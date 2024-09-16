from line_profiler import LineProfiler
from mhs_fenicsx.problem.helpers import indices_to_function
from mpi4py import MPI
from dolfinx import mesh, fem, cpp, io
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.submesh import build_subentity_to_parent_mapping, find_submesh_interface, \
compute_dg0_interpolation_data
from mhs_fenicsx_cpp import mesh_collision
from mhs_fenicsx.drivers.staggered_drivers import StaggeredRRDriver, StaggeredDNDriver, interpolate_dg0_cells_to_cells
import mhs_fenicsx.geometry
import numpy as np
import shutil
import typing
from petsc4py import PETSc

class MHSSubsteppingDriver:
    def __init__(self,slow_problem:Problem,writepos=True):
        self.ps = slow_problem
        self.fraction_macro_step = 0
        self.do_writepos = writepos
        self.writers = dict()

    def __del__(self):
        for w in self.writers.values():
            w.close()

    def initialize_post(self):
        if not(self.do_writepos):
            return
        self.name = "staggered_substepper"
        self.result_folder = f"post_{self.name}_tstep#{self.ps.iter}"
        shutil.rmtree(self.result_folder,ignore_errors=True)
        for p in [self.ps,self.pf]:
            self.writers[p] = io.VTKFile(p.domain.comm, f"{self.result_folder}/{self.name}_{p.name}.pvd", "wb")

    def writepos(self,case="macro"):
        if not(self.do_writepos):
            return
        (ps,pf) = (self.ps,self.pf)
        sd = self.staggered_driver
        if not(self.writers):
            self.initialize_post()
        if case=="micro":
            p = pf
            time = (self.macro_iter-1) + self.fraction_macro_step
        else:
            p = ps
            time = self.macro_iter
        funs = [p.u,p.gamma_nodes,p.source.fem_function,p.active_els_func,p.grad_u,
                p.u_prev,self.u_prev[p]]
        for fun_dic in [sd.ext_flux,sd.net_ext_flux,sd.ext_conductivity,sd.ext_sol,
                        sd.prev_ext_flux,sd.prev_ext_sol]:
            try:
                funs.append(fun_dic[p])
            except (AttributeError,KeyError):
                pass

        p.compute_gradient()
        self.writers[p].write_function(funs,t=time)


    def predictor_step(self):
        ps = self.ps

        # MACRO-STEP
        ps.pre_iterate()
        ps.set_forms_domain()
        if ps.convection_coeff:
            ps.set_forms_boundary()
        ps.compile_forms()
        ps.pre_assemble()
        ps.assemble()
        ps.solve()
        ps.post_iterate()

    def micro_steps(self):
        (ps,pf) = (self.ps,self.pf)
        sd = self.staggered_driver
        self.micro_iter = 0
        while (self.t1_macro_step - pf.time) > 1e-7:
            forced_time_derivative = (self.micro_iter==0)
            pf.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
            self.micro_iter += 1
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            if type(sd)==StaggeredRRDriver:
                f = self.fraction_macro_step
                sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                        f*self.ext_sol_array_tnp1[:]
                sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                        f*self.ext_flux_array_tnp1[:]
            elif type(sd)==StaggeredDNDriver and sd.p_dirichlet==pf:
                sd.dirichlet_tcon.g.x.array[:] = (1-sd.relaxation_coeff[pf].value)*pf.u.x.array[:] + \
                                                 sd.relaxation_coeff[pf].value*((1-self.fraction_macro_step)*\
                                                 sd.prev_ext_sol[pf].x.array[:] + self.fraction_macro_step*\
                                                 sd.ext_sol[pf].x.array[:])
            elif type(sd)==StaggeredDNDriver and sd.p_neumann==pf:
                sd.relaxation_coeff[pf].value = self.fraction_macro_step
            pf.assemble()
            pf.solve()
            #pf.post_iterate()
            self.micro_post_iterate()
            self.writepos(case="micro")

    def extract_subproblem(self):
        ps = self.ps
        self.t0_macro_step = ps.time
        self.t1_macro_step = ps.time + ps.dt.value
        cdim = ps.dim
        # Determine geometry of subproblem
        # To do it properly, get initial time of macro step, final time of macro step
        # Do collision tests across track and accumulate elements to extract
        # Here we just do a single collision test
        track_t0 = ps.source.path.get_track(self.t0_macro_step)
        track_t1 = ps.source.path.get_track(self.t1_macro_step)
        #assert(track_t0==ps.source.path.current_track)
        hs_radius = ps.source.R
        # TODO: Can't use this speed!
        hs_speed  = track_t0.speed
        back_pad = 5*hs_radius
        front_pad = 2*hs_radius
        side_pad = 3*hs_radius
        p0 = track_t0.get_position(self.t0_macro_step)
        p1 = track_t1.get_position(self.t1_macro_step)
        direction = track_t0.get_direction()
        p0 -= direction*back_pad
        p1 += direction*front_pad
        obb = mhs_fenicsx.geometry.OBB(p0,p1,width=back_pad,height=side_pad,depth=side_pad,dim=cdim,
                                       shrink=False)
        obb_mesh = obb.get_dolfinx_mesh()
        #subproblem_els = mhs_fenicsx.geometry.mesh_collision(ps.domain,obb_mesh,bb_tree_mesh_big=ps.bb_tree)
        subproblem_els = mesh_collision(ps.domain._cpp_object,obb_mesh._cpp_object,bb_tree_big=ps.bb_tree._cpp_object)
        subproblem_els = np.array(subproblem_els,dtype=np.int32)
        # Extract subproblem:
        self.submesh_data = {}
        submesh_data = mesh.create_submesh(ps.domain,cdim,subproblem_els)
        submesh = submesh_data[0]
        self.submesh_data["subcell_map"] = submesh_data[1]
        self.submesh_data["subvertex_map"] = submesh_data[2]
        self.submesh_data["subgeom_map"] = submesh_data[3]
        micro_params = ps.input_parameters.copy()
        micro_params["dt"] = micro_params["micro_adim_dt"] * (hs_radius / hs_speed)
        micro_params["petsc_opts"] = micro_params["petsc_opts_micro"]
        self.pf = Problem(submesh_data[0],micro_params, name="small")
        pf = self.pf
        self.submesh_data["parent"] = ps
        self.submesh_data["child"] = pf
        self.submesh_data["subfacet_map"] = build_subentity_to_parent_mapping(cdim-1,
                                                               ps.domain,
                                                               submesh,
                                                               self.submesh_data["subcell_map"],
                                                               self.submesh_data["subvertex_map"])
        find_submesh_interface(ps,pf,self.submesh_data)
        compute_dg0_interpolation_data(ps,pf,self.submesh_data)
        pf.u.interpolate(ps.u,cells0=self.submesh_data["subcell_map"],cells1=np.arange(len(self.submesh_data["subcell_map"])))

    def subtract_child(self):
        # Subtract child from parent
        # TODO: Make this more efficient, redundancy in setting active_els_func
        # here and inside of set_activation
        (ps,pf) = self.ps, self.pf
        ps.active_els_func.x.array[self.submesh_data["subcell_map"]] = 0
        ps.active_els_func.x.scatter_forward()
        active_els = ps.active_els_func.x.array[:ps.cell_map.size_local].nonzero()[0]
        ps.set_activation(active_els)

    def iterate_substepped_rr(self):
        (ps,pf) = (self.ps,self.pf)
        rr_driver = self.staggered_driver
        assert(type(rr_driver)==StaggeredRRDriver)
        self.update_robin_fast()
        self.micro_steps()
        # have solution at tnp1
        # Updated sol, grad, conduc from n+1
        # I have to apply curr sol, grad, conduc from n+1 weighted with the one from n

        rr_driver.update_robin(ps)
        ps.pre_iterate(forced_time_derivative=True)
        ps.assemble()
        ps.solve()

    def iterate_substepped_dn(self):
        dn_driver = self.staggered_driver
        assert(type(dn_driver)==StaggeredDNDriver)
        (pn,pd) = (self.ps,self.pf)

        # Solve fast/Dirichlet problem
        self.micro_steps()

        dn_driver.update_neumann_interface()
        # Solve slow/Neumann problem
        pn.pre_iterate(forced_time_derivative=True)
        pn.assemble()
        pn.solve()
        dn_driver.update_relaxation_factor()
        dn_driver.update_dirichlet_interface()

    def pre_iterate(self):
        (ps,pf) = (self.ps,self.pf)
        self.macro_iter += 1
        for p in [ps,pf]:
            p.time = self.t0_macro_step
            p.iter = self.prev_iter[p]
            p.u_prev.x.array[:] = self.u_prev[p].x.array[:]
        self.relaxation_coeff_pf = self.staggered_driver.relaxation_coeff[pf].value
        # TODO: Undo pre-iterate of source
        # TODO: Undo pre-iterate of domain
        # TODO: Undo post-iterate of problem

    def update_robin_fast(self):
        '''
        Update Robin condition of fast problem before micro-steps
        '''
        sd = self.staggered_driver
        pf = self.pf
        assert(type(sd)==StaggeredRRDriver)
        sd.update_robin(pf)
        self.ext_sol_array_tnp1 = sd.net_ext_sol[pf].x.array.copy()
        self.ext_flux_array_tnp1 = sd.net_ext_flux[pf].x.array.copy()

    def post_iterate(self):
        (ps,pf) = (self.ps,self.pf)
        self.staggered_driver.relaxation_coeff[pf].value = self.relaxation_coeff_pf
        for p in [ps,pf]:
            p.post_iterate()

    def micro_post_iterate(self):
        '''
        post_iterate of micro_step
        TODO: Maybe refactor? Seems like ps needs the same thing
        '''
        pf = self.pf
        current_track = pf.source.path.get_track(pf.time)
        dt2track_end = current_track.t1 - pf.time
        if abs(pf.dt.value - dt2track_end) < 1e-9:
            pf.dt.value = dt2track_end

    def pre_loop(self,sd:typing.Union[StaggeredDNDriver,StaggeredRRDriver]):
        (ps,pf) = (self.ps,self.pf)
        self.staggered_driver = sd
        if type(sd)==StaggeredRRDriver:
            self.iterate = self.iterate_substepped_rr
        elif type(sd)==StaggeredDNDriver:
            self.iterate = self.iterate_substepped_dn
        else:
            raise ValueError("Unknown staggered driver type.")
        self.macro_iter = 0
        self.prev_iter = {ps:ps.iter,pf:pf.iter}
        self.u_prev = {ps:ps.u.copy(),pf:pf.u.copy()}
        for u in self.u_prev.values():
            u.name = "u_prev_driver"
        # Add a compute gradient around here!
        (p,p_ext) = (pf,ps)
        self.ext_flux_tn = {p:fem.Function(p.dg0_vec,name="ext_flux_tn")}
        self.ext_conductivity_tn = {p:fem.Function(p.dg0_bg,name="ext_conduc_tn")}
        p_ext.compute_gradient()
        # TODO: Are these necessary? Can I get them from my own data?
        if type(sd)==StaggeredRRDriver:
            self.ext_sol_tn = {p:fem.Function(p.v,name="ext_sol_tn")}
            interpolate_dg0_cells_to_cells(self.ext_flux_tn[p],p_ext.grad_u,sd.active_gamma_cells[p_ext],sd.active_gamma_cells[p])
            interpolate_dg0_cells_to_cells(self.ext_conductivity_tn[p],p_ext.k,sd.active_gamma_cells[p_ext],sd.active_gamma_cells[p])
            self.ext_sol_tn[p].interpolate(p_ext.u,cells0=sd.submesh_cells[p_ext],cells1=sd.submesh_cells[p])
            self.ext_sol_tn[p].x.scatter_forward()
        elif type(sd)==StaggeredDNDriver:
            if sd.p_dirichlet==pf:
                self.ext_sol_tn[p].interpolate(p_ext.u,cells0=sd.submesh_cells[p_ext],cells1=sd.submesh_cells[p])
            else:
                interpolate_dg0_cells_to_cells(self.ext_flux_tn[p],p_ext.grad_u,sd.active_gamma_cells[p_ext],sd.active_gamma_cells[p])
                interpolate_dg0_cells_to_cells(self.ext_conductivity_tn[p],p_ext.k,sd.active_gamma_cells[p_ext],sd.active_gamma_cells[p])

        dim = self.ext_flux_tn[p].function_space.value_size
        for cell in sd.active_gamma_cells[p]:
            self.ext_flux_tn[p].x.array[cell*dim:cell*dim+dim] *= self.ext_conductivity_tn[p].x.array[cell]
        self.ext_flux_tn[p].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def post_loop(self):
        self.ps.initialize_activation()
