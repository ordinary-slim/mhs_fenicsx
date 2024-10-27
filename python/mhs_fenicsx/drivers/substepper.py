from line_profiler import LineProfiler
from mhs_fenicsx.problem.helpers import get_mask, indices_to_function
from mpi4py import MPI
from dolfinx import mesh, fem, cpp, io
from mhs_fenicsx.problem import Problem, ufl
from mhs_fenicsx.submesh import build_subentity_to_parent_mapping, find_submesh_interface, \
compute_dg0_interpolation_data
from mhs_fenicsx_cpp import mesh_collision
from mhs_fenicsx.drivers.staggered_drivers import StaggeredRRDriver, StaggeredDNDriver, interpolate_dg0_cells_to_cells
from mhs_fenicsx.drivers.newton_raphson import NewtonRaphson
import mhs_fenicsx.geometry
import numpy as np
import shutil
import typing
from petsc4py import PETSc
from abc import ABC, abstractmethod

class MHSSubstepper(ABC):
    def __init__(self,slow_problem:Problem,writepos=True,
                 max_nr_iters=25,max_ls_iters=5,):
        self.ps = slow_problem
        self.pf: Problem = None
        self.fraction_macro_step = 0
        self.do_writepos = writepos
        self.writers = dict()
        self.max_nr_iters = max_nr_iters
        self.max_ls_iters = max_ls_iters
        self.physical_domain_restriction = self.ps.restriction
    
    def __del__(self):
        for w in self.writers.values():
            w.close()

    @abstractmethod
    def define_subproblem(self):
        pass

    @abstractmethod
    def pre_loop(self):
        pass

    @abstractmethod
    def pre_iterate(self):
        pass

    @abstractmethod
    def subtract_fast(self):
        '''
        I think this can be in the parent class, maybe not
        '''
        pass

    def find_subproblem_els(self):
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
        back_pad = 5*hs_radius
        front_pad = 5*hs_radius
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
        return np.array(subproblem_els,dtype=np.int32)

    def predictor_step(self):
        '''
        Always linear
        '''
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
        # Reset iter to prev.
        # Useful for writepos
        ps.iter -= 1

    def micro_post_iterate(self):
        '''
        post_iterate of micro_step
        TODO: Maybe refactor? Seems like ps needs the same thing
        '''
        pf = self.pf
        current_track = pf.source.path.get_track(pf.time)
        dt2track_end = current_track.t1 - pf.time
        if abs(pf.dt.value - dt2track_end) < 1e-9:
            pf.set_dt(dt2track_end)

    def post_loop(self):
        self.ps.initialize_activation()

class MHSSemiMonolithicSubstepper(MHSSubstepper):
    def __init__(self,slow_problem: Problem ,writepos=True,
                 max_nr_iters=25,max_ls_iters=5,):
        super().__init__(slow_problem,writepos,max_nr_iters,max_ls_iters)
        self.pf = self.ps.copy(name=f"{self.ps.name}_micro_iters")

    def define_subproblem(self, subproblem_els=None):
        (ps, pf) = (self.ps, self.pf)
        # Store this and use it for deactivation?
        if not(subproblem_els):
            subproblem_els = self.find_subproblem_els()
        hs_radius = pf.source.R
        track_t0 = pf.source.path.get_track(self.t0_macro_step)
        hs_speed  = track_t0.speed# TODO: Can't use this speed!
        pf.set_dt( ps.input_parameters["micro_adim_dt"] * (hs_radius / hs_speed) )
        pf.set_activation(subproblem_els)
        pf.set_linear_solver(pf.input_parameters["petsc_opts_micro"])
        self.set_interface()
        self.dofs_fast = fem.locate_dofs_topological(pf.v, pf.dim, subproblem_els)

    def set_interface(self):
        # Find Gamma facets
        (ps, pf) = (self.ps, self.pf)
        gamma_facets_candidates = pf.bfacets_tag.values.nonzero()[0]
        gamma_facets_subindices = np.where(ps.bfacets_tag.values[gamma_facets_candidates] == 0)[0]
        gamma_facets = gamma_facets_candidates[gamma_facets_subindices]
        cdim = ps.domain.topology.dim
        gamma_facets_tag = mesh.meshtags(ps.domain, cdim-1,
                              np.arange(ps.num_facets, dtype=np.int32),
                              get_mask(ps.num_facets,gamma_facets))
        for p in [pf, ps]:
            p.set_gamma(gamma_facets_tag)

    def subtract_fast(self):
        (ps, pf) = (self.ps, self.pf)
        ps.set_activation((ps.active_els_func.x.array - pf.active_els_func.x.array).nonzero()[0])


    def pre_loop(self):
        (ps,pf) = (self.ps,self.pf)
        self.num_micro_steps = np.round(ps.dt.value / pf.dt.value, 9).astype(np.int32)
        self.macro_iter = 0
        self.prev_iter = {ps:ps.iter,pf:pf.iter}
        # Prepare fast problem
        # TODO: Set Dirichlet BC
        self.set_dirichlet_fast()
        pf.set_forms_domain()
        pf.set_forms_boundary()
        pf.compile_forms()
        pf.pre_assemble()
        # Prepare slow problem
        self.u_prev = {ps:ps.u.copy(),pf:pf.u.copy()}
        for f in self.u_prev.values():
            f.name = "u_prev_driver"
        self.initialize_post()

    def pre_iterate(self):
        (ps,pf) = (self.ps,self.pf)
        self.macro_iter += 1
        for p in [ps,pf]:
            p.time = self.t0_macro_step
            p.iter = self.prev_iter[p]
            p.u_prev.x.array[:] = self.u_prev[p].x.array[:]

    def post_iterate(self):
        (ps,pf) = (self.ps,self.pf)
        for p in [ps,pf]:
            p.post_iterate()

    def set_dirichlet_fast(self):
        '''
        - Called right before micro-steps
        - Interpolate macro tnp1 sol to Gamma nodes
        '''
        (ps,pf) = (self.ps,self.pf)
        pf.clear_dirchlet_bcs()
        dirichlet_fun_fast = fem.Function(pf.v, name="dirichlet_con_fast")
        self.gamma_dofs_fast = fem.locate_dofs_topological(pf.v, pf.dim-1, pf.gamma_facets.find(1))
        # Set Gamma dirichlet
        self.fast_dirichlet_tcon = fem.dirichletbc(dirichlet_fun_fast, self.gamma_dofs_fast)
        pf.dirichlet_bcs.append(self.fast_dirichlet_tcon)

    def micro_steps(self):
        (ps,pf) = (self.ps,self.pf)
        self.micro_iter = 0
        while self.micro_iter < (self.num_micro_steps - 1):
            self.micro_iter += 1
            forced_time_derivative = (self.micro_iter==0)
            pf.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
            f = self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            #TODO: Update Dirichlet BC pf here!
            self.fast_dirichlet_tcon.g.x.array[self.gamma_dofs_fast] = \
                    (1-f)*ps.u_prev.x.array[self.gamma_dofs_fast] + \
                    f*ps.u.x.array[self.gamma_dofs_fast]
            if pf.phase_change:
                nr_driver = NewtonRaphson(pf)
                nr_driver.solve()
            else:
                pf.assemble()
                pf.solve()
            #pf.post_iterate()
            self.micro_post_iterate()
            if self.writepos:
                self.writepos("micro")

    def monolithic_step(self):
        (ps, pf) = (self.ps, self.pf)
        pf.clear_dirchlet_bcs()
        pf.pre_iterate()
        ps.pre_iterate(forced_time_derivative=True)

        # Set-up incremental solve of both problem in ps
        ps.u.x.array[self.dofs_fast] = pf.u.x.array[self.dofs_fast]
        ps.dt_func.x.array[self.gamma_dofs_fast] = pf.dt.value
        ps.u_prev.x.array[self.gamma_dofs_fast] = pf.u_prev.x.array[self.gamma_dofs_fast]

        subdomain_data = [(idx+1, active_els) for (idx, active_els) in enumerate([p.local_active_els for p in [ps, pf]])]
        ps.set_forms_domain((1,subdomain_data),argument=ps.u)
        pf.set_forms_domain((2,subdomain_data),argument=ps.u)

        ps.a_ufl += pf.a_ufl
        ps.l_ufl += pf.l_ufl
        ps.r_ufl = ps.a_ufl - ps.l_ufl#residual
        ps.j_ufl = ufl.derivative(ps.r_ufl, ps.u)
        ps.mr_compiled = fem.form(-ps.r_ufl)
        ps.j_compiled = fem.form(ps.j_ufl)

        ps_restriction = ps.restriction
        ps.restriction = self.physical_domain_restriction

        if ps.phase_change or pf.phase_change:
            nr_driver = NewtonRaphson(ps,max_ls_iters=0)
            nr_driver.solve()
        else:
            ps.assemble()
            ps.solve()
        pf.u.x.array[:] = ps.u.x.array[:]

        # POST-ITERATE
        ps.post_iterate()
        pf.post_iterate()
        ps.restriction = ps_restriction
        ps.set_dt(ps.dt.value)
        pf.dirichlet_bcs = [self.fast_dirichlet_tcon]

        self.micro_iter += 1
        self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
        if self.writepos:
            self.writepos("micro")

    def initialize_post(self):
        if not(self.do_writepos):
            return
        self.name = "semi_monolithic_substepper"
        self.result_folder = f"post_{self.name}_tstep#{self.ps.iter}"
        shutil.rmtree(self.result_folder,ignore_errors=True)
        p = self.pf
        self.writers[p] = io.VTKFile(p.domain.comm, f"{self.result_folder}/{self.name}_{p.name}.pvd", "wb")

    def writepos(self,case="macro",extra_funs=[]):
        if not(self.do_writepos):
            return
        (pf, ps) = (self.pf, self.ps)
        if not(self.writers):
            self.initialize_post()
        if case=="predictor":
            p = ps
            time = 0.0
        elif case=="micro":
            p = pf
            time = (self.macro_iter-1) + self.fraction_macro_step
        else:
            p = ps
            time = self.macro_iter
        funs = [p.u,p.gamma_nodes,p.source.fem_function,p.active_els_func,p.grad_u,
                p.u_prev,self.u_prev[p]]
        funs += extra_funs

        p.compute_gradient()
        #print(f"time = {time}, micro_iter = {self.micro_iter}, macro_iter = {self.macro_iter}")
        self.writers[pf].write_function(funs,t=time)


class MHSStaggeredSubstepper(MHSSubstepper):
    def __init__(self,slow_problem:Problem,writepos=True,
                 max_nr_iters=25,max_ls_iters=5,):
        super().__init__(slow_problem,writepos,max_nr_iters,max_ls_iters)

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
        if case=="predictor":
            p = ps
            time = 0.0
        elif case=="micro":
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
            if not(pf.phase_change):
                pf.assemble()
                pf.solve()
            else:
                nr_driver = NewtonRaphson(pf)
                nr_driver.solve()
            #pf.post_iterate()
            self.micro_post_iterate()
            self.writepos(case="micro")

    def define_subproblem(self):
        ps = self.ps
        cdim = ps.domain.topology.dim
        subproblem_els = self.find_subproblem_els()
        # Extract subproblem:
        self.submesh_data = {}
        submesh_data = mesh.create_submesh(ps.domain,cdim,subproblem_els)
        submesh = submesh_data[0]
        self.submesh_data["subcell_map"] = submesh_data[1]
        self.submesh_data["subvertex_map"] = submesh_data[2]
        self.submesh_data["subgeom_map"] = submesh_data[3]
        micro_params = ps.input_parameters.copy()
        hs_radius = ps.source.R
        track_t0 = ps.source.path.get_track(self.t0_macro_step)
        hs_speed  = track_t0.speed# TODO: Can't use this speed!
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

    def subtract_fast(self):
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
        if ps.phase_change:
            nr_driver = NewtonRaphson(ps)
            nr_driver.solve()
        else:
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
        if pn.phase_change:
            nr_driver = NewtonRaphson(pn)
            nr_driver.solve()
        else:
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

    def set_staggered_driver(self, sd:typing.Union[StaggeredDNDriver,StaggeredRRDriver]):
        self.staggered_driver = sd

    def pre_loop(self):
        (ps,pf) = (self.ps,self.pf)
        super().pre_loop()
        sd = self.staggered_driver
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
        self.ext_conductivity_tn = {p:fem.Function(p.dg0,name="ext_conduc_tn")}
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
        self.ext_flux_tn[p].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
