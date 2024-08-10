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
import yaml
import numpy as np
import shutil
import typing
from petsc4py import PETSc
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)
radius = params["heat_source"]["radius"]
speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))

def get_dt(adim_dt):
    return adim_dt * (radius / speed)

def write_gcode():
    L = params["L"]
    half_len = L / 2.0
    gcode_lines = []
    gcode_lines.append(f"G0 F{speed} X{0.5-half_len} Y1 Z0\n")
    gcode_lines.append(f"G1 X{0.5+half_len} E0.1\n")
    with open(params["path"],'w') as f:
        f.writelines(gcode_lines)

class MHSSubsteppingDriver:
    def __init__(self,slow_problem:Problem):
        self.ps = slow_problem
        self.fraction_macro_step = 0
        self.writers = dict()

    def __del__(self):
        for w in self.writers.values():
            w.close()

    def initialize_post(self):
        self.name = "staggered_substepper"
        self.result_folder = f"post_{self.name}"
        shutil.rmtree(self.result_folder,ignore_errors=True)
        for p in [self.ps,self.pf]:
            self.writers[p] = io.VTKFile(p.domain.comm, f"{self.result_folder}/{self.name}_{p.name}.pvd", "wb")

    def writepos(self,case="macro"):
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
        funs = [p.u,p.gamma_nodes,p.source_rhs,p.active_els_func,p.grad_u,
                p.u_prev,self.u_prev[p]]
        for fun_dic in [sd.ext_flux,sd.ext_conductivity,sd.ext_sol,
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
        initial_track_macro_step = ps.source.path.get_track(self.t0_macro_step)
        assert(initial_track_macro_step==ps.source.path.current_track)
        pad = 4*radius
        p0 = initial_track_macro_step.get_position(self.t0_macro_step)
        p1 = initial_track_macro_step.get_position(self.t1_macro_step)
        direction = initial_track_macro_step.get_direction()
        p0 -= direction*pad
        p1 += direction*pad
        obb = mhs_fenicsx.geometry.OBB(p0,p1,width=pad,height=pad,depth=pad,dim=cdim,
                                       shrink=False)
        obb_mesh = obb.get_dolfinx_mesh()
        #subproblem_els = mhs_fenicsx.geometry.mesh_collision(ps.domain,obb_mesh,bb_tree_mesh_big=ps.bb_tree)
        subproblem_els = mesh_collision(ps.domain._cpp_object,obb_mesh._cpp_object,bb_tree_big=ps.bb_tree._cpp_object)
        subproblem_els = np.array(subproblem_els)
        # Extract subproblem:
        self.submesh_data = {}
        submesh_data = mesh.create_submesh(ps.domain,cdim,subproblem_els)
        submesh = submesh_data[0]
        self.submesh_data["subcell_map"] = submesh_data[1]
        self.submesh_data["subvertex_map"] = submesh_data[2]
        self.submesh_data["subgeom_map"] = submesh_data[3]
        micro_params = ps.input_parameters.copy()
        micro_params["dt"] = get_dt(params["micro_adim_dt"])
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
        pf.u.interpolate(ps.u_prev,cells0=self.submesh_data["subcell_map"],cells1=np.arange(len(self.submesh_data["subcell_map"])))

    def subtract_child(self):
        # Subtract child from parent
        # TODO: Make this more efficient, redundancy in setting active_els_func
        # here and inside of set_activation
        (ps,pf) = self.ps, self.pf
        ps.active_els_func.x.array[self.submesh_data["subcell_map"]] = 0
        ps.active_els_func.x.scatter_forward()
        active_els = ps.active_els_func.x.array.nonzero()[0]
        active_els = [el for el in active_els if el < ps.cell_map.size_local]
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

def main(initial_condition=False):
    write_gcode()
    els_per_radius = params["els_per_radius"]
    driver_type = params["driver_type"]
    if   driver_type=="robin":
        driver_constructor = StaggeredRRDriver
        initial_relaxation_factors=[1.0,1.0]
    elif driver_type=="dn":
        driver_constructor = StaggeredDNDriver
        initial_relaxation_factors=[0.5,1]
    else:
        raise ValueError("Undefined staggered driver type.")
    points_side = np.round(1.0 / radius * els_per_radius).astype(int) + 1
    big_mesh  = mesh.create_unit_square(MPI.COMM_WORLD,
                                        points_side,
                                        points_side,
                                        #mesh.CellType.quadrilateral,
                                        )

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"])
    big_p = Problem(big_mesh, macro_params, name="big_sta")
    if initial_condition:
        initial_condition = lambda x : 100*np.cos(4*(x[0]-0.5)*(x[1]-0.5))
        big_p.set_initial_condition(  initial_condition )

    substeppin_driver = MHSSubsteppingDriver(big_p)

    substeppin_driver.extract_subproblem() # generates driver.fast_problem
    (ps,pf) = (substeppin_driver.ps,substeppin_driver.pf)
    staggered_driver = driver_constructor(pf,ps,
                                   submesh_data=substeppin_driver.submesh_data,
                                   max_staggered_iters=params["max_staggered_iters"],
                                   initial_relaxation_factors=initial_relaxation_factors,)
    if (type(staggered_driver)==StaggeredRRDriver):
        h = 1.0 / (points_side-1)
        k = float(params["material_metal"]["conductivity"])
        staggered_driver.dirichlet_coeff[staggered_driver.p1] = 1.0/4.0
        staggered_driver.dirichlet_coeff[staggered_driver.p2] =  k / (2 * h)
        staggered_driver.relaxation_coeff[staggered_driver.p1].value = 3.0 / 3.0
    # Move extra_subproblem here
    #TODO: Check on pre_iterate / post_iterate of problems
    staggered_driver.pre_loop(prepare_subproblems=False)
    substeppin_driver.pre_loop(staggered_driver)
    #substeppin_driver.predictor_step()
    substeppin_driver.subtract_child()
    staggered_driver.prepare_subproblems()
    for _ in range(staggered_driver.max_staggered_iters):
        substeppin_driver.pre_iterate()
        staggered_driver.pre_iterate()
        substeppin_driver.iterate()
        substeppin_driver.post_iterate()
        staggered_driver.post_iterate(verbose=True)
        substeppin_driver.writepos(case="macro")

        if staggered_driver.convergence_crit < staggered_driver.convergence_threshold:
            break
    #TODO: Interpolate solution to inactive ps
    ps.u.interpolate(pf.u,
                     cells0=np.arange(pf.num_cells),
                     cells1=substeppin_driver.submesh_data["subcell_map"])
    big_p.writepos()

def run_reference(initial_condition=True):
    write_gcode()
    els_per_radius = params["els_per_radius"]
    points_side = np.round(1.0 / radius * els_per_radius).astype(int) + 1
    big_mesh  = mesh.create_unit_square(MPI.COMM_WORLD,
                                        points_side,
                                        points_side,
                                        #mesh.CellType.quadrilateral,
                                        )

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["micro_adim_dt"])
    final_t = get_dt(params["macro_adim_dt"])
    big_p = Problem(big_mesh, macro_params, name="big")

    if initial_condition:
        initial_condition = lambda x : 100*np.cos(4*(x[0]-0.5)*(x[1]-0.5))
        big_p.set_initial_condition(  initial_condition )

    big_p.set_forms_domain()
    big_p.set_forms_boundary()
    big_p.compile_forms()
    while (final_t - big_p.time) > 1e-7:
        big_p.pre_iterate()
        big_p.pre_assemble()
        big_p.assemble()
        big_p.solve()
        big_p.post_iterate()
        big_p.writepos()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--run-sub',action='store_true')
    parser.add_argument('-r','--run-ref',action='store_true')
    args = parser.parse_args()
    lp = LineProfiler()
    lp.add_module(Problem)
    if args.run_sub:
        lp.add_module(MHSSubsteppingDriver)
        lp.add_module(StaggeredRRDriver)
        lp.add_function(mhs_fenicsx.geometry.mesh_collision)
        lp.add_function(build_subentity_to_parent_mapping)
        lp.add_function(compute_dg0_interpolation_data)
        lp.add_function(find_submesh_interface)
        lp.add_function(indices_to_function)
        lp_wrapper = lp(main)
        lp_wrapper()
        profiling_file = f"profiling_sub_{rank}.txt"
    elif args.run_ref:
        lp_wrapper = lp(run_reference)
        lp_wrapper()
        profiling_file = f"profiling_ref_{rank}.txt"
    else:
        exit()
    with open(profiling_file, 'w') as pf:
        lp.print_stats(stream=pf)
