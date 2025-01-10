import yaml
from mpi4py import MPI
from test_2d_substepping import get_initial_condition, get_dt, write_gcode, get_mesh
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.drivers import MHSSubstepper, MHSStaggeredSubstepper, NewtonRaphson, StaggeredRRDriver, MonolithicRRDriver
from mhs_fenicsx.problem.helpers import propagate_dg0_at_facets_same_mesh, interpolate
from mhs_fenicsx.chimera import build_moving_problem, interpolate_solution_to_inactive
import shutil
from dolfinx import mesh, fem, io
import numpy as np
from petsc4py import PETSc
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class MHSStaggeredChimeraSubstepper(MHSStaggeredSubstepper):
    def __init__(self, slow_problem: Problem, moving_problem : Problem,
                 writepos=True,
                 max_nr_iters=25,max_ls_iters=5,):
        self.pm = moving_problem
        super().__init__(slow_problem,writepos,max_nr_iters,max_ls_iters)
        self.quadrature_degree = 2 # Gamma Chimera

    def define_subproblem(self, subproblem_els=None):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        super().define_subproblem(subproblem_els)
        pm.set_dt(pf.dt.value)

    def check_assumptions(self):
        ''' No overlap between pm and ps '''
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        is_false = np.logical_and(pf.gamma_facets[pm].values, ps.bfacets_tag.values).all()
        return not(is_false)

    def prepare_slow_problem(self):
        ps = self.ps
        sd = self.staggered_driver
        ps.set_forms_domain()
        ps.set_forms_boundary()
        sd.set_robin(ps)
        ps.compile_forms()
        ps.pre_assemble()

    def pre_loop(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        MHSSubstepper.pre_loop(self)
        sd = self.staggered_driver
        self.macro_iter = 0
        self.prev_iter = {p : p.iter for p in plist}
        self.u_prev = {p : p.u.copy() for p in plist}
        # TODO: Move pm to initial position
        for u in self.u_prev.values():
            u.name = "u_prev_driver"
        (p,p_ext) = (pf,ps)
        self.ext_flux_tn = {p:fem.Function(p.dg0_vec,name="ext_flux_tn")}
        self.ext_conductivity_tn = {p:fem.Function(p.dg0,name="ext_conduc_tn")}
        p_ext.compute_gradient()
        self.ext_sol_tn = {p:fem.Function(p.v,name="ext_sol_tn")}
        propagate_dg0_at_facets_same_mesh(p_ext, p_ext.grad_u, p, self.ext_flux_tn[p])
        propagate_dg0_at_facets_same_mesh(p_ext, p_ext.k, p, self.ext_conductivity_tn[p])
        self.ext_sol_tn[p].x.array[:] = p_ext.u.x.array[:]

        bsize = self.ext_flux_tn[p].function_space.value_size
        for idx in range(bsize):
            self.ext_flux_tn[p].x.array[sd.active_gamma_cells[p]*bsize+idx] *= self.ext_conductivity_tn[p].x.array[sd.active_gamma_cells[p]]
        self.ext_flux_tn[p].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def pre_iterate(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        self.macro_iter += 1
        for p in plist:
            p.time = self.t0_macro_step
            p.iter = self.prev_iter[p]
            p.u_prev.x.array[:] = self.u_prev[p].x.array[:]
        self.relaxation_coeff_pf = self.staggered_driver.relaxation_coeff[pf].value

    def post_iterate(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        for p in plist:
            p.post_iterate()

    def micro_post_iterate(self):
        '''
        post_iterate of micro_step
        TODO: Maybe refactor? Seems like ps needs the same thing
        '''
        pf, pm = plist = (self.pf, self.pm)
        for p in plist:
            current_track = p.source.path.get_track(p.time)
            dt2track_end = current_track.t1 - p.time
            if abs(p.dt.value - dt2track_end) < 1e-9:
                p.set_dt(dt2track_end)


    def micro_steps(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        sd = self.staggered_driver
        chimera_driver = MonolithicRRDriver(pf, pm,
                                            quadrature_degree=self.quadrature_degree)
        self.micro_iter = 0
        while (self.t1_macro_step - pf.time) > 1e-7:
            forced_time_derivative = (self.micro_iter==0)

            for p in [pm, pf]:
                p.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)

            fast_subproblem_els = pf.active_els_func.x.array.nonzero()[0]
            pm.intersect_problem(pf, finalize=False)
            pf.subtract_problem(pm, finalize=False)
            for p, p_ext in [(pm, pf), (pf, pm)]:
                p.finalize_activation()
                p.find_gamma(p_ext)#TODO: Re-use previous data here
            assert(self.check_assumptions())

            self.micro_iter += 1
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            f = self.fraction_macro_step
            sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                    f*self.ext_sol_array_tnp1[:]
            sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                    f*self.ext_flux_array_tnp1[:]
            for p in [pm, pf]:
                p.set_forms_domain()
                p.set_forms_boundary()
            sd.set_robin(pf)
            for p in [pm, pf]:
                p.compile_forms()
                p.pre_assemble()
                p.assemble_residual()
                p.assemble_jacobian(finalize=False)
            chimera_driver.setup_coupling()
            for p in [pm, pf]:
                p.A.assemble()
            chimera_driver.solve()

            self.micro_post_iterate()
            interpolate_solution_to_inactive(pf,pm)
            self.writepos(case="micro")
            pf.set_activation(fast_subproblem_els, finalize=False)

    def initialize_post(self):
        if not(self.do_writepos):
            return
        self.name = "staggered_chimera_substepper"
        self.result_folder = f"post_{self.name}_tstep#{self.ps.iter}"
        shutil.rmtree(self.result_folder,ignore_errors=True)
        for p in [self.ps,self.pf, self.pm]:
            self.writers[p] = io.VTKFile(p.domain.comm, f"{self.result_folder}/{self.name}_{p.name}.pvd", "wb")

    def writepos(self,case="macro",extra_funs=[]):
        if not(self.do_writepos):
            return
        (ps,pf,pm) = (self.ps,self.pf,self.pm)
        sd = self.staggered_driver
        if not(self.writers):
            self.initialize_post()

        if case=="predictor":
            p, p_ext = (ps, pf)
            time = 0.0
        elif case=="micro":
            p, p_ext = (pf, ps)
            time = (self.macro_iter-1) + self.fraction_macro_step
        else:
            p, p_ext = (ps, pf)
            time = (self.macro_iter-1) + self.fraction_macro_step
        get_funs = lambda p : [p.u,p.source.fem_function,p.active_els_func,p.grad_u, p.u_prev,self.u_prev[p]] + [gn for gn in p.gamma_nodes.values()]
        p_funs = get_funs(p)
        for fun_dic in [sd.ext_flux,sd.net_ext_flux,sd.ext_conductivity,sd.ext_sol,
                        sd.prev_ext_flux,sd.prev_ext_sol]:
            try:
                p_funs.append(fun_dic[p])
            except (AttributeError,KeyError):
                continue
        p_funs += extra_funs
        p.compute_gradient()
        self.writers[p].write_function(p_funs,t=time)
        if case=="micro":
            self.writers[pm].write_function(get_funs(pm),t=time)

def run_staggered_RR(params, writepos=True):
    els_per_radius = params["els_per_radius"]
    radius = params["heat_source"]["radius"]
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    driver_constructor = StaggeredRRDriver
    initial_relaxation_factors=[1.0,1.0]
    big_mesh = get_mesh(params, els_per_radius, radius)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=f"big_chimera_ss_RR")
    p_moving = build_moving_problem(big_p, els_per_radius)
    initial_condition_fun = get_initial_condition(params)
    big_p.set_initial_condition(  initial_condition_fun )

    max_timesteps = params["max_timesteps"]
    for _ in range(max_timesteps):
        substeppin_driver = MHSStaggeredChimeraSubstepper(big_p, p_moving,
                                                          writepos=(params["substepper_writepos"] and writepos))
        (ps,pf) = (substeppin_driver.ps,substeppin_driver.pf)
        staggered_driver = driver_constructor(pf,ps,
                                       max_staggered_iters=params["max_staggered_iters"],
                                       initial_relaxation_factors=initial_relaxation_factors,)
        substeppin_driver.set_staggered_driver(staggered_driver)

        el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
        h = 1.0 / el_density
        k = float(params["material_metal"]["conductivity"])
        staggered_driver.dirichlet_coeff[staggered_driver.p1] = 1.0/4.0
        staggered_driver.dirichlet_coeff[staggered_driver.p2] =  k / (2 * h)
        staggered_driver.relaxation_coeff[staggered_driver.p1].value = 3.0 / 3.0

        staggered_driver.pre_loop(prepare_subproblems=False)
        substeppin_driver.pre_loop()
        if params["predictor_step"]:
            substeppin_driver.predictor_step(writepos=substeppin_driver.do_writepos and writepos)
                
        substeppin_driver.prepare_slow_problem()
        for _ in range(staggered_driver.max_staggered_iters):
            substeppin_driver.pre_iterate()
            staggered_driver.pre_iterate()
            substeppin_driver.iterate_substepped_rr()
            substeppin_driver.post_iterate()
            staggered_driver.post_iterate(verbose=True)
            if writepos:
                substeppin_driver.writepos(case="macro")
            if staggered_driver.convergence_crit < staggered_driver.convergence_threshold:
                break
        substeppin_driver.post_loop()
        # Interpolate solution to inactive ps
        ps.u.interpolate(pf.u,
                         cells0=pf.active_els_func.x.array.nonzero()[0],
                         cells1=pf.active_els_func.x.array.nonzero()[0])
        if writepos:
            ps.writepos()
    return big_p


if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    lp = LineProfiler()
    lp.add_module(MHSStaggeredChimeraSubstepper)
    lp.add_module(MHSStaggeredSubstepper)
    lp.add_module(MHSSubstepper)
    lp.add_module(MonolithicRRDriver)
    lp.add_module(Problem)
    lp.add_function(interpolate_solution_to_inactive)
    lp.add_function(interpolate)
    lp_wrapper = lp(run_staggered_RR)
    lp_wrapper(params,True)
    profiling_file = f"profiling_chimera_rss_{rank}.txt"
    with open(profiling_file, 'w') as pf:
        lp.print_stats(stream=pf)
