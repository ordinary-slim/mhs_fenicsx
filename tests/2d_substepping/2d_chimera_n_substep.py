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
    def __init__(self,slow_problem:Problem, moving_problem : Problem,
                 writepos=True,
                 max_nr_iters=25,max_ls_iters=5,
                 do_predictor=False):
        super().__init__(slow_problem,writepos,max_nr_iters,max_ls_iters, do_predictor)
        self.pm = moving_problem
        self.plist.append(self.pm)
        self.quadrature_degree = 2 # Gamma Chimera
        self.name = "staggered_chimera_substepper"

    def pre_loop(self):
        super().pre_loop()
        self.pm.set_dt(self.pf.dt.value)

    def check_assumptions(self):
        ''' No overlap between pm and ps '''
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        is_false = np.logical_and(pf.gamma_facets[pm].values, ps.bfacets_tag.values).all()
        return not(is_false)

    def post_iterate(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        self.staggered_driver.relaxation_coeff[pf].value = self.relaxation_coeff_pf
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
        for p in [pm, pf]:
            p.set_forms_domain()
            p.set_forms_boundary()
            if p == pf:
                sd.set_robin(pf)
            p.compile_forms()

        self.micro_iter = 0
        while (self.t1_macro_step - pf.time) > 1e-7:
            forced_time_derivative = (self.micro_iter==0)

            for p in [pm, pf]:
                p.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)

            fast_subproblem_els = pf.active_els_func.x.array.nonzero()[0]
            pm.intersect_problem(pf, finalize=False)
            ps_pf_integration_data = pf.form_subdomain_data[fem.IntegralType.exterior_facet][-1]
            assert(ps_pf_integration_data[0] == sd.gamma_integration_tag)
            pf.subtract_problem(pm, finalize=False)
            for p, p_ext in [(pm, pf), (pf, pm)]:
                p.finalize_activation()
                p.find_gamma(p_ext)#TODO: Re-use previous data here
            pf.form_subdomain_data[fem.IntegralType.exterior_facet].append(ps_pf_integration_data)
            assert(self.check_assumptions())

            self.micro_iter += 1
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            f = self.fraction_macro_step
            sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                    f*self.ext_sol_array_tnp1[:]
            sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                    f*self.ext_flux_array_tnp1[:]
            for p in [pm, pf]:
                p.instantiate_forms()
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
    initial_relaxation_factors=[1.0,1.0]
    big_mesh = get_mesh(params, els_per_radius, radius)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, name=f"big_chimera_ss_RR")
    pm = build_moving_problem(ps, els_per_radius)
    initial_condition_fun = get_initial_condition(params)
    ps.set_initial_condition(  initial_condition_fun )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSStaggeredChimeraSubstepper(ps, pm,
                                                      writepos=(params["substepper_writepos"] and writepos),
                                                      do_predictor=params["predictor_step"])
    pf = substeppin_driver.pf
    staggered_driver = StaggeredRRDriver(pf,ps,
                                         max_staggered_iters=params["max_staggered_iters"],
                                         initial_relaxation_factors=initial_relaxation_factors,)

    el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
    h = 1.0 / el_density
    k = float(params["material_metal"]["conductivity"])
    staggered_driver.set_dirichlet_coefficients(h, k)

    for _ in range(max_timesteps):
        substeppin_driver.update_fast_problem()
        substeppin_driver.set_staggered_driver(staggered_driver)
        staggered_driver.pre_loop(prepare_subproblems=False)
        substeppin_driver.pre_loop()
        if substeppin_driver.do_predictor:
            substeppin_driver.predictor_step(writepos=substeppin_driver.do_writepos and writepos)
        staggered_driver.prepare_subproblems()
        for _ in range(staggered_driver.max_staggered_iters):
            substeppin_driver.pre_iterate()
            staggered_driver.pre_iterate()
            substeppin_driver.iterate()
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
        ps.is_grad_computed = False
        pf.u.x.array[:] = ps.u.x.array[:]
        pf.is_grad_computed = False
        if writepos:
            ps.writepos()
    return ps

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
