import yaml
from mpi4py import MPI

from mhs_fenicsx.drivers import MonolithicRRDriver, MonolithicDomainDecompositionDriver, StaggeredRRDriver
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, ChimeraSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicChimeraSubstepper
from test_2d_substepping import get_initial_condition, get_dt, write_gcode, get_mesh
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.chimera import build_moving_problem
from mhs_fenicsx.problem.helpers import assert_pointwise_vals, print_vals
import shutil
from dolfinx import mesh, fem, io
import numpy as np
import argparse
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run_staggered_RR(params, writepos=True):
    els_per_radius = params["els_per_radius"]
    radius = params["source_terms"][0]["radius"]
    initial_relaxation_factors=[1.0,1.0]
    big_mesh = get_mesh(params, els_per_radius, radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, name=f"big_chimera_ss_RR")
    pm = build_moving_problem(ps, els_per_radius)
    initial_condition_fun = get_initial_condition(params)
    ps.set_initial_condition(  initial_condition_fun )
    pm.set_initial_condition(  initial_condition_fun )

    max_timesteps = params["max_timesteps"]

    if params["substepping_parameters"]["chimera_steadiness_workflow"]["enabled"]:
        class MyMHSStaggeredChimeraSubstepper(MHSStaggeredChimeraSubstepper):
            def is_steady_enough(self):
                yes = (len(self.steadiness_measurements) > 1) and (((self.steadiness_measurements[-1] - self.steadiness_measurements[-2]) / self.steadiness_measurements[-2]) < 0.05)
                if yes:
                    self.steadiness_measurements.clear()
                return yes
        driver_type = MyMHSStaggeredChimeraSubstepper
    else:
        driver_type = MHSStaggeredChimeraSubstepper

    substeppin_driver = driver_type(
            StaggeredRRDriver,
            initial_relaxation_factors,
            ps, pm)

    staggered_driver = substeppin_driver.staggered_driver
    el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
    h = 1.0 / el_density
    k = float(params["material_metal"]["conductivity"])
    staggered_driver.set_dirichlet_coefficients(h, k)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos()
    return ps

def run_hodge(params, writepos=True):
    els_per_radius = params["els_per_radius"]
    radius = params["source_terms"][0]["radius"]
    big_mesh = get_mesh(params, els_per_radius, radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, name=f"big_chimera_sms")
    pm = build_moving_problem(ps, els_per_radius)
    initial_condition_fun = get_initial_condition(params)
    ps.set_initial_condition(  initial_condition_fun )
    pm.set_initial_condition(  initial_condition_fun )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSSemiMonolithicChimeraSubstepper(ps, pm)
    pf = substeppin_driver.pf
    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps,pf]:
                p.writepos(extra_funcs=[p.u_prev])
    return ps

def test_staggered_robin_chimera_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    params["material_metal"].pop("phase_change")
    p = run_staggered_RR(params, writepos=False)
    points = np.array([
        [-0.250, -0.250, 0.0],
        [-0.250, -0.375, 0.0],
        [+0.250, +0.000, 0.0],
        [+0.375, -0.125, 0.0],
        ])
    vals = np.array([230.21145903,  767.64713976,   24.99873494, 1338.92131082])
    assert_pointwise_vals(p,points,vals)

def test_sms_chimera_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    params["material_metal"].pop("phase_change")
    p = run_hodge(params, writepos=False)
    points = np.array([
        [-0.250, -0.250, 0.0],
        [-0.250, -0.375, 0.0],
        [+0.250, +0.000, 0.0],
        [+0.375, -0.125, 0.0],
        ])
    vals = np.array([230.91602869,  767.95991237,   25.23704842, 1340.40649002])
    assert_pointwise_vals(p,points,vals)

if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss','--run-sub-sta',action='store_true')
    parser.add_argument('-sms','--run-sub-mon',action='store_true')
    parser.add_argument('-t','--testing',action='store_true')
    write_gcode(params)
    lp = LineProfiler()
    lp.add_module(Problem)
    args = parser.parse_args()
    if args.run_sub_sta:
        from mhs_fenicsx.chimera import interpolate_solution_to_inactive
        from mhs_fenicsx.problem.helpers import interpolate_cg1
        lp.add_module(MHSStaggeredChimeraSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(MHSSubstepper)
        lp.add_module(MonolithicDomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp.add_function(interpolate_solution_to_inactive)
        lp.add_function(interpolate_cg1)
        lp_wrapper = lp(run_staggered_RR)
        lp_wrapper(params, writepos = True)
        profiling_file = f"profiling_chimera_rss_{rank}.txt"
    if args.run_sub_mon:
        from mhs_fenicsx.chimera import interpolate_solution_to_inactive
        from mhs_fenicsx.problem.helpers import interpolate_cg1
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicChimeraSubstepper)
        lp.add_module(MonolithicDomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp.add_function(interpolate_solution_to_inactive)
        lp.add_function(interpolate_cg1)
        lp_wrapper = lp(run_hodge)
        lp_wrapper(params,True)
        profiling_file = f"profiling_chimera_hodge_{rank}.txt"
    if args.testing:
        test_staggered_robin_chimera_substepper()
        test_sms_chimera_substepper()
        exit()
    with open(profiling_file, 'w') as pf:
        lp.print_stats(stream=pf)
