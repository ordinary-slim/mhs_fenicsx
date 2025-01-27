import yaml
from mpi4py import MPI

from mhs_fenicsx.drivers import MonolithicRRDriver, MonolithicDomainDecompositionDriver, StaggeredRRDriver
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicChimeraSubstepper
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
    pm.set_initial_condition(  initial_condition_fun )

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
        ps.u.x.array[substeppin_driver.dofs_fast] = pf.u.x.array[substeppin_driver.dofs_fast]
        ps.u.x.scatter_forward()
        ps.is_grad_computed = False
        pf.u.x.array[:] = ps.u.x.array[:]
        pf.is_grad_computed = False
        if writepos:
            ps.writepos()
    return ps

def run_hodge(params, writepos=True):
    els_per_radius = params["els_per_radius"]
    radius = params["heat_source"]["radius"]
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    big_mesh = get_mesh(params, els_per_radius, radius)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, name=f"big_chimera_sms")
    pm = build_moving_problem(ps, els_per_radius)
    initial_condition_fun = get_initial_condition(params)
    ps.set_initial_condition(  initial_condition_fun )
    pm.set_initial_condition(  initial_condition_fun )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSSemiMonolithicChimeraSubstepper(ps, pm,
                                                           writepos=(params["substepper_writepos"] and writepos),
                                                           do_predictor=params["predictor_step"])
    pf = substeppin_driver.pf
    for _ in range(max_timesteps):
        substeppin_driver.update_fast_problem()
        substeppin_driver.pre_loop(prepare_fast_problem=False)
        if substeppin_driver.do_predictor:
            substeppin_driver.predictor_step(writepos=substeppin_driver.do_writepos and writepos)
        for _ in range(params["max_staggered_iters"]):
            substeppin_driver.pre_iterate()
            substeppin_driver.micro_steps()
            substeppin_driver.monolithic_step()
            substeppin_driver.post_iterate()
        substeppin_driver.post_loop()
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
    vals = np.array([
        229.38574,
        782.87969,
        24.998454,
        1625.5163,
        ])
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
    vals = np.array([
        230.22777,
        783.1233,
        25.236909,
        1626.8163,
        ])
    assert_pointwise_vals(p,points,vals)

if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss','--run-sub-sta',action='store_true')
    parser.add_argument('-sms','--run-sub-mon',action='store_true')
    write_gcode(params)
    lp = LineProfiler()
    lp.add_module(Problem)
    args = parser.parse_args()
    if args.run_sub_sta:
        from mhs_fenicsx.chimera import interpolate_solution_to_inactive
        from mhs_fenicsx.problem.helpers import interpolate
        lp.add_module(MHSStaggeredChimeraSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(MHSSubstepper)
        lp.add_module(MonolithicDomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_function(interpolate_solution_to_inactive)
        lp.add_function(interpolate)
        lp_wrapper = lp(run_staggered_RR)
        lp_wrapper(params,True)
        profiling_file = f"profiling_chimera_rss_{rank}.txt"
    if args.run_sub_mon:
        from mhs_fenicsx.chimera import interpolate_solution_to_inactive
        from mhs_fenicsx.problem.helpers import interpolate
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicChimeraSubstepper)
        lp.add_module(MonolithicDomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_function(interpolate_solution_to_inactive)
        lp.add_function(interpolate)
        lp_wrapper = lp(run_hodge)
        lp_wrapper(params,True)
        profiling_file = f"profiling_chimera_hodge_{rank}.txt"
    with open(profiling_file, 'w') as pf:
        lp.print_stats(stream=pf)
