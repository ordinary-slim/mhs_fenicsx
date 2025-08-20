import yaml
from mpi4py import MPI

from mhs_fenicsx.drivers import MonolithicRRDriver, DomainDecompositionDriver, StaggeredInterpRRDriver
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, ChimeraSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicChimeraSubstepper
from test_2d_substepping import get_initial_condition, get_dt, write_gcode, get_mesh, get_max_timestep
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

def run_staggered_RR(params, descriptor="", writepos=True):
    write_gcode(params)
    radius = params["source_terms"][0]["radius"]
    big_mesh = get_mesh(params, radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    descriptor = "chimera_ss_robin_" + descriptor
    ps = Problem(big_mesh, macro_params, name=descriptor)
    els_per_radius = macro_params["moving_domain_params"]["els_per_radius"]
    pm = build_moving_problem(ps, els_per_radius)
    initial_condition_fun = get_initial_condition(params)
    ps.set_initial_condition(  initial_condition_fun )
    pm.set_initial_condition(  initial_condition_fun )

    max_timesteps = get_max_timestep(params)

    substeppin_driver = MHSStaggeredChimeraSubstepper(
            StaggeredInterpRRDriver,
            ps, pm,
            staggered_relaxation_factors=[1.0,1.0],)

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
            ps.writepos(extension="vtx")
    return ps

def run_hodge(params, descriptor="", writepos=True):
    write_gcode(params)
    els_per_radius = params["els_per_radius"]
    radius = params["source_terms"][0]["radius"]
    big_mesh = get_mesh(params, radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    descriptor = f"chimera_sms_" + descriptor
    ps = Problem(big_mesh, macro_params, name=descriptor)
    els_per_radius = macro_params["moving_domain_params"]["els_per_radius"]
    pm = build_moving_problem(ps, els_per_radius)
    initial_condition_fun = get_initial_condition(params)
    ps.set_initial_condition(  initial_condition_fun )
    pm.set_initial_condition(  initial_condition_fun )

    max_timesteps = get_max_timestep(params)

    substeppin_driver = MHSSemiMonolithicChimeraSubstepper(ps, pm)
    pf = substeppin_driver.pf
    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps, pf]:
                p.writepos(extension="vtx")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss','--run-sub-sta',action='store_true')
    parser.add_argument('-sms','--run-sub-mon',action='store_true')
    parser.add_argument('-t','--testing',action='store_true')
    parser.add_argument('-tsms','--test-semi-mono-substep',action='store_true')
    parser.add_argument('-d', '--descriptor', default="")
    parser.add_argument('-i', '--input-file', default="input.yaml")

    args = parser.parse_args()
    input_file = args.input_file
    with open(input_file, 'r') as f:
        params = yaml.safe_load(f)

    lp = LineProfiler()
    lp.add_module(Problem)
    if args.run_sub_sta:
        from mhs_fenicsx.chimera import interpolate_solution_to_inactive
        from mhs_fenicsx.problem.helpers import interpolate_cg1
        lp.add_module(MHSStaggeredChimeraSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(MHSSubstepper)
        lp.add_module(DomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp.add_function(interpolate_solution_to_inactive)
        lp.add_function(interpolate_cg1)
        lp_wrapper = lp(run_staggered_RR)
        lp_wrapper(params, descriptor=args.descriptor, writepos = True)
        profiling_file = f"profiling_chimera_rss_{rank}.txt"
    if args.run_sub_mon:
        from mhs_fenicsx.chimera import interpolate_solution_to_inactive
        from mhs_fenicsx.problem.helpers import interpolate_cg1
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicChimeraSubstepper)
        lp.add_module(DomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp.add_function(interpolate_solution_to_inactive)
        lp.add_function(interpolate_cg1)
        lp_wrapper = lp(run_hodge)
        lp_wrapper(params, descriptor=args.descriptor, writepos = True)
        profiling_file = f"profiling_chimera_hodge_{rank}.txt"
    if args.testing:
        test_staggered_robin_chimera_substepper()
        test_sms_chimera_substepper()
        exit()
    if args.test_semi_mono_substep:
        test_sms_chimera_substepper()
        exit()
    with open(profiling_file, 'w') as pf:
        lp.print_stats(stream=pf)
