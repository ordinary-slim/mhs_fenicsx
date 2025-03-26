import numpy as np
import yaml
from dolfinx import mesh
from mpi4py import MPI
from meshing import get_mesh
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.gcode import TrackType
from mhs_fenicsx.chimera import build_moving_problem
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper, ChimeraSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicChimeraSubstepper
from mhs_fenicsx.drivers import MonolithicRRDriver, MonolithicDomainDecompositionDriver, StaggeredRRDriver
import argparse
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_adim_back_len(fine_adim_dt : float = 0.5, adim_dt : float = 2):
    ''' Back length of moving domain'''
    return max(adim_dt + 4 * fine_adim_dt, 4)

def get_k(p):
    return p.materials[0].k.Ys[:-1].mean()

def get_h(p):
    radius = params["source_terms"][0]["radius"]
    el_density = np.round((1.0 / radius) * params["els_per_radius"]).astype(np.int32)
    return 1.0 / el_density

def write_gcode(params):
    gcode_lines = []
    L = params["part"][0]
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    gcode_lines.append(f"G0 F{speed:g} X{-L/2:g} Y0.0 Z0.0")
    gcode_lines.append(f"G1 X{+L/2:g} E0.1")
    gcode_file = params["source_terms"][0]["path"]
    with open(gcode_file,'w') as f:
        f.writelines("\n".join(gcode_lines))

def run_reference(params, writepos=True):
    domain = get_mesh(params)
    params["petsc_opts"] = params["petsc_opts_macro"]
    ps = Problem(domain, params, name="ref")

    ps.set_initial_condition(  params["environment_temperature"] )

    ps.set_forms()
    ps.compile_forms()
    adim_dt_print = params["substepping_parameters"]["micro_adim_dt"]
    adim_dt_cooling = params["substepping_parameters"]["cooling_adim_dt"]
    itime_step = 0
    while (not(ps.is_path_over()) and itime_step < params["max_timesteps"]):
        itime_step += 1
        track = ps.source.path.get_track(ps.time)
        if ps.source.path.get_track(ps.time).type in [TrackType.RECOATING,
                                                      TrackType.DWELLING]:
            ps.set_dt(ps.dimensionalize_waiting_timestep(track, adim_dt_cooling))
        else:
            ps.set_dt(ps.dimensionalize_mhs_timestep(track, adim_dt_print))
        ps.pre_iterate()
        ps.instantiate_forms()
        ps.pre_assemble()
        ps.non_linear_solve()
        ps.post_iterate()
        if writepos:
            ps.writepos(extra_funcs=[ps.u_av])
    return ps

def run_staggered(params, writepos=True):
    big_mesh = get_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, finalize_activation=False, name="staggered_rr")
    ps.set_initial_condition(  params["environment_temperature"] )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSStaggeredSubstepper(StaggeredRRDriver,
                                               [1.0, 1.0],
                                               ps,)
    ps = substeppin_driver.ps
    staggered_driver = substeppin_driver.staggered_driver
    staggered_driver.set_dirichlet_coefficients(get_h(ps), get_k(ps))

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos(extra_funcs=[ps.u_prev])
    return ps

def run_hodge(params, writepos=True):
    big_mesh = get_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, finalize_activation=True, name="hodge")
    ps.set_initial_condition(  params["environment_temperature"] )
    max_timesteps = params["max_timesteps"]
    substeppin_driver = MHSSemiMonolithicSubstepper(ps,)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps,pf]:
                p.writepos(extra_funcs=[p.u_prev])
    return ps

def run_staggered_chimera_rr(params, writepos=True):
    initial_relaxation_factors=[1.0,1.0]
    big_mesh = get_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, name=f"chimera_staggered_rr")
    pm = build_moving_problem(ps,
                              macro_params["moving_domain_params"]["els_per_radius"],
                              custom_get_adim_back_len=get_adim_back_len)
    ps.set_initial_condition(params["environment_temperature"])
    pm.set_initial_condition(params["environment_temperature"])

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSStaggeredChimeraSubstepper(
            StaggeredRRDriver,
            initial_relaxation_factors,
            ps, pm)

    staggered_driver = substeppin_driver.staggered_driver
    staggered_driver.set_dirichlet_coefficients(get_h(ps), get_k(ps))

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos()
    return ps

def run_chimera_hodge(params, writepos=True):
    big_mesh = get_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, name=f"chimera_hodge")
    pm = build_moving_problem(ps,
                              macro_params["moving_domain_params"]["els_per_radius"],
                              custom_get_adim_back_len=get_adim_back_len)
    ps.set_initial_condition(params["environment_temperature"])
    pm.set_initial_condition(params["environment_temperature"])

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSSemiMonolithicChimeraSubstepper(ps, pm)
    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps, pm]:
                p.writepos(extra_funcs=[p.u_prev])
    return ps
if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-ref', action='store_true')
    parser.add_argument('-ss','--run-stagg-sub',action='store_true')
    parser.add_argument('-sms','--run-hodge',action='store_true')
    parser.add_argument('-css','--run-chimera-stagg',action='store_true')
    parser.add_argument('-csms','--run-chimera-hodge',action='store_true')
    lp = LineProfiler()
    lp.add_module(Problem)
    args = parser.parse_args()
    profiling_file = ""
    if args.run_ref:
        lp_wrapper = lp(run_reference)
        lp_wrapper(params, writepos = True)
        profiling_file = f"profiling_ref_{rank}.txt"
    if args.run_stagg_sub:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(StaggeredRRDriver)
        lp_wrapper = lp(run_staggered)
        lp_wrapper(params, writepos = True)
        profiling_file = f"profiling_ss_{rank}.txt"
    if args.run_hodge:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicSubstepper)
        lp_wrapper = lp(run_hodge)
        lp_wrapper(params, writepos = True)
        profiling_file = f"profiling_sms_{rank}.txt"
    if args.run_chimera_stagg:
        lp.add_module(MHSStaggeredChimeraSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(MHSSubstepper)
        lp.add_module(MonolithicDomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp_wrapper = lp(run_staggered_chimera_rr)
        lp_wrapper(params, writepos = True)
        profiling_file = f"profiling_chimera_rss_{rank}.txt"
    if args.run_chimera_hodge:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicChimeraSubstepper)
        lp.add_module(MonolithicDomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp_wrapper = lp(run_chimera_hodge)
        lp_wrapper(params, True)
        profiling_file = f"profiling_chimera_hodge_{rank}.txt"
    if profiling_file:
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
