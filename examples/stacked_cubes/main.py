import numpy as np
import yaml
from dolfinx import mesh
from mpi4py import MPI
from mesh import create_stacked_cubes_mesh
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.gcode import TrackType
from mhs_fenicsx.chimera import build_moving_problem
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper, ChimeraSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicChimeraSubstepper
from mhs_fenicsx.drivers import MonolithicRRDriver, MonolithicDomainDecompositionDriver, StaggeredRRDriver
import argparse
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def chimera_get_adim_back_len(fine_adim_dt: float = 0.5, adim_dt: float = 2):
    ''' Back length of moving domain'''
    return 4.0

def write_gcode(params):
    num_layers = params["num_layers"]
    layer_thickness = params["layer_thickness"]
    hatch_spacing = params["hatch_spacing"]
    width = params["width"]
    half_len = width / 2.0
    num_hatches = np.rint(width / hatch_spacing).astype(int)
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    gcode_lines = []
    gcode_lines.append(f"G0 F{speed:g}")
    p0, p1 = np.zeros(3), np.zeros(3)
    E = 0.0
    for ilayer in range(num_layers):
        if ilayer % 2 == 0:
            const_idx = 0
            mov_idx = 1
        else:
            const_idx = 1
            mov_idx = 0
        z = layer_thickness * (ilayer + 1)
        p0[2], p1[2] = z, z
        for ihatch in range(num_hatches):
            E += 0.1
            fixed_coord = -half_len + (ihatch + 0.5) * layer_thickness
            sign = (ihatch + 1) % 2
            mov_coord0 = (-1)**sign * half_len
            mov_coord1 = -mov_coord0
            p0[const_idx], p1[const_idx] = fixed_coord, fixed_coord
            p0[mov_idx], p1[mov_idx] = mov_coord0, mov_coord1
            if ihatch==0:
                gcode_lines.append(f"G4 X{p0[0]:g} Y{p0[1]:g} Z{z} P0.5")
                gcode_lines.append(f"G4 P0.5 R1")
            else:
                positionning_line = f"G0 X{p0[0]:g} Y{p0[1]:g}"
                gcode_lines.append(positionning_line)
            printing_line = f"G1 X{p1[0]:g} Y{p1[1]:g} E{E:g}"
            gcode_lines.append(printing_line)

    gcode_file = params["source_terms"][0]["path"]
    with open(gcode_file,'w') as f:
        f.writelines("\n".join(gcode_lines))

def get_k(p):
    return p.materials[0].k.Ys[:-1].mean()

def deactivate_below_surface(p):
    # Deactivate below surface
    midpoints_cells = mesh.compute_midpoints(p.domain, p.dim, np.arange(p.num_cells))
    substrate_els = (midpoints_cells[:, 2] <= 0.0).nonzero()[0]
    p.set_activation(substrate_els, finalize=True)
    p.update_material_at_cells(substrate_els, p.materials[1])

def run_reference(params, descriptor=""):
    writepos = params.get("writepos", True)
    domain = create_stacked_cubes_mesh(params)
    params["petsc_opts"] = params["petsc_opts_macro"]
    ps = Problem(domain, params, name="ref" + descriptor)

    ps.set_initial_condition(  params["environment_temperature"] )
    deactivate_below_surface(ps)

    ps.set_forms()
    ps.compile_forms()
    adim_dt_print = params["substepping_parameters"]["micro_adim_dt"]
    adim_dt_cooling = params["substepping_parameters"]["cooling_adim_dt"]
    itime_step = 0
    max_timesteps = params.get("max_timesteps", 1e9)
    while (not(ps.is_path_over()) and itime_step < max_timesteps):
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
            ps.writepos(extension="vtx", extra_funcs=[ps.u_av])
    return ps

def run_staggered(params, descriptor=""):
    writepos = params.get("writepos", True)
    domain = create_stacked_cubes_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False, name="staggered_rr" + descriptor)

    ps.set_initial_condition(  params["environment_temperature"] )
    deactivate_below_surface(ps)

    substeppin_driver = MHSStaggeredSubstepper(StaggeredRRDriver,
                                               ps,
                                               staggered_relaxation_factors=[1.0, 1.0],)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)
    staggered_driver = substeppin_driver.staggered_driver
    staggered_driver.set_dirichlet_coefficients(
            params["fine_el_size"], get_k(ps))

    max_timesteps = params.get("max_timesteps", 1e9)
    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos(extension="vtx", extra_funcs=[ps.u_av])
    return ps

def run_hodge(params, descriptor=""):
    writepos = params.get("writepos", True)
    domain = create_stacked_cubes_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False, name="hodge" + descriptor)

    ps.set_initial_condition(  params["environment_temperature"] )
    deactivate_below_surface(ps)

    substeppin_driver = MHSSemiMonolithicSubstepper(ps,)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)

    max_timesteps = params.get("max_timesteps", 1e9)
    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos(extension="vtx", extra_funcs=[ps.u_prev])
    return ps

def run_staggered_chimera_rr(params, writepos=True, descriptor=""):
    writepos = params.get("writepos", True)
    domain = create_stacked_cubes_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False,
                 name="chimera_staggered_rr" + descriptor)

    pm = build_moving_problem(ps,
                              macro_params["moving_domain_params"]["els_per_radius"],
                              #custom_get_adim_back_len=get_adim_back_len,
                              )
    for p in [ps, pm]:
        p.set_initial_condition(params["environment_temperature"])
        deactivate_below_surface(p)


    substeppin_driver = MHSStaggeredChimeraSubstepper(
        StaggeredRRDriver,
        ps, pm,
        staggered_relaxation_factors=[1.0, 1.0],)

    staggered_driver = substeppin_driver.staggered_driver
    staggered_driver.set_dirichlet_coefficients(
        params["fine_el_size"], get_k(ps))

    itime_step = 0
    max_timesteps = params.get("max_timesteps", 1e9)
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps, substeppin_driver.pf, pm]:
                p.writepos(extension="vtx")
    return ps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-ref', action='store_true')
    parser.add_argument('-ss','--run-stagg-sub', action='store_true')
    parser.add_argument('-sms','--run-hodge', action='store_true')
    parser.add_argument('-css','--run-chimera-stagg', action='store_true')
    parser.add_argument('-csms','--run-chimera-hodge', action='store_true')
    parser.add_argument('-d','--descriptor', default="")
    lp = LineProfiler()
    lp.add_module(Problem)
    args = parser.parse_args()
    params_file = "input.yaml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    profiling_file = ""
    if args.run_ref:
        lp_wrapper = lp(run_reference)
        lp_wrapper(params, descriptor = args.descriptor)
        profiling_file = f"profiling_ref_{rank}.txt"
    if args.run_stagg_sub:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(StaggeredRRDriver)
        lp_wrapper = lp(run_staggered)
        lp_wrapper(params, descriptor = args.descriptor)
        profiling_file = f"profiling_ss_{rank}.txt"
    if args.run_hodge:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicSubstepper)
        lp_wrapper = lp(run_hodge)
        lp_wrapper(params, descriptor = args.descriptor)
        profiling_file = f"profiling_sms_{rank}.txt"
    if args.run_chimera_stagg:
        lp.add_module(MHSStaggeredChimeraSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(MHSSubstepper)
        lp.add_module(MonolithicDomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp_wrapper = lp(run_staggered_chimera_rr)
        lp_wrapper(params, descriptor = args.descriptor)
        profiling_file = f"profiling_chimera_rss_{rank}.txt"
    if args.run_chimera_hodge:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicChimeraSubstepper)
        lp.add_module(MonolithicDomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp_wrapper = lp(run_chimera_hodge)
        lp_wrapper(params, descriptor = args.descriptor)
        profiling_file = f"profiling_chimera_hodge_{rank}.txt"
    if profiling_file:
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
