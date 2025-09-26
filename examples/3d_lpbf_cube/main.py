import numpy as np
import yaml
from dolfinx import mesh
from mpi4py import MPI
from mesh import create_stacked_cubes_mesh
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.gcode import TrackType
from mhs_fenicsx.chimera import build_moving_problem
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper, ChimeraSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicChimeraSubstepper
from mhs_fenicsx.drivers import StaggeredRRDriver, MonolithicRRDriver, DomainDecompositionDriver, StaggeredInterpRRDriver
import argparse
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def chimera_get_adim_back_len(fine_adim_dt: float = 0.5, adim_dt: float = 2):
    ''' Back length of moving domain'''
    back_len = np.round(4*(1 + 1.5*adim_dt + (adim_dt**2.3)*fine_adim_dt)) / 4
    if rank == 0:
        print(f"adim dt = {adim_dt}, Back length = {back_len}", flush=True)
    return back_len

def get_pm(ps):
    params = ps.input_parameters
    hr = float(params["half_radius"])
    els_per_radius = 2*(hr / params["fine_el_size"])
    return build_moving_problem(ps,
                                els_per_radius,
                                custom_get_adim_back_len=chimera_get_adim_back_len,
                                shift=np.array([0.0, hr/2, 0.0]),
                                )

def get_dof_ratio(pf, ps):
    nlocdofs = {p: p.domain.topology.index_map(0).size_local for p in [pf, ps]}
    nactivelocdofs = {p: p.active_nodes_func.x.array[:nlocdofs[p]].sum() for p in [pf, ps]}
    nactivedofs = {p: p.domain.comm.allreduce(nactivelocdofs[p]) for p in [pf, ps]}
    return nactivedofs[pf] / nactivedofs[ps]

class Logger:
    def __init__(self):
        self.dof_ratios = []
        self.num_micro_iters = []

    def add_data(self, substepper):
        (pf, ps) = (substepper.pf, substepper.ps)
        if pf.source.path.current_track.type == TrackType.PRINTING:
            self.dof_ratios.append(get_dof_ratio(pf, ps))
            self.num_micro_iters.append(substepper.micro_iter)

    def __str__(self):
        if not self.dof_ratios:
            return "Logger: no data collected"
        mean_ratio = np.mean(self.dof_ratios)
        mean_num_micro_iters = np.mean(self.num_micro_iters)
        return f"Logger: mean DOF ratio = {mean_ratio:.3f}, mean num. micro-iters = {mean_num_micro_iters} (from {len(self.dof_ratios)} time-steps)"

def get_max_timesteps(params):
    mt = params.get("max_timesteps", 1e9)
    mt = mt if mt >= 0 else 1e9
    return mt

def get_gamma_coeffs(p):
    el_size = p.input_parameters["fine_el_size"]
    k = p.materials[-1].k.Ys.mean()
    a = 8.0
    return 1.0 / a, 2 * k / (a * el_size)
    #a = 2.0
    #return a * k / np.sqrt(el_size), k / np.sqrt(el_size) / a

def write_gcode(params):
    num_layers = params["num_layers"]
    layer_thickness = params["layer_thickness"]
    hatch_spacing = params["hatch_spacing"]
    width = params["width"]
    half_len = width / 2.0
    num_hatches = np.rint(width / hatch_spacing).astype(int)
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    pre_recoating_dwell_time = params["pre_recoating_dwell_time"]
    post_recoating_dwell_time = params["post_recoating_dwell_time"]
    final_dwelling_time = params["final_dwelling_time"]
    gcode_lines = []
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
            fixed_coord = -half_len + (ihatch + 0.5) * hatch_spacing
            sign = (ihatch + 1) % 2
            mov_coord0 = (-1)**sign * half_len
            mov_coord1 = -mov_coord0
            p0[const_idx], p1[const_idx] = fixed_coord, fixed_coord
            p0[mov_idx], p1[mov_idx] = mov_coord0, mov_coord1
            if ihatch==0:
                gcode_lines.append(f"G0 X{p0[0]:g} Y{p0[1]:g} Z{z} F{speed:g}")
                gcode_lines.append(f"G4 P{pre_recoating_dwell_time}")
                gcode_lines.append(f"G4 P{post_recoating_dwell_time} R1")
            else:
                positionning_line = f"G0 X{p0[0]:g} Y{p0[1]:g}"
                gcode_lines.append(positionning_line)
            printing_line = f"G1 X{p1[0]:g} Y{p1[1]:g} E{E:g}"
            gcode_lines.append(printing_line)
    gcode_lines.append(f"G4 P{final_dwelling_time}")

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

def do_timestep(p):
    p.pre_iterate()
    p.instantiate_forms()
    p.pre_assemble()
    p.non_linear_solve()
    p.post_iterate()

def do_printing_timestep(p):
    ''' Wrapper for profiling '''
    do_timestep(p)

def do_cooling_timestep(p):
    ''' Wrapper for profiling '''
    do_timestep(p)

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
    macro_adim_dt_print = params["substepping_parameters"]["macro_adim_dt"]
    adim_dt_cooling = params["substepping_parameters"]["cooling_adim_dt"]
    adim_dt_dwelling = params["substepping_parameters"]["dwelling_adim_dt"]
    final_adim_dt_dwelling = params["substepping_parameters"]["final_dwelling_adim_dt"]
    itime_step = 0
    max_timesteps = get_max_timesteps(params)
    while (not(ps.is_path_over()) and itime_step < max_timesteps):
        track = ps.source.path.get_track(ps.time)
        if track.type == TrackType.PRINTING:
            # Set timestep
            ps.set_dt(ps.dimensionalize_mhs_timestep(track, adim_dt_print))
            ps.cap_timestep()
            itime_step += ps.adimensionalize_mhs_timestep(ps.source.path.current_track) / macro_adim_dt_print

            do_printing_timestep(ps)
        else:
            if track.type == TrackType.COOLING:
                adim_dt = adim_dt_cooling
            else:
                if track == ps.source.path.tracks[-1]:
                    adim_dt = final_adim_dt_dwelling
                else:
                    adim_dt = adim_dt_dwelling
            ps.set_dt(ps.dimensionalize_waiting_timestep(track, adim_dt))
            itime_step += 1

            do_cooling_timestep(ps)

        itime_step = np.round(itime_step, 5)
        if writepos and (itime_step % 1 == 0):
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

    substeppin_driver = MHSStaggeredSubstepper(StaggeredInterpRRDriver,
                                               ps,
                                               staggered_relaxation_factors=[1.0, 1.0],)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)
    staggered_driver = substeppin_driver.staggered_driver
    staggered_driver.set_dirichlet_coefficients(
            params["fine_el_size"], get_k(ps))

    max_timesteps = get_max_timesteps(params)
    itime_step = 0
    logger = Logger()
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        logger.add_data(substeppin_driver)
        if writepos:
            ps.writepos(extension="vtx", extra_funcs=[ps.u_av])
    if rank == 0:
        print(logger, flush=True)
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

    max_timesteps = get_max_timesteps(params)
    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos(extension="vtx", extra_funcs=[ps.u_prev])
    return ps

def run_staggered_chimera_rr(params, descriptor=""):
    writepos = params.get("writepos", True)
    domain = create_stacked_cubes_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False,
                 name="chimera_staggered_rr" + descriptor)

    pm = get_pm(ps)
    for p in [ps, pm]:
        p.set_initial_condition(params["environment_temperature"])
        deactivate_below_surface(p)


    is_staggered = (params["substepping_parameters"]["chimera_driver"]["type"] == "staggered")
    if is_staggered:
        gc1, gc2 = get_gamma_coeffs(ps)
    else:
        gc1, gc2 = 1.0, 1.0

    params["substepping_parameters"]["chimera_driver"]["gamma_coeff1"] = gc1
    params["substepping_parameters"]["chimera_driver"]["gamma_coeff2"] = gc2

    substeppin_driver = MHSStaggeredChimeraSubstepper(
        StaggeredInterpRRDriver,
        ps, pm,
        staggered_relaxation_factors=[1.0, 1.0],)
    pf = substeppin_driver.pf

    staggered_driver = substeppin_driver.staggered_driver
    staggered_driver.set_dirichlet_coefficients(
        params["fine_el_size"], get_k(ps))

    itime_step = 0
    logger = Logger()
    max_timesteps = get_max_timesteps(params)
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        logger.add_data(substeppin_driver)
        if writepos:
            ps.writepos(extension="vtx")
    if rank == 0:
        if is_staggered:
            print(f"Average staggered iter: {substeppin_driver.chimera_driver.get_average_staggered_iter()}")
        print(logger, flush=True)
    return ps

def run_chimera_hodge(params, descriptor=""):
    writepos = params.get("writepos", True)
    domain = create_stacked_cubes_mesh(params)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False,
                 name="chimera_hodge" + descriptor)

    pm = get_pm(ps)
    for p in [ps, pm]:
        p.set_initial_condition(params["environment_temperature"])
        deactivate_below_surface(p)



    if params["substepping_parameters"]["chimera_driver"]["type"] == "staggered":
        gc1, gc2 = get_gamma_coeffs(ps)
    else:
        gc1, gc2 = 1.0, 1.0

    params["substepping_parameters"]["chimera_driver"]["gamma_coeff1"] = gc1
    params["substepping_parameters"]["chimera_driver"]["gamma_coeff2"] = gc2
    substeppin_driver = MHSSemiMonolithicChimeraSubstepper(ps, pm)

    itime_step = 0
    max_timesteps = get_max_timesteps(params)
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos(extension="vtx")
    return ps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '-ref', '--run-ref', action='store_true')
    parser.add_argument('-ss','--run-stagg-sub', action='store_true')
    parser.add_argument('-sms','--run-hodge', action='store_true')
    parser.add_argument('-css','--run-chimera-stagg', action='store_true')
    parser.add_argument('-csms','--run-chimera-hodge', action='store_true')
    parser.add_argument('-d','--descriptor', default="")
    parser.add_argument('-i','--input-file', default="input.yaml")
    lp = LineProfiler()
    lp.add_module(Problem)
    args = parser.parse_args()
    params_file = args.input_file
    desc = args.descriptor
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    profiling_file = ""
    if desc and desc[0] != "_":
        desc = "_" + desc
    if args.run_ref:
        lp.add_function(do_timestep)
        lp.add_function(do_printing_timestep)
        lp.add_function(do_cooling_timestep)
        lp_wrapper = lp(run_reference)
        lp_wrapper(params, descriptor = desc)
        profiling_file = f"profiling_ref_{desc}_{rank}.txt"
    if args.run_stagg_sub:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(StaggeredInterpRRDriver)
        lp_wrapper = lp(run_staggered)
        lp_wrapper(params, descriptor = desc)
        profiling_file = f"profiling_ss_{desc}_{rank}.txt"
    if args.run_hodge:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicSubstepper)
        lp_wrapper = lp(run_hodge)
        lp_wrapper(params, descriptor = desc)
        profiling_file = f"profiling_sms_{desc}_{rank}.txt"
    if args.run_chimera_stagg:
        lp.add_module(MHSStaggeredChimeraSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(MHSSubstepper)
        lp.add_module(DomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(StaggeredRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp_wrapper = lp(run_staggered_chimera_rr)
        lp_wrapper(params, descriptor = desc)
        profiling_file = f"profiling_css_{desc}_{rank}.txt"
    if args.run_chimera_hodge:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicChimeraSubstepper)
        lp.add_module(DomainDecompositionDriver)
        lp.add_module(MonolithicRRDriver)
        lp.add_module(StaggeredRRDriver)
        lp.add_module(ChimeraSubstepper)
        lp_wrapper = lp(run_chimera_hodge)
        lp_wrapper(params, descriptor = desc)
        profiling_file = f"profiling_csms_{desc}_{rank}.txt"
    if profiling_file:
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
