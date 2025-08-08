import numpy as np
import yaml
from dolfinx import mesh
from mpi4py import MPI
from mhs_fenicsx.problem import Problem
import argparse
from petsc4py import PETSc
import subprocess
from scipy.optimize import least_squares
import shutil
from meshing import get_mesh
from mhs_fenicsx.chimera import build_moving_problem
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper, ChimeraSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicChimeraSubstepper
from mhs_fenicsx.drivers import MonolithicRRDriver, DomainDecompositionDriver, StaggeredInterpRRDriver
from main import get_h, get_k, write_gcode
from mhs_fenicsx.gcode import gcode_to_path
import mhs_fenicsx.problem
from mhs_fenicsx.problem.heatsource import PenetratingGaussian

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

paraview_python = r"/data0/home/mslimani/bin/ParaView-5.13.3-MPI-Linux-Python3.10-x86_64/bin/pvpython"

def get_adim_back_len(fine_adim_dt : float = 0.5, adim_dt : float = 2):
    ''' Back length of moving domain'''
    return 8

def get_gamma_coeffs(p):
    els_per_radius1 = p.input_parameters["moving_domain_params"]["els_per_radius"]
    els_per_radius2 = p.input_parameters["els_per_radius"]
    els_per_radius = (els_per_radius1 + els_per_radius2) / 2.0
    radius = p.source.R
    el_size = radius / els_per_radius
    k = p.materials[-1].k.Ys.mean()
    a = 8.0
    return 1.0 / a, 2 * k / (a * el_size)

def define_substepper(domain, params, descriptor):
    run_type = params.get("run_type", "chimera")
    params["petsc_opts"] = params["petsc_opts_macro"]
    if run_type == "chimera":
        ps = Problem(domain, params, name="chimera_staggered_rr" + descriptor,
                     finalize_activation=True)
        pm = build_moving_problem(ps,
                                  params["moving_domain_params"]["els_per_radius"],
                                  custom_get_adim_back_len=get_adim_back_len,
                                  symmetries=[(1, 1)])
        for p in [ps, pm]:
            p.set_initial_condition(params["environment_temperature"])

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

        staggered_driver = substeppin_driver.staggered_driver
        staggered_driver.set_dirichlet_coefficients(
                get_h(ps, params["els_per_radius"]), get_k(ps))
    else:
        ps = Problem(domain, params, name="chimera_staggered_rr" + descriptor,
                     finalize_activation=True)
        ps.set_initial_condition(params["environment_temperature"])

        substeppin_driver = MHSStaggeredSubstepper(StaggeredInterpRRDriver,
                                                   ps,
                                                   staggered_relaxation_factors=[1.0, 1.0],)

        staggered_driver = substeppin_driver.staggered_driver
        staggered_driver.set_dirichlet_coefficients(
                get_h(ps, params["els_per_radius"]), get_k(ps))
    return substeppin_driver

def set_problem(p, params, problem_name):
    p.name = problem_name
    p.initialize_post()
    p.set_initial_condition(params["environment_temperature"])
    p.time = 0.0
    p.iter = 0
    for idx, source in enumerate(p.sources):
        hs_params = params["source_terms"][idx]
        path = gcode_to_path(hs_params["path"], default_power=hs_params["power"])
        source.set_path(path)
        source.power = source.path.tracks[0].power
        if "depth" in hs_params:
            source.depth = hs_params["depth"]
        source.R = hs_params["radius"]
        p.smoothing_cte_phase_change.value = float(params["smoothing_cte_phase_change"])
        # Absorptivity
        input_absorptivity = params["material_in625"]["absorptivity"]
        nu = next(iter(p.absorptivity.values()))
        if isinstance(input_absorptivity, list):
            T0, nu0 = input_absorptivity[0][0], input_absorptivity[0][1]
            T1, nu1 = input_absorptivity[1][0], input_absorptivity[1][1]
            nu.ufl_operands[0].ufl_operands[1].value = T1
            nu.ufl_operands[1].value = nu1
            m = (nu1 - nu0) / (T1 - T0)
            c = nu0 - m * T0
            nu.ufl_operands[2].ufl_operands[0].value = c
            nu.ufl_operands[2].ufl_operands[1].ufl_operands[0].value = m
        else:
            nu.ufl_operands[1].value = input_absorptivity
            nu.ufl_operands[2].ufl_operands[0].value = input_absorptivity
            nu.ufl_operands[2].ufl_operands[1].ufl_operands[0].value = 0.0

def run_simulation(domain, params, descriptor="",
                   writepos_every_iter=True, substepper=None):
    if rank == 0:
        write_gcode(params)
    comm.barrier()

    run_type = params.get("run_type", "chimera")

    skip_setting = False
    if substepper is None:
        substepper = define_substepper(domain, params, descriptor)
        skip_setting = True

    ps, pf, pm = substepper.ps, substepper.pf, None
    if run_type == "chimera":
        pm = substepper.pm

    if not(skip_setting):
        base_name = {ps: "case",
                     pf: "case_micro_iters"}

        if run_type == "chimera":
            base_name[pm] = "case_moving"
            dx = pm.source.path.tracks[0].p0 - pm.source.x.copy()
            pm.move(dx)

        for p in substepper.plist:
            set_problem(p, params, f"{base_name[p]}-{descriptor}")

    itime_step = 0
    max_timesteps = params["max_timesteps"]
    while ((itime_step < max_timesteps) and not (ps.is_path_over())):
        itime_step += 1
        substepper.do_timestep()
        if writepos_every_iter:
            ps.writepos(extension="vtx")
            if pm is not None:
                pm.writepos(extension="vtx")
    if not (writepos_every_iter):
        ps.writepos(extension="vtx")
    return substepper.plist

def get_meltpool_dims(domain, params, descriptor, writepos_every_iter=False, remove=True, substepper=None):
    plist = run_simulation(domain, params, descriptor,
                           writepos_every_iter=writepos_every_iter,
                           substepper=substepper)
    ps = plist[0]
    post_folder = "./" + ps.result_folder
    bp_file = post_folder + "/" + ps.name + ".bp/"
    if rank == 0:
        result = subprocess.run([paraview_python, "contour.py", bp_file],
                                capture_output=True, text=True)
        if remove:
            subprocess.run(["rm", "-rf", post_folder])
        stdout = result.stdout
        dims = stdout.rstrip()
        dims = dims.split(",")
        dims = [float(measurement) for measurement in dims]
        print(f"L = {dims[0]}, W = {dims[1]}, T = {dims[2]}", flush=True)
    else:
        dims = None
    dims = comm.bcast(dims, root=0)
    return np.array(dims)

def loop(params, writepos_every_iter=False):
    calibration_type = params["calibration_type"]
    requires_constant_absorptivity = {"Snu0n1depth"}
    domain = get_mesh(params, symmetry=True)

    params_file = "calibration_input.yaml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    materials = []
    for key in params.keys():
        if key.startswith("material"):
            name = key.split('_')[-1]
            input_nu = params[key]["absorptivity"]
            if calibration_type in requires_constant_absorptivity:
                params[key]["absorptivity"] = input_nu[0][1]
            material = mhs_fenicsx.problem.Material(params[key], name=name)
            materials.append(material)
    params["materials"] = materials
    substepper = define_substepper(domain, params, "looping")

    def set_speed(case):
        if case == 0:
            speed = 0.8
        else:
            speed = 1.2
        assert ("initial_speed" in params["source_terms"][0])
        params["source_terms"][0]["initial_speed"][0] = speed
        return speed

    def set_params_Snu0nu1depth(S, nu1, nu2, depth_factor, case):
        for hs_params in params["source_terms"]:
            assert ("power" in hs_params)
            if "depth" in hs_params:
                hs_params["depth"] = params["radius"] * depth_factor
        nu = nu1 if case == 0 else nu2
        params["material_in625"]["absorptivity"] = nu
        params["smoothing_cte_phase_change"] = S
        return f"S{S}-nu{nu}-d{depth_factor}"

    def set_params_absorptivity(nu0, T1, nu1, case):
        nu_params = params["material_in625"]["absorptivity"]
        nu_params[0][1] = nu0
        nu_params[1][0] = T1
        nu_params[1][1] = nu1
        return f"nu0={nu0}-T1={T1}-nu1={nu1}"

    def wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter):
        speed = set_speed(case)
        descriptor = (f"V{speed}-" + descriptor).replace(".", "_")
        return get_meltpool_dims(domain, params, descriptor,
                                 substepper=substepper,
                                 writepos_every_iter=writepos_every_iter)

    def iterate_Snu0nu1depth(smoothing_cte_phase_change, absorptivity1, absorptivity2, depth_factor, case=0, writepos_every_iter=False):
        descriptor = set_params_Snu0nu1depth(smoothing_cte_phase_change, absorptivity1, absorptivity2, depth_factor, case)
        return wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)

    def iterate_absorptivity(nu0, T1, nu1, case=0, writepos_every_iter=False):
        descriptor = set_params_absorptivity(nu0, T1, nu1, case)
        return wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)

    target = {0 : np.array([359.0, 66.0, 36.0]),
              1 : np.array([370.0, 56.5, 29.0])}

    cache = {case_idx : {} for case_idx in range(2)}

    def get_residual_case_Snu0nu1depth(params, case):
        # WARNING: This is hacky
        if case == 0:
            current_params = (params[0], params[1], params[3])
        else:
            current_params = (params[0], params[2], params[3])

        if current_params not in cache[case]:
            dims = iterate_Snu0nu1depth(*params, case=case, writepos_every_iter=writepos_every_iter)
            cache[case][current_params] = dims
        else:
            dims = cache[case][current_params]
            if rank == 0:
                print("Cache hit!", flush=True)
                print(f"params S = {current_params[0]}, nu = {current_params[1]}, d = {current_params[2]}",
                      flush=True)
                print(f"L = {dims[0]}, W = {dims[1]}, T = {dims[2]}", flush=True)

        t = target[case]
        res = (dims - t) / t

        if rank == 0:
            print(f"t: L = {t[0]}, W = {t[1]}, T = {t[2]}",
                  flush=True)
            print(f"res = {res}", flush=True)

        return res

    def get_residual_case_absorptivities(params, case):
        dims = iterate_absorptivity(*params, case=case, writepos_every_iter=writepos_every_iter)
        t = target[case]
        res = (dims - t) / t
        if rank == 0:
            print(f"t: L = {t[0]}, W = {t[1]}, T = {t[2]}",
                  flush=True)
            print(f"res = {res}", flush=True)

        return res

    def residuals_Snu0nu1depth(params):
        res0 = get_residual_case_Snu0nu1depth(params, 0)
        res1 = get_residual_case_Snu0nu1depth(params, 1)
        return np.hstack((res0, res1))

    def residuals_absorptivities(params):
        res0 = get_residual_case_absorptivities(params, 0)
        res1 = get_residual_case_absorptivities(params, 1)
        return np.hstack((res0, res1))

    if calibration_type == "Snu0n1depth":
        #initial_guess = [0.22036165690870016, 0.34526776845021717, 0.3143414300451578, 0.02]
        #initial_guess = [0.2, 0.8, 1.0, 0.5]
        #S, nu1, nu2, depth_factor
        initial_guess = [0.3, 0.3, 0.3, 0.3]
        #initial_guess = [0.23336689, 0.33574148, 0.37668996, 0.32177079]
        bounds = ([0.1, 0.0, 0.0, 0.0],
                  [2.0, 1.0, 1.0, 1.0])
        close_to_sol = False
        if close_to_sol:
            method = 'lm'
            bounds = ([-np.inf, -np.inf, -np.inf, -np.inf,],
                      [+np.inf, +np.inf, +np.inf, +np.inf,])
        else:
            method = 'trf'
        residuals = residuals_Snu0nu1depth
    elif calibration_type == "absorptivities":
        initial_guess = [0.3, 2000.0, 0.4]
        bounds = ([0.1,  500.0, 0.1],
                  [0.9, 4000.0, 0.9])
        close_to_sol = False
        if close_to_sol:
            method = 'lm'
            bounds = ([-np.inf, -np.inf, -np.inf, -np.inf,],
                      [+np.inf, +np.inf, +np.inf, +np.inf,])
        else:
            method = 'trf'
        residuals = residuals_absorptivities
    else:
        raise ValueError(f"Unknown calibration type: {calibration_type}")

    result = least_squares(residuals,
                           initial_guess,
                           verbose=2,
                           method=method,
                           bounds=bounds)

    if rank == 0:
        print("Optimized parameters:", result.x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loop', action='store_true')
    parser.add_argument('-i', '--input-file',
                        default="calibration_input.yaml")
    parser.add_argument('-d', '--descriptor',
                        default="_test")
    args = parser.parse_args()
    params_file = args.input_file
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    if args.loop:
        loop(params, writepos_every_iter=False)
    else:
        domain = get_mesh(params, symmetry=True)
        get_meltpool_dims(domain, params, args.descriptor, writepos_every_iter=True,
                          remove=False)
