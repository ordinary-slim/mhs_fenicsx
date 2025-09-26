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
        shift = np.array([+0.00125 + ps.source.R / 4.0, 0.0, 0.0])
        if params["moving_domain_params"]["els_per_radius"] == 8:
            shift[0] += ps.source.R / 8.0
        pm = build_moving_problem(ps,
                                  params["moving_domain_params"]["els_per_radius"],
                                  custom_get_adim_back_len=get_adim_back_len,
                                  symmetries=[(1, 1)],
                                  shift=shift)
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

        params["substepping_parameters"]["chimera_driver"]["gamma_coeff1"] = gc1
        params["substepping_parameters"]["chimera_driver"]["gamma_coeff2"] = gc2

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

def set_speed(case, params):
    if case == 0:
        speed = 800.0
    else:
        speed = 1200.0
    assert ("initial_speed" in params["source_terms"][0])
    params["source_terms"][0]["initial_speed"][0] = speed
    return speed

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
        input_nu = params["material_in625"]["absorptivity"]
        nu = next(iter(p.absorptivity.values()))
        if isinstance(input_nu, list):
            num_vals = len(input_nu)
            if num_vals == 2:
                T0, nu0 = input_nu[0][0], input_nu[0][1]
                T1, nu1 = input_nu[1][0], input_nu[1][1]
                nu.ufl_operands[0].ufl_operands[1].value = T1
                nu.ufl_operands[1].value = nu1
                m = (nu1 - nu0) / (T1 - T0)
                c = nu0 - m * T0
                nu.ufl_operands[2].ufl_operands[0].value = c
                nu.ufl_operands[2].ufl_operands[1].ufl_operands[0].value = m
            elif num_vals == 3:
                T0, nu0 = input_nu[0][0], input_nu[0][1]
                T1, nu1 = input_nu[1][0], input_nu[1][1]
                T2, nu2 = input_nu[2][0], input_nu[2][1]
                m01 = (nu1 - nu0) / (T1 - T0)
                c01 = nu0 - m01 * T0
                m12 = (nu2 - nu1) / (T2 - T1)
                c12 = nu1 - m12 * T1
                nu.ufl_operands[0].ufl_operands[1].value = T2
                # If first conditional true
                nu.ufl_operands[1].value = nu2
                # Else second conditional
                nested_gt = nu.ufl_operands[2]
                nested_gt.ufl_operands[0].ufl_operands[1].value = T1
                # True of second conditional
                nested_gt.ufl_operands[1].ufl_operands[0].value = c12
                nested_gt.ufl_operands[1].ufl_operands[1].ufl_operands[0].value = m12
                # False of second conditional
                nested_gt.ufl_operands[2].ufl_operands[0].value = c01
                nested_gt.ufl_operands[2].ufl_operands[1].ufl_operands[0].value = m01
            else:
                raise Exception
        else:
            nu.ufl_operands[1].value = input_nu
            nu.ufl_operands[2].value = input_nu

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

def loop(params, run_type="calibration", writepos_every_iter=False):
    calibration_type = params["calibration_type"]
    requires_constant_absorptivity = {"Snu0n1depth"}
    domain = get_mesh(params, symmetry=True)

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

    def wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter):
        speed = set_speed(case, params)
        descriptor = (f"V{speed}-" + descriptor).replace(".", "_")
        return get_meltpool_dims(domain, params, descriptor,
                                 substepper=substepper,
                                 writepos_every_iter=writepos_every_iter)

    if run_type=="calibration":
        def set_params_Snu0nu1depth(S, nu1, nu2, depth_factor, case):
            for hs_params in params["source_terms"]:
                assert ("power" in hs_params)
                if "depth" in hs_params:
                    hs_params["depth"] = params["radius"] * depth_factor
            nu = nu1 if case == 0 else nu2
            params["material_in625"]["absorptivity"] = nu
            params["smoothing_cte_phase_change"] = S
            return f"S{S}-nu{nu}-d{depth_factor}"

        def set_params_absorptivities(nu0, s, nu_delta, T2, nu2, case):
            nu_params = params["material_in625"]["absorptivity"]
            T0 = nu_params[0][0]
            T1 = T0 + s*(T2 - T0)
            nu_c = nu0 + s*(nu2 - nu0)
            nu1 = nu_c + nu_delta
            nu_params[0][1] = nu0
            nu_params[1][0] = T1
            nu_params[1][1] = nu1
            nu_params[2][0] = T2
            nu_params[2][1] = nu2
            return f"nu0={nu0}-T1={T1}-nu1={nu1}-T2={T2}-nu2={nu2}"

        def set_params_Snus(S, nu0, s, nu_delta, T2, nu2, case):
            params["smoothing_cte_phase_change"] = S
            nu_params = params["material_in625"]["absorptivity"]
            T0 = nu_params[0][0]
            T1 = T0 + s*(T2 - T0)
            nu_c = nu0 + s*(nu2 - nu0)
            nu1 = nu_c + nu_delta
            nu_params[0][1] = nu0
            nu_params[1][0] = T1
            nu_params[1][1] = nu1
            nu_params[2][0] = T2
            nu_params[2][1] = nu2
            return f"S={S}-nu0={nu0}-T1={T1}-nu1={nu1}-T2={T2}-nu2={nu2}"

        def set_params_S_depth_nus(S, depth_factor, nu0, s, nu_delta, T2, nu2, case):
            params["smoothing_cte_phase_change"] = S
            for hs_params in params["source_terms"]:
                assert ("power" in hs_params)
                if "depth" in hs_params:
                    hs_params["depth"] = params["radius"] * depth_factor
            nu_params = params["material_in625"]["absorptivity"]
            T0 = nu_params[0][0]
            T1 = T0 + s*(T2 - T0)
            nu_c = nu0 + s*(nu2 - nu0)
            nu1 = nu_c + nu_delta
            nu_params[0][1] = nu0
            nu_params[1][0] = T1
            nu_params[1][1] = nu1
            nu_params[2][0] = T2
            nu_params[2][1] = nu2
            return f"S={S}-d={depth_factor}-nu0={nu0}-T1={T1}-nu1={nu1}-T2={T2}-nu2={nu2}"

        def set_params_T1nu1(s, nu_delta, case):
            nu_params = params["material_in625"]["absorptivity"]
            T0, nu0 = nu_params[0][0], nu_params[0][1]
            T2, nu2 = nu_params[2][0], nu_params[2][1]
            T1 = T0 + s*(T2 - T0)
            m02 = (nu2 - nu0)/(T2 - T0)
            nu_c = nu0 + m02*(T1 - T0)
            nu1 = nu_c + nu_delta
            nu_params[1][0] = T1
            nu_params[1][1] = nu1
            return f"T1={T1}-nu1={nu1}"

        def set_params_depth(depth_factor, case):
            for hs_params in params["source_terms"]:
                assert ("power" in hs_params)
                if "depth" in hs_params:
                    hs_params["depth"] = params["radius"] * depth_factor
            return f"d={depth_factor}"

        def iterate_Snu0nu1depth(smoothing_cte_phase_change, absorptivity1, absorptivity2, depth_factor, case=0, writepos_every_iter=False):
            descriptor = set_params_Snu0nu1depth(smoothing_cte_phase_change, absorptivity1, absorptivity2, depth_factor, case)
            return wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)

        def iterate_absorptivities(nu0, s, nu_delta, T2, nu2, case=0, writepos_every_iter=False):
            descriptor = set_params_absorptivities(nu0, s, nu_delta, T2, nu2, case)
            return wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)

        def iterate_Snus(S, nu0, s, nu_delta, T2, nu2, case=0, writepos_every_iter=False):
            descriptor = set_params_Snus(S, nu0, s, nu_delta, T2, nu2, case)
            return wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)

        def iterate_S_depth_nus(S, depth_factor, nu0, s, nu_delta, T2, nu2, case=0, writepos_every_iter=False):
            descriptor = set_params_S_depth_nus(S, depth_factor, nu0, s, nu_delta, T2, nu2, case)
            return wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)

        def iterate_T1nu1(s, nu_delta, case=0, writepos_every_iter=False):
            descriptor = set_params_T1nu1(s, nu_delta, case)
            return wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)

        def iterate_depth(depth_factor, case=0, writepos_every_iter=False):
            descriptor = set_params_depth(depth_factor, case)
            return wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)

        # Targets compensated with fine coarse bias
        target = {#0 : np.array([359.0, 66.0, 36.0]),
                  0 : np.array([362.41, 66.0, 36.0]),
                  #1 : np.array([370.0, 56.5, 29.0])}
                  1 : np.array([376.26, 56.5, 29.0])}

        cache = {case_idx : {} for case_idx in range(2)}

        def get_res(dims, target, echo=True):
            d, t = dims, target
            res = (d - t) / t

            if echo and rank == 0:
                print(f"t: L = {t[0]}, W = {t[1]}, T = {t[2]}",
                      flush=True)
                print(f"res = {res}", flush=True)
            return res

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
            return get_res(dims, target[case])

        def get_residual_case_absorptivities(params, case):
            dims = iterate_absorptivities(*params, case=case, writepos_every_iter=writepos_every_iter)
            return get_res(dims, target[case])

        def get_residual_case_T1nu1(params, case):
            dims = iterate_T1nu1(*params, case=case, writepos_every_iter=writepos_every_iter)
            return get_res(dims, target[case])

        def get_residual_case_depth(params, case):
            dims = iterate_depth(*params, case=case, writepos_every_iter=writepos_every_iter)
            return get_res(dims, target[case])

        def residuals_Snu0nu1depth(params):
            res0 = get_residual_case_Snu0nu1depth(params, 0)
            res1 = get_residual_case_Snu0nu1depth(params, 1)
            return np.hstack((res0, res1))

        def residuals_absorptivities(params):
            res0 = get_residual_case_absorptivities(params, 0)
            res1 = get_residual_case_absorptivities(params, 1)
            res0[0] *= 10.0
            res1[0] *= 10.0
            return np.hstack((res0, res1))

        def get_residual_case_Snus(params, case):
            dims = iterate_Snus(*params, case=case, writepos_every_iter=writepos_every_iter)
            return get_res(dims, target[case])

        def residuals_Snus(params):
            res0 = get_residual_case_Snus(params, 0)
            res1 = get_residual_case_Snus(params, 1)
            res0[0] *= 10.0
            res1[0] *= 10.0
            return np.hstack((res0, res1))

        #def get_residual_case_S_depth_nus(params, case):
        #    dims = iterate_S_depth_nus(*params, case=case, writepos_every_iter=writepos_every_iter)
        #    return get_res(dims, target[case])

        def residuals_S_depth_nus(params):
            dims0 = iterate_S_depth_nus(*params, case=0, writepos_every_iter=writepos_every_iter)
            dims1 = iterate_S_depth_nus(*params, case=1, writepos_every_iter=writepos_every_iter)
            res0 = get_res(dims0, target[0], echo=False)
            res1 = get_res(dims1, target[1], echo=False)

            # Ordering / separation penalty on L
            L0, L1 = dims0[0], dims1[0]
            Lt0, Lt1 = target[0][0], target[1][0]
            gap_t = Lt1 - Lt0
            gap_m = L1 - L0

            # Scale for normalization (avoid zero)
            gap_scale = max(abs(gap_t), 1e-6)
            sign_t = 1.0 if gap_t >= 0 else -1.0
            margin = 6.0 / gap_scale     # set >0 if you want a minimum separation
            tau = 0.05              # softness; smaller = sharper hinge

            # Smooth hinge: penalize when (sign_t * gap_m) < margin
            z = (margin - sign_t * (gap_m / gap_scale)) / tau
            # softplus ~ log(1+exp(z)) ≈ max(0,z) smoothly
            penalty = tau * np.log1p(np.exp(z))

            lambda_ord = 1.0
            r_ord = lambda_ord * penalty
            res = np.hstack((res0, res1, [r_ord]))

            if rank == 0:
                print(f"Case 0, prediction = {dims0}, target = {target[0]}", flush=True)
                print(f"Case 1, prediction = {dims1}, target = {target[1]}", flush=True)
                print(f"Combined residual: {res}", flush=True)

            return res

        def residuals_T1nu1(params):
            res0 = get_residual_case_T1nu1(params, 0)
            res1 = get_residual_case_T1nu1(params, 1)
            res0[0] *= 10.0
            res1[0] *= 10.0
            return np.hstack((res0, res1))

        def residuals_depth(params):
            res0 = get_residual_case_depth(params, 0)
            res1 = get_residual_case_depth(params, 1)
            res0[0] *= 0.5
            res1[0] *= 0.5
            return np.hstack((res0, res1))

        diff_step = None
        jac='2-point'
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
            # nu0, s, nu_delta, T2, nu2
            initial_guess = [0.6, 0.5, 0.0, 2400.0, 0.1]
            bounds = ([0.1,  1e-3, -np.inf,  500.0, 0.05],
                      [0.9, 0.999, +np.inf, 4000.0,  0.9])
            x_scale = [1.0, 1.0, 1.0, 1000.0, 1.0]
            diff_step = [1e-3]*len(initial_guess)
            close_to_sol = False
            if close_to_sol:
                method = 'lm'
                bounds = ([-np.inf]*len(initial_guess),
                          [+np.inf]*len(initial_guess))
            else:
                method = 'trf'
            residuals = residuals_absorptivities
        elif calibration_type == "Snus":
            # S, nu0, s, nu_delta, T2, nu2
            initial_guess = [0.30576, 0.56672, 0.723457353, 0.206325267, 1864.2652925328898, 0.05078731296148058]
            bounds = ([0.2, 0.2,  1e-3, -np.inf,  500.0, 0.05],
                      [0.4, 0.9, 0.999, +np.inf, 4000.0,  0.9])
            x_scale = [1.0, 1.0, 1.0, 1.0, 1000.0, 1.0]
            diff_step = [1e-3]*len(initial_guess)
            method = 'trf'
            residuals = residuals_Snus
        elif calibration_type == "S_depth_nus":
            # S depth nu0 s nu_delta T2 nu2
            initial_guess = [
                0.30576,
                0.42,
                0.5667207857263901,
                0.723457812,
                0.202333473,
                1864.2652925328898,
                0.05078731296148058,
            ]
            bounds = ([0.2, 0.01, 0.1, 1e-3, -np.inf,  500.0, 0.05],
                      [0.4, 1.0, 0.9, 0.999, +np.inf, 4000.0, 0.9])
            x_scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, 1.0]
            diff_step = [1e-3]*len(initial_guess)
            close_to_sol = False
            if close_to_sol:
                method = 'lm'
                bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,],
                          [+np.inf, +np.inf, +np.inf, +np.inf, +np.inf, +np.inf, +np.inf,])
            else:
                method = 'trf'
            residuals = residuals_S_depth_nus
        elif calibration_type == "T1nu1":
            # s, nu_delta
            initial_guess = [0.5, 0.0]
            bounds = ([1e-3,  -np.inf],
                      [0.999, +np.inf])
            x_scale = [1.0, 1.0]
            diff_step = [1e-3, 1e-3]
            residuals = residuals_T1nu1
            method = 'trf'
        elif calibration_type == "depth":
            # depth_factor
            initial_guess = [0.43165379]
            x_scale = [1.0]
            bounds = ([1e-3],
                      [0.999])
            residuals = residuals_depth
            method = 'trf'
        else:
            raise ValueError(f"Unknown calibration type: {calibration_type}")

        result = least_squares(residuals,
                               initial_guess,
                               verbose=2,
                               method=method,
                               x_scale=x_scale,
                               diff_step=diff_step,
                               jac=jac,
                               bounds=bounds)

        if rank == 0:
            print("Optimized parameters:", result.x)
    elif run_type == "simple":
        nus = np.linspace(0.3, 0.5, 10)
        for case in [0, 1]:
            for nu in nus:
                params["material_in625"]["absorptivity"] = nu
                descriptor = f"nu-{nu}"
                wrapper_get_meltpool_dims(case, descriptor, writepos_every_iter)
    else:
        raise ValueError("Unknown run type.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loop', action='store_true')
    parser.add_argument('-i', '--input-file',
                        default="calibration_input.yaml")
    parser.add_argument('-r', '--run-type',
                        default="calibration")
    parser.add_argument('-d', '--descriptor',
                        default="_test")
    args = parser.parse_args()
    params_file = args.input_file
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    if args.loop:
        loop(params, run_type=args.run_type, writepos_every_iter=False)
    else:
        domain = get_mesh(params, symmetry=True)
        get_meltpool_dims(domain, params, args.descriptor, writepos_every_iter=True,
                          remove=False)
