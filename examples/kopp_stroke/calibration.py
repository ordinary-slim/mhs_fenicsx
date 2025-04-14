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
from mhs_fenicsx.drivers import MonolithicRRDriver, MonolithicDomainDecompositionDriver, StaggeredRRDriver
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


def define_substepper(domain, params, descriptor):
    initial_relaxation_factors = [1.0, 1.0]
    params["petsc_opts"] = params["petsc_opts_macro"]
    ps = Problem(domain, params, name="chimera_staggered_rr" + descriptor,
                 finalize_activation=True)
    pm = build_moving_problem(ps,
                              params["moving_domain_params"]["els_per_radius"],
                              custom_get_adim_back_len=get_adim_back_len,
                              symmetries=[(1, 1)])
    for p in [ps, pm]:
        p.set_initial_condition(params["environment_temperature"])

    substeppin_driver = MHSStaggeredChimeraSubstepper(
            StaggeredRRDriver,
            initial_relaxation_factors,
            ps, pm)
    staggered_driver = substeppin_driver.staggered_driver
    staggered_driver.set_dirichlet_coefficients(
            get_h(ps, params["els_per_radius"]), get_k(ps))
    return substeppin_driver


def run_staggered_chimera_rr(domain, params, descriptor="",
                             writepos_every_iter=True, substepper=None):
    if rank == 0:
        write_gcode(params)
    comm.barrier()

    if substepper is None:
        substepper = define_substepper(domain, params, descriptor)
        ps, pf, pm = substepper.ps, substepper.pf, substepper.pm
    else:
        ps, pf, pm = substepper.ps, substepper.pf, substepper.pm
        base_name = {ps: "case",
                     pf: "case_micro_iters",
                     pm: "case_moving"}
        dx = pm.source.path.tracks[0].p0 - pm.source.x.copy()
        pm.move(dx)
        for p in [ps, pf, pm]:
            p.name = f"{base_name[p]}-{descriptor}"
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

    S = ps.smoothing_cte_phase_change.value.copy()
    itime_step = 0
    max_timesteps = params["max_timesteps"]
    while ((itime_step < max_timesteps) and not (ps.is_path_over())):
        itime_step += 1
        #print(f"{rank}: id of problem: {id(ps)} id of source: {id(ps.source)}, id of path: {id(ps.source.path)}, speed of first track: {ps.source.path.tracks[0].speed}", flush=True)
        substepper.do_timestep()
        for p in [ps, pf, pm]:
            if itime_step == 1:
                p.smoothing_cte_phase_change.value = S / 2.0
            else:
                p.smoothing_cte_phase_change.value = S
        writepos_every_iter = True
        if writepos_every_iter:
            ps.writepos(extension="vtx")
    if not (writepos_every_iter):
        ps.writepos(extension="vtx")
    return ps, pf, pm


def get_meltpool_dims(domain, params, descriptor, writepos_every_iter=False, remove=True, substepper=None):
    ps, pf, pm = run_staggered_chimera_rr(domain, params, descriptor,
                                          writepos_every_iter=writepos_every_iter,
                                          substepper=substepper)
    post_folder = "./" + ps.result_folder
    bp_file = post_folder + "/" + ps.name + ".bp/"
    if rank == 0:
        result = subprocess.run([paraview_python, "contour.py", bp_file],
                                capture_output=True, text=True)
        if remove:
            subprocess.run(["rm", "-rf", post_folder])
        stdout = result.stdout
        res = stdout.rstrip()
        res = res.split(",")
        res = [float(measurement) for measurement in res]
        print(f"L = {res[0]}, W = {res[1]}, T = {res[2]}", flush=True)
    else:
        res = None
    res = comm.bcast(res, root=0)
    return np.array(res)


def loop(params, writepos_every_iter=False):
    domain = get_mesh(params, symmetry=True)

    params_file = "calibration_input.yaml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    materials = []
    for key in params.keys():
        if key.startswith("material"):
            name = key.split('_')[-1]
            material = mhs_fenicsx.problem.Material(params[key], name=name)
            materials.append(material)
    params["materials"] = materials
    substepper = define_substepper(domain, params, "looping")

    def iterate(absorptivity, depth_factor, density_factor, latent_heat_factor, case=0, writepos_every_iter=False):
        for hs_params in params["source_terms"]:
            assert ("power" in hs_params)
            hs_params["power"] = 179.2 * absorptivity if hs_params["power"] > 0.0 else 0.0
            if "depth" in hs_params:
                hs_params["depth"] = params["radius"] * depth_factor
        #assert ("liquidus_temperature" in params["material_in625"]["phase_change"])
        #Tl = 1330 + (1370 - 1330) * fliquidus_tem
        #params["material_in625"]["phase_change"]["liquidus_temperature"] = Tl
        #materials[0].T_l.value = Tl
        if case == 0:
            speed = 0.8
        else:
            speed = 1.2
        assert ("initial_speed" in params["source_terms"][0])
        params["source_terms"][0]["initial_speed"][0] = speed
        #
        assert ("density" in params["material_in625"])
        rho = (1.0 + density_factor)*8.44E+03
        params["material_in625"]["density"] = rho
        materials[0].rho.value = rho
        #
        assert ("latent_heat" in params["material_in625"]["phase_change"])
        L = (1.0 + latent_heat_factor)*280.0E+03
        params["material_in625"]["phase_change"]["latent_heat"] = L
        materials[0].L.value = L

        descriptor = f"nu{absorptivity}-d{depth_factor}-rho{rho}-L{L}".replace(".", "_")
        return get_meltpool_dims(domain, params, descriptor,
                                 substepper=substepper,
                                 writepos_every_iter=writepos_every_iter)

    #target0 = np.array([359.0, 66.0, 36.0])
    target1 = np.array([370.0, 56.5, 29.0])

    def residuals(params):
        #dims0 = iterate(*params, case=0, writepos_every_iter=writepos_every_iter)
        #res0 = (dims0 - target0) / target0
        #if rank == 0:
        #    print(f"target: L = {target0[0]}, W = {target0[1]}, T = {target0[2]}",
        #          flush=True)
        #    print(f"res0 = {res0}", flush=True)
        dims1 = iterate(*params, case=1, writepos_every_iter=writepos_every_iter)
        res1 = (dims1 - target1) / target1
        if rank == 0:
            print(f"target: L = {target1[0]}, W = {target1[1]}, T = {target1[2]}",
                  flush=True)
            print(f"res1 = {res1}", flush=True)
        #return np.hstack((res0, res1))
        return res1

    #initial_guess = [0.22036165690870016, 0.34526776845021717, 0.3143414300451578, 0.02]
    #initial_guess = [0.2, 0.8, 1.0, 0.5]
    initial_guess = [0.77692017, 0.95789302, 0.0, 0.0]
    result = least_squares(residuals, initial_guess,
                           verbose=2, bounds=([0.0, 1e-3, -0.2, -0.4],
                                              [1.0, 2.0, +0.2, +0.4]),
                           x_scale=[1.0, 1.0, 0.01, 0.01],
                           )
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
