import numpy as np
import yaml
from dolfinx import mesh
from mpi4py import MPI
from mhs_fenicsx.problem import Problem
import argparse
from petsc4py import PETSc
import subprocess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

paraview_python = r"/home/mslimani/bin/ParaView-5.13.2-MPI-Linux-Python3.10-x86_64/bin/pvpython"

def build_moving_problem(params):
    mdparams = params["moving_domain_params"]
    adim_back_len = 24
    adim_front_len = mdparams["adim_front_len"]
    adim_side_len = mdparams["adim_side_len"]
    adim_bot_len = mdparams["adim_bot_len"]
    adim_top_len = mdparams["adim_top_len"]
    center_of_mesh = np.zeros(3, dtype=np.float64)
    radius = params["radius"]
    els_per_radius = mdparams["els_per_radius"]
    el_size      = radius / float(els_per_radius)
    center_of_mesh = np.array(center_of_mesh)
    back_length  = radius * adim_back_len
    front_length = radius * adim_front_len
    side_length  = radius * adim_side_len
    bot_length   = radius * adim_bot_len
    top_length   = radius * adim_top_len
    mesh_bounds  = [
            center_of_mesh[0]-back_length,
            center_of_mesh[1]-side_length,
            center_of_mesh[2]-bot_length,
            center_of_mesh[0]+front_length,
            center_of_mesh[1]+side_length,
            center_of_mesh[2]+top_length,
            ]
    nx = np.rint((back_length+front_length)/el_size).astype(int)
    ny = np.rint(side_length*2/el_size).astype(int)
    nz = np.rint((top_length+bot_length)/el_size).astype(int)
    domain = mesh.create_box(MPI.COMM_WORLD,
                             [mesh_bounds[:3],mesh_bounds[3:]],
                             [nx,ny,nz],
                             mesh.CellType.hexahedron,
                             ghost_mode=mesh.GhostMode.shared_facet,
                             )
    # Placeholders
    params["domain_speed"] = np.array([0.0, 0.0, 0.0])
    params["advection_speed"] = -np.array(params["source_terms"][0]["initial_speed"])
    params["source_terms"][0]["initial_speed"] = [0.0, 0.0, 0.0]
    params["source_terms"][0]["initial_position"] = [0.0, 0.0, 0.0]
    p = Problem(domain, params,name="calibration")
    return p

def run(params, descriptor=""):
    pm = build_moving_problem(params)
    pm.set_initial_condition(  params["environment_temperature"])
    pm.set_forms()
    pm.name = "calibration_" + descriptor
    pm.compile_create_forms()
    dt = float(float(params["adim_dt"]) * pm.source.R / np.linalg.norm(pm.advection_speed.value))
    pm.set_dt(dt)
    for _ in range(params["max_timesteps"]):
        pm.pre_iterate()
        pm.pre_assemble()
        pm.non_linear_solve()
        pm.post_iterate()
    pm.writepos(extension="vtx")
    return pm

def iterate(smoothing_cte, absorptivity, depth_factor):
    params_file = "calibration_input.yaml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    params["smoothing_cte"] = smoothing_cte
    params["source_terms"][0]["power"] = 179.2 * absorptivity
    params["source_terms"][0]["depth"] = params["radius"] * depth_factor
    descriptor = f"S{smoothing_cte}-nu{absorptivity}-d{depth_factor}".replace(".", "_")
    pm = run(params, descriptor)
    bp_file = "./" + pm.result_folder + "/" + pm.name + ".bp/"
    result = subprocess.run([paraview_python, bp_file], capture_output=True, text=True)
    stdout = result.stdout

if __name__ == "__main__":
    iterate(0.25, 0.32, 0.28)
