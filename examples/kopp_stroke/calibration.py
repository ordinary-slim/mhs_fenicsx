import numpy as np
import yaml
from dolfinx import mesh
from mpi4py import MPI
from mhs_fenicsx.problem import Problem
import argparse
from line_profiler import LineProfiler
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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

def run(params, writepos=True, descriptor=""):
    pm = build_moving_problem(params)
    pm.set_initial_condition(  params["environment_temperature"])
    pm.set_forms()
    pm.compile_create_forms()
    dt = float(float(params["adim_dt"]) * pm.source.R / np.linalg.norm(pm.advection_speed.value))
    pm.set_dt(dt)
    for _ in range(params["max_timesteps"]):
        pm.pre_iterate()
        pm.pre_assemble()
        pm.non_linear_solve()
        pm.post_iterate()
        pm.writepos(extension="vtx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--descriptor', default="")
    lp = LineProfiler()
    lp.add_module(Problem)
    args = parser.parse_args()
    params_file = "calibration_input.yaml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    lp_wrapper = lp(run)
    lp_wrapper(params, writepos = True, descriptor = args.descriptor)
    profiling_file = f"profiling_chimera_rss_{rank}.txt"
    if profiling_file:
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
