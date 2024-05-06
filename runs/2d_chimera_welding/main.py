from mhs_fenicsx.problem import Problem
from mhs_fenicsx.drivers.single_problem_driver import SingleProblemDriver
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
import yaml
from helpers import mesh_around_hs, build_moving_problem

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

radius = params["heat_source"]["radius"]
max_iter = params["max_iter"]

def get_el_size(resolution=2.0):
    return params["heat_source"]["radius"] / resolution
def get_dt(adim_dt):
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    return adim_dt * (radius / speed)

box = [-7*radius,-3*radius,+7*radius,+3*radius]
params["dt"] = get_dt(params["adim_dt"])
params["heat_source"]["initial_position"] = [-5*radius, 0.0, 0.0]

def main():
    point_density = np.round(1/get_el_size()).astype(np.int32)
    nx = np.round((box[2]-box[0]) * point_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * point_density).astype(np.int32)
    domain  = mesh.create_rectangle(MPI.COMM_WORLD,
              [box[:2], box[2:]],
              [nx, ny],
              mesh.CellType.quadrilateral,
              )
    p_fixed = Problem(domain, params, name=params["case_name"])
    p_moving = build_moving_problem(p_fixed)
    for _ in range(10):
        p_fixed.pre_iterate()
        p_moving.pre_iterate()
        p_fixed.writepos()
        p_moving.writepos()

if __name__=="__main__":
    main()
