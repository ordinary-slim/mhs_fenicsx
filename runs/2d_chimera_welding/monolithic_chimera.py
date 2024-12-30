from mhs_fenicsx import problem
import numpy as np
import yaml
from mpi4py import MPI
from helpers import build_moving_problem
from dolfinx import mesh

def main():
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    def get_el_size(resolution=4.0):
        return params["heat_source"]["radius"] / resolution
    def get_dt(adim_dt):
        speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
        return adim_dt * (radius / speed)
    radius = params["heat_source"]["radius"]
    T_env = params["environment_temperature"]
    els_per_radius = params["els_per_radius"]
    box = [-10*radius,-4*radius,+10*radius,+4*radius]
    params["dt"] = get_dt(params["adim_dt"])
    params["heat_source"]["initial_position"] = [-4*radius, 0.0, 0.0]

    point_density = np.round(1/get_el_size(els_per_radius)).astype(np.int32)
    nx = np.round((box[2]-box[0]) * point_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * point_density).astype(np.int32)
    domain  = mesh.create_rectangle(MPI.COMM_WORLD,
              [box[:2], box[2:]],
              [nx, ny],
              mesh.CellType.quadrilateral,
              )

    p_fixed = problem.Problem(domain, params, name=params["case_name"])
    p_moving = build_moving_problem(p_fixed,els_per_radius)


if __name__=="__main__":
    main()
