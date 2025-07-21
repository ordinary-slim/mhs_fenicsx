from dolfinx import fem, mesh, io
import yaml
from mesh import mesh_double_ellipse
import numpy as np
from mpi4py import MPI
from mhs_fenicsx import problem
from mhs_fenicsx.drivers import MonolithicRRDriver
from mhs_fenicsx.chimera import build_moving_problem, interpolate_solution_to_inactive

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main(params):
    mesh_moving = mesh_double_ellipse(params)

    def get_el_size(resolution=4.0):
        return params["source_terms"][0]["radius"] / resolution
    def get_dt(adim_dt):
        speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
        return adim_dt * (radius / speed)

    radius = params["source_terms"][0]["radius"]
    max_temporal_iters = params["max_iter"]
    T_env = params["environment_temperature"]
    els_per_radius_bg = params["els_per_radius_bg"]
    box = [-10*radius,-4*radius,+10*radius,+4*radius]
    params["dt"] = get_dt(params["adim_dt"])
    initial_position = radius*np.array(params["source_terms"][0]["initial_position"])
    params["source_terms"][0]["initial_position"] = initial_position

    point_density = np.round(1/get_el_size(els_per_radius_bg)).astype(np.int32)
    nx = np.round((box[2]-box[0]) * point_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * point_density).astype(np.int32)
    domain  = mesh.create_rectangle(MPI.COMM_WORLD,
              [box[:2], box[2:]],
              [nx, ny],
              mesh.CellType.quadrilateral,
              )

    p_fixed = problem.Problem(domain, params, name="monolithic")
    p_moving = build_moving_problem(p_fixed, els_per_radius_bg, domain=mesh_moving)

    p_fixed.set_initial_condition(T_env)
    p_moving.set_initial_condition(T_env)

    driver = MonolithicRRDriver(p_fixed, p_moving, 1.0, 1.0)

    for p in [p_fixed, p_moving]:
        p.set_forms()
        p.compile_forms()

    for _ in range(max_temporal_iters):
        for p in [p_fixed, p_moving]:
            p.pre_iterate()
        physical_active_els = p_fixed.local_active_els
        p_fixed.subtract_problem(p_moving, finalize=True)
        p_moving.find_gamma(p_fixed)

        for p in [p_fixed, p_moving]:
            p.instantiate_forms()
            p.pre_assemble()

        driver.non_linear_solve()

        driver.post_iterate()
        extra_funs = {
                p_fixed : [p_fixed.u_prev],
                p_moving : [p_moving.u_prev],
                }
        interpolate_solution_to_inactive(p_fixed,p_moving)
        for p in [p_fixed, p_moving]:
            p.post_iterate()
            p.writepos(extra_funcs=extra_funs[p])
        p_fixed.set_activation(physical_active_els)
    return p_fixed, p_moving


if __name__=="__main__":
    with open("input.yaml", "r") as file:
        params = yaml.safe_load(file)

    main(params)
