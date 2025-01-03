from mhs_fenicsx import problem
from mhs_fenicsx.drivers.monolithic_drivers import MonolithicRRDriver
import numpy as np
import yaml
from mpi4py import MPI
from helpers import build_moving_problem, interpolate_solution_to_inactive
from dolfinx import mesh
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main():
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    def get_el_size(resolution=4.0):
        return params["heat_source"]["radius"] / resolution
    def get_dt(adim_dt):
        speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
        return adim_dt * (radius / speed)
    radius = params["heat_source"]["radius"]
    max_temporal_iters = params["max_iter"]
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

    p_fixed = problem.Problem(domain, params, name="monolithic")
    p_moving = build_moving_problem(p_fixed,els_per_radius)

    p_fixed.set_initial_condition(T_env)
    p_moving.set_initial_condition(T_env)

    driver = MonolithicRRDriver(p_fixed, p_moving, quadrature_degree=2)

    for _ in range(max_temporal_iters):
        for p in [p_fixed, p_moving]:
            p.pre_iterate()
        p_fixed.subtract_problem(p_moving)
        p_moving.find_gamma(p_moving.get_active_in_external(p_fixed))

        for p in [p_fixed, p_moving]:
            p.set_forms_domain()
            p.set_forms_boundary()
            p.compile_forms()
            p.pre_assemble()
            p.assemble_residual()
            p.assemble_jacobian(finalize=False)
        driver.setup_coupling()
        for p in [p_fixed, p_moving]:
            p.A.assemble()
        driver.solve()

        driver.post_iterate()
        extra_funs = {
                p_fixed : [p_fixed.u_prev],
                p_moving : [p_moving.u_prev],
                }
        interpolate_solution_to_inactive(p_fixed,p_moving)
        for p in [p_fixed, p_moving]:
            p.post_iterate()
            p.writepos(extra_funcs=extra_funs[p])

if __name__=="__main__":
    lp = LineProfiler()
    lp.add_module(problem)
    lp.add_function(build_moving_problem)
    lp.add_function(interpolate_solution_to_inactive)
    lp.add_module(MonolithicRRDriver)
    lp_wrapper = lp(main)
    lp_wrapper()
    with open(f"monolithic_profiling_{rank}.txt", 'w') as pf:
        lp.print_stats(stream=pf)
