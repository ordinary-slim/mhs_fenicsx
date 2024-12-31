from mhs_fenicsx import problem
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
import yaml
from helpers import interpolate_solution_to_inactive, build_moving_problem
from line_profiler import LineProfiler
from mhs_fenicsx.drivers.staggered_drivers import StaggeredDNDriver, StaggeredRRDriver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

radius = params["heat_source"]["radius"]
max_temporal_iters = params["max_iter"]
T_env = params["environment_temperature"]
els_per_radius = params["els_per_radius"]

def get_el_size(resolution=4.0):
    return params["heat_source"]["radius"] / resolution
def get_dt(adim_dt):
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    return adim_dt * (radius / speed)

box = [-10*radius,-4*radius,+10*radius,+4*radius]
params["dt"] = get_dt(params["adim_dt"])
params["heat_source"]["initial_position"] = [-4*radius, 0.0, 0.0]

def main():
    point_density = np.round(1/get_el_size(els_per_radius)).astype(np.int32)
    nx = np.round((box[2]-box[0]) * point_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * point_density).astype(np.int32)
    domain  = mesh.create_rectangle(MPI.COMM_WORLD,
              [box[:2], box[2:]],
              [nx, ny],
              mesh.CellType.quadrilateral,
              )
    params["petsc_opts"] = params["petsc_opts_fixed"]
    p_fixed = problem.Problem(domain, params, name="staggered")
    p_moving = build_moving_problem(p_fixed,els_per_radius)

    p_fixed.set_initial_condition(T_env)
    p_moving.set_initial_condition(T_env)

    driver = StaggeredRRDriver(p_moving,p_fixed,params["max_staggered_iters"],
                               initial_relaxation_factors=[1.0,1.0])
    if (type(driver)==StaggeredRRDriver):
        h = get_el_size(els_per_radius)
        k = float(params["material_metal"]["conductivity"])
        driver.dirichlet_coeff[driver.p1] = 1/2.0
        driver.dirichlet_coeff[driver.p2] = k / h
        driver.relaxation_coeff[driver.p1].value = 1.0

    for _ in range(max_temporal_iters):
        p_fixed.pre_iterate()
        p_moving.pre_iterate()
        p_fixed.subtract_problem(p_moving)
        p_moving.find_gamma(p_moving.get_active_in_external(p_fixed))
        driver.pre_loop()
        for _ in range(driver.max_staggered_iters):
            driver.pre_iterate()
            driver.iterate()
            driver.post_iterate(verbose=True)
            driver.writepos()
            if driver.convergence_crit < driver.convergence_threshold:
                break
        driver.post_loop()
        #TODO: Interpolate solution at inactive nodes
        interpolate_solution_to_inactive(p_fixed,p_moving)
        p_moving.writepos()
        p_fixed.writepos()

if __name__=="__main__":
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(problem)
        lp.add_function(build_moving_problem)
        lp.add_module(StaggeredDNDriver)
        lp_wrapper = lp(main)
        lp_wrapper()
        with open(f"profiling_{rank}.txt", 'w') as pf:
            lp.print_stats(stream=pf)

    else:
        main()
