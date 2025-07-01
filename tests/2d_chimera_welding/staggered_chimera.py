from mhs_fenicsx import problem
from mhs_fenicsx.chimera import build_moving_problem, interpolate_solution_to_inactive
from mhs_fenicsx.drivers import StaggeredDNDriver, StaggeredRRDriver
import numpy as np
import yaml
from mpi4py import MPI
from dolfinx import mesh
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

radius = params["source_terms"][0]["radius"]
max_temporal_iters = params["max_iter"]
T_env = params["environment_temperature"]
els_per_radius = params["els_per_radius"]

def get_el_size(resolution=4.0):
    return params["source_terms"][0]["radius"] / resolution
def get_dt(adim_dt):
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    return adim_dt * (radius / speed)

box = [-10*radius,-4*radius,+10*radius,+4*radius]
params["dt"] = get_dt(params["adim_dt"])
params["source_terms"][0]["initial_position"] = [-4*radius, 0.0, 0.0]

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

    for p in [p_fixed, p_moving]:
        p.set_initial_condition(T_env)

    dd_type=params["dd_type"]
    if dd_type=="robin":
        driver_type = StaggeredRRDriver
    elif dd_type=="dn":
        driver_type = StaggeredDNDriver
    else:
        raise ValueError("dd_type must be 'dn' or 'robin'")

    driver = driver_type(p_moving,
                         p_fixed,
                         initial_relaxation_factors=params["initial_relaxation_factors"],
                         max_staggered_iters=params["max_staggered_iters"])

    if (type(driver)==StaggeredRRDriver):
        h = 1.0 / get_el_size(els_per_radius)
        k = float(params["material_metal"]["conductivity"])
        driver.dirichlet_coeff[driver.p1].value = k / (np.sqrt(h))
        driver.dirichlet_coeff[driver.p2].value =  k / (np.sqrt(h))
        driver.relaxation_coeff[driver.p1].value = 3.0 / 3.0

    for _ in range(max_temporal_iters):
        for p in [p_fixed, p_moving]:
            p.pre_iterate()
        physical_active_els = p_fixed.local_active_els
        p_fixed.subtract_problem(p_moving, finalize=True)
        p_moving.find_gamma(p_fixed)

        driver.pre_loop(prepare_subproblems=True, preassemble=True)
        for _ in range(driver.max_staggered_iters):
            driver.pre_iterate()
            driver.iterate()
            driver.post_iterate(verbose=True)
            #driver.writepos()
            if driver.convergence_crit < driver.convergence_threshold:
                break
        driver.post_loop()
        #TODO: Interpolate solution at inactive nodes
        interpolate_solution_to_inactive(p_fixed,p_moving)
        for p in [p_fixed, p_moving]:
            p.post_iterate()
            p.writepos()
        p_fixed.set_activation(physical_active_els)

if __name__=="__main__":
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(problem)
        lp.add_function(build_moving_problem)
        lp.add_module(StaggeredDNDriver)
        lp_wrapper = lp(main)
        lp_wrapper()
        with open(f"staggered_profiling_{rank}.txt", 'w') as pf:
            lp.print_stats(stream=pf)

    else:
        main()
