from mhs_fenicsx import problem
from mhs_fenicsx.chimera import build_moving_problem, interpolate_solution_to_inactive, shape_moving_problem
from mhs_fenicsx.drivers import StaggeredDNDriver, StaggeredRRDriver, MonolithicRRDriver
import numpy as np
import yaml
from mpi4py import MPI
from dolfinx import mesh
from line_profiler import LineProfiler
Problem = problem.Problem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

def get_adim_back_len(fine_adim_dt : float = 0.5, adim_dt : float = 2):
    ''' Back length of moving domain'''
    return 2.0 + (3 * adim_dt)

radius = params["source_terms"][0]["radius"]
T_env = params["environment_temperature"]
els_per_radius = params["els_per_radius"]

def get_el_size(resolution=4.0):
    return params["source_terms"][0]["radius"] / resolution
def get_dt(adim_dt):
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    return adim_dt * (radius / speed)

def chimera_pre_iterate(pf: Problem, pm : Problem):
    adim_dts = params["adim_dts"]
    thresholds = params["adim_tresholds"]
    adim_dt = adim_dts[-1]
    for idx, threshold in enumerate(thresholds):
        if pf.time < get_dt(threshold):
            adim_dt = adim_dts[idx-1]
            break
    print(f"Using adimensional dt {adim_dt} for time {pf.time} with thresholds {thresholds} and adim_dts {adim_dts}", flush=True)
    for p in [pf, pm]:
        p.set_dt(get_dt(adim_dt))

    prev_pm_active_nodes_mask = pm.active_nodes_func.x.array.copy()
    shape_moving_problem(pm)
    pm.intersect_problem(pf, finalize=False)
    pm.update_active_dofs()
    newly_activated_dofs = np.logical_and(pm.active_nodes_func.x.array,
                                          np.logical_not(prev_pm_active_nodes_mask)).nonzero()[0]
    num_dofs_to_interpolate = pm.domain.comm.allreduce(newly_activated_dofs.size)
    if num_dofs_to_interpolate > 0:
        pm.interpolate(pf, dofs_to_interpolate=newly_activated_dofs)
    for p in [pf, pm]:
        p.pre_iterate()
    physical_active_els = pf.local_active_els
    pm.intersect_problem(pf, finalize=False)

    pf.subtract_problem(pm, finalize=False)

    for p, p_ext in [(pm, pf), (pf, pm)]:
        p.finalize_activation()
        p.find_gamma(p_ext)#TODO: Re-use previous data here
    return physical_active_els

box = [-10*radius,-4*radius,+20*radius,+4*radius]
params["dt"] = get_dt(params["adim_dts"][0])
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
    p_fixed = problem.Problem(domain, params, name="welding")
    p_moving = build_moving_problem(p_fixed,
                                    2*els_per_radius,
                                    #custom_get_adim_back_len=get_adim_back_len,
                                    )
    (pf, pm) = (p_fixed, p_moving)

    for p in [pf, pm]:
        p.set_initial_condition(T_env)

    run_type = params["run_type"]
    if run_type == "staggered":
        minstaggits, maxstaggits = {dt : 1e3 for dt in params["adim_dts"]}, \
            {dt : -1 for dt in params["adim_dts"]}
        for p in [pf, pm]:
            p.name += "_staggered"
            p.initialize_post()
        dd_type=params["dd_type"]
        if dd_type=="robin":
            driver_type = StaggeredRRDriver
        elif dd_type=="dn":
            driver_type = StaggeredDNDriver
        else:
            raise ValueError("dd_type must be 'dn' or 'robin'")

        driver = driver_type(pm,
                             pf,
                             initial_relaxation_factors=params["initial_relaxation_factors"],
                             max_staggered_iters=params["max_staggered_iters"],
                             convergence_threshold=params["convergence_threshold"],
                             )

        if (type(driver)==StaggeredRRDriver):
            h = get_el_size(els_per_radius)
            k = float(params["material_metal"]["conductivity"])
            driver.dirichlet_coeff[driver.p1].value = 1.0 / 2.0
            driver.dirichlet_coeff[driver.p2].value =  k / h
            driver.relaxation_coeff[driver.p1].value = 3.0 / 3.0

        while pf.time < get_dt(params["adim_t_final"]):
            physical_active_els = chimera_pre_iterate(pf, pm)

            driver.pre_loop(prepare_subproblems=True, preassemble=True)
            for _ in range(driver.max_staggered_iters):
                driver.pre_iterate()
                driver.iterate()
                driver.post_iterate(verbose=True)
                #driver.writepos()
                if driver.convergence_crit < driver.convergence_threshold:
                    if driver.iter > maxstaggits[pf.adimensionalize_mhs_timestep(pf.source.path.current_track)]:
                        maxstaggits[pf.adimensionalize_mhs_timestep(pf.source.path.current_track)] = driver.iter
                    elif driver.iter < minstaggits[pf.adimensionalize_mhs_timestep(pf.source.path.current_track)]:
                        minstaggits[pf.adimensionalize_mhs_timestep(pf.source.path.current_track)] = driver.iter
                    break
            driver.post_loop()
            #TODO: Interpolate solution at inactive nodes
            interpolate_solution_to_inactive(pf,pm)
            for p in [pf, pm]:
                p.post_iterate()
                p.writepos()
            pf.set_activation(physical_active_els)
        print(f"Minstaggits: {minstaggits}, Maxstaggits: {maxstaggits}", flush=True)

    elif run_type == "monolithic":
        for p in [pf, pm]:
            p.name += "_monolithic"
            p.initialize_post()
        driver = MonolithicRRDriver(p_fixed, p_moving, 1.0, 1.0, quadrature_degree=2)

        for p in [p_fixed, p_moving]:
            p.set_forms()
            p.compile_forms()

        while pf.time < get_dt(params["adim_t_final"]):
            physical_active_els = chimera_pre_iterate(pf, pm)

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

    elif run_type == "ref":
        pf.name = "ref"
        pf.initialize_post()
        pf.set_forms()
        pf.compile_forms()
        itime_step = 0
        fine_dt = get_dt(params["fine_dt"])
        pf.set_dt(fine_dt)
        macro_dt = get_dt(params["adim_dts"][0])
        while pf.time < get_dt(params["adim_t_final"]):
            itime_step += 1
            pf.pre_iterate()
            pf.instantiate_forms()
            pf.pre_assemble()
            pf.non_linear_solve()
            pf.post_iterate()
            if (abs(pf.time / macro_dt - np.rint(pf.time / macro_dt)) < 1e-7):
                pf.writepos(extension="vtx")
        return pf
    else:
        raise ValueError("run_type must be 'staggered', 'monolithic' or 'ref'")


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
