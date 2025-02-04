from line_profiler import LineProfiler
from mpi4py import MPI
from dolfinx import mesh
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper
import yaml
import numpy as np
import argparse
from mhs_fenicsx.problem.helpers import assert_pointwise_vals, print_vals
from mhs_fenicsx.drivers import StaggeredRRDriver, StaggeredDNDriver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_initial_condition(params):
    T_env = np.float64(params["environment_temperature"])
    if params["initial_condition"]:
        initial_condition_fun = lambda x : abs(300*np.cos(4*(x[0]-0.5)*(x[1]-0.5)))
    else:
        initial_condition_fun = lambda x : T_env*np.ones_like(x[0])
    return initial_condition_fun

def get_dt(adim_dt, radius, speed):
    return adim_dt * (radius / speed)

def write_gcode(params):
    (Lx, Ly) = (params["domain_width"], params["domain_height"])
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    hlenx = + 3.0 * Lx / 8.0
    hleny = + 3.0 * Ly / 8.0
    gcode_lines = []
    gcode_lines.append(f"G0 F{speed} X-{hlenx} Y-{hleny} Z0\n")
    gcode_lines.append(f"G1 X+{hlenx} E0.1\n")
    gcode_lines.append(f"G1 Y+{hleny} E0.2\n")
    gcode_lines.append(f"G1 X-{hlenx} E0.3\n")
    gcode_lines.append(f"G1 Y-{hleny} E0.4\n")
    with open(params["path"],'w') as f:
        f.writelines(gcode_lines)

def get_mesh(params, els_per_radius, radius, dim):
    el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
    (Lx, Ly) = (params["domain_width"], params["domain_height"])
    Lz = params["domain_depth"] if "domain_depth" in params else 1.0
    box = [-Lx/2.0, -Ly/2.0, 0.0, +Lx/2.0, +Ly/2.0, Lz]
    nx = np.round((box[3]-box[0]) * el_density).astype(np.int32)
    ny = np.round((box[4]-box[1]) * el_density).astype(np.int32)
    nz = np.round((box[5]-box[2]) * el_density).astype(np.int32)
    if dim==2:
        return mesh.create_rectangle(MPI.COMM_WORLD,
               [box[:2], box[3:5]],
               [nx, ny],
               #mesh.CellType.quadrilateral,
               )
    else:
        return mesh.create_box(MPI.COMM_WORLD,
               [box[:3], box[3:]],
               [nx, ny, nz],
               )


def run_staggered(params, driver_type, els_per_radius, writepos=True):
    radius = params["heat_source"]["radius"]
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    if   driver_type=="robin":
        driver_constructor = StaggeredRRDriver
        initial_relaxation_factors=[1.0,1.0]
    elif driver_type=="dn":
        driver_constructor = StaggeredDNDriver
        initial_relaxation_factors=[0.5,1]
    else:
        raise ValueError("Undefined staggered driver type.")
    big_mesh = get_mesh(params, els_per_radius, radius, 2)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=f"big_ss_{driver_type}")
    initial_condition_fun = get_initial_condition(params)
    big_p.set_initial_condition(  initial_condition_fun )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSStaggeredSubstepper(big_p,writepos=(params["substepper_writepos"] and writepos), do_predictor=params["predictor_step"])
    (ps,pf) = (substeppin_driver.ps,substeppin_driver.pf)
    staggered_driver = driver_constructor(pf,ps,
                                   max_staggered_iters=params["max_staggered_iters"],
                                   initial_relaxation_factors=initial_relaxation_factors,)

    if (type(staggered_driver)==StaggeredRRDriver):
        el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
        h = 1.0 / el_density
        k = float(params["material_metal"]["conductivity"])
        staggered_driver.set_dirichlet_coefficients(h, k)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.update_fast_problem()
        substeppin_driver.set_staggered_driver(staggered_driver)
        staggered_driver.pre_loop(prepare_subproblems=False)
        substeppin_driver.pre_loop()
        if substeppin_driver.do_predictor:
            substeppin_driver.predictor_step(writepos=substeppin_driver.do_writepos and writepos)
        staggered_driver.prepare_subproblems()
        for _ in range(staggered_driver.max_staggered_iters):
            substeppin_driver.pre_iterate()
            staggered_driver.pre_iterate()
            substeppin_driver.iterate()
            substeppin_driver.post_iterate()
            staggered_driver.post_iterate(verbose=True)
            if writepos:
                substeppin_driver.writepos(case="macro")

            if staggered_driver.convergence_crit < staggered_driver.convergence_threshold:
                break
        substeppin_driver.post_loop()
        # Interpolate solution to inactive ps
        ps.u.x.array[substeppin_driver.dofs_fast] = pf.u.x.array[substeppin_driver.dofs_fast]
        ps.u.x.scatter_forward()
        ps.is_grad_computed = False
        pf.u.x.array[:] = ps.u.x.array[:]
        pf.is_grad_computed = False
        if writepos:
            ps.writepos()
    return big_p

def run_semi_monolithic(params, els_per_radius, writepos=True):
    radius = params["heat_source"]["radius"]
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    big_mesh = get_mesh(params, els_per_radius, radius, 2)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=f"big_sms")
    initial_condition_fun = get_initial_condition(params)
    big_p.set_initial_condition(  initial_condition_fun )

    max_timesteps = params["max_timesteps"]
    substeppin_driver = MHSSemiMonolithicSubstepper(big_p,writepos=(params["substepper_writepos"] and writepos), do_predictor=params["predictor_step"])

    itime_step = 0
    while ((itime_step < max_timesteps) and not(big_p.is_path_over())):
        itime_step += 1
        substeppin_driver.update_fast_problem()
        (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)
        substeppin_driver.pre_loop()
        if substeppin_driver.do_predictor:
            substeppin_driver.predictor_step(writepos=substeppin_driver.do_writepos and writepos)
        for _ in range(params["max_staggered_iters"]):
            substeppin_driver.pre_iterate()
            substeppin_driver.micro_steps()
            substeppin_driver.monolithic_step()
            substeppin_driver.post_iterate()
        substeppin_driver.post_loop()
        if writepos:
            for p in [ps,pf]:
                p.writepos(extra_funcs=[p.u_prev])
    return big_p

def run_reference(params, els_per_radius):
    radius = params["heat_source"]["radius"]
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    big_mesh = get_mesh(params, els_per_radius, radius, 2)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["micro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name="big")

    initial_condition_fun = get_initial_condition(params)
    big_p.set_initial_condition(  initial_condition_fun )

    big_p.set_forms_domain()
    big_p.set_forms_boundary()
    big_p.compile_create_forms()
    itime_step = 0
    while ((itime_step < params["max_timesteps"]) and not(big_p.is_path_over())):
        itime_step += 1
        big_p.pre_iterate()
        big_p.pre_assemble()

        big_p.non_linear_solve()

        big_p.post_iterate()
        big_p.writepos()

def test_staggered_robin_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_staggered(params, "robin", 2, writepos=False)
    points = np.array([
        [-0.250, -0.250, 0.0],
        [-0.250, -0.375, 0.0],
        [+0.250, +0.000, 0.0],
        [+0.375, -0.125, 0.0],
        ])
    vals = np.array([
        234.5589654,
        805.9203145,
        24.99666684,
        1571.867472,
        ])
    assert_pointwise_vals(p,points,vals)

def test_hodge_semi_monolothic_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    p = run_semi_monolithic(params, 2, writepos=False)
    points = np.array([
        [-0.250, -0.250, 0.0],
        [-0.250, -0.375, 0.0],
        [+0.250, +0.000, 0.0],
        [+0.375, -0.125, 0.0],
        ])
    vals = np.array([
        235.5031273,
        805.9198418,
        25.23722158,
        1572.531289,
        ])
    assert_pointwise_vals(p,points,vals)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ss','--run-sub-sta',action='store_true')
    parser.add_argument('-sms','--run-sub-mon',action='store_true')
    parser.add_argument('-r','--run-ref',action='store_true')
    parser.add_argument('-t','--run-test',action='store_true')
    args = parser.parse_args()
    lp = LineProfiler()
    lp.add_module(Problem)
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    els_per_radius = params["els_per_radius"]
    profiling_file = None
    if args.run_sub_sta:
        from mhs_fenicsx.submesh import build_subentity_to_parent_mapping, find_submesh_interface, \
        compute_dg0_interpolation_data
        import mhs_fenicsx.geometry
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(StaggeredRRDriver)
        lp.add_function(mhs_fenicsx.geometry.mesh_collision)
        lp.add_function(build_subentity_to_parent_mapping)
        lp.add_function(compute_dg0_interpolation_data)
        lp.add_function(find_submesh_interface)
        driver_type = params["driver_type"]
        lp_wrapper = lp(run_staggered)
        lp_wrapper(params, driver_type, els_per_radius)
        profiling_file = f"profiling_ss_{rank}.txt"
    if args.run_sub_mon:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicSubstepper)
        lp_wrapper = lp(run_semi_monolithic)
        lp_wrapper(params, els_per_radius)
        profiling_file = f"profiling_sms_{rank}.txt"
    if args.run_ref:
        lp_wrapper = lp(run_reference)
        lp_wrapper(params, els_per_radius)
        profiling_file = f"profiling_ref_{rank}.txt"
    if args.run_test:
        test_staggered_robin_substepper()
        test_hodge_semi_monolothic_substepper()
    if profiling_file is not None:
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
