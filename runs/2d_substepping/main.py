from line_profiler import LineProfiler
from mpi4py import MPI
from dolfinx import mesh
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.drivers import MHSSubsteppingDriver
import yaml
import numpy as np
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)
radius = params["heat_source"]["radius"]
speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))

case_name = "big"

def get_dt(adim_dt):
    return adim_dt * (radius / speed)

def write_gcode():
    (Lx, Ly) = (params["domain_width"], params["domain_height"])
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

def mesh_rectangle(box,el_density):
    nx = np.round((box[2]-box[0]) * el_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * el_density).astype(np.int32)
    return mesh.create_rectangle(MPI.COMM_WORLD,
           [box[:2], box[2:]],
           [nx, ny],
           #mesh.CellType.quadrilateral,
           )

def run_substepped(initial_condition=False):
    try:
        initial_condition = params["initial_condition"]
    except KeyError:
        initial_condition = True
    els_per_radius = params["els_per_radius"]
    driver_type = params["driver_type"]
    if   driver_type=="robin":
        driver_constructor = StaggeredRRDriver
        initial_relaxation_factors=[1.0,1.0]
    elif driver_type=="dn":
        driver_constructor = StaggeredDNDriver
        initial_relaxation_factors=[0.5,1]
    else:
        raise ValueError("Undefined staggered driver type.")
    el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
    (Lx, Ly) = (params["domain_width"], params["domain_height"])
    big_mesh  = mesh_rectangle([-Lx/2.0, -Ly/2.0, +Lx/2.0, +Ly/2.0], el_density)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"])
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=f"{case_name}_substepped")
    if initial_condition:
        initial_condition = lambda x : 100*np.cos(4*(x[0]-0.5)*(x[1]-0.5))
        big_p.set_initial_condition(  initial_condition )

    num_timesteps = params["num_timesteps"]
    for _ in range(num_timesteps):
        substeppin_driver = MHSSubsteppingDriver(big_p,writepos=False)

        substeppin_driver.extract_subproblem() # generates driver.fast_problem
        (ps,pf) = (substeppin_driver.ps,substeppin_driver.pf)
        staggered_driver = driver_constructor(pf,ps,
                                       submesh_data=substeppin_driver.submesh_data,
                                       max_staggered_iters=params["max_staggered_iters"],
                                       initial_relaxation_factors=initial_relaxation_factors,)
        if (type(staggered_driver)==StaggeredRRDriver):
            h = 1.0 / el_density
            k = float(params["material_metal"]["conductivity"])
            staggered_driver.dirichlet_coeff[staggered_driver.p1] = 1.0/4.0
            staggered_driver.dirichlet_coeff[staggered_driver.p2] =  k / (2 * h)
            staggered_driver.relaxation_coeff[staggered_driver.p1].value = 3.0 / 3.0
        # Move extra_subproblem here
        #TODO: Check on pre_iterate / post_iterate of problems
        staggered_driver.pre_loop(prepare_subproblems=False)
        substeppin_driver.pre_loop(staggered_driver)
        #substeppin_driver.predictor_step()
        substeppin_driver.subtract_child()
        staggered_driver.prepare_subproblems()
        for _ in range(staggered_driver.max_staggered_iters):
            substeppin_driver.pre_iterate()
            staggered_driver.pre_iterate()
            substeppin_driver.iterate()
            substeppin_driver.post_iterate()
            staggered_driver.post_iterate(verbose=True)
            substeppin_driver.writepos(case="macro")

            if staggered_driver.convergence_crit < staggered_driver.convergence_threshold:
                break
        substeppin_driver.post_loop()
        #TODO: Interpolate solution to inactive ps
        ps.u.interpolate(pf.u,
                         cells0=np.arange(pf.num_cells),
                         cells1=substeppin_driver.submesh_data["subcell_map"])
        big_p.writepos()

def run_reference():
    try:
        initial_condition = params["initial_condition"]
    except KeyError:
        initial_condition = True
    els_per_radius = params["els_per_radius"]
    points_side = np.round(1.0 / radius * els_per_radius).astype(int) + 1
    big_mesh  = mesh.create_unit_square(MPI.COMM_WORLD,
                                        points_side,
                                        points_side,
                                        #mesh.CellType.quadrilateral,
                                        )

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["micro_adim_dt"])
    final_t = get_dt(params["macro_adim_dt"])*params["num_timesteps"]
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=case_name)

    if initial_condition:
        initial_condition = lambda x : 100*np.cos(4*(x[0]-0.5)*(x[1]-0.5))
        big_p.set_initial_condition(  initial_condition )

    big_p.set_forms_domain()
    big_p.set_forms_boundary()
    big_p.compile_forms()
    while (final_t - big_p.time) > 1e-7:
        big_p.pre_iterate()
        big_p.pre_assemble()
        big_p.assemble()
        big_p.solve()
        big_p.post_iterate()
        big_p.writepos()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--run-sub',action='store_false')
    parser.add_argument('-r','--run-ref',action='store_true')
    args = parser.parse_args()
    lp = LineProfiler()
    lp.add_module(Problem)
    write_gcode()
    if args.run_sub:
        from mhs_fenicsx.submesh import build_subentity_to_parent_mapping, find_submesh_interface, \
        compute_dg0_interpolation_data
        from mhs_fenicsx.drivers import StaggeredRRDriver, StaggeredDNDriver
        from mhs_fenicsx.problem.helpers import indices_to_function
        import mhs_fenicsx.geometry
        lp.add_module(MHSSubsteppingDriver)
        lp.add_module(StaggeredRRDriver)
        lp.add_function(mhs_fenicsx.geometry.mesh_collision)
        lp.add_function(build_subentity_to_parent_mapping)
        lp.add_function(compute_dg0_interpolation_data)
        lp.add_function(find_submesh_interface)
        lp.add_function(indices_to_function)
        lp_wrapper = lp(run_substepped)
        lp_wrapper()
        profiling_file = f"profiling_sub_{rank}.txt"
    if args.run_ref:
        lp_wrapper = lp(run_reference)
        lp_wrapper()
        profiling_file = f"profiling_ref_{rank}.txt"
    try:
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
    except NameError:
        pass
