from line_profiler import LineProfiler
from mpi4py import MPI
from dolfinx import mesh, io
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper
import yaml
import numpy as np
import argparse
from mhs_fenicsx.problem.helpers import assert_pointwise_vals, print_vals
from mhs_fenicsx.drivers import StaggeredInterpRRDriver, StaggeredInterpDNDriver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_initial_condition(params):
    T_env = np.float64(params["environment_temperature"])
    if params["initial_condition"]:
        initial_condition_fun = lambda x : abs(300*np.cos(4*(x[0]-0.5)*(x[1]-0.5)))
    else:
        initial_condition_fun = lambda x : T_env*np.ones_like(x[0])
    return initial_condition_fun

def get_max_timestep(params):
    return int(params["max_timesteps"]) if params["max_timesteps"] > 0.0 else np.iinfo(int).max

def get_dt(adim_dt, radius, speed):
    return adim_dt * (radius / speed)

def write_gcode(params):
    (Lx, Ly) = (params["domain_width"], params["domain_height"])
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    width_path_fraction = params.get("width_path_fraction", 3.0/4.0)
    height_path_fraction = params.get("height_path_fraction", 3.0/4.0)
    hlenx = np.round(Lx * width_path_fraction / 2.0, 4)
    hleny = np.round(Ly * height_path_fraction / 2.0, 4)
    gcode_lines = []
    is_straight_line = params.get("straight_line", False)
    if not is_straight_line:
        gcode_lines.append(f"G0 F{speed} X-{hlenx} Y-{hleny} Z0\n")
        gcode_lines.append(f"G1 X+{hlenx} E0.1\n")
        gcode_lines.append(f"G1 Y+{hleny} E0.2\n")
        gcode_lines.append(f"G1 X-{hlenx} E0.3\n")
        gcode_lines.append(f"G1 Y-{hleny} E0.4\n")
    else:
        gcode_lines.append(f"G0 F{speed} X-{hlenx} Y0.0 Z0\n")
        gcode_lines.append(f"G1 X+{hlenx} E0.1\n")
    with open(params["source_terms"][0]["path"],'w') as f:
        f.writelines(gcode_lines)

def get_mesh(params, radius, dim):
    els_per_radius = params["els_per_radius"]
    if (dim == 2) and (params["el_type"] == "quadrilateral"):
        el_type = mesh.CellType.quadrilateral
    elif (dim == 3) and (params["el_type"] == "hexahedron"):
        el_type = mesh.CellType.hexahedron
    else:
        el_type = mesh.CellType.triangle if dim == 2 else mesh.CellType.tetrahedron

    el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
    (Lx, Ly) = (params["domain_width"], params["domain_height"])
    Lz = params["domain_depth"] if "domain_depth" in params else 1.0
    box = [-Lx/2.0, -Ly/2.0, 0.0, +Lx/2.0, +Ly/2.0, Lz]
    nx = np.round((box[3]-box[0]) * el_density).astype(np.int32)
    ny = np.round((box[4]-box[1]) * el_density).astype(np.int32)
    nz = np.round((box[5]-box[2]) * el_density).astype(np.int32)
    if dim == 2:
        return mesh.create_rectangle(MPI.COMM_WORLD,
                                     [box[:2], box[3:5]],
                                     [nx, ny],
                                     el_type,)
    else:
        return mesh.create_box(MPI.COMM_WORLD,
                               [box[:3], box[3:]],
                               [nx, ny, nz],
                               el_type)


def run_staggered(params, driver_type, descriptor="", writepos=True):
    write_gcode(params)
    radius = params["source_terms"][0]["radius"]
    if   driver_type=="robin":
        driver_constructor = StaggeredInterpRRDriver
        initial_relaxation_factors = [1.0, 1.0]
    elif driver_type=="dn":
        driver_constructor = StaggeredInterpDNDriver
        initial_relaxation_factors = [1.0, 1.0]
    else:
        raise ValueError("Undefined staggered driver type.")
    big_mesh = get_mesh(params, radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    descriptor = f"ss{driver_type}" + descriptor
    big_p = Problem(big_mesh, macro_params, name=descriptor)
    initial_condition_fun = get_initial_condition(params)
    big_p.set_initial_condition(  initial_condition_fun )

    max_timesteps = get_max_timestep(params)

    substeppin_driver = MHSStaggeredSubstepper(driver_constructor,
                                               big_p,
                                               staggered_relaxation_factors=initial_relaxation_factors,)
    staggered_driver = substeppin_driver.staggered_driver
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)

    if (type(staggered_driver)==StaggeredInterpRRDriver):
        el_density = np.round((1.0 / radius) * params["els_per_radius"]).astype(np.int32)
        h = 1.0 / el_density
        k = float(params["material_metal"]["conductivity"])
        staggered_driver.set_dirichlet_coefficients(h, k)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos(extension="vtx")
    return big_p

def run_semi_monolithic(params, descriptor="", writepos=True):
    write_gcode(params)
    theoretical_predictor = params["substepping_parameters"]["predictor"].get("theoretical", False)
    radius = params["source_terms"][0]["radius"]
    big_mesh = get_mesh(params, radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]

    descriptor = f"sms" + descriptor
    big_p = Problem(big_mesh, macro_params, name=descriptor)
    initial_condition_fun = get_initial_condition(params)
    big_p.set_initial_condition(  initial_condition_fun )

    max_timesteps = get_max_timestep(params)
    substeppin_driver = MHSSemiMonolithicSubstepper(big_p)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)

    if theoretical_predictor:
        ps2 = ps.copy(name=f"{descriptor}_theoretical_predictor")
        micro_dt = ps.dimensionalize_mhs_timestep(ps.source.path.tracks[0], params["substepping_parameters"]["micro_adim_dt"])
        macro_dt = ps.dimensionalize_mhs_timestep(ps.source.path.tracks[0], params["substepping_parameters"]["macro_adim_dt"])
        num_substeps = np.rint(macro_dt / micro_dt).astype(int)
        ps2.set_dt(micro_dt)
        ps2.set_forms()
        ps2.compile_forms()
        ps2.reset_activation(finalize=True)
        ps.input_parameters["substepping_parameters"]["predictor"]["custom"] = ps2.u

    itime_step = 0
    while ((itime_step < max_timesteps) and not(big_p.is_path_over())):
        if theoretical_predictor:
            # This while loop is not super robust, but should work for simple
            # case without cooling
            dt_s = min(macro_dt, ps.source.path.times[-1] - ps.time)
            writer = io.VTXWriter(ps2.domain.comm,
                                  f"post_substep#{itime_step}_{ps2.name}.bp",
                                  output=[ps2.u, ps2.u_prev])
            isubstep_pred = 0
            while (np.round(ps.time + dt_s - ps2.time, 6) > 0):
                isubstep_pred += 1
                ps2.pre_iterate()
                ps2.instantiate_forms()
                ps2.pre_assemble()
                ps2.non_linear_solve()
                ps2.post_iterate()
                if isubstep_pred == (num_substeps - 1):
                    ps.input_parameters["substepping_parameters"]["predictor"]["force_last_substep"] = ps2.u.copy()
                t = np.round((ps2.time - ps.time) / dt_s, 6)
                writer.write(np.round(t, 7))
            writer.close()

        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps,pf]:
                p.writepos(extension="vtx")
        if theoretical_predictor:
            ps2.u.x.array[:] = ps.u.x.array[:]

    return big_p

def run_reference(params, descriptor="", writepos=True):
    write_gcode(params)
    radius = params["source_terms"][0]["radius"]
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    domain = get_mesh(params, radius, 2)

    macro_params = params.copy()
    print_dt = get_dt(params["substepping_parameters"]["micro_adim_dt"], radius, speed)
    macro_params["dt"] = print_dt
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    descriptor = "ref" + descriptor
    ps = Problem(domain, macro_params, finalize_activation=True, name=descriptor)

    initial_condition_fun = get_initial_condition(params)
    ps.set_initial_condition(initial_condition_fun)

    ps.set_forms()
    ps.compile_forms()
    itime_step = 0
    ps.set_dt(print_dt)
    max_timesteps = get_max_timestep(params) * (params["substepping_parameters"]["macro_adim_dt"] / params["substepping_parameters"]["micro_adim_dt"])
    macro_dt = ps.dimensionalize_mhs_timestep(ps.source.path.tracks[0], params["substepping_parameters"]["macro_adim_dt"])
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        ps.pre_iterate()
        ps.instantiate_forms()
        ps.pre_assemble()
        ps.non_linear_solve()
        ps.post_iterate()
        #if writepos and (abs(ps.time / macro_dt - np.rint(ps.time / macro_dt)) < 1e-7):
        #    ps.writepos(extension="vtx")
    return ps

def test_staggered_robin_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_staggered(params, "robin", writepos=False)
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
    p = run_semi_monolithic(params, writepos=False)
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
    parser.add_argument('-d', '--descriptor', default="")
    parser.add_argument('-i','--input-file', default='input.yaml')

    args = parser.parse_args()
    input_file = args.input_file
    with open(input_file, 'r') as f:
        params = yaml.safe_load(f)

    lp = LineProfiler()
    lp.add_module(Problem)

    profiling_file = None
    if args.run_sub_sta:
        from mhs_fenicsx.submesh import build_subentity_to_parent_mapping, find_submesh_interface, \
        compute_dg0_interpolation_data
        import mhs_fenicsx.geometry
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSStaggeredSubstepper)
        lp.add_module(StaggeredInterpRRDriver)
        lp.add_function(mhs_fenicsx.geometry.mesh_collision)
        lp.add_function(build_subentity_to_parent_mapping)
        lp.add_function(compute_dg0_interpolation_data)
        lp.add_function(find_submesh_interface)
        driver_type = params["driver_type"]
        lp_wrapper = lp(run_staggered)
        lp_wrapper(params, driver_type, descriptor=args.descriptor)
        profiling_file = f"profiling_ss_{rank}.txt"
    if args.run_sub_mon:
        lp.add_module(MHSSubstepper)
        lp.add_module(MHSSemiMonolithicSubstepper)
        lp_wrapper = lp(run_semi_monolithic)
        lp_wrapper(params, descriptor=args.descriptor)
        profiling_file = f"profiling_sms_{rank}.txt"
    if args.run_ref:
        lp_wrapper = lp(run_reference)
        lp_wrapper(params)
        profiling_file = f"profiling_ref_{rank}.txt"
    if args.run_test:
        test_staggered_robin_substepper()
        test_hodge_semi_monolothic_substepper()
    if profiling_file is not None:
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
