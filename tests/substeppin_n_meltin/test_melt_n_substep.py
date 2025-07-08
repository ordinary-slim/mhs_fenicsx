from mhs_fenicsx.problem import Problem
from dolfinx import mesh
from mpi4py import MPI
import numpy as np
import yaml
import argparse
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicSubstepper, MHSSemiMonolithicChimeraSubstepper
from mhs_fenicsx.drivers import MonolithicRRDriver, MonolithicDomainDecompositionDriver, StaggeredRRDriver
from mhs_fenicsx.chimera import build_moving_problem
from mhs_fenicsx.problem.helpers import assert_pointwise_vals, print_vals

def write_gcode(params):
    (Lx, Ly) = (params["domain_width"], params["domain_height"])
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    hlenx = + 3.0 * Lx / 8.0
    hleny = + 3.0 * Ly / 8.0
    gcode_lines = []
    gcode_lines.append(f"G0 F{speed} X-{hlenx} Y-{hleny} Z0\n")
    gcode_lines.append(f"G1 X+{hlenx} E0.1\n")
    with open(params["source_terms"][0]["path"],'w') as f:
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
               mesh.CellType.quadrilateral,
               )
    else:
        return mesh.create_box(MPI.COMM_WORLD,
               [box[:3], box[3:]],
               [nx, ny, nz],
               mesh.CellType.hexahedron,
               )


def get_dt(adim_dt, radius, speed):
    return adim_dt * (radius / speed)

def run_reference(params, writepos=True):
    radius = params["source_terms"][0]["radius"]
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    big_mesh = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["micro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name="big_melting")

    big_p.set_initial_condition(  params["environment_temperature"] )

    big_p.set_forms()
    big_p.compile_forms()# UNTESTED
    itime_step = 0
    while not(big_p.is_path_over()):
        itime_step += 1
        big_p.pre_iterate()
        big_p.instantiate_forms()# UNTESTED
        big_p.pre_assemble()
        big_p.non_linear_solve()
        big_p.post_iterate()
        if writepos:
            big_p.writepos(extra_funcs=[big_p.u_av])

def run_staggered(params, writepos=True):
    radius = params["source_terms"][0]["radius"]
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    big_mesh = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=f"big_ss_RR_melting")
    big_p.set_initial_condition(  params["environment_temperature"] )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSStaggeredSubstepper(StaggeredRRDriver,
                                               [1.0, 1.0],
                                               big_p,)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)
    staggered_driver = substeppin_driver.staggered_driver
    el_density = np.round((1.0 / radius) * params["els_per_radius"]).astype(np.int32)
    h = 1.0 / el_density
    k = float(params["material_metal"]["conductivity"])
    staggered_driver.set_dirichlet_coefficients(h, k)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos()
    return big_p

def run_semi_monolithic(params, writepos=True):
    radius = params["source_terms"][0]["radius"]
    big_mesh = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=f"big_sms_melting")
    big_p.set_initial_condition(params["environment_temperature"])

    max_timesteps = params["max_timesteps"]
    substeppin_driver = MHSSemiMonolithicSubstepper(big_p,)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(big_p.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps,pf]:
                p.writepos(extra_funcs=[p.u_prev])
    return big_p

def run_chimera_staggered(params, writepos=True):
    els_per_radius = params["els_per_radius"]
    radius = params["source_terms"][0]["radius"]
    el_size = radius / els_per_radius
    initial_relaxation_factors = [1.0,1.0]
    big_mesh = get_mesh(params, els_per_radius, radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, name=f"big_chimera_ss_RR_melting")
    if els_per_radius == 2:
        shift = np.array([0.0, el_size / 2.0, 0.0], dtype=np.float64)
    else:
        shift = None
    pm = build_moving_problem(ps, els_per_radius, shift=shift)
    ps.set_initial_condition(params["environment_temperature"])
    pm.set_initial_condition(params["environment_temperature"])

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSStaggeredChimeraSubstepper(StaggeredRRDriver, initial_relaxation_factors, ps, pm,)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)
    staggered_driver = substeppin_driver.staggered_driver

    k = float(params["material_metal"]["conductivity"])
    staggered_driver.set_dirichlet_coefficients(el_size, k)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos()
    return ps

def run_chimera_hodge(params, writepos=True):
    els_per_radius = params["els_per_radius"]
    radius = params["source_terms"][0]["radius"]
    el_size = radius / els_per_radius
    big_mesh = get_mesh(params, els_per_radius, radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(big_mesh, macro_params, name=f"big_chimera_sms_melting")
    if els_per_radius == 2:
        shift = np.array([0.0, el_size / 2.0, 0.0], dtype=np.float64)
    else:
        shift = None
    pm = build_moving_problem(ps, els_per_radius, shift=shift)
    ps.set_initial_condition(  params["environment_temperature"] )
    pm.set_initial_condition(  params["environment_temperature"] )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSSemiMonolithicChimeraSubstepper(ps, pm,)
    pf = substeppin_driver.pf
    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps,pf]:
                p.writepos(extra_funcs=[p.u_prev])
    return ps

def test_staggered_rr():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_staggered(params, writepos=False)
    points = np.array([
        [+0.06125, -0.1875, 0.0],
        [-0.1000, -0.0875, 0.0],
        [-0.4125, -0.1500, 0.0],
        [-0.1500, -0.0125, 0.0],
        ])
    vals = np.array([1741.17490832,  220.79734639,  282.20719521,   25.44636481])
    mats = np.array([ 2, 1, 1, 1, ])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)

def test_hodge():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_semi_monolithic(params, writepos=False)
    points = np.array([
        [+0.06125, -0.1875, 0.0],
        [-0.1000, -0.0875, 0.0],
        [-0.4125, -0.1500, 0.0],
        [-0.1500, -0.0125, 0.0],
        ])
    vals = np.array([1741.17490633,  220.80107027,  275.14547577,   25.14675407])
    mats = np.array([ 2, 1, 1, 1, ])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)

def test_chimera_staggered_rr():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_chimera_staggered(params, writepos=False)
    points = np.array([
        [+0.06125, -0.1875, 0.0],
        [-0.1000, -0.0875, 0.0],
        [-0.4125, -0.1500, 0.0],
        [-0.1500, -0.0125, 0.0],
        ])
    vals = np.array([1753.49372323,  219.63642619,  286.68875939,   25.15615151])
    mats = np.array([ 2, 1, 1, 1, ])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)

def test_chimera_hodge():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_chimera_hodge(params, writepos=False)
    points = np.array([
        [+0.06125, -0.1875, 0.0],
        [-0.1000, -0.0875, 0.0],
        [-0.4125, -0.1500, 0.0],
        [-0.1500, -0.0125, 0.0],
        ])
    vals = np.array([1753.49372288,  219.64521511,  282.18803562,   25.10785478])
    mats = np.array([ 2, 1, 1, 1, ])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--run-ref',action='store_true')
    parser.add_argument('-ss','--run-sub-sta',action='store_true')
    parser.add_argument('-sms','--run-hodge',action='store_true')
    parser.add_argument('-css','--run-chimera-sub-sta',action='store_true')
    parser.add_argument('-csms','--run-chimera-hodge',action='store_true')
    args = parser.parse_args()
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    if args.run_ref:
        run_reference(params)
    if args.run_sub_sta:
        run_staggered(params)
    if args.run_hodge:
        run_semi_monolithic(params)
    if args.run_chimera_sub_sta:
        run_chimera_staggered(params)
    if args.run_chimera_hodge:
        run_chimera_hodge(params)
