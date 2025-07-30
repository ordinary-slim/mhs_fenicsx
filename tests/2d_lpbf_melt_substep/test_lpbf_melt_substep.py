import argparse
from dolfinx import fem, mesh
import numpy as np
import yaml
from mpi4py import MPI
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.gcode import TrackType
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicSubstepper, MHSSemiMonolithicChimeraSubstepper
from mhs_fenicsx.drivers import MonolithicRRDriver, DomainDecompositionDriver, StaggeredInterpRRDriver
from mhs_fenicsx.chimera import build_moving_problem
from mhs_fenicsx.problem.helpers import assert_pointwise_vals, print_vals

def write_gcode(params):
    dim = params["dim"]
    if dim != 2:
        raise KeyError('not ready for dim not 2')
    (L, H) = (params["domain_width"], params["domain_height"])
    t = params["printer"]["layer_thickness"]
    num_layers = np.rint(H / t).astype(np.int32)
    num_layers = 2
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    hlenx = + L / 2.0
    gcode_lines = []
    gcode_lines.append(f"G0 X{-hlenx} Y0.0 F{speed}")
    E = 0.0
    for layer in range(num_layers):
        gcode_lines.append(f"G0 Y{t*(layer+1):2.2f}")
        gcode_lines.append(f"G4 P0.5")
        gcode_lines.append(f"G4 P0.5 R1")
        E += 0.1
        gcode_lines.append(f"G1 X{np.power(-1, layer)*hlenx} E{E:2.2f}")
    with open(params["source_terms"][0]["path"],'w') as f:
        f.writelines("\n".join(gcode_lines))

def get_mesh(params, els_per_radius, radius, dim):
    el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
    (Lx, Ly) = (params["domain_width"], params["domain_height"])
    Lz = params["domain_depth"] if "domain_depth" in params else 1.0
    box = [-Lx/2.0, -Ly/2.0, 0.0, +Lx/2.0, +Ly/2.0, Lz]
    nx = np.round((box[3]-box[0]) * el_density).astype(np.int32)
    ny = np.round((box[4]-box[1]) * el_density).astype(np.int32)
    nz = np.round((box[5]-box[2]) * el_density).astype(np.int32)
    if dim==2:
        box[1], box[4] = 0.0, +Ly
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
    domain = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    print_dt = get_dt(params["substepping_parameters"]["micro_adim_dt"], radius, speed)
    macro_params["dt"] = print_dt
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False, name="2dlpbf")
    ps.set_activation(np.array([] ,dtype=np.int32))

    ps.set_initial_condition(  params["environment_temperature"] )

    ps.set_forms()
    ps.compile_forms()
    adim_dt_cooling = params["substepping_parameters"]["cooling_adim_dt"]
    itime_step = 0
    while not(ps.is_path_over()):
        itime_step += 1
        track = ps.source.path.get_track(ps.time)
        if ps.source.path.get_track(ps.time).type in [TrackType.RECOATING,
                                                      TrackType.DWELLING]:
            ps.set_dt(adim_dt_cooling*(track.t1 - track.t0))
        else:
            ps.set_dt(print_dt)
        ps.pre_iterate()
        ps.instantiate_forms()
        ps.pre_assemble()
        ps.non_linear_solve()
        ps.post_iterate()
        if writepos:
            ps.writepos(extra_funcs=[ps.u_av])
    return ps

def run_staggered(params, writepos=True):
    radius = params["source_terms"][0]["radius"]
    domain = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False, name="2dlpbf")
    ps.set_activation(np.array([] ,dtype=np.int32))
    ps.set_initial_condition(  params["environment_temperature"] )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSStaggeredSubstepper(StaggeredInterpRRDriver,
                                               ps,
                                               staggered_relaxation_factors=[1.0, 1.0],)
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
            ps.writepos(extra_funcs=[ps.u_prev])
    return ps

def run_hodge(params, writepos=True):
    radius = params["source_terms"][0]["radius"]
    domain = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False, name="2dlpbf")
    ps.set_activation(np.array([] ,dtype=np.int32))
    ps.set_initial_condition(  params["environment_temperature"] )

    max_timesteps = params["max_timesteps"]
    substeppin_driver = MHSSemiMonolithicSubstepper(ps,)
    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            for p in [ps,pf]:
                p.writepos(extra_funcs=[p.u_prev])
    return ps

def run_chimera_staggered(params, writepos=True):
    radius = params["source_terms"][0]["radius"]
    domain = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False, name="2dlpbf")
    pm = build_moving_problem(ps, params["els_per_radius"])
    ps.set_activation(np.array([] ,dtype=np.int32))
    for p in [pm, ps]:
        p.set_initial_condition(  params["environment_temperature"] )

    max_timesteps = params["max_timesteps"]

    class MyMHSStaggeredChimeraSubstepper(MHSStaggeredChimeraSubstepper):
        def is_steady_enough(self):
            next_track = self.pf.source.path.get_track(pf.time)
            return (((pf.time - next_track.t0) / (next_track.t1 - next_track.t0)) >= 0.15)

    substeppin_driver = MyMHSStaggeredChimeraSubstepper(StaggeredInterpRRDriver,
                                                        ps, pm,
                                                        staggered_relaxation_factors=[1.0, 1.0])

    (ps, pf) = (substeppin_driver.ps, substeppin_driver.pf)
    staggered_driver = substeppin_driver.staggered_driver

    el_density = np.rint((1.0 / radius) * params["els_per_radius"]).astype(np.int32)
    h = 1.0 / el_density
    k = float(params["material_metal"]["conductivity"])
    staggered_driver.set_dirichlet_coefficients(h, k)

    itime_step = 0
    while ((itime_step < max_timesteps) and not(ps.is_path_over())):
        itime_step += 1
        substeppin_driver.do_timestep()
        if writepos:
            ps.writepos(extra_funcs=[ps.u_prev, ps.u_av])
    return ps

def run_chimera_hodge(params, writepos=True):
    radius = params["source_terms"][0]["radius"]
    domain = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    ps = Problem(domain, macro_params, finalize_activation=False, name="2dlpbf")
    pm = build_moving_problem(ps, params["els_per_radius"])
    ps.set_activation(np.array([] ,dtype=np.int32))
    for p in [pm, ps]:
        p.set_initial_condition(  params["environment_temperature"] )

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

def test_2dlpbf_ref():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_reference(params, writepos=False)
    points = np.array([
        [-0.4625, +0.0875, 0.0],
        [-0.4375, +0.0875, 0.0],
        [-0.0500, +0.0500, 0.0],
        [+0.4625, +0.0375, 0.0],
        [-0.5000,  0.0000, 0.0]
        ])
    vals = np.array([1686.47343084, 1706.81739879, 1126.69490132,  398.62514368,
                     333.59490359])
    mats = np.array([2., 2., 2., 2., 1.])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)

def test_staggered_rr_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_staggered(params, writepos=False)
    points = np.array([
        [-0.4625, +0.0875, 0.0],
        [-0.4375, +0.0875, 0.0],
        [-0.0500, +0.0500, 0.0],
        [+0.4625, +0.0375, 0.0],
        [-0.5000,  0.0000, 0.0]
        ])
    vals = np.array([1686.47343084, 1706.81739878, 1126.69490977, 411.3817135,
                     333.59490358])
    mats = np.array([2., 2., 2., 2., 1.])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)

def test_hodge_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_hodge(params, writepos=False)
    points = np.array([
        [-0.4625, +0.0875, 0.0],
        [-0.4375, +0.0875, 0.0],
        [-0.0500, +0.0500, 0.0],
        [+0.4625, +0.0375, 0.0],
        [-0.5000,0.0000, 0.0]
        ])
    vals = np.array([1686.47343053, 1706.8173969 , 1126.74657231,  428.43461819,
                     333.59490353])

    mats = np.array([1., 2., 2., 1., 1.])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)

def test_staggered_chimera_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_chimera_staggered(params, writepos=False)
    points = np.array([
        [-0.4625, +0.0875, 0.0],
        [-0.4375, +0.0875, 0.0],
        [-0.0500, +0.0500, 0.0],
        [+0.4625, +0.0375, 0.0],
        [-0.5000,0.0000, 0.0]
        ])
    vals = np.array([1577.75299862, 1651.87475776, 1175.1756622 ,  401.54335612,
                     216.5882932])

    mats = np.array([1., 2., 2., 1., 1.])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)

def test_hodge_chimera_substepper():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    p = run_chimera_hodge(params, writepos=False)
    points = np.array([
        [-0.4625, +0.0875, 0.0],
        [-0.4375, +0.0875, 0.0],
        [-0.0500, +0.0500, 0.0],
        [+0.4625, +0.0375, 0.0],
        [-0.5000,0.0000, 0.0]
        ])
    vals = np.array([1577.75299873, 1651.87475785, 1175.17568643,  401.57314081,
                     216.5882932])

    mats = np.array([1., 2., 2., 1., 1.])
    assert_pointwise_vals(p, points, vals, f=p.u)
    assert_pointwise_vals(p, points, mats, f=p.material_id)

if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-ref', action='store_true')
    parser.add_argument('-tr', '--run-test-ref', action='store_true')
    parser.add_argument('-ss','--run-sub-sta',action='store_true')
    parser.add_argument('-sms','--run-hodge',action='store_true')
    parser.add_argument('-tsms','--run-test-hodge',action='store_true')
    parser.add_argument('-css','--run-chimera-sub-sta',action='store_true')
    parser.add_argument('-tcss', '--test-chimera-sub-sta', action='store_true')
    parser.add_argument('-csms','--run-chimera-hodge',action='store_true')
    parser.add_argument('-tcsms', '--test-chimera-hodge', action='store_true')
    args = parser.parse_args()
    if args.run_ref:
        run_reference(params)
    if args.run_test_ref:
        test_2dlpbf_ref()
    if args.run_sub_sta:
        run_staggered(params)
    if args.run_hodge:
        run_hodge(params)
    if args.run_hodge:
        run_hodge(params)
    if args.run_test_hodge:
        test_hodge_substepper()
    if args.run_chimera_sub_sta:
        run_chimera_staggered(params)
    if args.run_chimera_hodge:
        run_chimera_hodge(params)
    if args.test_chimera_sub_sta:
        test_staggered_chimera_substepper()
    if args.test_chimera_hodge:
        test_hodge_chimera_substepper()
