import argparse
from dolfinx import fem, mesh
import numpy as np
import yaml
from mpi4py import MPI
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.gcode import TrackType
from mhs_fenicsx.drivers.substeppers import MHSSubstepper, MHSStaggeredSubstepper, MHSStaggeredChimeraSubstepper, MHSSemiMonolithicSubstepper, MHSSemiMonolithicChimeraSubstepper
from mhs_fenicsx.drivers import MonolithicRRDriver, MonolithicDomainDecompositionDriver, StaggeredRRDriver
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
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    hlenx = + L / 2.0
    gcode_lines = []
    gcode_lines.append(f"G0 X{-hlenx} Y0.0 F{speed}")
    E = 0.0
    for layer in range(num_layers):
        gcode_lines.append(f"G0 Y{t*(layer+1):2.2f}")
        gcode_lines.append(f"G4 P1.0 R1")
        E += 0.1
        gcode_lines.append(f"G1 X{np.pow(-1, layer)*hlenx} E{E:2.2f}")
    with open(params["path"],'w') as f:
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
    radius = params["heat_source"]["radius"]
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    big_mesh = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    print_dt = get_dt(params["micro_adim_dt"], radius, speed)
    macro_params["dt"] = print_dt
    recoat_dt = 0.5
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name="2dlpbf")
    big_p.set_activation(np.array([] ,dtype=np.int32))

    big_p.set_initial_condition(  params["environment_temperature"] )

    big_p.set_forms()
    big_p.compile_forms()
    itime_step = 0
    while not(big_p.is_path_over()):
        itime_step += 1
        if big_p.source.path.get_track(big_p.time).type == TrackType.RECOATING:
            big_p.set_dt(recoat_dt)
        else:
            big_p.set_dt(print_dt)
        big_p.pre_iterate()
        big_p.instantiate_forms()
        big_p.pre_assemble()
        big_p.non_linear_solve()
        big_p.post_iterate()
        if writepos:
            big_p.writepos(extra_funcs=[big_p.u_av])

def run_staggered(params, writepos=True):
    radius = params["heat_source"]["radius"]
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    big_mesh = get_mesh(params, params["els_per_radius"], radius, 2)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=f"big_ss_RR_melting")
    big_p.set_activation(np.array([] ,dtype=np.int32))
    big_p.set_initial_condition(  params["environment_temperature"] )

    max_timesteps = params["max_timesteps"]

    substeppin_driver = MHSStaggeredSubstepper(big_p,writepos=(params["substepper_writepos"] and writepos), do_predictor=params["predictor_step"])
    (ps,pf) = (substeppin_driver.ps,substeppin_driver.pf)
    staggered_driver = StaggeredRRDriver(pf,ps,
                                   max_staggered_iters=params["max_staggered_iters"],
                                   initial_relaxation_factors=[1.0, 1.0],)

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

if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-ref', action='store_true')
    parser.add_argument('-ss','--run-sub-sta',action='store_true')
    parser.add_argument('-sms','--run-hodge',action='store_true')
    parser.add_argument('-css','--run-chimera-sub-sta',action='store_true')
    parser.add_argument('-csms','--run-chimera-hodge',action='store_true')
    args = parser.parse_args()
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
