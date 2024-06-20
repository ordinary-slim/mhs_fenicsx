from line_profiler import LineProfiler
from mpi4py import MPI
from dolfinx import mesh, fem, cpp
from mhs_fenicsx.problem import Problem
import mhs_fenicsx.geometry
import yaml
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)
radius = params["heat_source"]["radius"]
speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))

def get_dt(adim_dt):
    return adim_dt * (radius / speed)

def write_gcode():
    L = params["L"]
    half_len = L / 2.0
    gcode_lines = []
    gcode_lines.append(f"G0 F{speed} X{0.5-half_len} Y1 Z0\n")
    gcode_lines.append(f"G1 X{0.5+half_len} E0.1\n")
    with open(params["path"],'w') as f:
        f.writelines(gcode_lines)

def main():
    write_gcode()
    els_per_radius = params["els_per_radius"]
    points_side = np.round(1.0 / radius * els_per_radius).astype(int) + 1
    big_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.quadrilateral)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"])
    big_p = Problem(big_mesh, macro_params, name="big")

    t0_macro_step = big_p.time
    # MACRO-STEP
    big_p.pre_iterate()
    big_p.set_forms_domain()
    big_p.compile_forms()
    big_p.pre_assemble()
    big_p.assemble()
    big_p.solve()
    big_p.post_iterate()

    t1_macro_step = big_p.time
    # MICRO-STEP
    # Determine geometry of subproblem
    # To do it properly, get initial time of macro step, final time of macro step
    # Do collision tests across track and accumulate elements to extract
    # Here we just do a single collision test
    initial_track_macro_step = big_p.source.path.get_track(t0_macro_step)
    assert(initial_track_macro_step==big_p.source.path.current_track)
    pad = 3*radius
    p0 = initial_track_macro_step.get_position(t0_macro_step)
    p1 = initial_track_macro_step.get_position(t1_macro_step)
    direction = initial_track_macro_step.get_direction()
    p0 -= direction*pad
    p1 += direction*pad
    obb = mhs_fenicsx.geometry.OBB(p0,p1,width=pad,height=pad,depth=pad,dim=big_p.dim,
                                   shrink=False)
    obb_mesh = obb.get_dolfinx_mesh()
    subproblem_els = mhs_fenicsx.geometry.mesh_collision(big_p.domain,obb_mesh,bb_tree_mesh_big=big_p.bb_tree)
    # Extract subproblem
    submesh, _, _, _ = mesh.create_submesh(big_mesh,big_p.dim,subproblem_els)
    micro_params = params.copy()
    micro_params["dt"] = get_dt(params["micro_adim_dt"])
    small_p = Problem(submesh,micro_params, name="small")

    # TODO 1: Solve on micro problem
    big_p.writepos()
    small_p.writepos()


if __name__=="__main__":
    main()
