import numpy as np
import yaml
from dolfinx import mesh
from mpi4py import MPI
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.gcode import TrackType

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_mesh(params):
    el_size = params["el_size"]
    edge_len = params["edge_length"]
    num_layers = params["num_layers"]
    Lx, Ly = edge_len, edge_len
    part_height = el_size * num_layers 
    Lz = edge_len + part_height
    box = [-Lx/2.0, -Ly/2.0, -edge_len, +Lx/2.0, +Ly/2.0, part_height]
    num_els = [np.round(L / el_size).astype(int) for L in [Lx, Ly, Lz]]
    return mesh.create_box(comm, [box[:3], box[3:]], num_els, cell_type=mesh.CellType.hexahedron)

def write_gcode(params):
    el_size = params["el_size"]
    num_layers = params["num_layers"]
    edge_len = params["edge_length"]
    half_len = edge_len / 2.0
    num_hatches = np.rint(edge_len / el_size).astype(int)
    speed = np.linalg.norm(np.array(params["source_terms"][0]["initial_speed"]))
    gcode_lines = []
    gcode_lines.append(f"G0 F{speed:g}")
    p0, p1 = np.zeros(3), np.zeros(3)
    E = 0.0
    for ilayer in range(num_layers):
        if ilayer % 2 == 0:
            const_idx = 0
            mov_idx = 1
        else:
            const_idx = 1
            mov_idx = 0
        z = el_size * (ilayer + 1)
        p0[2], p1[2] = z, z
        for ihatch in range(num_hatches):
            E += 0.1
            fixed_coord = -half_len + (ihatch + 0.5) * el_size
            sign = (ihatch + 1) % 2
            mov_coord0 = (-1)**sign * half_len
            mov_coord1 = -mov_coord0
            p0[const_idx], p1[const_idx] = fixed_coord, fixed_coord
            p0[mov_idx], p1[mov_idx] = mov_coord0, mov_coord1
            if ihatch==0:
                gcode_lines.append(f"G4 X{p0[0]:g} Y{p0[1]:g} Z{z} P0.5")
                gcode_lines.append(f"G4 P0.5 R1")
            else:
                positionning_line = f"G0 X{p0[0]:g} Y{p0[1]:g}"
                gcode_lines.append(positionning_line)
            printing_line = f"G1 X{p1[0]:g} Y{p1[1]:g} E{E:g}"
            gcode_lines.append(printing_line)

    gcode_file = params["source_terms"][0]["path"]
    with open(gcode_file,'w') as f:
        f.writelines("\n".join(gcode_lines))

def reference_run(params, writepos=True):
    domain = get_mesh(params)
    write_gcode(params)
    params["petsc_opts"] = params["petsc_opts_macro"]
    ps = Problem(domain, params, finalize_activation=False, name="ref")

    # Deactivate below surface
    midpoints_cells = mesh.compute_midpoints(ps.domain, ps.dim, np.arange(ps.num_cells))
    substrate_els = (midpoints_cells[:, 2] <= 0.0).nonzero()[0]
    ps.set_activation(substrate_els, finalize=True)
    ps.update_material_at_cells(substrate_els, ps.materials[1])

    ps.set_initial_condition(  params["environment_temperature"] )

    ps.set_forms()
    ps.compile_forms()
    adim_dt_print = params["substepping_parameters"]["micro_adim_dt"]
    adim_dt_cooling = params["substepping_parameters"]["cooling_adim_dt"]
    itime_step = 0
    while (not(ps.is_path_over()) and itime_step < params["max_timesteps"]):
        itime_step += 1
        track = ps.source.path.get_track(ps.time)
        if ps.source.path.get_track(ps.time).type in [TrackType.RECOATING,
                                                      TrackType.DWELLING]:
            ps.set_dt(ps.dimensionalize_waiting_timestep(track, adim_dt_cooling))
        else:
            ps.set_dt(ps.dimensionalize_mhs_timestep(track, adim_dt_print))
        ps.pre_iterate()
        ps.instantiate_forms()
        ps.pre_assemble()
        ps.non_linear_solve()
        ps.post_iterate()
        if writepos:
            ps.writepos(extra_funcs=[ps.u_av])
    return ps

if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    reference_run(params)
