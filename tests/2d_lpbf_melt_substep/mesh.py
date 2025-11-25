import gmsh
import yaml
import numpy as np
running_w_dolfinx = True
try:
    from dolfinx.io.gmshio import model_to_mesh
    from dolfinx.mesh import GhostMode, create_cell_partitioner
except ImportError:
    running_w_dolfinx = False
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def create_stacked_squares_mesh(params):
    gmsh.initialize()
    if rank == 0:
        gmsh.model.add("stacked_cubes")

        els_per_radius = params["els_per_radius"]
        radius = params["radius"]
        adim_pad_width = params["adim_pad_width"]
        part_el_size = radius / els_per_radius
        substrate_el_size = radius / params["substrate_els_per_radius"]

        layer_thickness = params["printer"]["layer_thickness"]
        num_layers = params["num_layers"]
        num_buffer_layers = params["num_buffer_layers"]
        substrate_depth = params["substrate_height"]
        L = params["path_width"] + 2 * params["adim_pad_width"] * params["radius"]
        part_height = layer_thickness * num_layers
        buffer_depth = layer_thickness * num_buffer_layers

        # ---------------- Points ----------------
        coords = {
            1: (-L/2, -buffer_depth, 0.0, part_el_size),
            2: (+L/2, -buffer_depth, 0.0, part_el_size),
            3: (+L/2, part_height, 0.0, part_el_size),
            4: (-L/2, part_height, 0.0, part_el_size),
            5: (-L/2, -substrate_depth, 0.0, substrate_el_size),
            6: (+L/2, -substrate_depth, 0.0, substrate_el_size),
        }

        for tag, (x, y, z, meshSize) in coords.items():
            gmsh.model.geo.addPoint(x, y, z, tag=tag, meshSize=meshSize)

        # ---------------- Lines ----------------
        lines = [
            (1,1,2), # H
            (2,2,3), # V
            (3,3,4), # H
            (4,4,1), # V
            (5,5,6), # H
            (6,6,2), # V
            (7,1,5), # V
        ]
        for tag, p1, p2 in lines:
            gmsh.model.geo.addLine(p1, p2, tag)

        gmsh.model.geo.addCurveLoop([1,2,3,4], 21)
        gmsh.model.geo.addPlaneSurface([21], 21)

        gmsh.model.geo.addCurveLoop([5,6,-1,7], 22)
        gmsh.model.geo.addPlaneSurface([22], 22)

        # ---------------- Mesh settings ----------------
        n_xy = int(np.rint(L / part_el_size) + 1)
        n_z = int(np.rint(((part_height + buffer_depth) / part_el_size)) + 1)
        for curve in [1,3]:
            gmsh.model.geo.mesh.setTransfiniteCurve(curve, n_xy)
        for curve in [2,4]:
            gmsh.model.geo.mesh.setTransfiniteCurve(curve, n_z)
        gmsh.model.geo.mesh.setTransfiniteSurface(21)

        # ---------------- Mesh generation ----------------
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [21, 22], name="Domain")
        gmsh.model.mesh.generate(2)

    if __name__ == "__main__":
        gmsh.fltk.run()
        gmsh.finalize()
        exit()
    elif running_w_dolfinx:
        model = MPI.COMM_WORLD.bcast(gmsh.model, root = 0)
        partitioner = create_cell_partitioner(GhostMode.shared_facet)
        msh_data = model_to_mesh(model, MPI.COMM_WORLD, 0,
                                 gdim=2, partitioner=partitioner)
        msh = msh_data[0]
        gmsh.finalize()
        MPI.COMM_WORLD.barrier()
        return msh

if __name__ == "__main__":
    with open("input.yaml", "r") as f:
        params = yaml.safe_load(f)
    create_stacked_squares_mesh(params)
