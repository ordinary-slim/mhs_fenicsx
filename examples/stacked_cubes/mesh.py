import gmsh
import yaml
running_w_dolfinx = True
try:
    from dolfinx.io.gmshio import model_to_mesh
    from dolfinx.mesh import GhostMode, create_cell_partitioner
except ImportError:
    running_w_dolfinx = False
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def create_stacked_cubes_mesh(params):
    gmsh.initialize()
    if rank == 0:
        gmsh.model.add("stacked_cubes")

        fine_el_size = params["fine_el_size"]
        coarse_el_size = params["coarse_el_size"]
        layer_thickness = params["layer_thickness"]
        num_layers = params["num_layers"]
        num_buffer_layers = params["num_buffer_layers"]
        substrate_depth = params["substrate_depth"]
        L = params["width"]
        part_height = layer_thickness * num_layers
        buffer_depth = layer_thickness * num_buffer_layers

        # ---------------- Points ----------------
        coords = {
            1: (-L/2, -L/2, -buffer_depth, fine_el_size),
            2: (+L/2, -L/2, -buffer_depth, fine_el_size),
            3: (+L/2, +L/2, -buffer_depth, fine_el_size),
            4: (-L/2, +L/2, -buffer_depth, fine_el_size),
            5: (-L/2, -L/2, part_height, fine_el_size),
            6: (+L/2, -L/2, part_height, fine_el_size),
            7: (+L/2, +L/2, part_height, fine_el_size),
            8: (-L/2, +L/2, part_height, fine_el_size),
             9: (-L/2, -L/2, -substrate_depth, coarse_el_size),
            10: (+L/2, -L/2, -substrate_depth, coarse_el_size),
            11: (+L/2, +L/2, -substrate_depth, coarse_el_size),
            12: (-L/2, +L/2, -substrate_depth, coarse_el_size),
        }

        for tag, (x, y, z, meshSize) in coords.items():
            gmsh.model.geo.addPoint(x, y, z, tag=tag, meshSize=meshSize)

        # ---------------- Lines ----------------
        lines = [
            (1,1,2), (2,2,3), (3,3,4), (4,4,1),
            (5,5,6), (6,6,7), (7,7,8), (8,8,5),
            (9,1,5), (10,2,6), (11,3,7), (12,4,8),
            (13,9,10), (14,10,11), (15,11,12), (16,12,9),
            (17,9,1), (18,10,2), (19,11,3), (20,12,4),
        ]
        for tag, p1, p2 in lines:
            gmsh.model.geo.addLine(p1, p2, tag)

        # ---------------- Surfaces for upper cube ----------------
        gmsh.model.geo.addCurveLoop([1,2,3,4], 21)
        gmsh.model.geo.addPlaneSurface([21], 21)

        gmsh.model.geo.addCurveLoop([5,6,7,8], 22)
        gmsh.model.geo.addPlaneSurface([22], 22)

        gmsh.model.geo.addCurveLoop([1,10,-5,-9],23)
        gmsh.model.geo.addPlaneSurface([23],23)

        gmsh.model.geo.addCurveLoop([2,11,-6,-10],24)
        gmsh.model.geo.addPlaneSurface([24],24)

        gmsh.model.geo.addCurveLoop([3,12,-7,-11],25)
        gmsh.model.geo.addPlaneSurface([25],25)

        gmsh.model.geo.addCurveLoop([4,9,-8,-12],26)
        gmsh.model.geo.addPlaneSurface([26],26)

        gmsh.model.geo.addSurfaceLoop([21,22,23,24,25,26],30)
        gmsh.model.geo.addVolume([30],1)

        # ---------------- Surfaces for lower cube ----------------
        gmsh.model.geo.addCurveLoop([13,14,15,16],27)
        gmsh.model.geo.addPlaneSurface([27],27)

        gmsh.model.geo.addCurveLoop([1,-18,-13,+17],29)
        gmsh.model.geo.addPlaneSurface([29],29)

        gmsh.model.geo.addCurveLoop([2,-19,-14,+18],30)
        gmsh.model.geo.addPlaneSurface([30],30)

        gmsh.model.geo.addCurveLoop([3,-20,-15,+19],31)
        gmsh.model.geo.addPlaneSurface([31],31)

        gmsh.model.geo.addCurveLoop([4,-17,-16,+20],32)
        gmsh.model.geo.addPlaneSurface([32],32)

        gmsh.model.geo.addSurfaceLoop([27,21,29,30,31,32],40)
        gmsh.model.geo.addVolume([40],2)

        # ---------------- Mesh settings ----------------
        fine_el_size = params["fine_el_size"]
        n_xy = int(L / fine_el_size) + 1
        n_z = int((part_height + buffer_depth) / fine_el_size) + 1
        for curve in [1,2,3,4,5,6,7,8]:
            gmsh.model.geo.mesh.setTransfiniteCurve(curve, n_xy)
        for curve in [9,10,11,12]:
            gmsh.model.geo.mesh.setTransfiniteCurve(curve, n_z)
        for surface in [21,22,23,24,25,26]:
            gmsh.model.geo.mesh.setTransfiniteSurface(surface)
        gmsh.model.geo.mesh.setTransfiniteVolume(1)

        # ---------------- Mesh generation ----------------
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [1, 2], name="Domain")
        gmsh.model.mesh.generate(3)

    if __name__ == "__main__":
        gmsh.fltk.run()
        gmsh.finalize()
        exit()
    elif running_w_dolfinx:
        model = MPI.COMM_WORLD.bcast(gmsh.model, root = 0)
        partitioner = create_cell_partitioner(GhostMode.shared_facet)
        msh_data = model_to_mesh(model, MPI.COMM_WORLD, 0,
                                 gdim=3, partitioner=partitioner)
        msh = msh_data[0]
        gmsh.finalize()
        MPI.COMM_WORLD.barrier()
        return msh

if __name__ == "__main__":
    with open("input.yaml", "r") as f:
        params = yaml.safe_load(f)
    create_stacked_cubes_mesh(params)
