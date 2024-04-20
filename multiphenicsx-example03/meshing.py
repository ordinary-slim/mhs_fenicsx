import mpi4py.MPI
import gmsh
import dolfinx.mesh
import dolfinx.io

## GEOMETRICAL PARAMS
r = 3
mesh_size = 1. / 4.

def getMesh():
    ## MESH
    gmsh.initialize()
    gmsh.model.add("mesh")
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
    p1 = gmsh.model.geo.addPoint(0.0, +r, 0.0, mesh_size)
    p2 = gmsh.model.geo.addPoint(0.0, -r, 0.0, mesh_size)
    c0 = gmsh.model.geo.addCircleArc(p1, p0, p2)
    c1 = gmsh.model.geo.addCircleArc(p2, p0, p1)
    l0 = gmsh.model.geo.addLine(p2, p1)
    line_loop_left = gmsh.model.geo.addCurveLoop([c0, l0])
    line_loop_right = gmsh.model.geo.addCurveLoop([c1, -l0])
    semicircle_left = gmsh.model.geo.addPlaneSurface([line_loop_left])
    semicircle_right = gmsh.model.geo.addPlaneSurface([line_loop_right])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [c0, c1], 1)
    gmsh.model.addPhysicalGroup(1, [l0], 2)
    gmsh.model.addPhysicalGroup(2, [semicircle_left], 1)
    gmsh.model.addPhysicalGroup(2, [semicircle_right], 2)
    gmsh.model.mesh.generate(2)

    partitioner = dolfinx.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh, subdomains, boundaries_and_interfaces = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, comm=mpi4py.MPI.COMM_WORLD, rank=0, gdim=2, partitioner=partitioner)
    gmsh.finalize()
    return mesh, subdomains, boundaries_and_interfaces
