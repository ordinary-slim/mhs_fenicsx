from dolfinx import fem, mesh
from basix.ufl import element
import ufl
from mpi4py import MPI
import basix
import numpy as np
import gmsh
from dolfinx.io.gmshio import model_to_mesh

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dolfinx_to_basix_ctype = {
                        mesh.CellType.point:basix.CellType.point,
                        mesh.CellType.interval:basix.CellType.interval,
                        mesh.CellType.triangle:basix.CellType.triangle,
                        mesh.CellType.quadrilateral:basix.CellType.quadrilateral,
                        mesh.CellType.tetrahedron:basix.CellType.tetrahedron,
                        mesh.CellType.hexahedron:basix.CellType.hexahedron,
                        }

def generate_facet_cell_quadrature(domain):
    tdim = domain.topology.dim
    fdim = tdim - 1
    cmap = domain.geometry.cmap
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_cells  = mesh.compute_incident_entities(domain.topology, boundary_facets, fdim, tdim) 
    fcelltype = dolfinx_to_basix_ctype[domain.topology.entity_types[-2][0]]
    celltype = dolfinx_to_basix_ctype[domain.topology.cell_type]
    sub_entity_connectivity = basix.cell.sub_entity_connectivity(celltype)
    num_facets_el = len(sub_entity_connectivity[tdim-1])

    gpoints_facet, gweights_facet = basix.make_quadrature(fcelltype, 2)
    # TODO: Tabulate facet element at gpoints
    felement = basix.create_element(
        basix.ElementFamily.P, fcelltype, cmap.degree, basix.LagrangeVariant.equispaced)
    ftab = felement.tabulate(0, gpoints_facet)

    num_gpoints_facet = len(gweights_facet)
    gpoints_cell = np.zeros((num_facets_el*num_gpoints_facet, tdim), dtype=gpoints_facet.dtype)
    gweights_cell = np.zeros(num_facets_el*num_gpoints_facet, dtype=gweights_facet.dtype)
    vertices = basix.geometry(celltype)
    for ifacet in range(num_facets_el):
        facet = sub_entity_connectivity[tdim-1][ifacet][0]
        for igp in range(num_gpoints_facet):
            idx = ifacet*num_gpoints_facet + igp
            gweights_cell[idx] = gweights_facet[igp]
            for idof in range(len(facet)):
                gpoints_cell[idx] += vertices[facet[idof]] * ftab[0][igp][idof][0]
    return gpoints_cell, gweights_cell, num_gpoints_facet

def ref_tria_mesh():
    # Define vertices of the tetrahedron
    vertices = np.array([
    [0.0, 0.0],  # Vertex 0
    [1.0, 0.0],  # Vertex 1
    [0.0, 1.0],  # Vertex 3
    ], dtype=np.float64)

    # Define the cell connectivity (single tetrahedron)
    cells = np.array([
        [0, 1, 2]  # Indices of the vertices making up the tetrahedron
    ], dtype=np.int32)

    # Create the mesh
    basix_d = basix.create_element(basix.ElementFamily.P, basix.CellType["triangle"], 1)
    return mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, basix_d)

def ref_quadri_mesh():
    # Define vertices of the tetrahedron
    vertices = np.array([
    [0.0, 0.0],  # Vertex 0
    [1.0, 0.0],  # Vertex 1
    [1.0, 1.0],  # Vertex 2
    [0.0, 1.0],  # Vertex 3
    ], dtype=np.float64)

    # Define the cell connectivity (single tetrahedron)
    cells = np.array([
        [0, 1, 2, 3]  # Indices of the vertices making up the tetrahedron
    ], dtype=np.int32)

    # Create the mesh
    basix_d = basix.create_element(basix.ElementFamily.P, basix.CellType["quadrilateral"], 1)
    return mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, basix_d)

def ref_tetra_mesh():
    # Define vertices of the tetrahedron
    vertices = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.0, 1.0, 0.0],  # Vertex 2
        [0.0, 0.0, 1.0],  # Vertex 3
    ], dtype=np.float64)

    # Define the cell connectivity (single tetrahedron)
    cells = np.array([
        [0, 1, 2, 3]  # Indices of the vertices making up the tetrahedron
    ], dtype=np.int32)

    # Create the mesh
    basix_d = basix.create_element(basix.ElementFamily.P, basix.CellType["tetrahedron"], 1)
    return mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, basix_d)

def ref_hexa_mesh():
    # Define vertices of the hexahedron (a unit cube)
    vertices = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.0, 1.0, 0.0],  # Vertex 2
        [1.0, 1.0, 0.0],  # Vertex 3
        [0.0, 0.0, 1.0],  # Vertex 4
        [1.0, 0.0, 1.0],  # Vertex 5
        [0.0, 1.0, 1.0],  # Vertex 6
        [1.0, 1.0, 1.0],  # Vertex 7
    ], dtype=np.float64) 

    # Define the cell connectivity (single hexahedron)
    cells = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7]  # Indices of the vertices making up the hexahedron
    ], dtype=np.int32)

    # Create the mesh
    basix_d = basix.create_element(basix.ElementFamily.P, basix.CellType["hexahedron"], 1)
    return mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, basix_d)

def ref_tria_mesh_gmsh():
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("SingleTriangle")

    # Define the points of the triangle
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)  # Vertex 0
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)  # Vertex 1
    p3 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)  # Vertex 2

    # Define the edges (lines) of the triangle
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p1)

    # Create a curve loop and a surface
    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Synchronize the Gmsh model to process the geometry
    gmsh.model.geo.synchronize()

    # Specify meshing options for a single element
    gmsh.option.setNumber("Mesh.MeshSizeMin", 1e3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1e3)

    # Set mesh element type to triangles (default)
    gmsh.model.mesh.generate(2)
    volumeTags = []
    for _, tag in gmsh.model.getEntities( 2 ):
        volumeTags.append( tag )
    gmsh.model.addPhysicalGroup(2, volumeTags, tag=1, name="Domain")
    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, _, _ = model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2,partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    return msh

def ref_quadri_mesh_gmsh():
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("SingleQuadrilateral")

    # Define the points of the quadrilateral (a unit square)
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)  # Vertex 0
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)  # Vertex 1
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0)  # Vertex 2
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)  # Vertex 3

    # Define the edges (lines) of the quadrilateral
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create a curve loop and a surface
    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Synchronize the Gmsh model
    gmsh.model.geo.synchronize()

    # Set transfinite meshing for the geometry
    # This ensures a structured grid with exactly one element
    gmsh.model.mesh.setTransfiniteCurve(l1, 2)  # 2 points (start and end)
    gmsh.model.mesh.setTransfiniteCurve(l2, 2)
    gmsh.model.mesh.setTransfiniteCurve(l3, 2)
    gmsh.model.mesh.setTransfiniteCurve(l4, 2)
    gmsh.model.mesh.setTransfiniteSurface(surface)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine to make quadrilaterals

    # Generate the mesh
    gmsh.model.mesh.generate(2)
    volumeTags = []
    for _, tag in gmsh.model.getEntities( 2 ):
        volumeTags.append( tag )
    gmsh.model.addPhysicalGroup(2, volumeTags, tag=1, name="Domain")
    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, _, _ = model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2,partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    return msh

def ref_tetra_mesh_gmsh():
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("SingleTetrahedron")

    # Define the points of the tetrahedron (a unit tetrahedron)
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)  # Vertex 0
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)  # Vertex 1
    p3 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)  # Vertex 2
    p4 = gmsh.model.geo.addPoint(0.0, 0.0, 1.0)  # Vertex 3

    # Define the edges (lines)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p1)
    l4 = gmsh.model.geo.addLine(p1, p4)
    l5 = gmsh.model.geo.addLine(p2, p4)
    l6 = gmsh.model.geo.addLine(p3, p4)

    # Define the faces (triangles) of the tetrahedron
    face1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3])])  # Base
    face2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l5, -l4])])  # Side 1
    face3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l2, l6, -l5])])  # Side 2
    face4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l3, l4, -l6])])  # Side 3

    # Create a volume (tetrahedron) from the faces
    volume = gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([face1, face2, face3, face4])])

    # Synchronize the Gmsh model
    gmsh.model.geo.synchronize()

    # Specify meshing options for a single element
    gmsh.option.setNumber("Mesh.MeshSizeMin", 1e3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1e3)

    # Generate the mesh
    gmsh.model.mesh.generate(3)
    volumeTags = []
    for _, tag in gmsh.model.getEntities( 3 ):
        volumeTags.append( tag )
    gmsh.model.addPhysicalGroup(3, volumeTags, tag=1, name="Domain")
    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, _, _ = model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3,partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    return msh

def ref_hexa_mesh_gmsh():
    # Initialize Gmsh
    gmsh.initialize()
    # Initialize Gmsh gmsh.initialize()
    gmsh.model.add("SingleHexahedron")

    # Define the points of the hexahedron (a unit cube)
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)  # Vertex 1 (0,0,0)
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0)  # Vertex 2 (1,0,0)
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0)  # Vertex 3 (1,1,0)
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0)  # Vertex 4 (0,1,0)
    p5 = gmsh.model.geo.addPoint(0.0, 0.0, 1.0)  # Vertex 5 (0,0,1)
    p6 = gmsh.model.geo.addPoint(1.0, 0.0, 1.0)  # Vertex 6 (1,0,1)
    p7 = gmsh.model.geo.addPoint(1.0, 1.0, 1.0)  # Vertex 7 (1,1,1)
    p8 = gmsh.model.geo.addPoint(0.0, 1.0, 1.0)  # Vertex 8 (0,1,1)

    # Define the edges (lines) of the hexahedron
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    l9 = gmsh.model.geo.addLine(p2, p6)
    l10 = gmsh.model.geo.addLine(p1, p5)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)

    # Define the faces (quadrilaterals) of the hexahedron
    f1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])  # Bottom
    f2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])])  # Top
    f3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l9, -l5, -l10])])  # Front
    f4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l9])])  # Right
    f5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([-l3, l11, l7, -l12])])  # Back
    f6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([-l4, l12, l8, -l10])])  # Left

    # Create a volume (hexahedron) from the faces
    volume = gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([f1, f2, f3, f4, f5, f6])])

    # Synchronize the Gmsh model
    gmsh.model.geo.synchronize()

    # Set transfinite meshing for the volume to enforce a single hexahedral element
    gmsh.model.mesh.setTransfiniteVolume(volume)
    gmsh.model.mesh.setTransfiniteSurface(f1)
    gmsh.model.mesh.setTransfiniteSurface(f2)
    gmsh.model.mesh.setTransfiniteSurface(f3)
    gmsh.model.mesh.setTransfiniteSurface(f4)
    gmsh.model.mesh.setTransfiniteSurface(f5)
    gmsh.model.mesh.setTransfiniteSurface(f6)
    for l in [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12]:
        gmsh.model.mesh.setTransfiniteCurve(l, 2)  # 2 points (start and end)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine to make quadrilaterals

    # Generate the mesh
    gmsh.model.mesh.generate(3)
    volumeTags = []
    for _, tag in gmsh.model.getEntities( 3 ):
        volumeTags.append( tag )
    gmsh.model.addPhysicalGroup(3, volumeTags, tag=1, name="Domain")
    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
    msh, _, _ = model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3,partitioner=partitioner)

    gmsh.finalize()
    MPI.COMM_WORLD.barrier()
    return msh


if __name__=="__main__":
    ref_hexa_mesh_gmsh()
