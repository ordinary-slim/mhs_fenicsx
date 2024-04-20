import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.io
import mpi4py.MPI
import numpy as np
import dolfinx.mesh
import petsc4py.PETSc
import ufl
import dolfinx.io
import gmsh

import multiphenicsx.fem
import multiphenicsx.fem.petsc

def getMesh():
    r = 3
    mesh_size = 1. / 4.
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

mesh, subdomains, boundaries_and_interfaces = getMesh()

cells_Omega1 = subdomains.indices[subdomains.values == 1]
cells_Omega2 = subdomains.indices[subdomains.values == 2]
facets_partial_Omega = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 1]
facets_Gamma = boundaries_and_interfaces.indices[boundaries_and_interfaces.values == 2]

# Define associated measures
dx = ufl.Measure("dx")(subdomain_data=subdomains)
dS = ufl.Measure("dS")(subdomain_data=boundaries_and_interfaces)
dS = dS(2)  # restrict to the interface, which has facet ID equal to 2
# Define function spaces
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
V1 = V.clone()
V2 = V.clone()
M = V.clone()

# Define restrictions
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
dofs_V1_Omega1 = dolfinx.fem.locate_dofs_topological(V1, subdomains.dim, cells_Omega1)
dofs_V2_Omega2 = dolfinx.fem.locate_dofs_topological(V2, subdomains.dim, cells_Omega2)
dofs_M_Gamma = dolfinx.fem.locate_dofs_topological(M, boundaries_and_interfaces.dim, facets_Gamma)
restriction_V1_Omega1 = multiphenicsx.fem.DofMapRestriction(V1.dofmap, dofs_V1_Omega1)
restriction_V2_Omega2 = multiphenicsx.fem.DofMapRestriction(V2.dofmap, dofs_V2_Omega2)
restriction_M_Gamma = multiphenicsx.fem.DofMapRestriction(M.dofmap, dofs_M_Gamma)
restriction = [restriction_V1_Omega1, restriction_V2_Omega2, restriction_M_Gamma]

# Define trial and test functions
(u1, u2, l) = (ufl.TrialFunction(V1), ufl.TrialFunction(V2), ufl.TrialFunction(M))
(v1, v2, m) = (ufl.TestFunction(V1), ufl.TestFunction(V2), ufl.TestFunction(M))

# Define problem block forms
zero = dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0))
a = [[ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx(1), None, ufl.inner(l("-"), v1("-")) * dS],
     [None, ufl.inner(ufl.grad(u2), ufl.grad(v2)) * dx(2), - ufl.inner(l("+"), v2("+")) * dS],
     [ufl.inner(u1("-"), m("-")) * dS, - ufl.inner(u2("+"), m("+")) * dS, None]]
f = [ufl.inner(1, v1) * dx(1), ufl.inner(1, v2) * dx(2), ufl.inner(zero, m("-")) * dS]
a_cpp = dolfinx.fem.form(a)
f_cpp = dolfinx.fem.form(f)

# Define boundary conditions
dofs_V1_partial_Omega = dolfinx.fem.locate_dofs_topological(
    V1, boundaries_and_interfaces.dim, facets_partial_Omega)
dofs_V2_partial_Omega = dolfinx.fem.locate_dofs_topological(
    V2, boundaries_and_interfaces.dim, facets_partial_Omega)
bc1 = dolfinx.fem.dirichletbc(zero, dofs_V1_partial_Omega, V1)
bc2 = dolfinx.fem.dirichletbc(zero, dofs_V2_partial_Omega, V2)
bcs = [bc1, bc2]

# Assemble the block linear system
A = multiphenicsx.fem.petsc.assemble_matrix_block(a_cpp, bcs=bcs, restriction=(restriction, restriction))
A.assemble()
F = multiphenicsx.fem.petsc.assemble_vector_block(f_cpp, a_cpp, bcs=bcs, restriction=restriction)

# Solve
u1u2l = multiphenicsx.fem.petsc.create_vector_block(f_cpp, restriction=restriction)
ksp = petsc4py.PETSc.KSP()
ksp.create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.getPC().setFactorSetUpSolverType()
ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=7, ival=4)
ksp.setFromOptions()
ksp.solve(F, u1u2l)
u1u2l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
ksp.destroy()

# Split the block solution in components
(u1, u2, l) = (dolfinx.fem.Function(V1), dolfinx.fem.Function(V2), dolfinx.fem.Function(M))
with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
        u1u2l, [V1.dofmap, V2.dofmap, M.dofmap], restriction) as u1u2l_wrapper:
    for u1u2l_wrapper_local, component in zip(u1u2l_wrapper, (u1, u2, l)):
        with component.vector.localForm() as component_local:
            component_local[:] = u1u2l_wrapper_local
u1u2l.destroy()

with dolfinx.io.XDMFFile(V1.mesh.comm, "out/res.xdmf", "w") as xdmf:
    xdmf.write_mesh(V1.mesh)
    xdmf.write_function(u1)
