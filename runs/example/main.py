from IPython import embed
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT

from mpi4py import MPI
import numpy as np
import dolfinx


mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)


V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2, )))
u = dolfinx.fem.Function(V)

u.interpolate(lambda x: (x[0], np.sin(x[1])))
u.x.scatter_forward()


def submesh_marker(x):
    return x[0] <= 0.5


submesh_entities = dolfinx.mesh.locate_entities(
    mesh, mesh.topology.dim, marker=lambda x: submesh_marker(x))
submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim, submesh_entities)

V_sub = dolfinx.fem.functionspace(submesh, V.ufl_element())
u_sub = dolfinx.fem.Function(V_sub)

num_sub_cells = submesh.topology.index_map(submesh.topology.dim).size_local
for cell in range(num_sub_cells):
    sub_dofs = V_sub.dofmap.cell_dofs(cell)
    parent_dofs = V.dofmap.cell_dofs(cell_map[cell])
    assert V_sub.dofmap.bs == V.dofmap.bs
    for parent, child in zip(parent_dofs, sub_dofs):
        for b in range(V_sub.dofmap.bs):
            u_sub.x.array[child*V_sub.dofmap.bs +
                          b] = u.x.array[parent*V.dofmap.bs+b]

u_sub.x.scatter_forward()
with dolfinx.io.XDMFFile(submesh.comm, "submesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(submesh)
    xdmf.write_function(u_sub)


with dolfinx.io.XDMFFile(submesh.comm, "parentmesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u)
