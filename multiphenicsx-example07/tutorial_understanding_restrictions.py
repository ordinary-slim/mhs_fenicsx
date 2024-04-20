#!/usr/bin/env python
# coding: utf-8

# # Tutorial 07: understanding restrictions
import os
import typing
import basix
import basix.ufl
import dolfinx.fem
import dolfinx.io
import dolfinx.mesh
import gmsh
import mpi4py.MPI
import mpl_toolkits.axes_grid1
import numpy as np
import multiphenicsx.fem


def count_dofs(restriction: multiphenicsx.fem.DofMapRestriction, comm: mpi4py.MPI.Intracomm) -> int:
    """Count the DOFs in a DofMapRestriction object."""
    u2r = restriction.unrestricted_to_restricted
    restricted_local_indices = np.array([r for (_, r) in u2r.items()], dtype=np.int32)
    dofs_V_restriction_global = restriction.index_map.local_to_global(restricted_local_indices)
    return len(set(gdof for gdofs in comm.allgather(dofs_V_restriction_global) for gdof in gdofs))


# In[ ]:


def locate_dofs_by_polar_coordinates(
    r: typing.Union[int, float, np.float64], theta: typing.Union[int, float, np.float64],
    V: dolfinx.fem.FunctionSpace, restriction: multiphenicsx.fem.DofMapRestriction
) -> set[np.int32]:
    """Determine which DOFs in a DofMapRestriction object are located at a point (written in polar coordinates)."""
    p = [r * np.cos(theta), r * np.sin(theta), 0.]
    dofs_V = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x.T, p).all(axis=1)).reshape(-1)
    u2r = restriction.unrestricted_to_restricted
    restricted_local_indices = np.array([u2r[dof] for dof in dofs_V if dof in u2r], dtype=np.int32)
    dofs_V_restriction_global = restriction.index_map.local_to_global(restricted_local_indices)
    return set(gdof for gdofs in V.mesh.comm.allgather(dofs_V_restriction_global) for gdof in gdofs)

r = 1
mesh_size = 2
# ### Mesh
# Generate a simple mesh consisting in an hexagon discretized with six equilateral triangle cells.
gmsh.initialize()
gmsh.model.add("mesh")
points = [
    gmsh.model.geo.addPoint(np.cos(t / 3 * np.pi), np.sin(t / 3 * np.pi), 0.0, mesh_size) for t in range(6)]
lines = [gmsh.model.geo.addLine(points[t], points[(t + 1) % 6]) for t in range(6)]
line_loop = gmsh.model.geo.addCurveLoop(lines)
domain = gmsh.model.geo.addPlaneSurface([line_loop])
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, lines, 0)
gmsh.model.addPhysicalGroup(2, [domain], 0)
gmsh.model.mesh.generate(2)
mesh, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, comm=mpi4py.MPI.COMM_WORLD, rank=0, gdim=2)
gmsh.finalize()

mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.topology.create_connectivity(mesh.topology.dim, 0)
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)

# ### Mesh restrictions on cells
# Define mesh tags on cells, which are equal to one on all cells.
cell_entities_all = dolfinx.mesh.locate_entities(
    mesh, mesh.topology.dim, lambda x: np.full((x.shape[1], ), True))
cell_values_all = np.full(cell_entities_all.shape, 1, dtype=np.int32)
cell_restriction_all = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim, cell_entities_all, cell_values_all)
cell_restriction_all.name = "cell_restriction_all"

# Define mesh tags on cells, which are equal to one on one half of the cells

# In[ ]:


eps = np.finfo(float).eps
cell_entities_subset = dolfinx.mesh.locate_entities(
    mesh, mesh.topology.dim,
    lambda x: np.logical_or(x[0] < eps, np.logical_and(x[1] < eps, x[0] < 0.5 + eps)))
cell_values_subset = np.full(cell_entities_subset.shape, 1, dtype=np.int32)
cell_restriction_subset = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim, cell_entities_subset, cell_values_subset)
cell_restriction_subset.name = "cell_restriction_subset"


# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    plot_mesh_tags(mesh, cell_restriction_subset)


# ### Mesh restrictions on facets
# 
# Define mesh tags on facets, which are equal to one on all facets

# In[ ]:


facet_entities_all = dolfinx.mesh.locate_entities(
    mesh, mesh.topology.dim - 1, lambda x: np.full((x.shape[1], ), True))
facet_values_all = np.full(facet_entities_all.shape, 1, dtype=np.int32)
facet_restriction_all = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim - 1, facet_entities_all, facet_values_all)
facet_restriction_all.name = "facet_restriction_all"


# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    plot_mesh_tags(mesh, facet_restriction_all)


# Define mesh tags on facets, which are equal to one on two facets

# In[ ]:


facet_entities_subset = dolfinx.mesh.locate_entities(
    mesh, mesh.topology.dim - 1, lambda x: np.fabs(x[1] + np.sqrt(3) * x[0]) < 0.01)
facet_values_subset = np.full(facet_entities_subset.shape, 1, dtype=np.int32)
facet_restriction_subset = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim - 1, facet_entities_subset, facet_values_subset)
facet_restriction_subset.name = "facet_restriction_subset"


# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    plot_mesh_tags(mesh, facet_restriction_subset)


# ### Mesh restrictions summary

# In[ ]:


cell_restrictions = (cell_restriction_all, cell_restriction_subset)
facet_restrictions = (facet_restriction_all, facet_restriction_subset)
all_restrictions = cell_restrictions + facet_restrictions


# ### Lagrange spaces
# 
# Define Lagrange FE spaces of order $k=1, 2, 3$, and plot the associated DofMap.

# In[ ]:


CG_elem = [
    basix.ufl.element(
        "Lagrange", mesh.basix_cell(), k, lagrange_variant=basix.LagrangeVariant.equispaced
    ) for k in (1, 2, 3)
]
CG = [dolfinx.fem.functionspace(mesh, CG_elem_k) for CG_elem_k in CG_elem]


# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 1, figsize=(10, 30))
    for (k, CGk) in enumerate(CG):
        plot_mesh(mesh, ax[k])
        plot_dofmap(CGk, ax[k])
        ax[k].set_title("CG " + str(k + 1) + " DofMap", fontsize=30)


# Define DofMapRestriction objects associated to the Lagrange FE spaces, for all four restrictions

# In[ ]:


dofmap_restriction_CG: dict[
    dolfinx.mesh.MeshTags, list[multiphenicsx.fem.DofMapRestriction]] = dict()
for restriction in all_restrictions:
    dofmap_restriction_CG[restriction] = list()
    for CGk in CG:
        restrict_CGk = dolfinx.fem.locate_dofs_topological(
            CGk, restriction.dim, restriction.indices[restriction.values == 1])
        dofmap_restriction_CG[restriction].append(
            multiphenicsx.fem.DofMapRestriction(CGk.dofmap, restrict_CGk))


# Compare DOFs for the case of cell restriction equal to one on the entire domain. There is indeed no difference.

# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for (k, CGk) in enumerate(CG):
        plot_mesh(mesh, ax[k, 0])
        plot_dofmap(CGk, ax[k, 0])
        ax[k, 0].set_title("CG " + str(k + 1) + " DofMap", fontsize=30)
    for (k, (CGk, dofmap_restriction_CGk)) in enumerate(zip(CG, dofmap_restriction_CG[cell_restriction_all])):
        plot_mesh_tags(mesh, cell_restriction_all, ax[k, 1])
        plot_dofmap_restriction(CGk, dofmap_restriction_CGk, ax[k, 1])
        ax[k, 1].set_title("CG " + str(k + 1) + " DofMapRestriction", fontsize=30)


# Assert that DOFs are at the expected locations

# In[ ]:


CG_1 = CG[0]
dofmap_restriction_CG_1 = dofmap_restriction_CG[cell_restriction_all][0]
assert count_dofs(dofmap_restriction_CG_1, mesh.comm) == 7
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 0, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1


# In[ ]:


CG_2 = CG[1]
dofmap_restriction_CG_2 = dofmap_restriction_CG[cell_restriction_all][1]
assert count_dofs(dofmap_restriction_CG_2, mesh.comm) == 19
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 0, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 4 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 0, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 3 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 5 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 7 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 9 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 11 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1


# In[ ]:


CG_3 = CG[2]
dofmap_restriction_CG_3 = dofmap_restriction_CG[cell_restriction_all][2]
assert count_dofs(dofmap_restriction_CG_3, mesh.comm) == 37
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 3 * np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 5 * np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 7 * np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 9 * np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 11 * np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 2 * np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 2 * np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 4 * np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 4 * np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 5 * np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 5 * np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1


# Compare DOFs for che case of cell restriction equal to one on a subset of the domain. Note how the DofMapRestriction has only a subset of the DOFs of the DofMap, and properly renumbers them.

# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for (k, CGk) in enumerate(CG):
        plot_mesh(mesh, ax[k, 0])
        plot_dofmap(CGk, ax[k, 0])
        ax[k, 0].set_title("CG " + str(k + 1) + " DofMap", fontsize=30)
    for (k, (CGk, dofmap_restriction_CGk)) in enumerate(zip(CG, dofmap_restriction_CG[cell_restriction_subset])):
        plot_mesh_tags(mesh, cell_restriction_subset, ax[k, 1])
        plot_dofmap_restriction(CGk, dofmap_restriction_CGk, ax[k, 1])
        ax[k, 1].set_title("CG " + str(k + 1) + " DofMapRestriction", fontsize=30)


# Assert that DOFs are at the expected locations

# In[ ]:


CG_1 = CG[0]
dofmap_restriction_CG_1 = dofmap_restriction_CG[cell_restriction_subset][0]
assert count_dofs(dofmap_restriction_CG_1, mesh.comm) == 5
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1


# In[ ]:


CG_2 = CG[1]
dofmap_restriction_CG_2 = dofmap_restriction_CG[cell_restriction_subset][1]
assert count_dofs(dofmap_restriction_CG_2, mesh.comm) == 12
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 4 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 5 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 7 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 9 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1


# In[ ]:


CG_3 = CG[2]
dofmap_restriction_CG_3 = dofmap_restriction_CG[cell_restriction_subset][2]
assert count_dofs(dofmap_restriction_CG_3, mesh.comm) == 22
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 5 * np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 7 * np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 9 * np.pi / 6, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 2 * np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 4 * np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 4 * np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 5 * np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1


# Compare DOFs for che case of facet restriction equal to one on the entire domain. Note how there is no difference for $k=1, 2$, but the cases $k=3$ differ (the DofMapRestriction does not have the DOF at the cell center).

# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for (k, CGk) in enumerate(CG):
        plot_mesh(mesh, ax[k, 0])
        plot_dofmap(CG[k], ax[k, 0])
        ax[k, 0].set_title("CG " + str(k + 1) + " DofMap", fontsize=30)
    for (k, (CGk, dofmap_restriction_CGk)) in enumerate(zip(CG, dofmap_restriction_CG[facet_restriction_all])):
        plot_mesh_tags(mesh, facet_restriction_all, ax[k, 1])
        plot_dofmap_restriction(CGk, dofmap_restriction_CGk, ax[k, 1])
        ax[k, 1].set_title("CG " + str(k + 1) + " DofMapRestriction", fontsize=30)


# Assert that DOFs are at the expected locations

# In[ ]:


CG_1 = CG[0]
dofmap_restriction_CG_1 = dofmap_restriction_CG[facet_restriction_all][0]
assert count_dofs(dofmap_restriction_CG_1, mesh.comm) == 7
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 0, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1


# In[ ]:


CG_2 = CG[1]
dofmap_restriction_CG_2 = dofmap_restriction_CG[facet_restriction_all][1]
assert count_dofs(dofmap_restriction_CG_2, mesh.comm) == 19
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 0, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 4 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 0, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 3 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 5 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 7 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 9 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 11 * np.pi / 6, CG_2, dofmap_restriction_CG_2)) == 1


# In[ ]:


CG_3 = CG[2]
dofmap_restriction_CG_3 = dofmap_restriction_CG[facet_restriction_all][2]
assert count_dofs(dofmap_restriction_CG_3, mesh.comm) == 31
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 2 * np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 2 * np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 4 * np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 4 * np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 5 * np.pi / 3 - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, 5 * np.pi / 3 + np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(7) / 3, np.pi - np.arctan(np.sqrt(3) / 5), CG_3, dofmap_restriction_CG_3)) == 1


# Compare DOFs for che case of facet restriction equal to one on a subset of the domain. Note how the DofMapRestriction has only a subset of the DOFs of the DofMap, and properly renumbers them.

# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for (k, CGk) in enumerate(CG):
        plot_mesh(mesh, ax[k, 0])
        plot_dofmap(CG[k], ax[k, 0])
        ax[k, 0].set_title("CG " + str(k + 1) + " DofMap", fontsize=30)
    for (k, (CGk, dofmap_restriction_CGk)) in enumerate(zip(CG, dofmap_restriction_CG[facet_restriction_subset])):
        plot_mesh_tags(mesh, facet_restriction_subset, ax[k, 1])
        plot_dofmap_restriction(CGk, dofmap_restriction_CGk, ax[k, 1])
        ax[k, 1].set_title("CG " + str(k + 1) + " DofMapRestriction", fontsize=30)


# Assert that DOFs are at the expected locations

# In[ ]:


CG_1 = CG[0]
dofmap_restriction_CG_1 = dofmap_restriction_CG[facet_restriction_subset][0]
assert count_dofs(dofmap_restriction_CG_1, mesh.comm) == 3
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_1, dofmap_restriction_CG_1)) == 1


# In[ ]:


CG_2 = CG[1]
dofmap_restriction_CG_2 = dofmap_restriction_CG[facet_restriction_subset][1]
assert count_dofs(dofmap_restriction_CG_2, mesh.comm) == 5
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_2, dofmap_restriction_CG_2)) == 1


# In[ ]:


CG_3 = CG[2]
dofmap_restriction_CG_3 = dofmap_restriction_CG[facet_restriction_subset][2]
assert count_dofs(dofmap_restriction_CG_3, mesh.comm) == 7
assert len(locate_dofs_by_polar_coordinates(
    0, 0, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1 / 3, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    2 / 3, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, CG_3, dofmap_restriction_CG_3)) == 1


# ### Discontinuous Galerkin spaces
# 
# Define Discontinuous Galerkin spaces of order $k = 0, 1, 2$, and plot the associated DofMap. This spaces will be used in combination with a cell restriction, as DG DOFs are associated to cells.

# In[ ]:


DG = [dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange", k)) for k in (0, 1, 2)]


# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 1, figsize=(10, 30))
    for (k, DGk) in enumerate(DG):
        plot_mesh(mesh, ax[k])
        plot_dofmap(DGk, ax[k])
        ax[k].set_title("DG " + str(k) + " DofMap", fontsize=30)


# Define Discontinuous Galerkin Trace spaces of order $k = 0, 1, 2, 3$, and plot the associated DofMap. This spaces will be used in combination with a facet restriction, as DGT DOFs are associated to facets.

# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
DGT = [dolfinx.fem.functionspace(mesh, ("Discontinuous Lagrange Trace", k)) for k in (0, 1, 2)]


# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 1, figsize=(10, 30))
    for (k, DGTk) in enumerate(DGT):
        plot_mesh(mesh, ax[k])
        plot_dofmap(DGTk, ax[k])
        ax[k].set_title("DGT " + str(k) + " DofMap\n", fontsize=30)


# Define DofMapRestriction objects associated to the Discontinuos Galerkin FE spaces, for all cell restrictions

# In[ ]:


dofmap_restriction_DG: dict[
    dolfinx.mesh.MeshTags, list[multiphenicsx.fem.DofMapRestriction]] = dict()
for restriction in cell_restrictions:
    dofmap_restriction_DG[restriction] = list()
    for DGk in DG:
        restrict_DGk = dolfinx.fem.locate_dofs_topological(
            DGk, restriction.dim, restriction.indices[restriction.values == 1])
        dofmap_restriction_DG[restriction].append(
            multiphenicsx.fem.DofMapRestriction(DGk.dofmap, restrict_DGk))


# Define DofMapRestriction objects associated to the Discontinuos Galerkin Trace FE spaces, for all facet restrictions

# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
dofmap_restriction_DGT: dict[
    dolfinx.mesh.MeshTags, list[multiphenicsx.fem.DofMapRestriction]] = dict()
for restriction in facet_restrictions:
    dofmap_restriction_DGT[restriction] = list()
    for DGTk in DGT:
        restrict_DGTk = dolfinx.fem.locate_dofs_topological(
            DGTk, restriction.dim, restriction.indices[restriction.values == 1])
        dofmap_restriction_DGT[restriction].append(
            multiphenicsx.fem.DofMapRestriction(DGTk.dofmap, restrict_DGTk))


# Compare DOFs for the case of cell restriction equal to one on the entire domain. There is indeed no difference.

# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for (k, DGk) in enumerate(DG):
        plot_mesh(mesh, ax[k, 0])
        plot_dofmap(DGk, ax[k, 0])
        ax[k, 0].set_title("DG " + str(k) + " DofMap", fontsize=30)
    for (k, (DGk, dofmap_restriction_DGk)) in enumerate(zip(DG, dofmap_restriction_DG[cell_restriction_all])):
        plot_mesh_tags(mesh, cell_restriction_all, ax[k, 1])
        plot_dofmap_restriction(DGk, dofmap_restriction_DGk, ax[k, 1])
        ax[k, 1].set_title("DG " + str(k) + " DofMapRestriction", fontsize=30)


# Assert that DOFs are at the expected locations

# In[ ]:


DG_0 = DG[0]
dofmap_restriction_DG_0 = dofmap_restriction_DG[cell_restriction_all][0]
assert count_dofs(dofmap_restriction_DG_0, mesh.comm) == 6
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 3 * np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 5 * np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 7 * np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 9 * np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 11 * np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1


# In[ ]:


DG_1 = DG[1]
dofmap_restriction_DG_1 = dofmap_restriction_DG[cell_restriction_all][1]
assert count_dofs(dofmap_restriction_DG_1, mesh.comm) == 18
assert len(locate_dofs_by_polar_coordinates(
    0, 0, DG_1, dofmap_restriction_DG_1)) == 6
assert len(locate_dofs_by_polar_coordinates(
    1, 0, DG_1, dofmap_restriction_DG_1)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, DG_1, dofmap_restriction_DG_1)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, DG_1, dofmap_restriction_DG_1)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, DG_1, dofmap_restriction_DG_1)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, DG_1, dofmap_restriction_DG_1)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, DG_1, dofmap_restriction_DG_1)) == 2


# In[ ]:


DG_2 = DG[2]
dofmap_restriction_DG_2 = dofmap_restriction_DG[cell_restriction_all][2]
assert count_dofs(dofmap_restriction_DG_2, mesh.comm) == 36
assert len(locate_dofs_by_polar_coordinates(
    0, 0, DG_2, dofmap_restriction_DG_2)) == 6
assert len(locate_dofs_by_polar_coordinates(
    0.5, 0, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, 4 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 0, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 3 * np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 5 * np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 7 * np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 9 * np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 11 * np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1


# Compare DOFs for che case of cell restriction equal to one on a subset of the domain. Note how the DofMapRestriction has only a subset of the DOFs of the DofMap, and properly renumbers them. Note also that the number of DOFs at the same physical location might be different between DofMap and DofMapRestriction (see e.g. the center of the hexagon).

# In[ ]:


if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for (k, DGk) in enumerate(DG):
        plot_mesh(mesh, ax[k, 0])
        plot_dofmap(DGk, ax[k, 0])
        ax[k, 0].set_title("DG " + str(k) + " DofMap", fontsize=30)
    for (k, (DGk, dofmap_restriction_DGk)) in enumerate(zip(DG, dofmap_restriction_DG[cell_restriction_subset])):
        plot_mesh_tags(mesh, cell_restriction_subset, ax[k, 1])
        plot_dofmap_restriction(DGk, dofmap_restriction_DGk, ax[k, 1])
        ax[k, 1].set_title("DG " + str(k) + " DofMapRestriction", fontsize=30)


# Assert that DOFs are at the expected locations

# In[ ]:


DG_0 = DG[0]
dofmap_restriction_DG_0 = dofmap_restriction_DG[cell_restriction_subset][0]
assert count_dofs(dofmap_restriction_DG_0, mesh.comm) == 3
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 5 * np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 7 * np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 3, 9 * np.pi / 6, DG_0, dofmap_restriction_DG_0)) == 1


# In[ ]:


DG_1 = DG[1]
dofmap_restriction_DG_1 = dofmap_restriction_DG[cell_restriction_subset][1]
assert count_dofs(dofmap_restriction_DG_1, mesh.comm) == 9
assert len(locate_dofs_by_polar_coordinates(
    0, 0, DG_1, dofmap_restriction_DG_1)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, DG_1, dofmap_restriction_DG_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, DG_1, dofmap_restriction_DG_1)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, DG_1, dofmap_restriction_DG_1)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, DG_1, dofmap_restriction_DG_1)) == 1


# In[ ]:


DG_2 = DG[2]
dofmap_restriction_DG_2 = dofmap_restriction_DG[cell_restriction_subset][2]
assert count_dofs(dofmap_restriction_DG_2, mesh.comm) == 18
assert len(locate_dofs_by_polar_coordinates(
    0, 0, DG_2, dofmap_restriction_DG_2)) == 3
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, 4 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 5 * np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 7 * np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 9 * np.pi / 6, DG_2, dofmap_restriction_DG_2)) == 1


# Compare DOFs for che case of facet restriction equal to one on the entire domain. There is indeed no difference.

# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for (k, DGTk) in enumerate(DGT):
        plot_mesh(mesh, ax[k, 0])
        plot_dofmap(DGTk, ax[k, 0])
        ax[k, 0].set_title("DGT " + str(k) + " DofMap\n", fontsize=30)
    for (k, (DGTk, dofmap_restriction_DGTk)) in enumerate(zip(DGT, dofmap_restriction_DGT[facet_restriction_all])):
        plot_mesh_tags(mesh, facet_restriction_all, ax[k, 1])
        plot_dofmap_restriction(DGTk, dofmap_restriction_DGTk, ax[k, 1])
        ax[k, 1].set_title("DGT " + str(k) + " DofMapRestriction\n", fontsize=30)


# Assert that DOFs are at the expected locations

# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
DGT_0 = DGT[0]
dofmap_restriction_DGT_0 = dofmap_restriction_DGT[facet_restriction_all][0]
assert count_dofs(dofmap_restriction_DGT_0, mesh.comm) == 12
assert len(locate_dofs_by_polar_coordinates(
    0.5, 0, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi / 3, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 4 * np.pi / 3, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, np.pi / 6, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 3 * np.pi / 6, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 5 * np.pi / 6, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 7 * np.pi / 6, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 9 * np.pi / 6, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 11 * np.pi / 6, DGT_0, dofmap_restriction_DGT_0)) == 1


# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
DGT_1 = DGT[1]
dofmap_restriction_DGT_1 = dofmap_restriction_DGT[facet_restriction_all][1]
assert count_dofs(dofmap_restriction_DGT_1, mesh.comm) == 24
assert len(locate_dofs_by_polar_coordinates(
    0, 0, DGT_1, dofmap_restriction_DGT_1)) == 6
assert len(locate_dofs_by_polar_coordinates(
    1, 0, DGT_1, dofmap_restriction_DGT_1)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, DGT_1, dofmap_restriction_DGT_1)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, DGT_1, dofmap_restriction_DGT_1)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, DGT_1, dofmap_restriction_DGT_1)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, DGT_1, dofmap_restriction_DGT_1)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, DGT_1, dofmap_restriction_DGT_1)) == 3


# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
DGT_2 = DGT[2]
dofmap_restriction_DGT_2 = dofmap_restriction_DGT[facet_restriction_all][2]
assert count_dofs(dofmap_restriction_DGT_2, mesh.comm) == 36
assert len(locate_dofs_by_polar_coordinates(
    0, 0, DGT_2, dofmap_restriction_DGT_2)) == 6
assert len(locate_dofs_by_polar_coordinates(
    0.5, 0, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, np.pi, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 4 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 0, DGT_2, dofmap_restriction_DGT_2)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, np.pi, DGT_2, dofmap_restriction_DGT_2)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, 4 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 3
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 3
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, np.pi / 6, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 3 * np.pi / 6, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 5 * np.pi / 6, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 7 * np.pi / 6, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 9 * np.pi / 6, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    np.sqrt(3) / 2, 11 * np.pi / 6, DGT_2, dofmap_restriction_DGT_2)) == 1


# Compare DOFs for che case of facet restriction equal to one on a subset of the domain. Note how the DofMapRestriction has only a subset of the DOFs of the DofMap, and properly renumbers them. Note also that the number of DOFs at the same physical location might be different between DofMap and DofMapRestriction (see e.g. the center of the hexagon).

# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
if "PYTEST_CURRENT_TEST" not in os.environ:
    _, ax = plt.subplots(3, 2, figsize=(20, 30))
    for (k, DGTk) in enumerate(DGT):
        plot_mesh(mesh, ax[k, 0])
        plot_dofmap(DGTk, ax[k, 0])
        ax[k, 0].set_title("DGT " + str(k) + " DofMap\n", fontsize=30)
    for (k, (DGTk, dofmap_restriction_DGTk)) in enumerate(zip(DGT, dofmap_restriction_DGT[facet_restriction_subset])):
        plot_mesh_tags(mesh, facet_restriction_subset, ax[k, 1])
        plot_dofmap_restriction(DGTk, dofmap_restriction_DGTk, ax[k, 1])
        ax[k, 1].set_title("DGT " + str(k) + " DofMapRestriction\n", fontsize=30)


# Assert that DOFs are at the expected locations

# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
DGT_0 = DGT[0]
dofmap_restriction_DGT_0 = dofmap_restriction_DGT[facet_restriction_subset][0]
assert count_dofs(dofmap_restriction_DGT_0, mesh.comm) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, DGT_0, dofmap_restriction_DGT_0)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, DGT_0, dofmap_restriction_DGT_0)) == 1


# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
DGT_1 = DGT[1]
dofmap_restriction_DGT_1 = dofmap_restriction_DGT[facet_restriction_subset][1]
assert count_dofs(dofmap_restriction_DGT_1, mesh.comm) == 4
assert len(locate_dofs_by_polar_coordinates(
    0, 0, DGT_1, dofmap_restriction_DGT_1)) == 2
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, DGT_1, dofmap_restriction_DGT_1)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, DGT_1, dofmap_restriction_DGT_1)) == 1


# In[ ]:


# PYTEST_XFAIL: Temporarily broken due to lack of suport for DGT in basix
DGT_2 = DGT[2]
dofmap_restriction_DGT_2 = dofmap_restriction_DGT[facet_restriction_subset][2]
assert count_dofs(dofmap_restriction_DGT_2, mesh.comm) == 6
assert len(locate_dofs_by_polar_coordinates(
    0, 0, DGT_2, dofmap_restriction_DGT_2)) == 2
assert len(locate_dofs_by_polar_coordinates(
    0.5, 2 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    0.5, 5 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 2 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 1
assert len(locate_dofs_by_polar_coordinates(
    1, 5 * np.pi / 3, DGT_2, dofmap_restriction_DGT_2)) == 1

