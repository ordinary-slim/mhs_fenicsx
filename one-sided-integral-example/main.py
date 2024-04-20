# Computing a onesided integral over an interior facet with consistent orientation
# Enabled by: https://github.com/FEniCS/dolfinx/pull/2269
# Copyright 2023 Jørgen S. Dokken
# SPDX: MIT

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
import pdb

def writepos(mesh):
    with dolfinx.io.VTKFile(mesh.comm, f"out/mesh.pvd", "w") as ofile:
        ofile.write_mesh(mesh)

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 10, 10, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

# Create connectivties required for defining integration entities
tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_entities(fdim)
mesh.topology.create_connectivity(fdim, tdim)
mesh.topology.create_connectivity(tdim, fdim)


# Get number of cells on process
cell_map = mesh.topology.index_map(tdim)
num_cells = cell_map.size_local + cell_map.num_ghosts

# Create markers for each size of the interface
cell_values = np.ones(num_cells, dtype=np.int32)
cell_values[dolfinx.mesh.locate_entities(
    mesh, tdim, lambda x: x[0] <= 0.5+1e-13)] = 2
ct = dolfinx.mesh.meshtags(mesh, tdim, np.arange(
    num_cells, dtype=np.int32), cell_values)

facet_map = mesh.topology.index_map(fdim)
num_facets = facet_map.size_local + facet_map.num_ghosts

# Create facet markers
facet_values = np.ones(num_facets, dtype=np.int32)
facet_values[dolfinx.mesh.locate_entities(
    mesh, fdim, lambda x: np.isclose(x[0], 0.5))] = 2
ft = dolfinx.mesh.meshtags(mesh, fdim, np.arange(
    num_facets, dtype=np.int32), facet_values)

# Give a set of facets marked with a value (in this case 2), get a consistent orientation for an interior integral
facets_to_integrate = ft.find(2)

f_to_c = mesh.topology.connectivity(fdim, tdim)
c_to_f = mesh.topology.connectivity(tdim, fdim)
# Compute integration entities for a single facet of a cell.
# Each facet is represented as a tuple (cell_index, local_facet_index), where cell_index is local to process
# local_facet_index is the local indexing of a facet for a given cell
integration_entities = []
for i, facet in enumerate(facets_to_integrate):
    # Only loop over facets owned by the process to avoid duplicate integration
    if facet >= facet_map.size_local:
        continue
    # Find cells connected to facet
    cells = f_to_c.links(facet)
    # Get value of cells
    marked_cells = ct.values[cells]
    # Get the cell marked with 2
    correct_cell = np.flatnonzero(marked_cells == 2)

    assert len(correct_cell) == 1
    # Get local index of facet
    local_facets = c_to_f.links(cells[correct_cell[0]])
    local_index = np.flatnonzero(local_facets == facet)
    assert len(local_index) == 1

    # Append integration entities
    integration_entities.append(cells[correct_cell[0]])
    integration_entities.append(local_index[0])
pdb.set_trace()

# Create custom integration measure for one-sided integrals
ds = ufl.Measure("ds", domain=mesh, subdomain_data=[
                 (8, np.asarray(integration_entities, dtype=np.int32))])
n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)
# Exact integral is [y/2**2]_0^1= 1/2
L = ufl.dot(ufl.as_vector((x[1], 0)), n)*ds(8)
L_compiled = dolfinx.fem.form(L)

print(
    f"Correct integral: {mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L_compiled), op=MPI.SUM)}")


# Create reference implementation where we use a restricted two-sided integral with no notion of orientation
dS = ufl.Measure("dS", domain=mesh, subdomain_data=ft, subdomain_id=2)
n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)
L2 = ufl.dot(ufl.as_vector((x[1], 0)), n("+"))*dS
L2_compiled = dolfinx.fem.form(L2)
print(
    f"Wrong integral: {mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_compiled), op=MPI.SUM)}")

writepos(mesh)
