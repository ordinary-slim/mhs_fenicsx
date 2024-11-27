from dolfinx import fem, mesh
import dolfinx.fem.petsc as dfp
from mpi4py import MPI
import ufl
from dolfinx.utils import numba_utils as petsc_numba
import numpy as np
from petsc4py import PETSc
import numba
from line_profiler import LineProfiler
from mhs_fenicsx_cpp import assemble_monolithic_robin
from dolfinx.cpp.mesh import DiagonalType, create_geometry, create_topology
import basix.ufl
from utils import generate_facet_cell_quadrature
from utils import ref_tria_mesh, ref_hexa_mesh, ref_tetra_mesh, ref_quadri_mesh
from utils import ref_tria_mesh_gmsh, ref_quadri_mesh_gmsh, ref_tetra_mesh_gmsh, ref_hexa_mesh_gmsh
from lpbf_mesh import get_mesh

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

MatSetValuesLocal = petsc_numba.MatSetValuesLocal

@numba.njit
def set_vals_numba(A, rows, cols, data, mode):
    MatSetValuesLocal(A, len(rows), rows.ctypes, len(cols), cols.ctypes, data.ctypes, mode)

@numba.njit(fastmath=True)
def modify_entries_petsc_mat(A, A_local, rows, cols,mode):
    set_vals_numba(A,rows,cols,A_local,mode)

numba_modif = False

def main():
    nelems_side = 2
    domain = mesh.create_unit_square(comm, nelems_side, nelems_side,
                                     #cell_type=mesh.CellType.quadrilateral,
                                     #diagonal=DiagonalType.right,
                                     )
    #domain = mesh.create_unit_cube(comm, nelems_side, nelems_side, nelems_side,
                                   ##cell_type=mesh.CellType.hexahedron,
                                   ##diagonal=DiagonalType.right,
                                   #)
    '''
    import yaml
    with open("lpbf.yaml", 'r') as f:
        params = yaml.safe_load(f)
    domain = get_mesh(params)
    '''

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_cells  = np.zeros_like(boundary_facets,dtype=np.int32)
    con_facet_cell = domain.topology.connectivity(fdim,tdim)
    for idx, ifacet in enumerate(boundary_facets):
        local_con = con_facet_cell.links(ifacet)
        assert len(local_con==1)
        boundary_cells[idx] = local_con[0]
    '''
    domain = mesh.create_unit_cube(comm, nelems_side, nelems_side, nelems_side,
                                   cell_type=mesh.CellType.hexahedron,
                                   )
    '''
    V  = fem.functionspace(domain, ("Lagrange", 1))
    Qe = basix.ufl.quadrature_element(domain.topology.entity_types[-2][0].name,
                                      degree=2)
    gamma_mesh = mesh.create_submesh(domain,fdim,boundary_facets)[0]
    Qs_gamma = fem.functionspace(gamma_mesh, Qe)

    (u, v) = (ufl.TrialFunction(V), ufl.TestFunction(V))
    a_ufl = u*v*ufl.ds
    a_cpd = fem.form(a_ufl)
    A = dfp.create_matrix(a_cpd)
    A_bis = dfp.create_matrix(a_cpd)
    dfp.assemble_matrix(A, a_cpd)

    if numba_modif:
        A_local = np.array([[33]], dtype=PETSc.ScalarType)
        rows = np.array([0], dtype=np.int32)
        cols = np.array([0], dtype=np.int32)
        modify_entries_petsc_mat(A.handle,A_local,rows,cols,PETSc.InsertMode.ADD_VALUES)
    else:
        gp_cell, gweigths_cell, num_gpoints_facet = generate_facet_cell_quadrature(domain)
        assemble_monolithic_robin(A_bis, Qs_gamma._cpp_object, V._cpp_object,
                                  boundary_facets,
                                  boundary_cells,
                                  gp_cell,
                                  gweigths_cell,
                                  num_gpoints_facet,
                                  )

    for mat in [A, A_bis]:
        mat.assemble()
    diff = (A-A_bis)
    norm_diff = diff.norm(2)
    print(f"=================================================================")
    print(f"Diff between my matrix and dolfinx assembled matrix = {norm_diff}")
    print(f"=================================================================")
    #print(diff.view())

if __name__=="__main__":
    lp = LineProfiler()
    lp_wrapper = lp(main)
    lp_wrapper()
    with open(f"profiling_rank{rank}.txt", 'w') as pf:
        lp.print_stats(stream=pf)
