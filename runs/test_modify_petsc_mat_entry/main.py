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
    '''
    domain = mesh.create_unit_square(comm, nelems_side, nelems_side,
                                     cell_type=mesh.CellType.quadrilateral,
                                     #diagonal=DiagonalType.right,
                                     )
    '''
    domain = mesh.create_unit_cube(comm, nelems_side, nelems_side, nelems_side,
                                   cell_type=mesh.CellType.hexahedron,
                                   )
    V  = fem.functionspace(domain, ("Lagrange", 1))

    (u, v) = (ufl.TrialFunction(V), ufl.TestFunction(V))
    a_ufl = ufl.dot(ufl.grad(u),ufl.grad(v))*ufl.dx
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
        assemble_monolithic_robin(A_bis, V._cpp_object)

    for mat in [A, A_bis]:
        mat.assemble()
    diff = (A-A_bis)
    norm_diff = diff.norm(2)
    print(f"Diff between my matrix and dolfinx assembled matrix = {norm_diff}")
    #print(diff.view())

if __name__=="__main__":
    lp = LineProfiler()
    lp_wrapper = lp(main)
    lp_wrapper()
    with open(f"profiling_rank{rank}.txt", 'w') as pf:
        lp.print_stats(stream=pf)
