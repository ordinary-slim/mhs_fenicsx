from dolfinx import fem, mesh
import dolfinx.fem.petsc as dfp
from mpi4py import MPI
import ufl
from dolfinx.utils import numba_utils as petsc_numba
import numpy as np
from petsc4py import PETSc
import numba

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
                                     cell_type=mesh.CellType.quadrilateral,
                                     )
    V  = fem.functionspace(domain, ("Lagrange", 1))

    (u, v) = (ufl.TrialFunction(V), ufl.TestFunction(V))
    a_ufl = u*v*ufl.dx
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
        from mhs_fenicsx_cpp import assemble_monolithic_robin
        assemble_monolithic_robin(A_bis, V._cpp_object)

    for mat in [A, A_bis]:
        mat.assemble()
    norm_diff = (A-A_bis).norm(2)
    print(f"Diff between my matrix and dolfinx assembled matrix = {norm_diff}")

if __name__=="__main__":
    main()

