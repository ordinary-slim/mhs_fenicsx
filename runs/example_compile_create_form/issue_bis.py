from dolfinx import fem, mesh
import ufl
from mpi4py import MPI
import numpy as np

def main():
    els_side = 4
    domain  = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side)
    V   = fem.functionspace(domain, ("Lagrange", 1),)
    solution = fem.Function(V, name="sol")
    conductivity = fem.Constant(domain, 1.0)
    v = ufl.TestFunction(V)
    dx = ufl.dx
    a_ufl = conductivity * solution * v * dx
    form = ufl.derivative(a_ufl, solution)
    j_compiled = fem.compile_form(domain.comm, form)

    coefficient_map = {}
    for coeff in form.coefficients():
        coefficient_map[coeff] = coeff
    constant_map = {}
    for const in form.constants():
        constant_map[const] = const
    j_instance = fem.create_form(j_compiled,
                                 [V, V],
                                 msh=domain,
                                 subdomains={},
                                 coefficient_map = coefficient_map,
                                 constant_map = constant_map)
    A = fem.assemble_matrix(j_instance)
    print(A.data)

if __name__=="__main__":
    main()
