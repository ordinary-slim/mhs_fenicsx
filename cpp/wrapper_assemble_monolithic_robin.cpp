#include "assemble_monolithic_robin.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/fem/petsc.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <caster_petsc.h>
#include <petsc4py/petsc4py.h>

namespace nb = nanobind;
void declare_assemble_monolithic_robin(nb::module_ &m) {
  import_petsc4py();
  m.def(
      "assemble_monolithic_robin",
      [](Mat A,
        dolfinx::fem::FunctionSpace<double>& V)
      {
        auto set_fn = dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES);
        assemble_monolithic_robin<double>(set_fn, V);
      });
      //nb::arg("A"), nb::arg("V"), "Hello");
}
