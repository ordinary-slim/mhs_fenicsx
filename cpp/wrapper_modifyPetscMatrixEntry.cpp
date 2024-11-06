#include "modifyPetscMatrixEntry.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/fem/petsc.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <caster_petsc.h>
#include <petsc4py/petsc4py.h>

namespace nb = nanobind;
template <dolfinx::scalar T>
void templated_declare_modifyPetscMatrixEntry(nb::module_ &m) {
  import_petsc4py();
  m.def(
      "modifyPetscMatrixEntry",
      [](Mat A)
      {
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>,
                          std::span<const PetscScalar>)>
        set_fn = dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES);
        modifyPetscMatrixEntry<T>(set_fn);
      },
      nb::arg("A"), "Hello");
}

void declare_modifyPetscMatrixEntry(nb::module_ &m) {
  templated_declare_modifyPetscMatrixEntry<double>(m);
  templated_declare_modifyPetscMatrixEntry<float>(m);
}
