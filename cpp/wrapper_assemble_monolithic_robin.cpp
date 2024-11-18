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
        dolfinx::fem::FunctionSpace<double>& V,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> facets_mesh_i,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells_mesh_i,
        nb::ndarray<const double, nb::c_contig> gpoints_cell,
        nb::ndarray<const double, nb::ndim<1>, nb::c_contig> gweights_cell,
        const size_t num_gp_per_facet)
      {
        auto set_fn = dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES);
        const std::size_t num_points = gpoints_cell.ndim() == 1 ? 1 : gpoints_cell.shape(0);
        const std::size_t gdim = V.mesh()->geometry().dim();
        std::span<const double> _gp(gpoints_cell.data(), gdim * num_points);
        assemble_monolithic_robin<double>(set_fn, V,
            std::span(facets_mesh_i.data(), facets_mesh_i.size()),
            std::span(cells_mesh_i.data(), cells_mesh_i.size()),
            _gp,
            std::span(gweights_cell.data(), gweights_cell.size()),
            num_gp_per_facet
            );
      });
      //nb::arg("A"), nb::arg("V"), "Hello");
}
