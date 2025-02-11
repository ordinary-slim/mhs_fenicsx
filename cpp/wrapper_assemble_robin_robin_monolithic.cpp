#include "assemble_robin_robin_monolithic.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/fem/petsc.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <caster_petsc.h>
#include <caster_mpi.h>
#include <petsc4py/petsc4py.h>

namespace nb = nanobind;
void declare_assemble_robin_robin_monolithic(nb::module_ &m) {
  using T = double;
  import_petsc4py();
  std::string pyclass_name = "MonolithicRobinRobinAssembler64";
  nb::class_<MonolithicRobinRobinAssembler<T>>(m, pyclass_name.c_str())
    .def(
        "__init__",
        [](MonolithicRobinRobinAssembler<T>* self)
        {
          new (self) MonolithicRobinRobinAssembler<T>();
        })
    .def(
       "preassemble",
       [](MonolithicRobinRobinAssembler<T>& self,
          nb::ndarray<const double, nb::ndim<2>, nb::c_contig> tabulated_gauss_points_gamma,
          nb::ndarray<const double, nb::ndim<2>, nb::c_contig> gauss_points_cell,
          nb::ndarray<const double, nb::ndim<1>, nb::c_contig> gweights_facet,
          dolfinx::fem::FunctionSpace<double>& V_i,
          multiphenicsx::fem::DofMapRestriction& restriction_i,
          dolfinx::fem::FunctionSpace<double>& V_j,
          multiphenicsx::fem::DofMapRestriction& restriction_j,
          nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> gamma_integration_data_i,
          nb::ndarray<const int, nb::ndim<1>, nb::c_contig> renumbering_cells_po_mesh_j,
          nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> dofs_cells_mesh_j,
          nb::ndarray<const double, nb::ndim<3>, nb::c_contig> geoms_cells_mesh_j)
       {
         const std::size_t num_points = gweights_facet.ndim() == 1 ? 1 : gweights_facet.shape(0);
         const std::size_t gdim = V_i.mesh()->geometry().dim();
         std::span<const double> _gp(gweights_facet.data(), gdim * num_points);
         Mat A = NULL;
         self.preassemble(
             A,
             std::span(tabulated_gauss_points_gamma.data(), tabulated_gauss_points_gamma.size()),
             std::span(gauss_points_cell.data(), gauss_points_cell.size()),
             std::span(gweights_facet.data(), gweights_facet.size()),
             V_i, restriction_i,
             V_j, restriction_j,
             std::span(gamma_integration_data_i.data(), gamma_integration_data_i.size()),
             std::span(renumbering_cells_po_mesh_j.data(), renumbering_cells_po_mesh_j.size()),
             std::span(dofs_cells_mesh_j.data(), dofs_cells_mesh_j.size()),
             std::span(geoms_cells_mesh_j.data(), geoms_cells_mesh_j.size())
             );
         return A;
       },
       nb::rv_policy::take_ownership)
    .def(
       "assemble",
       [](MonolithicRobinRobinAssembler<T>& self,
          Mat A,
          const dolfinx::fem::Function<T> &ext_conductivity,
          nb::ndarray<const double, nb::ndim<2>, nb::c_contig> tabulated_gauss_points_gamma,
          nb::ndarray<const double, nb::ndim<2>, nb::c_contig> gauss_points_cell,
          nb::ndarray<const double, nb::ndim<1>, nb::c_contig> gweights_facet,
          dolfinx::fem::FunctionSpace<double>& V_i,
          multiphenicsx::fem::DofMapRestriction& restriction_i,
          dolfinx::fem::FunctionSpace<double>& V_j,
          multiphenicsx::fem::DofMapRestriction& restriction_j,
          nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> gamma_integration_data_i,
          nb::ndarray<const int, nb::ndim<1>, nb::c_contig> renumbering_cells_po_mesh_j,
          nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> dofs_cells_mesh_j)
       {
         const std::size_t num_points = gweights_facet.ndim() == 1 ? 1 : gweights_facet.shape(0);
         const std::size_t gdim = V_i.mesh()->geometry().dim();
         std::span<const double> _gp(gweights_facet.data(), gdim * num_points);
         self.assemble(
             set_fn(A, ADD_VALUES),
             ext_conductivity,
             std::span(tabulated_gauss_points_gamma.data(), tabulated_gauss_points_gamma.size()),
             std::span(gauss_points_cell.data(), gauss_points_cell.size()),
             std::span(gweights_facet.data(), gweights_facet.size()),
             V_i, restriction_i,
             V_j, restriction_j,
             std::span(gamma_integration_data_i.data(), gamma_integration_data_i.size()),
             std::span(renumbering_cells_po_mesh_j.data(), renumbering_cells_po_mesh_j.size()),
             std::span(dofs_cells_mesh_j.data(), dofs_cells_mesh_j.size())
             );
       });
}
