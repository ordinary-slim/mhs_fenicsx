#include "tabulate_gamma_qpoints.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
void declare_tabulate_gamma_quadrature(nb::module_ &m) {
  m.def(
      "tabulate_gamma_quadrature",
      [](
        const dolfinx::mesh::Mesh<U> &mesh,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> gamma_integration_data,
        size_t num_gps_facet,
        nb::ndarray<const double, nb::ndim<2>, nb::c_contig> quadrature_points_cell
        )
      {
        assert(gamma_integration_data.size() % 2 == 0);
        size_t num_local_gamma_cells = gamma_integration_data.size() / 2;
        std::vector<U> _tabulated_gamma_qpoints =
          tabulate_gamma_quadrature(
              mesh,
              std::span(gamma_integration_data.data(), gamma_integration_data.size()),
              num_gps_facet,
              std::span(quadrature_points_cell.data(), quadrature_points_cell.size())
              );
        nb::ndarray<const U, nb::ndim<2>, nb::numpy> tabulated_gamma_qpoints(
              _tabulated_gamma_qpoints.data(), {num_gps_facet*num_local_gamma_cells, 3});
        return tabulated_gamma_qpoints.cast();
      });
}
