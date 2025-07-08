#include "tabulate_facet_qpoints.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
void declare_tabulate_facet_quadrature(nb::module_ &m) {
  m.def(
      "tabulate_facet_quadrature",
      [](
        const dolfinx::mesh::Mesh<U> &mesh,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> facet_integration_data,
        size_t num_gps_facet,
        nb::ndarray<const double, nb::ndim<2>, nb::c_contig> quadrature_points_cell
        )
      {
        assert(facet_integration_data.size() % 2 == 0);
        size_t num_local_facet_cells = facet_integration_data.size() / 2;
        std::vector<U> _tabulated_facet_qpoints =
          tabulate_facet_quadrature(
              mesh,
              std::span(facet_integration_data.data(), facet_integration_data.size()),
              num_gps_facet,
              std::span(quadrature_points_cell.data(), quadrature_points_cell.size())
              );
        nb::ndarray<const U, nb::ndim<2>, nb::numpy> tabulated_facet_qpoints(
              _tabulated_facet_qpoints.data(), {num_gps_facet*num_local_facet_cells, 3});
        return tabulated_facet_qpoints.cast();
      });
}
