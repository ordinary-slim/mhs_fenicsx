#include "diffmesh_utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <array.h>

namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_diffmesh_utils(nb::module_ &m) {
  m.def("scatter_cell_integration_data_po",
      [](const dolfinx::geometry::PointOwnershipData<T> &po,
         const dolfinx::fem::FunctionSpace<double>& V,
         const multiphenicsx::fem::DofMapRestriction& restriction)
      {
        auto [num_diff_points_rcv, _cell_indices, _gdofs_cells, _geom_cells] = scatter_cell_integration_data_po(po, V, restriction);
        auto con_v = restriction.dofmap()->map();
        const size_t num_dofs_cell = con_v.extent(1);
        nb::ndarray<const std::int64_t, nb::ndim<2>, nb::numpy> gdofs_cells(
            _gdofs_cells.data(), {num_diff_points_rcv, num_dofs_cell});

        const std::size_t num_dofs_g = V.mesh()->geometry().cmap().dim();
        nb::ndarray<const T, nb::ndim<3>, nb::numpy> geom_cells(
            _geom_cells.data(), {num_diff_points_rcv, num_dofs_g, 3});
        auto cell_indices = dolfinx_wrappers::as_nbarray(std::move(_cell_indices));
        return std::tuple(cell_indices,
                          gdofs_cells.cast(),
                          geom_cells.cast());
      });
}

void declare_diffmesh_utils(nb::module_ &m) {
  templated_declare_diffmesh_utils<float>(m);
  templated_declare_diffmesh_utils<double>(m);
}
