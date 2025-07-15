#include "diffmesh_utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <dolfinx_wrappers/array.h>

namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_diffmesh_utils(nb::module_ &m) {
  m.def("scatter_cell_integration_data_po",
      [](const dolfinx::geometry::PointOwnershipData<T> &po,
         const dolfinx::fem::FunctionSpace<double>& V,
         const multiphenicsx::fem::DofMapRestriction& restriction,
         const dolfinx::fem::Function<T>& mat_id_func)
      {
        auto [num_diff_points_rcv, _cell_indices, _mat_ids, _gdofs_cells, _geom_cells] = scatter_cell_integration_data_po(po, V, restriction, mat_id_func);
        auto con_v = restriction.dofmap()->map();
        const size_t num_dofs_cell = con_v.extent(1);
        nb::ndarray<const std::int64_t, nb::ndim<2>, nb::numpy> gdofs_cells(
            _gdofs_cells.data(), {num_diff_points_rcv, num_dofs_cell});

        const std::size_t num_dofs_g = V.mesh()->geometry().cmap().dim();
        nb::ndarray<const T, nb::ndim<3>, nb::numpy> geom_cells(
            _geom_cells.data(), {num_diff_points_rcv, num_dofs_g, 3});
        auto cell_indices = dolfinx_wrappers::as_nbarray(std::move(_cell_indices));
        auto mat_ids = dolfinx_wrappers::as_nbarray(std::move(_mat_ids));
        return std::tuple(cell_indices,
                          mat_ids,
                          gdofs_cells.cast(),
                          geom_cells.cast());
      });
  m.def(
      "find_owner_rank",
      [](nb::ndarray<const T, nb::c_contig> points,
         const dolfinx::geometry::BoundingBoxTree<T>& cell_bb_tree,
         const dolfinx::fem::Function<T>& active_els_func)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);
        std::vector<int> owner_rank = find_owner_rank<T>(_p,
                                                         cell_bb_tree,
                                                         active_els_func);
        return nb::ndarray<const int, nb::numpy>(owner_rank.data(),
            {owner_rank.size()}).cast();
      },
      nb::arg("points"),nb::arg("cell_bb_tree"),nb::arg("active_els_func"));
  m.def(
      "cellwise_determine_point_ownership",
      [](const dolfinx::mesh::Mesh<T>& mesh,
         nb::ndarray<const T, nb::c_contig> points,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
         T padding,
         bool extrapolate = true)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);
        return determine_point_ownership<T>(mesh,
                                            _p,
                                            std::span(cells.data(),cells.size()),
                                            padding,
                                            extrapolate);
      },
    nb::arg("mesh"), nb::arg("points"), nb::arg("cells"), nb::arg("padding"),
    nb::arg("extrapolate") = true);
  m.def(
      "determine_facet_points_ownership",
      [](const dolfinx::mesh::Mesh<T>& mesh,
         nb::ndarray<const T, nb::c_contig> points,
         size_t points_per_facet,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
         T padding,
         bool extrapolate = true)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);
        return determine_facet_points_ownership<T>(mesh,
                                                   _p,
                                                   points_per_facet,
                                                   std::span(cells.data(),cells.size()),
                                                   padding,
                                                   extrapolate);
      },
    nb::arg("mesh"), nb::arg("points"), nb::arg("points_per_facet"), nb::arg("cells"), nb::arg("padding"),
    nb::arg("extrapolate") = true);
}

void declare_diffmesh_utils(nb::module_ &m) {
  templated_declare_diffmesh_utils<float>(m);
  templated_declare_diffmesh_utils<double>(m);
}
