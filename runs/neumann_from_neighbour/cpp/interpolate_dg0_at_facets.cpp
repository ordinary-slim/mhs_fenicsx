#include <cstdio>
#include <cassert>
#include <dolfinx/fem/interpolate.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/geometry/utils.h>
#include <vector>

using int_vector = std::vector<int>;

template <std::floating_point T>
void interpolate_dg0_at_facets(const dolfinx::fem::Function<T> &sending_f,
                               dolfinx::fem::Function<T> &receiving_f,
                               int_vector &facets) {
  /*
   * Facets are local facets
   */
  std::shared_ptr smesh = sending_f.function_space()->mesh();//sending mesh
  std::shared_ptr rmesh = receiving_f.function_space()->mesh();//receiving mesh
  int cdim = rmesh->topology()->dim();
  // 1. Compute facet midpoints
  std::vector<T> midpoints = mesh::compute_midpoints<T>(*rmesh,cdim-1,facets);
  // 2. Determine point ownership
  geometry::PointOwnershipData po = geometry::determine_point_ownership<T>(
      *smesh,
      midpoints,
      T(0));
  auto& dest_ranks = po.src_owner;
  auto& src_ranks = po.dest_owners;
  auto& recv_points = po.dest_points;
  auto& evaluation_cells = po.dest_cells;
  // 3. Eval my points
  // Code copied from dolfinx interpolate.h
  const std::size_t value_size = sending_f.function_space()->value_size();
  // Evaluate the interpolating function where possible
  std::vector<T> send_values(recv_points.size() / 3 * value_size);
  /*
  sending_f.eval(recv_points, {recv_points.size() / 3, (std::size_t)3},
                 evaluation_cells, send_values, {recv_points.size() / 3, value_size});
  */
  auto s_x = sending_f.x()->array();
  for (int i = 0; i < evaluation_cells.size(); ++i) {
    for (int j = 0; j < value_size; ++j) {
      send_values[i*value_size+j] = s_x[evaluation_cells[i]*value_size+j];
    }
  }

  using dextents2 = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;

  // 4. Call dolfinx scatter
  // Send values back to owning process
  std::vector<T> values_b(dest_ranks.size() * value_size);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const T, dextents2> _send_values(
      send_values.data(), src_ranks.size(), value_size);
  fem::impl::scatter_values(rmesh->comm(), src_ranks, dest_ranks, _send_values,
                       std::span(values_b));
  // Transpose received data
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const T, dextents2> values(
      values_b.data(), dest_ranks.size(), value_size);
  std::vector<T> valuesT_b(value_size * dest_ranks.size());
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T, dextents2> valuesT(
      valuesT_b.data(), value_size, dest_ranks.size());
  for (std::size_t i = 0; i < values.extent(0); ++i)
    for (std::size_t j = 0; j < values.extent(1); ++j)
      valuesT(j, i) = values(i, j);

  // 5. Insert vals into receiving_f
  auto con_facet_cell = rmesh->topology()->connectivity(cdim-1,cdim);
  assert(con_facet_cell);
  auto facet_map = rmesh->topology()->index_map(cdim-1);
  auto r_x = receiving_f.x()->mutable_array();
  for (int i = 0; i < facets.size(); ++i) {
    int ifacet = facets[i];
    auto local_con = con_facet_cell->links(ifacet);
    assert(local_con.size() > 0);
    int incident_el = local_con[0];
    for (int j = 0; j < value_size; ++j) {
      r_x[incident_el*value_size+j] = values_b[i*value_size+j];
    }
  }
}

namespace nb = nanobind;
NB_MODULE(interpolate_dg0_at_facets, m) {
  nb::bind_vector<int_vector>(m, "int_vector");
  m.def("interpolate_dg0_at_facets", &interpolate_dg0_at_facets<double>);
  m.def("interpolate_dg0_at_facets", &interpolate_dg0_at_facets<float>);
}
