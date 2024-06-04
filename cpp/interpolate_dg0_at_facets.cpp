#include <cstdio>
#include <cassert>
#include <dolfinx/fem/interpolate.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/geometry/utils.h>
#include <vector>
#include <dolfinx/la/Vector.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

using int_vector = std::vector<int>;

template <std::floating_point T>
void interpolate_dg0_at_facets(const dolfinx::fem::Function<T> &sending_f,
                               dolfinx::fem::Function<T> &receiving_f,
                               const dolfinx::fem::Function<T> &receiving_active_els_f,
                               const mesh::MeshTags<std::int32_t> &facet_tag,
                               std::span<const std::int32_t> incident_cells,
                               geometry::PointOwnershipData<T> &po,
                               const dolfinx::common::IndexMap &gamma_index_map,
                               std::map<std::int32_t,std::int32_t> gamma_im_to_global_imap)
{
  /*
   * Facets are local facets
   */
  std::shared_ptr smesh = sending_f.function_space()->mesh();//sending mesh
  std::shared_ptr rmesh = receiving_f.function_space()->mesh();//receiving mesh
  int cdim = rmesh->topology()->dim();
  const std::size_t value_size = sending_f.function_space()->value_size();

  la::Vector interpolated_vals_vec = la::Vector<T>(
      std::shared_ptr<const dolfinx::common::IndexMap>(&gamma_index_map, [](const dolfinx::common::IndexMap*){}),
      value_size);
  auto interpolated_vals_array = interpolated_vals_vec.mutable_array();

  auto& dest_ranks = po.src_owner;
  auto& src_ranks = po.dest_owners;
  auto& recv_points = po.dest_points;
  auto& evaluation_cells = po.dest_cells;
  // 1. Eval my points
  // Code copied from dolfinx interpolate.h
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

  // 2. Call dolfinx scatter
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

  // 3. Insert vals into receiving_f
  auto active_els_array = receiving_active_els_f.x()->array();
  auto con_cell_facet = rmesh->topology()->connectivity(cdim,cdim-1);
  assert(con_cell_facet);
  auto rfacet_map = rmesh->topology()->index_map(cdim-1);
  std::int32_t rnum_facets = rfacet_map->size_local();
  auto r_x = receiving_f.x()->mutable_array();

  std::span<const std::int32_t> facet_tag_vals = facet_tag.values();

  // Build facet indices (local and ghost)
  std::size_t n = std::count_if(facet_tag_vals.begin(), facet_tag_vals.end(), [](std::int32_t i){return i>0;});
  std::vector<std::int32_t> facet_indices;
  facet_indices.reserve(n);
  for (std::int32_t i = 0; i < facet_tag_vals.size(); ++i)
  {
    if (facet_tag_vals[i] > 0)
      facet_indices.push_back(i);//dirty
  }

  // TODO: Clean this up
  for (int i = 0; i < facet_indices.size(); ++i) {
    int ifacet = facet_indices[i];
    if (ifacet >= rnum_facets)
      continue;
    for (int j = 0; j < value_size; ++j) {
      interpolated_vals_array[i*value_size+j] = values_b[i*value_size+j];
    }
  }

  // Ghost communication
  interpolated_vals_vec.scatter_fwd();



  for (int i = 0; i < incident_cells.size(); ++i) {
    int icell = incident_cells[i];
    auto local_con = con_cell_facet->links(icell);
    auto it = std::find_if(local_con.begin(),local_con.end(),[&facet_tag_vals](std::int32_t ifacet){return (facet_tag_vals[ifacet]>0);});
    assert(it != std::end(local_con));
    std::int32_t ifacet = *it;
    std::int32_t ifacet_gamma_imap  = gamma_im_to_global_imap[ifacet];
    for (int j = 0; j < value_size; ++j) {
      r_x[icell*value_size+j] = interpolated_vals_array[ifacet_gamma_imap*value_size+j];
    }
  }
}

namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_interpolate_dg0_at_facets(nb::module_ &m) {
  m.def(
      "interpolate_dg0_at_facets",
      [](const dolfinx::fem::Function<T> &sending_f,
         dolfinx::fem::Function<T> &receiving_f,
         const dolfinx::fem::Function<T> &receiving_active_els_f,
         const mesh::MeshTags<std::int32_t> &facet_tag,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
         geometry::PointOwnershipData<T> &po,
         const dolfinx::common::IndexMap &gamma_index_map,
         std::map<std::int32_t,std::int32_t> gamma_im_to_global_imap)
      {
        return interpolate_dg0_at_facets<T>(sending_f,
                                            receiving_f,
                                            receiving_active_els_f,
                                            facet_tag,
                                            std::span(cells.data(),cells.size()),
                                            po,
                                            gamma_index_map,
                                            gamma_im_to_global_imap);
      }
      );
}

void declare_interpolate_dg0_at_facets(nb::module_ &m) {
  templated_declare_interpolate_dg0_at_facets<double>(m);
  templated_declare_interpolate_dg0_at_facets<float>(m);
}
