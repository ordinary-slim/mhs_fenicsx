#include <cassert>
#include <cstdint>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/la/Vector.h>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

template <std::floating_point T>
std::vector<int> locate_active_boundary(const dolfinx::mesh::Mesh<T> &domain,
                                  const dolfinx::fem::Function<T> &active_els_func) {
  std::vector<int> bfacets_indices;
  // Values of function
  std::span<const T> func_vals = active_els_func.x()->array();
  int cdim = domain.topology()->dim();
  domain.topology_mutable()->create_connectivity(cdim-1,cdim);
  auto con_facet_cell = domain.topology()->connectivity(cdim-1,cdim);
  assert(con_facet_cell);
  auto facet_map = domain.topology()->index_map(cdim-1);
  std::int32_t num_facets = facet_map->size_local();
  la::Vector<int> bfacet_marker = la::Vector<int>(facet_map, 1);
  auto bfacet_marker_vals = bfacet_marker.mutable_array();
  float num_incident_active_els;
  int num_bfacets = 0;
  for (int ifacet = 0; ifacet < num_facets; ++ifacet) {
    num_incident_active_els = 0.0;
    auto local_con = con_facet_cell->links(ifacet);
    for (auto &el : local_con) {
      num_incident_active_els += func_vals[el];
    }
    if (std::abs(num_incident_active_els - 1.0) < 1e-7) {
      bfacet_marker_vals[ifacet] = 1;
      ++num_bfacets;
    }
  }

  bfacet_marker.scatter_fwd();

  bfacets_indices.reserve(num_bfacets);

  for (int i = 0; i < bfacet_marker_vals.size(); ++i) {
    if (bfacet_marker_vals[i]==1)
      bfacets_indices.push_back(i);
  }
  return bfacets_indices;
}

template <std::floating_point T>
std::vector<int> deactivate_from_nodes(const dolfinx::mesh::Mesh<T> &domain,
                                       const std::span<const T> nodes_to_subtract) {
  // 1. Make vector
  auto topology = domain.topology();
  int cdim = topology->dim();
  auto cell_map = domain.topology()->index_map(cdim);
  la::Vector active_els_mask = la::Vector<T>(cell_map, 1);
  active_els_mask.set(1);// Start out as all active
  auto active_els_mask_vals = active_els_mask.mutable_array();
  // 2. Loop over els
  auto con_cell_nodes = topology->connectivity(cdim,0);
  assert(con_cell_nodes);
  for (int icell = 0; icell < cell_map->size_local(); ++icell) {
    auto incident_nodes = con_cell_nodes->links(icell);
    bool all_active_in_ext = std::all_of(incident_nodes.begin(),
                                         incident_nodes.end(),
                                         [&nodes_to_subtract](auto inode){
                                           return (nodes_to_subtract[inode]==1.0);
                                         });
    if (all_active_in_ext)
      active_els_mask_vals[icell] = 0;
  }
  // 3. Scatter
  active_els_mask.scatter_fwd();
  // 4. Extract indices
  std::vector<int> active_els;
  for (int icell = 0; icell < cell_map->size_local(); ++icell) {
    if (active_els_mask_vals[icell]==1.0)
      active_els.push_back(icell);
  }
  return active_els;
}


namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_activation_utils(nb::module_ &m) {
  m.def("locate_active_boundary", &locate_active_boundary<T>);
  m.def(
      "deactivate_from_nodes",
      [](const dolfinx::mesh::Mesh<T> &domain,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> nodes_to_subtract)
      {
        return deactivate_from_nodes<T>(domain,
                                        std::span(nodes_to_subtract.data(),nodes_to_subtract.size()));
      }
      );
}

void declare_activation_utils(nb::module_ &m) {
  templated_declare_activation_utils<float>(m);
  templated_declare_activation_utils<double>(m);
}
