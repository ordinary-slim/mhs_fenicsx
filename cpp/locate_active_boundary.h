#pragma once
#include <cassert>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/la/Vector.h>
#include <vector>

using int_vector = std::vector<int>;

template <std::floating_point T>
int_vector locate_active_boundary(const dolfinx::mesh::Mesh<T> &domain,
                                  const dolfinx::fem::Function<T> &active_els_func) {
  int_vector bfacets_indices;
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
