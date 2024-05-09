#pragma once
#include <cassert>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
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
  std::int32_t num_facets = domain.topology()->index_map(cdim-1)->size_local();
  float num_incident_active_els;
  for (int ifacet = 0; ifacet < num_facets; ++ifacet) {
    num_incident_active_els = 0.0;
    auto local_con = con_facet_cell->links(ifacet);
    for (auto &el : local_con) {
      num_incident_active_els += func_vals[el];
    }
    if (std::abs(num_incident_active_els - 1.0) < 1e-7) {
      bfacets_indices.push_back(ifacet);
    }
  }
  return bfacets_indices;
}
