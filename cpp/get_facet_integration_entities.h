#pragma once
#include <cassert>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <vector>

template <std::floating_point T>
std::vector<int32_t> get_facet_integration_entities(const dolfinx::mesh::Mesh<T> &domain,
    std::span<const std::int32_t> facets,
    const dolfinx::fem::Function<T> &active_els_func) {

  std::vector<int32_t> integration_entities;

  std::span<const T> active_els_vals = active_els_func.x()->array();
  int cdim = domain.topology()->dim();
  auto con_facet_cell = domain.topology()->connectivity(cdim-1,cdim);
  auto con_cell_facet = domain.topology()->connectivity(cdim,cdim-1);
  std::int32_t num_local_facets = domain.topology()->index_map(cdim-1)->size_local();

  for (const int &ifacet : facets) {
    if (ifacet >= num_local_facets) {
      continue;
    }
    auto incident_cells = con_facet_cell->links(ifacet);
    int owner_el = -1;
    for (auto &ielem : incident_cells) {
      if (std::abs(active_els_vals[ielem] - 1) < 1e-7) {
        owner_el = ielem;
        break;
      }
    }
    assert(owner_el>-1);

    //Get local index of facet
    auto incident_facets = con_cell_facet->links(owner_el);
    int local_index = -1;
    for (int i = 0; i < incident_facets.size(); ++i) {
      if (ifacet==incident_facets[i]) {
        local_index = i;
        break;
      }
    }
    assert(local_index>-1);
    integration_entities.push_back(owner_el);
    integration_entities.push_back(local_index);
  }

  return integration_entities;
}
