#pragma once
#include <cassert>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <vector>

/*
 * facets must be both local and ghosted indices of ents of interest
 */
template <std::floating_point T>
std::vector<int32_t> get_facet_integration_entities(const dolfinx::mesh::Mesh<T> &domain,
    std::span<const std::int32_t> facets,
    const dolfinx::fem::Function<T> &active_els_func,
    bool use_inactive = false,
    bool use_ghosted = false)
{


  std::span<const T> active_els_vals = active_els_func.x()->array();
  int cdim = domain.topology()->dim();
  auto con_facet_cell = domain.topology()->connectivity(cdim-1,cdim);
  auto con_cell_facet = domain.topology()->connectivity(cdim,cdim-1);
  size_t num_local_cells = domain.topology()->index_map(cdim)->size_local();

  std::vector<int32_t> integration_entities;
  for (const std::int32_t &ifacet : facets) {
    // Find local active cell that owns the facet
    auto incident_cells = con_facet_cell->links(ifacet);
    std::int32_t owner_cell = -1;
    for (auto &ielem : incident_cells) {
      // Skip inactive or ghosted elements
      if ((not(use_inactive) && active_els_vals[ielem] < 1.0) || (not(use_ghosted) && ielem >= num_local_cells))
        continue;
      owner_cell = ielem;
      break;
    }
    if (owner_cell<=-1)
      continue;

    //Get local index of facet
    auto incident_facets = con_cell_facet->links(owner_cell);
    auto it = std::find(incident_facets.begin(), incident_facets.end(), ifacet);
    assert(it != incident_facets.end());
    const std::int32_t local_index = std::distance(incident_facets.begin(), it);

    integration_entities.push_back(owner_cell);
    integration_entities.push_back(local_index);
  }

  return integration_entities;
}
