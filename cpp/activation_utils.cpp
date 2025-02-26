#include <cassert>
#include <cstdint>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/la/Vector.h>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <dolfinx/mesh/MeshTags.h>
#include <numeric>

template <std::floating_point T>
std::vector<int> locate_active_boundary(const dolfinx::mesh::Mesh<T> &domain,
                                  const dolfinx::fem::Function<T> &active_els_func) {
  std::vector<int> bfacets_indices;
  // Values of function
  std::span<const T> func_vals = active_els_func.x()->array();
  auto topology = domain.topology();
  int cdim = topology->dim();
  domain.topology_mutable()->create_connectivity(cdim-1,cdim);
  auto con_facet_cell = topology->connectivity(cdim-1,cdim);
  assert(con_facet_cell);
  auto facet_map = topology->index_map(cdim-1);
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
std::tuple<la::Vector<std::int8_t>,
  la::Vector<std::int8_t>> node_mask_to_el_mask(const dolfinx::mesh::Mesh<T> &domain,
                                      const std::span<const bool> node_mask) {
  // 1. Make vector
  auto topology = domain.topology();
  auto dofmap_x = domain.geometry().dofmap();
  auto cell_map = topology->index_map(topology->dim());
  const std::int32_t num_local_cells = cell_map->size_local();
  la::Vector<std::int8_t> els_mask(cell_map, 1), colliding_els_mask(cell_map, 1);
  els_mask.set(0);
  colliding_els_mask.set(0);
  auto els_mask_vals = els_mask.mutable_array();
  auto colliding_els_mask_vals = colliding_els_mask.mutable_array();
  // 2. Loop over els
  for (int icell = 0; icell < num_local_cells; ++icell) {
    auto xdofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dofmap_x, icell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    bool all_el_nodes_active = true;
    bool one_el_nodes_active = false;
    for (size_t i = 0; i < dofmap_x.extent(1); ++i) {
      bool is_node_inactive = (node_mask[xdofs[i]]!=1);
      one_el_nodes_active = (one_el_nodes_active || not(is_node_inactive));
      if (is_node_inactive) {
        all_el_nodes_active = false;
        break;
      }
    }
    if (all_el_nodes_active)
      els_mask_vals[icell] = 1;
    if (one_el_nodes_active)
      colliding_els_mask_vals[icell] = 1;
  }
  // 3. Scatter
  els_mask.scatter_fwd();
  colliding_els_mask.scatter_fwd();
  return std::tuple(els_mask, colliding_els_mask);
}

template <std::floating_point T>
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>>
deactivate_from_nodes(const dolfinx::mesh::Mesh<T> &domain,
      const dolfinx::fem::Function<T> &active_els_func,
      const std::span<const bool> nodes_to_subtract) {
  auto topology = domain.topology();
  auto cell_map = topology->index_map(topology->dim());
  const std::int32_t num_local_cells = cell_map->size_local(), num_ghost_cells = cell_map->num_ghosts();
  auto [ext_active_els_mask, _colliding_els_mask] = node_mask_to_el_mask(domain, nodes_to_subtract);
  std::span<const std::int8_t> ext_active_els_mask_vals = ext_active_els_mask.array();
  std::span<const std::int8_t> colliding_els_mask = _colliding_els_mask.array();

  auto curr_active_els_mask = active_els_func.x()->array();
  std::vector<std::int32_t> active_els, colliding_els;
  for (int icell = 0; icell < (num_local_cells+num_ghost_cells); ++icell) {
    if ((curr_active_els_mask[icell]) and (not(ext_active_els_mask_vals[icell]))) {
      active_els.push_back(icell);
    } else if (colliding_els_mask[icell]) {
      colliding_els.push_back(icell);
    }
  }
  return {active_els, colliding_els};
}

template <std::floating_point T>
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>>
intersect_from_nodes(const dolfinx::mesh::Mesh<T> &domain,
    const dolfinx::fem::Function<T> &active_els_func,
    const std::span<const bool> nodes_to_intersect) {

  auto topology = domain.topology();
  auto cell_map = topology->index_map(topology->dim());
  const std::int32_t num_local_cells = cell_map->size_local(), num_ghost_cells = cell_map->num_ghosts();
  auto [ext_active_els_mask, _colliding_els_mask] = node_mask_to_el_mask(domain, nodes_to_intersect);
  std::span<const std::int8_t> ext_active_els_mask_vals = ext_active_els_mask.array();
  std::span<const std::int8_t> colliding_els_mask = _colliding_els_mask.array();

  auto curr_active_els_mask = active_els_func.x()->array();
  std::vector<std::int32_t> active_els, colliding_els;
  for (int icell = 0; icell < (num_local_cells+num_ghost_cells); ++icell) {
    if ((curr_active_els_mask[icell]) and (ext_active_els_mask_vals[icell])) {
      active_els.push_back(icell);
    } else if (colliding_els_mask[icell]) {
      colliding_els.push_back(icell);
    }
  }
  return {active_els, colliding_els};
}

template <std::floating_point T>
mesh::MeshTags<std::int8_t> find_interface(const dolfinx::mesh::Mesh<T> &domain,
    mesh::MeshTags<std::int8_t> &bfacet_tag,
    const dolfinx::common::IndexMap& dofs_index_map,
    const std::span<const bool> &ext_active_dofs_flags) {
  // Initialize arrays
  auto topology = domain.topology();
  int cdim = topology->dim();
  auto node_map = topology->index_map(0);
  auto facet_map = topology->index_map(cdim-1);
  auto con_facet_nodes = topology->connectivity(cdim-1,0);

  size_t total_facet_count = facet_map->size_local() + facet_map->num_ghosts();
  std::vector<std::int8_t> tags(total_facet_count, 0);
  std::vector<std::int32_t> bfacets = bfacet_tag.find(1);
  size_t num_nodes_facet = con_facet_nodes->num_links(0);// num links of first facet
                                                         // assuming constant
  std::vector<std::int64_t> global_indices_local_con(num_nodes_facet);
  std::vector<std::int32_t> local_dofs(num_nodes_facet);
  for (std::int32_t ifacet : bfacets) {
    auto local_con = con_facet_nodes->links(ifacet);
    node_map->local_to_global(local_con,global_indices_local_con);
    dofs_index_map.global_to_local(global_indices_local_con,local_dofs);
    bool all_active = std::all_of(local_dofs.begin(),
                                  local_dofs.end(),
                                  [&ext_active_dofs_flags](std::int32_t idof){
                                    return ext_active_dofs_flags[idof];
                                  });
    if (all_active) {
      if (ifacet < facet_map->size_local()) {
        tags[ifacet] = 1;
      } else {
        tags[ifacet] = 2;
      }
    }
  }

  std::vector<std::int32_t> ents(total_facet_count);
  std::iota(ents.begin(),ents.end(),0);
  return mesh::MeshTags<std::int8_t>(topology,cdim-1,std::move(ents),std::move(tags));
}

namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_activation_utils(nb::module_ &m) {
  m.def("locate_active_boundary", &locate_active_boundary<T>);
  m.def(
      "deactivate_from_nodes",
      [](const dolfinx::mesh::Mesh<T> &domain,
         const dolfinx::fem::Function<T> &active_els_func,
         nb::ndarray<const bool, nb::ndim<1>, nb::c_contig> nodes_to_subtract)
      {
      auto [active_els, colliding_els] = deactivate_from_nodes<T>(domain,
          active_els_func,
          std::span(nodes_to_subtract.data(),nodes_to_subtract.size()));
      return std::tuple(
          nb::ndarray<const std::int32_t, nb::numpy>(active_els.data(),
            {active_els.size()}).cast(),
          nb::ndarray<const std::int32_t, nb::numpy>(colliding_els.data(),
            {colliding_els.size()}).cast());

      }
      );
  m.def(
      "intersect_from_nodes",
      [](const dolfinx::mesh::Mesh<T> &domain,
         const dolfinx::fem::Function<T> &active_els_func,
         nb::ndarray<const bool, nb::ndim<1>, nb::c_contig> nodes_to_subtract)
      {
      auto [active_els, colliding_els] = intersect_from_nodes<T>(domain,
          active_els_func,
          std::span(nodes_to_subtract.data(),nodes_to_subtract.size()));
      return std::tuple(
          nb::ndarray<const std::int32_t, nb::numpy>(active_els.data(),
            {active_els.size()}).cast(),
          nb::ndarray<const std::int32_t, nb::numpy>(colliding_els.data(),
            {colliding_els.size()}).cast());

      }
      );
  m.def("find_interface",
      [](const dolfinx::mesh::Mesh<T> &domain,
         mesh::MeshTags<std::int8_t> &bfacet_tag,
         const dolfinx::common::IndexMap& dofs_index_map,
         nb::ndarray<const bool, nb::ndim<1>, nb::c_contig> ext_active_dofs_flags)
      {
        return find_interface(domain,
                              bfacet_tag,
                              dofs_index_map,
                              std::span(ext_active_dofs_flags.data(),ext_active_dofs_flags.size()));
      }
      );
}

void declare_activation_utils(nb::module_ &m) {
  templated_declare_activation_utils<float>(m);
  templated_declare_activation_utils<double>(m);
}
