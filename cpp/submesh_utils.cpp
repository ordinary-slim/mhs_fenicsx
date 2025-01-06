#include <concepts>
#include <dolfinx/mesh/Mesh.h>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

template <std::floating_point T>
std::vector<std::int32_t> build_subentity_to_parent_mapping(
    int edim,
    const dolfinx::mesh::Mesh<T> &pmesh,
    const dolfinx::mesh::Mesh<T> &cmesh,
    std::span<const std::int32_t> subcell_map,
    std::span<const std::int32_t> subvertex_map) {

  auto ptopo = pmesh.topology_mutable();
  auto ctopo = cmesh.topology_mutable();
  int tdim = ptopo->dim();
  assert(tdim==ctopo->dim());

  auto centity_map = ctopo->index_map(edim);
  int num_cents = centity_map->size_local() + centity_map->num_ghosts();

  std::vector<std::int32_t> subentity_map(num_cents,-1);

  if (num_cents == 0) {
    return subentity_map;
  }

  ctopo->create_connectivity(edim,tdim);
  auto ccon_e2c = ctopo->connectivity(edim,tdim);
  auto ccon_e2v = ctopo->connectivity(edim,0);

  ptopo->create_connectivity(tdim,edim);
  ptopo->create_connectivity(edim,tdim);
  auto pcon_e2v = ptopo->connectivity(edim,0);
  auto pcon_c2e = ptopo->connectivity(tdim,edim);

  // If num_cents > 0, this should work
  const std::size_t nnodes_per_ent = ccon_e2v->num_links(0);
  assert(nnodes_per_ent==pcon_e2v->num_links(0));
  std::vector<std::int32_t> pinodes(nnodes_per_ent), pinodes2compare_sorted(nnodes_per_ent);

  for (int ient = 0; ient < num_cents; ++ient) {
    bool entity_found = false;
    auto cicells = ccon_e2c->links(ient);//child incident cells
    auto cinodes = ccon_e2v->links(ient);
    std::transform(cinodes.begin(),cinodes.end(),pinodes.begin(),[&subvertex_map](auto &cinode){return subvertex_map[cinode];});
    sort( pinodes.begin(), pinodes.end() );
    for (auto cicell : cicells) {
      if (entity_found) {
        break;
      }
      auto picell = subcell_map[cicell];
      auto pients = pcon_c2e->links(picell);
      for (auto pient : pients) {
        auto pinodes2compare_unsorted = pcon_e2v->links(pient);
        std::copy_n(pinodes2compare_unsorted.begin(),pinodes2compare_unsorted.size(),pinodes2compare_sorted.begin());
        sort( pinodes2compare_sorted.begin(), pinodes2compare_sorted.end() );
        entity_found = (pinodes==pinodes2compare_sorted);
        if (entity_found) {
          subentity_map[ient] = pient;
          break;
        }
      }
    }
    if (not(entity_found)) {
      throw std::runtime_error("Did not find mesh-submesh match!");
    }
  }

  return subentity_map;
}

namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_build_subentity_to_parent_mapping(nb::module_ &m) {
  m.def(
      "build_subentity_to_parent_mapping",
      [](int edim,
         const dolfinx::mesh::Mesh<T> &pmesh,
         const dolfinx::mesh::Mesh<T> &cmesh,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> subcell_map,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> subvertex_map)
      {
        std::vector<std::int32_t> mapping = build_subentity_to_parent_mapping<T>(edim,
            pmesh, cmesh, std::span(subcell_map.data(),subcell_map.size()),
            std::span(subvertex_map.data(),subvertex_map.size()));
        return nb::ndarray<const std::int32_t, nb::numpy>(mapping.data(),
                                                  {mapping.size()}).cast();
      }
      );
}

void declare_submesh_utils(nb::module_ &m) {
  templated_declare_build_subentity_to_parent_mapping<double>(m);
  templated_declare_build_subentity_to_parent_mapping<float>(m);
}
