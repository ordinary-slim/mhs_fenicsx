#include <cassert>
#include <cstdio>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <dolfinx/geometry/gjk.h>
#include <basix/mdspan.hpp>


template <std::floating_point T>
void extract_cell_geometry(std::vector<T> &cell_geo, int ielem, int nnodes_per_el,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan< const std::int32_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>> dofmap,
    std::span<const T> x_geo) {

    auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(dofmap, ielem, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t k = 0; k < nnodes_per_el; ++k) {
      std::copy_n(std::next(x_geo.begin(), 3 * dofs[k]), 3,
                  std::next(cell_geo.begin(), 3 * k));
    }
}

template <std::floating_point T>
std::vector<std::int32_t> mesh_collision(
    const dolfinx::mesh::Mesh<T> &mesh_big,
    const dolfinx::mesh::Mesh<T> &mesh_small,
    const dolfinx::geometry::BoundingBoxTree<T> *bb_tree_big = nullptr) {
    /*
    mesh_small is in MPI.COMM_SELF
    mesh_big is in MPI.COMM_WORLD
    */
  // TODO: Handle case where nullptr is passed.
  double tol = 1e-7;
  if (not(bb_tree_big)) {
    throw std::runtime_error("Expected valid ptr to bb_tree_big");
  }
  int cdim = mesh_small.topology()->dim();
  assert(cdim==mesh_big.topology()->dim());
  auto cmap_small = mesh_small.topology_mutable()->index_map(cdim);
  const std::int32_t num_entities_small = cmap_small->size_local() + cmap_small->num_ghosts();
  std::vector<std::int32_t> ents_small(num_entities_small);
  std::iota(ents_small.begin(), ents_small.end(), 0);
  auto bb_tree_small = dolfinx::geometry::BoundingBoxTree<T>(mesh_small, cdim, ents_small, tol);
  std::vector<std::int32_t> ent_pairs = dolfinx::geometry::compute_collisions(*bb_tree_big,bb_tree_small);

  auto dofmap_small = mesh_small.geometry().dofmap();
  auto dofmap_big = mesh_big.geometry().dofmap();
  auto x_geo_small = mesh_small.geometry().x();
  auto x_geo_big = mesh_big.geometry().x();
  const std::size_t nnodes_per_el_small = dofmap_small.extent(1);
  const std::size_t nnodes_per_el_big = dofmap_big.extent(1);
  std::vector<T> cell_geo_small(nnodes_per_el_small * 3); 
  std::vector<T> cell_geo_big(nnodes_per_el_big * 3); 

  int num_potential_collisions = ent_pairs.size()/2;
  std::vector<std::int32_t> big_colliding_cells;
  big_colliding_cells.reserve(num_potential_collisions);
  for (int ipair = 0; ipair < num_potential_collisions; ++ipair) {
    std::int32_t cell_big = ent_pairs[2*ipair];
    std::int32_t cell_small = ent_pairs[2*ipair+1];
    extract_cell_geometry(cell_geo_small, cell_small, nnodes_per_el_small, dofmap_small, x_geo_small);
    extract_cell_geometry(cell_geo_big, cell_big, nnodes_per_el_big, dofmap_big, x_geo_big);
    std::array<T, 3> d = dolfinx::geometry::compute_distance_gjk<T>(cell_geo_small, cell_geo_big);
    T d2 = std::reduce(d.begin(), d.end(), T(0), [](auto d, auto e) { return d + e * e; });
    if (d2 < tol) {
      big_colliding_cells.push_back(cell_big);
    }
  }
  sort( big_colliding_cells.begin(), big_colliding_cells.end() );
  big_colliding_cells.erase( unique( big_colliding_cells.begin(), big_colliding_cells.end() ), big_colliding_cells.end() );

  return big_colliding_cells;
}

namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_mesh_collision(nb::module_ &m) {
  m.def(
      "mesh_collision",
      [](
        const dolfinx::mesh::Mesh<T> &mesh_big,
        const dolfinx::mesh::Mesh<T> &mesh_small,
        const dolfinx::geometry::BoundingBoxTree<T>& bb_tree_big
        )
      {
        std::vector<std::int32_t> elements = mesh_collision(
            mesh_big,
            mesh_small,
            &bb_tree_big
            );
        return nb::ndarray<const std::int32_t, nb::numpy>(elements.data(),
                                                  {elements.size()}).cast();
      }, nb::arg("mesh_big"),nb::arg("mesh_small"),nb::arg("bb_tree_big").none());
}

void declare_mesh_collision(nb::module_ &m) {
  templated_declare_mesh_collision<double>(m);
  templated_declare_mesh_collision<float>(m);
}
