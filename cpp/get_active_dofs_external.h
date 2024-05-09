#pragma once
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/gjk.h>
#include <vector>
#include <span>

using int_vector = std::vector<int>;

template <std::floating_point T>
std::vector<int> get_active_dofs_external(dolfinx::fem::Function<T> &loc_active_nodes,
                                          const dolfinx::fem::Function<T> &ext_active_els,
                                          const dolfinx::geometry::BoundingBoxTree<T> &local_nodal_tree,
                                          const dolfinx::mesh::Mesh<T> &local_domain,
                                          const dolfinx::geometry::BoundingBoxTree<T> &ext_cell_tree,
                                          const dolfinx::mesh::Mesh<T> &ext_domain) {

  std::vector<int> active_nodes;
  const mesh::Geometry<T>& loc_geo = local_domain.geometry();
  const mesh::Geometry<T>& ext_geo = ext_domain.geometry();
  std::span<const T> loc_x = loc_geo.x();
  std::span<const T> ext_x = ext_geo.x();
  auto loc_dofmap = loc_geo.dofmap();
  auto ext_dofmap = ext_geo.dofmap();
  const std::size_t ext_nnodes_el = ext_dofmap.extent(1);
  std::vector<T> el_coordinates(ext_nnodes_el * 3);

  std::vector coll = dolfinx::geometry::compute_collisions<T>(local_nodal_tree, ext_cell_tree);

  for (int i = 0; i < coll.size()/2; ++i) {
    int inode = coll[2*i];
    int ielem = coll[2*i+1];
    auto ext_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        ext_dofmap, ielem, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t k = 0; k < ext_nnodes_el; ++k)
    {
      std::copy_n(std::next(ext_x.begin(), 3 * ext_dofs[k]), 3,
                  std::next(el_coordinates.begin(), 3 * k));
    }
    std::array<T, 3> d = dolfinx::geometry::compute_distance_gjk<T>(loc_x.subspan(3 * inode, 3), el_coordinates);

    T d2 = std::reduce(d.begin(), d.end(), T(0),
                       [](auto d, auto e) { return d + e * e; });
    //if (d2 < 1e-7) active_nodes.push_back(inode);
    active_nodes.push_back(inode);
  }
  return active_nodes;
}
