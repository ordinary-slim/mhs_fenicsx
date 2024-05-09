#pragma once
#include <algorithm>
#include <numeric>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/Constant.h>

template <std::floating_point T>
void compute_el_size_along_vector(dolfinx::fem::Function<T> &dg_func,
                                  dolfinx::fem::Constant<T> &vector) {
  auto mesh = dg_func.function_space()->mesh();
  int cdim = mesh->topology()->dim();
  auto cell_map = mesh->topology()->index_map(cdim);
  auto dof_map = mesh->geometry().dofmap();
  std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
  assert(cell_map);
  auto x = dg_func.x()->mutable_array();
  std::int32_t num_dofs_cell = dof_map.extent(1);
  std::vector<T> coordinate(3);
  std::vector<T> vector_direction = vector.value;
  double vector_norm = sqrt(std::inner_product(vector_direction.begin(),
        vector_direction.end(),vector_direction.begin(),0.0));
  assert(vector_norm!=0);
  std::transform(vector_direction.begin(), vector_direction.end(), vector_direction.begin(), [&vector_norm](auto &c){return c/vector_norm;});

  for (int icell = 0; icell < num_cells; ++icell) {
    double min_proj = 99e+99, max_proj = -99e+99;
    auto dofs_cell = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dof_map, icell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t inode = 0; inode < num_dofs_cell; ++inode) {
      std::copy_n(std::next(mesh->geometry().x().begin(), 3 * dofs_cell[inode]), 3,
                  std::next(coordinate.begin(), 0));
      double projection = std::inner_product(vector_direction.begin(),
                                             vector_direction.end(),
                                             coordinate.begin(),
                                             0.0);
      min_proj = std::min(projection, min_proj);
      max_proj = std::max(projection, max_proj);
    }
    x[icell] = max_proj - min_proj;
  }
}
