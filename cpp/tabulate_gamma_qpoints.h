#pragma once
#include <dolfinx/fem/assemble_matrix_impl.h>
#include <basix/quadrature.h>

using U = double;

inline std::vector<U> tabulate_gamma_quadrature(
    const dolfinx::mesh::Mesh<U> &mesh,
    std::span<const std::int32_t> gamma_integration_data,
    size_t num_gps_facet,
    std::span<const U> _quadrature_points_cell
    ) {
  using cmdspan2 = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using mdspan2 = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using cmdspan4 = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

  int cdim = mesh.topology()->dim();
  int num_facets_cell = basix::cell::num_sub_entities(mesh::cell_type_to_basix_type(mesh.topology()->cell_type()), cdim-1);
  assert(gamma_integration_data.size() % 2 == 0);
  size_t num_local_gamma_cells = gamma_integration_data.size() / 2;

  int gdim = mesh.geometry().dim();
  const dolfinx::fem::CoordinateElement<U>& coordinate_element = mesh.geometry().cmap();
  auto x_dofmap = mesh.geometry().dofmap();
  std::span<const U> x_geo = mesh.geometry().x();
  int num_dofs_cell = mesh.geometry().dofmap().extent(1);

  std::vector<U> _gamma_qpoints(num_local_gamma_cells*num_gps_facet*3);
  mdspan2 gamma_qpoints(_gamma_qpoints.data(), num_local_gamma_cells*num_gps_facet, 3);
  cmdspan2 quadrature_points_cell(_quadrature_points_cell.data(), num_gps_facet*num_facets_cell, cdim);

  // TODO: Move to outside of this function
  std::array<std::size_t, 4> phi_shape = coordinate_element.tabulate_shape(0, quadrature_points_cell.extent(0));
  std::vector<U> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(),
        1, std::multiplies{}));
  cmdspan4 phi_full(phi_b.data(), phi_shape);
  coordinate_element.tabulate(0, std::span(_quadrature_points_cell.data(), _quadrature_points_cell.size()), {quadrature_points_cell.extent(0), quadrature_points_cell.extent(1)}, phi_b);
  auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi_full, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  std::vector<U> cell_coords_b(num_dofs_cell * gdim);
  mdspan2 cell_coords(cell_coords_b.data(), x_dofmap.extent(1), gdim);

  for (int i = 0; i < num_local_gamma_cells; ++i) {
    std::int32_t icell = gamma_integration_data[2*i];
    std::int32_t lifacet = gamma_integration_data[2*i+1];
    auto dofs_cell = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, icell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t j = 0; j < num_dofs_cell; ++j) {
      const int pos = 3 * dofs_cell[j];
      for (std::size_t k = 0; k < gdim; ++k) {
        cell_coords(j, k) = x_geo[pos + k];
      }
    }
    auto qpoints = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        gamma_qpoints,
        std::pair(i*num_gps_facet, i*num_gps_facet + num_gps_facet),
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto phi_sub = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        phi, 
        std::pair(lifacet*num_gps_facet,lifacet*num_gps_facet+num_gps_facet),
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    // Push forward
    coordinate_element.push_forward(qpoints, cell_coords, phi_sub);
  }
  return _gamma_qpoints;
}
