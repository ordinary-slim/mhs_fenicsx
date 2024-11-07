#pragma once
#include <dolfinx/fem/assemble_matrix_impl.h>
#include <basix/quadrature.h>
#include <stdexcept>


template <dolfinx::scalar T>
void assemble_monolithic_robin(la::MatSet<T> auto mat_add, const dolfinx::fem::FunctionSpace<double>& V) {

  using U = double;
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using mdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

  auto mesh = V.mesh();
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  std::shared_ptr<const common::IndexMap> cmap = mesh->topology()->index_map(tdim);

  auto element = V.element();
  const int order = element->basix_element().degree();

  // Construct quadrature rule
  const int max_degree = 2 * order;
  auto quadrature_type = basix::quadrature::get_default_rule(
      element->basix_element().cell_type(), max_degree);
  auto [X_b, weights] = basix::quadrature::make_quadrature<T>(
      quadrature_type, element->basix_element().cell_type(),
      basix::polyset::type::standard, max_degree);
  mdspan2_t X(X_b.data(), weights.size(), 2);

  // Tabulate basis functions at quadrature points
  auto e_shape = element->basix_element().tabulate_shape(0, weights.size());
  std::size_t length
      = std::accumulate(e_shape.begin(), e_shape.end(), 1, std::multiplies<>{});
  std::vector<T> phi_b(length);
  mdspan4_t phi(phi_b.data(), e_shape);
  element->basix_element().tabulate(0, X, phi);

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell (affine cells)
  const dolfinx::fem::CoordinateElement<U>& coordinate_element = mesh->geometry().cmap();
  const std::size_t num_dofs_g = coordinate_element.dim();
  std::array<std::size_t, 4> phi0_shape = coordinate_element.tabulate_shape(1, 1);
  std::vector<U> phi0_b(
      std::reduce(phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi0(phi0_b.data(), phi0_shape);
  coordinate_element.tabulate(1, std::vector<U>(tdim, 0), {1, tdim}, phi0_b);
  auto dphi0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi0, std::pair(1, tdim + 1), 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  /*
  for (size_t ndsize = 0; ndsize < e_shape[0]; ++ndsize) {
    for (size_t ipoint = 0; ipoint < e_shape[1]; ++ipoint) {
      printf("point = (%g, %g)\n", X(ipoint, 0), X(ipoint, 1));
      for (size_t idof = 0; idof < e_shape[2]; ++idof) {
        for (size_t ival = 0; ival < e_shape[3]; ++ival) {
          printf("phi(iderivate %li, ipoint %li, idof %li, ival %li) = %g\n",
              ndsize, ipoint, idof, ival, phi(ndsize, ipoint, idof, ival));
        }
      }
    }
  }
  */

  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = mesh->geometry().dofmap();
  std::span<const U> x_g = mesh->geometry().x();
  // Create buffer for coordinate dofs and point in physical space
  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
  std::vector<U> xp_b(1 * gdim);
  mdspan2_t xp(xp_b.data(), 1, gdim);

  // Create buffers for geometry data
  std::vector<U> J_b(gdim * tdim);
  mdspan2_t J(J_b.data(), gdim, tdim);
  std::vector<U> K_b(tdim * gdim);
  mdspan2_t K(K_b.data(), tdim, gdim);
  std::vector<U> det_scratch(2 * gdim * tdim);

  std::vector<std::int32_t> cells(cmap->size_local());
  std::iota(cells.begin(), cells.end(), 0);

  std::shared_ptr<const fem::DofMap> dofmap = V.dofmap();
  auto connectivity = dofmap->map();
  const int num_dofs_cell = connectivity.extent(1);

  /*
  if (not(mesh->geometry().cmap().is_affine())) {
    throw std::invalid_argument("Only affine\n");
  }
  */
  const size_t num_entries_locmat = num_dofs_cell*num_dofs_cell;
  std::vector<T> A_eb(num_entries_locmat);
  mdspan2_t A_e(A_eb.data(), num_dofs_cell, num_dofs_cell);
  for (size_t icell = 0; icell < cells.size(); ++icell) {
    // Get cell geometry (coordinate dofs)
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, icell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }
    std::fill(J_b.begin(), J_b.end(), U(0.0));
    // Compute detJ
    dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi0, coord_dofs, J);
    U detJ = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(J, det_scratch);


    std::fill(A_eb.begin(), A_eb.end(), T(0.0));
    for (size_t k = 0; k < phi.extent(1); ++k) {//igauss
      for (size_t i = 0; i < num_dofs_cell; ++i) {//inode
        for (size_t j = 0; j < num_dofs_cell; ++j) {//jnode
          A_e(i, j) += weights[k] * phi(0, k, i, 0) * phi(0, k, j, 0) * abs(detJ);
        }
      }
    }
    auto dofs = std::span(connectivity.data_handle() + icell * num_dofs_cell, num_dofs_cell);
    mat_add(dofs, dofs, A_eb);
  }



  //element->basix_element().tabulate_shape(0, 2);

  /*
  std::array<int, 1> dofs0 = {0};
  std::array<int, 1> dofs1 = {0};
  std::vector<T> Ae = {44};
  mat_add(dofs0, dofs1, Ae);
  */
}

/*
template <dolfinx::scalar T>
void assemble_monolithic_robin(auto mat_add) {
  assemble_monolithic_robin(mat_add);
}
*/
