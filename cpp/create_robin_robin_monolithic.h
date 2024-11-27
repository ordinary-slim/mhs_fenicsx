#pragma once
#include <dolfinx/fem/assemble_matrix_impl.h>
#include <basix/quadrature.h>
#include <basix/cell.h>

using bct = basix::cell::type;

template <dolfinx::scalar T>
void assemble_monolithic_robin(la::MatSet<T> auto mat_add,
    const dolfinx::fem::FunctionSpace<double>& V,
    std::span<const std::int32_t> facets_mesh_i,
    std::span<const std::int32_t> cells_mesh_i,
    std::span<const T> gpoints_cell_b,
    std::span<const T> gweights_cell,
    const size_t num_gp_per_facet)
{

  // Dictionary cell type to facet cell type
  std::map<bct, bct> bcell_type {
    {bct::interval, bct::point},
    {bct::triangle, bct::interval},
    {bct::quadrilateral, bct::interval},
    {bct::tetrahedron, bct::triangle},
    {bct::hexahedron, bct::quadrilateral}
  };

  using U = double;
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using mdspan3_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>;
  using mdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

  auto mesh = V.mesh();
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  std::shared_ptr<const common::IndexMap> cmap = mesh->topology()->index_map(tdim);
  auto con_cell_facet = mesh->topology()->connectivity(tdim,tdim-1);
  assert(con_cell_facet);

  auto element = V.element();
  const int order = element->basix_element().degree();

  assert(gpoints_cell_b.size() % gdim == 0);
  const std::size_t num_points = gpoints_cell_b.size() / gdim;
  cmdspan2_t gpoints_cell(gpoints_cell_b.data(), num_points, gdim);

  // Tabulate basis functions at quadrature points
  int nderivative = 1;
  auto e_shape = element->basix_element().tabulate_shape(nderivative, gweights_cell.size());
  std::size_t length
      = std::accumulate(e_shape.begin(), e_shape.end(), 1, std::multiplies<>{});
  std::vector<T> phi_b(length);
  mdspan4_t phi(phi_b.data(), e_shape);
  element->basix_element().tabulate(nderivative, gpoints_cell, phi);
  auto dphi_ref = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi, std::pair(1, tdim + 1), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  std::vector<T> dphi_b(dphi_ref.extent(0)*dphi_ref.extent(1)*dphi_ref.extent(2));
  // dim, gp, basis func
  mdspan3_t dphi(dphi_b.data(), dphi_ref.extent(0), dphi_ref.extent(1), dphi_ref.extent(2));

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

  // Create buffer for local contribution
  auto fcell_type = bcell_type[element->basix_element().cell_type()];
  const size_t num_dofs_facet = basix::cell::num_sub_entities(fcell_type, 0);
  auto sub_entity_connectivity = basix::cell::sub_entity_connectivity(element->basix_element().cell_type());
  const size_t num_entries_locmat = num_dofs_facet*num_dofs_facet;
  std::vector<T> A_eb(num_entries_locmat);
  mdspan2_t A_e(A_eb.data(), num_dofs_facet, num_dofs_facet);

  for (size_t idx = 0; idx < facets_mesh_i.size(); ++idx) {
    std::int32_t ifacet_i = facets_mesh_i[idx];
    std::int32_t icell_i = cells_mesh_i[idx];
    // Find local index of facet
    auto local_con_cell_facets = con_cell_facet->links(icell_i);
    auto itr = std::find(local_con_cell_facets.begin(),
                         local_con_cell_facets.end(),
                         ifacet_i);
    assert(itr!=local_con_cell_facets.end());
    size_t lifacet_i = std::distance(local_con_cell_facets.begin(), itr);
    // relevant dofs are sub_entity_connectivity[fdim][lifacet_i][0]
    const auto& lfacet_dofs = sub_entity_connectivity[tdim-1][lifacet_i][0];

    // Get cell geometry (coordinate dofs)
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, icell_i, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }
    std::fill(J_b.begin(), J_b.end(), U(0.0));
    std::fill(K_b.begin(), K_b.end(), U(0.0));
    // Compute reference coordinates X, and J, detJ and K
    dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi0, coord_dofs, J);
    dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(J, K);
    U detJ = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(J, det_scratch);
    U fdetJ = 0.0;
    switch (fcell_type) {
      case (bct::interval):
        for (size_t i = 0; i < gdim; ++i) {
          fdetJ += pow(coord_dofs(lfacet_dofs[1],i) - coord_dofs(lfacet_dofs[0],i),2);
        }
        fdetJ = pow(fdetJ, 0.5);
        break;
      case (bct::triangle): case (bct::quadrilateral) : {
        std::array<double, 3> u = {coord_dofs(lfacet_dofs[2],0) - coord_dofs(lfacet_dofs[0],0), coord_dofs(lfacet_dofs[2],1) - coord_dofs(lfacet_dofs[0],1), coord_dofs(lfacet_dofs[2],2) - coord_dofs(lfacet_dofs[0],2)};
        std::array<double, 3> v = {coord_dofs(lfacet_dofs[1],0) - coord_dofs(lfacet_dofs[0],0), coord_dofs(lfacet_dofs[1],1) - coord_dofs(lfacet_dofs[0],1), coord_dofs(lfacet_dofs[1],2) - coord_dofs(lfacet_dofs[0],2)};
        std::array<double, 3> cross =  {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
        for (size_t i = 0; i < 3; ++i) {
          fdetJ += pow(cross[i],2);
        }
        fdetJ = pow(fdetJ, 0.5);
        break;}
      default:
        throw std::invalid_argument("AAAAAAAAAAAAA");
        break;
    }

    // Compute local contribution
    std::fill(A_eb.begin(), A_eb.end(), T(0.0));
    for (size_t k = 0; k < num_gp_per_facet; ++k) {//igauss
      //TODO: Get right k
      for (size_t i = 0; i < lfacet_dofs.size(); ++i) {//inode
        //TODO: Get right i
        for (size_t j = 0; j < lfacet_dofs.size(); ++j) {//inode
          //TODO: Get right j
          A_e(i, j) += gweights_cell[lifacet_i*num_dofs_facet+k] *
                       phi(0, lifacet_i*num_dofs_facet+k, lfacet_dofs[i], 0) *
                       phi(0, lifacet_i*num_dofs_facet+k, lfacet_dofs[j], 0) *
                       abs(fdetJ);
        }
      }
    }
    auto dofs = std::span(connectivity.data_handle() + icell_i * num_dofs_cell, num_dofs_cell);
    std::vector<int> facet_dofs(lfacet_dofs.size());
    std::transform(lfacet_dofs.begin(), lfacet_dofs.end(), facet_dofs.begin(),
                   [&dofs](size_t index) { return dofs[index]; });
    mat_add(facet_dofs, facet_dofs, A_eb);
  }
  /*
    // Compute physical J
    std::fill(dphi_b.begin(), dphi_b.end(), U(0.0));
    for (std::size_t g = 0; g < dphi_ref.extent(1); g++)//gp
      for (std::size_t i = 0; i < K.extent(1); i++)
        for (std::size_t j = 0; j < dphi_ref.extent(2); j++)
          for (std::size_t k = 0; k < K.extent(0); k++)
            dphi(i,g,j) += K(k,i) * dphi_ref(k,g,j);

    std::fill(A_eb.begin(), A_eb.end(), T(0.0));
    for (size_t k = 0; k < phi.extent(1); ++k) {//igauss
      for (size_t i = 0; i < num_dofs_cell; ++i) {//inode
        for (size_t j = 0; j < num_dofs_cell; ++j) {//jnode
          T grad_phi_dot = T(0.0);
          for (size_t dim = 0; dim < dphi_ref.extent(0); ++dim)
            grad_phi_dot += dphi(dim,k,i) * dphi(dim,k,j);
          A_e(i, j) += weights[k] * grad_phi_dot * abs(detJ);
        }
      }
    }
    auto dofs = std::span(connectivity.data_handle() + icell * num_dofs_cell, num_dofs_cell);
  }
  */
}
