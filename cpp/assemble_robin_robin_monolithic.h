#pragma once
#include <dolfinx/fem/assemble_matrix_impl.h>
#include <basix/quadrature.h>
#include <basix/cell.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/petsc.h>
#include <multiphenicsx/DofMapRestriction.h>
#include "CustomSparsityPattern.h"

using bct = basix::cell::type;
using U = double;

using cmdspan2_i = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const std::int32_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

struct Triplet
{
  std::int64_t row, col;
  double value;
  Triplet(size_t row_, size_t col_, double value_) :
  row(row_), col(col_), value(value_) {
  }
};

template <dolfinx::scalar T>
std::tuple<std::vector<U>, std::vector<U>, std::vector<U>>
     compute_geometry_data(const dolfinx::mesh::Mesh<T> &mesh,
                           std::span<const T> _geoms_cells_mesh_j)
{
  const std::size_t tdim = mesh.topology()->dim();
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t num_dofs_g = mesh.geometry().cmap().dim();
  assert((_geoms_cells_mesh_j.size() % (num_dofs_g * 3))==0);
  const std::size_t num_cells = _geoms_cells_mesh_j.size() / (num_dofs_g * 3);

  using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using mdspan3_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>;
  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

  const dolfinx::fem::CoordinateElement<U>& cmap = mesh.geometry().cmap();
  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Assuming affine
  std::array<std::size_t, 4> phi0_shape = cmap.tabulate_shape(1, 1);
  std::vector<U> phi0_b(
      std::reduce(phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi0(phi0_b.data(), phi0_shape);
  cmap.tabulate(1, std::vector<U>(tdim, 0), {1, tdim}, phi0_b);
  auto dphi0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi0, std::pair(1, tdim + 1), 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  // Geometry data at each cell
  std::vector<U> J_b(num_cells * gdim * tdim);
  mdspan3_t J(J_b.data(), num_cells, gdim, tdim);
  std::vector<U> K_b(num_cells * tdim * gdim);
  mdspan3_t K(K_b.data(), num_cells, tdim, gdim);
  std::vector<U> detJ(num_cells);
  std::vector<U> det_scratch(2 * gdim * tdim);

  for (int idx = 0; idx < num_cells; ++idx) {
    // Get cell geometry (coordinate dofs)
    cmdspan2_t coord_dofs(_geoms_cells_mesh_j.data() + idx * (num_dofs_g * 3), num_dofs_g, 3);

    auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        J, idx, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        K, idx, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    // Compute reference coordinates X, and J, detJ and K
    dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi0, coord_dofs, _J);
    dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
    detJ[idx] = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
            _J, det_scratch);
  }
  return {J_b, K_b, detJ};
}

inline auto set_fn(Mat A, InsertMode mode)
{
  return [A, mode, cache = std::vector<PetscInt>()](
             std::int64_t row,
             std::int64_t col,
             double val) mutable -> int
  {
    PetscErrorCode ierr;
    ierr = MatSetValue(A, row, col, val, mode);

#ifndef NDEBUG
    if (ierr != 0)
      dolfinx::la::petsc::error(ierr, __FILE__, "MatSetValue");
#endif
    return ierr;
  };
}

template <dolfinx::scalar T>
class MonolithicRobinRobinAssembler {
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

  public:
  MonolithicRobinRobinAssembler() { }

  void preassemble(
      Mat& A,
      std::span<const T> _tabulated_gauss_points_gamma,
      std::span<const T> _gauss_points_cell,
      std::span<const T> gweights_facet,
      const dolfinx::fem::FunctionSpace<double>& V_i,
      const multiphenicsx::fem::DofMapRestriction& restriction_i,
      const dolfinx::fem::FunctionSpace<double>& V_j,
      const multiphenicsx::fem::DofMapRestriction& restriction_j,
      std::span<const std::int32_t> gamma_integration_data_i,
      std::span<const int> renumbering_cells_po_mesh_j,
      std::span<const std::int64_t> _dofs_cells_mesh_j,
      std::span<const double> _geoms_cells_mesh_j)
  {
    auto index_map_i = restriction_i.index_map;
    auto index_map_j = restriction_j.index_map;
    int bs_i = restriction_i.index_map_bs();
    int bs_j = restriction_j.index_map_bs();

    auto mesh_i = V_i.mesh();
    auto element_i = V_i.element()->basix_element();
    auto element_j = V_j.element()->basix_element();
    const std::size_t tdim = mesh_i->topology()->dim();
    const std::size_t gdim = mesh_i->geometry().dim();

    auto fcell_type_i = basix::cell::sub_entity_type(element_i.cell_type(), tdim-1, 0);
    size_t num_facets_cell = basix::cell::num_sub_entities(element_i.cell_type(), tdim-1);
    size_t num_gps_facet = gweights_facet.size();
    size_t num_gps_cell = num_gps_facet * num_facets_cell;

    auto con_v_i = restriction_i.dofmap()->map();
    auto con_v_j = restriction_j.dofmap()->map();
    size_t num_dofs_cell_i = con_v_i.extent(1);
    size_t num_dofs_cell_j = con_v_j.extent(1);
    const size_t num_dofs_facet_i = basix::cell::num_sub_entities(fcell_type_i, 0);
    size_t num_diff_cells_j = *std::max_element(renumbering_cells_po_mesh_j.begin(),
                                                renumbering_cells_po_mesh_j.end());
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const std::int64_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>> dofs_cells_mesh_j(
          _dofs_cells_mesh_j.data(), num_diff_cells_j, num_dofs_cell_j);


    assert(_tabulated_gauss_points_gamma.size() % 3 == 0);
    size_t num_gps_processor = _tabulated_gauss_points_gamma.size() / 3;
    cmdspan2_t tabulated_gauss_points_gamma(_tabulated_gauss_points_gamma.data(), num_gps_processor, 3);
    cmdspan2_t gauss_points_cell(_gauss_points_cell.data(), num_gps_cell, tdim);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x_dofmap_i = mesh_i->geometry().dofmap();
    const size_t num_dofs_g_i = x_dofmap_i.extent(1);
    std::span<const U> x_g_i = mesh_i->geometry().x();
    const size_t num_dofs_g_j = V_j.mesh()->geometry().dofmap().extent(1);

    auto e_shape_i = element_i.tabulate_shape(_nderivative_i, num_gps_cell);
    phi_i_b.resize(std::accumulate(e_shape_i.begin(), e_shape_i.end(), 1, std::multiplies<>{}));
    std::fill(phi_i_b.begin(), phi_i_b.end(), 0.0);
    mdspan4_t phi_i(phi_i_b.data(), e_shape_i);
    element_i.tabulate(_nderivative_i, gauss_points_cell, phi_i);

    e_shape_j = element_j.tabulate_shape(_nderivative_j, num_gps_processor);
    phi_j_b.resize(std::accumulate(e_shape_j.begin(), e_shape_j.end(), 1, std::multiplies<>{}));
    dphi_j_b.resize(tdim*e_shape_j[1]*e_shape_j[2]);
    mdspan3_t dphi_j_all_gps(dphi_j_b.data(), tdim, e_shape_j[1], e_shape_j[2]);
    mdspan4_t phi_j(phi_j_b.data(), e_shape_j);
    // Prepare mesh_j integration data
    std::vector<U> _gauss_points_ref_j(num_gps_processor*tdim, 0.0);
    auto [J_j_b, K_j_b, detJ_j] = compute_geometry_data(*V_j.mesh(), _geoms_cells_mesh_j);
    mdspan3_t J_j(J_j_b.data(), num_diff_cells_j, gdim, tdim);
    mdspan3_t K_j(K_j_b.data(), num_diff_cells_j, tdim, gdim);

    std::vector<int> facet_dofs_i(num_dofs_facet_i);
    std::vector<std::int64_t> gfacet_dofs_i(num_dofs_facet_i);
    std::vector<std::int64_t> gdofs_j(num_dofs_cell_j);


    auto sp = CustomSparsityPattern(V_i.mesh()->comm(), {index_map_i, index_map_j}, {bs_i, bs_j});
    auto sub_cell_con_i = basix::cell::sub_entity_connectivity(element_i.cell_type());
    size_t num_gamma_facets = gamma_integration_data_i.size() / 2;
    facet_normals.resize(3*num_gamma_facets);
    std::fill(facet_normals.begin(), facet_normals.end(), 0.0);
    facet_dets.resize(num_gamma_facets);
    std::fill(facet_dets.begin(), facet_dets.end(), 0.0);
    // Buffer for coords facet cell i
    std::vector<U> coord_dofs_i_b(num_dofs_g_i * gdim);
    mdspan2_t coord_dofs_i(coord_dofs_i_b.data(), num_dofs_g_i, gdim);

    // LOOP 1: get sparsity pattern and pull back gps on mesh j
    for (size_t idx = 0; idx < num_gamma_facets; ++idx) {
      std::int32_t icell_i = gamma_integration_data_i[2*idx];
      std::int32_t lifacet_i = gamma_integration_data_i[2*idx+1];
      // relevant dofs are sub_entity_connectivity[fdim][lifacet_i][0]
      const auto& lfacet_dofs = sub_cell_con_i[tdim-1][lifacet_i][0];
      // DOFS i
      auto dofs_cell_i = restriction_i.cell_dofs(icell_i);
      auto udofs_cell_i = std::span(con_v_i.data_handle() + icell_i * num_dofs_cell_i, num_dofs_cell_i);
      std::transform(lfacet_dofs.begin(), lfacet_dofs.end(), facet_dofs_i.begin(),
                     [&dofs_cell_i](size_t index) {return dofs_cell_i[index];});
      restriction_i.index_map->local_to_global(facet_dofs_i, gfacet_dofs_i);

      // Get cell geometry (coordinate dofs)
      std::array<U, 3> cell_centroid_i = {0, 0, 0};
      std::array<U, 3> facet_centroid = {0, 0, 0};
      auto dofs_cell_i_unrestricted = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap_i, icell_i, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < num_dofs_g_i; ++i)
      {
        const int pos = 3 * dofs_cell_i_unrestricted[i];
        for (std::size_t j = 0; j < gdim; ++j) {
          coord_dofs_i(i, j) = x_g_i[pos + j];
          cell_centroid_i[j] += x_g_i[pos + j];
        }
      }
      for (std::size_t i = 0; i < lfacet_dofs.size(); ++i) {
        int idof = lfacet_dofs[i];
        for (std::size_t j = 0; j < gdim; ++j)
          facet_centroid[j] += coord_dofs_i(idof,j);
      }
      for (std::size_t i = 0; i < gdim; ++i) {
        cell_centroid_i[i] /= num_dofs_g_i;
        facet_centroid[i] /= lfacet_dofs.size();
      }

      // Compute facet det and facet normal
      std::span<U> normal(facet_normals.data()+3*idx, 3);
      switch (fcell_type_i) {
        case (bct::point):
          facet_dets[idx] = 1.0;
          normal[0] = 1.0;
          break;
        case (bct::interval):
          for (size_t i = 0; i < gdim; ++i) {
            facet_dets[idx] += pow(coord_dofs_i(lfacet_dofs[1],i) - coord_dofs_i(lfacet_dofs[0],i),2);
            normal[i] = coord_dofs_i(lfacet_dofs[1],i) - coord_dofs_i(lfacet_dofs[0],i);
          }
          std::swap(normal[0], normal[1]);
          facet_dets[idx] = pow(facet_dets[idx], 0.5);
          normal[0] *= -1;
          break;
        case (bct::triangle): case (bct::quadrilateral) : {
          std::array<double, 3> u = {coord_dofs_i(lfacet_dofs[2],0) - coord_dofs_i(lfacet_dofs[0],0), coord_dofs_i(lfacet_dofs[2],1) - coord_dofs_i(lfacet_dofs[0],1), coord_dofs_i(lfacet_dofs[2],2) - coord_dofs_i(lfacet_dofs[0],2)};
          std::array<double, 3> v = {coord_dofs_i(lfacet_dofs[1],0) - coord_dofs_i(lfacet_dofs[0],0), coord_dofs_i(lfacet_dofs[1],1) - coord_dofs_i(lfacet_dofs[0],1), coord_dofs_i(lfacet_dofs[1],2) - coord_dofs_i(lfacet_dofs[0],2)};
          normal[0] = u[1] * v[2] - u[2] * v[1];
          normal[1] = u[2] * v[0] - u[0] * v[2];
          normal[2] = u[0] * v[1] - u[1] * v[0];
          for (size_t i = 0; i < 3; ++i) {
            facet_dets[idx] += pow(normal[i],2);
          }
          facet_dets[idx] = pow(facet_dets[idx], 0.5);
          break;}
        default:
          throw std::invalid_argument("Not ready for this type of facet.");
          break;
      }
      for (std::size_t i = 0; i < gdim; ++i)
        normal[i] /= facet_dets[idx];
      // If normal is pointing to inside the el, swap it
      U dot = 0;
      for (std::size_t i = 0; i < gdim; ++i)
        dot += normal[i] * (facet_centroid[i] - cell_centroid_i[i]);
      if (dot < 0) {
        for (std::size_t i = 0; i < gdim; ++i)
          normal[i] = -normal[i];
      }

      for (size_t k = 0; k < num_gps_facet; ++k) {//igauss
        // SPARSITY PATTERN
        size_t igp = idx*num_gps_facet + k;
        int icell_j = renumbering_cells_po_mesh_j[igp];
        // Dofs j
        auto gdofs_j = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          dofs_cells_mesh_j, icell_j, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        // Update sparsity pattern
        sp.insert(facet_dofs_i, std::span(gdofs_j.data_handle(), gdofs_j.extent(0)));

        // Store all ref gps j positions
        auto point = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          tabulated_gauss_points_gamma, igp, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        // Pullback GP assuming affine
        cmdspan2_t point_as_matrix(point.data_handle(), 1, tdim);
        MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            U, MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                   std::size_t, 1, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
            Xp(_gauss_points_ref_j.data()+tdim*igp, 1, tdim);
        auto _K_j = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K_j, icell_j, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        cmdspan2_t coord_dofs_j(_geoms_cells_mesh_j.data() + icell_j * (num_dofs_g_j * 3), num_dofs_g_j, 3);
        std::array<U, 3> x0_j = {0, 0, 0};
        for (std::size_t i = 0; i < coord_dofs_j.extent(1); ++i)
          x0_j[i] += coord_dofs_j(0, i);
        dolfinx::fem::CoordinateElement<U>::pull_back_affine(Xp, _K_j, x0_j, point_as_matrix);

      }
    }

    // Pre-allocate matrix
    sp.finalize();
    custom_create_matrix(A, sp.comm(), sp);

    // Precompute values
    if (num_gps_processor>0) {
      cmdspan2_t gauss_points_ref_j(_gauss_points_ref_j.data(), num_gps_processor, tdim);
      element_j.tabulate(_nderivative_j, gauss_points_ref_j, phi_j);

      // Loop 2: precompute gradients
      std::fill(dphi_j_b.begin(), dphi_j_b.end(), U(0.0));
      for (size_t idx = 0; idx < num_gamma_facets; ++idx) {
        for (size_t k = 0; k < num_gps_facet; ++k) {//igauss
          size_t igp = idx*num_gps_facet + k;
          int icell_j = renumbering_cells_po_mesh_j[igp];
          auto _K_j = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            K_j, icell_j, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

          auto dphi_ref_j = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              phi_j, std::pair(1, tdim + 1), igp,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
          auto dphi_j = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              dphi_j_all_gps, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, igp,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          // Aplly transformation on gradient
          for (std::size_t i = 0; i < _K_j.extent(1); i++)
            for (std::size_t j = 0; j < dphi_ref_j.extent(2); j++)
              for (std::size_t k = 0; k < _K_j.extent(0); k++)
                dphi_j(i,j) += _K_j(k,i) * dphi_ref_j(k,j);
        }
      }
    }
  }

  void assemble(
      const std::function<int(const std::int64_t,
                              const std::int64_t,
                              const double)>& mat_add,
      std::reference_wrapper<const dolfinx::fem::Function<T>> ext_conductivity,
      std::span<const T> _tabulated_gauss_points_gamma,
      std::span<const T> _gauss_points_cell,
      std::span<const T> gweights_facet,
      const dolfinx::fem::FunctionSpace<double>& V_i,
      const multiphenicsx::fem::DofMapRestriction& restriction_i,
      const dolfinx::fem::FunctionSpace<double>& V_j,
      const multiphenicsx::fem::DofMapRestriction& restriction_j,
      std::span<const std::int32_t> gamma_integration_data_i,
      std::span<const int> renumbering_cells_po_mesh_j,
      std::span<const std::int64_t> _dofs_cells_mesh_j)
  {

    // Dictionary cell type to facet cell type
    auto index_map_i = restriction_i.index_map;
    auto index_map_j = restriction_j.index_map;

    auto mesh_i = V_i.mesh();
    auto element_i = V_i.element()->basix_element();
    auto element_j = V_j.element()->basix_element();

    const std::size_t tdim = mesh_i->topology()->dim();
    const std::size_t gdim = mesh_i->geometry().dim();
    std::shared_ptr<const common::IndexMap> cmap = mesh_i->topology()->index_map(tdim);
    auto con_cf_i = mesh_i->topology()->connectivity(tdim,tdim-1);
    assert(con_cf_i);

    auto fcell_type_i = basix::cell::sub_entity_type(element_i.cell_type(), tdim-1, 0);
    size_t num_facets_cell = basix::cell::num_sub_entities(element_i.cell_type(), tdim-1);
    size_t num_gps_facet = gweights_facet.size();
    size_t num_gps_cell = num_gps_facet * num_facets_cell;

    assert(_tabulated_gauss_points_gamma.size() % 3 == 0);
    size_t num_gps_processor = _tabulated_gauss_points_gamma.size() / 3;
    cmdspan2_t tabulated_gauss_points_gamma(_tabulated_gauss_points_gamma.data(), num_gps_processor, 3);
    cmdspan2_t gauss_points_cell(_gauss_points_cell.data(), num_gps_cell, tdim);
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x_dofmap_i = mesh_i->geometry().dofmap();
    const size_t num_dofs_g_i = x_dofmap_i.extent(1);
    std::span<const U> x_g_i = mesh_i->geometry().x();
    const size_t num_dofs_g_j = V_j.mesh()->geometry().dofmap().extent(1);

    auto con_v_i = restriction_i.dofmap()->map();
    auto con_v_j = restriction_j.dofmap()->map();
    auto unrestricted_to_restricted_i = restriction_i.unrestricted_to_restricted();
    auto unrestricted_to_restricted_j = restriction_j.unrestricted_to_restricted();
    size_t num_dofs_cell_i = con_v_i.extent(1);
    size_t num_dofs_cell_j = con_v_j.extent(1);
    const size_t num_dofs_facet_i = basix::cell::num_sub_entities(fcell_type_i, 0);
    auto sub_entity_connectivity = basix::cell::sub_entity_connectivity(V_i.element()->basix_element().cell_type());

    size_t num_diff_cells_j = *std::max_element(renumbering_cells_po_mesh_j.begin(),
                                                renumbering_cells_po_mesh_j.end());
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const std::int64_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>> dofs_cells_mesh_j(
          _dofs_cells_mesh_j.data(), num_diff_cells_j, num_dofs_cell_j);

    // Create buffer for local contribution
    std::vector<int> facet_dofs_i(num_dofs_facet_i);
    std::vector<std::int64_t> gfacet_dofs_i(num_dofs_facet_i);
    std::vector<std::int64_t> gdofs_j(num_dofs_cell_j);

    // Prepare mesh_j integration data
    mdspan4_t phi_j(phi_j_b.data(), e_shape_j);
    mdspan3_t dphi_j(dphi_j_b.data(), tdim, e_shape_j[1], e_shape_j[2]);

    auto sub_cell_con_i = basix::cell::sub_entity_connectivity(element_i.cell_type());

    assert(gamma_integration_data_i.size() % 2 == 0);
    size_t num_gamma_facets = gamma_integration_data_i.size() / 2;
    assert(num_gps_facet*num_gamma_facets == num_gps_processor);
    auto e_shape_i = element_i.tabulate_shape(_nderivative_i, num_gps_cell);
    mdspan4_t phi_i(phi_i_b.data(), e_shape_i);

    auto ext_k_arr = ext_conductivity.get().x()->array();

    // Estimate total number of contributions
    std::vector<Triplet> contribs;
    contribs.reserve(num_gps_processor * num_dofs_facet_i * num_dofs_cell_j);
    for (size_t idx = 0; idx < num_gamma_facets; ++idx) {
      std::int32_t icell_i = gamma_integration_data_i[2*idx];
      std::int32_t lifacet_i = gamma_integration_data_i[2*idx+1];
      // relevant dofs are sub_entity_connectivity[fdim][lifacet_i][0]
      const auto& lfacet_dofs = sub_cell_con_i[tdim-1][lifacet_i][0];
      // DOFS i
      auto dofs_cell_i = restriction_i.cell_dofs(icell_i);
      auto udofs_cell_i = std::span(con_v_i.data_handle() + icell_i * num_dofs_cell_i, num_dofs_cell_i);
      std::transform(lfacet_dofs.begin(), lfacet_dofs.end(), facet_dofs_i.begin(),
                     [&dofs_cell_i](size_t index) {return dofs_cell_i[index];});
      restriction_i.index_map->local_to_global(facet_dofs_i, gfacet_dofs_i);
      std::span<U> normal(facet_normals.data()+3*idx, 3);
      for (size_t k = 0; k < num_gps_facet; ++k) {//igauss
        size_t igp = idx*num_gps_facet + k;
        int icell_j = renumbering_cells_po_mesh_j[igp];
        size_t idx_gp_i = lifacet_i*num_gps_facet + k;
        // DOFS j
        auto gdofs_j = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          dofs_cells_mesh_j, icell_j, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

        // Concatenate triplets with contribs
        double v;
        for (size_t i = 0; i < num_dofs_facet_i; ++i) {
          for (size_t j = 0; j < num_dofs_cell_j; ++j) {
            v = 0.0;
            // TODO Consider adding Robin coeff
            v -= phi_j(0, igp, j, 0);
            // dphi: dim, gp, basis func
            double dot_gradj_n = 0.0;
            for (int w = 0; w < tdim; ++w)
              dot_gradj_n += normal[w] * dphi_j(w,igp,j);
            v -= ext_k_arr[icell_i] * dot_gradj_n;
            v *= phi_i(0,
                       idx_gp_i,
                       lfacet_dofs[i],
                       0);
            v *= gweights_facet[k];
            v *= facet_dets[idx];
            contribs.push_back( Triplet(gfacet_dofs_i[i], gdofs_j[j], v) );
          }
        }
      }
    }

    // Assemble triplets
    for (Triplet& t: contribs) {
      mat_add(t.row, t.col, t.value);
    }
  }
  private:
  std::vector<T> phi_i_b;
  std::vector<T> phi_j_b;
  std::vector<T> dphi_j_b;
  std::vector<T> facet_normals;
  std::vector<T> facet_dets;
  int _nderivative_i = 0;
  int _nderivative_j = 1;
  std::array<std::size_t, 4> e_shape_j;
};
