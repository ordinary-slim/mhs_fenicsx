#pragma once
#include <dolfinx/fem/assemble_matrix_impl.h>
#include <basix/quadrature.h>
#include <basix/cell.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/petsc.h>
#include <multiphenicsx/DofMapRestriction.h>
#include <petscsystypes.h>

using bct = basix::cell::type;
using U = double;

using cmdspan2_i = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const std::int32_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

struct Triplet
{
  size_t row, col;
  double value;
  Triplet(size_t row_, size_t col_, double value_) :
  row(row_), col(col_), value(value_) {
  }
};

template <dolfinx::scalar T>
Mat create_robin_robin_monolithic(
    MPI_Comm comm, const dolfinx::fem::FunctionSpace<double>& Qs_gamma,
    const dolfinx::fem::FunctionSpace<double>& V_i,
    const multiphenicsx::fem::DofMapRestriction& restriction_i,
    const dolfinx::fem::FunctionSpace<double>& V_j,
    const multiphenicsx::fem::DofMapRestriction& restriction_j,
    std::span<const std::int32_t> facets_mesh_i,
    std::span<const std::int32_t> cells_mesh_i,
    const geometry::PointOwnershipData<U>& po_mesh_j,
    std::span<const std::int32_t> cells_mesh_j,
    std::span<const T> gweights_facet)
{

  // Dictionary cell type to facet cell type
  std::map<bct, bct> bcell_type {
    {bct::interval, bct::point},
    {bct::triangle, bct::interval},
    {bct::quadrilateral, bct::interval},
    {bct::tetrahedron, bct::triangle},
    {bct::hexahedron, bct::quadrilateral}
  };

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

  auto index_map_i = restriction_i.index_map;
  auto index_map_j = restriction_j.index_map;
  int bs_i = restriction_i.index_map_bs();
  int bs_j = restriction_j.index_map_bs();
  PetscErrorCode ierr;
  Mat A;
  ierr = MatCreate(comm, &A);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatCreate");

  // Get global and local dimensions
  const std::int64_t M = bs_i * index_map_i->size_global();
  const std::int64_t N = bs_j * index_map_j->size_global();
  const std::int32_t m = bs_i * index_map_i->size_local();
  const std::int32_t n = bs_j * index_map_j->size_local();

  // Set matrix size
  ierr = MatSetSizes(A, m, n, M, N);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatSetSizes");

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(A);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatSetFromOptions");

  auto mesh = V_i.mesh();
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  std::shared_ptr<const common::IndexMap> cmap = mesh->topology()->index_map(tdim);
  auto con_cf_i = mesh->topology()->connectivity(tdim,tdim-1);
  assert(con_cf_i);

  cmdspan2_i dofmap_qs_gamma = Qs_gamma.dofmap()->map();
  auto con_v_i = restriction_i.dofmap()->map();
  auto con_v_j = restriction_j.dofmap()->map();
  size_t num_dofs_cell_i = con_v_i.extent(1);
  size_t num_dofs_cell_j = con_v_j.extent(1);
  auto fcell_type_i = bcell_type[V_i.element()->basix_element().cell_type()];
  const size_t num_dofs_facet_i = basix::cell::num_sub_entities(fcell_type_i, 0);
  auto sub_entity_connectivity = basix::cell::sub_entity_connectivity(V_i.element()->basix_element().cell_type());

  // Create buffer for local contribution
  const size_t num_entries_locmat = num_dofs_facet_i*num_dofs_cell_j;
  std::vector<T> A_eb(num_entries_locmat);
  mdspan2_t A_e(A_eb.data(), num_dofs_facet_i, num_dofs_cell_j);
  std::vector<int> facet_dofs_i(num_dofs_facet_i);
  std::vector<std::int64_t> gfacet_dofs_i(num_dofs_facet_i);
  std::vector<std::int64_t> gdofs_j(num_dofs_cell_j);

  // Estimate total number of contributions
  size_t total_num_gps = Qs_gamma.dofmap()->index_map->size_local();
  std::vector<Triplet> contribs;
  contribs.reserve(total_num_gps * num_dofs_facet_i * num_dofs_cell_j);

  auto element_i = V_i.element();
  auto sub_cell_con_i = basix::cell::sub_entity_connectivity(element_i->basix_element().cell_type());
  for (int idx = 0; idx < facets_mesh_i.size(); ++idx) {
    int ifacet_i = facets_mesh_i[idx];
    int icell_i = cells_mesh_i[idx];
    int icell_j = cells_mesh_j[idx];
    // Find local index of facet
    auto loc_con_cf_i = con_cf_i->links(icell_i);
    auto itr = std::find(loc_con_cf_i.begin(),
                         loc_con_cf_i.end(),
                         ifacet_i);
    assert(itr!=loc_con_cf_i.end());
    size_t lifacet_i = std::distance(loc_con_cf_i.begin(), itr);
    // relevant dofs are sub_entity_connectivity[fdim][lifacet_i][0]
    const auto& lfacet_dofs = sub_cell_con_i[tdim-1][lifacet_i][0];

    // Compute local contribution
    auto gp_indices = Qs_gamma.dofmap()->cell_dofs(idx);
    std::fill(A_eb.begin(), A_eb.end(), 1.0);
    for (size_t k = 0; k < gp_indices.size(); ++k) {//igauss
      size_t igp = gp_indices[k];
      int rank_j = po_mesh_j.src_owner[igp];
      int icell_j = cells_mesh_j[igp];
    }

    // DOFS i
    auto dofs_i = restriction_i.cell_dofs(icell_i);
    std::transform(lfacet_dofs.begin(), lfacet_dofs.end(), facet_dofs_i.begin(),
                   [&dofs_i](size_t index) { return dofs_i[index]; });
    // DOFS j
    auto dofs_j = restriction_j.cell_dofs(icell_j);

    printf("cell %i, local facet idx %li:\n", icell_i, lifacet_i);
    for (size_t idx = 0; idx < lfacet_dofs.size(); ++idx) {
      printf("(%i, %i), ", lfacet_dofs[idx], facet_dofs_i[idx]);
    }
    printf("\n");
    // Global DOFS
    restriction_i.index_map->local_to_global(facet_dofs_i, gfacet_dofs_i);
    restriction_j.index_map->local_to_global(dofs_j, gdofs_j);

    // Concatenate triplets with contribs
    for (size_t i = 0; i < A_e.extent(0); ++i) {
      for (size_t j = 0; j < A_e.extent(1); ++j) {
        contribs.push_back( Triplet(gfacet_dofs_i[i], gdofs_j[j], A_e(i,j)) );
      }
    }
  }

  // Assemble triplets
  for (Triplet& t: contribs) {
    printf("(%li, %li, %g)\n", t.row, t.col, t.value);
    MatSetValue(A, t.row, t.col, t.value, ADD_VALUES);
  }

  return A;
}
