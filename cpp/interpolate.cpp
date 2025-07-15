#include <cstdio>
#include <cassert>
#include <dolfinx/fem/interpolate.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/geometry/utils.h>
#include <vector>
#include <dolfinx/la/Vector.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include "diffmesh_utils.h"
#include "my_eval_affine.h"

template <std::floating_point T>
void interpolate_dg0_at_facets(std::vector<std::reference_wrapper<const dolfinx::fem::Function<T>>> &sending_fs,
                               std::vector<std::shared_ptr<dolfinx::fem::Function<T>>> &receiving_fs,
                               const dolfinx::fem::Function<T> &receiving_active_els_f,
                               const dolfinx::mesh::MeshTags<std::int8_t> &facet_tag,
                               std::span<const std::int32_t> incident_cells,
                               dolfinx::geometry::PointOwnershipData<T> &po,
                               const dolfinx::common::IndexMap &gamma_index_map,
                               std::span<const std::int32_t> gamma_imap_to_global_imap)
{
  /*
   * Facets are local facets
   */
  //TODO: Add more asserts
  const size_t num_funs = sending_fs.size();
  assert(num_funs == receiving_fs.size());
  std::shared_ptr smesh = sending_fs[0].get().function_space()->mesh();//sending mesh
  std::shared_ptr rmesh = receiving_fs[0]->function_space()->mesh();//receiving mesh
  int cdim = rmesh->topology()->dim();

  auto& dest_ranks = po.src_owner;
  auto& src_ranks = po.dest_owners;
  auto& recv_points = po.dest_points;
  auto& evaluation_cells = po.dest_cells;

  size_t* block_sizes = new size_t[num_funs];
  std::vector<dolfinx::la::Vector<T>> interpolated_vals_vectors;
  std::vector<std::vector<T>> send_values;
  for (int ifun = 0; ifun < num_funs; ++ifun) {
    assert(sending_fs[ifun].get().function_space()->mesh()==smesh);
    assert(receiving_fs[ifun]->function_space()->mesh()==rmesh);
    block_sizes[ifun] = sending_fs[ifun].get().function_space()->element()->value_size();
    assert(block_sizes[ifun] == receiving_fs[ifun]->function_space()->element()->value_size());
    interpolated_vals_vectors.push_back(dolfinx::la::Vector<T>(
          std::shared_ptr<const dolfinx::common::IndexMap>(&gamma_index_map,
            [](const dolfinx::common::IndexMap*){}),
          block_sizes[ifun])
        );
    send_values.push_back(std::vector<T>(recv_points.size() / 3 * block_sizes[ifun]));
  }

  // 1. Eval my points
  // Code copied from dolfinx interpolate.h
  // Evaluate the interpolating function where possible
  for (size_t ifun = 0; ifun < num_funs; ++ifun ) {
    auto s_x = sending_fs[ifun].get().x()->array();
    for (int icell = 0; icell < evaluation_cells.size(); ++icell) {
      for (int i = 0; i < block_sizes[ifun]; ++i) {
        send_values[ifun][icell*block_sizes[ifun]+i] = s_x[evaluation_cells[icell]*block_sizes[ifun]+i];
      }
    }
  }


  // 2. Call dolfinx scatter
  // Send values back to owning process
  using dextents2 = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;
  std::vector<std::vector<T>> values_b;
  for (size_t ifun = 0; ifun < num_funs; ++ifun) {
    values_b.push_back(std::vector<T>(dest_ranks.size() * block_sizes[ifun]));
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const T, dextents2> _send_values(
        send_values[ifun].data(), src_ranks.size(), block_sizes[ifun]);
    dolfinx::fem::impl::scatter_values(rmesh->comm(), src_ranks, dest_ranks, _send_values,
                         std::span(values_b[ifun]));
    // Transpose received data
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const T, dextents2> values(
        values_b[ifun].data(), dest_ranks.size(), block_sizes[ifun]);
    std::vector<T> valuesT_b(block_sizes[ifun] * dest_ranks.size());
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T, dextents2> valuesT(
        valuesT_b.data(), block_sizes[ifun], dest_ranks.size());
    for (std::size_t i = 0; i < values.extent(0); ++i)
      for (std::size_t j = 0; j < values.extent(1); ++j)
        valuesT(j, i) = values(i, j);
  }

  // 3. Insert vals into receiving_f
  auto con_cell_facet = rmesh->topology()->connectivity(cdim,cdim-1);
  assert(con_cell_facet);
  auto rfacet_map = rmesh->topology()->index_map(cdim-1);
  std::int32_t rnum_facets = rfacet_map->size_local();

  std::span<const std::int8_t> facet_tag_vals = facet_tag.values();

  // Build facet indices (local and ghost)
  std::size_t n = std::count_if(facet_tag_vals.begin(), facet_tag_vals.end(), [](std::int32_t i){return i>0;});
  std::vector<std::int32_t> facet_indices;
  facet_indices.reserve(n);
  for (std::int32_t i = 0; i < facet_tag_vals.size(); ++i)
  {
    if (facet_tag_vals[i] > 0)
      facet_indices.push_back(i);//dirty
  }

  // TODO: Clean this up
  for (size_t ifun = 0; ifun < num_funs; ++ifun) {
    auto interpolated_vals_array = interpolated_vals_vectors[ifun].mutable_array();
    for (int i = 0; i < facet_indices.size(); ++i) {
      int ifacet = facet_indices[i];
      if (ifacet >= rnum_facets)
        continue;
      for (int j = 0; j < block_sizes[ifun]; ++j) {
        interpolated_vals_array[i*block_sizes[ifun]+j] = values_b[ifun][i*block_sizes[ifun]+j];
      }
    }
    // Ghost communication
    interpolated_vals_vectors[ifun].scatter_fwd();
  }

  // Invert gamma_imap_to_global_imap
  std::unordered_map<std::int32_t, std::int32_t> global_imap_to_gamma_imap(gamma_imap_to_global_imap.size());
  for (int i = 0; i < gamma_imap_to_global_imap.size(); ++i)
    global_imap_to_gamma_imap[gamma_imap_to_global_imap[i]] = i;

  for (int i = 0; i < incident_cells.size(); ++i) {
    int icell = incident_cells[i];
    auto local_con = con_cell_facet->links(icell);
    auto it = std::find_if(local_con.begin(),local_con.end(),[&facet_tag_vals](std::int32_t ifacet){return (facet_tag_vals[ifacet]>0);});
    assert(it != std::end(local_con));
    std::int32_t ifacet = *it;
    std::int32_t ifacet_gamma_imap  = global_imap_to_gamma_imap[ifacet];
    for (size_t ifun = 0; ifun < num_funs; ++ifun) {
      auto r_x = receiving_fs[ifun]->x()->mutable_array();
      auto interpolated_vals_array = interpolated_vals_vectors[ifun].mutable_array();
      for (int j = 0; j < block_sizes[ifun]; ++j) {
        r_x[icell*block_sizes[ifun]+j] = interpolated_vals_array[ifacet_gamma_imap*block_sizes[ifun]+j];
      }
    }
  }
  delete[] block_sizes;
}

template <std::floating_point T>
void propagate_dg0_at_facets_same_mesh(const dolfinx::fem::Function<T> &sending_f,
                                       dolfinx::fem::Function<T> &receiving_f,
                                       const dolfinx::fem::Function<T> &sending_active_els_f,
                                       const dolfinx::fem::Function<T> &receiving_active_els_f,
                                       const dolfinx::common::IndexMap &gamma_index_map,
                                       std::span<const std::int32_t> gamma_imap_to_global_imap)
{
  std::shared_ptr smesh = sending_f.function_space()->mesh();//sending mesh
  std::shared_ptr rmesh = receiving_f.function_space()->mesh();//receiving mesh
  assert(smesh == rmesh);
  int sbsize = sending_f.function_space().get()->element()->value_size();
  assert(sbsize == receiving_f.function_space().get()->element()->value_size());
  auto vals = dolfinx::la::Vector<T>(std::shared_ptr<const dolfinx::common::IndexMap>(&gamma_index_map, [](const dolfinx::common::IndexMap*){}), sbsize);
  auto vals_arr = vals.mutable_array();
  int cdim = smesh->topology()->dim();
  auto cell_map = smesh->topology()->index_map(cdim);
  assert(cdim == rmesh->topology()->dim());
  auto sactive_els_arr = sending_active_els_f.x()->array();
  auto sending_f_arr = sending_f.x()->array();
  // Pack
  auto con_facet_cell = smesh->topology()->connectivity(cdim-1,cdim);
  for (int i = 0; i < gamma_index_map.size_local(); ++i) {
    std::int32_t ifacet = gamma_imap_to_global_imap[i];
    auto incident_cells = con_facet_cell->links(ifacet);
    int owner_el = -1;
    for (auto &ielem : incident_cells) {
      if (sactive_els_arr[ielem]) {
        owner_el = ielem;
        break;
      }
    }
    assert(owner_el > -1);
    for (int j = 0; j < sbsize; ++j) {
      vals_arr[i*sbsize+j] = sending_f_arr[owner_el*sbsize + j];
    }
  }
  // Communicate accross ranks
  vals.scatter_fwd();
  // Unpack
  int num_gamma_facets_processor = (gamma_index_map.size_local() + gamma_index_map.num_ghosts());
  auto receiving_f_marr = receiving_f.x()->mutable_array();
  auto ractive_els_arr = receiving_active_els_f.x()->array();
  for (int i = 0; i < num_gamma_facets_processor; ++i) {
    std::int32_t ifacet = gamma_imap_to_global_imap[i];
    auto incident_cells = con_facet_cell->links(ifacet);
    int owner_el = -1;
    for (auto &ielem : incident_cells) {
      if ((ractive_els_arr[ielem]) and (ielem < cell_map->size_local())) {
        owner_el = ielem;
        break;
      }
    }
    if (owner_el == -1)
      continue;
    for (int j = 0; j < sbsize; ++j)
      receiving_f_marr[owner_el*sbsize + j] = vals_arr[i*sbsize+j];
  }
  receiving_f.x()->scatter_fwd();
}

template <std::floating_point T>
void interpolate_cg1_affine(const dolfinx::fem::Function<T> &sending_f,
                            dolfinx::fem::Function<T> &receiving_f,
                            std::span<const std::int32_t> sending_cells,
                            std::span<const std::int32_t> dofs_to_interpolate,
                            std::span<const T> _coords_dofs_to_interpolate,
                            T padding = 1e-7)
{
  std::shared_ptr smesh = sending_f.function_space()->mesh();//sending mesh
  std::shared_ptr rmesh = receiving_f.function_space()->mesh();//receiving mesh
  MPI_Comm comm = smesh->comm();
  {
    int result;
    MPI_Comm_compare(comm, rmesh->comm(), &result);
    if (result == MPI_UNEQUAL)
    {
      throw std::runtime_error("Interpolation on different meshes is only "
                               "supported on the same communicator.");
    }
  }
  assert(_coords_dofs_to_interpolate.size() % 3 == 0);
  size_t num_dofs_processor = _coords_dofs_to_interpolate.size() / 3;
  assert(num_dofs_processor == dofs_to_interpolate.size());

  dolfinx::geometry::PointOwnershipData<T> interpolation_data =
    determine_point_ownership(*smesh,
                              _coords_dofs_to_interpolate,
                              sending_cells,
                              padding);

  const std::vector<int>& dest_ranks = interpolation_data.src_owner;
  const std::vector<int>& src_ranks = interpolation_data.dest_owners;
  const std::vector<T>& recv_points = interpolation_data.dest_points;
  const std::vector<std::int32_t>& evaluation_cells
      = interpolation_data.dest_cells;

  const std::size_t sbsize = sending_f.function_space()->element()->value_size();
  assert(sbsize == receiving_f.function_space().get()->element()->value_size());
  // Evaluate the interpolating function where possible
  std::vector<T> send_values(recv_points.size() / 3 * sbsize);
  eval_affine<T,T>(sending_f, recv_points, {recv_points.size() / 3, (std::size_t)3},
      evaluation_cells, send_values, {recv_points.size() / 3, sbsize});

  using dextents2 = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;

  // Send values back to owning process
  std::vector<T> values_b(dest_ranks.size() * sbsize);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const T, dextents2> _send_values(
      send_values.data(), src_ranks.size(), sbsize);
  scatter_values(comm, src_ranks, dest_ranks, _send_values,
      std::span(values_b));

  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const T, dextents2> values(
      values_b.data(), dest_ranks.size(), sbsize);

  // Insert vals into receiving_f
  auto rvals = receiving_f.x()->mutable_array();
  for (int i = 0; i < num_dofs_processor; ++i) {
    std::int32_t dof = dofs_to_interpolate[i];
    for (int j = 0; j < sbsize; ++j)
      rvals[dof*sbsize + j] = values(i,j);
  }
  receiving_f.x()->scatter_fwd();
}

template <std::floating_point T>
void interpolate_dg0(const dolfinx::fem::Function<T> &sending_f,
                    dolfinx::fem::Function<T> &receiving_f,
                    std::span<const std::int32_t> sending_cells,
                    std::span<const std::int32_t> receiving_cells,
                    T padding = 1e-7) {
    std::shared_ptr smesh = sending_f.function_space()->mesh();//sending mesh
    std::shared_ptr rmesh = receiving_f.function_space()->mesh();//receiving mesh
    MPI_Comm comm = smesh->comm();
    {
      int result;
      MPI_Comm_compare(comm, rmesh->comm(), &result);
      if (result == MPI_UNEQUAL)
      {
        throw std::runtime_error("Interpolation on different meshes is only "
                                 "supported on the same communicator.");
      }
    }
    // Compute midpoints of rcells
    int cdim = rmesh->topology()->dim();
    const std::vector<T> midpoints = dolfinx::mesh::compute_midpoints(*rmesh, cdim, receiving_cells);
    // Point ownership with midpoints and scells
    dolfinx::geometry::PointOwnershipData<T> interpolation_data =
      determine_point_ownership(*smesh,
                                std::span<const T>(midpoints.data(), midpoints.size()),
                                sending_cells,
                                padding);
    // Gather sf info at cells
    auto s_x = sending_f.x()->array();
    std::vector<T> rvals(interpolation_data.src_owner.size());
    std::vector<T> _svals(interpolation_data.dest_cells.size());
    std::transform(interpolation_data.dest_cells.begin(), interpolation_data.dest_cells.end(),
        _svals.begin(),
        [s_x](auto& icell){ return s_x[icell]; });
    using dextents2 = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const T, dextents2> svals(
        _svals.data(), _svals.size(), 1);
    scatter_values(comm,
        interpolation_data.dest_owners,
        interpolation_data.src_owner,
        svals,
        std::span(rvals));
    // Unpack info
    auto r_x = receiving_f.x()->mutable_array();
    for (size_t i = 0; i < rvals.size(); ++i)
      r_x[receiving_cells[i]] = rvals[i];
    receiving_f.x()->scatter_fwd();
}


namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_interpolate(nb::module_ &m) {
  m.def(
      "interpolate_dg0_at_facets",
      [](
         std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> ptrs_sending_fs,
         std::vector<std::shared_ptr<dolfinx::fem::Function<T>>> ptrs_receiving_fs,
         const dolfinx::fem::Function<T> &receiving_active_els_f,
         const dolfinx::mesh::MeshTags<std::int8_t> &facet_tag,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
         dolfinx::geometry::PointOwnershipData<T> &po,
         const dolfinx::common::IndexMap &gamma_index_map,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> gamma_imap_to_global_imap)
      {
        std::vector<std::reference_wrapper<const dolfinx::fem::Function<T>>> refs_sending_fs;
        for (auto &sf : ptrs_sending_fs) {
          refs_sending_fs.push_back(*sf);
        }
        return interpolate_dg0_at_facets<T>(refs_sending_fs,
                                            ptrs_receiving_fs,
                                            receiving_active_els_f,
                                            facet_tag,
                                            std::span(cells.data(),cells.size()),
                                            po,
                                            gamma_index_map,
                                            std::span(gamma_imap_to_global_imap.data(), gamma_imap_to_global_imap.size()));
      }
      );
  m.def(
      "propagate_dg0_at_facets_same_mesh",
      [](
        const dolfinx::fem::Function<T> &sending_f,
        dolfinx::fem::Function<T> &receiving_f,
        const dolfinx::fem::Function<T> &sending_active_els_f,
        const dolfinx::fem::Function<T> &receiving_active_els_f,
        const dolfinx::common::IndexMap &gamma_index_map,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> gamma_imap_to_global_imap
      ) {
        return propagate_dg0_at_facets_same_mesh<T>(sending_f,
                                                    receiving_f,
                                                    sending_active_els_f,
                                                    receiving_active_els_f,
                                                    gamma_index_map,
                                                    std::span(gamma_imap_to_global_imap.data(), gamma_imap_to_global_imap.size()));
      }
      );
  m.def(
      "interpolate_cg1_affine",
      [](
        const dolfinx::fem::Function<T> &sending_f,
        dolfinx::fem::Function<T> &receiving_f,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> sending_cells,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> dofs_to_interpolate,
        nb::ndarray<const T, nb::ndim<2>, nb::c_contig> coords_dofs_to_interpolate,
        T padding
      ) {
        interpolate_cg1_affine<T>(sending_f,
            receiving_f,
            std::span(sending_cells.data(), sending_cells.size()),
            std::span(dofs_to_interpolate.data(), dofs_to_interpolate.size()),
            std::span(coords_dofs_to_interpolate.data(), coords_dofs_to_interpolate.size()),
            padding);
      }
      );
  m.def(
      "interpolate_dg0",
      [](
        const dolfinx::fem::Function<T> &sending_f,
        dolfinx::fem::Function<T> &receiving_f,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> sending_cells,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> receiving_cells,
        T padding
      ) {
        interpolate_dg0<T>(sending_f,
            receiving_f,
            std::span(sending_cells.data(), sending_cells.size()),
            std::span(receiving_cells.data(), receiving_cells.size()),
            padding);
      }
      );
}

void declare_interpolate(nb::module_ &m) {
  templated_declare_interpolate<double>(m);
  templated_declare_interpolate<float>(m);
}
