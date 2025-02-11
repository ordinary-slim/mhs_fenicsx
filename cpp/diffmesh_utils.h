#pragma once
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/geometry/utils.h>
#include <multiphenicsx/DofMapRestriction.h>

/// COPIED FROM DOLFINX CODE
/// dolfinx::scalar concept too restrictive
template <typename T, std::size_t D>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, D>>;
/// @brief Scatter data into non-contiguous memory.
///
/// Scatter blocked data `send_values` to its corresponding `src_rank`
/// and insert the data into `recv_values`. The insert location in
/// `recv_values` is determined by `dest_ranks`. If the j-th dest rank
/// is -1, then `recv_values[j*block_size:(j+1)*block_size]) = 0`.
///
/// @param[in] comm The MPI communicator.
/// @param[in] src_ranks Rank owning the values of each row in
/// `send_values`.
/// @param[in] dest_ranks List of ranks receiving data. Size of array is
/// how many values we are receiving (not unrolled for block_size).
/// @param[in] send_values Values to send back to owner. Shape is
/// `(src_ranks.size(), block_size)`.
/// @param[in,out] recv_values Array to fill with values.  Shape
/// `(dest_ranks.size(), block_size)`. Storage is row-major.
/// @pre It is required that src_ranks are sorted.
/// @note `dest_ranks` can contain repeated entries.
/// @note `dest_ranks` might contain -1 (no process owns the point).
template <typename T>
void scatter_values(MPI_Comm comm, std::span<const std::int32_t> src_ranks,
                    std::span<const std::int32_t> dest_ranks,
                    mdspan_t<const T, 2> send_values, std::span<T> recv_values)
{
  const std::size_t block_size = send_values.extent(1);
  assert(src_ranks.size() * block_size == send_values.size());
  assert(recv_values.size() == dest_ranks.size() * block_size);

  // Build unique set of the sorted src_ranks
  std::vector<std::int32_t> out_ranks(src_ranks.size());
  out_ranks.assign(src_ranks.begin(), src_ranks.end());
  auto [unique_end, range_end] = std::ranges::unique(out_ranks);
  out_ranks.erase(unique_end, range_end);
  out_ranks.reserve(out_ranks.size() + 1);

  // Remove negative entries from dest_ranks
  std::vector<std::int32_t> in_ranks;
  in_ranks.reserve(dest_ranks.size());
  std::copy_if(dest_ranks.begin(), dest_ranks.end(),
               std::back_inserter(in_ranks),
               [](auto rank) { return rank >= 0; });

  // Create unique set of sorted in-ranks
  {
    std::ranges::sort(in_ranks);
    auto [unique_end, range_end] = std::ranges::unique(in_ranks);
    in_ranks.erase(unique_end, range_end);
  }
  in_ranks.reserve(in_ranks.size() + 1);

  // Create neighborhood communicator
  MPI_Comm reverse_comm;
  MPI_Dist_graph_create_adjacent(
      comm, in_ranks.size(), in_ranks.data(), MPI_UNWEIGHTED, out_ranks.size(),
      out_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &reverse_comm);

  std::vector<std::int32_t> comm_to_output;
  std::vector<std::int32_t> recv_sizes(in_ranks.size());
  recv_sizes.reserve(1);
  std::vector<std::int32_t> recv_offsets(in_ranks.size() + 1, 0);
  {
    // Build map from parent to neighborhood communicator ranks
    std::vector<std::pair<std::int32_t, std::int32_t>> rank_to_neighbor;
    rank_to_neighbor.reserve(in_ranks.size());
    for (std::size_t i = 0; i < in_ranks.size(); i++)
      rank_to_neighbor.push_back({in_ranks[i], i});
    std::ranges::sort(rank_to_neighbor);

    // Compute receive sizes
    std::ranges::for_each(
        dest_ranks,
        [&dest_ranks, &rank_to_neighbor, &recv_sizes, block_size](auto rank)
        {
          if (rank >= 0)
          {
            auto it = std::ranges::lower_bound(rank_to_neighbor, rank,
                                               std::ranges::less(),
                                               [](auto e) { return e.first; });
            assert(it != rank_to_neighbor.end() and it->first == rank);
            recv_sizes[it->second] += block_size;
          }
        });

    // Compute receiving offsets
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::next(recv_offsets.begin(), 1));

    // Compute map from receiving values to position in recv_values
    comm_to_output.resize(recv_offsets.back() / block_size);
    std::vector<std::int32_t> recv_counter(recv_sizes.size(), 0);
    for (std::size_t i = 0; i < dest_ranks.size(); ++i)
    {
      if (const std::int32_t rank = dest_ranks[i]; rank >= 0)
      {
        auto it = std::ranges::lower_bound(rank_to_neighbor, rank,
                                           std::ranges::less(),
                                           [](auto e) { return e.first; });
        assert(it != rank_to_neighbor.end() and it->first == rank);
        int insert_pos = recv_offsets[it->second] + recv_counter[it->second];
        comm_to_output[insert_pos / block_size] = i * block_size;
        recv_counter[it->second] += block_size;
      }
    }
  }

  std::vector<std::int32_t> send_sizes(out_ranks.size());
  send_sizes.reserve(1);
  {
    // Compute map from parent MPI rank to neighbor rank for outgoing
    // data. `out_ranks` is sorted, so rank_to_neighbor will be sorted
    // too.
    std::vector<std::pair<std::int32_t, std::int32_t>> rank_to_neighbor;
    rank_to_neighbor.reserve(out_ranks.size());
    for (std::size_t i = 0; i < out_ranks.size(); i++)
      rank_to_neighbor.push_back({out_ranks[i], i});

    // Compute send sizes. As `src_ranks` is sorted, we can move 'start'
    // in search forward.
    auto start = rank_to_neighbor.begin();
    std::ranges::for_each(
        src_ranks,
        [&rank_to_neighbor, &send_sizes, block_size, &start](auto rank)
        {
          auto it = std::ranges::lower_bound(start, rank_to_neighbor.end(),
                                             rank, std::ranges::less(),
                                             [](auto e) { return e.first; });
          assert(it != rank_to_neighbor.end() and it->first == rank);
          send_sizes[it->second] += block_size;
          start = it;
        });
  }

  // Compute sending offsets
  std::vector<std::int32_t> send_offsets(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_offsets.begin(), 1));

  // Send values to dest ranks
  std::vector<T> values(recv_offsets.back());
  values.reserve(1);
  MPI_Neighbor_alltoallv(send_values.data_handle(), send_sizes.data(),
                         send_offsets.data(), dolfinx::MPI::mpi_t<T>,
                         values.data(), recv_sizes.data(), recv_offsets.data(),
                         dolfinx::MPI::mpi_t<T>, reverse_comm);
  MPI_Comm_free(&reverse_comm);

  // Insert values received from neighborhood communicator in output
  // span
  std::ranges::fill(recv_values, T(0));
  for (std::size_t i = 0; i < comm_to_output.size(); i++)
  {
    auto vals = std::next(recv_values.begin(), comm_to_output[i]);
    auto vals_from = std::next(values.begin(), i * block_size);
    std::copy_n(vals_from, block_size, vals);
  }
};

template <std::floating_point T>
dolfinx::geometry::PointOwnershipData<T> my_determine_point_ownership(const dolfinx::mesh::Mesh<T>& mesh,
                                                                      std::span<const T> points,
                                                                      std::span<const std::int32_t> cells,
                                                                      T padding)
{
  MPI_Comm comm = mesh.comm();

  // Create a global bounding-box tree to find candidate processes with
  // cells that could collide with the points
  const int tdim = mesh.topology()->dim();
  auto cell_map = mesh.topology()->index_map(tdim);
  const std::int32_t num_cells = cell_map->size_local();
  // NOTE: Should we send the cells in as input?
  //std::vector<std::int32_t> cells(num_cells, 0);
  //std::iota(cells.begin(), cells.end(), 0);
  dolfinx::geometry::BoundingBoxTree bb(mesh, tdim, cells, padding);
  dolfinx::geometry::BoundingBoxTree global_bbtree = bb.create_global_tree(comm);

  // Compute collisions:
  // For each point in `points` get the processes it should be sent to
  dolfinx::graph::AdjacencyList collisions = compute_collisions(global_bbtree, points);

  // Get unique list of outgoing ranks
  std::vector<std::int32_t> out_ranks = collisions.array();
  std::sort(out_ranks.begin(), out_ranks.end());
  out_ranks.erase(std::unique(out_ranks.begin(), out_ranks.end()),
                  out_ranks.end());
  // Compute incoming edges (source processes)
  std::vector in_ranks = dolfinx::MPI::compute_graph_edges_nbx(comm, out_ranks);
  std::sort(in_ranks.begin(), in_ranks.end());

  // Create neighborhood communicator in forward direction
  MPI_Comm forward_comm;
  MPI_Dist_graph_create_adjacent(
      comm, in_ranks.size(), in_ranks.data(), MPI_UNWEIGHTED, out_ranks.size(),
      out_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &forward_comm);

  // Compute map from global mpi rank to neighbor rank, "collisions"
  // uses global rank
  std::map<std::int32_t, std::int32_t> rank_to_neighbor;
  for (std::size_t i = 0; i < out_ranks.size(); i++)
    rank_to_neighbor[out_ranks[i]] = i;

  // Count the number of points to send per neighbor process
  std::vector<std::int32_t> send_sizes(out_ranks.size());
  for (std::size_t i = 0; i < points.size() / 3; ++i)
    for (auto p : collisions.links(i))
      send_sizes[rank_to_neighbor[p]] += 3;

  // Compute receive sizes
  std::vector<std::int32_t> recv_sizes(in_ranks.size());
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Request sizes_request;
  MPI_Ineighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                         MPI_INT, forward_comm, &sizes_request);

  // Compute sending offsets
  std::vector<std::int32_t> send_offsets(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_offsets.begin(), 1));

  // Pack data to send and store unpack map
  std::vector<T> send_data(send_offsets.back());
  std::vector<std::int32_t> counter(send_sizes.size(), 0);
  // unpack map: [index in adj list][pos in x]
  std::vector<std::int32_t> unpack_map(send_offsets.back() / 3);
  for (std::size_t i = 0; i < points.size(); i += 3)
  {
    for (auto p : collisions.links(i / 3))
    {
      int neighbor = rank_to_neighbor[p];
      int pos = send_offsets[neighbor] + counter[neighbor];
      auto it = std::next(send_data.begin(), pos);
      std::copy_n(std::next(points.begin(), i), 3, it);
      unpack_map[pos / 3] = i / 3;
      counter[neighbor] += 3;
    }
  }

  MPI_Wait(&sizes_request, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_offsets(in_ranks.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_offsets.begin(), 1));

  std::vector<T> received_points((std::size_t)recv_offsets.back());
  MPI_Neighbor_alltoallv(
      send_data.data(), send_sizes.data(), send_offsets.data(),
      dolfinx::MPI::mpi_t<T>, received_points.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_t<T>, forward_comm);

  // Get mesh geometry for closest entity
  const dolfinx::mesh::Geometry<T>& geometry = mesh.geometry();
  std::span<const T> geom_dofs = geometry.x();
  auto x_dofmap = geometry.dofmap();

  // Compute candidate cells for collisions (and extrapolation)
  const dolfinx::graph::AdjacencyList<std::int32_t> candidate_collisions
      = compute_collisions(bb, std::span<const T>(received_points.data(),
                                                  received_points.size()));

  // Each process checks which points collide with a cell on the process
  const int rank = dolfinx::MPI::rank(comm);
  std::vector<std::int32_t> cell_indicator(received_points.size() / 3);
  std::vector<std::int32_t> closest_cells(received_points.size() / 3);
  for (std::size_t p = 0; p < received_points.size(); p += 3)
  {
    std::array<T, 3> point;
    std::copy_n(std::next(received_points.begin(), p), 3, point.begin());
    // Find first colliding cell among the cells with colliding bounding boxes
    const int colliding_cell = dolfinx::geometry::compute_first_colliding_cell(
        mesh, candidate_collisions.links(p / 3), point,
        10 * std::numeric_limits<T>::epsilon());
    // If a collding cell is found, store the rank of the current process
    // which will be sent back to the owner of the point
    cell_indicator[p / 3] = (colliding_cell >= 0) ? rank : -1;
    // Store the cell index for lookup once the owning processes has determined
    // the ownership of the point
    closest_cells[p / 3] = colliding_cell;
  }

  // Create neighborhood communicator in the reverse direction: send
  // back col to requesting processes
  MPI_Comm reverse_comm;
  MPI_Dist_graph_create_adjacent(
      comm, out_ranks.size(), out_ranks.data(), MPI_UNWEIGHTED, in_ranks.size(),
      in_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &reverse_comm);

  // Reuse sizes and offsets from first communication set
  // but divide by three
  {
    auto rescale = [](auto& x)
    {
      std::transform(x.cbegin(), x.cend(), x.begin(),
                     [](auto e) { return (e / 3); });
    };
    rescale(recv_sizes);
    rescale(recv_offsets);
    rescale(send_sizes);
    rescale(send_offsets);

    // The communication is reversed, so swap recv to send offsets
    std::swap(recv_sizes, send_sizes);
    std::swap(recv_offsets, send_offsets);
  }

  std::vector<std::int32_t> recv_ranks(recv_offsets.back());
  MPI_Neighbor_alltoallv(cell_indicator.data(), send_sizes.data(),
                         send_offsets.data(), MPI_INT32_T, recv_ranks.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT32_T,
                         reverse_comm);

  std::vector<int> point_owners(points.size() / 3, -1);
  for (std::size_t i = 0; i < unpack_map.size(); i++)
  {
    const std::int32_t pos = unpack_map[i];
    // Only insert new owner if no owner has previously been found
    if (recv_ranks[i] >= 0 && point_owners[pos] == -1)
      point_owners[pos] = recv_ranks[i];
  }

  // Create extrapolation marker for those points already sent to other
  // process
  std::vector<std::uint8_t> send_extrapolate(recv_offsets.back());
  for (std::int32_t i = 0; i < recv_offsets.back(); i++)
  {
    const std::int32_t pos = unpack_map[i];
    send_extrapolate[i] = point_owners[pos] == -1;
  }

  // Swap communication direction, to send extrapolation marker to other
  // processes
  std::swap(send_sizes, recv_sizes);
  std::swap(send_offsets, recv_offsets);
  std::vector<std::uint8_t> dest_extrapolate(recv_offsets.back());
  MPI_Neighbor_alltoallv(send_extrapolate.data(), send_sizes.data(),
                         send_offsets.data(), MPI_UINT8_T,
                         dest_extrapolate.data(), recv_sizes.data(),
                         recv_offsets.data(), MPI_UINT8_T, forward_comm);

  std::vector<T> squared_distances(received_points.size() / 3, -1);

  for (std::size_t i = 0; i < dest_extrapolate.size(); i++)
  {
    if (dest_extrapolate[i] == 1)
    {
      assert(closest_cells[i] == -1);
      std::array<T, 3> point;
      std::copy_n(std::next(received_points.begin(), 3 * i), 3, point.begin());

      // Find shortest distance among cells with colldiing bounding box
      T shortest_distance = std::numeric_limits<T>::max();
      std::int32_t closest_cell = -1;
      for (auto cell : candidate_collisions.links(i))
      {
        auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        std::vector<T> nodes(3 * dofs.size());
        for (std::size_t j = 0; j < dofs.size(); ++j)
        {
          const int pos = 3 * dofs[j];
          for (std::size_t k = 0; k < 3; ++k)
            nodes[3 * j + k] = geom_dofs[pos + k];
        }
        const std::array<T, 3> d = dolfinx::geometry::compute_distance_gjk<T>(
            std::span<const T>(point.data(), point.size()), nodes);
        if (T current_distance = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            current_distance < shortest_distance)
        {
          shortest_distance = current_distance;
          closest_cell = cell;
        }
      }
      closest_cells[i] = closest_cell;
      squared_distances[i] = shortest_distance;
    }
  }

  std::swap(recv_sizes, send_sizes);
  std::swap(recv_offsets, send_offsets);

  // Get distances from closest entity of points that were on the other process
  std::vector<T> recv_distances(recv_offsets.back());
  MPI_Neighbor_alltoallv(
      squared_distances.data(), send_sizes.data(), send_offsets.data(),
      dolfinx::MPI::mpi_t<T>, recv_distances.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_t<T>, reverse_comm);

  // Update point ownership with extrapolation information
  std::vector<T> closest_distance(point_owners.size(),
                                  std::numeric_limits<T>::max());
  for (std::size_t i = 0; i < out_ranks.size(); i++)
  {
    for (std::int32_t j = recv_offsets[i]; j < recv_offsets[i + 1]; j++)
    {
      const std::int32_t pos = unpack_map[j];
      auto current_dist = recv_distances[j];
      // Update if closer than previous guess and was found
      if (auto d = closest_distance[pos];
          (current_dist > 0) and (current_dist < d))
      {
        point_owners[pos] = out_ranks[i];
        closest_distance[pos] = current_dist;
      }
    }
  }

  // Communication is reversed again to send dest ranks to all processes
  std::swap(send_sizes, recv_sizes);
  std::swap(send_offsets, recv_offsets);

  // Pack ownership data
  std::vector<std::int32_t> send_owners(send_offsets.back());
  std::fill(counter.begin(), counter.end(), 0);
  for (std::size_t i = 0; i < points.size() / 3; ++i)
  {
    for (auto p : collisions.links(i))
    {
      int neighbor = rank_to_neighbor[p];
      send_owners[send_offsets[neighbor] + counter[neighbor]++]
          = point_owners[i];
    }
  }

  // Send ownership info
  std::vector<std::int32_t> dest_ranks(recv_offsets.back());
  MPI_Neighbor_alltoallv(send_owners.data(), send_sizes.data(),
                         send_offsets.data(), MPI_INT32_T, dest_ranks.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT32_T,
                         forward_comm);

  // Unpack dest ranks if point owner is this rank
  std::vector<int> owned_recv_ranks;
  owned_recv_ranks.reserve(recv_offsets.back());
  std::vector<T> owned_recv_points;
  std::vector<std::int32_t> owned_recv_cells;
  for (std::size_t i = 0; i < in_ranks.size(); i++)
  {
    for (std::int32_t j = recv_offsets[i]; j < recv_offsets[i + 1]; j++)
    {
      if (rank == dest_ranks[j])
      {
        owned_recv_ranks.push_back(in_ranks[i]);
        owned_recv_points.insert(
            owned_recv_points.end(), std::next(received_points.cbegin(), 3 * j),
            std::next(received_points.cbegin(), 3 * (j + 1)));
        owned_recv_cells.push_back(closest_cells[j]);
      }
    }
  }

  MPI_Comm_free(&forward_comm);
  MPI_Comm_free(&reverse_comm);
  return dolfinx::geometry::PointOwnershipData<T>{.src_owner = std::move(point_owners),
    .dest_owners = std::move(owned_recv_ranks),
    .dest_points = std::move(owned_recv_points),
    .dest_cells = std::move(owned_recv_cells)};
}

template <std::floating_point T>
std::vector<int> find_owner_rank(
    std::span<const T> points,
    const dolfinx::geometry::BoundingBoxTree<T>& cell_bb_tree,
    const dolfinx::fem::Function<T>& active_els_func)
{

  auto mesh = active_els_func.function_space()->mesh();
  MPI_Comm comm = mesh->comm();

  dolfinx::geometry::BoundingBoxTree global_bbtree = cell_bb_tree.create_global_tree(comm);

  // Compute collisions:
  // For each point in `points` get the processes it should be sent to
  dolfinx::graph::AdjacencyList collisions = compute_collisions(global_bbtree, points);

  // Get unique list of outgoing ranks
  std::vector<std::int32_t> out_ranks = collisions.array();
  std::sort(out_ranks.begin(), out_ranks.end());
  out_ranks.erase(std::unique(out_ranks.begin(), out_ranks.end()),
                  out_ranks.end());
  // Compute incoming edges (source processes)
  std::vector in_ranks = dolfinx::MPI::compute_graph_edges_nbx(comm, out_ranks);
  std::sort(in_ranks.begin(), in_ranks.end());

  // Create neighborhood communicator in forward direction
  MPI_Comm forward_comm;
  MPI_Dist_graph_create_adjacent(
      comm, in_ranks.size(), in_ranks.data(), MPI_UNWEIGHTED, out_ranks.size(),
      out_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &forward_comm);

  // Compute map from global mpi rank to neighbor rank, "collisions"
  // uses global rank
  std::map<std::int32_t, std::int32_t> rank_to_neighbor;
  for (std::size_t i = 0; i < out_ranks.size(); i++)
    rank_to_neighbor[out_ranks[i]] = i;

  // Count the number of points to send per neighbor process
  std::vector<std::int32_t> send_sizes(out_ranks.size());
  for (std::size_t i = 0; i < points.size() / 3; ++i)
    for (auto p : collisions.links(i))
      send_sizes[rank_to_neighbor[p]] += 3;

  // Compute receive sizes
  std::vector<std::int32_t> recv_sizes(in_ranks.size());
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Request sizes_request;
  MPI_Ineighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                         MPI_INT, forward_comm, &sizes_request);

  // Compute sending offsets
  std::vector<std::int32_t> send_offsets(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_offsets.begin(), 1));

  // Pack data to send and store unpack map
  std::vector<T> send_data(send_offsets.back());
  std::vector<std::int32_t> counter(send_sizes.size(), 0);
  // unpack map: [index in adj list][pos in x]
  std::vector<std::int32_t> unpack_map(send_offsets.back() / 3);
  for (std::size_t i = 0; i < points.size(); i += 3)
  {
    for (auto p : collisions.links(i / 3))
    {
      int neighbor = rank_to_neighbor[p];
      int pos = send_offsets[neighbor] + counter[neighbor];
      auto it = std::next(send_data.begin(), pos);
      std::copy_n(std::next(points.begin(), i), 3, it);
      unpack_map[pos / 3] = i / 3;
      counter[neighbor] += 3;
    }
  }

  MPI_Wait(&sizes_request, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_offsets(in_ranks.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_offsets.begin(), 1));

  std::vector<T> received_points((std::size_t)recv_offsets.back());
  MPI_Neighbor_alltoallv(
      send_data.data(), send_sizes.data(), send_offsets.data(),
      dolfinx::MPI::mpi_t<T>, received_points.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_t<T>, forward_comm);

  // Get mesh geometry for closest entity
  const dolfinx::mesh::Geometry<T>& geometry = mesh->geometry();
  std::span<const T> geom_dofs = geometry.x();
  auto x_dofmap = geometry.dofmap();

  // Compute candidate cells for collisions (and extrapolation)
  const dolfinx::graph::AdjacencyList<std::int32_t> candidate_collisions
      = compute_collisions(cell_bb_tree, std::span<const T>(received_points.data(),
                                                  received_points.size()));

  // Each process checks which points collide with a cell on the process
  const int rank = dolfinx::MPI::rank(comm);
  std::vector<std::int32_t> cell_indicator(received_points.size() / 3);
  std::vector<std::int32_t> closest_cells(received_points.size() / 3);
  std::vector<std::int32_t> candidate_cells;
  auto active_els_array = active_els_func.x()->array();
  for (std::size_t p = 0; p < received_points.size(); p += 3)
  {
    std::array<T, 3> point;
    std::copy_n(std::next(received_points.begin(), p), 3, point.begin());
    // Find first colliding cell among the cells with colliding bounding boxes
    std::span<const std::int32_t> all_candidate_cells = candidate_collisions.links(p / 3);
    candidate_cells.clear();
    candidate_cells.reserve(all_candidate_cells.size());
    std::copy_if(all_candidate_cells.begin(),
        all_candidate_cells.end(),
        std::back_inserter(candidate_cells),
        [&active_els_array](std::int32_t cell){ return bool(active_els_array[cell]);});
    const int colliding_cell = dolfinx::geometry::compute_first_colliding_cell(
        *mesh, candidate_cells, point,
        10 * std::numeric_limits<T>::epsilon());
    // If a collding cell is found, store the rank of the current process
    // which will be sent back to the owner of the point
    cell_indicator[p / 3] = (colliding_cell >= 0) ? rank : -1;
    // Store the cell index for lookup once the owning processes has determined
    // the ownership of the point
    closest_cells[p / 3] = colliding_cell;
  }

  // Create neighborhood communicator in the reverse direction: send
  // back col to requesting processes
  MPI_Comm reverse_comm;
  MPI_Dist_graph_create_adjacent(
      comm, out_ranks.size(), out_ranks.data(), MPI_UNWEIGHTED, in_ranks.size(),
      in_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &reverse_comm);

  // Reuse sizes and offsets from first communication set
  // but divide by three
  {
    auto rescale = [](auto& x)
    {
      std::transform(x.cbegin(), x.cend(), x.begin(),
                     [](auto e) { return (e / 3); });
    };
    rescale(recv_sizes);
    rescale(recv_offsets);
    rescale(send_sizes);
    rescale(send_offsets);

    // The communication is reversed, so swap recv to send offsets
    std::swap(recv_sizes, send_sizes);
    std::swap(recv_offsets, send_offsets);
  }

  std::vector<std::int32_t> recv_ranks(recv_offsets.back());
  MPI_Neighbor_alltoallv(cell_indicator.data(), send_sizes.data(),
                         send_offsets.data(), MPI_INT32_T, recv_ranks.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT32_T,
                         reverse_comm);

  std::vector<int> point_owners(points.size() / 3, -1);
  for (std::size_t i = 0; i < unpack_map.size(); i++)
  {
    const std::int32_t pos = unpack_map[i];
    // Only insert new owner if no owner has previously been found
    if (recv_ranks[i] >= 0 && point_owners[pos] == -1)
      point_owners[pos] = recv_ranks[i];
  }

  // Create extrapolation marker for those points already sent to other
  // process
  std::vector<std::uint8_t> send_extrapolate(recv_offsets.back());
  for (std::int32_t i = 0; i < recv_offsets.back(); i++)
  {
    const std::int32_t pos = unpack_map[i];
    send_extrapolate[i] = point_owners[pos] == -1;
  }

  // Swap communication direction, to send extrapolation marker to other
  // processes
  std::swap(send_sizes, recv_sizes);
  std::swap(send_offsets, recv_offsets);
  std::vector<std::uint8_t> dest_extrapolate(recv_offsets.back());
  MPI_Neighbor_alltoallv(send_extrapolate.data(), send_sizes.data(),
                         send_offsets.data(), MPI_UINT8_T,
                         dest_extrapolate.data(), recv_sizes.data(),
                         recv_offsets.data(), MPI_UINT8_T, forward_comm);

  std::vector<T> squared_distances(received_points.size() / 3, -1);

  for (std::size_t i = 0; i < dest_extrapolate.size(); i++)
  {
    if (dest_extrapolate[i] == 1)
    {
      assert(closest_cells[i] == -1);
      std::array<T, 3> point;
      std::copy_n(std::next(received_points.begin(), 3 * i), 3, point.begin());

      // Find shortest distance among cells with colldiing bounding box
      T shortest_distance = std::numeric_limits<T>::max();
      std::int32_t closest_cell = -1;
      for (auto cell : candidate_collisions.links(i))
      {
        if (not(active_els_array[cell])) continue;
        auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        std::vector<T> nodes(3 * dofs.size());
        for (std::size_t j = 0; j < dofs.size(); ++j)
        {
          const int pos = 3 * dofs[j];
          for (std::size_t k = 0; k < 3; ++k)
            nodes[3 * j + k] = geom_dofs[pos + k];
        }
        const std::array<T, 3> d = dolfinx::geometry::compute_distance_gjk<T>(
            std::span<const T>(point.data(), point.size()), nodes);
        if (T current_distance = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            current_distance < shortest_distance)
        {
          shortest_distance = current_distance;
          closest_cell = cell;
        }
      }
      closest_cells[i] = closest_cell;
      squared_distances[i] = shortest_distance;
    }
  }

  std::swap(recv_sizes, send_sizes);
  std::swap(recv_offsets, send_offsets);

  // Get distances from closest entity of points that were on the other process
  std::vector<T> recv_distances(recv_offsets.back());
  MPI_Neighbor_alltoallv(
      squared_distances.data(), send_sizes.data(), send_offsets.data(),
      dolfinx::MPI::mpi_t<T>, recv_distances.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_t<T>, reverse_comm);

  // Update point ownership with extrapolation information
  std::vector<T> closest_distance(point_owners.size(),
                                  std::numeric_limits<T>::max());
  for (std::size_t i = 0; i < out_ranks.size(); i++)
  {
    for (std::int32_t j = recv_offsets[i]; j < recv_offsets[i + 1]; j++)
    {
      const std::int32_t pos = unpack_map[j];
      auto current_dist = recv_distances[j];
      // Update if closer than previous guess and was found
      if (auto d = closest_distance[pos];
          (current_dist > 0) and (current_dist < d))
      {
        point_owners[pos] = out_ranks[i];
        closest_distance[pos] = current_dist;
      }
    }
  }
  return point_owners;
}

using dextents2 = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;
template <std::floating_point T>
std::tuple<size_t, std::vector<int>, std::vector<std::int32_t>, std::vector<std::int64_t>, std::vector<T>>
                        scatter_cell_integration_data_po(
                          const dolfinx::geometry::PointOwnershipData<T> &po,
                          const dolfinx::fem::FunctionSpace<double>& V,
                          const multiphenicsx::fem::DofMapRestriction& restriction,
                          const dolfinx::fem::Function<T> &mat_id_func
                        )
{
  size_t num_pts_snd = po.dest_cells.size();
  size_t num_pts_rcv = po.src_owner.size();
  auto mesh = V.mesh();
  // Scatter loc cell indices and materials
  std::span<const T> mat_ids = mat_id_func.x()->array();
  std::vector<std::int32_t> _indices_n_mats_to_rcv(2*num_pts_rcv);
  std::vector<std::int32_t> _indices_n_mats_to_snd(2*num_pts_snd);
  for (size_t i = 0; i < num_pts_snd; ++i) {
    std::int32_t icell = po.dest_cells[i];
    _indices_n_mats_to_snd[2*i]   = icell;
    _indices_n_mats_to_snd[2*i+1] = mat_ids[icell];
  }
  using dextents2 = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const std::int32_t, dextents2> indices_n_mats_to_send(
      _indices_n_mats_to_snd.data(), num_pts_snd, 2);
  scatter_values(mesh->comm(), po.dest_owners, po.src_owner, indices_n_mats_to_send,
                 std::span(_indices_n_mats_to_rcv));
  // Unpack
  std::vector<std::int32_t> mat_ids_rcv(num_pts_rcv);
  std::vector<std::int32_t> owner_cells(num_pts_rcv);
  for (size_t i = 0; i < num_pts_rcv; ++i) {
    owner_cells[i] = _indices_n_mats_to_rcv[2*i];
    mat_ids_rcv[i] = _indices_n_mats_to_rcv[2*i+1];
  }
  // Scatter global dofs
  auto con_v = restriction.dofmap()->map();
  auto imap = restriction.index_map;
  size_t num_dofs_cell = con_v.extent(1);
  std::vector<std::int64_t> _gdofs_cells_snd(num_pts_snd * num_dofs_cell);
  for (size_t idx = 0; idx < po.dest_cells.size(); ++idx) {
    std::int32_t icell = po.dest_cells[idx];
    auto ldofs_cell = restriction.cell_dofs(icell);
    std::span gdofs_cell(_gdofs_cells_snd.data() + idx * num_dofs_cell, num_dofs_cell);
    imap->local_to_global(ldofs_cell, gdofs_cell);
  }
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const std::int64_t, dextents2> gdofs_cells_snd(
      _gdofs_cells_snd.data(), num_pts_snd, num_dofs_cell);
  std::vector<std::int64_t> gdofs_cells_rcv(num_pts_rcv*num_dofs_cell);
  scatter_values(mesh->comm(), po.dest_owners, po.src_owner, gdofs_cells_snd,
                 std::span(gdofs_cells_rcv));
  // Scatter cell geometries
  const std::size_t num_dofs_g = mesh->geometry().cmap().dim();
  std::vector<T> _cell_geometries_snd(num_pts_snd * num_dofs_g * 3);
  std::vector<T> _cell_geometries_rcv(num_pts_rcv * num_dofs_g * 3);
  auto x = mesh->geometry().x();
  auto map = mesh->geometry().dofmap();
  for (size_t i = 0; i < num_pts_snd; ++i) {
    auto cell_dofs_g = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
    map, po.dest_cells[i], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (size_t j = 0; j < num_dofs_g; ++j) {
      int32_t idof = cell_dofs_g[j];
      std::copy(x.begin() + idof * 3, x.begin() + idof * 3 + 3, _cell_geometries_snd.begin() + i * 3 * num_dofs_g + 3 * j);
    }
  }
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const T, dextents2> cell_geometries_snd(
      _cell_geometries_snd.data(), num_pts_snd, 3*num_dofs_g);
  scatter_values(mesh->comm(), po.dest_owners, po.src_owner, cell_geometries_snd,
                 std::span(_cell_geometries_rcv));
  // Define (rank, loc cell idx) -> idx map
  std::map<std::pair<int, std::int32_t>, int> unique_cell_map;
  int curr_value = 0;
  std::vector<int> cell_indices(num_pts_rcv);
  std::vector<size_t> unique_cell_indices;
  unique_cell_indices.reserve(num_pts_rcv);
  for (size_t i = 0; i < owner_cells.size(); ++i) {
    // rank = po.src_owner[i], loc cell idx = owner_cells[idx]
    std::pair<int, std::int32_t> unique_cell_id(po.src_owner[i], owner_cells[i]);
    if (unique_cell_map.contains(unique_cell_id)) {
      cell_indices[i] = unique_cell_map[unique_cell_id];
    } else {
      unique_cell_map[unique_cell_id] = curr_value;
      unique_cell_indices.push_back(i);
      cell_indices[i] = curr_value;
      ++curr_value;
    }
  }
  size_t num_unique_pts_rcv = unique_cell_indices.size();
  // Subvectors with unique data
  std::vector<std::int32_t> unique_mat_ids_rcv(num_unique_pts_rcv);
  std::vector<std::int64_t> unique_cell_gdofs_rcv(num_unique_pts_rcv * num_dofs_cell);
  std::vector<T> unique_cell_geometries_rcv(num_unique_pts_rcv * num_dofs_g * 3);
  for (int i = 0; i < unique_cell_indices.size(); ++i) {
    unique_mat_ids_rcv[i] = mat_ids_rcv[unique_cell_indices[i]];
    std::move(gdofs_cells_rcv.begin() + unique_cell_indices[i]*num_dofs_cell,
              gdofs_cells_rcv.begin() + unique_cell_indices[i]*num_dofs_cell + num_dofs_cell,
              unique_cell_gdofs_rcv.begin() + i * num_dofs_cell);
    std::move(_cell_geometries_rcv.begin() + unique_cell_indices[i]*num_dofs_g*3,
              _cell_geometries_rcv.begin() + unique_cell_indices[i]*num_dofs_g*3 + num_dofs_g * 3,
              unique_cell_geometries_rcv.begin() + i * num_dofs_g * 3);
  }

  return {num_unique_pts_rcv, cell_indices, unique_mat_ids_rcv, unique_cell_gdofs_rcv, unique_cell_geometries_rcv};
}
