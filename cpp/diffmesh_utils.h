#pragma once
#include <dolfinx/geometry/utils.h>

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
std::vector<std::int32_t> scatter_cells_po(
                          const dolfinx::mesh::Mesh<T> &mesh,
                          dolfinx::geometry::PointOwnershipData<T> &po)
{
  //// Cast to T, do communication, cast back
  std::vector<std::int32_t> _owner_cells(po.src_owner.size());
  using dextents2 = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const std::int32_t, dextents2> _send_values(
      po.dest_cells.data(), po.dest_cells.size(), 1);
  scatter_values(mesh.comm(), po.dest_owners, po.src_owner, _send_values,
                 std::span(_owner_cells));
  return _owner_cells;
}
