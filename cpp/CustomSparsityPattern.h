#pragma once
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <petscsystypes.h>

// Same as dolfinx except it expects global cols
// instead upon insertion
class CustomSparsityPattern
{
public:
  CustomSparsityPattern(
      MPI_Comm comm,
      const std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2>& maps,
      const std::array<int, 2>& bs)
    : _comm(comm), _index_maps(maps), _bs(bs),
      _row_cache(maps[0]->size_local() + maps[0]->num_ghosts())
  {
  }
MPI_Comm comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
std::shared_ptr<const dolfinx::common::IndexMap> index_map(int dim) const
{
  return _index_maps.at(dim);
}
//-----------------------------------------------------------------------------
int block_size(int dim) const { return _bs[dim]; }
//-----------------------------------------------------------------------------
// Insert local rows and global cols
void insert(std::span<const std::int32_t> rows,
            std::span<const std::int64_t> cols)
{
  if (!_offsets.empty())
  {
    throw std::runtime_error(
        "Cannot insert into sparsity pattern. It has already been finalized");
  }

  assert(_index_maps[0]);
  const std::int32_t max_row
      = _index_maps[0]->size_local() + _index_maps[0]->num_ghosts() - 1;

  for (std::int32_t row : rows)
  {
    if (row > max_row or row < 0)
    {
      throw std::runtime_error(
          "Cannot insert rows that do not exist in the IndexMap.");
    }
    _row_cache[row].insert(_row_cache[row].end(), cols.begin(), cols.end());
  }
}
//-----------------------------------------------------------------------------
void finalize()
{
  if (!_offsets.empty())
    throw std::runtime_error("Sparsity pattern has already been finalised.");

  dolfinx::common::Timer t0("SparsityPattern::finalize");

  const int rank = dolfinx::MPI::rank(_comm.comm());
  const int comm_size = dolfinx::MPI::size(_comm.comm());

  assert(_index_maps[0]);
  const std::int32_t local_size0 = _index_maps[0]->size_local();
  const std::array local_range0 = _index_maps[0]->local_range();
  std::span ghosts0 = _index_maps[0]->ghosts();
  std::span owners0 = _index_maps[0]->owners();
  std::span src0 = _index_maps[0]->src();

  assert(_index_maps[1]);
  const std::int32_t local_size1 = _index_maps[1]->size_local();
  const std::array local_range1 = _index_maps[1]->local_range();

  // Share bounds of col index map
  std::vector<std::int64_t> all_local_ranges1(comm_size);
  PetscErrorCode ierr = MPI_Allgather(&local_range1[1], 1, MPI_INT64_T,
      all_local_ranges1.data(), 1, MPI_INT64_T, _comm.comm());

  /*
  _col_ghosts.assign(_index_maps[1]->ghosts().begin(),
                     _index_maps[1]->ghosts().end());
  _col_ghost_owners.assign(_index_maps[1]->owners().begin(),
                           _index_maps[1]->owners().end());
 */

  // Compute size of data to send to each process
  std::vector<int> send_sizes(src0.size(), 0);
  for (std::size_t i = 0; i < owners0.size(); ++i)
  {
    auto it = std::ranges::lower_bound(src0, owners0[i]);
    assert(it != src0.end() and *it == owners0[i]);
    const int neighbour_rank = std::distance(src0.begin(), it);
    send_sizes[neighbour_rank] += 3 * _row_cache[i + local_size0].size();
  }

  // Compute send displacements
  std::vector<int> send_disp(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_disp.begin(), 1));

  // For each ghost row, pack and send (global row, global col,
  // col_owner) triplets to send to neighborhood
  std::vector<int> insert_pos(send_disp);
  std::vector<std::int64_t> ghost_data(send_disp.back());
  for (std::size_t i = 0; i < owners0.size(); ++i)
  {
    auto it = std::ranges::lower_bound(src0, owners0[i]);
    assert(it != src0.end() and *it == owners0[i]);
    const int neighbour_rank = std::distance(src0.begin(), it);

    for (std::int64_t col_global : _row_cache[i + local_size0])
    {
      // Get index in send buffer
      const std::int32_t pos = insert_pos[neighbour_rank];

      // Pack send data
      ghost_data[pos] = ghosts0[i];
      // Find owning rank
      const auto it = std::find_if(all_local_ranges1.begin(), all_local_ranges1.end(),
          [&col_global](const std::int64_t end){return col_global < end;});
      std::int64_t owner = std::distance(all_local_ranges1.begin(), it);

      ghost_data[pos + 1] = col_global;
      ghost_data[pos + 2] = owner;

      insert_pos[neighbour_rank] += 3;
    }
  }

  // Exchange data between processes
  std::vector<std::int64_t> ghost_data_in;
  {
    MPI_Comm comm;
    std::span dest0 = _index_maps[0]->dest();
    MPI_Dist_graph_create_adjacent(
        _index_maps[0]->comm(), dest0.size(), dest0.data(), MPI_UNWEIGHTED,
        src0.size(), src0.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

    std::vector<int> recv_sizes(dest0.size());
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, comm);

    // Build recv displacements
    std::vector<int> recv_disp = {0};
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::back_inserter(recv_disp));

    ghost_data_in.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(ghost_data.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, ghost_data_in.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           comm);
    MPI_Comm_free(&comm);
  }

  /* TODO: Do I need this?
  std::map<std::int64_t, std::int32_t> global_to_local;
  std::int32_t local_i = local_size1;
  for (std::int64_t global_i : _col_ghosts)
    global_to_local.insert({global_i, local_i++});
  */

  // Insert data received from the neighborhood
  for (std::size_t i = 0; i < ghost_data_in.size(); i += 3)
  {
    const std::int32_t row_local = ghost_data_in[i] - local_range0[0];
    const std::int64_t col = ghost_data_in[i + 1];
    _row_cache[row_local].push_back(col);
  }

  // Sort and remove duplicate column indices in each row
  std::vector<std::int32_t> adj_counts(local_size0 + owners0.size(), 0);
  _diagonal_beg_offsets.resize(local_size0 + owners0.size());
  _diagonal_end_offsets.resize(local_size0 + owners0.size());
  for (std::size_t i = 0; i < local_size0 + owners0.size(); ++i)
  {
    std::vector<std::int64_t>& row = _row_cache[i];
    std::ranges::sort(row);
    auto it_end = std::ranges::unique(row).begin();

    _diagonal_beg_offsets[i] = std::distance(
        row.begin(), std::find_if(row.begin(), it_end,
          [&local_range1](std::int64_t col){return (col >= local_range1[0]);}
          )
        );
    _diagonal_end_offsets[i] = std::distance(
        row.begin(), std::find_if(row.begin(), it_end,
          [&local_range1](std::int64_t col){ return (col >= local_range1[1]);}
          )
        );

    _edges.insert(_edges.end(), row.begin(), it_end);
    adj_counts[i] += std::distance(row.begin(), it_end);
  }
  // Clear cache
  std::vector<std::vector<std::int64_t>>().swap(_row_cache);

  // Compute offsets for adjacency list
  _offsets.resize(local_size0 + owners0.size() + 1, 0);
  std::partial_sum(adj_counts.begin(), adj_counts.end(), _offsets.begin() + 1);

  _edges.shrink_to_fit();

  // Column count increased due to received rows from other processes
  spdlog::info("Column ghost size increased from {} to {}",
               _index_maps[1]->ghosts().size(), _col_ghosts.size());
}
//-----------------------------------------------------------------------------
std::int32_t nnz_diag(std::int32_t row) const
{
  if (_offsets.empty())
    throw std::runtime_error("Sparsity pattern has not be finalized.");
  return (_diagonal_end_offsets[row] - _diagonal_beg_offsets[row]);
}
//-----------------------------------------------------------------------------
std::int32_t nnz_off_diag(std::int32_t row) const
{
  if (_offsets.empty())
    throw std::runtime_error("Sparsity pattern has not be finalized.");
  return (_offsets[row + 1] - _offsets[row]) - (_diagonal_end_offsets[row] - _diagonal_beg_offsets[row]);
}
//-----------------------------------------------------------------------------
private:
  // MPI communicator
  dolfinx::MPI::Comm _comm;

  // Index maps for each dimension
  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> _index_maps;

  // Block size
  std::array<int, 2> _bs;

  // Non-zero ghost columns in owned rows
  std::vector<std::int64_t> _col_ghosts;

  // Owning process of ghost columns in owned rows
  std::vector<std::int32_t> _col_ghost_owners;

  // Cache for unassembled entries on owned and unowned (ghost) rows
  std::vector<std::vector<std::int64_t>> _row_cache;

  // Sparsity pattern adjacency data (computed once pattern is
  // finalised). _edges holds the edges (connected dofs). The edges for
  // node i are in the range [_offsets[i], _offsets[i + 1]).
  std::vector<std::int32_t> _edges;
  std::vector<std::int64_t> _offsets;

  // Start of off-diagonal (unowned columns) on each row (row-wise)
  std::vector<std::int32_t> _diagonal_beg_offsets;
  std::vector<std::int32_t> _diagonal_end_offsets;
};

// Copy of original create_matrix to pass my own SparsityPattern
void custom_create_matrix(Mat& A, MPI_Comm comm, const CustomSparsityPattern& sp,
                         std::string type = std::string());
