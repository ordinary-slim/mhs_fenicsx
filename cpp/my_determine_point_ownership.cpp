#include <dolfinx/geometry/utils.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

using namespace dolfinx;

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
  geometry::BoundingBoxTree bb(mesh, tdim, cells, padding);
  geometry::BoundingBoxTree global_bbtree = bb.create_global_tree(comm);

  // Compute collisions:
  // For each point in `points` get the processes it should be sent to
  graph::AdjacencyList collisions = compute_collisions(global_bbtree, points);

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
      dolfinx::MPI::mpi_type<T>(), received_points.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_type<T>(), forward_comm);

  // Get mesh geometry for closest entity
  const mesh::Geometry<T>& geometry = mesh.geometry();
  std::span<const T> geom_dofs = geometry.x();
  auto x_dofmap = geometry.dofmap();

  // Compute candidate cells for collisions (and extrapolation)
  const graph::AdjacencyList<std::int32_t> candidate_collisions
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
    const int colliding_cell = geometry::compute_first_colliding_cell(
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
        const std::array<T, 3> d = geometry::compute_distance_gjk<T>(
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
      dolfinx::MPI::mpi_type<T>(), recv_distances.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_type<T>(), reverse_comm);

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
  return geometry::PointOwnershipData<T>{.src_owner = std::move(point_owners),
                                         .dest_owners = std::move(owned_recv_ranks),
                                         .dest_points = std::move(owned_recv_points),
                                         .dest_cells = std::move(owned_recv_cells)};
}

namespace nb = nanobind;

template <std::floating_point T>
void templated_declare_my_determine_point_ownership(nb::module_ &m) {
  m.def(
      "cellwise_determine_point_ownership",
      [](const dolfinx::mesh::Mesh<T>& mesh,
         nb::ndarray<const T, nb::c_contig> points,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
         T padding)
      {
        const std::size_t p_s0 = points.ndim() == 1 ? 1 : points.shape(0);
        std::span<const T> _p(points.data(), 3 * p_s0);
        return my_determine_point_ownership<T>(mesh,
                                               _p,
                                               std::span(cells.data(),cells.size()),
                                               padding);
      },
      nb::arg("mesh"),nb::arg("points"),nb::arg("cells"),nb::arg("padding"));
}

void declare_my_determine_point_ownership(nb::module_ &m) {
  templated_declare_my_determine_point_ownership<double>(m);
  templated_declare_my_determine_point_ownership<float>(m);
};
