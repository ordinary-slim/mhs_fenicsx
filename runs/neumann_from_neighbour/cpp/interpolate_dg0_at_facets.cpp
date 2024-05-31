#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <cassert>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/geometry/utils.h>
#include <vector>

using int_vector = std::vector<int>;

template <std::floating_point T>
void interpolate_dg0_at_facets(const dolfinx::fem::Function<T> &sending_f,
                               dolfinx::fem::Function<T> &receiving_f,
                               int_vector &facets) {
  std::shared_ptr smesh = sending_f.function_space()->mesh();//sending mesh
  std::shared_ptr rmesh = receiving_f.function_space()->mesh();//receiving mesh
  int cdim = rmesh->topology()->dim();
  // 1. Compute facet midpoints
  std::vector<T> midpoints = mesh::compute_midpoints<T>(*rmesh,cdim-1,facets);
  // 2. Determine point ownership
  geometry::PointOwnershipData po = geometry::determine_point_ownership<T>(
      *rmesh,
      midpoints,
      T(0));
  auto& dest_ranks = po.src_owner;
  auto& src_ranks = po.dest_owners;
  auto& recv_points = po.dest_points;
  auto& evaluation_cells = po.dest_cells;
  // 3. Eval my points
  // Code copied from dolfinx interpolate.h
  const std::size_t value_size = sending_f.function_space()->value_size();
  // Evaluate the interpolating function where possible
  std::vector<T> send_values(recv_points.size() / 3 * value_size);
  sending_f.eval(recv_points, {recv_points.size() / 3, (std::size_t)3},
                 evaluation_cells, send_values, {recv_points.size() / 3, value_size});
  std::cout << "value to send = " << std::endl;
  for (int i = 0; i < evaluation_cells.size(); ++i) {
    for (int j = 0; j < value_size; ++j) {
      std::cout << send_values[i*value_size+j] << ", ";
    }
    std::cout << std::endl;
  }

  using dextents2 = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>;

  // 4. Call dolfinx scatter
  // 5. Insert vals into receiving_f
}

namespace nb = nanobind;
NB_MODULE(interpolate_dg0_at_facets, m) {
  nb::bind_vector<int_vector>(m, "int_vector");
  m.def("interpolate_dg0_at_facets", &interpolate_dg0_at_facets<double>);
  m.def("interpolate_dg0_at_facets", &interpolate_dg0_at_facets<float>);
}
