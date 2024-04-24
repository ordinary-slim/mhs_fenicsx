#include <nanobind/nanobind.h>
#include <dolfinx/mesh/Mesh.h>
#include <vector>

namespace nb = nanobind;
using int_vector = std::vector<int>;

template <std::floating_point T>
int_vector locate_active_boundary(const dolfinx::mesh::Mesh<T> &domain) {
  return int_vector();
}

NB_MODULE(cpp, m) {
  m.def("locate_active_boundary", &locate_active_boundary);
}
