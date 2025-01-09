#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include "compute_el_size_along_vector.h"
#include <nanobind/stl/bind_map.h>

namespace nb = nanobind;
using int_vector = std::vector<int>;

void declare_interpolate(nb::module_ &m);
void declare_my_determine_point_ownership(nb::module_ &m);
void declare_activation_utils(nb::module_ &m);
void declare_mesh_collision(nb::module_ &m);
void declare_submesh_utils(nb::module_ &m);
void declare_create_robin_robin_monolithic(nb::module_ &m);
void declare_diffmesh_utils(nb::module_ &m);
void declare_get_facet_integration_entities(nb::module_ &m);

NB_MODULE(mhs_fenicsx_cpp, m) {
  nb::bind_vector<int_vector>(m, "int_vector");
  m.def("compute_el_size_along_vector", &compute_el_size_along_vector<double>);
  m.def("compute_el_size_along_vector", &compute_el_size_along_vector<float>);
  declare_get_facet_integration_entities(m);
  declare_diffmesh_utils(m);
  declare_activation_utils(m);
  declare_interpolate(m);
  declare_my_determine_point_ownership(m);
  declare_mesh_collision(m);
  declare_submesh_utils(m);
  declare_create_robin_robin_monolithic(m);
}
