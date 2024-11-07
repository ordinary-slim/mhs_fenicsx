#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include "get_active_dofs_external.h"
#include "compute_el_size_along_vector.h"
#include "get_facet_integration_entities.h"
#include <nanobind/stl/bind_map.h>

namespace nb = nanobind;
using int_vector = std::vector<int>;

void declare_interpolate_dg0_at_facets(nb::module_ &m);
void declare_my_determine_point_ownership(nb::module_ &m);
void declare_activation_utils(nb::module_ &m);
void declare_mesh_collision(nb::module_ &m);
void declare_submesh_utils(nb::module_ &m);
void declare_assemble_monolithic_robin(nb::module_ &m);

NB_MODULE(mhs_fenicsx_cpp, m) {
  nb::bind_vector<int_vector>(m, "int_vector");
  nb::bind_map<std::map<std::int32_t,std::int32_t>>(m,"int_map");
  m.def("get_active_dofs_external", &get_active_dofs_external<double>);
  m.def("get_active_dofs_external", &get_active_dofs_external<float>);
  m.def("compute_el_size_along_vector", &compute_el_size_along_vector<double>);
  m.def("compute_el_size_along_vector", &compute_el_size_along_vector<float>);
  m.def("get_facet_integration_entities", &get_facet_integration_entities<double>);
  m.def("get_facet_integration_entities", &get_facet_integration_entities<float>);
  declare_activation_utils(m);
  declare_interpolate_dg0_at_facets(m);
  declare_my_determine_point_ownership(m);
  declare_mesh_collision(m);
  declare_submesh_utils(m);
  declare_assemble_monolithic_robin(m);
}
