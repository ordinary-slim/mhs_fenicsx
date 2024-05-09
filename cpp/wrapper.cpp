#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include "locate_active_boundary.h"
#include "get_active_dofs_external.h"
#include "compute_el_size_along_vector.h"

namespace nb = nanobind;
using int_vector = std::vector<int>;

NB_MODULE(mhs_fenicsx_cpp, m) {
  nb::bind_vector<int_vector>(m, "int_vector");
  m.def("locate_active_boundary", &locate_active_boundary<double>);
  m.def("locate_active_boundary", &locate_active_boundary<float>);
  m.def("get_active_dofs_external", &get_active_dofs_external<double>);
  m.def("get_active_dofs_external", &get_active_dofs_external<float>);
  m.def("compute_el_size_along_vector", &compute_el_size_along_vector<double>);
  m.def("compute_el_size_along_vector", &compute_el_size_along_vector<float>);
}
