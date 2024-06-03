#include <nanobind/nanobind.h>
#include <vector>

namespace nb = nanobind;
using int_vector = std::vector<int>;

void declare_interpolate_dg0_at_facets(nb::module_ &m);
void declare_my_determine_point_ownership(nb::module_ &m);

NB_MODULE(my_utils, m) {
  declare_interpolate_dg0_at_facets(m);
  declare_my_determine_point_ownership(m);
}
