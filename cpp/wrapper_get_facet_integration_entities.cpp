#include "get_facet_integration_entities.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <array.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
template <std::floating_point T>
void templated_declare_get_facet_integration_entities(nb::module_ &m) {
  m.def("get_facet_integration_entities",
      [](const dolfinx::mesh::Mesh<T> &domain,
        nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> facets,
        const dolfinx::fem::Function<T> &active_els_func,
        std::optional<std::pair<std::reference_wrapper<const dolfinx::common::IndexMap>, std::span<const std::int32_t>>> subindex_map)
      {
        std::vector<std::int32_t> facet_integration_data =
          get_facet_integration_entities(
              domain,
              std::span(facets.data(), facets.size()),
              active_els_func,
              subindex_map);
        return nb::ndarray<const std::int32_t, nb::numpy>(facet_integration_data.data(),
                                                  {facet_integration_data.size()}).cast();
      }, nb::arg("domain"), nb::arg("facets"), nb::arg("active_els_func"), nb::arg("subindex_map")=nb::none());
}

void declare_get_facet_integration_entities(nb::module_ &m) {
  templated_declare_get_facet_integration_entities<float>(m);
  templated_declare_get_facet_integration_entities<double>(m);
}
