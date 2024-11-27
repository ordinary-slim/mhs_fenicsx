#include "create_robin_robin_monolithic.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/fem/petsc.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <caster_petsc.h>
#include <caster_mpi.h>
#include <petsc4py/petsc4py.h>

namespace nb = nanobind;
void declare_create_robin_robin_monolithic(nb::module_ &m) {
  import_petsc4py();
  m.def(
      "create_robin_robin_monolithic",
      [](dolfinx_wrappers::MPICommWrapper comm,
         dolfinx::fem::FunctionSpace<double>& Qs_gamma,
         dolfinx::fem::FunctionSpace<double>& V_i,
         multiphenicsx::fem::DofMapRestriction& restriction_i,
         dolfinx::fem::FunctionSpace<double>& V_j,
         multiphenicsx::fem::DofMapRestriction& restriction_j,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> facets_mesh_i,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells_mesh_i,
         const geometry::PointOwnershipData<U>& po_mesh_j,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells_mesh_j,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> gweights_facet)
      {
        const std::size_t num_points = gweights_facet.ndim() == 1 ? 1 : gweights_facet.shape(0);
        const std::size_t gdim = V_i.mesh()->geometry().dim();
        std::span<const double> _gp(gweights_facet.data(), gdim * num_points);
        Mat A  = create_robin_robin_monolithic<double>(comm.get(), Qs_gamma,
                        V_i, restriction_i,
                        V_j, restriction_j,
                        std::span(facets_mesh_i.data(), facets_mesh_i.size()),
                        std::span(cells_mesh_i.data(), cells_mesh_i.size()),
                        po_mesh_j,
                        std::span(cells_mesh_j.data(), cells_mesh_j.size()),
                        std::span(gweights_facet.data(), gweights_facet.size())
                        );
        PyObject* obj = PyPetscMat_New(A);
        PetscObjectDereference((PetscObject)A);
        return nb::borrow(obj);
      });
      //nb::arg("A"), nb::arg("V"), "Hello");
}
