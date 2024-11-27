from dolfinx import fem, mesh
import numpy as np
import basix, basix.ufl
from mpi4py import MPI
from main import exact_sol, Rhs, left_marker_dirichlet, right_marker_dirichlet
import yaml
from mhs_fenicsx.problem import Problem
from mhs_fenicsx_cpp import cellwise_determine_point_ownership, scatter_cells_po, \
                            create_robin_robin_monolithic

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

def generate_facet_cell_quadrature(domain):
    dolfinx_to_basix_ctype = {
                            mesh.CellType.point:basix.CellType.point,
                            mesh.CellType.interval:basix.CellType.interval,
                            mesh.CellType.triangle:basix.CellType.triangle,
                            mesh.CellType.quadrilateral:basix.CellType.quadrilateral,
                            mesh.CellType.tetrahedron:basix.CellType.tetrahedron,
                            mesh.CellType.hexahedron:basix.CellType.hexahedron,
                            }
    tdim = domain.topology.dim
    fdim = tdim - 1
    cmap = domain.geometry.cmap
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_cells  = mesh.compute_incident_entities(domain.topology, boundary_facets, fdim, tdim) 
    fcelltype = dolfinx_to_basix_ctype[domain.topology.entity_types[-2][0]]
    celltype = dolfinx_to_basix_ctype[domain.topology.cell_type]
    sub_entity_connectivity = basix.cell.sub_entity_connectivity(celltype)
    num_facets_el = len(sub_entity_connectivity[tdim-1])

    gpoints_facet, gweights_facet = basix.make_quadrature(fcelltype, 2)
    # TODO: Tabulate facet element at gpoints
    felement = basix.create_element(
        basix.ElementFamily.P, fcelltype, cmap.degree, basix.LagrangeVariant.equispaced)
    ftab = felement.tabulate(0, gpoints_facet)

    num_gpoints_facet = len(gweights_facet)
    gpoints_cell = np.zeros((num_facets_el*num_gpoints_facet, tdim), dtype=gpoints_facet.dtype)
    gweights_cell = np.zeros(num_facets_el*num_gpoints_facet, dtype=gweights_facet.dtype)
    vertices = basix.geometry(celltype)
    for ifacet in range(num_facets_el):
        facet = sub_entity_connectivity[tdim-1][ifacet][0]
        for igp in range(num_gpoints_facet):
            idx = ifacet*num_gpoints_facet + igp
            gweights_cell[idx] = gweights_facet[igp]
            for idof in range(len(facet)):
                gpoints_cell[idx] += vertices[facet[idof]] * ftab[0][igp][idof][0]
    return gpoints_cell, gweights_cell, num_gpoints_facet

rhs = Rhs(params["material"]["density"],
          params["material"]["specific_heat"],
          params["material"]["conductivity"],
          params["advection_speed"])

def main():
    # Mesh and problems
    els_side = params["els_side"]
    left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.quadrilateral)
    right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.triangle)
    p_left = Problem(left_mesh, params, name=f"diff_mesh_robin_left")
    p_right = Problem(right_mesh, params, name=f"diff_mesh_robin_right")

    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= 0.5 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= 0.5 )

    for p in [p_left,p_right]:
        p.set_activation(active_els[p])
        p.set_rhs(rhs)

    # Set up interface Gamma
    gamma_facets = {}
    gamma_cells  = {}
    for p in [p_left, p_right]:
        gf = mesh.locate_entities(p.domain, p.dim-1,
                                               lambda x : np.isclose(x[0], 0.5))
        gamma_facets[p] = gf[:np.searchsorted(gf, p.facet_map.size_local)]
        gamma_cells[p]  = np.zeros_like(gamma_facets[p],dtype=np.int32)
        gc = gamma_cells[p]
        con_facet_cell = p.domain.topology.connectivity(p.dim-1,p.dim)
        for idx, ifacet in enumerate(gamma_facets[p]):
            local_con = con_facet_cell.links(ifacet)
            if p.active_els_func.x.array[local_con[0]]:
                gc[idx] = local_con[0]
            else:
                gc[idx] = local_con[1]

    gamma_mesh = {}
    Qs_gamma = {}
    Qs_gamma_x = {}
    Qs_gamma_po = { p_left : {}, p_right : {}}
    gamma_src_cells = { p_left : {}, p_right : {}}
    # Robin coupling: i = left, j = right
    # TODO 1: Make quadrature on Gamma left
    gamma_mesh[p_left], \
    subfacet_map, subvertex_map, subnodes_map = \
            mesh.create_submesh(p_left.domain,p_left.dim-1,gamma_facets[p_left])
    Qe = basix.ufl.quadrature_element(p_left.domain.topology.entity_types[-2][0].name,
                                      degree=2)
    Qs_gamma[p_left] = fem.functionspace(gamma_mesh[p_left], Qe)
    Qs_gamma_x[p_left] = Qs_gamma[p_left].tabulate_dof_coordinates()
    Qs_gamma_po[p_left][p_right] = cellwise_determine_point_ownership(
                                        p_right.domain._cpp_object,
                                        Qs_gamma_x[p_left],
                                        gamma_cells[p_right],
                                        np.float64(1e-7))
    gamma_src_cells[p_left][p_right] = np.array(
            scatter_cells_po(p_right.domain._cpp_object,
                             Qs_gamma_po[p_left][p_right])
            )
    A_lr = create_robin_robin_monolithic(comm,
                                         Qs_gamma[p_left]._cpp_object,
                                         p_left.v._cpp_object,
                                         p_left.restriction,
                                         p_right.v._cpp_object,
                                         p_right.restriction,
                                         gamma_facets[p_left],
                                         gamma_cells[p_left],
                                         Qs_gamma_po[p_left][p_right],
                                         gamma_src_cells[p_left][p_right],
                                         Qe._weights,
                                         )
    A_lr.assemble()
    A_lr.view()


if __name__=="__main__":
    main()
