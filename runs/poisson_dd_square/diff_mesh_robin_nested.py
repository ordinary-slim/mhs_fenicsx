from dolfinx import fem, mesh
import numpy as np
import basix, basix.ufl
from mpi4py import MPI
from main import exact_sol, grad_exact_sol, Rhs
import yaml
from mhs_fenicsx.problem import Problem
from mhs_fenicsx_cpp import cellwise_determine_point_ownership, scatter_cell_integration_data_po, \
                            create_robin_robin_monolithic, interpolate_dg0_at_facets
from ffcx.ir.elementtables import permute_quadrature_interval, \
                                  permute_quadrature_triangle, \
                                  permute_quadrature_quadrilateral
from ffcx.element_interface import map_facet_points
import multiphenicsx.fem.petsc
import ufl
import petsc4py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

rhs = Rhs(params["material"]["density"],
          params["material"]["specific_heat"],
          params["material"]["conductivity"],
          params["advection_speed"])

def left_marker_dirichlet(x):
    return np.isclose(x[0],0)
def right_marker_dirichlet(x):
    return np.isclose(x[0],1)
def left_marker_neumann(x):
    return np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] <= 0.5)
def right_marker_neumann(x):
    return np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] >= 0.5)
def left_marker_neumann_debug(x):
    return np.logical_or(np.logical_and(np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)), x[0] <= 0.5), np.isclose(x[0], 0.5))
def right_marker_neumann_debug(x):
    return np.logical_or(np.logical_and(np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)), x[0] >= 0.5), np.isclose(x[0], 0.5))
def marker_gamma(x):
    return np.isclose(x[0], 0.5)

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

    # Set-up
    for p, marker in zip([p_left,p_right], [left_marker_dirichlet, right_marker_dirichlet]):
        p.set_activation(active_els[p])
        p.set_rhs(rhs)
        # Dirichlet
        u_ex = fem.Function(p.v,name="exact")
        u_ex.interpolate(exact_sol)
        bdofs_dir  = fem.locate_dofs_geometrical(p.v,marker)
        p.dirichlet_bcs = [fem.dirichletbc(u_ex, bdofs_dir)]

    # Set up interface Gamma
    for p, p_ext in zip([p_left, p_right], [p_right, p_left]):
        p.find_gamma(p.get_active_in_external( p_ext ))
    gamma_facets = {}
    gamma_cells  = {}
    # TODO: Check if this is slow
    for p in [p_left, p_right]:
        gf = mesh.locate_entities(p.domain, p.dim-1, marker_gamma)
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
    gamma_renumbered_cells_ext = { p_left : {}, p_right : {}}
    gamma_dofs_cells_ext = { p_left : {}, p_right : {}}
    gamma_geoms_cells_ext = { p_left : {}, p_right : {}}
    gamma_iid = {p_left:{}, p_right:{}}
    ext_conductivity = {}
    midpoints_gamma = {p_left:None, p_right:None}
    # Robin coupling: i = left, j = right
    for p in [p_left, p_right]:
        gamma_mesh[p], \
        _, _, _ = \
                mesh.create_submesh(p.domain,p.dim-1,gamma_facets[p])

    def assemble_robin_matrix(p:Problem, p_ext:Problem, quadrature_degree=2):
        # GENERATE QUADRATURE
        cell_type =  p.domain.topology.entity_types[-1][0].name
        facet_type = p.domain.topology.entity_types[-2][0].name
        Qe = basix.ufl.quadrature_element(facet_type,
                                          degree=quadrature_degree)
        # GENERATE ALL PERMUTATIONS OF QUADRATURE
        cdim = p.domain.topology.dim
        fdim = cdim - 1
        num_gps = Qe.num_entity_dofs[-1][0]
        GPs = []
        if cdim==2:
            for ref in range(2):
                GPs.append(permute_quadrature_interval(Qe._points, ref))
        elif cdim==3:
            if facet_type == "triangle":
                for rot in range(3):
                    for ref in range(2):
                        GPs.append(permute_quadrature_triangle(Qe._points, ref, rot))
            elif facet_type == "quadrilateral":
                for rot in range(4):
                    for ref in range(2):
                        GPs.append(permute_quadrature_quadrilateral(Qe._points, ref, rot))
            else:
                raise Exception
        else:
            raise Exception
        permuted_quadrature_points_facet = np.vstack(GPs)
        num_gps_facet = permuted_quadrature_points_facet.shape[0]
        num_facets_cell = p.domain.ufl_cell().num_facets()
        permuted_quadrature_points_cell  = np.zeros((num_gps_facet * num_facets_cell, cdim), dtype=permuted_quadrature_points_facet.dtype)
        for ifacet in range(num_facets_cell):
            permuted_quadrature_points_cell[ifacet*num_gps_facet : ifacet*num_gps_facet + num_gps_facet, :cdim] = map_facet_points(permuted_quadrature_points_facet, ifacet, cell_type)

        Qs_gamma[p] = fem.functionspace(gamma_mesh[p], Qe)
        Qs_gamma_x[p] = Qs_gamma[p].tabulate_dof_coordinates()
        Qs_gamma_po[p][p_ext] = cellwise_determine_point_ownership(
                                            p_ext.domain._cpp_object,
                                            Qs_gamma_x[p],
                                            gamma_cells[p_ext],
                                            np.float64(1e-7))
        gamma_renumbered_cells_ext[p][p_ext], \
        gamma_dofs_cells_ext[p][p_ext], \
        gamma_geoms_cells_ext[p][p_ext] = \
                        scatter_cell_integration_data_po(Qs_gamma_po[p][p_ext],
                                                          p_ext.v._cpp_object,
                                                          p_ext.restriction)
        midpoints_gamma[p] = mesh.compute_midpoints(p.domain,p.domain.topology.dim-1,p.gamma_facets.find(1))
        gamma_iid[p][p_ext] = cellwise_determine_point_ownership(
                                            p_ext.domain._cpp_object,
                                            midpoints_gamma[p],
                                            gamma_cells[p_ext],
                                            np.float64(1e-6))
        ext_conductivity[p] = fem.Function(p.dg0, name="ext_k")
        interpolate_dg0_at_facets([p_ext.k._cpp_object],
                                  [ext_conductivity[p]._cpp_object],
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets._cpp_object,
                                  gamma_cells[p],
                                  gamma_iid[p][p_ext],
                                  p.gamma_facets_index_map,
                                  p.gamma_imap_to_global_imap)

        A = create_robin_robin_monolithic(Qs_gamma[p]._cpp_object,
                                          ext_conductivity[p]._cpp_object,
                                          Qs_gamma_x[p],
                                          permuted_quadrature_points_cell,
                                          Qe._weights,
                                          p.v._cpp_object,
                                          p.restriction,
                                          p_ext.v._cpp_object,
                                          p_ext.restriction,
                                          gamma_facets[p],
                                          gamma_cells[p],
                                          Qs_gamma_po[p][p_ext],
                                          gamma_renumbered_cells_ext[p][p_ext],
                                          gamma_dofs_cells_ext[p][p_ext],
                                          gamma_geoms_cells_ext[p][p_ext],
                                          )
        A.assemble()
        return A

    quadrature_degree = 2
    A_lr = assemble_robin_matrix(p_left, p_right, quadrature_degree)
    A_rl = assemble_robin_matrix(p_right, p_left, quadrature_degree)
    diag_matrix = {}
    rhs_vector = {}
    neumann_facets = {}
    for p, marker in zip([p_left, p_right], [left_marker_neumann, right_marker_neumann]):
        p.set_forms_domain()
        # Set-up remaining terms
        neumann_tag = 66
        neumann_facets[p] = mesh.locate_entities(p.domain, p.dim-1, marker)
        neumann_int_ents = p.get_facet_integrations_entities(neumann_facets[p])
        gamma_tag = 44
        gamma_integral_ents = p.get_facet_integrations_entities()
        subdomain_data = [(neumann_tag, np.asarray(neumann_int_ents, dtype=np.int32)),
                          (gamma_tag, np.asarray(gamma_integral_ents, dtype=np.int32))]
        # Neumann condition
        ds = ufl.Measure('ds', domain=p.domain, subdomain_data=subdomain_data)
        n = ufl.FacetNormal(p.domain)
        v = ufl.TestFunction(p.v)
        p.l_ufl += +ufl.inner(n, p.k * grad_exact_sol(p.domain)) * v * ds(neumann_tag)
        # LHS term Robin
        dS = ufl.Measure('ds', domain=p.domain, subdomain_data=subdomain_data)
        p.a_ufl += + p.u * v * dS(gamma_tag)

        p.compile_forms()
        # Pre-assemble
        p.pre_assemble()
        p.assemble_jacobian()
        p.assemble_residual()
        diag_matrix[p] = p.A
        rhs_vector[p] = p.L

    # Create nest system
    A = petsc4py.PETSc.Mat().createNest([[p_left.A, A_lr], [A_rl, p_right.A]])
    L = petsc4py.PETSc.Vec().createNest([p_left.L, p_right.L])
    A.assemble()
    L.assemble()

    # TODO: Solve
    l_cpp = [p_left.mr_compiled, p_right.mr_compiled]
    restriction = [p_left.restriction, p_right.restriction]
    ulur = multiphenicsx.fem.petsc.create_vector_nest(l_cpp, restriction=restriction)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(p_left.domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    petsc4py.PETSc.Options().setValue('-ksp_error_if_not_converged', 'true')
    ksp.getPC().setType("lu")
    #ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    '''
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=7, ival=4)
    '''
    ksp.setFromOptions()
    ksp.solve(L, ulur)
    print(f"rank {rank}, converged reason = {ksp.getConvergedReason()}")
    for ulur_sub in ulur.getNestSubVecs():
        ulur_sub.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()

    # Split the block solution in components
    with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(
            ulur, [p_left.v.dofmap, p_right.v.dofmap], restriction) as ulur_wrapper:
        for ulur_wrapper_local, component in zip(ulur_wrapper, (p_left.u, p_right.u)):
            with component.x.petsc_vec.localForm() as component_local:
                component_local[:] = ulur_wrapper_local
    ulur.destroy()
    for mat in [L, A]:
        mat.destroy()

    p_left.writepos(extra_funcs=[ext_conductivity[p_left], p_left.dirichlet_bcs[0].g])
    p_right.writepos(extra_funcs=[p_right.dirichlet_bcs[0].g])


if __name__=="__main__":
    main()
