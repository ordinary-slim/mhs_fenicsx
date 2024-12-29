from dolfinx import fem, mesh, la
import numpy as np
import basix, basix.ufl
from mpi4py import MPI
from main import exact_sol_2d, grad_exact_sol_2d, Rhs, \
                 exact_sol_3d, grad_exact_sol_3d
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
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

rhs = Rhs(params["material"]["density"],
          params["material"]["specific_heat"],
          params["material"]["conductivity"],
          params["advection_speed"],
          params["dim"])

def left_marker_dirichlet(x):
    return np.isclose(x[0],0)
def right_marker_dirichlet(x):
    return np.isclose(x[0],1)
def left_marker_neumann_2d(p:Problem):
    return mesh.locate_entities(p.domain, p.dim-1,
                                lambda x : np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] <= 0.5)
                                )
def right_marker_neumann_2d(p:Problem):
    return mesh.locate_entities(p.domain, p.dim-1,
                                lambda x : np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] >= 0.5)
                                )
def left_marker_neumann_3d(p:Problem):
    y_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[1], 0),
                                          np.isclose(x[1], 1)),
                            x[0] <= 0.5))
    z_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[2], 0),
                                          np.isclose(x[2], 1)),
                            x[0] <= 0.5))
    return np.hstack((y_facets,z_facets))

def left_marker_neumann_3d_debug(p:Problem):
    y_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[1], 0),
                                          np.isclose(x[1], 1)),
                            x[0] <= 0.5))
    z_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[2], 0),
                                          np.isclose(x[2], 1)),
                            x[0] <= 0.5))
    m_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.isclose(x[0], 0.5))
    return np.hstack((y_facets,z_facets,m_facets))

def right_marker_neumann_3d(p:Problem):
    y_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[1], 0),
                                          np.isclose(x[1], 1)),
                            x[0] >= 0.5))
    z_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[2], 0),
                                          np.isclose(x[2], 1)),
                            x[0] >= 0.5))
    return np.hstack((y_facets,z_facets))

def marker_gamma(x):
    return np.isclose(x[0], 0.5)

def get_matrix_row_as_func(p : Problem, p_ext:Problem, mat):
    ''' Debugging util '''
    if p.dim==2:
        middle_dof = fem.locate_dofs_geometrical(p.v, lambda x : np.logical_and(np.isclose(x[0], 0.5), np.isclose(x[1], 0.5)))
    else:
        middle_dof = fem.locate_dofs_geometrical(p.v, lambda x : np.logical_and(np.logical_and(np.isclose(x[0], 0.5), np.isclose(x[1], 0.5)), np.isclose(x[2], 0.5)))
    res_left = p.restriction
    res_right = p_ext.restriction
    middle_dof = middle_dof[:np.searchsorted(middle_dof, res_left.dofmap.index_map.size_local)]
    vals = np.zeros(res_right.dofmap.index_map.size_global)
    found = np.array([0], dtype=np.int32)
    found_mask = np.zeros(comm.size,dtype=np.int32)
    if middle_dof.size:
        row_middle_dof = res_left.unrestricted_to_restricted[middle_dof[0]]
        grow_middle_dof = res_left.index_map.local_to_global(np.array([row_middle_dof]))
        c, v = mat.getRow(grow_middle_dof)
        vals[c] = v
        found[0] = 1
    comm.Allgather(found, found_mask)
    middle_dof_rank = found_mask.nonzero()[0][0]
    comm.Bcast(vals, root=middle_dof_rank)
    # TODO: middle_dof_rank is restricted, unrestrict
    middle_row_func = fem.Function(p_ext.v, name="middle_row_f")
    rgrange = np.arange(res_right.index_map.local_range[0],
                        res_right.index_map.local_range[1])
    rlrange = np.arange(rgrange.size)
    for rgidx, rlidx in zip(rgrange, rlrange):
        ulidx = res_right.restricted_to_unrestricted[rlidx]
        middle_row_func.x.array[ulidx] = vals[rgidx]
    middle_row_func.x.scatter_forward()
    return middle_row_func

def main():
    # Mesh and problems
    els_side = params["els_side"]
    dim = params["dim"]

    if dim==2:
        left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.quadrilateral)
        right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.triangle)
        exact_sol = exact_sol_2d
        grad_exact_sol = grad_exact_sol_2d
        left_marker_neumann = left_marker_neumann_2d
        left_marker_neumann_debug = left_marker_neumann_2d_debug
        right_marker_neumann = right_marker_neumann_2d
    else:
        cell_type = mesh.CellType.tetrahedron
        if params["cell_type"] == "hexa":
            cell_type = mesh.CellType.hexahedron
        left_mesh  = mesh.create_unit_cube(MPI.COMM_WORLD, els_side, els_side, els_side, cell_type=cell_type)
        right_mesh  = mesh.create_unit_cube(MPI.COMM_WORLD, els_side, els_side, els_side, cell_type=cell_type)
        exact_sol = exact_sol_3d
        grad_exact_sol = grad_exact_sol_3d
        left_marker_neumann = left_marker_neumann_3d
        left_marker_neumann_debug = left_marker_neumann_3d_debug
        right_marker_neumann = right_marker_neumann_3d

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

    gamma_cells = {
            p_left : p_left.gamma_integration_data[::2],
            p_right : p_right.gamma_integration_data[::2],
            }
    gamma_qpoints_po = {
            p_left : { p_right : None },
            p_right : { p_left : None },
            }
    gamma_renumbered_cells_ext = { p_left : {}, p_right : {}}
    gamma_dofs_cells_ext = { p_left : {}, p_right : {}}
    gamma_geoms_cells_ext = { p_left : {}, p_right : {}}
    gamma_iid = {p_left:{}, p_right:{}}
    ext_conductivity = {}
    midpoints_gamma = {p_left:None, p_right:None}

    def assemble_robin_matrix(p:Problem, p_ext:Problem, quadrature_degree=2):
        cdim = p.domain.topology.dim
        fdim = cdim - 1
        # GENERATE QUADRATURE
        cell_type =  p.domain.topology.entity_types[-1][0].name
        facet_type = p.domain.topology.entity_types[-2][0].name
        Qe = basix.ufl.quadrature_element(facet_type,
                                          degree=quadrature_degree)

        num_gps_facet = Qe.num_entity_dofs[-1][0]
        num_facets_cell = p.domain.ufl_cell().num_facets()
        quadrature_points_cell  = np.zeros((num_gps_facet * num_facets_cell, cdim), dtype=Qe._points.dtype)
        for ifacet in range(num_facets_cell):
            quadrature_points_cell[ifacet*num_gps_facet : ifacet*num_gps_facet + num_gps_facet, :cdim] = map_facet_points(Qe._points, ifacet, cell_type)

        # Manually tabulate
        num_local_gamma_cells = p.gamma_integration_data.size // 2
        gamma_qpoints = np.zeros((num_local_gamma_cells * \
                num_gps_facet, 3), dtype=np.float64)
        pgeo = p.domain.geometry
        for idx in range(num_local_gamma_cells):
            icell   = p.gamma_integration_data[2*idx]
            lifacet = p.gamma_integration_data[2*idx+1]
            ref_points = quadrature_points_cell[lifacet*num_gps_facet:lifacet*num_gps_facet+num_gps_facet, :]
            # Push forward
            gamma_qpoints[idx*num_gps_facet:\
                idx*num_gps_facet + num_gps_facet, :] =  \
                pgeo.cmap.push_forward(
                        ref_points, pgeo.x[pgeo.dofmap[icell]])

        gamma_qpoints_po[p][p_ext] = \
                cellwise_determine_point_ownership(p_ext.domain._cpp_object,
                                                   gamma_qpoints,
                                                   gamma_cells[p_ext],
                                                   np.float64(1e-7))
        gamma_renumbered_cells_ext[p][p_ext], \
        gamma_dofs_cells_ext[p][p_ext], \
        gamma_geoms_cells_ext[p][p_ext] = \
                        scatter_cell_integration_data_po(gamma_qpoints_po[p][p_ext],
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

        p.domain.topology.create_entity_permutations()
        A = create_robin_robin_monolithic(ext_conductivity[p]._cpp_object,
                                          gamma_qpoints,
                                          quadrature_points_cell,
                                          Qe._weights,
                                          p.v._cpp_object,
                                          p.restriction,
                                          p_ext.v._cpp_object,
                                          p_ext.restriction,
                                          p.gamma_integration_data,
                                          gamma_qpoints_po[p][p_ext],
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
        neumann_facets[p] = marker(p)
        neumann_int_ents = p.get_facet_integrations_entities(neumann_facets[p])
        gamma_tag = 44
        subdomain_data = [(neumann_tag, np.asarray(neumann_int_ents, dtype=np.int32)),
                          (gamma_tag, np.asarray(p.gamma_integration_data, dtype=np.int32))]
        # Neumann condition
        ds = ufl.Measure('ds', domain=p.domain, subdomain_data=subdomain_data)
        n = ufl.FacetNormal(p.domain)
        v = ufl.TestFunction(p.v)
        p.l_ufl += +p.k * ufl.inner(n, grad_exact_sol(p.domain)) * v * ds(neumann_tag)
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
    #A.assemble()
    #L.assemble()

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
    #ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=7, ival=4)
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

    middle_row_func_rl = get_matrix_row_as_func(p_right, p_left, A_rl)
    middle_row_func_lr = get_matrix_row_as_func(p_left, p_right, A_lr)

    for mat in [L, A, A_lr, A_rl]:
        mat.destroy()

    p_left.writepos(extra_funcs=[ext_conductivity[p_left], p_left.dirichlet_bcs[0].g, middle_row_func_rl])
    p_right.writepos(extra_funcs=[p_right.dirichlet_bcs[0].g, middle_row_func_lr])

if __name__=="__main__":
    lp = LineProfiler()
    lp.add_module(Problem)
    lp_wrapper = lp(main)
    lp_wrapper()
    with open(f"profiling_diff_mesh_robin_rank{rank}.txt", 'w') as pf:
        lp.print_stats(stream=pf)
