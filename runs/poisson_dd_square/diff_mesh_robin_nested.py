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
def left_marker_neumann_2d(x):
    return np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] <= 0.5)
def left_marker_neumann_2d_debug(x):
    return np.logical_or(
               np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] <= 0.5),
               np.isclose(x[0], 0.5))
def right_marker_neumann_2d(x):
    return np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] >= 0.5)
def left_marker_neumann_3d(x):
    return np.logical_or(
            np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] <= 0.5),
            np.logical_and(np.logical_or(np.isclose(x[2],0), np.isclose(x[2],1)), x[0] <= 0.5))
def left_marker_neumann_3d_debug(x):
    return np.logical_or(np.isclose(x[0], 0.5),
            np.logical_or(
                np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] <= 0.5),
                np.logical_and(np.logical_or(np.isclose(x[2],0), np.isclose(x[2],1)), x[0] <= 0.5))
          )
def right_marker_neumann_3d(x):
    return np.logical_or(
            np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] >= 0.5),
            np.logical_and(np.logical_or(np.isclose(x[2],0), np.isclose(x[2],1)), x[0] >= 0.5))
def marker_gamma(x):
    return np.isclose(x[0], 0.5)

def get_matrix_row_as_func(p : Problem, p_ext:Problem, mat):
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
        left_mesh  = mesh.create_unit_cube(MPI.COMM_WORLD, els_side, els_side, els_side, mesh.CellType.hexahedron)
        right_mesh = mesh.create_unit_cube(MPI.COMM_WORLD, els_side, els_side, els_side)
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
    gamma_facets = {}
    gamma_cells  = {}
    # TODO: Check if this is slow
    # TODO: Rework this
    # NOTE: Gamma facets are local but owner cells can be ghosted!
    for p in [p_left, p_right]:
        gamma_facets[p] = p.gamma_facets.find(1) # local
        # TODO: Go over all gamma facets
        gamma_cells[p]  = np.zeros_like(gamma_facets[p],dtype=np.int32)
        gc = gamma_cells[p]
        con_facet_cell = p.domain.topology.connectivity(p.dim-1,p.dim)
        for idx, ifacet in enumerate(gamma_facets[p]):
            local_con = con_facet_cell.links(ifacet)
            if p.active_els_func.x.array[local_con[0]]:
                gc[idx] = local_con[0]
            else:
                gc[idx] = local_con[1]
            if gc[idx] > p.cell_map.size_local:
                print(f"Rank {rank}, ifacet {ifacet} (/{p.facet_map.size_local}) is connected to {gc[idx]}(/{p.cell_map.size_local})", flush=True)

    gamma_mesh = {}#TODO: Remove
    Qs_gamma = {}#TODO: Remove
    Qs_gamma_x = {}#TODO: Check occurences
    Qs_gamma_po = { p_left : {}, p_right : {}}#TODO: Check occurences
    gamma_renumbered_cells_ext = { p_left : {}, p_right : {}}
    gamma_dofs_cells_ext = { p_left : {}, p_right : {}}
    gamma_geoms_cells_ext = { p_left : {}, p_right : {}}
    gamma_iid = {p_left:{}, p_right:{}}
    ext_conductivity = {}
    midpoints_gamma = {p_left:None, p_right:None}
    # Robin coupling: i = left, j = right
    for p in [p_left, p_right]:
        gamma_mesh[p], \
        ent_map, _, _ = \
                mesh.create_submesh(p.domain,p.dim-1,gamma_facets[p])
        #print(f"Rank {rank}, entity map = {ent_map}, size local = {gamma_mesh[p].topology.index_map(p.dim-1).size_local}", flush=True)

    def assemble_robin_matrix(p:Problem, p_ext:Problem, quadrature_degree=2):
        cdim = p.domain.topology.dim
        fdim = cdim - 1
        # GENERATE QUADRATURE
        cell_type =  p.domain.topology.entity_types[-1][0].name
        facet_type = p.domain.topology.entity_types[-2][0].name
        Qe = basix.ufl.quadrature_element(facet_type,
                                          degree=quadrature_degree)

        # GENERATE ALL PERMUTATIONS OF QUADRATURE
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

        num_unpermuted_gps_per_facet = Qe.num_entity_dofs[-1][0]
        # TODO: manually tabulate
        num_local_gamma_cells = p.gamma_lcell_facet_map.size
        gamma_qpoints = np.zeros((num_local_gamma_cells * \
                num_unpermuted_gps_per_facet, 3), dtype=np.float64)
        pgeo = p.domain.geometry
        for idx in range(num_local_gamma_cells):
            # TODO: Make points on ref cell. Maybe tabulate outside of this loop?
            icell   = p.gamma_integration_data[2*idx]
            lifacet = p.gamma_integration_data[2*idx+1]
            # Grab unpermuted points. Could also grab permuted points respecting facet permutation
            ref_points = permuted_quadrature_points_cell[lifacet*num_gps_facet:lifacet*num_gps_facet+num_unpermuted_gps_per_facet, :]
            # Push forward
            gamma_qpoints[idx*num_unpermuted_gps_per_facet:\
                idx*num_unpermuted_gps_per_facet + num_unpermuted_gps_per_facet, :] =  \
                pgeo.cmap.push_forward(
                        ref_points, pgeo.x[pgeo.dofmap[icell]])
        print(f"Rank {rank}, Qs_gamma_x = {Qs_gamma_x[p_left]}", flush=True)
        print(f"Rank {rank}, gamma_qpoints = {gamma_qpoints}", flush=True)
        exit()


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
    # DEBUG
    owner_cells_po_lr = fem.Function(p_right.dg0, name="owner_cells_po")
    for idx, cell in enumerate(Qs_gamma_po[p_left][p_right].dest_cells):
        print(f"Rank {rank}, Point {Qs_gamma_po[p_left][p_right].dest_points[idx]} is owned by cell {cell} but local size {p_right.cell_map.size_local}")
        cells = fem.locate_dofs_topological(p_right.dg0, p_right.dim,
                                           np.asarray([cell], dtype=np.int32),
                                           remote=False)
        assert(len(cells)==1)
        owner_cells_po_lr.x.array[cells[0]] += 1
    s = comm.allreduce(owner_cells_po_lr.x.array.sum())
    num_gamma_facets1 = comm.allreduce(len(gamma_facets[p_left]))
    num_gamma_facets2 = comm.allreduce(len(p_left.gamma_lcell_facet_map))
    if rank == 0:
        print(f"Sum dest_cells as func = {s}", flush=True)
        print(f"num gamma facets OLD = {num_gamma_facets1}, num_gamma_facets NEW = {num_gamma_facets2}", flush=True)
    owner_cells_po_lr.x.scatter_forward()
    # EDEBUG
    #A_rl = assemble_robin_matrix(p_right, p_left, quadrature_degree)
    diag_matrix = {}
    rhs_vector = {}
    neumann_facets = {}
    neumann_dofs_debug = {}
    for p, marker in zip([p_left, p_right], [left_marker_neumann_debug, right_marker_neumann]):
        p.set_forms_domain()
        # Set-up remaining terms
        neumann_tag = 66
        neumann_facets[p] = mesh.locate_entities(p.domain, p.dim-1, marker)
        neumann_int_ents = p.get_facet_integrations_entities(neumann_facets[p])
        ##### BDEBUG
        from mhs_fenicsx.problem import indices_to_function
        neumann_dofs_debug[p] = indices_to_function(p.v, neumann_facets[p],
                                           p.dim-1,name="neumann")
        ##### EDEBUG
        gamma_tag = 44
        subdomain_data = [(neumann_tag, np.asarray(neumann_int_ents, dtype=np.int32)),
                          (gamma_tag, np.asarray(p.gamma_integration_data, dtype=np.int32))]
        # Neumann condition
        ds = ufl.Measure('ds', domain=p.domain, subdomain_data=subdomain_data)
        n = ufl.FacetNormal(p.domain)
        v = ufl.TestFunction(p.v)
        p.l_ufl += +ufl.inner(n, p.k * grad_exact_sol(p.domain)) * v * ds(neumann_tag)
        # LHS term Robin
        if p == p_right:
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
    #A = petsc4py.PETSc.Mat().createNest([[p_left.A, None], [A_rl, p_right.A]])
    L = petsc4py.PETSc.Vec().createNest([p_left.L, p_right.L])
    #A.assemble()
    L.assemble()

    '''
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
    for mat in [L, A]:
        mat.destroy()
    '''

    #middle_row_func_rl = get_matrix_row_as_func(p_right, p_left, A_rl)
    middle_row_func_lr = get_matrix_row_as_func(p_left, p_right, A_lr)

    #p_left.writepos(extra_funcs=[ext_conductivity[p_left], p_left.dirichlet_bcs[0].g, neumann_dofs_debug[p_left], middle_row_func_rl])
    p_right.writepos(extra_funcs=[p_right.dirichlet_bcs[0].g, neumann_dofs_debug[p_right], middle_row_func_lr, owner_cells_po_lr])

if __name__=="__main__":
    lp = LineProfiler()
    lp_wrapper = lp(main)
    lp_wrapper()
    with open(f"profiling_diff_mesh_robin_rank{rank}.txt", 'w') as pf:
        lp.print_stats(stream=pf)
