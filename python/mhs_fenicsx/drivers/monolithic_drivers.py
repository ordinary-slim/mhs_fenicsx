from mhs_fenicsx.problem import Problem
from dolfinx import fem, mesh, la
import basix.ufl
from mhs_fenicsx_cpp import cellwise_determine_point_ownership, scatter_cell_integration_data_po, \
                            create_robin_robin_monolithic, interpolate_dg0_at_facets, \
                            tabulate_gamma_quadrature
from mhs_fenicsx.problem.helpers import get_identity_maps
from ffcx.element_interface import map_facet_points
import numpy as np
from petsc4py import PETSc
import multiphenicsx, multiphenicsx.fem.petsc
import ufl

class MonolithicDomainDecompositionDriver:
    def __init__(self, sub_problem_1:Problem,sub_problem_2:Problem, quadrature_degree):
        (p1,p2) = (sub_problem_1,sub_problem_2)
        self.p1 = p1
        self.p2 = p2
        self.quadrature_degree = quadrature_degree

    def setup_coupling(self):
        ''' Find interface '''
        (p1,p2) = (self.p1,self.p2)
        self.gamma_cells = {
                    p1 : p1.gamma_integration_data[p2][::2],
                    p2 : p2.gamma_integration_data[p1][::2],
                    }
        self.gamma_qpoints_po = {
                    p1 : { p2 : None },
                    p2 : { p1 : None },
                    }
        self.gamma_renumbered_cells_ext = { p1 : {}, p2 : {}}
        self.gamma_dofs_cells_ext = { p1 : {}, p2 : {}}
        self.gamma_geoms_cells_ext = { p1 : {}, p2 : {}}
        self.gamma_iid = {p1:{}, p2:{}}
        self.ext_conductivity = {}
        self.midpoints_gamma = {p1:None, p2:None}

    def solve(self):
        ''' Solve '''
        # Create nest system
        (p1, p2) = (self.p1, self.p2)
        self.A = PETSc.Mat().createNest([[p1.A, self.A12], [self.A21, p2.A]])
        self.L = PETSc.Vec().createNest([p1.L, p2.L])
        # SOLVE
        l_cpp = [p1.mr_instance, p2.mr_instance]
        restriction = [p1.restriction, p2.restriction]
        du1du2 = multiphenicsx.fem.petsc.create_vector_nest(l_cpp, restriction=restriction)
        ksp = PETSc.KSP()
        ksp.create(p1.domain.comm)
        ksp.setOperators(self.A)
        ksp.setType("preonly")
        PETSc.Options().setValue('-ksp_error_if_not_converged', 'true')
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.getPC().setFactorSetUpSolverType()
        ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=7, ival=4)
        #ksp.setFromOptions()
        ksp.solve(self.L, du1du2)
        for du1du2_sub in du1du2.getNestSubVecs():
            du1du2_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        ksp.destroy()
        # Split the block solution in components
        with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(
                du1du2, [p1.v.dofmap, p2.v.dofmap], restriction) as u1u2_wrapper:
            for u1u2_wrapper_local, component in zip(u1u2_wrapper, (p1.du, p2.du)):
                with component.x.petsc_vec.localForm() as component_local:
                    component_local[:] = u1u2_wrapper_local
        du1du2.destroy()
        for p in [p1, p2]:
            p.u.x.array[:] += p.du.x.array[:]
            p.is_grad_computed = False

    def post_iterate(self):
        for mat in [self.A12, self.A21]:
            mat.destroy()

class MonolithicRRDriver(MonolithicDomainDecompositionDriver):
    def __init__(self, sub_problem_1:Problem,sub_problem_2:Problem, quadrature_degree):
        (p1,p2) = (sub_problem_1,sub_problem_2)
        super().__init__(p1, p2, quadrature_degree)
        self.compile_forms()
        self.Qe = dict()
        self.quadrature_points_cell = dict()
        for p in (p1, p2):
            cdim = p.domain.topology.dim
            cell_type =  p.domain.topology.entity_types[-1][0].name
            facet_type = p.domain.topology.entity_types[-2][0].name
            self.Qe[p] = basix.ufl.quadrature_element(facet_type, degree=quadrature_degree)
            num_gps_facet = self.Qe[p].num_entity_dofs[-1][0]
            num_facets_cell = p.domain.ufl_cell().num_facets()
            self.quadrature_points_cell[p]  = np.zeros((num_gps_facet * num_facets_cell, cdim), dtype=self.Qe[p]._points.dtype)
            for ifacet in range(num_facets_cell):
                self.quadrature_points_cell[p][ifacet*num_gps_facet : ifacet*num_gps_facet + num_gps_facet, :cdim] = map_facet_points(self.Qe[p]._points, ifacet, cell_type)

    def compile_forms(self):
        '''TODO: Make this domain independent'''
        (p1,p2) = (self.p1,self.p2)
        self.gamma_integration_tag = 44
        self.r_ufl, self.j_ufl = {}, {}
        self.r_compiled, self.j_compiled = {}, {}
        for p in [p1, p2]:
            # LHS term Robin
            ds = ufl.Measure('ds')
            (u, v) = (p.u, ufl.TestFunction(p.v))
            a_ufl  = + u * v * ds(self.gamma_integration_tag)
            self.j_ufl[p] = ufl.derivative(a_ufl, p.u)
            # LOC CONTRIBUTION
            self.r_ufl[p] = a_ufl
            self.r_compiled[p] = fem.compile_form(p.domain.comm, self.r_ufl[p],
                                               form_compiler_options={"scalar_type": np.float64})
            self.j_compiled[p] = fem.compile_form(p.domain.comm, self.j_ufl[p],
                                               form_compiler_options={"scalar_type": np.float64})

    def instantiate_forms(self):
        (p1,p2) = (self.p1,self.p2)
        self.r_instance, self.j_instance = {}, {}
        for p, p_ext in [(p1, p2), (p2, p1)]:
            rcoeffmap, rconstmap = get_identity_maps(self.r_ufl[p])
            form_subdomain_data = {fem.IntegralType.exterior_facet :
                                   [(self.gamma_integration_tag, p.gamma_integration_data[p_ext])]
                                     }
            self.r_instance[p] = fem.create_form(self.r_compiled[p],
                                                 [p.v],
                                                 msh=p.domain,
                                                 subdomains=form_subdomain_data,
                                                 coefficient_map=rcoeffmap,
                                                 constant_map=rconstmap)
            lcoeffmap, lconstmap = get_identity_maps(self.j_ufl[p])
            self.j_instance[p] = fem.create_form(self.j_compiled[p],
                                                 [p.v, p.v],
                                                 msh=p.domain,
                                                 subdomains=form_subdomain_data,
                                                 coefficient_map=lcoeffmap,
                                                 constant_map=lconstmap)

    def setup_coupling(self):
        # WARNING: Incompatibility with strongly enforced Dirichlet BC!
        super().setup_coupling()
        (p1,p2) = (self.p1,self.p2)
        self.A12 = self.assemble_robin_matrix(p1, p2)
        self.A21 = self.assemble_robin_matrix(p2, p1)
        # Add LHS term
        for p, p_ext in zip([p1, p2], [p2, p1]):
            self.instantiate_forms()
            p.A.assemble(PETSc.Mat.AssemblyType.FLUSH)
            multiphenicsx.fem.petsc.assemble_matrix(
                    p.A,
                    self.j_instance[p],
                    bcs=p.dirichlet_bcs,
                    restriction=(p.restriction, p.restriction))
            # RHS term Robin for residual formulation
            res = p.restriction
            ext_res = p_ext.restriction
            in_flux = la.create_petsc_vector(res.index_map, p.v.value_size)
            ext_flux = la.create_petsc_vector(res.index_map, p.v.value_size)

            # EXT CONTRIBUTION
            indices = np.array(tuple(ext_res.restricted_to_unrestricted.values())[:ext_res.index_map.size_local],dtype=np.int32)
            indices = ext_res.dofmap.index_map.local_to_global(indices).astype(np.int32)
            iset = PETSc.IS().createGeneral(indices, p_ext.domain.comm)
            ext_res_sol = p_ext.u.x.petsc_vec.getSubVector(iset)
            A_coupling = self.A12
            if p_ext == p1:
                A_coupling = self.A21
            A_coupling.mult(ext_res_sol, ext_flux)
            ext_flux.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            p_ext.u.x.petsc_vec.restoreSubVector(iset, ext_res_sol)

            multiphenicsx.fem.petsc.assemble_vector(
                    in_flux,
                    self.r_instance[p],
                    restriction=p.restriction)
            in_flux.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

            # Add
            p.L += - in_flux - ext_flux
            for f in [in_flux, ext_flux]:
                f.destroy()
            p.L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def assemble_robin_matrix(self, p:Problem, p_ext:Problem):
        # GENERATE QUADRATURE
        # Manually tabulate
        num_gps_facet = self.Qe[p].num_entity_dofs[-1][0]
        gamma_qpoints = tabulate_gamma_quadrature(
                p.domain._cpp_object,
                p.gamma_integration_data[p_ext],
                num_gps_facet,
                self.quadrature_points_cell[p]
                )

        self.gamma_qpoints_po[p][p_ext] = \
                cellwise_determine_point_ownership(p_ext.domain._cpp_object,
                                                   gamma_qpoints,
                                                   self.gamma_cells[p_ext],
                                                   np.float64(1e-7))
        self.gamma_renumbered_cells_ext[p][p_ext], \
        self.gamma_dofs_cells_ext[p][p_ext], \
        self.gamma_geoms_cells_ext[p][p_ext] = \
                        scatter_cell_integration_data_po(self.gamma_qpoints_po[p][p_ext],
                                                          p_ext.v._cpp_object,
                                                          p_ext.restriction)
        self.midpoints_gamma[p] = mesh.compute_midpoints(p.domain,p.domain.topology.dim-1,p.gamma_facets[p_ext].find(1))
        self.gamma_iid[p][p_ext] = cellwise_determine_point_ownership(
                                            p_ext.domain._cpp_object,
                                            self.midpoints_gamma[p],
                                            self.gamma_cells[p_ext],
                                            np.float64(1e-6))
        self.ext_conductivity[p] = fem.Function(p.dg0, name="ext_k")
        interpolate_dg0_at_facets([p_ext.k._cpp_object],
                                  [self.ext_conductivity[p]._cpp_object],
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets[p_ext]._cpp_object,
                                  self.gamma_cells[p],
                                  self.gamma_iid[p][p_ext],
                                  p.gamma_facets_index_map[p_ext],
                                  p.gamma_imap_to_global_imap[p_ext])

        A = create_robin_robin_monolithic(self.ext_conductivity[p]._cpp_object,
                                          gamma_qpoints,
                                          self.quadrature_points_cell[p],
                                          self.Qe[p]._weights,
                                          p.v._cpp_object,
                                          p.restriction,
                                          p_ext.v._cpp_object,
                                          p_ext.restriction,
                                          p.gamma_integration_data[p_ext],
                                          self.gamma_qpoints_po[p][p_ext],
                                          self.gamma_renumbered_cells_ext[p][p_ext],
                                          self.gamma_dofs_cells_ext[p][p_ext],
                                          self.gamma_geoms_cells_ext[p][p_ext],
                                          )
        A.assemble()
        return A
