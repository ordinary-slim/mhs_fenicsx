from mhs_fenicsx.problem import Problem
from dolfinx import fem, mesh, la
import basix.ufl
from mhs_fenicsx_cpp import cellwise_determine_point_ownership, scatter_cell_integration_data_po, \
                            MonolithicRobinRobinAssembler64, interpolate_dg0_at_facets, \
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
        self.gamma_qpoints = {
                    p1 : None,
                    p2 : None,
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

    def post_iterate(self):
        pass

class MonolithicRRDriver(MonolithicDomainDecompositionDriver):
    def __init__(self, sub_problem_1:Problem,sub_problem_2:Problem, quadrature_degree):
        (p1,p2) = (sub_problem_1,sub_problem_2)
        super().__init__(p1, p2, quadrature_degree)
        self.compile_forms()
        self.Qe = dict()
        self.quadrature_points_cell = dict()
        self.monolithicRrAssembler = dict()
        for p, p_ext in zip((p1, p2), (p2, p1)):
            self.monolithicRrAssembler[p] = MonolithicRobinRobinAssembler64()
            cdim = p.domain.topology.dim
            cell_type =  p.domain.topology.entity_types[-1][0].name
            facet_type = p.domain.topology.entity_types[-2][0].name
            points, weights= None, None
            if facet_type=='point':
                points  = np.array([[0.0]], dtype=np.float64)
                weights = np.array([1.0], dtype=np.float64)
                quadrature_degree = 1
            self.Qe[p] = basix.ufl.quadrature_element(facet_type,
                                                      points=points,
                                                      weights=weights,
                                                      degree=quadrature_degree,
                                                      )
            num_gps_facet = self.Qe[p].num_entity_dofs[-1][0]
            num_facets_cell = p.domain.ufl_cell().num_facets()
            self.quadrature_points_cell[p]  = np.zeros((num_gps_facet * num_facets_cell, cdim), dtype=self.Qe[p]._points.dtype)
            for ifacet in range(num_facets_cell):
                self.quadrature_points_cell[p][ifacet*num_gps_facet : ifacet*num_gps_facet + num_gps_facet, :cdim] = map_facet_points(self.Qe[p]._points, ifacet, cell_type)

    def __del__(self):
        self._destroy()

    def compile_forms(self):
        '''TODO: Make this domain independent'''
        (p1,p2) = (self.p1,self.p2)
        assert(p1.materials == p2.materials)
        base_tag = 500
        self.gamma_integration_tags = {}
        for idx, mat in enumerate(p1.materials):
            self.gamma_integration_tags[mat] = base_tag+idx
        self.r_ufl, self.j_ufl = {}, {}
        self.r_compiled, self.j_compiled = {}, {}
        ds = ufl.Measure('ds')
        for p in [p1, p2]:
            # LHS term Robin
            (u, v) = (p.u, ufl.TestFunction(p.v))
            a_ufl = []
            for mat in p.materials:
                a_ufl.append(+ u * v * ds(self.gamma_integration_tags[mat]))
            a_ufl = sum(a_ufl)
            self.j_ufl[p] = ufl.derivative(a_ufl, p.u)
            # LOC CONTRIBUTION
            self.r_ufl[p] = a_ufl
            self.r_compiled[p] = fem.compile_form(p.domain.comm, self.r_ufl[p],
                                               form_compiler_options={"scalar_type": np.float64})
            self.j_compiled[p] = fem.compile_form(p.domain.comm, self.j_ufl[p],
                                               form_compiler_options={"scalar_type": np.float64})

    def set_form_subdomain_data(self):
        (p1,p2) = (self.p1,self.p2)
        self.form_subdomain_data = {p1:[], p2:[]}
        for p, p_ext in [(p1, p2), (p2, p1)]:
            self.form_subdomain_data[p] = p.get_facets_subdomain_data(p.gamma_integration_data[p_ext], self.gamma_integration_tags)

    def instantiate_forms(self):
        (p1,p2) = (self.p1,self.p2)
        self.r_instance, self.j_instance = {}, {}
        self.set_form_subdomain_data()
        for p, p_ext in [(p1, p2), (p2, p1)]:
            rcoeffmap, rconstmap = get_identity_maps(self.r_ufl[p])
            form_subdomain_data = {fem.IntegralType.exterior_facet :
                                   self.form_subdomain_data[p]}
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

    def contribute_to_diagonal_blocks(self):
        (p1,p2) = (self.p1,self.p2)
        for p, p_ext in zip([p1, p2], [p2, p1]):
            p.A.assemble(PETSc.Mat.AssemblyType.FLUSH)
            multiphenicsx.fem.petsc.assemble_matrix(
                    p.A,
                    self.j_instance[p],
                    bcs=p.dirichlet_bcs,
                    restriction=(p.restriction, p.restriction))

    def contribute_to_residuals(self, R_vec):
        (p1,p2) = (self.p1,self.p2)
        for p, p_ext, R_sub_vec in zip([p1, p2], [p2, p1], R_vec.getNestSubVecs()):
            # RHS term Robin for residual formulation
            res = p.restriction
            ext_res = p_ext.restriction
            in_flux = la.create_petsc_vector(res.index_map, p.v.value_size)
            ext_flux = la.create_petsc_vector(res.index_map, p.v.value_size)

            # EXT CONTRIBUTION
            indices = np.array(tuple(ext_res.restricted_to_unrestricted.values()) ,dtype=np.int32)
            indices.sort()
            indices = indices[:ext_res.index_map.size_local]
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
            R_sub_vec += - in_flux - ext_flux
            for f in [in_flux, ext_flux]:
                f.destroy()
            R_sub_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def preassemble_robin_matrix(self, p:Problem, p_ext:Problem):
        # GENERATE QUADRATURE
        # Manually tabulate
        num_gps_facet = self.Qe[p].num_entity_dofs[-1][0]
        self.gamma_qpoints[p] = tabulate_gamma_quadrature(
                p.domain._cpp_object,
                p.gamma_integration_data[p_ext],
                num_gps_facet,
                self.quadrature_points_cell[p]
                )
        self.gamma_qpoints_po[p][p_ext] = \
                cellwise_determine_point_ownership(p_ext.domain._cpp_object,
                                                   self.gamma_qpoints[p],
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
        return self.monolithicRrAssembler[p].preassemble(self.gamma_qpoints[p],
                                                         self.quadrature_points_cell[p],
                                                         self.Qe[p]._weights,
                                                         p.v._cpp_object,
                                                         p.restriction,
                                                         p_ext.v._cpp_object,
                                                         p_ext.restriction,
                                                         p.gamma_integration_data[p_ext],
                                                         self.gamma_renumbered_cells_ext[p][p_ext],
                                                         self.gamma_dofs_cells_ext[p][p_ext],
                                                         self.gamma_geoms_cells_ext[p][p_ext],
                                                         )

    def assemble_robin_matrix(self, p:Problem, p_ext:Problem):
        A_coupling = self.A12
        if p_ext == self.p1:
            A_coupling = self.A21
        A_coupling.zeroEntries()
        self.ext_conductivity[p] = fem.Function(p.dg0, name="ext_k")
        interpolate_dg0_at_facets([p_ext.k._cpp_object],
                                  [self.ext_conductivity[p]._cpp_object],
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets[p_ext]._cpp_object,
                                  self.gamma_cells[p],
                                  self.gamma_iid[p][p_ext],
                                  p.gamma_facets_index_map[p_ext],
                                  p.gamma_imap_to_global_imap[p_ext])

        self.monolithicRrAssembler[p].assemble(A_coupling, self.ext_conductivity[p]._cpp_object,
                                               self.gamma_qpoints[p],
                                               self.quadrature_points_cell[p],
                                               self.Qe[p]._weights,
                                               p.v._cpp_object,
                                               p.restriction,
                                               p_ext.v._cpp_object,
                                               p_ext.restriction,
                                               p.gamma_integration_data[p_ext],
                                               self.gamma_renumbered_cells_ext[p][p_ext],
                                               self.gamma_dofs_cells_ext[p][p_ext],
                                               )
        A_coupling.assemble()
        return A_coupling

    def pre_assemble(self):
        (p1,p2) = (self.p1,self.p2)
        self._destroy()
        self.restriction = [p1.restriction, p2.restriction]
        self.dofmaps = [p1.v.dofmap, p2.v.dofmap]
        self.setup_coupling()
        self.instantiate_forms()
        mr_instance = [p1.mr_instance, p2.mr_instance]

        # TODO: Efficient structure needed
        self.A12 = self.preassemble_robin_matrix(p1, p2)
        self.A21 = self.preassemble_robin_matrix(p2, p1)
        self.assemble_robin_matrix(p1, p2)
        self.assemble_robin_matrix(p2, p1)

        self.L = PETSc.Vec().createNest([p1.L, p2.L])
        self.A = PETSc.Mat().createNest([[p1.A, self.A12], [self.A21, p2.A]])
        self.x = multiphenicsx.fem.petsc.create_vector_nest(mr_instance, restriction=self.restriction)
        self.obj_vec = multiphenicsx.fem.petsc.create_vector_nest(mr_instance, restriction=self.restriction)

    def set_snes_sol_vector(self) -> PETSc.Vec:  # type: ignore[no-any-unimported]
        """
        Set PETSc.Vec to be passed to PETSc.SNES.solve to initial guess
        """
        (p1,p2) = (self.p1,self.p2)
        sol_vector = self.x
        with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(
                sol_vector, self.dofmaps, self.restriction) as nest_sol:
            for sol_sub, u_sub in zip(nest_sol, [p1.u, p2.u]):
                with u_sub.x.petsc_vec.localForm() as u_sol_sub_vector_local:
                    sol_sub[:] = u_sol_sub_vector_local[:]

    def update_solution(self, sol_vector):
        (p1,p2) = (self.p1,self.p2)
        with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(sol_vector, self.dofmaps, self.restriction) as nest_sol:
            for sol_sub, u_sub in zip(nest_sol, [p1.u, p2.u]):
                with u_sub.x.petsc_vec.localForm() as u_sub_vector_local:
                    u_sub_vector_local[:] = sol_sub[:]
        p1.u.x.scatter_forward()
        p2.u.x.scatter_forward()

    def assemble_jacobian(
            self, snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P_mat: PETSc.Mat):
        (p1,p2) = (self.p1,self.p2)
        for p in [p1, p2]:
            p.assemble_jacobian(finalize=False)
        # Could move this to `assemble_residual`
        self.contribute_to_diagonal_blocks()
        J_mat.assemble()

    def assemble_residual(self, snes: PETSc.SNES, x: PETSc.Vec, R_vec: PETSc.Vec): 
        (p1,p2) = (self.p1,self.p2)
        # TODO: Make it so that the residual is assembled into R_vec
        self.update_solution(x)
        for p, R_sub_vec in zip([p1, p2], R_vec.getNestSubVecs()):
            p.assemble_residual(R_sub_vec)
        # TODO: Check if expensive
        self.assemble_robin_matrix(p1, p2)
        self.assemble_robin_matrix(p2, p1)
        self.contribute_to_residuals(R_vec)
        R_vec.scale(-1)


    def obj(  # type: ignore[no-any-unimported]
            self, snes: PETSc.SNES, x: PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.assemble_residual(snes, x, self.obj_vec)
        return self.obj_vec.norm()  # type: ignore[no-any-return]

    def non_linear_solve(self):
        (p1,p2) = (self.p1,self.p2)
        self.pre_assemble()

        # Solve
        snes = PETSc.SNES().create(p1.domain.comm)
        snes.setTolerances(max_it=50)

        snes.getKSP().setType("preonly")
        opts = PETSc.Options()
        opts.setValue('-ksp_error_if_not_converged', 'true')
        opts.setValue('-snes_type', 'newtonls')
        #opts.setValue('-snes_line_search_type', 'l2')
        snes.setFromOptions()
        #snes.getKSP().setFromOptions()
        snes.getKSP().getPC().setType("lu")
        snes.getKSP().getPC().setFactorSolverType("mumps")
        #snes.getKSP().getPC().setType("fieldsplit")
        #nested_IS = self.A.getNestISs()
        #snes.getKSP().getPC().setFieldSplitIS(["u1", nested_IS[0][0]], ["u2", nested_IS[1][1]])

        snes.setObjective(self.obj)
        snes.setFunction(self.assemble_residual, self.L)
        snes.setJacobian(self.assemble_jacobian, J=self.A, P=None)
        snes.setMonitor(lambda _, it, residual: print(it, residual))
        self.set_snes_sol_vector()
        snes.solve(None, self.x)
        self.update_solution(self.x)
        assert (snes.getConvergedReason() > 0)
        snes.destroy()
        [opts.__delitem__(k) for k in opts.getAll().keys()] # Clear options data-base
        opts.destroy()

        for p in [p1, p2]:
            p.is_grad_computed = False

    def _destroy(self):
        for attr in ["x", "A12", "A21", "obj_vec"]:
            try:
                self.__dict__[attr].destroy()
            except KeyError:
                pass
