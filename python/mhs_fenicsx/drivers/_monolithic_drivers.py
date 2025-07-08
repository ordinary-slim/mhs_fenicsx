from mhs_fenicsx.problem import Problem
from dolfinx import fem, la
import basix.ufl
from mhs_fenicsx_cpp import cellwise_determine_point_ownership, scatter_cell_integration_data_po, \
                            MonolithicRobinRobinAssembler64, interpolate_dg0_at_facets
from mhs_fenicsx.problem.helpers import get_identity_maps
import numpy as np
from petsc4py import PETSc
import multiphenicsx, multiphenicsx.fem.petsc
import ufl
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class MonolithicDomainDecompositionDriver:
    def __init__(self, sub_problem_1: Problem, sub_problem_2: Problem):
        (p1, p2) = (sub_problem_1, sub_problem_2)
        self.p1 = p1
        self.p2 = p2
        if "petsc_opts_mono_robin" in p1.input_parameters:
            self.solver_opts = dict(p1.input_parameters["petsc_opts_mono_robin"])
        else:
            self.solver_opts = {"pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

    def setup_coupling(self):
        ''' Find interface '''
        (p1,p2) = (self.p1,self.p2)
        self.gdofs_communicators = {}

    def post_iterate(self):
        pass

class MonolithicRRDriver(MonolithicDomainDecompositionDriver):
    def __init__(self, sub_problem_1:Problem, sub_problem_2:Problem,
                 robin_coeff1: float, robin_coeff2: float):
        (p1,p2) = (sub_problem_1,sub_problem_2)
        super().__init__(p1, p2)
        self.robin_coeff1, self.robin_coeff2 = robin_coeff1, robin_coeff2
        self.set_n_compile_forms()
        self.monolithicRrAssembler = dict()
        for p in [p1, p2]:
            self.monolithicRrAssembler[p] = MonolithicRobinRobinAssembler64()

    def __del__(self):
        self._destroy()

    def set_n_compile_forms(self):
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
        for p, robin_coeff in zip([p1, p2], [self.robin_coeff1, self.robin_coeff2]):
            # LHS term Robin
            (u, v) = (p.u, ufl.TestFunction(p.v))
            a_ufl = []
            for mat in p.materials:
                a_ufl.append(+ robin_coeff * u * v * ds(self.gamma_integration_tags[mat]))
            a_ufl = sum(a_ufl)
            self.j_ufl[p] = ufl.derivative(a_ufl, u)
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

    def assemble_robin_jacobian_p_p(self):
        (p1,p2) = (self.p1,self.p2)
        for p, p_ext in zip([p1, p2], [p2, p1]):
            p.A.assemble(PETSc.Mat.AssemblyType.FLUSH)
            multiphenicsx.fem.petsc.assemble_matrix(
                    p.A,
                    self.j_instance[p],
                    bcs=p.dirichlet_bcs,
                    restriction=(p.restriction, p.restriction))

    def assemble_robin_residual(self, R_vec):
        (p1,p2) = (self.p1,self.p2)
        for p, p_ext, R_sub_vec in zip([p1, p2], [p2, p1], R_vec.getNestSubVecs()):
            # RHS term Robin for residual formulation
            res = p.restriction
            robin_residual = la.petsc.create_vector(res.index_map, p.v.value_size)

            # EXT CONTRIBUTION
            self.assemble_robin_residual_p_p_ext(robin_residual, p, p_ext)

            # IN CONTRIBUTION
            multiphenicsx.fem.petsc.assemble_vector(
                    robin_residual,
                    self.r_instance[p],
                    restriction=p.restriction)
            robin_residual.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            # TODO: Check sign here?
            R_sub_vec += robin_residual
            robin_residual.destroy()
            R_sub_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def preassemble_robin_matrix(self, p:Problem, p_ext:Problem):
        self.gdofs_communicators[p] = GdofsCommunicator(p, p_ext, p.boun_dofs_cells_ext[p_ext])
        A_coupling = self.monolithicRrAssembler[p].preassemble(p.bqpoints,
                                                               p.quadrature_points_cell,
                                                               p.Qe._weights,
                                                               p.v._cpp_object,
                                                               p.restriction,
                                                               p_ext.v._cpp_object,
                                                               p_ext.restriction,
                                                               p.bfacets_integration_data,
                                                               p.boun_marker_gamma[p_ext],
                                                               p.boun_renumbered_cells_ext[p_ext],
                                                               p.boun_dofs_cells_ext[p_ext],
                                                               p.boun_geoms_cells_ext[p_ext],
                                                               )
        return A_coupling

    def assemble_robin_jacobian_p_p_ext(self, p:Problem, p_ext:Problem):
        A = self.A12
        robin_coeff = self.robin_coeff1
        if p_ext == self.p1:
            A = self.A21
            robin_coeff = self.robin_coeff2
        A.zeroEntries()
        u_ext_coeffs = self.gdofs_communicators[p].point_to_point_comm()
        self.monolithicRrAssembler[p].assemble_jacobian(A,
                                                        np.array([mat.k.compiled_func for mat in p_ext.materials], np.uintp),
                                                        np.array([mat.k.compiled_dfunc for mat in p_ext.materials], np.uintp),
                                                        p.bqpoints,
                                                        p.quadrature_points_cell,
                                                        p.Qe._weights,
                                                        p.v._cpp_object,
                                                        p.restriction,
                                                        p_ext.v._cpp_object,
                                                        p_ext.restriction,
                                                        p.bfacets_integration_data,
                                                        p.bqpoints_po[p_ext],
                                                        p.boun_marker_gamma[p_ext],
                                                        p.boun_renumbered_cells_ext[p_ext],
                                                        p.boun_dofs_cells_ext[p_ext],
                                                        u_ext_coeffs,
                                                        p.boun_mat_ids[p_ext],
                                                        robin_coeff
                                                        )
        A.assemble()

    def assemble_robin_residual_p_p_ext(self, R_sub_vec: PETSc.Vec, p:Problem, p_ext:Problem):
        u_ext_coeffs = self.gdofs_communicators[p].point_to_point_comm()
        robin_coeff = self.robin_coeff1
        if p_ext == self.p1:
            robin_coeff = self.robin_coeff2
        with R_sub_vec.localForm() as R_loc:
            self.monolithicRrAssembler[p].assemble_residual(R_loc.array_w,
                                                            np.array([mat.k.compiled_func for mat in p_ext.materials], np.uintp),
                                                            np.array([mat.k.compiled_dfunc for mat in p_ext.materials], np.uintp),
                                                            p.bqpoints,
                                                            p.quadrature_points_cell,
                                                            p.Qe._weights,
                                                            p.v._cpp_object,
                                                            p.restriction,
                                                            p_ext.v._cpp_object,
                                                            p_ext.restriction,
                                                            p.bfacets_integration_data,
                                                            p.bqpoints_po[p_ext],
                                                            p.boun_marker_gamma[p_ext],
                                                            p.boun_renumbered_cells_ext[p_ext],
                                                            p.boun_dofs_cells_ext[p_ext],
                                                            u_ext_coeffs,
                                                            p.boun_mat_ids[p_ext],
                                                            robin_coeff
                                                            )

    def pre_assemble(self):
        (p1,p2) = (self.p1,self.p2)
        self._destroy()
        self.restriction = [p1.restriction, p2.restriction]
        self.dofmaps = [p1.v.dofmap, p2.v.dofmap]
        self.setup_coupling()
        self.instantiate_forms()
        r_instance = [p1.r_instance, p2.r_instance]

        self.A12 = self.preassemble_robin_matrix(p1, p2)
        self.A21 = self.preassemble_robin_matrix(p2, p1)

        self.L = PETSc.Vec().createNest([p1.L, p2.L])
        self.A = PETSc.Mat().createNest([[p1.A, self.A12], [self.A21, p2.A]])
        self.x = multiphenicsx.fem.petsc.create_vector(r_instance, kind=PETSc.Vec.Type.NEST, restriction=self.restriction)
        self.obj_vec = multiphenicsx.fem.petsc.create_vector(r_instance, kind=PETSc.Vec.Type.NEST, restriction=self.restriction)

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

    def J_snes(
            self, snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P_mat: PETSc.Mat):
        (p1,p2) = (self.p1,self.p2)
        for p in [p1, p2]:
            p.assemble_jacobian(finalize=False)
        self.assemble_robin_jacobian_p_p_ext(p1, p2)
        self.assemble_robin_jacobian_p_p_ext(p2, p1)
        self.assemble_robin_jacobian_p_p()
        J_mat.assemble()

    def R_snes(self, snes: PETSc.SNES, x: PETSc.Vec, R_vec: PETSc.Vec): 
        (p1,p2) = (self.p1,self.p2)
        self.update_solution(x)
        for p, R_sub_vec in zip([p1, p2], R_vec.getNestSubVecs()):
            p.assemble_residual(R_sub_vec)
        # TODO: Check if expensive
        self.assemble_robin_residual(R_vec)

    def obj_snes(  # type: ignore[no-any-unimported]
            self, snes: PETSc.SNES, x: PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.R_snes(snes, x, self.obj_vec)
        return self.obj_vec.norm()  # type: ignore[no-any-return]

    def non_linear_solve(self):
        (p1,p2) = (self.p1,self.p2)
        self.pre_assemble()

        # Solve
        snes = PETSc.SNES().create(p1.domain.comm)
        snes.setTolerances(max_it=50)

        opts = PETSc.Options()
        for k,v in self.solver_opts.items():
            opts[k] = v
        snes.setFromOptions()
        pc = snes.getKSP().getPC()
        if pc.getType() == "fieldsplit":
            index_sets = self.A.getNestISs()
            pc.setFieldSplitIS(["u1", index_sets[0][0]], ["u2", index_sets[1][1]])

        snes.setObjective(self.obj_snes)
        snes.setFunction(self.R_snes, self.L)
        snes.setJacobian(self.J_snes, J=self.A, P=None)
        snes.setMonitor(lambda _, it, residual: print(it, residual, flush=True) if rank == 0 else None)
        self.set_snes_sol_vector()

        def solve():
            snes.solve(None, self.x)
            self.update_solution(self.x)
            return snes.getConvergedReason()

        converged_reason = solve()
        assert (converged_reason > 0), f"did not converge : {converged_reason}"

        snes.destroy()
        [opts.__delitem__(k) for k in opts.getAll().keys()] # Clear options data-base
        opts.destroy()

        for p in [p1, p2]:
            p.post_modify_solution()

    def _destroy(self):
        for attr in ["x", "A12", "A21", "obj_vec"]:
            try:
                self.__dict__[attr].destroy()
            except KeyError:
                pass

class GdofsCommunicator:
    def __init__(self, p, p_ext, gdofs_p_needs):
        '''Right after calling scatter_cell_integration_data_po'''
        self.p, self.p_ext = p, p_ext
        self.ndofs_cell_ext = gdofs_p_needs.shape[1]
        self.gdofs_p_needs, self.uiperm_gdofs_ext = np.unique(gdofs_p_needs, return_inverse=True)
        imap = p_ext.restriction.index_map
        lrange = imap.local_range
        bounds = np.zeros(comm.size+1, np.int64)
        comm.Allgather(np.array(lrange[0]), bounds[:-1])
        bounds[-1] = imap.size_global
        self.owners = np.searchsorted(bounds, self.gdofs_p_needs, side="right")
        self.owners -= 1
        self.mask_local = (self.owners == rank)
        self.mask_ghost = np.logical_not(self.mask_local)

        # Counts I need from other procs
        rcv_sizes = np.zeros(comm.size, np.int32)
        for ighost in self.mask_ghost.nonzero()[0]:
            rcv_sizes[self.owners[ighost]] += 1
        snd_sizes = np.zeros(comm.size, np.int32)
        comm.Alltoall(rcv_sizes, snd_sizes)

        # dofs i need
        other_ranks = np.hstack((np.arange(0, rank), np.arange(rank+1, comm.size)))
        self.dofs_i_rcv = {ghost_owner : self.gdofs_p_needs[(self.owners==ghost_owner).nonzero()[0]] for ghost_owner in other_ranks}
        self.dofs_i_snd = {ghost_owner : np.zeros(snd_sizes[ghost_owner], dtype=np.int64) for ghost_owner in other_ranks}
        
        reqs = []
        for orank in other_ranks:
            reqs.append(comm.Isend(self.dofs_i_rcv[orank], dest=orank))

        for orank in other_ranks:
            reqs.append(comm.Irecv(self.dofs_i_snd[orank], source=orank))

        MPI.Request.Waitall(reqs)

        for other_rank, dofs in self.dofs_i_snd.items():
            self.dofs_i_snd[other_rank] = p_ext.restriction.unrestrict(imap.global_to_local(dofs))
        lindices = self.mask_local.nonzero()[0]
        llindices = imap.global_to_local(self.gdofs_p_needs[lindices])
        self.my_dofs_i_need = p_ext.restriction.unrestrict(llindices)

        self.data_i_rcv = {ghost_owner: np.zeros(rcv_sizes[ghost_owner], dtype=np.float64) for ghost_owner in other_ranks}
        self.data_i_snd = {ghost_owner: np.zeros(snd_sizes[ghost_owner], dtype=np.float64) for ghost_owner in other_ranks}
        self.u_ext_coeffs = np.zeros(self.gdofs_p_needs.size, np.float64)

    def point_to_point_comm(self):
        p_ext = self.p_ext
        other_ranks = np.hstack((np.arange(0, rank), np.arange(rank+1, comm.size)))
        reqs = []
        for orank in other_ranks:
            self.data_i_snd[orank][:] = p_ext.u.x.array[self.dofs_i_snd[orank]]
            reqs.append(comm.Isend(self.data_i_snd[orank], dest=orank))

        for orank in other_ranks:
            reqs.append(comm.Irecv(self.data_i_rcv[orank], source=orank))
        MPI.Request.Waitall(reqs)

        self.u_ext_coeffs[self.mask_local.nonzero()[0]] = self.p_ext.u.x.array[self.my_dofs_i_need]
        for other_rank in self.data_i_rcv:
            self.u_ext_coeffs[(self.owners==other_rank).nonzero()[0]] = self.data_i_rcv[other_rank]
        return self.u_ext_coeffs[self.uiperm_gdofs_ext].reshape((-1, self.ndofs_cell_ext))
