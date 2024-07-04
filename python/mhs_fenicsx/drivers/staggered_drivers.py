from dolfinx import fem, mesh, io
from mhs_fenicsx.problem.helpers import indices_to_function
import ufl
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, GammaL2Dotter
from mhs_fenicsx_cpp import interpolate_dg0_at_facets, cellwise_determine_point_ownership
from line_profiler import LineProfiler
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class StaggeredDomainDecompositionDriver:
    def __init__(self,sub_problem_1,sub_problem_2,
                 max_staggered_iters=40,):
        self.p1 = sub_problem_1
        self.p2 = sub_problem_2
        self.max_staggered_iters = max_staggered_iters
        self.convergence_crit = 1e9
        self.convergence_threshold = 1e-6
        self.writer1 = io.VTKFile(self.p1.domain.comm, f"staggered_out/staggered_iters_{self.p1.name}.pvd", "wb")
        self.writer2 = io.VTKFile(self.p2.domain.comm, f"staggered_out/staggered_iters_{self.p2.name}.pvd", "wb")
        self.iter = -1

    def post_loop(self):
        self.p1.post_iterate()
        self.p2.post_iterate()

    def pre_iterate(self):
        if self.iter == 0:
            self.previous_u1 = self.p1.u.copy();self.previous_u1.name="previous_u"
            self.previous_u2 = self.p2.u.copy();self.previous_u2.name="previous_u"
        else:
            self.previous_u1.x.array[:] = self.p1.u.x.array
            self.previous_u2.x.array[:] = self.p2.u.x.array
        self.iter += 1

    def write_results(self,extra_funcs_p1=[],extra_funcs_p2=[]):
        (p1, p2) = (self.p1, self.p2)
        pos_functions_p1 = [p1.u,
                            p1.active_els_func,
                            p1.source_rhs,
                            self.previous_u1,]
        pos_functions_p2 = [p2.u,
                            p2.active_els_func,
                            p2.source_rhs,
                            self.previous_u2,]

        pos_functions_p1.extend(extra_funcs_p1)
        pos_functions_p2.extend(extra_funcs_p2)
        # PARTITION
        ranks1 = fem.Function(p1.dg0_bg,name="partition")
        ranks2 = fem.Function(p2.dg0_bg,name="partition")
        ranks1.x.array[:] = rank
        ranks2.x.array[:] = rank
        pos_functions_p1.append(ranks1)
        pos_functions_p2.append(ranks2)
        # EPARTITION
        self.writer1.write_function(pos_functions_p1,t=self.iter)
        self.writer2.write_function(pos_functions_p2,t=self.iter)

    def set_interface(self):
        (p1, p2) = (self.p1, self.p2)
        self.gamma_dofs = dict()
        self.gamma_cells = dict()
        self.iid = dict()
        self.active_gamma_cells = dict()
        for p,p_ext in zip([p1,p2],[p2,p1]):
            p.find_gamma(p.get_active_in_external( p_ext ))
            self.gamma_dofs[p] = p.gamma_nodes.x.array.nonzero()[0]
            # Interpolation data: TODO: Cleanup
            self.gamma_cells[p] = mesh.compute_incident_entities(p.domain.topology,
                                                                np.hstack((p.gamma_facets.find(1),p.gamma_facets.find(2))),
                                                                p.dim-1,
                                                                p.dim)
            self.iid[p_ext] = dict()
            self.iid[p_ext][p] = fem.create_interpolation_data(
                                                 p.v,
                                                 p_ext.v,
                                                 self.gamma_cells[p],
                                                 padding=1e-6,)
            self.active_gamma_cells[p] = self.gamma_cells[p][p.active_els_func.x.array[self.gamma_cells[p]].nonzero()[0]]
            p_index_ghost = np.searchsorted(self.active_gamma_cells[p],p.cell_map.size_local-1,side='right')
            self.active_gamma_cells[p] = self.active_gamma_cells[p][:p_index_ghost]


    def pre_loop(self,set_bc=None):
        (p1, p2) = (self.p1, self.p2)
        self.iter = 0
        # TODO: Call parent pre-loop
        # TODO: Complete with spec tasks
        p1.clear_dirchlet_bcs()
        p2.clear_dirchlet_bcs()
        self.set_interface()
        # Ext bc
        if set_bc is not None:
            set_bc(p1,p2)

class StaggeredDNDriver(StaggeredDomainDecompositionDriver):
    def __init__(self,
                 p_dirichlet:Problem,
                 p_neumann:Problem,
                 max_staggered_iters=40,
                 initial_relaxation_factor=0.5,):
        StaggeredDomainDecompositionDriver.__init__(self,
                                                    p_dirichlet,
                                                    p_neumann,
                                                    max_staggered_iters,)
        self.p_dirichlet = p_dirichlet
        self.p_neumann = p_neumann
        self.initial_relaxation_factor = initial_relaxation_factor
        self.relaxation_factor = initial_relaxation_factor
        self.dirichlet_tcon = None

    def writepos(self,extra_funcs_p1=[],extra_funcs_p2=[]):
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        extra_funcs_p1 = [pd.dirichlet_gamma, pd.grad_u] + extra_funcs_p1
        extra_funcs_p2 = [pn.neumann_flux, self.ext_conductivity,] + extra_funcs_p2
                            

        StaggeredDomainDecompositionDriver.write_results(self,extra_funcs_p1=extra_funcs_p1,extra_funcs_p2=extra_funcs_p2)

    def pre_loop(self,set_bc=None):
        StaggeredDomainDecompositionDriver.pre_loop(self,set_bc=set_bc)
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        midpoints_neumann_facets = mesh.compute_midpoints(pn.domain,pn.domain.topology.dim-1,pn.gamma_facets.find(1))
        self.iid_d2n_border = cellwise_determine_point_ownership(
                                    pd.domain._cpp_object,
                                    midpoints_neumann_facets,
                                    self.active_gamma_cells[pd],
                                    np.float64(1e-6))

        # Neumann Gamma funcs
        pn.neumann_flux = fem.Function(pn.dg0_vec,name="flux")
        self.ext_conductivity = fem.Function(pn.dg0_bg,name="ext_conduc")
        # Aitken
        self.neumann_res = fem.Function(pn.v,name="residual")
        self.neumann_prev_res = fem.Function(pn.v,name="previous residual")
        self.neumann_res_diff = fem.Function(pn.v,name="residual difference")
        self.dirichlet_res = fem.Function(pd.v,name="residual")
        self.dirichlet_prev_res = fem.Function(pd.v,name="previous residual")
        self.dirichlet_res_diff = fem.Function(pd.v,name="residual difference")

        # Forms and allocation
        self.set_dirichlet_interface()
        pd.set_forms_domain()
        pd.set_forms_boundary()
        pd.compile_forms()
        pn.set_forms_domain()
        pn.set_forms_boundary()
        self.set_neumann_interface()
        pn.compile_forms()

        pd.pre_assemble()
        pn.pre_assemble()

        self.l2_dot_neumann = GammaL2Dotter(pn)
        self.l2_dot_dirichlet = GammaL2Dotter(pd)

    def post_iterate(self, verbose=False):
        (pn, pd) = (self.p_neumann, self.p_dirichlet)
        self.neumann_res.x.array[:] = pn.u.x.array-self.previous_u2.x.array
        norm_diff_neumann    = self.l2_dot_neumann(self.neumann_res)
        norm_current_neumann = self.l2_dot_neumann(pn.u)
        self.convergence_crit = np.sqrt(norm_diff_neumann) / np.sqrt(norm_current_neumann)
        norm_res = self.l2_dot_dirichlet(self.dirichlet_res)
        if rank==0:
            if verbose:
                print(f"Staggered iteration DN #{self.iter}, omega = {self.relaxation_factor}, relative norm of difference: {self.convergence_crit}, norm residual: {norm_res}")

    def set_dirichlet_interface(self):
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        # Get Gamma DOFS right
        dofs_gamma_right = fem.locate_dofs_topological(pd.v, pd.dim-1, pd.gamma_facets.find(1))
        self.update_dirichlet_interface()
        # Set Gamma dirichlet
        self.dirichlet_tcon = pd.add_dirichlet_bc(pd.dirichlet_gamma,bdofs=dofs_gamma_right, reset=False)
        pd.is_dirichlet_gamma = True

    def update_relaxation_factor(self):
        (pn,pd) = (self.p_neumann,self.p_dirichlet)
        # Interpolate
        pd.dirichlet_gamma.interpolate_nonmatching(pn.u,
                                                  cells=self.gamma_cells[pd],
                                                  interpolation_data=self.iid[pn][pd])
        pd.dirichlet_gamma.x.scatter_forward()
        if self.iter > 1:
            self.dirichlet_prev_res.x.array[self.gamma_dofs[self.p1]] = self.dirichlet_res.x.array[self.gamma_dofs[self.p1]]

        self.dirichlet_res.x.array[self.gamma_dofs[self.p1]] = pd.dirichlet_gamma.x.array[self.gamma_dofs[self.p1]] - \
                               pd.u.x.array[self.gamma_dofs[self.p1]]
        self.dirichlet_res.x.scatter_forward()
        if self.iter > 1:
            self.dirichlet_res_diff.x.array[self.gamma_dofs[self.p1]] = self.dirichlet_res.x.array[self.gamma_dofs[self.p1]] - \
                    self.dirichlet_prev_res.x.array[self.gamma_dofs[self.p1]]
            self.dirichlet_res_diff.x.scatter_forward()
            self.relaxation_factor = - self.relaxation_factor * self.l2_dot_dirichlet(self.dirichlet_prev_res,self.dirichlet_res_diff)
            self.relaxation_factor /= self.l2_dot_dirichlet(self.dirichlet_res_diff)


    def update_dirichlet_interface(self):
        (pd, pn) = (self.p_dirichlet,self.p_neumann)
        pd.dirichlet_gamma.x.array[:] = self.relaxation_factor*pd.dirichlet_gamma.x.array + \
                                 (1-self.relaxation_factor)*pd.u.x.array
        pd.dirichlet_gamma.x.scatter_forward()
        if self.dirichlet_tcon is not None:
            self.dirichlet_tcon.g.x.array[:] = pd.dirichlet_gamma.x.array
    
    def set_neumann_interface(self):
        p = self.p_neumann
        self.update_neumann_interface()
        # Custom measure
        gammaIntegralEntities = p.get_facet_integrations_entities()
        dS = ufl.Measure('ds', domain=p.domain, subdomain_data=[
            (8,np.asarray(gammaIntegralEntities, dtype=np.int32))])
        v = ufl.TestFunction(p.v)
        n = ufl.FacetNormal(p.domain)
        p.l_ufl += +self.ext_conductivity * ufl.inner(n,p.neumann_flux) * v * dS(8)

    def update_neumann_interface(self):
        (p, p_ext) = (self.p_neumann,self.p_dirichlet)
        p_ext.compute_gradient()
        # Update functions
        interpolate_dg0_at_facets(p_ext.grad_u._cpp_object,
                                  p.neumann_flux._cpp_object,
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets._cpp_object,
                                  self.active_gamma_cells[p],
                                  self.iid_d2n_border,
                                  p.gamma_facets_index_map,
                                  p.gamma_imap_to_global_imap)
        p.neumann_flux.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        interpolate_dg0_at_facets(p_ext.k._cpp_object,
                                  self.ext_conductivity._cpp_object,
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets._cpp_object,
                                  self.active_gamma_cells[p],
                                  self.iid_d2n_border,
                                  p.gamma_facets_index_map,
                                  p.gamma_imap_to_global_imap)
        self.ext_conductivity.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def iterate(self):
        (pn, pd) = (self.p_neumann,self.p_dirichlet)
        # Solve right with Dirichlet from left
        pd.assemble()
        pd.solve()

        self.update_neumann_interface()
        # Solve left with Neumann from right
        pn.assemble()
        pn.solve()
        self.update_relaxation_factor()
        self.update_dirichlet_interface()

class StaggeredRRDriver(StaggeredDomainDecompositionDriver):
    def __init__(self,
                 p_dirichlet:Problem,
                 p_neumann:Problem,
                 max_staggered_iters=40,
                 initial_relaxation_factor=0.5,):
        StaggeredDomainDecompositionDriver.__init__(self,
                                                    p_dirichlet,
                                                    p_neumann,
                                                    max_staggered_iters,)
        self.initial_relaxation_factor = initial_relaxation_factor
        self.relaxation_factor = initial_relaxation_factor
        self.dirichlet_tcon = None

    def writepos(self,extra_funcs_p1=[],extra_funcs_p2=[]):
        (p1,p2) = (self.p1,self.p2)
        extra_funcs_p1 = [p1.dirichlet_gamma, p1.neumann_flux, self.ext_conductivity[p1],] + extra_funcs_p1
        extra_funcs_p2 = [p2.dirichlet_gamma, p2.neumann_flux, self.ext_conductivity[p2],] + extra_funcs_p2
        StaggeredDomainDecompositionDriver.write_results(self,extra_funcs_p1=extra_funcs_p1,extra_funcs_p2=extra_funcs_p2)

    def pre_loop(self,set_bc=None):
        StaggeredDomainDecompositionDriver.pre_loop(self,set_bc=set_bc)
        (p1,p2) = (self.p1,self.p2)
        midpoints_facets = dict()
        self.iid_border = dict()
        self.ext_conductivity = dict()
        self.gamma_residual = dict()
        self.l2_dot = dict()
        for p,p_ext in zip([p1,p2],[p2,p1]):
            self.iid_border[p_ext] = dict()
            midpoints_facets[p] = mesh.compute_midpoints(p.domain,p.domain.topology.dim-1,p.gamma_facets.find(1))

            self.iid_border[p_ext][p] = cellwise_determine_point_ownership(
                                        p_ext.domain._cpp_object,
                                        midpoints_facets[p],
                                        self.active_gamma_cells[p_ext],
                                        np.float64(1e-6))

            # Flux funcs
            p.neumann_flux = fem.Function(p.dg0_vec,name="flux")
            self.ext_conductivity[p] = fem.Function(p.dg0_bg,name="ext_conduc")
            self.gamma_residual[p] = fem.Function(p.v,name="residual")

            # Forms and allocation
            p.set_forms_domain()
            p.set_forms_boundary()
            self.set_robin(p)
            p.compile_forms()
            p.pre_assemble()
            self.l2_dot[p] = GammaL2Dotter(p)

    def post_iterate(self, verbose=False):
        (p2, p1) = (self.p2, self.p1)
        self.gamma_residual[p2].x.array[:] = p2.u.x.array-self.previous_u2.x.array
        norm_diff_neumann    = self.l2_dot[p2](self.gamma_residual[p2])
        norm_current_neumann = self.l2_dot[p2](p2.u)
        self.convergence_crit = np.sqrt(norm_diff_neumann) / np.sqrt(norm_current_neumann)
        if rank==0:
            if verbose:
                print(f"Staggered iteration RR #{self.iter}, relative norm of difference: {self.convergence_crit}")

    def set_robin(self,p):
        if p==self.p1:
            p_ext=self.p2
        else:
            p_ext=self.p1
        self.update_robin(p)
        # Custom measure
        gammaIntegralEntities = p.get_facet_integrations_entities()
        dS = ufl.Measure('ds', domain=p.domain, subdomain_data=[
            (8,np.asarray(gammaIntegralEntities, dtype=np.int32))])
        v = ufl.TestFunction(p.v)
        n = ufl.FacetNormal(p.domain)
        (u, v) = (ufl.TrialFunction(p.v),ufl.TestFunction(p.v))
        p.a_ufl += + u * v * dS(8)
        p.l_ufl += + (p.dirichlet_gamma + self.ext_conductivity[p] * ufl.inner(n,p.neumann_flux)) * v * dS(8)

    def update_robin(self,p):
        if p==self.p1:
            p_ext=self.p2
        else:
            p_ext=self.p1
        p_ext.compute_gradient()
        # Update flux
        interpolate_dg0_at_facets(p_ext.grad_u._cpp_object,
                                  p.neumann_flux._cpp_object,
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets._cpp_object,
                                  self.active_gamma_cells[p],
                                  self.iid_border[p_ext][p],
                                  p.gamma_facets_index_map,
                                  p.gamma_imap_to_global_imap)
        p.neumann_flux.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        interpolate_dg0_at_facets(p_ext.k._cpp_object,
                                  self.ext_conductivity[p]._cpp_object,
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets._cpp_object,
                                  self.active_gamma_cells[p],
                                  self.iid_border[p_ext][p],
                                  p.gamma_facets_index_map,
                                  p.gamma_imap_to_global_imap)
        self.ext_conductivity[p].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # Update ext solution
        p.dirichlet_gamma.interpolate_nonmatching(p_ext.u,
                                                  cells=self.gamma_cells[p],
                                                  interpolation_data=self.iid[p_ext][p])
        p.dirichlet_gamma.x.scatter_forward()

    def iterate(self):
        (p1, p2) = (self.p1,self.p2)

        self.update_robin(p1)
        p1.assemble()
        p1.solve()

        self.update_robin(p2)
        p2.assemble()
        p2.solve()
