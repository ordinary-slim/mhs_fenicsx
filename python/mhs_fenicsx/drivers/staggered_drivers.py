from dolfinx import fem, mesh, io
from mhs_fenicsx.problem.helpers import indices_to_function, assert_gamma_tag
import ufl
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, GammaL2Dotter, propagate_dg0_at_facets_same_mesh
from mhs_fenicsx_cpp import interpolate_dg0_at_facets, cellwise_determine_point_ownership
from line_profiler import LineProfiler
from petsc4py import PETSc
import shutil
from abc import ABC, abstractmethod

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class StaggeredDomainDecompositionDriver(ABC):
    def __init__(self,sub_problem_1:Problem,sub_problem_2:Problem,
                 max_staggered_iters=40,
                 initial_relaxation_factors=[1.0,1.0]):
        (p1,p2) = (sub_problem_1,sub_problem_2)
        self.p1 = p1
        self.p2 = p2
        self.max_staggered_iters = max_staggered_iters
        self.convergence_crit = 1e9
        self.convergence_threshold = 1e-6
        self.writers = dict()
        self.iter = -1
        self.is_chimera = not(p1.domain == p2.domain) # Different meshes
        self.active_gamma_cells = dict()
        self.gamma_integration_tag = 4
        if self.is_chimera:
            self.midpoints_facets = dict()
            self.iid_border = dict()
        else:
            self.active_gamma_cells = dict()
        # Relaxation
        self.relaxation_coeff = {p1:fem.Constant(p1.domain, 1.0),p2:fem.Constant(p2.domain, 1.0)}
        for relaxation,p in zip(initial_relaxation_factors,[p1,p2]):
            if relaxation<1.0:
                self.relaxation_coeff[p].value = relaxation
        self.ext_conductivity = dict()
        self.ext_flux = dict()
        self.net_ext_flux = dict()
        self.ext_sol = dict()
        self.net_ext_sol = dict()
        self.prev_ext_flux = dict()
        self.prev_ext_sol = dict()
        self.l2_dot = {p:GammaL2Dotter(p) for p in (p1, p2)}
        self.gamma_residual = dict()
        self.initialize_coupling_functions()
        self.compile_forms()
        self.store_forms()

    def __del__(self):
        for w in self.writers.values():
            w.close()

    @abstractmethod
    def initialize_coupling_functions(self):
        pass

    @abstractmethod
    def compile_forms(self):
        pass
    
    def store_forms(self):
        (p1, p2) = plist = (self.p1, self.p2)
        self.mr_ufl = {p:p.mr_ufl for p in plist}
        self.j_ufl = {p:p.j_ufl for p in plist}
        self.mr_compiled = {p:p.mr_compiled for p in plist}
        self.j_compiled = {p:p.j_compiled for p in plist}

    def prepare_subproblems(self, finalize=True):
        (p1, p2) = plist = (self.p1, self.p2)
        for p in plist:
            p.mr_ufl = self.mr_ufl[p]
            p.mr_compiled = self.mr_compiled[p]
            p.j_ufl = self.j_ufl[p]
            p.j_compiled = self.j_compiled[p]
            p.instantiate_forms()
            if finalize:
                p.instantiate_forms()
                p.pre_assemble()

    def initialize_post(self):
        self.result_folder = f"staggered_out"
        shutil.rmtree(self.result_folder,ignore_errors=True)
        for p in [self.p1,self.p2]:
            self.writers[p] = io.VTKFile(p.domain.comm, f"{self.result_folder}/staggered_iters_{p.name}.pvd", "wb")

    def post_loop(self):
        for p in [self.p1, self.p2]:
            p.post_iterate()

    def pre_iterate(self):
        (p1,p2) = (self.p1, self.p2)
        if self.iter == 0:
            self.previous_u = dict()
            for p in [p1,p2]:
                self.previous_u[p] = p.u.copy();self.previous_u[p].name="previous_u"
        else:
            for p in [p1,p2]:
                self.previous_u[p].x.array[:] = p.u.x.array
        self.iter += 1

    def write_results(self,extra_funcs_p1=[],extra_funcs_p2=[]):
        (p1, p2) = (self.p1, self.p2)
        pos_funs = dict()

        # PARTITION
        ranks = dict()
        for p in [self.p1,self.p2]:
            ranks[p] = fem.Function(p.dg0,name="partition")
            ranks[p].x.array[:] = rank

        pos_funs[self.p1] = [p1.u,
                            p1.active_els_func,
                            p1.source.fem_function,
                            self.previous_u[self.p1],
                            ranks[p1],]

        pos_funs[self.p2] = [p2.u,
                            p2.active_els_func,
                            p2.source.fem_function,
                            self.previous_u[self.p2],
                            ranks[p2],]

        for p,extra_funs in zip([self.p1,self.p2],[extra_funcs_p1,extra_funcs_p2]):
            pos_funs[p].extend(extra_funs)
        # EPARTITION
        for p in [self.p1,self.p2]:
            self.writers[p].write_function(pos_funs[p],t=self.iter)

    def set_interface(self):
        (p1, p2) = (self.p1, self.p2)
        self.gamma_cells = dict()
        self.iid = dict()
        for p,p_ext in zip([p1,p2],[p2,p1]):
            p.find_gamma(p_ext)
            # Interpolation data: TODO: Cleanup
            self.gamma_cells[p] = mesh.compute_incident_entities(p.domain.topology,
                                                                np.hstack((p.gamma_facets[p_ext].find(1),p.gamma_facets[p_ext].find(2))),
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
        if self.is_chimera:
            self.set_interface()
        else:
            for p, p_ext in [(p1,p2),(p2,p1)]:
                self.active_gamma_cells[p] = p.gamma_integration_data[p_ext][::2]
        self.gamma_dofs = dict()
        for p, p_ext in [(p1,p2),(p2,p1)]:
            self.gamma_dofs[p] = fem.locate_dofs_topological(p.v,0,p.gamma_nodes[p_ext].x.array.nonzero()[0],True)
            p.form_subdomain_data[fem.IntegralType.exterior_facet].append((self.gamma_integration_tag,p.gamma_integration_data[p_ext]))
            self.l2_dot[p].set_gamma(p_ext)
        # Ext bc
        if set_bc is not None:
            #TODO: Fix this, bug prone!
            set_bc(p1,p2)

    def assert_tag(self, p):
        assert_gamma_tag(self.gamma_integration_tag, p)

class StaggeredDNDriver(StaggeredDomainDecompositionDriver):
    def __init__(self,
                 p_dirichlet:Problem,
                 p_neumann:Problem,
                 max_staggered_iters=40,
                 initial_relaxation_factors=[1.0,1.0]):
        initial_relaxation_factors[0] = 0.5 # Force relaxation of Dirichlet problem (Aitken)
        (self.p_dirichlet, self.p_neumann) = (p_dirichlet, p_neumann)
        StaggeredDomainDecompositionDriver.__init__(self,
                                                    p_dirichlet,
                                                    p_neumann,
                                                    max_staggered_iters,
                                                    initial_relaxation_factors)

    def initialize_coupling_functions(self):
        (pd,pn) = plist = (self.p_dirichlet,self.p_neumann)
        for p in plist:
            # Neumann Gamma funcs
            self.ext_conductivity[pn] = fem.Function(pn.dg0,name="ext_conduc")
            self.ext_flux[pn] = fem.Function(pn.dg0_vec,name="ext_grad")
            self.net_ext_flux[pn] = fem.Function(pn.dg0_vec,name="net_ext_flux")
            self.prev_ext_flux[pn] = fem.Function(pn.dg0_vec,name="prev_flux")

            self.ext_sol[pd] = fem.Function(pd.v,name="ext_sol")
            # TODO: Use this instead of actual sol
            self.prev_ext_sol[pd] = fem.Function(pd.v,name="prev_ext_sol")
            # Aitken
            self.neumann_res = fem.Function(pn.v,name="residual")
            self.neumann_prev_res = fem.Function(pn.v,name="previous residual")
            self.neumann_res_diff = fem.Function(pn.v,name="residual difference")
            self.dirichlet_res = fem.Function(pd.v,name="residual")
            self.dirichlet_prev_res = fem.Function(pd.v,name="previous residual")
            self.dirichlet_res_diff = fem.Function(pd.v,name="residual difference")

    def compile_forms(self):
        (pd,pn) = plist = (self.p_dirichlet,self.p_neumann)
        for p in plist:
            p.set_forms_domain()
            p.set_forms_boundary()
        self.set_neumann_interface()
        for p in plist:
            p.compile_forms()

    def writepos(self,extra_funcs_p1=[],extra_funcs_p2=[]):
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        if not(self.writers):
            self.initialize_post()
        extra_funcs_p1 = [self.ext_sol[pd], pd.grad_u] + extra_funcs_p1
        extra_funcs_p2 = [self.ext_flux[pn]] + extra_funcs_p2
                            

        StaggeredDomainDecompositionDriver.write_results(self,extra_funcs_p1=extra_funcs_p1,extra_funcs_p2=extra_funcs_p2)

    def pre_loop(self,set_bc=None,prepare_subproblems=True):
        '''
        1. Set interface between subproblems
        2. Initialize vars to receive ext data
        '''
        StaggeredDomainDecompositionDriver.pre_loop(self,set_bc=set_bc)
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        if self.is_chimera:
            midpoints_neumann_facets = mesh.compute_midpoints(pn.domain,pn.domain.topology.dim-1,pn.gamma_facets[pd].find(1))
            self.iid_d2n_border = cellwise_determine_point_ownership(
                                        pd.domain._cpp_object,
                                        midpoints_neumann_facets,
                                        self.active_gamma_cells[pd],
                                        np.float64(1e-6))

        # Forms and allocation
        if prepare_subproblems:
            self.prepare_subproblems()

    def prepare_subproblems(self, finalize=True):
        (pn, pd) = plist = (self.p_neumann, self.p_dirichlet)
        StaggeredDomainDecompositionDriver.prepare_subproblems(self,finalize=finalize)
        self.set_dirichlet_interface()

    def post_iterate(self, verbose=False):
        (pn, pd) = (self.p_neumann, self.p_dirichlet)
        self.neumann_res.x.array[:] = pn.u.x.array-self.previous_u[self.p2].x.array
        norm_diff_neumann    = self.l2_dot[pn](self.neumann_res)
        norm_current_neumann = self.l2_dot[pn](pn.u)
        self.convergence_crit = np.sqrt(norm_diff_neumann) / np.sqrt(norm_current_neumann)
        norm_res = self.l2_dot[pd](self.dirichlet_res)
        if rank==0:
            if verbose:
                print(f"Staggered iteration DN #{self.iter}, relaxation factor = {self.relaxation_coeff[pd].value}, relative norm of difference: {self.convergence_crit}, norm residual: {norm_res}")

    def set_dirichlet_interface(self,update=True):
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        # Get Gamma DOFS right
        gamma_dofs = fem.locate_dofs_topological(pd.v, pd.dim-1, pd.gamma_facets[pn].find(1))
        # Set Gamma dirichlet
        self.dirichlet_tcon = pd.add_dirichlet_bc(self.ext_sol[pd],bdofs=gamma_dofs, reset=False)
        if update:
            self.update_dirichlet_interface()

    def pre_iterate(self):
        super().pre_iterate()
        pn = self.p_neumann
        if self.relaxation_coeff[pn].value < 1.0:
            self.prev_ext_flux[pn].x.array[:] = self.ext_flux[pn].x.array[:]


    def update_ext_sol(self):
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        if self.is_chimera:
            self.ext_sol[pd].interpolate_nonmatching(pn.u,
                                                     cells=self.gamma_cells[pd],
                                                     interpolation_data=self.iid[pn][pd])
        else:
            # If this is expensive, consider `self.ext_sol[pd] = pn.u`
            self.ext_sol[pd].x.array[:] = pn.u.x.array[:]
        self.ext_sol[pd].x.scatter_forward()

    def update_relaxation_factor(self):
        (pn,pd) = (self.p_neumann,self.p_dirichlet)

        if self.iter > 1:
            self.dirichlet_prev_res.x.array[self.gamma_dofs[self.p1]] = self.dirichlet_res.x.array[self.gamma_dofs[self.p1]]
            self.prev_ext_sol[pd].x.array[:] = self.ext_sol[pd].x.array[:]

        self.update_ext_sol()

        self.dirichlet_res.x.array[self.gamma_dofs[pd]] = self.ext_sol[pd].x.array[self.gamma_dofs[pd]] - \
                               pd.u.x.array[self.gamma_dofs[pd]]
        self.dirichlet_res.x.scatter_forward()
        if self.iter > 1:
            self.dirichlet_res_diff.x.array[self.gamma_dofs[pd]] = self.dirichlet_res.x.array[self.gamma_dofs[pd]] - \
                    self.dirichlet_prev_res.x.array[self.gamma_dofs[pd]]
            self.dirichlet_res_diff.x.scatter_forward()
            self.relaxation_coeff[pd].value = - self.relaxation_coeff[pd].value * self.l2_dot[pd](self.dirichlet_prev_res,self.dirichlet_res_diff)
            self.relaxation_coeff[pd].value /= self.l2_dot[pd](self.dirichlet_res_diff)


    def update_dirichlet_interface(self):
        (pd, pn) = (self.p_dirichlet,self.p_neumann)
        self.dirichlet_tcon.g.x.array[:] = self.relaxation_coeff[pd].value*self.ext_sol[pd].x.array + \
                                 (1-self.relaxation_coeff[pd].value)*pd.u.x.array
        self.dirichlet_tcon.g.x.scatter_forward()
    
    def set_neumann_interface(self):
        (p, p_ext) = (self.p_neumann, self.p_dirichlet)
        # Custom measure
        dS = ufl.Measure('ds')
        v = ufl.TestFunction(p.v)
        n = ufl.FacetNormal(p.domain)
        neumann_con = +ufl.inner(n,self.net_ext_flux[p])
        p.l_ufl += neumann_con * v * dS(self.gamma_integration_tag)

    def update_neumann_interface(self):
        (p, p_ext) = (self.p_neumann,self.p_dirichlet)
        self.assert_tag(p)
        p_ext.compute_gradient(cells=self.active_gamma_cells[p_ext])
        # Update functions
        if self.is_chimera:
            interpolate_dg0_at_facets([p_ext.grad_u._cpp_object,p_ext.k._cpp_object],
                                      [self.ext_flux[p]._cpp_object,self.ext_conductivity[p]._cpp_object],
                                      p.active_els_func._cpp_object,
                                      p.gamma_facets[p_ext]._cpp_object,
                                      self.active_gamma_cells[p],
                                      self.iid_d2n_border,
                                      p.gamma_facets_index_map[p_ext],
                                      p.gamma_imap_to_global_imap[p_ext])
        else:
            propagate_dg0_at_facets_same_mesh(p_ext, p_ext.grad_u, p, self.ext_flux[p])
            propagate_dg0_at_facets_same_mesh(p_ext, p_ext.k, p, self.ext_conductivity[p])

        # TODO: Pass this to c++
        dim = self.ext_flux[p].function_space.value_size
        for idx in range(dim):
            self.ext_flux[p].x.array[self.active_gamma_cells[p]*dim+idx] *= self.ext_conductivity[p].x.array[self.active_gamma_cells[p]]
        self.ext_flux[p].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.net_ext_flux[p].x.array[:] = self.ext_flux[p].x.array[:]
        if self.relaxation_coeff[p].value < 1.0:
            theta = self.relaxation_coeff[p].value
            self.net_ext_flux[p].x.array[:] *= theta
            self.net_ext_flux[p].x.array[:] += (1-theta)*self.prev_ext_flux[p].x.array[:]

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
                 p1:Problem,
                 p2:Problem,
                 max_staggered_iters=40,
                 initial_relaxation_factors=[1.0,1.0]):
        self.dirichlet_coeff = {p1:fem.Constant(p1.domain, 1.0),p2:fem.Constant(p2.domain, 1.0)}
        StaggeredDomainDecompositionDriver.__init__(self,
                                                    p1,
                                                    p2,
                                                    max_staggered_iters,
                                                    initial_relaxation_factors)
        self.dirichlet_tcon = None

    def initialize_coupling_functions(self):
        (p1,p2) = plist = (self.p1,self.p2)
        for p in plist:
            self.ext_flux[p] = fem.Function(p.dg0_vec,name="ext_grad")
            self.net_ext_flux[p] = fem.Function(p.dg0_vec,name="net_ext_flux")
            self.ext_conductivity[p] = fem.Function(p.dg0,name="ext_conduc")
            self.ext_sol[p] = fem.Function(p.v,name="ext_sol")
            self.net_ext_sol[p] = fem.Function(p.v,name="net_ext_sol")
            self.prev_ext_flux[p] = fem.Function(p.dg0_vec,name="prev_ext_grad")
            self.prev_ext_sol[p] = fem.Function(p.v,name="prev_ext_sol")
            self.gamma_residual[p] = fem.Function(p.v,name="residual")

    def compile_forms(self):
        (p1,p2) = plist = (self.p1,self.p2)
        for p in plist:
            p.set_forms_domain()
            p.set_forms_boundary()
            self.set_robin(p)
            p.compile_forms()

    def set_dirichlet_coefficients(self, h:float, k:float):
        (p1,p2) = (self.p1,self.p2)
        self.dirichlet_coeff[self.p1].value = 1.0/4.0
        self.dirichlet_coeff[self.p2].value =  k / (2 * h)
        self.relaxation_coeff[self.p1].value = 3.0 / 3.0

    def writepos(self,extra_funcs_p1=[],extra_funcs_p2=[]):
        (p1,p2) = (self.p1,self.p2)
        extra_funcs = {p1:extra_funcs_p1,p2:extra_funcs_p2}
        if not(self.writers):
            self.initialize_post()
        for p in [p1,p2]:
            extra_funcs[p] = [p.grad_u, self.ext_sol[p], self.ext_flux[p]] + extra_funcs[p]
        StaggeredDomainDecompositionDriver.write_results(self,extra_funcs_p1=extra_funcs[p1],extra_funcs_p2=extra_funcs[p2])

    def pre_iterate(self):
        super().pre_iterate()
        (p1,p2) = (self.p1,self.p2)
        for p in [p1,p2]:
            if self.relaxation_coeff[p].value < 1.0:
                self.prev_ext_flux[p].x.array[:] = self.ext_flux[p].x.array[:]
                self.prev_ext_sol[p].x.array[:] = self.ext_sol[p].x.array[:]

    def pre_loop(self,set_bc=None,prepare_subproblems=True):
        '''
        1. Interface data
        2. Vars to receive p_ext data
        '''
        StaggeredDomainDecompositionDriver.pre_loop(self,set_bc=set_bc)
        (p1,p2) = (self.p1,self.p2)
        for p,p_ext in zip([p1,p2],[p2,p1]):
            if self.is_chimera: # If different meshes, build interp data
                self.iid_border[p_ext] = dict()
                self.midpoints_facets[p] = mesh.compute_midpoints(p.domain,p.domain.topology.dim-1,p.gamma_facets[p_ext].find(1))

                self.iid_border[p_ext][p] = cellwise_determine_point_ownership(
                                            p_ext.domain._cpp_object,
                                            self.midpoints_facets[p],
                                            self.active_gamma_cells[p_ext],
                                            np.float64(1e-6))
        if prepare_subproblems:
            self.prepare_subproblems()

    def prepare_subproblems(self, finalize=True):
        (p2, p1) = plist = (self.p2, self.p1)
        StaggeredDomainDecompositionDriver.prepare_subproblems(self,finalize=finalize)

    def post_iterate(self, verbose=False):
        (p2, p1) = (self.p2, self.p1)
        self.gamma_residual[p2].x.array[:] = p2.u.x.array-self.previous_u[self.p2].x.array
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
        # Custom measure
        dS = ufl.Measure('ds')
        v = ufl.TestFunction(p.v)
        n = ufl.FacetNormal(p.domain)
        (u, v) = (p.u,ufl.TestFunction(p.v))
        p.a_ufl += + self.dirichlet_coeff[p] * u * v * dS(self.gamma_integration_tag)
        robin_con = self.dirichlet_coeff[p]*self.net_ext_sol[p] + ufl.inner(n,self.net_ext_flux[p])
        p.l_ufl += robin_con * v * dS(self.gamma_integration_tag)

    def update_robin(self,p):
        if p==self.p1:
            p_ext=self.p2
        else:
            p_ext=self.p1
        self.assert_tag(p)
        p_ext.compute_gradient(cells=self.active_gamma_cells[p_ext])
        if self.is_chimera:
            # Update flux
            interpolate_dg0_at_facets([p_ext.grad_u._cpp_object,
                                       p_ext.k._cpp_object],
                                      [self.ext_flux[p]._cpp_object,
                                       self.ext_conductivity[p]._cpp_object],
                                      p.active_els_func._cpp_object,
                                      p.gamma_facets[p_ext]._cpp_object,
                                      self.active_gamma_cells[p],
                                      self.iid_border[p_ext][p],
                                      p.gamma_facets_index_map[p_ext],
                                      p.gamma_imap_to_global_imap[p_ext])
            # Update ext solution
            self.ext_sol[p].interpolate_nonmatching(p_ext.u,
                                                      cells=self.gamma_cells[p],
                                                      interpolation_data=self.iid[p_ext][p])
            self.ext_sol[p].x.scatter_forward()
        else:
            propagate_dg0_at_facets_same_mesh(p_ext, p_ext.grad_u, p, self.ext_flux[p])
            propagate_dg0_at_facets_same_mesh(p_ext, p_ext.k, p, self.ext_conductivity[p])
            # If this is expensive, consider `self.ext_sol[p] = p_ext.u`
            self.ext_sol[p].x.array[:] = p_ext.u.x.array[:]

        # Compute flux
        bsize = self.ext_flux[p].function_space.value_size
        for idx in range(bsize):
            self.ext_flux[p].x.array[self.active_gamma_cells[p]*bsize+idx] *= self.ext_conductivity[p].x.array[self.active_gamma_cells[p]]
        self.ext_flux[p].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.net_ext_sol[p].x.array[:] = self.ext_sol[p].x.array[:]
        self.net_ext_flux[p].x.array[:] = self.ext_flux[p].x.array[:]

        if self.relaxation_coeff[p].value < 1.0:
            theta = self.relaxation_coeff[p].value
            self.net_ext_sol[p].x.array[:] *= theta
            self.net_ext_sol[p].x.array[:] += (1-theta)*self.prev_ext_sol[p].x.array[:]
            self.net_ext_flux[p].x.array[:] *= theta
            self.net_ext_flux[p].x.array[:] += (1-theta)*self.prev_ext_flux[p].x.array[:]

    def iterate(self):
        (p1, p2) = (self.p1,self.p2)

        self.update_robin(p1)
        p1.assemble()
        p1.solve()

        self.update_robin(p2)
        p2.assemble()
        p2.solve()
