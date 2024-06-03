from dolfinx import fem, mesh, io
from mhs_fenicsx.problem.helpers import indices_to_function
import ufl
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, interpolate_dg_at_facets, interpolate
from mhs_fenicsx_cpp import interpolate_dg0_at_facets, cellwise_determine_point_ownership
from line_profiler import LineProfiler
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class StaggeredDNDriver:
    def __init__(self,
                 p_dirichlet:Problem,
                 p_neumann:Problem,
                 max_staggered_iters=40,
                 initial_relaxation_factor=0.5,
                 ):
        self.p_neumann = p_neumann
        self.p_dirichlet = p_dirichlet
        self.max_staggered_iters = max_staggered_iters
        self.convergence_crit = 1e9
        self.convergence_threshold = 1e-6
        self.iter = 0
        self.initial_relaxation_factor = initial_relaxation_factor
        self.relaxation_factor = initial_relaxation_factor
        self.dirichlet_tcon = None

    def pre_iterate(self):
        if self.iter == 0:
            self.previous_u_neumann = self.p_neumann.u.copy();self.previous_u_neumann.name="previous_u"
            self.previous_u_dirichlet = self.p_dirichlet.u.copy();self.previous_u_dirichlet.name="previous_u"
        else:
            self.previous_u_neumann.x.array[:] = self.p_neumann.u.x.array
            self.previous_u_dirichlet.x.array[:] = self.p_dirichlet.u.x.array
        self.iter += 1

    def pre_loop(self,set_bc=None):
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        self.iter = 0
        self.writer_neumann = io.VTKFile(pn.domain.comm, f"staggered_out/staggered_iters_{pn.name}.pvd", "wb")
        self.writer_dirichlet = io.VTKFile(pd.domain.comm, f"staggered_out/staggered_iters_{pd.name}.pvd", "wb")
        pd.clear_dirchlet_bcs()
        pn.clear_dirchlet_bcs()
        # Find interface
        pn.find_gamma(pn.get_active_in_external( pd ))
        pd.find_gamma(pd.get_active_in_external( pn ))
        self.gamma_dofs_neumann   = pn.gamma_nodes.x.array.nonzero()[0]
        self.gamma_dofs_dirichlet = pd.gamma_nodes.x.array.nonzero()[0]
        # Interpolation data
        self.gamma_cells_d = mesh.compute_incident_entities(pd.domain.topology,
                                                            np.hstack((pd.gamma_facets.find(1),pd.gamma_facets.find(2))),
                                                            pd.dim-1,
                                                            pd.dim)
        self.gamma_cells_n = mesh.compute_incident_entities(pn.domain.topology,
                                                            np.hstack((pn.gamma_facets.find(1),pn.gamma_facets.find(2))),
                                                            pn.dim-1,
                                                            pn.dim)
        self.iid_d2n = fem.create_interpolation_data(
                                             pn.v,
                                             pd.v,
                                             self.gamma_cells_n,
                                             padding=1e-6,)
        self.iid_n2d = fem.create_interpolation_data(
                                             pd.v,
                                             pn.v,
                                             self.gamma_cells_d,
                                             padding=1e-6,)
        midpoints_neumann_facets = mesh.compute_midpoints(pn.domain,pn.domain.topology.dim-1,pn.gamma_facets.find(1))
        active_gamma_cells_d = self.gamma_cells_d[pd.active_els_func.x.array[self.gamma_cells_d].nonzero()[0]]
        self.iid_d2n_border = cellwise_determine_point_ownership(
                                    pd.domain._cpp_object,
                                    midpoints_neumann_facets,
                                    active_gamma_cells_d,
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

        # Ext bc
        if set_bc is not None:
            set_bc(pn,pd)

        # Forms and allocation
        # Interpolate
        self.set_dirichlet_interface()
        pd.set_forms_domain()
        pd.set_forms_boundary()
        pd.compile_forms()
        pd.pre_assemble()
        pn.set_forms_domain()
        pn.set_forms_boundary()
        self.set_neumann_interface()
        pn.compile_forms()
        self.p_neumann.pre_assemble()

    def writepos(self,extra_funcs_dirichlet=[],extra_funcs_neumann=[]):
        (pn, pd) = (self.p_neumann, self.p_dirichlet)
        fs_dirichlet = [pd.u,
                     pd.dirichlet_gamma,
                     pd.active_els_func,
                     pd.source_rhs,
                     self.previous_u_dirichlet,
                     ]
        fs_neumann = [pn.u,
                    pn.neumann_flux,
                    pn.active_els_func,
                    pn.source_rhs,
                    self.previous_u_neumann,
                      ]
        fs_dirichlet.extend(extra_funcs_dirichlet)
        fs_neumann.extend(extra_funcs_neumann)
        # PARTITION
        partition_d = fem.Function(pd.dg0_bg,name="partition")
        partition_n = fem.Function(pn.dg0_bg,name="partition")
        partition_d.x.array[:] = rank
        partition_n.x.array[:] = rank
        fs_dirichlet.append(partition_d)
        fs_neumann.append(partition_n)
        # EPARTITION
        self.writer_neumann.write_function(fs_neumann,t=self.iter)
        self.writer_dirichlet.write_function(fs_dirichlet,t=self.iter)

    def post_loop(self):
        self.p_neumann.post_iterate()
        self.p_dirichlet.post_iterate()

    def post_iterate(self, verbose=False):
        (pn, pd) = (self.p_neumann, self.p_dirichlet)
        norm_diff_neumann    = pn.l2_dot_gamma(pn.u-self.previous_u_neumann)
        norm_current_neumann = pn.l2_dot_gamma(pn.u)
        self.convergence_crit = np.sqrt(norm_diff_neumann) / np.sqrt(norm_current_neumann)
        norm_res = pd.l2_dot_gamma(self.dirichlet_res)
        if rank==0:
            if verbose:
                print(f"Staggered iteration #{self.iter}, omega = {self.relaxation_factor}, relative norm of difference: {self.convergence_crit}, norm residual: {norm_res}")

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
                                                  cells=self.gamma_cells_d,
                                                  interpolation_data=self.iid_n2d)
        pd.dirichlet_gamma.x.scatter_forward()
        if self.iter > 1:
            self.dirichlet_prev_res.x.array[self.gamma_dofs_dirichlet] = self.dirichlet_res.x.array[self.gamma_dofs_dirichlet]

        self.dirichlet_res.x.array[self.gamma_dofs_dirichlet] = pd.dirichlet_gamma.x.array[self.gamma_dofs_dirichlet] - \
                               pd.u.x.array[self.gamma_dofs_dirichlet]
        self.dirichlet_res.x.scatter_forward()
        if self.iter > 1:
            self.dirichlet_res_diff.x.array[self.gamma_dofs_dirichlet] = self.dirichlet_res.x.array[self.gamma_dofs_dirichlet] - \
                    self.dirichlet_prev_res.x.array[self.gamma_dofs_dirichlet]
            self.dirichlet_res_diff.x.scatter_forward()
            self.relaxation_factor = - self.relaxation_factor * pd.l2_dot_gamma(self.dirichlet_prev_res,self.dirichlet_res_diff)
            self.relaxation_factor /= pd.l2_dot_gamma(self.dirichlet_res_diff)
            #self.relaxation_factor = self.initial_relaxation_factor
            #self.relaxation_factor = np.sign(self.relaxation_factor)*min(abs(self.relaxation_factor),abs(self.initial_relaxation_factor))


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
                                  self.gamma_cells_n,
                                  self.iid_d2n_border,
                                  p.gamma_facets_index_map)
        interpolate_dg0_at_facets(p_ext.k._cpp_object,
                                  self.ext_conductivity._cpp_object,
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets._cpp_object,
                                  self.gamma_cells_n,
                                  self.iid_d2n_border,
                                  p.gamma_facets_index_map)
        '''
        interpolate_dg_at_facets(p_ext.grad_u,
                                 p.neumann_flux,
                                 p.gamma_facets.find(1),
                                 p_ext.bb_tree,
                                 p.active_els_tag,
                                 p_ext.active_els_tag,
                                 )

        interpolate_dg_at_facets(p_ext.k,
                                 self.ext_conductivity,
                                 p.gamma_facets.find(1),
                                 p_ext.bb_tree,
                                 p.active_els_tag,
                                 p_ext.active_els_tag,
                                 )
        '''

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
