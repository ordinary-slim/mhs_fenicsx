from dolfinx import fem, mesh, io
from mhs_fenicsx.problem.helpers import indices_to_function
import ufl
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, interpolate_dg_at_facets, interpolate
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Driver:
    def __init__(self,
                 p_dirichlet:Problem,
                 p_neumann:Problem,
                 max_staggered_iters=40,
                 relaxation_factor=0.5,
                 ):
        self.p_neumann = p_neumann
        self.p_dirichlet = p_dirichlet
        self.max_staggered_iters = max_staggered_iters
        self.convergence_crit = 1e9
        self.convergence_threshold = 1e-6
        self.iter = 0
        self.relaxation_factor = relaxation_factor

    def pre_iterate(self):
        if self.iter == 0:
            self.previous_u_neumann = self.p_neumann.u.copy();self.previous_u_neumann.name="previous_u"
            self.previous_u_dirichlet = self.p_dirichlet.u.copy();self.previous_u_dirichlet.name="previous_u"
        else:
            self.previous_u_neumann.x.array[:] = self.p_neumann.u.x.array
            self.previous_u_dirichlet.x.array[:] = self.p_dirichlet.u.x.array
        self.iter += 1
        
        for p in [self.p_dirichlet, self.p_neumann]:
            p.time = self.iter

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
        # Interpolation data
        self.gamma_cells_d = mesh.compute_incident_entities(pd.domain.topology,
                                                            pd.gammaFacets.find(1),
                                                            pd.dim-1,
                                                            pd.dim)
        self.gamma_cells_n = mesh.compute_incident_entities(pn.domain.topology,
                                                            pn.gammaFacets.find(1),
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
        if set_bc is not None:
            set_bc(pn,pd)
        # Forms and allocation
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
        fs_dirichlet = [self.p_dirichlet.u,
                     self.p_dirichlet.dirichlet_gamma,
                     self.p_dirichlet.active_els_func,
                     self.p_dirichlet.source_rhs,
                     self.previous_u_dirichlet,
                     ]
        fs_neumann = [self.p_neumann.u,
                    self.p_neumann.neumann_flux,
                    self.p_neumann.active_els_func,
                    self.p_neumann.source_rhs,
                    self.previous_u_neumann,
                      ]
        fs_dirichlet.extend(extra_funcs_dirichlet)
        fs_neumann.extend(extra_funcs_neumann)
        self.writer_neumann.write_function(fs_neumann,t=self.iter)
        self.writer_dirichlet.write_function(fs_dirichlet,t=self.iter)

    def post_loop(self):
        pass

    def post_iterate(self, verbose=False):
        (p_neumann,p_dirichlet)=(self.p_neumann,self.p_dirichlet)
        norm_diff_neumann    = p_neumann.l2_norm_gamma(p_neumann.u-self.previous_u_neumann)
        norm_current_neumann = p_neumann.l2_norm_gamma(p_neumann.u)
        self.convergence_crit = np.sqrt(norm_diff_neumann) / np.sqrt(norm_current_neumann)
        if rank==0:
            if verbose:
                print(f"Staggered iteration #{self.iter}, relative norm of difference: {self.convergence_crit}")
    
    def set_dirichlet_interface(self):
        pd = self.p_dirichlet
        # Get Gamma DOFS right
        dofs_gamma_right = fem.locate_dofs_topological(pd.v, pd.dim-1, pd.gammaFacets.find(1))
        self.update_dirichlet_interface()
        # Set Gamma dirichlet
        pd.add_dirichlet_bc(pd.dirichlet_gamma,bdofs=dofs_gamma_right, reset=False)
        pd.is_dirichlet_gamma = True

    def update_dirichlet_interface(self):
        (p, p_ext) = (self.p_dirichlet,self.p_neumann)
        # Interpolate
        p.dirichlet_gamma.interpolate_nonmatching(p_ext.u,
                                                  cells=self.gamma_cells_d,
                                                  interpolation_data=self.iid_n2d)
        p.dirichlet_gamma.x.array[:] = self.relaxation_factor*p.dirichlet_gamma.x.array + \
                                 (1-self.relaxation_factor)*p.u.x.array
    
    def set_neumann_interface(self):
        (p, p_ext) = (self.p_neumann,self.p_dirichlet)
        p_ext.compute_gradient()
        gammaIntegralEntities = p.get_facet_integrations_entities()
        # Custom measure
        dS = ufl.Measure('ds', domain=p.domain, subdomain_data=[
            (8,np.asarray(gammaIntegralEntities, dtype=np.int32))])
        v = ufl.TestFunction(p.v)
        n = ufl.FacetNormal(p.domain)
        self.update_neumann_interface()
        p.l_ufl += +self.ext_conductivity * ufl.inner(n,p.neumann_flux) * v * dS(8)

    def update_neumann_interface(self):
        (p, p_ext) = (self.p_neumann,self.p_dirichlet)
        # Update functions
        p.neumann_flux = interpolate_dg_at_facets(p_ext.grad_u,
                                        p.gammaFacets.find(1),
                                        p.dg0_vec,
                                        p_ext.bb_tree,
                                        p.active_els_tag,
                                        p_ext.active_els_tag,
                                        name="flux",
                                        )

        self.ext_conductivity = interpolate_dg_at_facets(p_ext.k,
                                        p.gammaFacets.find(1),
                                        p.dg0_bg,
                                        p_ext.bb_tree,
                                        p.active_els_tag,
                                        p_ext.active_els_tag,
                                        name="ext_conduc",
                                        )

    def iterate(self):
        (pn, pd) = (self.p_neumann,self.p_dirichlet)
        # Solve right with Dirichlet from left
        self.set_dirichlet_interface()
        pd.set_forms_domain()
        pd.set_forms_boundary()
        pd.compile_forms()
        pd.assemble()
        pd.solve()
        # Solve left with Neumann from right
        pn.set_forms_domain()
        pn.set_forms_boundary()
        self.set_neumann_interface()
        pn.compile_forms()
        pn.assemble()
        pn.solve()

