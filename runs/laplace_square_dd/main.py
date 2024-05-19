from dolfinx import fem, mesh, io
from mhs_fenicsx.problem.helpers import indices_to_function
import ufl
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, interpolate_dg_at_facets, interpolate
from line_profiler import LineProfiler
import yaml

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

def exact_sol(x):
    return 2 -(x[0]**2 + x[1]**2)

class Rhs:
    def __init__(self,rho,cp,k,v):
        self.rho = rho
        self.cp = cp
        self.k = k
        self.v = v

    def __call__(self,x):
        return_val = -2*self.rho*self.cp*self.v[0]*x[0] + -2*self.rho*self.cp*self.v[1]*x[1] + 4*self.k
        return return_val

rhs = Rhs(params["material"]["density"],
          params["material"]["specific_heat"],
          params["material"]["conductivity"],
          params["advection_speed"])

# Bcs
def left_marker_dirichlet(x):
    return np.logical_or( np.isclose(x[1],1), np.logical_or(
            np.isclose(x[0],0), np.isclose(x[1],0)) )
def right_marker_gamma_dirichlet(x):
    return np.logical_or( np.isclose(x[1],1), np.logical_or(
            np.isclose(x[0],1), np.isclose(x[1],0)) )
def right_marker_neumann(x):
    return np.isclose( x[0],0.5 )

def getPartition(p:Problem):
    f = fem.Function(p.dg0_bg,name="partition")
    f.x.array.fill(rank)
    return f

class Driver:
    def __init__(self, p_neumann:Problem,
                 p_dirichlet:Problem,
                 max_staggered_iters=40):
        self.p_neumann = p_neumann
        self.p_dirichlet = p_dirichlet
        self.max_staggered_iters = max_staggered_iters
        self.convergence_crit = 1e9
        self.convergence_threshold = 1e-6
        self.iter = 0
        self.relaxation_factor = 1

    def pre_iterate(self):
        self.previous_u_neumann = self.p_neumann.u.copy();self.previous_u_neumann.name="previous_u"
        self.previous_u_dirichlet = self.p_dirichlet.u.copy();self.previous_u_dirichlet.name="previous_u"
        self.iter += 1
        
        for p in [self.p_dirichlet, self.p_neumann]:
            p.time = self.iter

    def pre_loop(self):
        (pd,pn) = (self.p_dirichlet,self.p_neumann)
        self.iter = 0
        self.writer_neumann = io.VTKFile(pn.domain.comm, f"staggered_iters_{pn.name}.pvd", "wb")
        self.writer_dirichlet = io.VTKFile(pd.domain.comm, f"staggered_iters_{pd.name}.pvd", "wb")
        pd.clear_dirchlet_bcs()
        pn.clear_dirchlet_bcs()
        # Interpolation data
        self.nmmid_d2n = fem.create_nonmatching_meshes_interpolation_data(
                                                     pn.domain,
                                                     pn.v.element,
                                                     pd.domain,
                                                     padding=1e-6,)
        self.nmmid_n2d = fem.create_nonmatching_meshes_interpolation_data(
                                                     pd.domain,
                                                     pd.v.element,
                                                     pn.domain,
                                                     padding=1e-6,)
        # Find interface
        pn.find_gamma(pn.get_active_in_external( pd ))
        pd.find_gamma(pd.get_active_in_external( pn ))
        # Set outside Dirichlet
        pn.add_dirichlet_bc(exact_sol,marker=left_marker_dirichlet,reset=True)
        pd.add_dirichlet_bc(exact_sol,marker=right_marker_gamma_dirichlet, reset=True)
        # Forms and allocation
        self.set_dirichlet_interface()
        pd.set_forms_domain(rhs)
        pd.compile_forms()
        pd.pre_assemble()
        pn.set_forms_domain(rhs)
        self.set_neumann_interface()
        pn.compile_forms()
        self.p_neumann.pre_assemble()

    def writepos(self):
        exact_dirichlet = fem.Function(self.p_dirichlet.v,name="exact")
        exact_neumann = fem.Function(self.p_neumann.v,name="exact")
        exact_dirichlet.interpolate(exact_sol)
        exact_neumann.interpolate(exact_sol)
        fs_dirichlet = [self.p_dirichlet.u,
                     self.p_dirichlet.dirichlet_gamma,
                     self.p_dirichlet.active_els_func,
                     self.p_dirichlet.source_rhs,
                     self.previous_u_dirichlet,
                     exact_dirichlet,
                     ]
        fs_neumann = [self.p_neumann.u,
                    self.p_neumann.neumann_flux,
                    self.p_neumann.active_els_func,
                    self.p_neumann.source_rhs,
                    self.previous_u_neumann,
                    exact_neumann,
                      ]
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
        (p, p_ext) = (self.p_dirichlet,self.p_neumann)
        # Get Gamma DOFS right
        dofs_gamma_right = fem.locate_dofs_topological(p.v, p.dim-1, p.gammaFacets.find(1))
        self.update_dirichlet_interface()
        # Set Gamma dirichlet
        p.add_dirichlet_bc(p.dirichlet_gamma,bdofs=dofs_gamma_right, reset=False)
        p.is_dirichlet_gamma = True

    def update_dirichlet_interface(self):
        (p, p_ext) = (self.p_dirichlet,self.p_neumann)
        # Interpolate
        gamma_els = mesh.compute_incident_entities(p.domain.topology,
                                                   p.gammaFacets.find(1),
                                                   p.domain.topology.dim-1,
                                                   p.domain.topology.dim,)
        p.dirichlet_gamma.interpolate(p_ext.u,
                                      #cells=gamma_els,
                                      nmm_interpolation_data=self.nmmid_n2d)
        p.dirichlet_gamma.x.array[:] = self.relaxation_factor*p.dirichlet_gamma.x.array[:] + \
                                 (1-self.relaxation_factor)*p.u.x.array[:]
    
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
        # Solve right with Dirichlet from left
        self.set_dirichlet_interface()
        self.p_dirichlet.set_forms_domain(rhs)
        self.p_dirichlet.compile_forms()
        self.p_dirichlet.assemble()
        self.p_dirichlet.solve()
        # Solve left with Neumann from right
        self.p_neumann.set_forms_domain(rhs)
        self.set_neumann_interface()
        self.p_neumann.compile_forms()
        self.p_neumann.assemble()
        self.p_neumann.solve()


def main():
    # Mesh and problems
    points_side = 64
    left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.quadrilateral)
    right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.triangle)
    p_left = Problem(left_mesh, params, name="left")
    p_right = Problem(right_mesh, params, name="right")
    # Activation
    active_els_left = fem.locate_dofs_geometrical(p_left.dg0_bg, lambda x : x[0] <= 0.5 )
    active_els_right = fem.locate_dofs_geometrical(p_right.dg0_bg, lambda x : x[0] >= 0.5 )

    p_left.set_activation( active_els_left )
    p_right.set_activation( active_els_right )

    driver = Driver(p_left,p_right,max_staggered_iters=params["max_staggered_iters"])
    driver.pre_loop()
    for _ in range(driver.max_staggered_iters):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate(verbose=True)
        driver.writepos()
        if driver.convergence_crit < driver.convergence_threshold and driver.iter > 10:
            break

if __name__=="__main__":
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(Driver)
        lp.add_module(Problem)
        lp.add_function(interpolate)
        lp.add_function(fem.Function.interpolate)
        lp.add_function(interpolate_dg_at_facets)
        lp_wrapper = lp(main)
        lp_wrapper()
        if rank==0:
            with open("profiling.txt", 'w') as pf:
                lp.print_stats(stream=pf)
    else:
        main()
