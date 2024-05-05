from dolfinx import fem, mesh
import ufl
import numpy as np
from mpi4py import MPI
from Problem import Problem, interpolate_dg_at_facets, interpolate
from line_profiler import LineProfiler
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def exact_sol(x):
    return 2 -(x[0]**2 + x[1]**2)
def exact_sol_ufl(mesh):
    x = ufl.SpatialCoordinate(mesh)
    return 2 -(x[0]**2 + x[1]**2)
def exact_flux_ufl(mesh):
    x = ufl.SpatialCoordinate(mesh)
    return ufl.as_vector((-2*x[0], -2*x[1]))
def rhs():
    return 4
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
    def __init__(self, p_neumann:Problem, p_dirichlet:Problem, max_iter=5):
        self.p_neumann = p_neumann
        self.p_dirichlet = p_dirichlet
        self.max_iter = max_iter
        self.convergence_crit = 1e9
        self.convergence_threshold = 1e-6
        self.iter = 0
        self.relaxation_factor = 0.7

    def pre_iterate(self):
        self.previous_u_neumann = self.p_neumann.u.copy();self.previous_u_neumann.name="previous_u"
        self.previous_u_dirichlet = self.p_dirichlet.u.copy();self.previous_u_dirichlet.name="previous_u"
        self.iter += 1
        self.p_dirichlet.clear_dirchlet_bcs()
        self.p_neumann.clear_dirchlet_bcs()
        
        for p in [self.p_dirichlet, self.p_neumann]:
            p.time = self.iter

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
        # Interpolate
        interpolate(p_ext.u, p.v,p.dirichlet_gamma)
        p.dirichlet_gamma.x.array[:] = self.relaxation_factor*p.dirichlet_gamma.x.array[:] + \
                                 (1-self.relaxation_factor)*p.u.x.array[:]
        # Set Gamma dirichlet
        p.add_dirichlet_bc(p.dirichlet_gamma,bdofs=dofs_gamma_right, reset=False)
        p.is_dirichlet_gamma = True
    
    def set_neumann_interface(self):
        (p, p_ext) = (self.p_neumann,self.p_dirichlet)
        p_ext.compute_gradient()
        gammaIntegralEntities = p.compute_gamma_integration_ents()
        # Custom measure
        dS = ufl.Measure('ds', domain=p.domain, subdomain_data=[
            (8,np.asarray(gammaIntegralEntities, dtype=np.int32))])
        v = ufl.TestFunction(p.v)
        n = ufl.FacetNormal(p.domain)
        p.neumann_flux = interpolate_dg_at_facets(p_ext.grad_u,
                                        p.gammaFacets.find(1),
                                        p.dg0_dim2,
                                        p_ext.bb_tree,
                                        p.active_els_tag,
                                        p_ext.active_els_tag,
                                        name="flux",
                                        )
        p.l_ufl += +ufl.inner(n,p.neumann_flux) * v * dS(8)


    def iterate(self):
        self.p_neumann.find_gamma(self.p_dirichlet)
        self.p_dirichlet.find_gamma(self.p_neumann)

        self.p_neumann.add_dirichlet_bc(exact_sol,marker=left_marker_dirichlet,reset=True)
        self.p_dirichlet.add_dirichlet_bc(exact_sol,marker=right_marker_gamma_dirichlet, reset=True)

        # Solve left with Neumann from right
        self.p_neumann.set_forms_domain(rhs)
        self.set_neumann_interface()
        self.p_neumann.assemble()
        self.p_neumann.solve()

        # Solve right with Dirichlet from left
        self.set_dirichlet_interface()
        self.p_dirichlet.set_forms_domain(rhs)
        self.p_dirichlet.assemble()
        self.p_dirichlet.solve()


    def writepos(self):
        # Post
        # exact sol
        ex_left = fem.Function(self.p_neumann.v, name="exact")
        ex_right = fem.Function(self.p_dirichlet.v, name="exact")
        ex_left.interpolate(fem.Expression(exact_sol_ufl(self.p_neumann.domain),self.p_neumann.v.element.interpolation_points()))
        ex_right.interpolate(fem.Expression(exact_sol_ufl(self.p_dirichlet.domain),self.p_dirichlet.v.element.interpolation_points()))
        # partition
        partition_left, partition_right = getPartition(self.p_neumann), getPartition(self.p_dirichlet)
        # els numbering
        els_numbering_right = fem.Function(self.p_dirichlet.dg0_bg, name="numbering")
        els_numbering_right.x.array[:] = np.arange(self.p_dirichlet.num_cells)[:]
        els_numbering_left = fem.Function(self.p_neumann.dg0_bg, name="numbering")
        els_numbering_left.x.array[:] = np.arange(self.p_neumann.num_cells)[:]
        # nodes numbering
        nodes_numbering = fem.Function(self.p_neumann.v, name="numbering")
        nodes_numbering.x.array[:] = np.arange(self.p_neumann.domain.topology.index_map(0).size_local+self.p_neumann.domain.topology.index_map(0).num_ghosts)[:]

        self.p_neumann.writepos(extra_funcs=[ex_left, partition_left,nodes_numbering,els_numbering_left,self.p_neumann.neumann_flux,self.previous_u_neumann])
        self.p_dirichlet.writepos(extra_funcs=[ex_right, partition_right,els_numbering_right,self.previous_u_dirichlet])

def main():
    # Mesh and problems
    points_side = 16
    left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.quadrilateral)
    right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.triangle)
    p_left = Problem(left_mesh, name="left")
    p_right = Problem(right_mesh, name="right")
    # Activation
    active_els_left = fem.locate_dofs_geometrical(p_left.dg0_bg, lambda x : x[0] <= 0.5 )
    active_els_right = fem.locate_dofs_geometrical(p_right.dg0_bg, lambda x : x[0] >= 0.5 )

    p_left.set_activation( active_els_left )
    p_right.set_activation( active_els_right )

    driver = Driver(p_left,p_right,max_iter=40)
    for _ in range(driver.max_iter):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate(verbose=True)
        driver.writepos()
        if driver.convergence_crit < driver.convergence_threshold:
            break

if __name__=="__main__":
    profiling = False
    if profiling:
        lp = LineProfiler()
        lp.add_module(Driver)
        lp.add_module(Problem)
        lp.add_function(interpolate)
        lp.add_function(fem.Function.interpolate)
        lp.add_function(interpolate_dg_at_facets)
        lp_wrapper = lp(main)
        lp_wrapper()
        lp.print_stats()
    else:
        main()
