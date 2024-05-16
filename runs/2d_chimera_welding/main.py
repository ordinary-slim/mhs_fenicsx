from mhs_fenicsx import problem, geometry
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
import ufl
import yaml
from helpers import mesh_around_hs, build_moving_problem, get_active_in_external_trees, interpolate
from line_profiler import LineProfiler
from mhs_fenicsx.geometry import mesh_containment

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

radius = params["heat_source"]["radius"]
max_temporal_iters = params["max_iter"]
T_env = params["environment_temperature"]

def get_el_size(resolution=4.0):
    return params["heat_source"]["radius"] / resolution
def get_dt(adim_dt):
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    return adim_dt * (radius / speed)

box = [-8*radius,-4*radius,+8*radius,+4*radius]
params["dt"] = get_dt(params["adim_dt"])
params["heat_source"]["initial_position"] = [-4*radius, 0.0, 0.0]

class Driver:
    def __init__(self,
                 p_dirichlet:problem.Problem,
                 p_neumann:problem.Problem,
                 max_staggered_iters=20):
        self.p_dirichlet =  p_dirichlet
        self.p_neumann = p_neumann
        self.max_staggered_iters = max_staggered_iters
        self.convergence_crit = 1e9
        self.convergence_threshold = 1e-6
        self.iter = 0
        self.relaxation_factor = 0.5
        self.previous_u_neumann = self.p_neumann.u.copy();self.previous_u_neumann.name="previous_u"
        self.previous_u_dirichlet = self.p_dirichlet.u.copy();self.previous_u_dirichlet.name="previous_u"

    def pre_iterate(self):
        self.iter += 1
        self.p_dirichlet.clear_dirchlet_bcs()
        self.p_neumann.clear_dirchlet_bcs()
        
    def pre_loop(self):
        self.iter = 0
        self.writer_neumann = io.VTKFile(self.p_neumann.domain.comm, f"staggered_iters_{self.p_neumann.name}.pvd", "wb")
        self.writer_dirichlet = io.VTKFile(self.p_dirichlet.domain.comm, f"staggered_iters_{self.p_dirichlet.name}.pvd", "wb")
        #self.writer_neumann.write_function(self.p_neumann.u)
        #self.writer_dirichlet.write_function(self.p_dirichlet.u)

    def writepos(self):
        fs_dirichlet = [self.p_dirichlet.u,
                     self.p_dirichlet.neumann_flux,
                     self.p_dirichlet.active_els_func,
                     self.previous_u_dirichlet
                     ]
        fs_neumann = [self.p_neumann.u,
                    self.p_neumann.dirichlet_gamma,
                    self.p_neumann.active_els_func,
                    self.previous_u_neumann]
        self.writer_neumann.write_function(fs_neumann,t=self.iter)
        self.writer_dirichlet.write_function(fs_dirichlet,t=self.iter)

    def post_loop(self):
        pass

    def post_iterate(self, verbose=False):
        (p_neumann,p_dirichlet)=(self.p_neumann,self.p_dirichlet)
        norm_diff_neumann    = p_neumann.l2_norm_gamma(p_neumann.u-self.previous_u_neumann)
        norm_current_neumann = p_neumann.l2_norm_gamma(p_neumann.u)
        self.convergence_crit = np.sqrt(norm_diff_neumann) / np.sqrt(norm_current_neumann)
        self.previous_u_neumann.x.array[:] = self.p_neumann.u.x.array[:]
        self.previous_u_dirichlet.x.array[:] = self.p_dirichlet.u.x.array[:]
        if rank==0:
            if verbose:
                print(f"Staggered iteration #{self.iter}, relative norm of difference: {self.convergence_crit}")

    def set_dirichlet_interface(self,p:problem.Problem,p_ext:problem.Problem):
        # Get Gamma DOFS right
        dofs_gamma_right = fem.locate_dofs_topological(p.v, p.dim-1, p.gammaFacets.find(1))
        # Interpolate
        interpolate(p_ext.u, p.v,p.dirichlet_gamma)
        p.dirichlet_gamma.x.array[:] = self.relaxation_factor*p.dirichlet_gamma.x.array[:] + \
                                 (1-self.relaxation_factor)*p.u.x.array[:]
        # Set Gamma dirichlet
        p.add_dirichlet_bc(p.dirichlet_gamma,bdofs=dofs_gamma_right, reset=False)
        p.is_dirichlet_gamma = True
    
    def set_neumann_interface(self,p:problem.Problem,p_ext:problem.Problem):
        p_ext.compute_gradient()
        gammaIntegralEntities = p.get_facet_integrations_entities(facet_indices=p.gammaFacets.find(1))
        # Custom measure
        dS = ufl.Measure('ds', domain=p.domain, subdomain_data=[
            (8,np.asarray(gammaIntegralEntities, dtype=np.int32))])
        v = ufl.TestFunction(p.v)
        n = ufl.FacetNormal(p.domain)
        p.neumann_flux = problem.interpolate_dg_at_facets(p_ext.grad_u,
                                                          p.gammaFacets.find(1),
                                                          p.dg0_vec,
                                                          p_ext.bb_tree,
                                                          p.active_els_tag,
                                                          p_ext.active_els_tag,
                                                          name="flux",
                                                          )
        p.l_ufl += +ufl.inner(n,p.neumann_flux) * v * dS(8)

    def iterate(self):
        # Solve right with Dirichlet from left
        self.p_neumann.set_forms_domain()
        self.p_neumann.set_forms_boundary()
        self.set_dirichlet_interface(self.p_neumann,self.p_dirichlet)
        self.p_neumann.compile_forms()
        self.p_neumann.assemble()
        self.p_neumann.solve()
        # Solve left with Neumann from right
        self.p_dirichlet.set_forms_domain()
        self.p_dirichlet.set_forms_boundary()
        self.set_neumann_interface(self.p_dirichlet,self.p_neumann)
        self.p_dirichlet.compile_forms()
        self.p_dirichlet.assemble()
        self.p_dirichlet.solve()

def main():
    point_density = np.round(1/get_el_size()).astype(np.int32)
    nx = np.round((box[2]-box[0]) * point_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * point_density).astype(np.int32)
    domain  = mesh.create_rectangle(MPI.COMM_WORLD,
              [box[:2], box[2:]],
              [nx, ny],
              mesh.CellType.quadrilateral,
              )
    p_fixed = problem.Problem(domain, params, name=params["case_name"])
    p_moving = build_moving_problem(p_fixed)

    p_fixed.set_initial_condition(T_env)
    p_moving.set_initial_condition(T_env)

    driver = Driver(p_fixed,p_moving,params["max_staggered_iters"])

    for _ in range(max_temporal_iters):
        p_fixed.pre_iterate()
        p_moving.pre_iterate()
        p_fixed.subtract_problem(p_moving)
        p_moving.find_gamma(p_moving.get_active_in_external(p_fixed))
        driver.pre_loop()
        for _ in range(driver.max_staggered_iters):
            driver.pre_iterate()
            driver.iterate()
            driver.post_iterate(verbose=True)
            driver.writepos()
            if driver.convergence_crit < driver.convergence_threshold and (driver.iter > 1):
                break
        driver.post_loop()
        p_moving.writepos()
        p_fixed.writepos()

if __name__=="__main__":
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(problem)
        lp.add_function(build_moving_problem)
        lp.add_class(Driver)
        lp_wrapper = lp(main)
        lp_wrapper()
        with open("profiling.txt", 'w') as pf:
            lp.print_stats(stream=pf)

    else:
        main()
