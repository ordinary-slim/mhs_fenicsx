from mhs_fenicsx import problem, geometry
from mhs_fenicsx.problem.helpers import interpolate, interpolate_dg_at_facets
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
import ufl
import yaml
from helpers import interpolate_solution_to_inactive, mesh_around_hs, build_moving_problem, get_active_in_external_trees
from line_profiler import LineProfiler
from mhs_fenicsx.geometry import mesh_containment
from driver2 import Driver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

radius = params["heat_source"]["radius"]
max_temporal_iters = params["max_iter"]
T_env = params["environment_temperature"]
els_per_radius = params["els_per_radius"]

def get_el_size(resolution=4.0):
    return params["heat_source"]["radius"] / resolution
def get_dt(adim_dt):
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    return adim_dt * (radius / speed)

box = [-10*radius,-4*radius,+10*radius,+4*radius]
params["dt"] = get_dt(params["adim_dt"])
params["heat_source"]["initial_position"] = [-4*radius, 0.0, 0.0]

'''
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

    def pre_iterate(self):
        self.previous_u_neumann = self.p_neumann.u.copy();self.previous_u_neumann.name="previous_u"
        self.previous_u_dirichlet = self.p_dirichlet.u.copy();self.previous_u_dirichlet.name="previous_u"
        self.iter += 1
        
    def pre_loop(self):
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
        # Forms and allocation
        self.set_dirichlet_interface()
        pd.set_forms_domain()
        pd.compile_forms()
        pd.pre_assemble()
        pn.set_forms_domain()
        self.set_neumann_interface()
        pn.compile_forms()
        self.p_neumann.pre_assemble()

    def writepos(self):
        fs_dirichlet = [self.p_dirichlet.u,
                     self.p_dirichlet.dirichlet_gamma,
                     self.p_dirichlet.active_els_func,
                     self.previous_u_dirichlet
                     ]
        fs_neumann = [self.p_neumann.u,
                    self.p_neumann.neumann_flux,
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

    def set_dirichlet_interface(self):
        pd= self.p_dirichlet
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
        p.dirichlet_gamma.x.array[:] = self.relaxation_factor*p.dirichlet_gamma.x.array[:] + \
                                 (1-self.relaxation_factor)*p.u.x.array[:]
    
    def set_neumann_interface(self):
        (pn, pd) = (self.p_neumann,self.p_dirichlet)
        pd.compute_gradient()
        gammaIntegralEntities = pn.get_facet_integrations_entities()
        # Custom measure
        dS = ufl.Measure('ds', domain=pn.domain, subdomain_data=[
            (8,np.asarray(gammaIntegralEntities, dtype=np.int32))])
        v = ufl.TestFunction(pn.v)
        n = ufl.FacetNormal(pn.domain)
        self.update_neumann_interface()
        pn.l_ufl += +self.ext_conductivity * ufl.inner(n,pn.neumann_flux) * v * dS(8)

    def update_neumann_interface(self):
        (pn, pd) = (self.p_neumann,self.p_dirichlet)
        # Update functions
        pn.neumann_flux = interpolate_dg_at_facets(pd.grad_u,
                                        pn.gammaFacets.find(1),
                                        pn.dg0_vec,
                                        pd.bb_tree,
                                        pn.active_els_tag,
                                        pd.active_els_tag,
                                        name="flux",
                                        )

        self.ext_conductivity = interpolate_dg_at_facets(pd.k,
                                        pn.gammaFacets.find(1),
                                        pn.dg0_bg,
                                        pd.bb_tree,
                                        pn.active_els_tag,
                                        pd.active_els_tag,
                                        name="ext_conduc",
                                        )

    def iterate(self):
        # Solve right with Dirichlet from left
        self.p_dirichlet.set_forms_domain()
        self.p_dirichlet.set_forms_boundary()
        self.set_dirichlet_interface()
        self.p_dirichlet.compile_forms()
        self.p_dirichlet.assemble()
        self.p_dirichlet.solve()
        # Solve left with Neumann from right
        self.p_neumann.set_forms_domain()
        self.p_neumann.set_forms_boundary()
        self.set_neumann_interface()
        self.p_neumann.compile_forms()
        self.p_neumann.assemble()
        self.p_neumann.solve()
'''

def main():
    point_density = np.round(1/get_el_size(els_per_radius)).astype(np.int32)
    nx = np.round((box[2]-box[0]) * point_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * point_density).astype(np.int32)
    domain  = mesh.create_rectangle(MPI.COMM_WORLD,
              [box[:2], box[2:]],
              [nx, ny],
              mesh.CellType.quadrilateral,
              )
    p_fixed = problem.Problem(domain, params, name=params["case_name"])
    p_moving = build_moving_problem(p_fixed,els_per_radius)

    p_fixed.set_initial_condition(T_env)
    p_moving.set_initial_condition(T_env)

    driver = Driver(p_moving,p_fixed,params["max_staggered_iters"],
                    relaxation_factor=0.5)

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
        #TODO: Interpolate solution at inactive nodes
        interpolate_solution_to_inactive(p_fixed,p_moving)
        p_moving.writepos()
        p_fixed.writepos()

if __name__=="__main__":
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(problem)
        lp.add_function(build_moving_problem)
        lp.add_module(Driver)
        lp_wrapper = lp(main)
        lp_wrapper()
        if rank==0:
            with open("profiling.txt", 'w') as pf:
                lp.print_stats(stream=pf)

    else:
        main()
