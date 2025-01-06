from dolfinx import fem, mesh
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.problem.helpers import assert_pointwise_vals
import yaml
from mhs_fenicsx.drivers.staggered_drivers import StaggeredDNDriver, StaggeredRRDriver
from mhs_fenicsx_cpp import cellwise_determine_point_ownership

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
    f = fem.Function(p.dg0,name="partition")
    f.x.array.fill(rank)
    return f

def set_bc(pright,pleft):
    # Set outside Dirichlet
    pright.add_dirichlet_bc(exact_sol,marker=right_marker_gamma_dirichlet, reset=True)
    pleft.add_dirichlet_bc(exact_sol,marker=left_marker_dirichlet,reset=True)

def run(dd_type="dn"):
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    rhs = Rhs(params["material"]["density"],
              params["material"]["specific_heat"],
              params["material"]["conductivity"],
              params["advection_speed"])
    if dd_type=="robin":
        driver_type = StaggeredRRDriver
        initial_relaxation_factors=[1.0,1.0]
    elif dd_type=="dn":
        driver_type = StaggeredDNDriver
        initial_relaxation_factors=[0.5,1.0]
    else:
        raise ValueError("dd_type must be 'dn' or 'robin'")
    # Mesh and problems
    points_side = params["points_side"]
    left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.quadrilateral)
    right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.triangle)
    p_left = Problem(left_mesh, params, name=f"left_{dd_type}")
    p_right = Problem(right_mesh, params, name=f"right_{dd_type}")
    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= 0.5 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= 0.5 )

    f_exact = dict()
    for p in [p_left,p_right]:
        p.set_activation(active_els[p])
        f_exact[p] = fem.Function(p.v,name="exact")
        f_exact[p].interpolate(exact_sol)
        p.set_rhs(rhs)

    driver = driver_type(p_right,p_left,max_staggered_iters=params["max_staggered_iters"],
                         initial_relaxation_factors=initial_relaxation_factors)

    driver.pre_loop(set_bc=set_bc)
    for _ in range(driver.max_staggered_iters):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate(verbose=True)
        if driver.convergence_crit < driver.convergence_threshold:
            break
    driver.post_loop()
    return driver


tol = 1e-7
points_left = np.array([[0.25,0.5,0.0],[0.5,0.5,0.0],])
points_right = np.array([[0.5,0.5,0.0],[0.75,0.5,0.0],])
vals_left_dn = np.array([1.68532511425681, 1.49462879481564])
vals_right_dn = np.array([1.4946288473, 1.1853250916])
vals_left_robin = np.array([1.6853234504, 1.4946274331])
vals_right_robin = np.array([1.4946278829, 1.1853250921])

def test_poisson_dd_dn():
    driver = run("dn")
    p_right,p_left = (driver.p1,driver.p2)
    assert (driver.iter == 11)
    for problem,points,vals in zip([p_left,p_right],[points_left,points_right],[vals_left_dn,vals_right_dn]):
        assert_pointwise_vals(problem,points,vals)

def test_poisson_dd_robin():
    driver = run("robin")
    p_right,p_left = (driver.p1,driver.p2)
    assert (driver.iter == 14)
    for problem,points,vals in zip([p_left,p_right],[points_left,points_right],[vals_left_robin,vals_right_robin]):
        assert_pointwise_vals(problem,points,vals)

if __name__=="__main__":
    test_poisson_dd_dn()
    test_poisson_dd_robin()
