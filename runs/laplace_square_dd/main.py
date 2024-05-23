from dolfinx import fem, mesh, io
from mhs_fenicsx.problem.helpers import indices_to_function
import ufl
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, interpolate_dg_at_facets, interpolate
from line_profiler import LineProfiler
import yaml
from mhs_fenicsx.drivers.staggered_dn_driver import StaggeredDNDriver
import trace, sys

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

def set_bc(pn,pd):
    # Set outside Dirichlet
    pn.add_dirichlet_bc(exact_sol,marker=left_marker_dirichlet,reset=True)
    pd.add_dirichlet_bc(exact_sol,marker=right_marker_gamma_dirichlet, reset=True)

def main():
    # Mesh and problems
    points_side = params["points_side"]
    left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.quadrilateral)
    right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.triangle)
    p_left = Problem(left_mesh, params, name="left")
    p_right = Problem(right_mesh, params, name="right")
    # Activation
    active_els_left = fem.locate_dofs_geometrical(p_left.dg0_bg, lambda x : x[0] <= 0.5 )
    active_els_right = fem.locate_dofs_geometrical(p_right.dg0_bg, lambda x : x[0] >= 0.5 )

    p_left.set_activation( active_els_left )
    p_right.set_activation( active_els_right )

    exact_left = fem.Function(p_left.v,name="exact")
    exact_right = fem.Function(p_right.v,name="exact")
    exact_left.interpolate(exact_sol)
    exact_right.interpolate(exact_sol)
    p_left.set_rhs(rhs)
    p_right.set_rhs(rhs)

    driver = StaggeredDNDriver(p_right,p_left,max_staggered_iters=params["max_staggered_iters"])

    driver.pre_loop(set_bc=set_bc)
    for _ in range(driver.max_staggered_iters):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate(verbose=True)
        driver.writepos(extra_funcs_neumann=[exact_left],extra_funcs_dirichlet=[exact_right])
        if driver.convergence_crit < driver.convergence_threshold:
            break
    driver.post_loop()

if __name__=="__main__":
    profiling = True
    tracing = False
    if profiling:
        lp = LineProfiler()
        lp.add_module(StaggeredDNDriver)
        lp.add_module(Problem)
        lp.add_function(interpolate)
        lp.add_function(fem.Function.interpolate)
        lp.add_function(interpolate_dg_at_facets)
        lp_wrapper = lp(main)
        lp_wrapper()
        if rank==0:
            with open("profiling.txt", 'w') as pf:
                lp.print_stats(stream=pf)
    elif tracing:
        # define Trace object: trace line numbers at runtime, exclude some modules
        tracer = trace.Trace(
            ignoredirs=[sys.prefix, sys.exec_prefix],
            ignoremods=[
                'inspect', 'contextlib', '_bootstrap',
                '_weakrefset', 'abc', 'posixpath', 'genericpath', 'textwrap'
            ],
            trace=1,
            count=0)
        tracer.runfunc(main)
    else:
        main()
