from dolfinx import fem, mesh, io
from mhs_fenicsx.problem.helpers import indices_to_function
import ufl
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, interpolate_dg_at_facets, interpolate
from mhs_fenicsx.submesh import build_subentity_to_parent_mapping,compute_dg0_interpolation_data,\
        find_submesh_interface
from line_profiler import LineProfiler
import yaml
from mhs_fenicsx.drivers.staggered_drivers import StaggeredDNDriver, StaggeredRRDriver
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

def set_bc(pd,pn):
    # Set outside Dirichlet
    pd.add_dirichlet_bc(exact_sol,marker=right_marker_gamma_dirichlet, reset=True)
    pn.add_dirichlet_bc(exact_sol,marker=left_marker_dirichlet,reset=True)

def run(dd_type="dn",submesh=False):
    dd_type=params["dd_type"] if "dd_type" in params else dd_type
    submesh=params["submesh"] if "submesh" in params else submesh
    if dd_type=="robin":
        driver_type = StaggeredRRDriver
    elif dd_type=="dn":
        driver_type = StaggeredDNDriver
    else:
        raise ValueError("dd_type must be 'dn' or 'robin'")

    # Mesh and problems
    points_side = params["points_side"]
    left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.quadrilateral)
    if submesh:
        dd_type += "_submesh"
    p_left = Problem(left_mesh, params, name=f"left_{dd_type}")
    submesh_data = {}
    if submesh:
        right_els = fem.locate_dofs_geometrical(p_left.dg0_bg, lambda x : x[0] >= 0.5 )
        submesh = mesh.create_submesh(left_mesh,2,right_els)
        right_mesh = submesh[0]
        submesh_data["subcell_map"] = submesh[1]
        submesh_data["subvertex_map"] = submesh[2]
        submesh_data["subgeom_map"] = submesh[3]
    else:
        right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.triangle)
    p_right = Problem(right_mesh, params, name=f"right_{dd_type}")

    if submesh:
        submesh_data["parent"] = p_left
        submesh_data["child"] = p_right
        submesh_data["subfacet_map"] = build_subentity_to_parent_mapping(1,p_left.domain,p_right.domain,
                                                                         submesh_data["subcell_map"],
                                                                         submesh_data["subvertex_map"],)
        find_submesh_interface(p_left,p_right,submesh_data)
        compute_dg0_interpolation_data(p_left,p_right,submesh_data)

    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0_bg, lambda x : x[0] <= 0.5 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0_bg, lambda x : x[0] >= 0.5 )

    f_exact = dict()
    for p in [p_left,p_right]:
        p.set_activation(active_els[p])
        f_exact[p] = fem.Function(p.v,name="exact")
        f_exact[p].interpolate(exact_sol)
        p.set_rhs(rhs)

    driver = driver_type(p_right,p_left,max_staggered_iters=params["max_staggered_iters"],
                         submesh_data=submesh_data)

    driver.pre_loop(set_bc=set_bc)
    for _ in range(driver.max_staggered_iters):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate(verbose=True)
        driver.writepos(extra_funcs_p1=[f_exact[p_right]],extra_funcs_p2=[f_exact[p_left]])
        if driver.convergence_crit < driver.convergence_threshold:
            break
    driver.post_loop()

if __name__=="__main__":
    profiling = True
    tracing = False
    if profiling:
        lp = LineProfiler()
        lp.add_module(StaggeredDNDriver)
        lp.add_module(StaggeredRRDriver)
        lp.add_module(Problem)
        lp.add_function(interpolate)
        lp.add_function(fem.Function.interpolate)
        lp.add_function(interpolate_dg_at_facets)
        lp_wrapper = lp(run)
        lp_wrapper()
        with open(f"profiling_rank{rank}.txt", 'w') as pf:
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
