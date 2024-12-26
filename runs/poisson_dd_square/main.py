from dolfinx import fem, mesh
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

def exact_sol_2d(x):
    return 2 -(x[0]**2 + x[1]**2)
def grad_exact_sol_2d(domain):
    x = ufl.SpatialCoordinate(domain)
    return ufl.as_vector((-2*x[0], -2*x[1]))
def exact_sol_3d(x):
    return 2 -(x[0]**2 + x[1]**2 + x[2]**2)
def grad_exact_sol_3d(domain):
    x = ufl.SpatialCoordinate(domain)
    return ufl.as_vector((-2*x[0], -2*x[1], -2*x[2]))

class Rhs:
    def __init__(self,rho,cp,k,v,dim=None):
        self.rho = rho
        self.cp = cp
        self.k = k
        self.v = v
        if dim==None:
            dim = len(v)
        self.dim = dim

    def __call__(self,x):
        return_val = -2*self.rho*self.cp*self.v[0]*x[0] + -2*self.rho*self.cp*self.v[1]*x[1] + (2*self.dim)*self.k
        return return_val

rhs = Rhs(params["material"]["density"],
          params["material"]["specific_heat"],
          params["material"]["conductivity"],
          params["advection_speed"],
          params["dim"])

# Bcs
def left_marker_dirichlet(x):
    return np.logical_or( np.isclose(x[1],1), np.logical_or(
            np.isclose(x[0],0), np.isclose(x[1],0)) )
def right_marker_dirichlet(x):
    return np.logical_or( np.isclose(x[1],1), np.logical_or(
            np.isclose(x[0],1), np.isclose(x[1],0)) )
def right_marker_neumann(x):
    return np.isclose( x[0],0.5 )

def getPartition(p:Problem):
    f = fem.Function(p.dg0,name="partition")
    f.x.array.fill(rank)
    return f

def set_bc(pd:Problem,pn:Problem):
    # Set outside Dirichlet
    pd.add_dirichlet_bc(exact_sol_2d,marker=right_marker_dirichlet, reset=True)
    pn.add_dirichlet_bc(exact_sol_2d,marker=left_marker_dirichlet,reset=True)

def run(run_type="dd"):
    dd_type=params["dd_type"]
    if dd_type=="robin":
        driver_type = StaggeredRRDriver
    elif dd_type=="dn":
        driver_type = StaggeredDNDriver
    else:
        raise ValueError("dd_type must be 'dn' or 'robin'")

    # Mesh and problems
    els_side = params["els_side"]
    left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.quadrilateral)
    if run_type=="submesh":
        dd_type += "_submesh"
    p_left = Problem(left_mesh, params, name=f"left_{dd_type}")
    submesh_data = {}
    if run_type=="submesh":
        right_els = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] >= 0.5 )
        submesh = mesh.create_submesh(left_mesh,2,right_els)
        right_mesh = submesh[0]
        submesh_data["subcell_map"] = submesh[1]
        submesh_data["subvertex_map"] = submesh[2]
        submesh_data["subgeom_map"] = submesh[3]
    else:
        right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.triangle)
    p_right = Problem(right_mesh, params, name=f"right_{dd_type}")

    if run_type=="submesh":
        submesh_data["parent"] = p_left
        submesh_data["child"] = p_right
        submesh_data["subfacet_map"] = build_subentity_to_parent_mapping(1,p_left.domain,p_right.domain,
                                                                         submesh_data["subcell_map"],
                                                                         submesh_data["subvertex_map"],)
        find_submesh_interface(p_left,p_right,submesh_data)
        compute_dg0_interpolation_data(p_left,p_right,submesh_data)

    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= 0.5 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= 0.5 )

    f_exact = dict()
    for p in [p_left,p_right]:
        p.set_activation(active_els[p])
        f_exact[p] = fem.Function(p.v,name="exact")
        f_exact[p].interpolate(exact_sol_2d)
        p.set_rhs(rhs)

    driver = driver_type(p_right,p_left,max_staggered_iters=params["max_staggered_iters"],
                         submesh_data=submesh_data,
                         initial_relaxation_factors=[0.5,1.0])
    if (type(driver)==StaggeredRRDriver):
        h = 1.0 / els_side
        k = float(params["material"]["conductivity"])
        driver.dirichlet_coeff[driver.p1] = 4.0
        driver.dirichlet_coeff[driver.p2] =  k / (4 * h)
        driver.relaxation_coeff[driver.p1].value = 2.0 / 3.0

    driver.pre_loop(set_bc=set_bc)
    for _ in range(driver.max_staggered_iters):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate(verbose=True)
        driver.writepos(extra_funcs_p1=[f_exact[p_right]],extra_funcs_p2=[f_exact[p_left]])
        if driver.convergence_crit < driver.convergence_threshold:
            break
    driver.post_loop()

def run_same_mesh(run_type="_"):
    from petsc4py import PETSc
    els_side = params["els_side"]
    domain  = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.quadrilateral)
    p_left = Problem(domain, params, name=f"sm_left")
    p_right = p_left.copy(name="sm_right")
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= 0.5 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= 0.5 )

    for p in [p_left,p_right]:
        p.set_activation(active_els[p])
        p.set_rhs(rhs)
    mask = p_left.active_els_func.x.array + 2*p_right.active_els_func.x.array
    subdomain_data = mesh.meshtags(p_left.domain, p_left.dim,
                                        np.arange(p_left.num_cells, dtype=np.int32),
                                        mask,)

    idx = 1
    for p in [p_left,p_right]:
        p.set_forms_domain(subdomain_data=(idx,subdomain_data))
        #p.set_forms_boundary()
        idx += 1

    res = dict()
    u2r = dict()

    for p in [p_left,p_right]:
        res[p] = fem.Function(p.v,name="restriction")
        u2r[p] = p.restriction.unrestricted_to_restricted.copy()
        for i in range(p.v.dofmap.index_map.size_global):
            try:
                res[p].x.array[i] = u2r[p][i]
            except KeyError:
                res[p].x.array[i] = -1

    a = p_left.a_ufl + p_right.a_ufl
    l = p_left.l_ufl + p_right.l_ufl
    a_cpp = fem.form(a)
    l_cpp = fem.form(l)

    p_left.add_dirichlet_bc(exact_sol_2d,marker=left_marker_dirichlet,reset=True)
    p_right.add_dirichlet_bc(exact_sol_2d,marker=right_marker_dirichlet, reset=True)

    bcs = [p_left.dirichlet_bcs[0], p_right.dirichlet_bcs[0]]

    A = fem.petsc.assemble_matrix(a_cpp, bcs=bcs)
    A.assemble()
    L = fem.petsc.assemble_vector(l_cpp)

    fem.petsc.apply_lifting(L, [a_cpp], [p_left.dirichlet_bcs + p_right.dirichlet_bcs])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(L,p_left.dirichlet_bcs + p_right.dirichlet_bcs)
    # Solve
    x = fem.petsc.create_vector(l_cpp)
    ksp = PETSc.KSP()
    ksp.create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=7, ival=4)
    ksp.setFromOptions()
    ksp.solve(L, p_left.u.vector)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    ksp.destroy()

    for p in [p_left, p_right]:
        p.writepos(extra_funcs=[res[p]])

if __name__=="__main__":
    profiling = True
    tracing = False
    run_type = params["run_type"]
    func = run_same_mesh if run_type=="same_mesh" else run
    if profiling:
        lp = LineProfiler()
        lp.add_module(StaggeredDNDriver)
        lp.add_module(StaggeredRRDriver)
        lp.add_module(Problem)
        lp.add_function(interpolate)
        lp.add_function(fem.Function.interpolate)
        lp.add_function(interpolate_dg_at_facets)
        lp_wrapper = lp(func)
        lp_wrapper(run_type=run_type)
        with open(f"profiling_rank{rank}.txt", 'w') as pf:
            lp.print_stats(stream=pf)
    else:
        func(run_type=run_type)
