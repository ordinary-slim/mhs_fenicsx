from dolfinx import fem, mesh, la
import numpy as np
from mpi4py import MPI
import yaml
from mhs_fenicsx.problem import Problem
import ufl
from line_profiler import LineProfiler
from mhs_fenicsx.drivers import MonolithicRRDriver
from mhs_fenicsx.problem.helpers import assert_pointwise_vals

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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

def left_marker_dirichlet(x):
    return np.isclose(x[0],0)
def right_marker_dirichlet(x):
    return np.isclose(x[0],1)
def left_marker_neumann_2d(p:Problem):
    return mesh.locate_entities(p.domain, p.dim-1,
                                lambda x : np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] <= 0.5)
                                )
def right_marker_neumann_2d(p:Problem):
    return mesh.locate_entities(p.domain, p.dim-1,
                                lambda x : np.logical_and(np.logical_or(np.isclose(x[1],0), np.isclose(x[1],1)), x[0] >= 0.5)
                                )
def left_marker_neumann_3d(p:Problem):
    y_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[1], 0),
                                          np.isclose(x[1], 1)),
                            x[0] <= 0.5))
    z_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[2], 0),
                                          np.isclose(x[2], 1)),
                            x[0] <= 0.5))
    return np.hstack((y_facets,z_facets))

def right_marker_neumann_3d(p:Problem):
    y_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[1], 0),
                                          np.isclose(x[1], 1)),
                            x[0] >= 0.5))
    z_facets = mesh.locate_entities(p.domain, p.dim-1,
                lambda x : np.logical_and(
                            np.logical_or(np.isclose(x[2], 0),
                                          np.isclose(x[2], 1)),
                            x[0] >= 0.5))
    return np.hstack((y_facets,z_facets))

def marker_gamma(x):
    return np.isclose(x[0], 0.5)

def get_matrix_row_as_func(p : Problem, p_ext:Problem, mat):
    ''' Debugging util '''
    if p.dim==2:
        middle_dof = fem.locate_dofs_geometrical(p.v, lambda x : np.logical_and(np.isclose(x[0], 0.5), np.isclose(x[1], 0.5)))
    else:
        middle_dof = fem.locate_dofs_geometrical(p.v, lambda x : np.logical_and(np.logical_and(np.isclose(x[0], 0.5), np.isclose(x[1], 0.5)), np.isclose(x[2], 0.5)))
    res_left = p.restriction
    res_right = p_ext.restriction
    middle_dof = middle_dof[:np.searchsorted(middle_dof, res_left.dofmap.index_map.size_local)]
    vals = np.zeros(res_right.dofmap.index_map.size_global)
    found = np.array([0], dtype=np.int32)
    found_mask = np.zeros(comm.size,dtype=np.int32)
    if middle_dof.size:
        row_middle_dof = res_left.unrestricted_to_restricted[middle_dof[0]]
        grow_middle_dof = res_left.index_map.local_to_global(np.array([row_middle_dof]))
        c, v = mat.getRow(grow_middle_dof[0])
        vals[c] = v
        found[0] = 1
    comm.Allgather(found, found_mask)
    middle_dof_rank = found_mask.nonzero()[0][0]
    comm.Bcast(vals, root=middle_dof_rank)
    middle_row_func = fem.Function(p_ext.v, name="middle_row_f")
    rgrange = np.arange(res_right.index_map.local_range[0],
                        res_right.index_map.local_range[1])
    rlrange = np.arange(rgrange.size)
    # middle_dof_rank is restricted, unrestrict
    for rgidx, rlidx in zip(rgrange, rlrange):
        ulidx = res_right.restricted_to_unrestricted[rlidx]
        middle_row_func.x.array[ulidx] = vals[rgidx]
    middle_row_func.x.scatter_forward()
    return middle_row_func

def run(dim, els_side, el_type, writepos=False):
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    rhs = Rhs(params["material"]["density"],
              params["material"]["specific_heat"],
              params["material"]["conductivity"],
              params["advection_speed"],
              dim)

    # Mesh and problems
    if dim==2:
        left_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.quadrilateral)
        right_mesh = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.triangle)
        exact_sol = exact_sol_2d
        grad_exact_sol = grad_exact_sol_2d
        left_marker_neumann = left_marker_neumann_2d
        right_marker_neumann = right_marker_neumann_2d
    else:
        cell_type = mesh.CellType.tetrahedron
        if el_type == "hexa":
            cell_type = mesh.CellType.hexahedron
        left_mesh  = mesh.create_unit_cube(MPI.COMM_WORLD, els_side, els_side, els_side, cell_type=cell_type)
        right_mesh  = mesh.create_unit_cube(MPI.COMM_WORLD, els_side, els_side, els_side, cell_type=cell_type)
        exact_sol = exact_sol_3d
        grad_exact_sol = grad_exact_sol_3d
        left_marker_neumann = left_marker_neumann_3d
        right_marker_neumann = right_marker_neumann_3d

    p_left = Problem(left_mesh, params, name=f"diff_mesh_robin_left")
    p_right = Problem(right_mesh, params, name=f"diff_mesh_robin_right")

    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= 0.5 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= 0.5 )

    # Set-up
    for p, marker in zip([p_left,p_right], [left_marker_dirichlet, right_marker_dirichlet]):
        p.set_activation(active_els[p])
        p.set_rhs(rhs)
        # Dirichlet
        u_ex = fem.Function(p.v,name="exact")
        u_ex.interpolate(exact_sol)
        bdofs_dir  = fem.locate_dofs_geometrical(p.v,marker)
        p.dirichlet_bcs = [fem.dirichletbc(u_ex, bdofs_dir)]

    # Set up interface Gamma
    for p, p_ext in zip([p_left, p_right], [p_right, p_left]):
        p.find_gamma(p_ext)

    quadrature_degree = 2
    driver = MonolithicRRDriver(p_left, p_right, quadrature_degree=quadrature_degree)

    neumann_facets = {}
    for p, marker in zip([p_left, p_right], [left_marker_neumann, right_marker_neumann]):
        p.set_forms_domain()
        # Set-up Neumann condition
        neumann_tag = 66
        neumann_facets[p] = marker(p)
        p.form_subdomain_data[fem.IntegralType.exterior_facet].append((neumann_tag, p.get_facet_integrations_entities(neumann_facets[p])))
        # Neumann condition
        ds = ufl.Measure('ds')
        n = ufl.FacetNormal(p.domain)
        v = ufl.TestFunction(p.v)
        p.l_ufl += +p.k * ufl.inner(n, grad_exact_sol(p.domain)) * v * ds(neumann_tag)

        p.compile_forms()
        # Pre-assemble
        p.pre_assemble()
        p.assemble_jacobian(finalize=False)
        p.assemble_residual()

    # Create nest system
    driver.setup_coupling()
    for p in [p_left, p_right]:
        p.A.assemble()
    driver.solve()

    middle_row_func_rl = get_matrix_row_as_func(p_right, p_left, driver.A21)
    middle_row_func_lr = get_matrix_row_as_func(p_left, p_right, driver.A12)

    driver.post_iterate()
    for p in [p_left, p_right]:
        p.post_iterate()

    if writepos:
        p_left.writepos(extra_funcs=[driver.ext_conductivity[p_left], p_left.dirichlet_bcs[0].g, middle_row_func_rl])
        p_right.writepos(extra_funcs=[p_right.dirichlet_bcs[0].g, middle_row_func_lr])

    # 16 elems per side
    points_2d = {
        p_left : np.array([[0.25, 0.25, 0.0],
                           [0.50, 0.50, 0.0],
                           [0.25, 1.00, 0.0]]),
        p_right : np.array([[0.75, 0.25, 0.0],
                            [0.50, 0.50, 0.0],
                            [0.75, 1.00, 0.0]]),
            }
    vals_2d = {
            p_left : np.array([1.86724959,
                               1.484520997,
                               0.9304105402]),
            p_right :  np.array([1.36746994,
                                 1.48441414,
                                 0.428867246]),
            }
    # 4 elems per side, tetra
    points_3d = {
        p_left : np.array([[0.25, 0.25, 0.25],
                           [0.50, 0.50, 0.50],
                           [0.25, 1.00, 1.00]]),
        p_right : np.array([[0.75, 0.25, 0.25],
                            [0.50, 0.50, 0.50],
                            [0.75, 1.00, 1.00]]),
            }
    vals_3d_tetra = {
            p_left : np.array([1.77741894,
                               1.1950978,
                               -0.0487715323]),
            p_right :  np.array([1.29849312,
                                 1.1950978,
                                 -0.589351976]),
            }
    vals_3d_hexa = {
            p_left : np.array([1.78125,
                               1.1875,
                               -0.09375]),
            p_right :  np.array([1.28125,
                                 1.1875,
                                 -0.59375]),
            }

    for p in [p_left, p_right]:
        if dim == 2:
            points   = points_2d[p]
            ref_vals = vals_2d[p]
        elif dim == 3:
            points   = points_3d[p]
            if el_type=="hexa":
                ref_vals = vals_3d_hexa[p]
            else:
                ref_vals = vals_3d_tetra[p]
        else:
            raise Exception
        assert_pointwise_vals(p, points, ref_vals)

def test_monolithic_RR_poisson_2d():
    run(dim=2, els_side=16, el_type="quadtri")

def test_monolithic_RR_poisson_3d_tetra():
    run(dim=3, els_side=4, el_type="tetra")

def test_monolithic_RR_poisson_3d_hexa():
    run(dim=3, els_side=4, el_type="hexa")

if __name__=="__main__":
    profiling = True
    writepos = True
    dim = 3
    els_side = 4
    el_type = "hexa"
    run_func = run
    if profiling:
        lp = LineProfiler()
        lp.add_module(Problem)
        lp_wrapper = lp(run)
        run_func = lp_wrapper
    run_func(dim=dim, els_side=els_side, el_type=el_type, writepos=writepos)
    if profiling:
        with open(f"profiling_diff_mesh_robin_rank{rank}.txt", 'w') as pf:
            lp.print_stats(stream=pf)
