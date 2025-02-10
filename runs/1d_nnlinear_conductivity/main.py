from dolfinx import mesh, fem
from mhs_fenicsx.problem import Problem, set_same_mesh_interface
from mpi4py import MPI
import yaml
import numpy as np
from mhs_fenicsx.drivers import StaggeredRRDriver, MonolithicRRDriver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ul, ur = 2.0, 7.0
L = 2.0
beta = 1.0
u_exact = lambda x : ((1 + beta * ul) * \
                      np.exp( x[0]/L * \
                      np.log((1 + beta*ur)/(1 + beta*ul))) -1.0) \
                      / beta

def get_mesh():
    nelems = 20
    return mesh.create_interval(comm,
                                nelems,
                                [0.0, L])

def ref(params):
    domain = get_mesh()
    c2 = ul*(ul+2)
    c1 = (ur*(ur+2) - ul*(ul+2)) / 2.0
    u_exact = lambda x : -1 + np.sqrt(1 + c2 + c1*x[0])
    p = Problem(domain, params, "nnlinearconduc")
    marker  = lambda x : np.logical_or(np.isclose(x[0],0),
                                       np.isclose(x[0],L))
    u_d = fem.Function(p.v,name="exact")
    u_d.interpolate(u_exact)
    bdofs_dir  = fem.locate_dofs_geometrical(p.v,marker)
    p.dirichlet_bcs = [fem.dirichletbc(u_d, bdofs_dir)]

    # Solve
    p.set_forms()
    p.compile_create_forms()
    p.pre_assemble()
    p.non_linear_solve()
    p.post_iterate()
    p.writepos(extra_funcs=[u_d])
    return p


def set_bc(pright,pleft):
    # Set outside Dirichlet
    pright.add_dirichlet_bc(u_exact, marker=(lambda x : np.isclose(x[0],L)), reset=True)
    pleft.add_dirichlet_bc(u_exact, marker=(lambda x : np.isclose(x[0],0)), reset=True)

def staggered_robin(params):
    domain = get_mesh()
    p_left = Problem(domain, params, "nnlinearconduc_Left_stag_robin")
    p_right = p_left.copy(name="nnlinearconduc_Right_stag_robin")
    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= L/2.0 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= L/2.0 )
    for p in [p_left,p_right]:
        p.set_activation(active_els[p])
    set_same_mesh_interface(p_left, p_right)
    driver = StaggeredRRDriver(p_right, p_left, max_staggered_iters=20,
                               initial_relaxation_factors=[1.0, 1.0])

    driver.pre_loop(set_bc=set_bc)
    for _ in range(driver.max_staggered_iters):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate(verbose=True)
        driver.writepos()
        if driver.convergence_crit < driver.convergence_threshold:
            break
    driver.post_loop()
    return driver

def monolithic_robin(params):
    domain = get_mesh()
    p_left = Problem(domain, params, "nnlinearconduc_Left_mono_robin")
    p_right = p_left.copy(name="nnlinearconduc_Right_mono_robin")
    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= L/2.0 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= L/2.0 )
    for p in [p_left,p_right]:
        p.set_activation(active_els[p])
    set_same_mesh_interface(p_left, p_right)
    set_bc(p_right, p_left)
    driver = MonolithicRRDriver(p_left, p_right, quadrature_degree=2)
    for p in [p_left, p_right]:
        p.set_forms()
        p.compile_create_forms()
        p.pre_assemble()
    driver.non_linear_solve()
    driver.post_iterate()
    for p in [p_left, p_right]:
        p.post_iterate()
        p.writepos()

if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    #ref(params)
    #staggered_robin(params)
    monolithic_robin(params)
