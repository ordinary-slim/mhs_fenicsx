from dolfinx import mesh, fem
from mhs_fenicsx.problem import Problem, set_same_mesh_interface
from mpi4py import MPI
import yaml
import numpy as np
from mhs_fenicsx.drivers import StaggeredRRDriver, MonolithicRRDriver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

u_hot = 7.0
u_cold = 2.0
kl = 1.0
kr = 10.0
c1 = (u_hot - u_cold) / (0.5 + (kl / kr)*0.5)
c2 = kl / kr * c1
d2 = u_cold + 0.5 * (c1 - c2)

def u_exact(x):
    left = np.array(x[0] <= 0.5, dtype=np.float64)*(u_cold + c1 * x[0])
    right = np.array(x[0] > 0.5, dtype=np.float64)*(d2 + c2*x[0])
    return (left + right)

def get_mesh():
    nelems_side = 20
    return mesh.create_unit_square(comm,
                                   nelems_side, nelems_side,
                                   )

def ref(params):
    domain = get_mesh()
    p = Problem(domain, params, "2mat_difussing")
    marker  = lambda x : np.logical_or(np.isclose(x[0],0.0),
                                       np.isclose(x[0],1.0))
    u_d = fem.Function(p.v,name="exact")
    u_d.interpolate(u_exact)
    bdofs_dir  = fem.locate_dofs_geometrical(p.v,marker)
    p.dirichlet_bcs = [fem.dirichletbc(u_d, bdofs_dir)]

    # Set materials
    right_els = fem.locate_dofs_geometrical(p.dg0, lambda x : x[0] >= 0.5 )
    p.update_material_at_cells(right_els, p.materials[1])

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
    pleft.add_dirichlet_bc(u_exact, marker=(lambda x : np.isclose(x[0],0)), reset=True)
    pright.add_dirichlet_bc(u_exact, marker=(lambda x : np.isclose(x[0],1.0)), reset=True)
    for p in (pright, pleft):
        p.dirichlet_bcs[0].g.name = "exact"

def staggered_robin(params):
    domain = get_mesh()
    p_left = Problem(domain, params, "2mat_diffusion_Left_stag_robin")
    right_els = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] >= 0.5 )
    p_left.update_material_at_cells(right_els, p_left.materials[1])
    p_right = p_left.copy(name="2mat_diffusion_Right_stag_robin")
    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= 0.5)
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= 0.5)
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
        driver.writepos(extra_funcs_p1=[driver.p1.dirichlet_bcs[0].g],
                        extra_funcs_p2=[driver.p2.dirichlet_bcs[0].g])
        if driver.convergence_crit < driver.convergence_threshold:
            break
    driver.post_loop()
    return p_left, p_right

def monolithic_robin(params):
    domain = get_mesh()
    p_left = Problem(domain, params, "2mat_diffusion_Left_mono_robin")
    right_els = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] >= 0.5 )
    p_left.update_material_at_cells(right_els, p_left.materials[1])
    p_right = p_left.copy(name="2mat_diffusion_Right_mono_robin")
    # Activation
    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.dg0, lambda x : x[0] <= 0.5 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.dg0, lambda x : x[0] >= 0.5 )
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
        p.writepos(extra_funcs=[p.dirichlet_bcs[0].g])
    return p_left, p_right

if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    #ref(params)
    staggered_robin(params)
    #monolithic_robin(params)
