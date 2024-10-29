import yaml
from mhs_fenicsx.problem import Problem
from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
from scipy.optimize import fsolve
from scipy.special import erf, erfc
from line_profiler import LineProfiler
from mhs_fenicsx.drivers import NewtonRaphson

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def transcendental_fun(x):
    return np.sqrt(np.pi)*x - \
        (stl / np.exp(np.pow(x,2)) / erf(x) ) + \
        (sts / np.exp(np.pow(x,2)) / erfc(x) )

def locate_interface(t):
    return 2*lamma*np.sqrt(diffusivity*t);

def exact_sol(x,t):
    interface_pos = locate_interface(t);
    sol = np.zeros(x.shape[1],dtype=x.dtype)
    liquid_indices = x[0]<=interface_pos
    solid_indices  = x[0]>interface_pos
    sol[liquid_indices] = T_l - (T_l - T_m)*erf(x[0,liquid_indices]/(2*np.sqrt(diffusivity*t)))/erf(lamma)
    sol[solid_indices] = T_s + (T_m - T_s)*erfc(x[0,solid_indices]/(2*np.sqrt(diffusivity*t)))/erfc(lamma)
    return sol

mat_params = params["material"]
phase_change_params = mat_params["phase_change"]
c_p = mat_params["specific_heat"]
rho = mat_params["density"]
k = mat_params["conductivity"]
T_l = phase_change_params["liquidus_temperature"]
T_s = phase_change_params["solidus_temperature"]
l = phase_change_params["latent_heat"]
T_m = (T_l + T_s)/2.0
stl = c_p*(T_l - T_m) / l
sts = c_p*(T_m - T_s) / l
diffusivity = k / rho / c_p
lamma = fsolve(transcendental_fun, 0.388150542167233)

def main(case_name="demo_phase_change"):
    nelems = 1000
    max_num_timesteps = 150
    max_nr_iters = 25
    max_ls_iters = 5
    x_left  = 0.0
    x_right = 0.1
    domain  = mesh.create_interval(MPI.COMM_WORLD,
                                   nelems,
                                   [x_left,x_right],)
    p = Problem(domain, params, name=case_name)
    p.set_initial_condition(p.T_env)
    f_exact = fem.Function(p.v, name="exact_sol")
    p.set_forms_domain()
    # Dirichlet BC
    bfacets = mesh.locate_entities_boundary(domain, p.dim-1, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(f_exact, fem.locate_dofs_topological(p.v, p.dim-1, bfacets))
    p.dirichlet_bcs.append(bc)

    p.compile_forms()
    p.pre_assemble()

    lamma_exact_sol = lambda x : exact_sol(x,p.time)
    for _ in range(max_num_timesteps):
        p.pre_iterate()
        f_exact.interpolate(lamma_exact_sol)

        if p.phase_change:
            nr_driver = NewtonRaphson(p,max_ls_iters=max_ls_iters)
            nr_driver.solve()
        else:
            p.assemble()
            p.solve()

        #p.post_iterate()
        p.writepos(extra_funcs=[f_exact])

if __name__=="__main__":
    lp = LineProfiler()
    lp.add_function(Problem.assemble_jacobian)
    lp.add_function(Problem.assemble_residual)
    import multiphenicsx.fem.petsc
    lp.add_function(multiphenicsx.fem.petsc.apply_lifting)
    lp_wrapper = lp(main)
    lp_wrapper()
    profiling_file = f"profiling_rank{rank}.txt"
    try:
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
    except NameError:
        pass
