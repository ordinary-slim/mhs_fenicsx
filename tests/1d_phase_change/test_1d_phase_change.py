import yaml
from mhs_fenicsx.problem import Problem
from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
from scipy.optimize import fsolve
from scipy.special import erf, erfc
from line_profiler import LineProfiler
import argparse
from mhs_fenicsx.problem.helpers import assert_pointwise_vals, print_vals

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main(params, case_name="demo_phase_change", writepos=True):
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

    def transcendental_fun(x):
        return np.sqrt(np.pi)*x - \
            (stl / np.exp(np.pow(x,2)) / erf(x) ) + \
            (sts / np.exp(np.pow(x,2)) / erfc(x) )
    diffusivity = k / rho / c_p
    lamma = fsolve(transcendental_fun, 0.388150542167233)

    def locate_interface(t):
        return 2*lamma*np.sqrt(diffusivity*t);

    def exact_sol(x,t):
        interface_pos = locate_interface(t);
        if rank==0:
            print(f"Interface position at t {t} s = {interface_pos[0]}", flush=True)
        sol = np.zeros(x.shape[1],dtype=x.dtype)
        liquid_indices = x[0]<=interface_pos
        solid_indices  = x[0]>interface_pos
        sol[liquid_indices] = T_l - (T_l - T_m)*erf(x[0,liquid_indices]/(2*np.sqrt(diffusivity*t)))/erf(lamma)
        sol[solid_indices] = T_s + (T_m - T_s)*erfc(x[0,solid_indices]/(2*np.sqrt(diffusivity*t)))/erfc(lamma)
        return sol

    nelems = params["nelems"]
    max_num_timesteps = params["max_num_timesteps"]
    max_nr_iters = 25
    max_ls_iters = 5
    x_left  = params["x_left"]
    x_right = params["x_right"]
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

    p.compile_create_forms()
    p.pre_assemble()

    exact_sol_at_t = lambda x : exact_sol(x,p.time)
    for _ in range(max_num_timesteps):
        p.pre_iterate()
        f_exact.interpolate(exact_sol_at_t)

        p.non_linear_solve()
        if writepos:
            p.writepos(extra_funcs=[f_exact])
    return p

def test_1d_phase_change():
    with open("test_input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    p = main(params=params, writepos=False)
    points = np.array([
        [+0.0010, 0.0, 0.0],
        [+0.0025, 0.0, 0.0],
        [+0.0080, 0.0, 0.0],
        ])
    vals = np.array([
        1895.205212,
        1748.631706,
        1561.736834
        ])
    assert_pointwise_vals(p,points,vals)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--run-test',action='store_true')
    parser.add_argument('-p','--run-profiling',action='store_true')
    args = parser.parse_args()
    if args.run_profiling:
        with open("input.yaml", 'r') as f:
            params = yaml.safe_load(f)
        lp = LineProfiler()
        lp.add_function(Problem.assemble_jacobian)
        lp.add_function(Problem.assemble_residual)
        import multiphenicsx.fem.petsc
        lp.add_function(multiphenicsx.fem.petsc.apply_lifting)
        lp_wrapper = lp(main)
        lp_wrapper(params=params)
        profiling_file = f"profiling_rank{rank}.txt"
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
    if args.run_test:
        test_1d_phase_change()
