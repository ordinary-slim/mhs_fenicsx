import yaml
from mhs_fenicsx.problem import Problem, L2Differ
from dolfinx import mesh, fem, io
from mpi4py import MPI
import numpy as np
from scipy.optimize import fsolve
from scipy.special import erf, erfc
from line_profiler import LineProfiler
import argparse
from mhs_fenicsx.problem.helpers import assert_pointwise_vals, print_vals
import csv
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main(params, case_name="demo_phase_change", writepos=True,
         write_csvs=False,
         write_l2errs=False,):
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
    x_left  = params["x_left"]
    x_right = params["x_right"]
    domain  = mesh.create_interval(MPI.COMM_WORLD,
                                   nelems,
                                   [x_left,x_right],)
    p = Problem(domain, params, name=case_name)
    p.set_initial_condition(p.T_env)
    f_exact = fem.Function(p.v, name="exact_sol")
    null_func = fem.Function(p.v, name="null_func")
    null_func.x.array.fill(0.0)
    model_name = p.latent_heat_treatment
    S = params["smoothing_cte_phase_change"]
    output_dir = f"./{model_name}_s{S}_nelems{nelems}_dt{params['dt']}.bp"
    writer = io.VTXWriter(p.domain.comm, output_dir, [p.u, f_exact])
    p.set_forms()
    # Dirichlet BC
    bfacets = mesh.locate_entities_boundary(domain, p.dim-1, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(f_exact, fem.locate_dofs_topological(p.v, p.dim-1, bfacets))
    p.dirichlet_bcs.append(bc)

    p.compile_create_forms()
    p.pre_assemble()

    l2differ = L2Differ(p)

    exact_sol_at_t = lambda x : exact_sol(x,p.time)
    l2errs = []
    for _ in range(max_num_timesteps):
        p.pre_iterate()
        f_exact.interpolate(exact_sol_at_t)

        p.non_linear_solve()
        if writepos:
            writer.write(p.time)
        if write_csvs and np.isclose(p.time, [1.0, 10, 30, 60, 120]).any():
            file_name = f"./{model_name}_csvs_s{S}/{np.round(p.time, 1)}.csv"
            dir_name = os.path.dirname(file_name)
            if rank==0 and not os.path.exists(dir_name):
                os.mkdir(dir_name)
            with open(file_name, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['x', 'u', 'exact'])
                for x, u, exact in zip(p.domain.geometry.x[:, 0],
                                       p.u.x.array,
                                       f_exact.x.array):
                    csv_writer.writerow([x, u, exact])
        # Compute L2 error
        l2err = l2differ(p.u, f_exact)
        l2normex = l2differ(f_exact, null_func)
        relerr = l2err / l2normex if l2normex > 0 else 0.0
        l2errs.append((p.time, relerr))
        if rank==0:
            print(f"L2 error at t = {p.time} = {l2err:.6e} = {relerr*100}%", flush=True)
    writer.close()
    if write_l2errs:
        with open(f"./{model_name}_l2errs_s{S}.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['time', 'l2err'])
            for t, l2err in l2errs:
                csv_writer.writerow([t, l2err])
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
        1894.398592,
        1748.72238,
        1563.268393
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
        lp_wrapper(params=params, write_csvs=True, write_l2errs=True)
        profiling_file = f"profiling_rank{rank}.txt"
        with open(profiling_file, 'w') as pf:
            lp.print_stats(stream=pf)
    if args.run_test:
        test_1d_phase_change()
