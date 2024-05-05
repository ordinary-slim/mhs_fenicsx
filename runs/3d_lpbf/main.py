from meshing import get_mesh
import mhs_fenicsx
from mhs_fenicsx.problem import Problem
import yaml
from mhs_fenicsx.drivers import SingleProblemDriver
from mpi4py import MPI
from write_gcode import write_gcode
from line_profiler import LineProfiler
import argparse
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_dt(adim_dt, params):
    r = params["heat_source"]["radius"]
    v = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    return adim_dt * (r / v)

def main():
    domain = get_mesh(params)
    p = Problem(domain, params, name=params["case_name"])
    driver = SingleProblemDriver(p,params)
    while (driver.p.time < driver.p.source.path.times[-1] - 1e-7) and (driver.p.iter<max_iter):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate()
        if driver.p.iter%post_frequency==0:
            p.writepos()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', default='input.yaml')
    args = parser.parse_args()
    with open(args.input_file, 'r') as f:
        params = yaml.safe_load(f)
    write_gcode(params)
    max_iter = params["max_iter"]
    post_frequency = int(params["post_frequency"])

    if "adim_dt" in params:
        params["dt"] = get_dt(params["adim_dt"],params)

    profiling = False
    if profiling:
        lp = LineProfiler()
        lp.add_module(SingleProblemDriver)
        lp.add_module(mhs_fenicsx.problem)
        lp.add_module(mhs_fenicsx.geometry)
        lp_wrapper = lp(main)
        lp_wrapper()
        if rank==0:
            with open("profiling.txt", 'w') as pf:
                lp.print_stats(stream=pf)
    else:
        main()
