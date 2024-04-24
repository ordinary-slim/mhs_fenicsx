from meshing import get_mesh
import mhs_fenicsx
from mhs_fenicsx.problem import Problem
import yaml
from mhs_fenicsx.drivers import SingleProblemDriver
from mpi4py import MPI
from write_gcode import write_gcode
from line_profiler import LineProfiler
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)
max_iter = params["max_iter"]

def main():
    domain = get_mesh()
    p = Problem(domain, params, name="case")
    driver = SingleProblemDriver(p,params)
    while (driver.p.time < driver.p.source.path.times[-1] - 1e-7) and (driver.p.iter<max_iter):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate()
        p.writepos()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', default=-1, type=int)
    parser.add_argument('--case-name', default='case')
    args = parser.parse_args()
    write_gcode( nLayers=args.layers )
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(SingleProblemDriver)
        lp.add_module(mhs_fenicsx.problem)
        lp.add_module(mhs_fenicsx.geometry)
        lp_wrapper = lp(main)
        lp_wrapper()
        with open("profiling.txt", 'w') as pf:
            lp.print_stats(stream=pf)
    else:
        main()
