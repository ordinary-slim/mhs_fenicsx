from meshing import get_mesh
import yaml
from mpi4py import MPI
from write_gcode import write_gcode
from line_profiler import LineProfiler
import mhs_fenicsx.problem
import mhs_fenicsx.drivers
import mhs_fenicsx.geometry
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main():
    write_gcode(params["path"])
    domain = get_mesh()
    p = mhs_fenicsx.problem.Problem(domain, params, name=params["case_name"])
    driver = mhs_fenicsx.drivers.SingleProblemDriver(p,params)
    while (driver.p.time < driver.p.source.path.times[-1] - 1e-7):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate()
        p.writepos()
        p.writepos_vtx()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', default='input.yaml')
    args = parser.parse_args()
    with open(args.input_file, 'r') as f:
        params = yaml.safe_load(f)
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(mhs_fenicsx.drivers)
        lp.add_module(mhs_fenicsx.problem)
        lp.add_module(mhs_fenicsx.geometry)
        lp_wrapper = lp(main)
        lp_wrapper()
        with open("profiling.txt", 'w') as pf:
            lp.print_stats(stream=pf)

    else:
        main()
