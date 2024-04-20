from meshing import get_mesh
from mhs_fenicsx.problem import Problem
import yaml
from mhs_fenicsx.drivers import SingleProblemDriver
from mpi4py import MPI
from write_gcode import write_gcode

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

def main():
    write_gcode(params["path"])
    domain = get_mesh()
    p = Problem(domain, params, name="case")
    driver = SingleProblemDriver(p,params)
    while (driver.p.time < driver.p.source.path.times[-1] - 1e-7):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate()
        p.writepos()

if __name__=="__main__":
    main()
