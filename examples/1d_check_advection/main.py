from basix.ufl import element
from mhs_fenicsx.problem import Problem
import yaml
from mpi4py import MPI
from dolfinx import mesh
import numpy as np

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

def run(case_name="case"):
    # Mesh and problems
    point_density = params["point_density"]
    bounds = [-3.0,3.0]
    nx = np.round((bounds[1]-bounds[0]) * float(params["point_density"]) ).astype(np.int32)
    domain  = mesh.create_interval(MPI.COMM_WORLD,
                                   nx,
                                   [bounds[0],bounds[1]],)

    p = Problem(domain, params, name=case_name)

    p.set_initial_condition(params["environment_temperature"])
    p.set_forms()
    p.compile_create_forms()
    for _ in range(20):
        p.pre_iterate()
        p.assemble()
        p.solve()
        p.post_iterate()
        p.writepos()

if __name__=="__main__":
    run()
