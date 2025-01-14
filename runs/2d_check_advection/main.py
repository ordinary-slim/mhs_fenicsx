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
    point_density = 2 / params["heat_source"]["radius"]
    bounds = [-7.0,-2.0,7.0,2.0]
    nx = np.round((bounds[2]-bounds[0]) * point_density ).astype(np.int32)
    ny = np.round((bounds[3]-bounds[1]) * point_density ).astype(np.int32)
    domain = mesh.create_rectangle(MPI.COMM_WORLD,
                                   [bounds[:2],bounds[2:]],
                                   [nx,ny],
                                   mesh.CellType.quadrilateral,
                                   )

    p = Problem(domain, params, name=case_name)

    p.set_initial_condition(params["environment_temperature"])
    p.set_forms_domain()
    p.compile_create_forms()
    for _ in range(20):
        p.pre_iterate()
        p.assemble()
        p.solve()
        p.post_iterate()
        p.writepos()

if __name__=="__main__":
    run()
