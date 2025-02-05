import argparse
import numpy as np
from mhs_fenicsx.problem import Problem, indices_to_function
from mhs_fenicsx.geometry import mesh_collision, Hatch, OBB
from mhs_fenicsx.gcode import Path, gcode_to_path
from dolfinx import fem, mesh, default_scalar_type, io
import numpy as np
from mpi4py import MPI
import yaml
from mhs_fenicsx.drivers import SingleProblemDriver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

inputFile = "input.yaml"
tol = 1e-7
box = [-10, -5, 10, 5]

def run(case_name="case"):
    # Mesh and problems
    point_density = params["point_density"]
    nx = np.round((box[2]-box[0]) * point_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * point_density).astype(np.int32)
    domain  = mesh.create_rectangle(MPI.COMM_WORLD,
              [box[:2], box[2:]],
              [nx, ny],
              mesh.CellType.quadrilateral,
              )

    p = Problem(domain, params, name=case_name)
    active_els = fem.locate_dofs_geometrical(p.dg0, lambda x : x[0] <= 0.0 )
    p.set_activation( active_els )

    p.set_initial_condition(10.0)
    p.set_forms()
    p.compile_create_forms()
    for _ in range(10):
        p.pre_iterate()
        p.pre_assemble()
        p.assemble()
        p.solve()
        p.post_iterate()
        p.writepos()

if __name__=="__main__":
    run()
