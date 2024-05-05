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
radiusHs = params["heat_source"]["radius"]
speedHs = np.linalg.norm(params["heat_source"]["initial_speed"])
adimR = radiusHs / speedHs
box = [-7, -2, 7, 2]
gcodeFile = params["path"]
fineElSize = radiusHs/2/params["elSizeFactor"]

def write_gcode():
    gcodeLines = []
    x0 = -5
    x1 = +5
    y = 0
    gcodeLines.append(
            "G0 F{} X{} Y{} Z0".format( speedHs, x0, y ))
    gcodeLines.append(
            "G1 X{} E0.1".format( x1 ))
    with open(params["path"], 'w') as gf:
        gf.writelines([l+"\n" for l in gcodeLines])

def run(case_name="case"):
    # Mesh and problems
    point_density = np.round(1/fineElSize).astype(np.int32)
    nx = np.round((box[2]-box[0]) * point_density).astype(np.int32)
    ny = np.round((box[3]-box[1]) * point_density).astype(np.int32)
    domain  = mesh.create_rectangle(MPI.COMM_WORLD,
              [box[:2], box[2:]],
              [nx, ny],
              mesh.CellType.quadrilateral,
              )

    p = Problem(domain, params, name=case_name)
    driver = SingleProblemDriver(p,params)
    while (driver.p.time < driver.p.source.path.times[-1] - 1e-7):
        driver.pre_iterate()
        driver.iterate()
        driver.post_iterate()
        p.writepos()

if __name__=="__main__":
    write_gcode()
    run()
