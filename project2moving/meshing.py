import numpy as np
from mpi4py import MPI
from dolfinx import mesh

def meshBox( box ):
    L = box[2] - box[0]
    H = box[3] - box[1]
    meshDen = 4
    nx = int(np.round(L*meshDen*10)/10)
    ny = int(np.round(H*meshDen*10)/10)
    return mesh.create_rectangle(MPI.COMM_WORLD,
         [np.array(box[:2]), np.array(box[2:])],
         [nx, ny],
         mesh.CellType.quadrilateral,
         )

