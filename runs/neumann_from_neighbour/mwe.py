from dolfinx import fem, mesh, io
from mpi4py import MPI
import numpy as np
import ufl
import basix.ufl

def mesh_rectangle(min_x, min_y, max_x, max_y, elsize, cell_type=mesh.CellType.triangle):
    nx = np.ceil((max_x-min_x)/elsize).astype(np.int32)
    ny = np.ceil((max_y-min_y)/elsize).astype(np.int32)
    return mesh.create_rectangle(MPI.COMM_WORLD,
                                 [np.array([min_x,min_y]),np.array([max_x,max_y])],
                                 [nx,ny],
                                 cell_type,)

class Problem:
    def __init__(self, domain):
        self.domain = domain
        self.dim    = domain.topology.dim
        self.V    = fem.functionspace(self.domain,("CG",1))
        self.uh   = fem.Function(self.V, name="uh")

el_size = 0.25
mesh_left  = mesh_rectangle(0, 0, 0.5, 1, elsize=el_size)
mesh_right = mesh_rectangle(0.5, 0, 1, 1, elsize=el_size, cell_type=mesh.CellType.quadrilateral)

p_left  = Problem(mesh_left)
p_right = Problem(mesh_right)

x = ufl.SpatialCoordinate(mesh_right)
p_right.uh.interpolate(fem.Expression(x[0]**2+x[1]**2,
                                     p_right.uh.function_space.element.interpolation_points()))
dg0_right     = fem.functionspace(mesh_right,("DG",0))
func2interpolate = fem.Function(dg0_right)
func2interpolate.interpolate(fem.Expression(ufl.grad(p_right.uh)[0],
                             func2interpolate.function_space.element.interpolation_points()))

rfacets = mesh.locate_entities_boundary(p_left.domain,p_left.dim-1,lambda x:np.isclose(x[0],0.5))
# I want a function that:
# 1. Has dofs at quadrature points of rfacets on mesh_left
# 2. I can put in a ufl form with other functions on mesh_left

#Qe = basix.ufl.quadrature_element(mesh_left.topology.entity_types[-2][0].name,
                                  #degree=1,)
#Qs = fem.functionspace(mesh_left,Qe)
