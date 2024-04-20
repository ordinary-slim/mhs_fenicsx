from mpi4py import MPI
import numpy as np
from dolfinx import io, mesh, geometry, cpp, fem
from petsc4py import PETSc

def mark_cells(msh, cell_index):
    num_cells = msh.topology.index_map(
        msh.topology.dim).size_local + msh.topology.index_map(
        msh.topology.dim).num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    values = np.full(cells.shape, 0, dtype=np.int32)
    values[cell_index] = np.full(len(cell_index), 1, dtype=np.int32)
    cell_tag = mesh.meshtags(msh, msh.topology.dim, cells, values)
    return cell_tag
def indices_to_func(mt,space,name="collision"):
    domain = space.mesh
    dofs = fem.locate_dofs_topological(space, domain.topology.dim, mt.find(1))
    f = fem.Function(space,name=name)
    f.x.array[dofs] = 1
    return f
def create_partition_tag(domain: mesh.Mesh) -> mesh.MeshTags:
    """
    Create a cell marker with all cells owned by the process tagged with the process rank
    """
    tdim = domain.topology.dim
    num_cells_local = domain.topology.index_map(tdim).size_local
    cells = np.arange(num_cells_local, dtype=np.int32)
    values = np.full(num_cells_local, domain.comm.rank, dtype=np.int32)
    return mesh.meshtags(domain, tdim, cells, values)
def extract_cell_geometry(input_mesh, cell: int):
    mesh_nodes = cpp.mesh.entities_to_geometry(
        input_mesh._cpp_object, input_mesh.topology.dim, np.array([cell], dtype=np.int32), False)[0]

    return input_mesh.geometry.x[mesh_nodes]


def mesh_collision(mesh_big,mesh_small):
    '''
    mesh_small is in MPI.COMM_SELF
    mesh_big is in MPI.COMM_WORLD
    '''
    num_big_cells = mesh_big.topology.index_map(mesh_big.topology.dim).size_local + \
        mesh_big.topology.index_map(mesh_big.topology.dim).num_ghosts

    bb_tree = geometry.bb_tree(
        mesh_big, mesh_big.topology.dim, np.arange(num_big_cells, dtype=np.int32))

    # For each local mesh, compute the bounding box, compute colliding cells
    tol = 1e-13
    big_cells = []
    local_cells_set = set()
    o_cell_idx = mesh_small.topology.original_cell_index
    bb_small = geometry.bb_tree(
        mesh_small, mesh_small.topology.dim)
    cell_cell_collisions = geometry.compute_collisions_trees(
        bb_small, bb_tree)
    for local_cell, big_cell in cell_cell_collisions:

        geom_small = extract_cell_geometry(mesh_small, local_cell)
        geom_big = extract_cell_geometry(mesh_big, big_cell)
        distance = geometry.compute_distance_gjk(geom_big, geom_small)
        if np.linalg.norm(distance) <= tol:
            big_cells.append(big_cell)
            local_cells_set = local_cells_set.union([o_cell_idx[local_cell]])

    sorted_unique_big_cells = np.unique(big_cells).astype(dtype=np.int32)
    colliding_big_marker = mark_cells(mesh_big, sorted_unique_big_cells)
    colliding_big_marker.name = "colliding cells"

    return colliding_big_marker

class Hatch:
    def __init__(self,x0,x1,width,height,depth):
        self.x0 = x0
        self.x1 = x1
        self.width = width
        self.height = height
        self.depth = depth

    def get_mesh(self):
        '''
        Return an oriented hexahedron/rectangle from hatch geometry
        '''
        pass

class OBB:
    def __init__(self,p0:np.ndarray,p1:np.ndarray,
                 width=1.0,height=1.0,depth=1.0,dim=3):
        '''
        Note: For both dim 2 and 3, vectors have 3 components
        i.e. 2D is nested in 3D
        '''
        self.dim = dim
        step = p1-p0
        step_len = np.linalg.norm(step)
        self.pos = (p0+p1)/2
        self.x_axis = step / step_len
        self.set_transverse_axes(dim)
        self.pos += ((height-depth)/2)*self.z_axis
        self.half_widths = np.array([step_len/2.0,
                                     width/2.0,
                                     (height+depth)/2.0,])

    def set_transverse_axes(self,dim):
        '''
        Set y and z axes depending on dimension
        '''
        if (dim==2):
            self.z_axis = np.array([0.0,1.0,0.0])
        else:
            self.z_axis = np.array([0.0,0.0,1.0])
        self.z_axis -= self.z_axis.dot(self.x_axis)*self.x_axis
        self.z_axis /= np.linalg.norm(self.z_axis)
        assert np.linalg.norm(self.z_axis)>0, "Steps along z-axis are not allowed."
        self.y_axis = np.cross(self.z_axis,self.x_axis)

    def get_dolfinx_mesh(self):
        points = np.empty((2**self.dim,self.dim))
        point_counter = 0
        if self.dim==3:
            for y_sign in [-1.0,+1.0]:
                for x_sign in [-1.0,+1.0]:
                    for z_sign in [-1.0,+1.0]:
                        points[point_counter,:] = \
                                self.pos +  \
                                x_sign*self.half_widths[0]*self.x_axis + \
                                y_sign*self.half_widths[1]*self.y_axis + \
                                z_sign*self.half_widths[2]*self.z_axis
                        point_counter+=1
            cells = [(0,1,2,3,4,5,6,7)]
            ufl_mesh = ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(self.dim,), dtype=PETSc.RealType))
        else:
            for x_sign in [-1.0,+1.0]:
                for z_sign in [-1.0,+1.0]:
                    points[point_counter,:] = \
                            self.pos[:2] +  \
                            x_sign*self.half_widths[0]*self.x_axis[:2] + \
                            z_sign*self.half_widths[2]*self.z_axis[:2]
                    point_counter+=1
            cells = [(0,1,2,3)]
            ufl_mesh = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(self.dim,), dtype=PETSc.RealType))
        return mesh.create_mesh(MPI.COMM_SELF,cells,points,ufl_mesh)
