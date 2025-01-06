from mpi4py import MPI
import numpy as np
from dolfinx import io, mesh, geometry, cpp, fem
from petsc4py import PETSc
import ufl
import basix

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


def mesh_collision(mesh_big,mesh_small,bb_tree_mesh_big=None):
    '''
    mesh_small is in MPI.COMM_SELF
    mesh_big is in MPI.COMM_WORLD
    '''
    if bb_tree_mesh_big is None:
        num_big_cells = mesh_big.topology.index_map(mesh_big.topology.dim).size_local + \
            mesh_big.topology.index_map(mesh_big.topology.dim).num_ghosts
        bb_tree = geometry.bb_tree( mesh_big, mesh_big.topology.dim, np.arange(num_big_cells, dtype=np.int32))
    else:
        bb_tree = bb_tree_mesh_big

    # For each local mesh, compute the bounding box, compute colliding cells
    tol = 1e-7
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
    return sorted_unique_big_cells

def mesh_containment(nodal_bb_tree,loc_mesh,cell_bb_tree,ext_mesh):
    big_process = nodal_bb_tree.create_global_tree(loc_mesh.comm)
    process_collisions = geometry.compute_collisions_trees(cell_bb_tree, big_process)
    outgoing_edges = set()
    num_outgoing_cells = np.zeros(loc_mesh.comm.size, dtype=np.int32)

    for cell_idx, process_idx in process_collisions:
        num_outgoing_cells[process_idx] += 1
        outgoing_edges = set.union(outgoing_edges, (process_idx,))
    outgoing_edges = np.asarray(np.unique(list(outgoing_edges)), dtype=np.int32)
    small_to_big_comm = ext_mesh.comm.Create_dist_graph(
        list([ext_mesh.comm.rank]), [len(np.unique(outgoing_edges))], outgoing_edges, reorder=False)

    source, dest, _ = small_to_big_comm.Get_dist_neighbors()

    num_vertices_per_cell_small = cpp.mesh.cell_num_vertices(
        ext_mesh.topology.cell_type)

    # Extract all mesh nodes per process
    process_offsets = np.zeros(len(dest)+1, dtype=np.int32)
    np.cumsum(num_outgoing_cells[dest], out=process_offsets[1:])
    sending_cells = np.full(process_offsets[-1], 10, dtype=np.int32)
    insert_counter = np.zeros_like(dest, dtype=np.int32)
    for cell_idx, process_idx in process_collisions:
        local_idx = np.flatnonzero(dest == process_idx)
        assert len(local_idx) == 1
        idx = local_idx[0]
        sending_cells[process_offsets[idx]+insert_counter[idx]] = cell_idx
        insert_counter[idx] += 1


    node_counter = np.zeros(ext_mesh.geometry.index_map().size_local +
                            ext_mesh.geometry.index_map().num_ghosts+1, dtype=np.int32)
    local_pos = np.zeros_like(node_counter, dtype=np.int32)
    send_geom = []
    send_top = []
    send_top_size = np.zeros_like(dest, dtype=np.int32)
    send_geom_size = np.zeros_like(dest, dtype=np.int32)

    for i in range(len(dest)):
        # Get nodes of all cells sent to a given process
        org_nodes = cpp.mesh.entities_to_geometry(
            ext_mesh._cpp_object, ext_mesh.topology.dim, sending_cells[process_offsets[i]:process_offsets[i+1]], False).reshape(-1)

        # Get the unique set of nodes sent to this process
        unique_nodes = np.unique(org_nodes)

        # Compute remapping of nodes
        node_counter[:] = 0
        node_counter[unique_nodes] = 1
        np.cumsum(node_counter, out=local_pos)
        local_pos -= 1  # Map to 0 index system
        send_geom.append(
            ext_mesh.geometry.x[unique_nodes][:, :ext_mesh.geometry.dim])
        send_geom_size[i] = np.size(send_geom[-1])
        send_top.append(local_pos[org_nodes])
        send_top_size[i] = np.size(send_top[-1])

    # Compute send and receive offsets for geometry and topology
    geom_offset = np.zeros(len(dest)+1, dtype=np.int32)
    top_offset = np.zeros(len(dest)+1, dtype=np.int32)
    np.cumsum(send_geom_size, out=geom_offset[1:])
    np.cumsum(send_top_size, out=top_offset[1:])

    if len(send_geom) == 0:
        send_geom = np.array([], dtype=ext_mesh.geometry.x.dtype)
    else:
        send_geom = np.vstack(send_geom).reshape(-1)
    if len(send_top) == 0:
        send_top = np.array([], dtype=np.int32)
    else:
        send_top = np.hstack(send_top)
    if len(send_geom_size) == 0:
        send_geom_size = np.zeros(1, dtype=np.int32)
    if len(send_top_size) == 0:
        send_top_size = np.zeros(1, dtype=np.int32)

    recv_geom_size = np.zeros(max(len(source), 1), dtype=np.int32)
    small_to_big_comm.Neighbor_alltoall(send_geom_size, recv_geom_size)
    recv_geom_size = recv_geom_size[:len(source)]
    send_geom_size = send_geom_size[:len(dest)]
    recv_geom_offsets = np.zeros(len(recv_geom_size)+1, dtype=np.int32)
    np.cumsum(recv_geom_size, out=recv_geom_offsets[1:])


    recv_top_size = np.zeros(max(len(source), 1), dtype=np.int32)
    small_to_big_comm.Neighbor_alltoall(send_top_size, recv_top_size)
    recv_top_size = recv_top_size[:len(source)]
    send_top_size = send_top_size[:len(dest)]

    recv_top_offsets = np.zeros(len(recv_top_size)+1, dtype=np.int32)
    np.cumsum(recv_top_size, out=recv_top_offsets[1:])

    numpy_to_mpi = {np.float64: MPI.DOUBLE,
                    np.float32: MPI.FLOAT, np.int8: MPI.INT8_T, np.int32: MPI.INT32_T}
    # Communicate data
    recv_geom = np.zeros(recv_geom_offsets[-1], dtype=ext_mesh.geometry.x.dtype)
    s_geom_msg = [send_geom, send_geom_size, numpy_to_mpi[send_geom.dtype.type]]
    r_geom_msg = [recv_geom, recv_geom_size, numpy_to_mpi[recv_geom.dtype.type]]
    small_to_big_comm.Neighbor_alltoallv(s_geom_msg, r_geom_msg)

    recv_top = np.zeros(recv_top_offsets[-1], dtype=np.int32)
    s_top_msg = [send_top, send_top_size, numpy_to_mpi[send_top.dtype.type]]
    r_top_msg = [recv_top, recv_top_size, numpy_to_mpi[recv_top.dtype.type]]
    small_to_big_comm.Neighbor_alltoallv(s_top_msg, r_top_msg)

    # For each received geometry, create a mesh
    local_meshes = []
    for i in range(len(source)):
        local_meshes.append(mesh.create_mesh(
            MPI.COMM_SELF,
            recv_top[recv_top_offsets[i]:recv_top_offsets[i+1]
                     ].reshape(-1, num_vertices_per_cell_small),
            recv_geom[recv_geom_offsets[i]:recv_geom_offsets[i+1]
                      ].reshape(-1, ext_mesh.geometry.dim),
            ext_mesh.ufl_domain()))

    # For each local mesh, compute the bounding box, compute colliding cells
    tol = 1e-13
    nodes = []
    num_local_cells = np.zeros(max(len(source), 1), dtype=np.int32)
    for i in range(len(source)):
        local_cells_i = set()
        o_cell_idx = local_meshes[i].topology.original_cell_index
        local_tree = geometry.bb_tree(
            local_meshes[i], local_meshes[i].topology.dim)
        cell_node_collisions = geometry.compute_collisions_trees(
            local_tree, nodal_bb_tree)
        for loc_cell, node in cell_node_collisions:

            geom_cell = extract_cell_geometry(local_meshes[i], loc_cell)
            geom_node = loc_mesh.geometry.x[node]
            d = geometry.compute_distance_gjk(geom_cell, geom_node)
            if np.linalg.norm(d) <= tol:
                nodes.append(node)

    return nodes

# TODO: Remove this?
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
                 width=1.0,height=1.0,depth=1.0,dim=3,shrink=True):
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
        if shrink:
            self.half_widths *= 0.999

    def set_transverse_axes(self,dim):
        '''
        Set y and z axes depending on dimension
        '''
        if (dim==2):
            self.z_axis = np.array([-self.x_axis[1],self.x_axis[0],0.0])
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
