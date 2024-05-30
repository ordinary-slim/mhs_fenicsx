from dolfinx import fem, mesh, io, geometry
from mpi4py import MPI
import numpy as np
import basix.ufl, basix.cell
import ufl
import mhs_fenicsx_cpp
import petsc4py

comm = MPI.COMM_WORLD
rank = comm.rank

def mesh_rectangle(min_x, min_y, max_x, max_y, elsize, cell_type=mesh.CellType.triangle):
    nx = np.ceil((max_x-min_x)/elsize).astype(np.int32)
    ny = np.ceil((max_y-min_y)/elsize).astype(np.int32)
    return mesh.create_rectangle(MPI.COMM_WORLD,
                                 [np.array([min_x,min_y]),np.array([max_x,max_y])],
                                 [nx,ny],
                                 cell_type,)

def interpolate_dg0_at_facets(sending_f,
                             receiving_f,
                             facets,bb_tree_ext,):
    domain           = receiving_f.function_space.mesh
    ext_domain       = sending_f.function_space.mesh
    cdim = domain.topology.dim
    function_dim = 1 if (len(receiving_f.ufl_shape) == 0) else receiving_f.ufl_shape[0]
    # Build Gamma midpoints array
    local_interface_midpoints = np.zeros((len(facets), 3), np.double)
    for i, ifacet in enumerate(facets):
        local_interface_midpoints[i,:] = mesh.compute_midpoints(domain,cdim-1,np.array([ifacet],dtype=np.int32))

    facet_counts  = np.zeros(comm.size, dtype=np.int32)
    facets_offsets = np.zeros(comm.size, dtype=np.int32)
    comm.Allgather(np.array([len(facets)], np.int32), facet_counts)
    facets_offsets[1:] = np.cumsum(facet_counts[:-1])
    total_facet_count = np.sum(facet_counts, dtype=int)

    global_interface_midpoints = np.zeros((total_facet_count,3), dtype=np.double, order='C')
    comm.Allgatherv(local_interface_midpoints,[global_interface_midpoints,facet_counts*local_interface_midpoints.shape[1],facets_offsets*local_interface_midpoints.shape[1],MPI.DOUBLE])

    # Collect values at midpoints
    local_vals  = np.zeros((total_facet_count,function_dim),dtype=np.double,order='C')
    global_vals = np.zeros((total_facet_count,function_dim),dtype=np.double,order='C')
    found_local  = np.zeros(total_facet_count,dtype=np.double,order='C')
    found_global = np.zeros(total_facet_count,dtype=np.double,order='C')
    for idx in range(total_facet_count):
        candidate_parents_ext = geometry.compute_collisions_points(bb_tree_ext,global_interface_midpoints[idx,:])
        potential_parent_els_ext = geometry.compute_colliding_cells(ext_domain, candidate_parents_ext, global_interface_midpoints[idx,:])
        #potential_parent_els_ext = potential_parent_els_ext.array[np.flatnonzero( ext_activation_tag.values[ potential_parent_els_ext.array] ) ]
        if len(potential_parent_els_ext.array)>0:
            idx_owner_el = potential_parent_els_ext.array[0]
            if idx_owner_el < ext_domain.topology.index_map(cdim).size_local:
                local_vals[idx,:]  = sending_f.eval(global_interface_midpoints[idx,:], idx_owner_el)
                found_local[idx] = 1
    comm.Allreduce([local_vals, MPI.DOUBLE], [global_vals, MPI.DOUBLE])
    comm.Allreduce([found_local, MPI.DOUBLE], [found_global, MPI.DOUBLE])

    f_to_c_left = domain.topology.connectivity(1,2)

    # build global parent el array for facets
    global_parent_els_proc = np.zeros(len(facets), np.int32)
    for idx, ifacet in enumerate(facets):
        parent_els  = f_to_c_left.links(ifacet)
        #parent_els  = parent_els[np.flatnonzero(activation_tag.values[parent_els])]
        assert (len(parent_els)) >= 1
        parent_el_glob  = domain.topology.index_map(domain.geometry.dim).local_to_global(parent_els)
        global_parent_els_proc[idx] = parent_el_glob[0]

    global_parent_els = np.zeros(total_facet_count, np.int32)
    comm.Allgatherv(global_parent_els_proc,[global_parent_els,facet_counts,facets_offsets,MPI.INT])
    local_parent_els  = domain.topology.index_map(domain.geometry.dim).global_to_local(global_parent_els)

    for idx, el in enumerate(local_parent_els):
        if el < 0:
            continue
        flat_idx    = el*function_dim
        receiving_f.x.array[flat_idx:flat_idx+2] = global_vals[idx]

    receiving_f.vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)

def new_interpolate_dg0_at_facets(sending_f,
                                  receiving_f,
                                  facets,bb_tree_ext,):
    domain           = receiving_f.function_space.mesh
    ext_domain       = sending_f.function_space.mesh
    cdim = domain.topology.dim
    function_dim = 1 if (len(receiving_f.ufl_shape) == 0) else receiving_f.ufl_shape[0]
    midpoint_tree = geometry.create_midpoint_tree(domain,1,facets)
    global_midpoint_tree = midpoint_tree.create_global_tree(comm)
    print(f"Rank = {rank}, num_bboxes of local = {midpoint_tree.num_bboxes}")
    print(f"Rank = {rank}, num_bboxes of global = {global_midpoint_tree.num_bboxes}")

class Problem:
    def __init__(self, domain, name="case"):
        self.domain = domain
        self.name = name
        self.V    = fem.functionspace(self.domain,("CG",1))
        self.dg0  = fem.functionspace(self.domain, ("DG", 0))
        dg0_dim2_element = basix.ufl.element("DG", self.domain.basix_cell(), 0, shape=(2,))
        self.dg0_dim2 = fem.functionspace(self.domain, dg0_dim2_element)
        self.uh   = fem.Function(self.V, name="uh")
        self.dim  = domain.topology.dim

        self.domain.topology.create_entities(self.dim-1)
        self.cell_map  = self.domain.topology.index_map(self.dim)
        self.facet_map = self.domain.topology.index_map(self.dim-1)

        self.bb_tree = geometry.bb_tree(domain,self.dim)
        self.domain.topology.create_connectivity(1,2)


    def writepos(self, funcs=[]):
        with io.VTKFile(self.domain.comm, f"out/{self.name}.pvd", "w") as ofile:
            ofile.write_function(funcs)

el_size = 0.125
mesh_left  = mesh_rectangle(0, 0, 0.5, 1, elsize=el_size)
mesh_right = mesh_rectangle(0.5, 0, 1, 1, elsize=el_size, cell_type=mesh.CellType.quadrilateral)

p_right = Problem(mesh_right, name="right")
p_left  = Problem(mesh_left, name="left")

d_f   = fem.Function(p_right.dg0,name="d_f")
der   = fem.Function(p_right.dg0_dim2,name="der")
c_f   = fem.Function(p_right.V,name="c_f")
x = ufl.SpatialCoordinate(mesh_right)
c_f.interpolate( fem.Expression(x[0]**2 + x[1]**2,c_f.function_space.element.interpolation_points()))
d_f.interpolate( fem.Expression(x[0],d_f.function_space.element.interpolation_points()))
der.interpolate( fem.Expression(ufl.grad(c_f),der.function_space.element.interpolation_points()))

rfacets = mesh.locate_entities_boundary(p_left.domain,p_left.dim-1,lambda x:np.isclose(x[0],0.5))

der_python  = fem.Function(p_left.dg0_dim2,name="der")
der_python2 = fem.Function(p_left.dg0_dim2,name="der")
interpolate_dg0_at_facets(der,
                          der_python,
                          rfacets,
                          p_right.bb_tree)
new_interpolate_dg0_at_facets(der,
                              der_python2,
                              rfacets,
                              p_right.bb_tree)

p_left.writepos(funcs=[der_python])
p_right.writepos(funcs=[c_f,d_f,der])
