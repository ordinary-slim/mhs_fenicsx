from dolfinx import fem, mesh, io, geometry
from mpi4py import MPI
import numpy as np
import basix.ufl, basix.cell
import ufl
import pdb

comm = MPI.COMM_WORLD
rank = comm.rank

def exact_sol_ufl(mesh):
    x = ufl.SpatialCoordinate(mesh)
    return 2 -(x[0]**2 + x[1]**2)
def exact_flux_ufl(mesh):
    x = ufl.SpatialCoordinate(mesh)
    return ufl.as_vector((-2*x[0], -2*x[1]))
def rhs():
    return 4


def mesh_rectangle(min_x, min_y, max_x, max_y, elsize, cell_type=mesh.CellType.triangle):
    nx = np.ceil((max_x-min_x)/elsize).astype(np.int32)
    ny = np.ceil((max_y-min_y)/elsize).astype(np.int32)
    return mesh.create_rectangle(MPI.COMM_WORLD,
                                 [np.array([min_x,min_y]),np.array([max_x,max_y])],
                                 [nx,ny],
                                 cell_type,)

def interpolate_dg_at_facets(f,facets,targetSpace,bb_tree_ext,):
    interpolated_f = fem.Function(targetSpace,name="der")
    domain           = targetSpace.mesh
    ext_domain       = f.function_space.mesh
    cdim = domain.topology.dim
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
    local_vals  = np.zeros((total_facet_count,cdim),dtype=np.double,order='C')
    global_vals = np.zeros((total_facet_count,cdim),dtype=np.double,order='C')
    for idx in range(total_facet_count):
        candidate_parents_ext = geometry.compute_collisions_points(bb_tree_ext,global_interface_midpoints[idx,:])
        parent_el_ext = geometry.compute_colliding_cells(ext_domain, candidate_parents_ext, global_interface_midpoints[idx,:])
        if len(parent_el_ext.array)>0:
            idx_owner_el = parent_el_ext.array[0]
            if idx_owner_el < ext_domain.topology.index_map(cdim).size_local:
                local_vals[idx,:]  = f.eval(global_interface_midpoints[idx,:], idx_owner_el)
    comm.Allreduce([local_vals, MPI.DOUBLE], [global_vals, MPI.DOUBLE])

    f_to_c_left = domain.topology.connectivity(1,2)

    for idx, ifacet in enumerate(facets):
        parent_el = f_to_c_left.links(ifacet)[0]
        flat_idx    = parent_el*interpolated_f.ufl_shape[0]
        value       = global_vals[facets_offsets[rank]+idx,:]
        interpolated_f.x.array[flat_idx:flat_idx+2] = value
    return interpolated_f


class Problem:
    def __init__(self, domain, name="case"):
        self.domain = domain
        self.name = name
        self.V    = fem.functionspace(self.domain,("CG",1))
        self.uh   = fem.Function(self.V, name="uh")
        self.dim  = domain.topology.dim

        self.domain.topology.create_entities(self.dim-1)
        self.cell_map  = self.domain.topology.index_map(self.dim)
        self.facet_map = self.domain.topology.index_map(self.dim-1)

        self.bb_tree = geometry.bb_tree(domain,self.dim)


    def writepos(self, extra_funcs=[], only_mesh=False):
        funcs = [self.uh]
        funcs.extend(extra_funcs)
        with io.VTKFile(self.domain.comm, f"out/{self.name}.pvd", "w") as ofile:
            ofile.write_mesh(self.domain)
            if not(only_mesh):
                ofile.write_function(funcs)

el_size = 0.125
mesh_left  = mesh_rectangle(0, 0, 0.5, 1, elsize=el_size)
mesh_right = mesh_rectangle(0.5, 0, 1, 1, elsize=el_size, cell_type=mesh.CellType.quadrilateral)

p_left  = Problem(mesh_left, name="left")
p_right = Problem(mesh_right, name="right")

dg0_right = fem.functionspace(mesh_right, ("DG", 0))
dg0_dim2_el_right = basix.ufl.element("DG", mesh_right.basix_cell(), 0, shape=(2,))
dg0_dim2_right = fem.functionspace(mesh_right, dg0_dim2_el_right)
d_f   = fem.Function(dg0_right,name="d_f")
der   = fem.Function(dg0_dim2_right,name="der")
c_f   = fem.Function(p_right.V,name="c_f")
x = ufl.SpatialCoordinate(mesh_right)
c_f.interpolate( fem.Expression(x[0]**2 + x[1]**2,c_f.function_space.element.interpolation_points()))
d_f.interpolate( fem.Expression(x[0],d_f.function_space.element.interpolation_points()))
der.interpolate( fem.Expression(ufl.grad(c_f),der.function_space.element.interpolation_points()))

dg0_dim2_el_left = basix.ufl.element("DG", mesh_left.basix_cell(), 0, shape=(2,))
dg0_dim2_left = fem.functionspace(mesh_left, dg0_dim2_el_left)

rfacets = mesh.locate_entities_boundary(p_left.domain,p_left.dim-1,lambda x:np.isclose(x[0],0.5))

interpolated_der = interpolate_dg_at_facets(der,rfacets,dg0_dim2_left,p_right.bb_tree)

p_left.writepos(extra_funcs=[interpolated_der])
p_right.writepos(extra_funcs=[c_f,d_f,der])
