import dolfinx.mesh
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, HeatSource
from mhs_fenicsx.geometry import mesh_containment
from dolfinx import fem
import mhs_fenicsx_cpp

def interpolate(func2project,
                targetSpace,
                interpolate,):
    nmmid = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
                                 targetSpace.mesh,
                                 targetSpace.element,
                                 func2project.ufl_function_space().mesh,
                                 padding=1e-6,)
    interpolate.interpolate(func2project, nmm_interpolation_data=nmmid)
    return interpolate

def get_active_in_external_trees(p_loc:Problem, p_ext:Problem ):
    '''
    Return nodal function with nodes active in p_ext
    '''
    # Get function on p_ext
    loc_active_dofs = fem.Function( p_loc.v )
    #inodes = mhs_fenicsx_cpp.get_active_dofs_external(loc_active_dofs._cpp_object,
                                                      #p_ext.active_els_func._cpp_object,
                                                      #p_loc.bb_tree_nodes._cpp_object,
                                                      #p_loc.domain._cpp_object,
                                                      #p_ext.bb_tree._cpp_object,
                                                      #p_ext.domain._cpp_object,)
    inodes = mesh_containment(p_loc.bb_tree_nodes,p_loc.domain,
                              p_ext.bb_tree,p_ext.domain,)
    loc_active_dofs.x.array[fem.locate_dofs_topological(p_loc.v, 0, inodes)] = 1.0
    loc_active_dofs.x.scatter_forward()
    return loc_active_dofs

def build_moving_problem(p_fixed:Problem):
    moving_domain = mesh_around_hs(p_fixed.source,p_fixed.domain.topology.dim)
    params = p_fixed.input_parameters.copy()
    if "path" in params:
        params.pop("path")
    params["domain_speed"]    = p_fixed.source.speed
    params["advection_speed"] = -p_fixed.source.speed
    return Problem(moving_domain, params,name=p_fixed.name+"_moving")

def mesh_around_hs(hs:HeatSource, dim:int):
    center_of_mesh = np.array(hs.x)
    back_length  = hs.R * 2
    front_length = hs.R * 2
    side_length  = hs.R * 2
    bot_length   = hs.R * 2
    top_length   = hs.R * 2
    el_size      = hs.R / 2.0
    mesh_bounds  = [
            center_of_mesh[0]-back_length,
            center_of_mesh[1]-side_length,
            center_of_mesh[2]-bot_length,
            center_of_mesh[0]+front_length,
            center_of_mesh[1]+side_length,
            center_of_mesh[2]+top_length,
            ]
    nx = np.ceil((back_length+front_length)/el_size).astype(int)
    ny = np.ceil(side_length*2/el_size).astype(int)
    nz = np.ceil((top_length+bot_length)/el_size).astype(int)
    if dim==1:
        return dolfinx.mesh.create_interval(MPI.COMM_WORLD,
                                            nx,
                                            [mesh_bounds[0],mesh_bounds[3]],
                                            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,)
    elif dim==2:
        return dolfinx.mesh.create_rectangle(MPI.COMM_WORLD,
                                             [mesh_bounds[:2],mesh_bounds[3:5]],
                                             [nx,ny],
                                             dolfinx.mesh.CellType.quadrilateral,
                                             ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
                                             )
    else:
        return dolfinx.mesh.create_box(MPI.COMM_WORLD,
                                       [mesh_bounds[:3],mesh_bounds[3:]],
                                       [nx,ny,nz],
                                       dolfinx.mesh.CellType.hexahedron,
                                       ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
                                       )
