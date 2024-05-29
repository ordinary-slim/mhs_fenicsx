import dolfinx.mesh
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, HeatSource
from mhs_fenicsx.problem.helpers import interpolate
from mhs_fenicsx.geometry import mesh_containment
from dolfinx import fem
import mhs_fenicsx_cpp

def interpolate_solution_to_inactive(p:Problem, p_ext:Problem):
    u_ext = fem.Function(p.v)
    interpolate(p_ext.u,u_ext)
    inactive_nodes = np.where(p.active_nodes_func.x.array==0)[0]
    p.u.x.array[inactive_nodes] = u_ext.x.array[inactive_nodes]

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

def build_moving_problem(p_fixed:Problem,els_per_radius=2):
    moving_domain = mesh_around_hs(p_fixed.source,p_fixed.domain.topology.dim,els_per_radius)
    params = p_fixed.input_parameters.copy()
    if "path" in params:
        params.pop("path")
    params["domain_speed"]    = p_fixed.source.speed
    params["advection_speed"] = -p_fixed.source.speed
    params["petsc_opts"] = params["petsc_opts_moving"]
    return Problem(moving_domain, params,name=p_fixed.name+"_moving")

def mesh_around_hs(hs:HeatSource, dim:int,els_per_radius:int):
    center_of_mesh = np.array(hs.x)
    back_length  = hs.R * 4
    front_length = hs.R * 2
    side_length  = hs.R * 2
    bot_length   = hs.R * 2
    top_length   = hs.R * 2
    el_size      = hs.R / float(els_per_radius)
    mesh_bounds  = [
            center_of_mesh[0]-back_length,
            center_of_mesh[1]-side_length,
            center_of_mesh[2]-bot_length,
            center_of_mesh[0]+front_length,
            center_of_mesh[1]+side_length,
            center_of_mesh[2]+top_length,
            ]
    nx = np.round((back_length+front_length)/el_size).astype(int)
    ny = np.round(side_length*2/el_size).astype(int)
    nz = np.round((top_length+bot_length)/el_size).astype(int)
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
