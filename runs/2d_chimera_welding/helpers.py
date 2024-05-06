import dolfinx.mesh
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, HeatSource

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
