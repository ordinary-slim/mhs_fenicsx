import dolfinx.mesh
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, HeatSource
import mhs_fenicsx_cpp

def interpolate_solution_to_inactive(p:Problem, p_ext:Problem):
    local_ext_inactive_dofs = np.logical_and((p.active_nodes_func.x.array == 0), p.ext_nodal_activation[p_ext]).nonzero()[0]
    local_ext_inactive_dofs = local_ext_inactive_dofs[:np.searchsorted(local_ext_inactive_dofs, p.domain.topology.index_map(0).size_local)]
    mhs_fenicsx_cpp.interpolate_cg1_affine(p_ext.u._cpp_object,
                                           p.u._cpp_object,
                                           np.arange(p_ext.cell_map.size_local),
                                           local_ext_inactive_dofs,
                                           p.dof_coords[local_ext_inactive_dofs],
                                           1e-6)

def build_moving_problem(p_fixed:Problem,els_per_radius=2):
    moving_domain = mesh_around_hs(p_fixed.source,p_fixed.domain.topology.dim,els_per_radius)
    params = p_fixed.input_parameters.copy()
    params["attached_to_hs"] = 1
    # Placeholders
    params["domain_speed"] = np.array([1.0, 0.0, 0.0])
    params["advection_speed"] = np.array([-1.0, 0.0, 0.0])
    if "petsc_opts_moving" in params:
        params["petsc_opts"] = params["petsc_opts_moving"]
    p = Problem(moving_domain, params,name=p_fixed.name+"_moving")
    return p

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
