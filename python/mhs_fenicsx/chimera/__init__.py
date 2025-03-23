import dolfinx.mesh
import numpy as np
from mpi4py import MPI
from mhs_fenicsx.problem import Problem, HeatSource
from mhs_fenicsx.problem.helpers import interpolate_cg1
from mhs_fenicsx.geometry import OBB

def interpolate_solution_to_inactive(p:Problem, p_ext:Problem, cells1 = None, finalize=True):
    local_ext_inactive_dofs = np.logical_and((p.active_nodes_func.x.array == 0), p.ext_nodal_activation[p_ext]).nonzero()[0]
    local_ext_inactive_dofs = local_ext_inactive_dofs[:np.searchsorted(local_ext_inactive_dofs, p.domain.topology.index_map(0).size_local)]
    if cells1 is None:
        cells1 = p_ext.local_active_els
    interpolate_cg1(p_ext.u,
                    p.u,
                    cells1,
                    local_ext_inactive_dofs,
                    p.dof_coords[local_ext_inactive_dofs],
                    1e-6)
    if finalize:
        p.post_modify_solution(cells=p.ext_colliding_els[p_ext])

def shape_moving_problem(pm : Problem):
    if not(pm.input_parameters["moving_domain_params"]["shape"]):
        pm.reset_activation(finalize=False)
    else:
        next_track = pm.source.path.get_track(pm.time)
        adim_dt = pm.adimensionalize_mhs_timestep(next_track)
        mdparams = pm.input_parameters["moving_domain_params"]
        radius = pm.source.R
        center = np.array(pm.source.x)
        e = next_track.get_direction()
        back_len = pm.get_adim_back_len(0.5, adim_dt) * radius
        front_len = mdparams["adim_front_len"] * radius
        side_len = mdparams["adim_side_len"] * radius
        bot_len = mdparams["adim_bot_len"] * radius
        top_len = mdparams["adim_top_len"] * radius
        p0 = center - back_len * e
        p1 = center + front_len * e
        obb = OBB(p0, p1, side_len, top_len, bot_len, pm.dim)
        colliding_els = obb.broad_collision(pm.bb_tree)
        pm.set_activation(colliding_els, finalize=False)

def get_adim_back_len_paper(fine_adim_dt : float = 0.5, adim_dt : float = 2):
    return np.round(4*(1 + adim_dt + (adim_dt**2.3)*fine_adim_dt)) / 4

def get_adim_back_len_simple(fine_adim_dt : float = 0.5, adim_dt : float = 2):
    return adim_dt + 4 * fine_adim_dt

def build_moving_problem(p_fixed : Problem, els_per_radius=2, shift=None, custom_get_adim_back_len=None):
    params = p_fixed.input_parameters.copy()
    mdparams = params["moving_domain_params"]
    if params["moving_domain_params"]["shape"]:
        if custom_get_adim_back_len:
            get_adim_back_len = custom_get_adim_back_len
        else:
            get_adim_back_len = get_adim_back_len_paper
    else:
        get_adim_back_len = get_adim_back_len_simple
    adim_back_len = get_adim_back_len(0.5, mdparams["max_adim_dt"])
    adim_front_len = mdparams["adim_front_len"]
    adim_side_len = mdparams["adim_side_len"]
    adim_bot_len = mdparams["adim_bot_len"]
    adim_top_len = mdparams["adim_top_len"]
    moving_domain = mesh_around_hs(p_fixed.source, p_fixed.domain.topology.dim, els_per_radius, shift,
                                   adim_back_len, adim_front_len, adim_side_len, adim_bot_len, adim_top_len)
    params["attached_to_hs"] = 1
    # Placeholders
    params["domain_speed"] = np.array([1.0, 0.0, 0.0])
    params["advection_speed"] = np.array([-1.0, 0.0, 0.0])
    if "petsc_opts_moving" in params:
        params["petsc_opts"] = params["petsc_opts_moving"]
    p = Problem(moving_domain, params,name=p_fixed.name+"_moving")
    p.get_adim_back_len = get_adim_back_len
    return p

def mesh_around_hs(hs:HeatSource,
                   dim:int,
                   els_per_radius:int,
                   shift=None,
                   adim_back_len = 4.0,
                   adim_front_len = 2.0,
                   adim_side_len = 2.0,
                   adim_bot_len = 2.0,
                   adim_top_len = 2.0):
    center_of_mesh = np.array(hs.x)
    back_length  = hs.R * adim_back_len
    front_length = hs.R * adim_front_len
    side_length  = hs.R * adim_side_len
    bot_length   = hs.R * adim_bot_len
    top_length   = hs.R * adim_top_len

    el_size      = hs.R / float(els_per_radius)

    mesh_bounds  = [
            center_of_mesh[0]-back_length,
            center_of_mesh[1]-side_length,
            center_of_mesh[2]-bot_length,
            center_of_mesh[0]+front_length,
            center_of_mesh[1]+side_length,
            center_of_mesh[2]+top_length,
            ]
    if shift is not None:
        assert(shift.size == 3)
        for i in range(3):
            mesh_bounds[i]   += shift[i]
            mesh_bounds[i+3] += shift[i]
    nx = np.rint((back_length+front_length)/el_size).astype(int)
    ny = np.rint(side_length*2/el_size).astype(int)
    nz = np.rint((top_length+bot_length)/el_size).astype(int)
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
