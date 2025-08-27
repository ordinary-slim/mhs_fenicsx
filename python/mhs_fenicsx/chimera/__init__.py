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

def shape_moving_problem(pm: Problem):
    mdparams = pm.input_parameters["moving_domain_params"]
    if not(mdparams["shape"]):
        pm.reset_activation(finalize=False)
    else:
        next_track = pm.source.path.get_track(pm.time)
        adim_dt = pm.adimensionalize_mhs_timestep(next_track)
        radius = pm.source.R
        center = np.array(pm.source.x)
        e = next_track.get_direction()
        back_len = pm.get_adim_back_len(0.5, adim_dt) * radius
        front_len = mdparams["adim_front_len"] * radius
        width = 2 * mdparams["adim_side_len"] * radius
        bot_len = mdparams["adim_bot_len"] * radius
        top_len = mdparams["adim_top_len"] * radius
        p0 = center - back_len * e
        p1 = center + front_len * e
        obb = OBB(p0, p1, width, top_len, bot_len, pm.dim)
        colliding_els = obb.broad_collision(pm.bb_tree)
        pm.set_activation(colliding_els, finalize=False)

def get_adim_back_len_paper(fine_adim_dt : float = 0.5, adim_dt : float = 2):
    return np.round(4*(1 + adim_dt + (adim_dt**2.3)*fine_adim_dt)) / 4

def get_adim_back_len_simple(fine_adim_dt : float = 0.5, adim_dt : float = 2):
    return adim_dt + 4 * fine_adim_dt

def build_moving_problem(p_fixed : Problem, els_per_radius=2, shift=None, custom_get_adim_back_len=None, symmetries=[], domain=None):
    params = p_fixed.input_parameters.copy()

    mdparams = params["moving_domain_params"]
    if params["moving_domain_params"]["shape"]:
        if custom_get_adim_back_len:
            get_adim_back_len = custom_get_adim_back_len
        else:
            get_adim_back_len = get_adim_back_len_paper
    else:
        get_adim_back_len = get_adim_back_len_simple

    if domain is not None:
        moving_domain = domain
    else:
        adim_back_len = get_adim_back_len(0.5, mdparams["max_adim_dt"])
        adim_front_len = mdparams["adim_front_len"]
        adim_side_len = mdparams["adim_side_len"]
        adim_bot_len = mdparams["adim_bot_len"]
        adim_top_len = mdparams["adim_top_len"]
        moving_domain = mesh_around_hs(p_fixed.source, p_fixed.domain.topology.dim, els_per_radius, shift,
                                       adim_back_len, adim_front_len, adim_side_len, adim_bot_len, adim_top_len,
                                       symmetries=symmetries)
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
                   adim_top_len = 2.0,
                   symmetries = []):
    center_of_mesh = np.array(hs.x)
    back_length  = hs.R * adim_back_len
    front_length = hs.R * adim_front_len
    side_length  = hs.R * adim_side_len
    bot_length   = hs.R * adim_bot_len
    top_length   = hs.R * adim_top_len

    el_size = hs.R / float(els_per_radius)

    lens = np.array([-back_length,
                     -side_length,
                     -bot_length,
                     +front_length,
                     +side_length,
                     +top_length,])

    for symmetry in symmetries:
        axis, side = symmetry[0], symmetry[1]
        # We are keeping side `side`and discarding the other side
        idx_bound = axis + (-side > 0) * 3
        lens[idx_bound] = 0.0

    mesh_bounds = np.hstack((center_of_mesh, center_of_mesh)) + lens

    if shift is not None:
        assert(shift.size == 3)
        for i in range(3):
            mesh_bounds[i]   += shift[i]
            mesh_bounds[i+3] += shift[i]

    nels = [-1] * 3
    for axis in range(3):
        len = abs(lens[axis]) + abs(lens[axis+3])
        nels[axis] = np.rint(len/el_size).astype(int)

    if dim==1:
        return dolfinx.mesh.create_interval(MPI.COMM_WORLD,
                                            nels[:dim],
                                            [mesh_bounds[0],mesh_bounds[3]],
                                            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,)
    elif dim==2:
        return dolfinx.mesh.create_rectangle(MPI.COMM_WORLD,
                                             [mesh_bounds[:2],mesh_bounds[3:5]],
                                             nels[:dim],
                                             dolfinx.mesh.CellType.quadrilateral,
                                             ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
                                             )
    else:
        return dolfinx.mesh.create_box(MPI.COMM_WORLD,
                                       [mesh_bounds[:3],mesh_bounds[3:]],
                                       nels[:dim],
                                       dolfinx.mesh.CellType.hexahedron,
                                       ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
                                       )
