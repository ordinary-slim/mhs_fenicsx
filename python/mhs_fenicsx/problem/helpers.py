from __future__ import annotations
from dolfinx import fem, mesh, geometry
import ufl
import numpy as np
import numpy.typing as npt
from mpi4py import MPI
import dolfinx.fem.petsc
import petsc4py.PETSc
import mhs_fenicsx_cpp
import typing

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def interpolate_cg1(sf : dolfinx.fem.Function,
                    rf : dolfinx.fem.Function,
                    scells : npt.NDArray[np.int32],
                    rdofs : npt.NDArray[np.int32],
                    coords_rdofs : npt.NDArray[np.float64],
                    padding=1e-6):
    mhs_fenicsx_cpp.interpolate_cg1_affine(sf._cpp_object,
                                           rf._cpp_object,
                                           scells,
                                           rdofs,
                                           coords_rdofs,
                                           padding)

def interpolate_dg0(sf : dolfinx.fem.Function,
                    rf : dolfinx.fem.Function,
                    scells : npt.NDArray[np.int32],
                    rcells : npt.NDArray[np.int32],
                    padding = 1e-6,
                    ):
    mhs_fenicsx_cpp.interpolate_dg0(sf._cpp_object,
                                    rf._cpp_object,
                                    scells,
                                    rcells,
                                    padding)

def l2_squared(f : dolfinx.fem.Function,active_els_tag):
    dx = ufl.Measure("dx")(subdomain_data=active_els_tag)
    l_ufl = f*f*dx(1)
    l2_norm = dolfinx.fem.assemble_scalar(fem.form(l_ufl))
    l2_norm = comm.allreduce(l2_norm, op=MPI.SUM)
    return l2_norm

def locate_active_boundary(domain, active_els_func):
    return mhs_fenicsx_cpp.locate_active_boundary(domain._cpp_object, active_els_func._cpp_object)

def get_mask(size, indices, dtype=np.int8, val=1):
    val = dtype(val)
    mask = np.zeros(size, dtype=dtype)
    if isinstance(val, np.ndarray):
        for i, v in zip(indices, val):
            mask[i] = v
    else:
        mask[indices] = val
    return mask

def interpolate_dg_at_facets(sending_f,
                             receiving_f,
                             facets,bb_tree_ext,
                             activation_tag,
                             ext_activation_tag,):
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
        potential_parent_els_ext = potential_parent_els_ext.array[np.flatnonzero( ext_activation_tag.values[ potential_parent_els_ext.array] ) ]
        if len(potential_parent_els_ext)>0:
            idx_owner_el = potential_parent_els_ext[0]
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
        parent_els  = parent_els[np.flatnonzero(activation_tag.values[parent_els])]
        assert (len(parent_els)) == 1
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

def inidices_to_nodal_meshtag(space, indices, dim):
    nodal_dofs = fem.locate_dofs_topological(space, dim, indices,)
    return mesh.meshtags(space.mesh, space.mesh.topology.dim,
                         nodal_dofs,
                         np.ones(len(nodal_dofs),
                                 dtype=np.int32),)

def indices_to_function(space, indices, dim, name="f", remote=True, f = None):
    dofs = fem.locate_dofs_topological(space, dim, indices,remote)
    if f is None:
        f = fem.Function(space,name=name)
    else:
        f.x.array[:] = 0
    f.x.array[dofs] = 1
    return f

def set_same_mesh_interface(p1:Problem, p2:Problem):
    assert(p1.domain == p2.domain)
    gamma_facets = np.logical_and(p1.bfacets_tag.values, p2.bfacets_tag.values).nonzero()[0]
    cdim = p1.domain.topology.dim
    gamma_facets_tag = mesh.meshtags(p1.domain, cdim-1,
                          np.arange(p1.num_facets, dtype=np.int32),
                          get_mask(p1.num_facets,gamma_facets))
    for p, p_ext in [(p2,p1), (p1,p2)]:
        p.set_gamma(p_ext, gamma_facets_tag)

def propagate_dg0_at_facets_same_mesh(ps:Problem, sf:fem.Function, pr:Problem, rf:fem.Function):
    assert(ps.domain == pr.domain)
    mhs_fenicsx_cpp.propagate_dg0_at_facets_same_mesh(
            sf._cpp_object,
            rf._cpp_object,
            ps.active_els_func._cpp_object,
            pr.active_els_func._cpp_object,
            ps.gamma_facets_index_map[pr],
            ps.gamma_imap_to_global_imap[pr]
            )

def assert_pointwise_vals(p:Problem, points, ref_vals, f = None, rtol=1.e-5):
    '''Test util'''
    if f is None:
        f = p.u
    po = mhs_fenicsx_cpp.cellwise_determine_point_ownership(
            p.domain._cpp_object,
            points,
            p.active_els_func.x.array.nonzero()[0],
            np.float64(1e-7),)
    indices_points_found = (po.src_owner == rank).nonzero()[0]
    rindices_points_found = (po.dest_owners == rank).nonzero()[0]

    vals = f.eval(po.dest_points[rindices_points_found], po.dest_cells[rindices_points_found]).reshape(-1)
    #for i, idx in zip(range(len(indices_points_found)), indices_points_found):
    #    print(f"{rank}: {po.dest_points[i]} ---> {vals[i]} / {ref_vals[idx]}", flush=True)
    assert np.isclose(ref_vals[indices_points_found], vals, rtol=rtol).all()

def print_vals(p:Problem, points, only_print=False):
    '''Test util'''
    po = mhs_fenicsx_cpp.cellwise_determine_point_ownership(
            p.domain._cpp_object,
            points,
            p.active_els_func.x.array.nonzero()[0],
            np.float64(1e-7),)

    rindices_points_found = (po.dest_owners == rank).nonzero()[0]
    vals = p.u.eval(po.dest_points[rindices_points_found], po.dest_cells[rindices_points_found])
    vals = vals.reshape(-1, 1)
    csv = np.hstack((po.dest_points[rindices_points_found], vals))
    if not(only_print):
        np.savetxt(f"{comm.size}_procs_run_proc#{rank}.csv", csv, delimiter=",")
    else:
        print(f"{rank}: {csv}", flush=True)

def get_identity_maps(form):
    coefficient_map = {}
    coefficient_map = {coeff:coeff for coeff in form.coefficients()}
    constant_map = {const:const for const in form.constants()}
    return coefficient_map, constant_map

def assert_gamma_tags(gamma_tags:typing.List[int], p:'Problem'):
    all_tags_found = True
    for gamma_tag in gamma_tags:
        tag_found = False
        for tag, _ in p.form_subdomain_data[fem.IntegralType.exterior_facet]:
            tag_found = (gamma_tag == tag)
            if tag_found:
                break
        all_tags_found = all_tags_found and tag_found
        if not(all_tags_found):
            break
    assert(all_tags_found)

def skip_assembly(x, last_x, dx):
    skip = False
    if last_x is not None:
        dx._copy(x)
        dx.axpy(-1.0, last_x)
        if dx.norm() < 1e-12:
            skip = True
        last_x._copy(x)
    else:
        dx, last_x = x.copy(), x.copy()
    return skip

