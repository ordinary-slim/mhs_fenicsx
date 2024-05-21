from __future__ import annotations
from dolfinx import fem, mesh, geometry
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import multiphenicsx
import multiphenicsx.fem.petsc
import dolfinx.fem.petsc
import petsc4py.PETSc
from dolfinx import default_scalar_type
from abc import ABC, abstractmethod
import mhs_fenicsx_cpp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def interpolate(sending_func,
                receiving_func,):
    '''
    Interpolate sending_func to receiving_func,
    each comming from separate meshes
    '''
    topology = receiving_func.function_space.mesh.topology
    cmap = topology.index_map(topology.dim)
    num_cells = cmap.size_local + cmap.num_ghosts
    cells = np.arange(num_cells,dtype=np.int32)
    nmmid = fem.create_interpolation_data(
                                 receiving_func.function_space,
                                 sending_func.function_space,
                                 cells,
                                 padding=1e-6,)
    receiving_func.interpolate_nonmatching(sending_func, cells, interpolation_data=nmmid)
    return receiving_func,

def l2_squared(f : dolfinx.fem.Function,active_els_tag):
    dx = ufl.Measure("dx")(subdomain_data=active_els_tag)
    l_ufl = f*f*dx(1)
    l2_norm = dolfinx.fem.assemble_scalar(fem.form(l_ufl))
    l2_norm = comm.allreduce(l2_norm, op=MPI.SUM)
    return l2_norm

def locate_active_boundary(domain, active_els_func):
    return mhs_fenicsx_cpp.locate_active_boundary(domain._cpp_object, active_els_func._cpp_object)

def get_facet_integration_entities(domain,facet_indices,active_els_func):
    return mhs_fenicsx_cpp.get_facet_integration_entities(domain._cpp_object,
                                                          facet_indices,
                                                          active_els_func._cpp_object,
                                                          )

def get_mask(size, indices, dtype=np.int32, true_val=1):
    true_val = dtype(true_val)
    mask = np.zeros(size, dtype=dtype)
    mask[indices] = true_val
    return mask

def interpolate_dg_at_facets(f,facets,targetSpace,bb_tree_ext,
                             activation_tag,
                             ext_activation_tag,
                             name="flux"):
    interpolated_f = fem.Function(targetSpace,name=name)
    domain           = targetSpace.mesh
    ext_domain       = f.function_space.mesh
    cdim = domain.topology.dim
    function_dim = 1 if (len(interpolated_f.ufl_shape) == 0) else interpolated_f.ufl_shape[0]
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
                local_vals[idx,:]  = f.eval(global_interface_midpoints[idx,:], idx_owner_el)
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
        interpolated_f.x.array[flat_idx:flat_idx+2] = global_vals[idx]

    interpolated_f.vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    return interpolated_f

def inidices_to_nodal_meshtag(space, indices, dim):
    nodal_dofs = fem.locate_dofs_topological(space, dim, indices,)
    return mesh.meshtags(space.mesh, space.mesh.topology.dim,
                         nodal_dofs,
                         np.ones(len(nodal_dofs),
                                 dtype=np.int32),)

def indices_to_function(space, indices, dim, name="f"):
    dofs = fem.locate_dofs_topological(space, dim, indices,)
    f = fem.Function(space,name=name)
    f.x.array[dofs] = 1
    return f

