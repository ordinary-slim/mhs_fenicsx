from dolfinx import fem, mesh, geometry
from mhs_fenicsx.problem import Problem
from mhs_fenicsx_cpp import build_subentity_to_parent_mapping as build_subentity_to_parent_mapping_cpp
import numpy as np

# removable
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def build_subentity_to_parent_mapping(edim:int,pdomain:mesh.Mesh,cdomain:mesh.Mesh,subcell_map,subvertex_map):
    return build_subentity_to_parent_mapping_cpp(edim,
                                                 pdomain._cpp_object,
                                                 cdomain._cpp_object,
                                                 subcell_map,
                                                 subvertex_map)

def find_submesh_interface(parent_problem:Problem,child_problem:Problem,submesh_data):
    (pp,cp) = (parent_problem,child_problem)
    cdim = pp.domain.topology.dim
    cmask = np.zeros(cp.num_facets, dtype=np.int32)
    fast_bfacet_indices = cp.bfacets_tag.values.nonzero()[0]
    bfacets_gamma_tag = np.int32(1) - pp.bfacets_tag.values[submesh_data["subfacet_map"][fast_bfacet_indices]]
    fgamma_facets_indices = fast_bfacet_indices[bfacets_gamma_tag.nonzero()[0]]
    cmask[fgamma_facets_indices] = 1
    cp.set_gamma(mesh.meshtags(cp.domain, cdim-1,
                               np.arange(cp.num_facets, dtype=np.int32),
                               cmask))
    pmask = np.zeros(pp.num_facets, dtype=np.int32)
    pmask[submesh_data["subfacet_map"][fgamma_facets_indices]] = 1
    pp.set_gamma(mesh.meshtags(pp.domain, cdim-1,
                               np.arange(pp.num_facets, dtype=np.int32),
                               pmask))

def compute_dg0_interpolation_data(parent_problem:Problem,child_problem:Problem,submesh_data):
    (pp,cp) = (parent_problem,child_problem)
    # Build interpolation data dg0 boundary
    tdim = cp.domain.topology.dim
    ccon_f2c = cp.domain.topology.connectivity(tdim-1,tdim)
    pcon_f2c = pp.domain.topology.connectivity(tdim-1,tdim)
    indices_gamma_facets = cp.gamma_facets.values.nonzero()[0]
    num_gamma_facets = len(indices_gamma_facets)
    submesh_data["cinterface_cells"] = np.full(num_gamma_facets,-1)
    submesh_data["pinterface_cells"] = np.full(num_gamma_facets,-1)
    for idx in range(num_gamma_facets):
        ifacet = indices_gamma_facets[idx]
        ccell = ccon_f2c.links(ifacet)
        assert(len(ccell)==1)
        ccell = ccell[0]
        if ccell >= cp.cell_map.size_local:
            continue
        submesh_data["cinterface_cells"][idx] = ccell
        pifacet = submesh_data["subfacet_map"][ifacet]
        pcells = pcon_f2c.links(pifacet)
        if pcells[0] == submesh_data["subcell_map"][ccell]:
            submesh_data["pinterface_cells"][idx] = pcells[1]
        else:
            submesh_data["pinterface_cells"][idx] = pcells[0]
