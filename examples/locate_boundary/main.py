from dolfinx import mesh, io, fem
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main():
    num_points_side = 16
    domain  = mesh.create_unit_square(comm,
              num_points_side,num_points_side,
              mesh.CellType.quadrilateral,)
    cdim = domain.topology.dim
    cell_map = domain.topology.index_map(cdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    print(f"Rank {rank}, num ghosts = {cell_map.num_ghosts}")
    cg1 = fem.functionspace(domain, ("CG", 1),)
    dg0 = fem.functionspace(domain, ("Discontinuous Lagrange", 0),)
    domain.topology.create_connectivity(cdim,cdim)
    active_els = mesh.locate_entities(domain,cdim,lambda x: x[cdim-1] < 0.3)
    active_els_tag = mesh.meshtags(domain, cdim,
                                   np.arange(num_cells, dtype=np.int32),
                                   get_mask(num_cells, active_els),)
    active_els_tag.name = "active_els"
    partition_func = fem.Function(dg0,name="partition")
    partition_func.x.array[:] = rank

    # TODO: Mark boundary of active_els
    bfacets = []
    domain.topology.create_connectivity(cdim-1, cdim)
    con_facet_cell = domain.topology.connectivity(cdim-1, cdim)
    num_facets_local = domain.topology.index_map(cdim-1).size_local
    for ifacet in range(con_facet_cell.num_nodes):
        local_con = con_facet_cell.links(ifacet)
        incident_active_els = 0
        for el in local_con:
            if (el in active_els):
                incident_active_els += 1
        if (incident_active_els==1) and (ifacet < num_facets_local):
            bfacets.append(ifacet)
    bnodes_func = indices_to_function(cg1,bfacets,cdim-1,name="bnodes")
    # ETODO

    dg0_funcs = []
    for tag in [active_els_tag]:
        dg0_funcs.append(indices_to_function(dg0,tag.find(1),cdim,name=tag.name))
    dg0_funcs.append(partition_func)
    funcs = [bnodes_func] + dg0_funcs

    with io.VTKFile(domain.comm,"out/case.pvd", 'w') as writer:
        writer.write_function(funcs)

def get_mask(size, indices, dtype=np.int32, true_val=1):
    true_val = dtype(true_val)
    mask = np.zeros(size, dtype=dtype)
    mask[indices] = true_val
    return mask
def indices_to_function(space, indices, dim, name="f"):
    dofs = fem.locate_dofs_topological(space, dim, indices,)
    f = fem.Function(space,name=name)
    f.x.array[dofs] = 1
    return f

if __name__=="__main__":
    main()
