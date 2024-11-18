from dolfinx import mesh, fem
from mpi4py import MPI
import basix.ufl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main():
    nelems_side = 2
    domain = mesh.create_unit_square(comm, nelems_side, nelems_side,)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    bmesh = mesh.create_submesh(domain,fdim,boundary_facets)[0]
    Qe = basix.ufl.quadrature_element(domain.topology.entity_types[-2][0].name,degree=2)
    Ve = fem.functionspace(bmesh,Qe)
    import pdb
    pdb.set_trace()

if __name__=="__main__":
    main()
