from dolfinx import io, fem, mesh, cpp
import ufl
import numpy as np
from mpi4py import MPI
import multiphenicsx
import multiphenicsx.fem.petsc
import dolfinx.fem.petsc
import petsc4py.PETSc
import pdb

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def exact_sol(x):
    return 2 -(x[0]**2 + x[1]**2)
def rhs():
    return 4

def main():
    # BACKGROUND MESH
    nels_side = 16
    bg_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, nels_side, nels_side, mesh.CellType.quadrilateral)
    cdim = bg_mesh.topology.dim
    fdim = cdim - 1
    
    # SOLUTION SPACES
    Vbg   = fem.functionspace(bg_mesh, ("Lagrange", 1),)
    dg0bg = fem.functionspace(bg_mesh, ("Discontinuous Lagrange", 0),)
    Vactive = Vbg.clone()
    # Activation
    active_dofs        = fem.locate_dofs_geometrical(Vactive, lambda x : x[0] <= 0.5)
    active_els         = np.array(fem.locate_dofs_geometrical(dg0bg, lambda x : x[0] <= 0.5), dtype=np.int32)
    restriction        = multiphenicsx.fem.DofMapRestriction(Vactive.dofmap, active_dofs)
    active_els_tag     = mesh.meshtags(bg_mesh, cdim,
                                       active_els,
                                       np.ones(active_els.shape, dtype=np.int32) )
    #print(f"Rank = {rank}, active_els = {active_els}", flush=True)
    #comm.Barrier()

    # LOCATE ACTIVE BOUNDARY
    bfacets = []
    bg_mesh.topology.create_connectivity(fdim, cdim)
    con_facet_cell = bg_mesh.topology.connectivity(1, 2)
    num_facets_local = bg_mesh.topology.index_map(1).size_local
    for ifacet in range(con_facet_cell.num_nodes):
        local_con = con_facet_cell.links(ifacet)
        incident_active_els = 0
        for el in local_con:
            if (el in active_els):
                incident_active_els += 1
        if (incident_active_els==1) and (ifacet < num_facets_local):
            bfacets.append(ifacet)
    # Optional
    bmesh = mesh.create_submesh(bg_mesh, 1, np.array(bfacets))[0]


    # BC
    bdofs = fem.locate_dofs_topological(Vactive, fdim, bfacets,)
    u_bc = fem.Function(Vactive)
    u_bc.interpolate(exact_sol)
    bc = fem.dirichletbc(u_bc, bdofs)

    # FORMS
    dx = ufl.Measure("dx")(subdomain_data=active_els_tag)
    (x, v) = (ufl.TrialFunction(Vactive),ufl.TestFunction(Vactive))
    a = ufl.dot(ufl.grad(x), ufl.grad(v))*dx
    l = rhs()*v*dx
    a_cpp = fem.form(a)
    l_cpp = fem.form(l)
    A = multiphenicsx.fem.petsc.assemble_matrix(a_cpp,
                                                bcs=[bc],
                                                restriction=(restriction, restriction))
    A.assemble()
    L = multiphenicsx.fem.petsc.assemble_vector(l_cpp,
                                                restriction=restriction,)
    multiphenicsx.fem.petsc.apply_lifting(L, [a_cpp], [[bc]], restriction=restriction,)
    L.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    multiphenicsx.fem.petsc.set_bc(L,[bc],restriction=restriction)

    # SOLVE
    x = multiphenicsx.fem.petsc.create_vector(l_cpp, restriction=restriction)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(bg_mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.solve(L, x)
    x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()

    # GET FUNCTION ON SUBDOMAIN
    usub = fem.Function(Vactive, name="uh")
    with usub.vector.localForm() as usub_vector_local, \
            multiphenicsx.fem.petsc.VecSubVectorWrapper(x, Vactive.dofmap, restriction) as x_wrapper:
                usub_vector_local[:] = x_wrapper
    x.destroy()
    uexact = fem.Function(Vactive, name="uex")
    uexact.interpolate( exact_sol )
    # WRITE
    with io.VTKFile(bg_mesh.comm, "out/res.pvd", "w") as ofile:
        ofile.write_function([usub, uexact])
    with io.XDMFFile(bg_mesh.comm, "out/bmesh.xdmf", "w") as ofile:
        ofile.write_mesh(bmesh)
        #ofile.write_meshtags(active_els_tag, bg_mesh.geometry)

if __name__=="__main__":
    main()
