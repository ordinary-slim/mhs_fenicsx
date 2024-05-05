from dolfinx import mesh, io, fem
from mpi4py import MPI

nels_per_side = 2
mesh  = mesh.create_unit_square(MPI.COMM_WORLD, nels_per_side, nels_per_side, mesh.CellType.quadrilateral)

V = fem.functionspace(mesh, ("Lagrange", 1))
D = fem.functionspace(mesh, ("DG",0))

writer = io.XDMFFile(mesh.comm,"test_multiple_calls.xdmf", 'w')
writer.write_mesh(mesh)

funcs = []
for it in range(10):
    funcs.append(fem.Function(V, name = f"u{it}"))
    funcs[it].interpolate( lambda x : x[0]**2 )
    writer.write_function(funcs[it])
writer.close()

'''
writer = io.VTXWriter(mesh.comm,"test.bp",[u1,u2])
writer.write(0.0)
writer.close()
'''
