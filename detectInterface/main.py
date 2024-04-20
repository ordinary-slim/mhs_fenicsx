from mpi4py import MPI
import numpy as np
from dolfinx import mesh, io, fem
import dolfinx.fem.petsc
import basix.ufl
import ufl
import pdb

def project( projectedFunction, targetSpace, projection ):
    # Compute function for fine grid at quadrature points of the coarse grid
    degree = 1
    Qe = basix.ufl.quadrature_element(
        targetSpace.mesh.topology.cell_name(), degree=degree)
    targetQuadratureSpace = fem.functionspace(targetSpace.mesh, Qe)
    nmmid = fem.create_nonmatching_meshes_interpolation_data(
                                 targetQuadratureSpace.mesh._cpp_object,
                                 targetQuadratureSpace.element,
                                 projectedFunction.ufl_function_space().mesh._cpp_object,
                                 padding=1e-5,)
    q_func = fem.Function(targetQuadratureSpace)
    q_func.interpolate(projectedFunction, nmm_interpolation_data=nmmid)
    # VARIATIONAL PROBLEM
    dx = ufl.Measure(
            "dx",
            #domain = targetSpace.ufl_domain(),
            #metadata={"quadrature_rule":"vertex",
                      #"quadrature_degree":1,
                      #}
            )
    u, v = ufl.TrialFunction(targetSpace), ufl.TestFunction(targetSpace)
    a = ufl.inner( u, v ) * dx
    L = ufl.inner( q_func, v ) * dx
    problem = dolfinx.fem.petsc.LinearProblem(a, L, u = projection )
    problem.solve()

def mesh_rectangle( box ):
    L = box[2] - box[0]
    H = box[3] - box[1]
    mesh_den = 4
    nx = int(np.round(L*mesh_den*10)/10)
    ny = int(np.round(H*mesh_den*10)/10)
    return mesh.create_rectangle(MPI.COMM_WORLD,
         [np.array(box[:2]), np.array(box[2:])],
         [nx, ny],
         mesh.CellType.quadrilateral,
         )

meshBig = mesh_rectangle( [-2, -2, 2, 2] ) 
meshSmall = mesh_rectangle( [-1, -1, 1, 1] ) 

dg0small = fem.functionspace(meshSmall, ("Discontinuous Lagrange", 0))
active_els_small = fem.Function( dg0small )
active_els_small.x.array.fill( 33 )

dg0big = fem.functionspace(meshBig, ("Discontinuous Lagrange", 0))
small_from_big = fem.Function( dg0big )
project(active_els_small, dg0big, small_from_big )

nelsBig = meshBig.topology.index_map(2).size_local
mt = mesh.meshtags( meshBig, meshBig.topology.dim,
                   range(nelsBig), np.array(small_from_big.x.array).astype(np.int32), )

with io.XDMFFile(meshSmall.comm, "out/center.xdmf", "w") as xdmf:
    xdmf.write_mesh(meshSmall)
    xdmf.write_function(active_els_small)
    #xdmf.write_meshtags( mt, meshSmall.geometry,)

with io.XDMFFile(meshBig.comm, "out/outside.xdmf", "w") as xdmf:
    xdmf.write_mesh(meshBig)
    xdmf.write_meshtags(mt, meshBig.geometry)
    #xdmf.write_function(small_from_big)
