from dolfinx import io, fem, mesh, cpp, geometry
import ufl
from mpi4py import MPI
import dolfinx.fem.petsc
import basix.ufl
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def exact_sol(x):
    return 2 -(x[0]**2 + x[1]**2)
def rhs():
    return 4

def interpolate(sending_func,
                receiving_func,):
    '''
    Interpolate sending_func to receiving_func,
    each comming from separate meshes
    '''
    targetSpace = receiving_func.function_space
    nmmid = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
                                 targetSpace.mesh,
                                 targetSpace.element,
                                 sending_func.ufl_function_space().mesh,
                                 padding=0,)
    receiving_func.interpolate(sending_func, nmm_interpolation_data=nmmid)
    return receiving_func

def solve(domain,u):
    '''
    Solve Poisson problem
    '''
    V = u.function_space
    # Dirichlet BC
    cdim = domain.topology.dim
    fdim = cdim-1
    domain.topology.create_connectivity(fdim,cdim)
    bfacets = mesh.exterior_facet_indices(domain.topology)
    u_bc = fem.Function(V)
    u_bc.interpolate(exact_sol)
    dirichlet_bcs = [fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, bfacets))]
    # Forms
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a =  ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx
    l =  rhs()*v*ufl.dx
    problem = dolfinx.fem.petsc.LinearProblem(a,l,
                                              bcs=dirichlet_bcs,)
    return problem.solve()

def write_post(mesh1,mesh2,u1,u2):
    writer1, writer2 = io.VTKFile(mesh1.comm, f"out/result1.vtk",'w'),io.VTKFile(mesh2.comm, f"out/result2.vtk",'w')
    writer1.write_function(u1)
    writer2.write_function(u2)
    writer1.close()
    writer2.close()

def main():
    # Mesh and problems
    points_side = 512
    mesh1 = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.triangle)
    mesh2 = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.quadrilateral)

    (V1, V2) = (fem.functionspace(mesh1, ("Lagrange", 1)),fem.functionspace(mesh2, ("Lagrange", 1)))
    (u1, u2)   = (fem.Function(V1),fem.Function(V2))
    # Solve Poisson problem on mesh1
    u1 = solve(mesh1,u1)
    # Interpolate from mesh1 to mesh2
    interpolate(u1,u2)
    # Write post
    #write_post(mesh1,mesh2,u1,u2)

if __name__=="__main__":
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_function(interpolate)
        lp.add_function(solve)
        lp.add_function(main)
        lp.add_function(write_post)
        lp.add_function(fem.Function.interpolate)
        lp_wrapper = lp(main)
        lp_wrapper()
        lp.print_stats()
