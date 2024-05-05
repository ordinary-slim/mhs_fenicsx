from Problem import Problem, indices_to_function
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
import multiphenicsx
import multiphenicsx.fem.petsc

def exact_sol(x):
    return 2 -(x[0]**2 + x[1]**2)
def exact_sol_ufl(mesh):
    x = ufl.SpatialCoordinate(mesh)
    return exact_sol(x)
def exact_flux_ufl(mesh):
    x = ufl.SpatialCoordinate(mesh)
    return ufl.as_vector((-2*x[0], -2*x[1]))
def rhs():
    return 4

def mesh_rectangle(min_x, min_y, max_x, max_y, elsize, cell_type=mesh.CellType.triangle):
    nx = np.ceil((max_x-min_x)/elsize).astype(np.int32)
    ny = np.ceil((max_y-min_y)/elsize).astype(np.int32)
    return mesh.create_rectangle(MPI.COMM_WORLD,
                                 [np.array([min_x,min_y]),np.array([max_x,max_y])],
                                 [nx,ny],
                                 cell_type,)

def run_conforming():
    el_size = 0.5/4
    mesh_left  = mesh_rectangle(0, 0, 0.5, 1, elsize=el_size)
    p_left  = Problem(mesh_left, name="c_neumann")
    def left_marker_dirichlet(x):
        return np.logical_or( np.isclose(x[1],1), np.logical_or(np.isclose(x[0],0), np.isclose(x[1],0)))
    dirichlet_dofs_left = fem.locate_dofs_geometrical(p_left.v,left_marker_dirichlet)
    p_left.domain.topology.create_connectivity(0,p_left.dim)
    p_left.setDirichletBc(exact_sol, bdofs=dirichlet_dofs_left)
    p_left.setForms(rhs, compileForms=False)

    #n = ufl.FacetNormal(p_left.domain)
    #v = ufl.TestFunction(p_left.v)
    #p_left.l_ufl += +ufl.inner(n, exact_flux_ufl(p_left.domain)) * v * ufl.ds
    right_boundary = mesh.locate_entities(p_left.domain,1,lambda x:np.isclose(x[0],0.5))
    int_entities   = p_left.getGammaIntegralEntities(gammaIndices=right_boundary)
    dS = ufl.Measure('ds', domain=p_left.domain, subdomain_data=[(8,np.asarray(int_entities, dtype=np.int32))])
    n = ufl.FacetNormal(p_left.domain)
    v = ufl.TestFunction(p_left.v)
    neumann_ufl = +ufl.inner(n, exact_flux_ufl(p_left.domain)) * v * dS(8)
    p_left.l_ufl += neumann_ufl

    p_left.compileForms()
    p_left.assemble()
    p_left.solve()
    
    # exact sol
    ex = fem.Function(p_left.v, name="exact")
    ex.interpolate(fem.Expression(exact_sol_ufl(p_left.domain),p_left.v.element.interpolation_points()))
    # dirichlet dofs
    dirichlet_dofs_func = indices_to_function(p_left.v,dirichlet_dofs_left,0)
    dirichlet_dofs_func.name = "dirichet"
    # rhs and diagonal
    l = fem.Function(p_left.v,name="rhs")
    d = fem.Function(p_left.v,name="diag")

    for k,v in p_left.restriction.restricted_to_unrestricted.items():
        l.x.array[v] = p_left.L.array[k]
        d.x.array[v] = p_left.A.getValue(k,k)

    p_left.writepos(extra_funcs=[ex, dirichlet_dofs_func,l,d])

def run_non_conforming():
    el_size = 0.5/4
    mesh_square  = mesh_rectangle(0, 0, 1, 1, elsize=el_size)
    p_left  = Problem(mesh_square, name="nc_neumann")

    active_els_left = fem.locate_dofs_geometrical(p_left.dg0_bg, lambda x : x[0] <= 0.5 )
    p_left.setActivation(active_els_left)

    def left_marker_dirichlet(x):
        return np.logical_or( np.isclose(x[1],1), np.logical_or(np.isclose(x[0],0), np.isclose(x[1],0)))
    dirichlet_dofs_left = fem.locate_dofs_geometrical(p_left.v,left_marker_dirichlet)
    p_left.domain.topology.create_connectivity(0,p_left.dim)
    p_left.setDirichletBc(exact_sol, bdofs=dirichlet_dofs_left)

    right_boundary = mesh.locate_entities(p_left.domain,1,lambda x:np.isclose(x[0],0.5))
    int_entities   = p_left.getGammaIntegralEntities(gammaIndices=right_boundary)

    dS = ufl.Measure('ds', domain=p_left.domain, subdomain_data=[(8,np.asarray(int_entities, dtype=np.int32))])
    n = ufl.FacetNormal(p_left.domain)
    v = ufl.TestFunction(p_left.v)
    p_left.setForms(rhs, compileForms=False) # set forms interior
    neumann_ufl = ufl.inner(n, exact_flux_ufl(p_left.domain)) * v * dS(8)
    p_left.l_ufl += neumann_ufl

    p_left.compileForms()
    p_left.assemble()
    p_left.solve()
    
    # EXTRA POST
    # exact sol
    ex = fem.Function(p_left.v, name="exact")
    ex.interpolate(fem.Expression(exact_sol_ufl(p_left.domain),p_left.v.element.interpolation_points()))
    # dirichlet dofs
    dirichlet_dofs_func = indices_to_function(p_left.v,dirichlet_dofs_left,0)
    dirichlet_dofs_func.name = "dirichet"
    # rfacets
    rbf = indices_to_function(p_left.v,right_boundary,1,name="rfacets")
    # rhs and diag
    l = fem.Function(p_left.v,name="rhs")
    d = fem.Function(p_left.v,name="diag")

    for k,v in p_left.restriction.restricted_to_unrestricted.items():
        l.x.array[v] = p_left.L.array[k]
        d.x.array[v] = p_left.A.getValue(k,k)

    p_left.writepos(extra_funcs=[ex, dirichlet_dofs_func,l,d])

if __name__=="__main__":
    run_conforming()
    run_non_conforming()
