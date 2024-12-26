from dolfinx import mesh, fem, io
import multiphenicsx
import multiphenicsx.fem.petsc
import numpy as np
from mpi4py import MPI
import ufl
import petsc4py
from line_profiler import LineProfiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Bcs
def left_marker_dirichlet(x):
    return np.logical_or( np.isclose(x[1],1), np.logical_or(
            np.isclose(x[0],0), np.isclose(x[1],0)) )
def right_marker_dirichlet(x):
    return np.logical_or( np.isclose(x[1],1), np.logical_or(
            np.isclose(x[0],1), np.isclose(x[1],0)) )
def marker_gamma(x):
    return np.isclose( x[0],0.5 )

def compute_interior_facet_integration_data(mesh, facet_indices, value, active_els_mask):
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    integration_data = fem.compute_integration_domains(
        fem.IntegralType.interior_facet, mesh.topology, facet_indices
    )
    is_second_el_active = (active_els_mask[integration_data[2::4]] > 0)
    # Order restriction on one side
    ordered_integration_data = integration_data.reshape(-1, 4).copy()
    if True in is_second_el_active:
        ordered_integration_data[np.ix_(is_second_el_active, [0, 1, 2, 3])] = ordered_integration_data[np.ix_(is_second_el_active, [2, 3, 0, 1])]
    return (value, ordered_integration_data.reshape(-1))

def run_same_mesh_robin_nested():
    els_side = 32
    domain  = mesh.create_unit_square(MPI.COMM_WORLD, els_side, els_side, mesh.CellType.quadrilateral)
    conductivity = 1.0
    def exact_sol(x):
        return 2 -(x[0]**2 + x[1]**2)
    rhs = fem.Constant(domain, np.float64(4*conductivity))

    class Problem:
        def __init__(self,domain:mesh.Mesh,name="problem"):
            self.domain = domain
            self.V      = fem.functionspace(domain, ("Lagrange", 1),)
            self.u      = fem.Function(self.V,name="sol")
            self.DG0    = fem.functionspace(domain, ("Discontinuous Lagrange", 0),)
            self.restriction = None
            self.dir_bcs     = []
            self.writers = {}
            self.name = name
            self.cdim = self.domain.topology.dim
            self.fdim = self.cdim - 1

        def __del__(self):
            for writer in self.writers.values():
                writer.close()

        def dim(self):
            return self.domain.topology.dim

        def set_restriction(self,active_els):
            self.active_dofs = fem.locate_dofs_topological(self.V, self.dim(), active_els,remote=True)
            self.restriction = multiphenicsx.fem.DofMapRestriction(self.V.dofmap, self.active_dofs)
            active_cells_dofs = fem.locate_dofs_topological(self.DG0, self.dim(), active_els,remote=False)
            self.active_els_fun = fem.Function(self.DG0,name="active_els")
            self.active_els_fun.x.array[active_cells_dofs] = 1
            self.local_active_els = self.active_els_fun.x.array.nonzero()[0][:np.searchsorted(self.active_els_fun.x.array.nonzero()[0], self.domain.topology.index_map(self.cdim).size_local)]

        def set_rhs( self, rhs ):
            self.rhs = fem.Constant(self.domain, np.float64(rhs))

        def get_forms(self,dx):
            (u,v) = (ufl.TrialFunction(self.V), ufl.TestFunction(self.V))
            self.a_ufl = conductivity*ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
            self.l_ufl = self.rhs * v * dx
            return self.a_ufl, self.l_ufl

        def writepos(self, extra_funs):
            if not(self.writers):
                self.writers["vtk"] = io.VTKFile(self.domain.comm, f"post_nested/{self.name}.pvd", "w")
            self.writers["vtk"].write_function([self.u,self.active_els_fun]+extra_funs)

    cdim = domain.topology.dim
    fdim = cdim-1
    p_left = Problem(domain,name="left")
    p_right= Problem(domain,name="right")

    active_els = dict()
    active_els[p_left] = fem.locate_dofs_geometrical(p_left.DG0, lambda x : x[0] <= 0.5 )
    active_els[p_right] = fem.locate_dofs_geometrical(p_right.DG0, lambda x : x[0] >= 0.5 )
    gamma_facets = mesh.locate_entities(p_left.domain,fdim,marker_gamma)

    domain.topology.create_connectivity(cdim,cdim)
    domain.topology.create_connectivity(cdim,fdim)
    domain.topology.create_connectivity(fdim,cdim)
    for p, marker in zip([p_left,p_right],[left_marker_dirichlet, right_marker_dirichlet]):
        u_ex = fem.Function(p.V,name="exact")
        u_ex.interpolate(exact_sol)
        bdofs_dir  = fem.locate_dofs_geometrical(p.V,marker)
        p.dir_bcs = [fem.dirichletbc(u_ex, bdofs_dir)]
        p.set_rhs(rhs)
        p.set_restriction(active_els[p])

    # Subdomain data 
    subdomain_data = [(1,p_left.local_active_els), (2, p_right.local_active_els)]
    gamma_tag = 8
    gamma_integration_data = compute_interior_facet_integration_data(p_left.domain,
                                                                gamma_facets,
                                                                gamma_tag,
                                                                p_left.active_els_fun.x.array)
    dx = ufl.Measure("dx")(subdomain_data=subdomain_data)
    dS = ufl.Measure("dS")(domain=p_left.domain,
                           subdomain_data=[gamma_integration_data])

    a_left, l_left = p_left.get_forms(dx(1))
    a_right, l_right = p_right.get_forms(dx(2))

    # Solve problem
    bcs =  p_right.dir_bcs + p_left.dir_bcs
    restriction = [p_left.restriction, p_right.restriction]
    a = [[a_left, None],[None, a_right]]
    l = [l_left, l_right]
    # Set Robin-Robin coupling
    (ul, vl) = (ufl.TrialFunction(p_left.V), ufl.TestFunction(p_left.V))
    (ur, vr) = (ufl.TrialFunction(p_right.V), ufl.TestFunction(p_right.V))
    n = ufl.FacetNormal(p_left.domain)
    a[0][0] += ul("+") * vl("+") * dS(gamma_tag)
    a[0][1]  = - (ur("-") + conductivity * ufl.inner( ufl.grad(ur)("-"), n("+") )) * vl("+") * dS(gamma_tag)
    a[1][1] += ur("-") * vr("-") * dS(gamma_tag)
    a[1][0]  = - (ul("+") + conductivity * ufl.inner( ufl.grad(ul)("+"), n("-") )) * vr("-") * dS(gamma_tag)

    a_cpp = fem.form(a)
    l_cpp = fem.form(l)

    # Assemble the block linear system
    As = [[], []]
    ps = [p_left, p_right]
    for i in range(2):
        for j in range(2):
            As[i].append(multiphenicsx.fem.petsc.assemble_matrix(a_cpp[i][j], bcs=bcs, restriction=(ps[i].restriction, ps[j].restriction)))
    A = petsc4py.PETSc.Mat().createNest(As)
    A.assemble()

    bcs_by_block = fem.bcs_by_block(a_cpp[0][1].function_spaces, bcs)
    for i, p in enumerate([p_left, p_right]):
        p.lc = fem.form(p.l_ufl)
        p.L  = multiphenicsx.fem.petsc.assemble_vector(p.lc, restriction=p.restriction)
        multiphenicsx.fem.petsc.apply_lifting(p.L, a_cpp[i], bcs_by_block, restriction=p.restriction)
        p.L.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD_VALUES, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        multiphenicsx.fem.petsc.set_bc(p.L, bcs=p.dir_bcs, restriction=p.restriction)
        p.L.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT_VALUES, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    L = petsc4py.PETSc.Vec().createNest([p_left.L, p_right.L])

    # Solve
    ulur = multiphenicsx.fem.petsc.create_vector_nest(l_cpp, restriction=restriction)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=7, ival=4)
    ksp.setFromOptions()
    ksp.solve(L, ulur)
    for ulur_sub in ulur.getNestSubVecs():
        ulur_sub.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()

    # Split the block solution in components
    with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(
            ulur, [p_left.V.dofmap, p_right.V.dofmap], restriction) as ulur_wrapper:
        for ulur_wrapper_local, component in zip(ulur_wrapper, (p_left.u, p_right.u)):
            with component.x.petsc_vec.localForm() as component_local:
                component_local[:] = ulur_wrapper_local
    ulur.destroy()
    for mat in [L, A]:
        mat.destroy()

    for p in [p_left,p_right]:
        p.writepos(extra_funs=[p.dir_bcs[0].g])


if __name__=="__main__":
    lp = LineProfiler()
    lp_wrapper = lp(run_same_mesh_robin_nested)
    lp_wrapper()
