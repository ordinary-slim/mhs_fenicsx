from __future__ import annotations
from dolfinx import io, fem, mesh, cpp, geometry
import ufl
import numpy as np
from mpi4py import MPI
import multiphenicsx
import multiphenicsx.fem.petsc
import dolfinx.fem.petsc
import petsc4py.PETSc
import basix.ufl
from dolfinx import default_scalar_type

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def interpolate(func2project,
                targetSpace,
                interpolate,):
    nmmid = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
                                 targetSpace.mesh,
                                 targetSpace.element,
                                 func2project.ufl_function_space().mesh,
                                 padding=1e-6,)
    interpolate.interpolate(func2project, nmm_interpolation_data=nmmid)
    return interpolate

def l2_squared(f : dolfinx.fem.Function,active_els_tag):
    dx = ufl.Measure("dx")(subdomain_data=active_els_tag)
    l_ufl = f*f*dx(1)
    l2_norm = dolfinx.fem.assemble_scalar(fem.form(l_ufl))
    l2_norm = comm.allreduce(l2_norm, op=MPI.SUM)
    return l2_norm

def locate_active_boundary(mesh, active_els):
    bfacets = []
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    con_facet_cell = mesh.topology.connectivity(mesh.topology.dim-1, mesh.topology.dim)
    num_facets_local = mesh.topology.index_map(mesh.topology.dim-1).size_local
    for ifacet in range(con_facet_cell.num_nodes):
        local_con = con_facet_cell.links(ifacet)
        incident_active_els = 0
        for el in local_con:
            if (el in active_els):
                incident_active_els += 1
        if (incident_active_els==1) and (ifacet < num_facets_local):
            bfacets.append(ifacet)
    return bfacets

def get_mask(size, indices, dtype=np.int32):
    mask = np.zeros(size, dtype=dtype)
    mask[indices] = 1
    return mask

def interpolate_dg_at_facets(f,facets,targetSpace,bb_tree_ext,
                             activation_tag,
                             ext_activation_tag,
                             name="flux"):
    interpolated_f = fem.Function(targetSpace,name=name)
    domain           = targetSpace.mesh
    ext_domain       = f.function_space.mesh
    cdim = domain.topology.dim
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
    local_vals  = np.zeros((total_facet_count,cdim),dtype=np.double,order='C')
    global_vals = np.zeros((total_facet_count,cdim),dtype=np.double,order='C')
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
        flat_idx    = el*interpolated_f.ufl_shape[0]
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

class Problem:
    def __init__(self, mesh, name="case"):
        self.domain   = mesh
        self.name = name
        # Initializations
        self.v_bg   = fem.functionspace(mesh, ("Lagrange", 1),)
        self.v      = self.v_bg.clone()
        self.dg0_bg = fem.functionspace(mesh, ("Discontinuous Lagrange", 0),)
        dg0_dim2_el = basix.ufl.element("DG", self.domain.basix_cell(), 0, shape=(2,))
        self.dg0_dim2    = fem.functionspace(self.domain, dg0_dim2_el)
        self.restriction = None
        self.dim = self.domain.topology.dim

        # Set num cells per processor
        self.domain.topology.create_entities(self.dim-1)
        self.cell_map = self.domain.topology.index_map(self.dim)
        self.facet_map = self.domain.topology.index_map(self.dim-1)
        self.num_cells = self.cell_map.size_local + self.cell_map.num_ghosts
        self.num_facets = self.facet_map.size_local + self.facet_map.num_ghosts
        self.bb_tree = geometry.bb_tree(self.domain,self.dim,padding=1e-7)
        self.set_activation()

        self.u = fem.Function(self.v, name="uh") # Solution
        self.grad_u = fem.Function(self.dg0_dim2,name="grad")
        self.is_grad_computed = False
        self.dirichlet_bcs = []
        self.neumann_flux = None
        self.dirichlet_gamma = fem.Function(self.v,name="dirichlet_gamma")
        self.is_dirichlet_gamma = False

        self.time = 0.0
        self.writer = io.VTKFile(self.domain.comm, f"out/results_{self.name}.pvd", "wb")

    def __del__(self):
        self.writer.close()

    def set_activation(self, active_els=None):
        if active_els is None:
            active_els = np.arange( self.num_cells, dtype=np.int32 )
        self.active_els_tag = mesh.meshtags(self.domain, self.dim,
                                            np.arange(self.num_cells, dtype=np.int32),
                                            get_mask(self.num_cells, active_els),)
        self.domain.topology.create_connectivity(self.dim,self.dim)
        self.active_els_func= indices_to_function(self.dg0_bg,active_els,self.dim,name="active_els")
        self.active_dofs = fem.locate_dofs_topological(self.v, self.dim, active_els,)
        self.restriction = multiphenicsx.fem.DofMapRestriction(self.v.dofmap, self.active_dofs)
        self.bfacets_tag  = mesh.meshtags(self.domain, self.dim-1,
                                         np.arange(self.num_facets, dtype=np.int32),
                                         get_mask(self.num_facets, locate_active_boundary( self.domain, active_els ), dtype=np.int32),
                                         )
    def compute_gradient(self):
        if not(self.is_grad_computed):
            self.grad_u.interpolate( fem.Expression(ufl.grad(self.u),self.grad_u.function_space.element.interpolation_points()) )
            self.is_grad_computed = True

    def get_active_in_external(self, p_ext:Problem ):
        '''
        Return nodal function with nodes active in p_ext
        '''
        # Get function on p_ext
        active_dofs_ext_func_ext = fem.Function( p_ext.v )
        active_dofs_ext_func_ext.x.array[p_ext.active_dofs] = 1.0
        # Interpolate to nodes of self
        nmmid = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
                                     self.domain,
                                     self.v.element,
                                     active_dofs_ext_func_ext.function_space.mesh,
                                     padding=1e-5,)
        active_dofs_ext_func_self = dolfinx.fem.Function(self.v_bg,
                                                         name="active_nodes_ext",)
        active_dofs_ext_func_self.interpolate(active_dofs_ext_func_ext, nmm_interpolation_data=nmmid)
        np.round(active_dofs_ext_func_self.x.array,decimals=7,out=active_dofs_ext_func_self.x.array)
        return active_dofs_ext_func_self

    def find_gamma(self, p_ext:Problem):
        gammaFacets = []
        ext_active_dofs_func = self.get_active_in_external( p_ext )
        ext_active_dofs_func.name = "ext act dofs"
        # Loop over boundary facets, get incident nodes,
        # if all nodes of facet are active in external --> gamma facet
        self.domain.topology.create_connectivity(self.dim-1, 0)
        con_facet_nodes = self.domain.topology.connectivity(self.dim-1, 0)
        for ifacet in self.bfacets_tag.find(1):
            local_con = con_facet_nodes.links(ifacet)
            local_con_global = self.domain.topology.index_map(0).local_to_global(local_con)
            local_con_space = self.v_bg.dofmap.index_map.global_to_local(local_con_global)
            all_nodes_active = True
            for inode in local_con_space:
                if not(ext_active_dofs_func.x.array[inode]):
                    all_nodes_active = False
                    break
            if all_nodes_active:
                gammaFacets.append(ifacet)
        self.gammaFacets = mesh.meshtags(self.domain, self.dim-1,
                                         np.arange(self.num_facets, dtype=np.int32),
                                         get_mask(self.num_facets, gammaFacets),)
        self.gammaNodes = indices_to_function(self.v,
                                         self.gammaFacets.find(1),
                                         self.dim-1,
                                         name="gammaNodes",)



    def add_dirichlet_bc(self, func, bdofs=None, bfacets_tag=None, marker=None, reset=False):
        if reset:
            self.dirichlet_bcs = []
        if bdofs is None:
            if bfacets_tag is None:
                if marker==None:
                    bfacets_tag = self.bfacets_tag
                    bdofs = fem.locate_dofs_topological(self.v, self.dim-1, bfacets_tag.find(1),)
                else:
                    bdofs  = fem.locate_dofs_geometrical(self.v,marker)
        u_bc = fem.Function(self.v)
        u_bc.interpolate(func)
        self.dirichlet_bcs.append(fem.dirichletbc(u_bc, bdofs))

    def compute_gamma_integration_ents(self, gammaIndices=None):
        gamma_integration_ents = []
        if gammaIndices is None:
            gammaIndices = self.gammaFacets.find(1)
        f_to_c = self.domain.topology.connectivity(self.dim-1,self.dim)
        c_to_f = self.domain.topology.connectivity(self.dim,self.dim-1)
        for ifacet in gammaIndices:
            if ifacet >= self.facet_map.size_local:
                continue
            # Find cells connected to facet
            cells = f_to_c.links(ifacet)
            # Get correct cell from activation
            active_cells = cells[np.flatnonzero(self.active_els_tag.values[cells])]

            assert len(active_cells) == 1
            # Get local index of ifacet
            local_facets = c_to_f.links(active_cells[0])
            local_index = np.flatnonzero(local_facets == ifacet)
            assert len(local_index) == 1

            # Append integration entities
            gamma_integration_ents.append(active_cells[0])
            gamma_integration_ents.append(local_index[0])
        return gamma_integration_ents

    def set_forms_domain(self, rhs, potential_ext=None, flux=None):
        dx = ufl.Measure("dx")(subdomain_data=self.active_els_tag)
        (x, v) = (ufl.TrialFunction(self.v),ufl.TestFunction(self.v))
        self.a_ufl = ufl.dot(ufl.grad(x), ufl.grad(v))*dx(1)
        self.l_ufl = fem.Constant(self.domain,default_scalar_type(0))*v*dx(1)
        self.l_ufl += rhs()*v*dx(1)

    def compile_forms(self):
        self.a_compiled = fem.form(self.a_ufl)
        self.l_compiled = fem.form(self.l_ufl)

    def assemble(self):
        self.compile_forms()
        self.A = multiphenicsx.fem.petsc.assemble_matrix(self.a_compiled,
                                                    bcs=self.dirichlet_bcs,
                                                    restriction=(self.restriction, self.restriction))
        self.A.assemble()
        self.L = multiphenicsx.fem.petsc.assemble_vector(self.l_compiled,
                                                    restriction=self.restriction,)
        multiphenicsx.fem.petsc.apply_lifting(self.L, [self.a_compiled], [self.dirichlet_bcs], restriction=self.restriction,)
        self.L.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        multiphenicsx.fem.petsc.set_bc(self.L,self.dirichlet_bcs,restriction=self.restriction)

    def _solveLinearSystem(self):
        self.x = multiphenicsx.fem.petsc.create_vector(self.l_compiled, restriction=self.restriction)
        ksp = petsc4py.PETSc.KSP()
        ksp.create(self.domain.comm)
        ksp.setOperators(self.A)
        ksp.setType("cg")
        #ksp.getPC().setType("lu")
        #ksp.getPC().setFactorSolverType("mumps")
        #ksp.setFromOptions()
        ksp.setTolerances(rtol=1e-10)
        ksp.solve(self.L, self.x)
        print(f"Residual norm = {ksp.norm}, # iterations : {ksp.its}")
        self.x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        ksp.destroy()

    def _restrictSolution(self):
        with self.u.vector.localForm() as usub_vector_local, \
                multiphenicsx.fem.petsc.VecSubVectorWrapper(self.x, self.v.dofmap, self.restriction) as x_wrapper:
                    usub_vector_local[:] = x_wrapper
        self.x.destroy()

    def solve(self):
        self._solveLinearSystem()
        self._restrictSolution()
        self.is_grad_computed   = False

    def writepos(self,extra_funcs=[]):
        funcs = [self.u, self.active_els_func,self.gammaNodes]
        if self.is_grad_computed:
            funcs.append(self.grad_u)
        if self.is_dirichlet_gamma:
            funcs.append(self.dirichlet_gamma)
        funcs.extend(extra_funcs)
        self.writer.write_function(funcs,t=self.time)

    def clear_dirchlet_bcs(self):
        self.dirichlet_bcs = []
        self.is_dirichlet_gamma = False

    def write_bmesh(self):
        bmesh = dolfinx.mesh.create_submesh(self.domain,self.dim-1,self.bfacets_tag.find(1))[0]
        with io.VTKFile(bmesh.comm, f"out/bmesh_{self.name}.pvd", "w") as ofile:
            ofile.write_mesh(bmesh)

    def l2_norm_gamma( self, f : dolfinx.fem.Function ):
        gamma_ents = self.compute_gamma_integration_ents()
        ds_neumann = ufl.Measure('ds', domain=self.domain, subdomain_data=[
            (8,np.asarray(gamma_ents, dtype=np.int32))])
        l_ufl = f*f*ds_neumann(8)
        l2_norm = dolfinx.fem.assemble_scalar(fem.form(l_ufl))
        l2_norm = comm.allreduce(l2_norm, op=MPI.SUM)
        return l2_norm
