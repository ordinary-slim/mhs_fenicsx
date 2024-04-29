from __future__ import annotations
from dolfinx import io, fem, mesh, cpp, geometry
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import multiphenicsx
import multiphenicsx.fem.petsc
import dolfinx.fem.petsc
import petsc4py.PETSc
import basix.ufl
from dolfinx import default_scalar_type
import shutil
from abc import ABC, abstractmethod
from mhs_fenicsx import gcode
from mhs_fenicsx.problem.helpers import *
from mhs_fenicsx.problem.heatsource import *
from mhs_fenicsx.problem.material import Material
import sys
sys.path.append('/data0/home/mslimani/cases/fenicsx-cases/mhs_fenicsx/problem/cpp/build')
import cpp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Problem:
    def __init__(self, domain, parameters, name="case"):
        self.domain   = domain
        self.dim = self.domain.topology.dim
        self.name = name
        self.v_bg    = fem.functionspace(domain, ("Lagrange", 1),)
        self.v       = self.v_bg.clone()
        self.dg0_bg  = fem.functionspace(domain, ("Discontinuous Lagrange", 0),)
        self.dg0_vec = fem.functionspace(self.domain,
                                         basix.ufl.element("DG",
                                                           self.domain.basix_cell(),
                                                           0,
                                                           shape=(self.dim,)))
        self.restriction = None

        self.domain.topology.create_entities(self.dim-1)
        self.domain.topology.create_connectivity(self.dim,self.dim)
        self.domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
        # Set num cells per processor
        self.cell_map = self.domain.topology.index_map(self.dim)
        self.facet_map = self.domain.topology.index_map(self.dim-1)
        self.node_map = self.domain.topology.index_map(0)
        self.num_cells = self.cell_map.size_local + self.cell_map.num_ghosts
        self.num_facets = self.facet_map.size_local + self.facet_map.num_ghosts
        self.num_nodes = self.node_map.size_local + self.node_map.num_ghosts
        self.bb_tree = geometry.bb_tree(self.domain,self.dim,padding=1e-7)
        self.set_activation()

        self.u   = fem.Function(self.v, name="uh")   # Solution
        self.u_prev = fem.Function(self.v, name="uh_n") # Previous solution
        self.grad_u = fem.Function(self.dg0_vec,name="grad")
        self.is_grad_computed = False
        self.dirichlet_bcs = []

        # BCs / Interface
        self.gammaNodes = None
        self.neumann_flux = None
        self.dirichlet_gamma = fem.Function(self.v,name="dirichlet_gamma")
        self.is_dirichlet_gamma = False

        # Source term
        if self.dim == 2:
            self.source = Gaussian2D(parameters)
        else:
            self.source = Gaussian3D(parameters)
        self.source_rhs   = fem.Function(self.v, name="source")   # Solution

        # Time
        self.isSteady = parameters["isSteady"]
        self.iter     = 0
        self.time     = 0.0
        self.dt       = fem.Constant(self.domain, parameters["dt"])
        # Material parameters
        self.define_materials(parameters)
        # Integration
        self.quadrature_metadata = {"quadrature_rule":"vertex",
                                    "quadrature_degree":1, }
        self.initialize_post()

    def __del__(self):
        self.writer.close()

    def set_initial_condition( self, expression ):
        try:
            constant = float(expression)
            self.u.x.array[:] = constant
        except TypeError:
            self.u.interpolate(expression)
        self.u_prev.x.array[:] = self.u.x.array[:]
    
    def define_materials(self,parameters):
        self.T_env = parameters["environment_temperature"]
        if "deposition_temperature" in parameters:
            self.T_dep = parameters["deposition_temperature"]
        if "convection_coeff" in parameters:
            self.convection_coeff = fem.Constant(
                    self.domain, PETSc.ScalarType(parameters["convection_coeff"]))
        else:
            self.convection_coeff = None

        self.materials = []
        for key in parameters.keys():
            if key.startswith("material"):
                self.materials.append(Material(parameters[key]))
        assert len(self.materials) > 0, "No materials defined!"
        # All domain starts out covered by material #0
        self.material_id = fem.Function(self.dg0_bg,name="material_id")
        self.material_id.x.array[:] = 0.0
        # Initialize material funcs
        self.k   = fem.Function(self.dg0_bg,name="conductivity")
        self.cp  = fem.Function(self.dg0_bg,name="specific_heat")
        self.rho = fem.Function(self.dg0_bg,name="density")
        self.set_material_funcs()

    def update_material_funcs(self,cells,new_id):
        self.material_id.x.array[cells] = new_id
        self.k.x.array[cells]   = self.materials[new_id].k
        self.rho.x.array[cells] = self.materials[new_id].rho
        self.cp.x.array[cells]  = self.materials[new_id].cp

    def set_material_funcs(self):
        for idx, material in enumerate(self.materials):
            cells = np.flatnonzero(abs(self.material_id.x.array-idx)<1e-7)
            self.k.x.array[cells]   = material.k
            self.rho.x.array[cells] = material.rho
            self.cp.x.array[cells]  = material.cp

    def pre_iterate(self):
        # Pre-iterate source first, current track is tn's
        self.source.pre_iterate(self.time,self.dt.value)
        self.source_rhs.interpolate(self.source)
        self.iter += 1
        self.time += self.dt.value
        self.u_prev.x.array[:] = self.u.x.array[:]
        if rank==0:
            print(f"Problem {self.name} about to solve for iter {self.iter}, time {self.time}.")

    def post_iterate(self):
        pass

    def set_activation(self, active_els=None):
        '''
        TODO: Refactor so that if arg None, do nothing.
        bfacets should be somewhere else possibly
        '''
        if active_els is None:
            active_els = np.arange(self.num_cells,dtype=np.int32)
        self.active_els_tag = mesh.meshtags(self.domain, self.dim,
                                            np.arange(self.num_cells, dtype=np.int32),
                                            get_mask(self.num_cells, active_els),)
        self.active_els_func= indices_to_function(self.dg0_bg,active_els,self.dim,name="active_els")
        try:
            old_active_dofs_array  = self.active_dofs_array.copy()
            self.active_dofs = fem.locate_dofs_topological(self.v, self.dim, active_els,)
            self.active_dofs_array = get_mask(self.num_nodes,self.active_dofs,dtype=np.int32)
            just_active_dofs_array = self.active_dofs_array - old_active_dofs_array
            self.just_activated_nodes = np.flatnonzero(just_active_dofs_array)
        except AttributeError:
            self.active_dofs = fem.locate_dofs_topological(self.v, self.dim, active_els,)
            self.active_dofs_array = get_mask(self.num_nodes,self.active_dofs,dtype=np.int32)

        self.restriction = multiphenicsx.fem.DofMapRestriction(self.v.dofmap, self.active_dofs)
        bfacets_indices  = locate_active_boundary( self.domain, self.active_els_func)
        self.bfacets_tag  = mesh.meshtags(self.domain, self.dim-1,
                                         np.arange(self.num_facets, dtype=np.int32),
                                         get_mask(self.num_facets,
                                                  bfacets_indices, dtype=np.int32),
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

    def get_facet_integrations_entities(self, facet_indices=None):
        facets_integration_ents = []
        if facet_indices is None:
            facet_indices = self.gammaFacets.find(1)
        f_to_c = self.domain.topology.connectivity(self.dim-1,self.dim)
        c_to_f = self.domain.topology.connectivity(self.dim,self.dim-1)
        for ifacet in facet_indices:
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
            facets_integration_ents.append(active_cells[0])
            facets_integration_ents.append(local_index[0])
        return facets_integration_ents

    def set_forms_domain(self):
        dx = ufl.Measure("dx", subdomain_data=self.active_els_tag,
                         metadata=self.quadrature_metadata)
        (u, v) = (ufl.TrialFunction(self.v),ufl.TestFunction(self.v))
        self.a_ufl = self.k*ufl.dot(ufl.grad(u), ufl.grad(v))*dx(1)
        self.l_ufl = self.source_rhs*v*dx(1)
        if not(self.isSteady):
            self.a_ufl += (self.rho*self.cp/self.dt)*u*v*dx(1)
            self.l_ufl += (self.rho*self.cp/self.dt)*self.u_prev*v*dx(1)
    def set_forms_boundary(self):
        '''
        rn must be called after set_forms_domain
        since a_ufl and l_ufl not initialized before
        '''
        boun_integral_entities = self.get_facet_integrations_entities(self.bfacets_tag.find(1))
        ds = ufl.Measure('ds', domain=self.domain, subdomain_data=[
                         (1,np.asarray(boun_integral_entities, dtype=np.int32))],
                         metadata=self.quadrature_metadata)
        (u, v) = (ufl.TrialFunction(self.v),ufl.TestFunction(self.v))
        # CONVECTION
        if self.convection_coeff is not None:
            self.a_ufl += self.convection_coeff * \
                          u*v* \
                          ds(1)
            T_env   = fem.Constant(self.domain, PETSc.ScalarType(self.T_env))
            self.l_ufl += self.convection_coeff * \
                          T_env*v* \
                          ds(1)

    def compile_forms(self):
        self.a_compiled = fem.form(self.a_ufl)
        self.l_compiled = fem.form(self.l_ufl)

    def pre_assemble(self):
        pass

    def assemble(self):
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
        ksp.solve(self.L, self.x)
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
    
    def initialize_post(self):
        self.result_folder = "post"
        shutil.rmtree(self.result_folder,ignore_errors=True)
        self.writer = io.VTKFile(self.domain.comm, f"{self.result_folder}/{self.name}.pvd", "wb")
        self.writer_vtx = io.VTXWriter(self.domain.comm, f"{self.result_folder}/{self.name}.bp",output=[self.u])

    def writepos(self,extra_funcs=[]):
        funcs = [self.u,
                 #self.u_prev,
                 self.active_els_func,
                 self.material_id,
                 self.source_rhs,
                 #self.k,
                 ]
        if self.gammaNodes is not None:
            funcs.append(self.gammaNodes)
        if self.is_grad_computed:
            funcs.append(self.grad_u)
        if self.is_dirichlet_gamma:
            funcs.append(self.dirichlet_gamma)

        #bnodes = indices_to_function(self.v,self.bfacets_tag.find(1),self.dim-1,name="bnodes")
        #funcs.append(bnodes)
        funcs.extend(extra_funcs)
        self.writer.write_function(funcs,t=np.round(self.time,7))

    def writepos_vtx(self):
        self.writer_vtx.write(self.time)

    def clear_dirchlet_bcs(self):
        self.dirichlet_bcs = []
        self.is_dirichlet_gamma = False

    def write_bmesh(self):
        bmesh = dolfinx.mesh.create_submesh(self.domain,self.dim-1,self.bfacets_tag.find(1))[0]
        with io.VTKFile(bmesh.comm, f"out/bmesh_{self.name}.pvd", "w") as ofile:
            ofile.write_mesh(bmesh)

    def l2_norm_gamma( self, f : dolfinx.fem.Function ):
        gamma_ents = self.get_facet_integrations_entities()
        ds_neumann = ufl.Measure('ds', domain=self.domain, subdomain_data=[
            (8,np.asarray(gamma_ents, dtype=np.int32))])
        l_ufl = f*f*ds_neumann(8)
        l2_norm = dolfinx.fem.assemble_scalar(fem.form(l_ufl))
        l2_norm = comm.allreduce(l2_norm, op=MPI.SUM)
        return l2_norm
