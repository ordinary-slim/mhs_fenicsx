from __future__ import annotations
from dolfinx import io, fem, mesh, cpp, geometry, la
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
import mhs_fenicsx_cpp
import typing

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Problem:
    def __init__(self, domain, parameters, name="case"):
        self.input_parameters = parameters.copy()
        self.domain   = domain
        self.dim = self.domain.topology.dim
        self.name = name
        # Function spaces
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
        self.set_bb_trees()
        self.initialize_activation()

        self.u   = fem.Function(self.v, name="uh")   # Solution
        self.u_prev = fem.Function(self.v, name="uh_n") # Previous solution
        self.grad_u = fem.Function(self.dg0_vec,name="grad")
        self.is_grad_computed = False
        self.dirichlet_bcs = []

        # BCs / Interface
        self.gamma_nodes = None
        self.neumann_flux = None
        self.dirichlet_gamma = fem.Function(self.v,name="dirichlet_gamma")
        self.is_dirichlet_gamma = False

        # Source term
        if self.dim == 1:
            self.source = Gaussian1D(parameters)
        elif self.dim == 2:
            self.source = Gaussian2D(parameters)
        else:
            self.source = Gaussian3D(parameters)
        self.rhs = None # For python functions
        self.source_rhs   = fem.Function(self.v, name="source")   # For moving hs

        # Time
        self.is_steady = parameters["isSteady"]
        self.iter     = 0
        self.time     = 0.0
        self.dt       = fem.Constant(self.domain, parameters["dt"]) if not(self.is_steady) else -1
        # Motion
        self.domain_speed     = np.array(parameters["domain_speed"]) if "domain_speed" in parameters else None
        advection_speed = parameters["advection_speed"][:self.domain.topology.dim] if "advection_speed" in parameters else np.zeros(self.domain.topology.dim)
        self.advection_speed = fem.Constant(self.domain,advection_speed)
        # Stabilization
        self.is_supg = False
        if ("supg" in parameters):
            self.is_supg = bool(parameters["supg"])
        self.supg_elwise_coeff = fem.Function(self.dg0_bg,name="supg_tau") if self.is_supg else None
        # Material parameters
        self.define_materials(parameters)
        # Integration
        self.quadrature_metadata = {"quadrature_rule":"vertex",
                                    "quadrature_degree":1, }
        self.initialize_post()
        self.set_linear_solver(parameters["petsc_opts"] if "petsc_opts" in parameters else None)

    def __del__(self):
        self.writer.close()
        self.writer_vtx.close()

    def set_initial_condition( self, expression ):
        try:
            constant = float(expression)
            self.u.x.array[:] = constant
        except TypeError:
            self.u.interpolate(expression)
        self.u_prev.x.array[:] = self.u.x.array[:]

    def set_rhs( self, rhs ):
        self.rhs = rhs

    def set_bb_trees(self):
        self.bb_tree = geometry.bb_tree(self.domain,self.dim,np.arange(self.num_cells,dtype=np.int32),padding=1e-7)
        self.bb_tree_nodes = geometry.bb_tree(self.domain,0,np.arange(self.num_nodes,dtype=np.int32),padding=1e-7)
    
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

    def compute_supg_coeff(self):
        if self.supg_elwise_coeff is None:
            self.supg_elwise_coeff = fem.Function(self.dg0_bg,name="supg_tau")
        mhs_fenicsx_cpp.compute_el_size_along_vector(self.supg_elwise_coeff._cpp_object,self.advection_speed._cpp_object)
        advection_norm = np.linalg.norm(self.advection_speed.value)
        self.supg_elwise_coeff.x.array[:] = self.supg_elwise_coeff.x.array[:]**2 / (
                2 * self.supg_elwise_coeff.x.array[:] * self.rho.x.array[:] * self.cp.x.array[:]*advection_norm + \
                 4 * self.k.x.array[:])

    def pre_iterate(self):
        # Pre-iterate source first, current track is tn's
        if rank==0:
            print(f"\nProblem {self.name} about to solve for iter {self.iter+1}, time {self.time+self.dt.value}")
        self.source.pre_iterate(self.time,self.dt.value)
        # Mesh motion
        if self.domain_speed is not None:
            self.domain.geometry.x[:] += np.round(self.domain_speed*self.dt.value,7)
            # This can be done more efficiently C++ level
            self.set_bb_trees()

        self.source_rhs.interpolate(self.source)
        self.iter += 1
        self.time += self.dt.value
        self.u_prev.x.array[:] = self.u.x.array

    def post_iterate(self):
        self._destroy()
        self.has_preassembled = False

    def initialize_activation(self):
        self.active_els_tag = mesh.meshtags(self.domain, self.dim,
                                            np.arange(self.num_cells, dtype=np.int32),
                                            np.ones(self.num_cells,dtype=np.int32))
        self.active_els_func= fem.Function(self.dg0_bg,name="active_els")
        self.active_els_func.x.array[:] = 1.0
        self.active_nodes_func = fem.Function(self.v,name="active_nodes")
        self.active_nodes_func.x.array[:] = 1.0
        self.active_dofs = np.arange(self.num_nodes,dtype=np.int32)
        self.restriction = multiphenicsx.fem.DofMapRestriction(self.v.dofmap, self.active_dofs)
        self.update_boundary()

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
        indices_to_function(self.dg0_bg,active_els,self.dim,name="active_els",remote=False, f=self.active_els_func)
        #self.active_els_func.x.scatter_forward()

        old_active_dofs_array  = self.active_nodes_func.x.array.copy()
        self.active_dofs = fem.locate_dofs_topological(self.v, self.dim, active_els,remote=False)
        self.active_nodes_func.x.array[:] = np.float64(0.0)
        self.active_nodes_func.x.array[self.active_dofs] = np.float64(1.0)
        #self.active_nodes_func.x.scatter_forward()
        just_active_dofs_array = self.active_nodes_func.x.array - old_active_dofs_array
        self.just_activated_nodes = np.flatnonzero(just_active_dofs_array)

        self.restriction = multiphenicsx.fem.DofMapRestriction(self.v.dofmap, self.active_dofs)
        self.update_boundary()

    def update_boundary(self):
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
        cells = np.arange(self.num_cells,dtype=np.int32)
        nmmid = dolfinx.fem.create_interpolation_data(
                                     self.v,
                                     p_ext.v,
                                     cells,
                                     padding=1e-5,)
        active_dofs_ext_func_self = dolfinx.fem.Function(self.v_bg,
                                                         name="active_nodes_ext",)
        active_dofs_ext_func_self.interpolate_nonmatching(active_dofs_ext_func_ext,
                                                          cells=cells,
                                                          interpolation_data=nmmid,)
        np.round(active_dofs_ext_func_self.x.array,decimals=7,out=active_dofs_ext_func_self.x.array)
        active_dofs_ext_func_self.x.scatter_forward()
        return active_dofs_ext_func_self

    def subtract_problem(self,p_ext:Problem):
        self.ext_nodal_activation = self.get_active_in_external(p_ext)
        self.ext_nodal_activation.name = "ext_act"
        ext_active_nodes = self.ext_nodal_activation.x.array.nonzero()[0]
        incident_cells = mesh.compute_incident_entities(self.domain.topology,
                                                        ext_active_nodes,
                                                        0,
                                                        self.domain.topology.dim,)
        active_els_mask = la.vector(self.cell_map,1,dtype=np.int32)
        active_els_mask.array[:] = np.int32(1)
        for cell in incident_cells:
            if cell >= self.cell_map.size_local:
                continue
            all_active_in_ext = True
            for idof in self.v.dofmap.cell_dofs(cell):
                if self.ext_nodal_activation.x.array[idof]==0:
                    all_active_in_ext = False
                    break
            if all_active_in_ext:
                active_els_mask.array[cell] = 0
        active_els_mask.scatter_forward()
        active_els = active_els_mask.array.nonzero()[0]
        self.set_activation(active_els)
        self.find_gamma(self.ext_nodal_activation)
        active_els_mask.petsc_vec.destroy()

    def find_gamma(self,ext_active_dofs_func):
        loc_gamma_facets = []
        ghost_gamma_facets = []
        all_gamma_facets = []
        #ext_active_dofs_func = self.get_active_in_external( p_ext )
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
                all_gamma_facets.append(ifacet)
                if ifacet < self.facet_map.size_local:
                    loc_gamma_facets.append(ifacet)
                else:
                    ghost_gamma_facets.append(ifacet)
        self.gamma_facets = mesh.meshtags(self.domain, self.dim-1,
                                         np.arange(self.num_facets, dtype=np.int32),
                                         get_mask(self.num_facets, [loc_gamma_facets,ghost_gamma_facets], val=[1,2]),)
        self.gamma_nodes = indices_to_function(self.v,
                                         self.gamma_facets.find(1),
                                         self.dim-1,
                                         name="gammaNodes",)
        self.gamma_facets_index_map, \
        gamma_imap_to_global_imap = cpp.common.create_sub_index_map(self.facet_map,
                                                    np.array(all_gamma_facets,dtype=np.int32),
                                                    False)
        self.gamma_imap_to_global_imap = mhs_fenicsx_cpp.int_map()
        for i in range(len(gamma_imap_to_global_imap)):
            self.gamma_imap_to_global_imap[gamma_imap_to_global_imap[i]] = i

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
        bc = fem.dirichletbc(u_bc, bdofs)
        self.dirichlet_bcs.append(bc)
        return bc

    def get_facet_integrations_entities(self, facet_indices=None):
        if facet_indices is None:
            facet_indices = self.gamma_facets.find(1)
        return get_facet_integration_entities(self.domain,facet_indices,self.active_els_func)

    def set_forms_domain(self):
        dx = ufl.Measure("dx", subdomain_data=self.active_els_tag,
                         metadata=self.quadrature_metadata,
                         )
        (u, v) = (ufl.TrialFunction(self.v),ufl.TestFunction(self.v))
        self.a_ufl = self.k*ufl.dot(ufl.grad(u), ufl.grad(v))*dx(1)
        if self.rhs is not None:
            self.source_rhs.interpolate(self.rhs)
        self.l_ufl = self.source_rhs*v*dx(1)
        if not(self.is_steady):
            self.a_ufl += (self.rho*self.cp/self.dt)*u*v*dx(1)
            self.l_ufl += (self.rho*self.cp/self.dt)*self.u_prev*v*dx(1)
        if np.linalg.norm(self.advection_speed.value):
            self.a_ufl += self.rho*self.cp*ufl.dot(self.advection_speed,ufl.grad(u))*v*dx(1)
            if self.is_supg:
                self.compute_supg_coeff()
                if not(self.is_steady):
                    self.a_ufl += (self.rho * self.cp / self.dt) * u * self.supg_elwise_coeff * \
                                    self.rho * self.cp * ufl.dot(self.advection_speed,ufl.grad(v)) * dx(1)
                    self.l_ufl += (self.rho * self.cp / self.dt) * self.u_prev * self.supg_elwise_coeff * \
                                    self.rho * self.cp * ufl.dot(self.advection_speed,ufl.grad(v)) * dx(1)
                self.a_ufl += (self.rho * self.cp) * ufl.dot(self.advection_speed,ufl.grad(u)) * \
                                self.supg_elwise_coeff * self.rho * self.cp * ufl.dot(self.advection_speed,ufl.grad(v)) * dx(1)
                self.l_ufl += self.source_rhs * \
                        self.supg_elwise_coeff * self.rho * self.cp * ufl.dot(self.advection_speed,ufl.grad(v)) * dx(1)

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
        self.A = multiphenicsx.fem.petsc.create_matrix(self.a_compiled,
                                                  (self.restriction, self.restriction),
                                                  )
        self.L = multiphenicsx.fem.petsc.create_vector(self.l_compiled,
                                                  self.restriction)
        self.x = multiphenicsx.fem.petsc.create_vector(self.l_compiled, restriction=self.restriction)
        self.has_preassembled = True

    def assemble(self):
        if not(self.has_preassembled):
            self.pre_assemble()
        self.A.zeroEntries()
        multiphenicsx.fem.petsc.assemble_matrix(self.A,
                                                self.a_compiled,
                                                bcs=self.dirichlet_bcs,
                                                restriction=(self.restriction, self.restriction))
        self.A.assemble()
        with self.L.localForm() as l_local:
            l_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector(self.L,
                                                self.l_compiled,
                                                restriction=self.restriction,)
        multiphenicsx.fem.petsc.apply_lifting(self.L, [self.a_compiled], [self.dirichlet_bcs], restriction=self.restriction,)
        self.L.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        multiphenicsx.fem.petsc.set_bc(self.L,self.dirichlet_bcs,restriction=self.restriction)

    def set_linear_solver(self, opts:typing.Optional[dict] = None):
        if opts is None:
            opts = {"pc_type" : "lu", "pc_factor_mat_solver_type" : "mumps",}
        self.linear_solver_opts = dict(opts)

    def _solve_linear_system(self):
        with self.x.localForm() as x_local:
            x_local.set(0.0)
        ksp = petsc4py.PETSc.KSP()
        ksp.create(self.domain.comm)
        ksp.setOperators(self.A)
        ksp_opts = PETSc.Options()
        for k,v in self.linear_solver_opts.items():
            ksp_opts[k] = v
        ksp.setFromOptions()
        ksp.solve(self.L, self.x)
        self.x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        ksp.destroy()

    def _restrict_solution(self):
        with self.u.vector.localForm() as usub_vector_local, \
                multiphenicsx.fem.petsc.VecSubVectorWrapper(self.x, self.v.dofmap, self.restriction) as x_wrapper:
                    usub_vector_local[:] = x_wrapper
        self.u.x.scatter_forward()

    def _destroy(self):
        self.x.destroy()
        self.A.destroy()
        self.L.destroy()

    def solve(self):
        self._solve_linear_system()
        self._restrict_solution()
        self.is_grad_computed   = False
    
    def initialize_post(self):
        self.result_folder = f"post_{self.name}"
        shutil.rmtree(self.result_folder,ignore_errors=True)
        self.writer = io.VTKFile(self.domain.comm, f"{self.result_folder}/{self.name}.pvd", "wb")
        self.writer_vtx = io.VTXWriter(self.domain.comm, f"{self.result_folder}/{self.name}.bp",output=[self.u,self.source_rhs,self.active_nodes_func])

    def writepos(self,extra_funcs=[]):
        funcs = [self.u,
                 #self.u_prev,
                 self.active_els_func,
                 self.active_nodes_func,
                 self.material_id,
                 self.source_rhs,
                 #self.k,
                 ]
        if self.gamma_nodes is not None:
            funcs.append(self.gamma_nodes)
        if self.is_grad_computed:
            funcs.append(self.grad_u)
        if self.is_dirichlet_gamma:
            funcs.append(self.dirichlet_gamma)
        #BPARTITIONTAG
        partition = fem.Function(self.dg0_bg,name="partition")
        partition.x.array[:] = rank
        funcs.append(partition)
        #EPARTITIONTAG
        #BDEBUG
        if self.supg_elwise_coeff is not None:
            funcs.append(self.supg_elwise_coeff)
        #EDEBUG

        bnodes = indices_to_function(self.v,self.bfacets_tag.find(1),self.dim-1,name="bnodes")
        funcs.append(bnodes)
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

    def l2_dot_gamma( self, f : dolfinx.fem.Function, g : typing.Optional[dolfinx.fem.Function] = None ):
        if g is None:
            g = f
        gamma_ents = self.get_facet_integrations_entities()
        ds_neumann = ufl.Measure('ds', domain=self.domain, subdomain_data=[
            (8,np.asarray(gamma_ents, dtype=np.int32))])
        l_ufl = f*g*ds_neumann(8)
        l2_norm = dolfinx.fem.assemble_scalar(fem.form(l_ufl))
        l2_norm = comm.allreduce(l2_norm, op=MPI.SUM)
        return l2_norm
