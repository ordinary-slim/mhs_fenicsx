from __future__ import annotations
from dolfinx import io, fem, mesh, cpp, geometry, la
import ufl
import numpy as np
import numpy.typing as npt
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
from mhs_fenicsx.problem.material import Material, base_mat_to_problem, phase_change_mat_to_problem
import mhs_fenicsx_cpp
import typing
import copy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Problem:
    def __init__(self, domain, parameters, name="case"):
        self.input_parameters = parameters.copy()
        self.writers = dict()
        self.domain   = domain
        self.dim = self.domain.topology.dim
        self.name = name
        # Function spaces
        self.v       = fem.functionspace(domain, ("Lagrange", 1),)
        self.dof_coords = self.v.tabulate_dof_coordinates()
        self.dg0  = fem.functionspace(domain, ("Discontinuous Lagrange", 0),)
        self.dg0_vec = fem.functionspace(self.domain,
                                         basix.ufl.element("DG",
                                                           self.domain.basix_cell(),
                                                           0,
                                                           shape=(self.dim,)))
        self.restriction: typing.Optional[multiphenicsx.fem.DofMapRestriction] = None

        for dim in [self.dim, self.dim-1]:
            self.domain.topology.create_entities(dim)
        self.domain.topology.create_connectivity(self.dim,self.dim)
        self.domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
        # Set num cells per processor
        self.cell_map = self.domain.topology.index_map(self.dim)
        self.facet_map = self.domain.topology.index_map(self.dim-1)
        self.node_map = self.domain.topology.index_map(0)
        self.num_cells = self.cell_map.size_local + self.cell_map.num_ghosts
        self.num_facets = self.facet_map.size_local + self.facet_map.num_ghosts
        self.num_nodes = self.node_map.size_local + self.node_map.num_ghosts
        self.bb_tree = geometry.bb_tree(self.domain,self.dim,np.arange(self.num_cells,dtype=np.int32),padding=1e-7)
        self.bb_tree_nodes = geometry.bb_tree(self.domain,0,np.arange(self.num_nodes,dtype=np.int32),padding=1e-7)
        self.initialize_activation()

        self.u   = fem.Function(self.v, name="uh")   # Solution
        self.u_prev = fem.Function(self.v, name="uh_n") # Previous solution
        self.grad_u = fem.Function(self.dg0_vec,name="grad")
        self.is_grad_computed = False
        self.dirichlet_bcs = []

        # BCs / Interface
        self.ext_nodal_activation : dict['Problem', npt.NDArray[np.bool]] = {}
        self.gamma_nodes : dict['Problem', fem.Function] = {}
        self.gamma_facets : dict['Problem', mesh.MeshTags] = {}
        self.gamma_facets_index_map : dict['Problem', dolfinx.common.IndexMap] = {}
        self.gamma_imap_to_global_imap : dict['Problem', npt.NDArray[np.int32]] = {}
        self.gamma_integration_data : dict['Problem', npt.NDArray[np.int32]] = {}
                         
        # Source term
        try:
            hs_type = self.input_parameters["heat_source"]["type"]
            if   hs_type == 1:
                self.source = Gaussian1D(self)
            elif hs_type == 2:
                self.source = Gaussian2D(self)
            elif hs_type == 3:
                self.source = Gaussian3D(self)
            elif hs_type == 4:
                self.source = LumpedHeatSource(self)
            else:
                raise ValueError(f"Unknown heat source type {hs_type}.")
        except KeyError:
            if self.dim == 1:
                self.source = Gaussian1D(self)
            elif self.dim == 2:
                self.source = Gaussian2D(self)
            else:
                self.source = Gaussian3D(self)

        self.rhs = None # For python functions

        # Time
        self.is_steady = parameters["isSteady"]
        self.iter     = 0
        self.time     = 0.0
        dt = parameters["dt"] if not(self.is_steady) else -1.0
        self.dt       = fem.Constant(self.domain, dt)
        self.dt_func  = fem.Function(self.v, name="dt")
        self.dt_func.x.array[:] = self.dt.value
        # Motion
        self.domain_speed     = np.array(parameters["domain_speed"]) if "domain_speed" in parameters else None
        advection_speed = parameters["advection_speed"][:self.domain.topology.dim] if "advection_speed" in parameters else np.zeros(self.domain.topology.dim)
        self.advection_speed = fem.Constant(self.domain,advection_speed)
        angular_advection_speed = parameters["angular_advection_speed"] if "angular_advection_speed" in parameters else np.zeros(3)
        if self.domain.topology.dim < 3:
            angular_advection_speed = np.sum(angular_advection_speed)
        self.angular_advection_speed = fem.Constant(self.domain,angular_advection_speed)
        rotation_center = parameters["rotation_center"][:self.domain.topology.dim] if "rotation_center" in parameters else np.zeros(self.domain.topology.dim)
        self.rotation_center = fem.Constant(self.domain,rotation_center)
        # Stabilization
        self.is_supg = (("supg" in parameters) and bool(parameters["supg"]))
        self.advected_el_size : typing.Optional[fem.Function] = fem.Function(self.dg0,name="supg_tau") if self.is_supg else None
        # Material parameters
        self.define_materials(parameters)
        # Integration
        self.quadrature_metadata = parameters["quadrature_metadata"] \
                if "quadrature_metadata" in parameters \
                else {"quadrature_rule":"vertex", "quadrature_degree":1, }
        self.is_post_initialized = False
        self.is_mesh_shared = False
        self.set_linear_solver(parameters["petsc_opts"] if "petsc_opts" in parameters else None)

    def __del__(self):
        for writer in self.writers.values():
            writer.close()
        self._destroy()

    def copy(self,name=None):
        self.is_mesh_shared = True
        to_be_skipped = set([
            "restriction",
            "is_post_initialized",
            "writers",
            ])
        to_be_deep_copied = set([
            "linear_solver_opts",
            "source",
            "dirichlet_bcs",
            ])
        to_be_reset = set([
            "gamma_nodes",
            "gamma_facets",
            "gamma_facets_index_map",
            "gamma_imap_to_global_imap",
            "gamma_integration_data",
            ])
        attributes = (set(self.__dict__.keys()) - to_be_skipped) - to_be_deep_copied
        result = object.__new__(self.__class__)
        for k in attributes:
            attr = self.__dict__[k]
            if   isinstance(attr, (fem.Function,np.ndarray)):
                setattr(result, k, attr.copy())
                if hasattr(attr, "name"):
                    result.__dict__[k].name = attr.name
            elif isinstance(attr, fem.Constant):
                result.__dict__[k] = fem.Constant(self.domain, attr.value)
            else:
                setattr(result, k, attr)
        for k in to_be_deep_copied:
            setattr(result, k, copy.deepcopy(self.__dict__[k]))
        for k in to_be_reset:
            setattr(result, k, type(self.__dict__[k])())
        if not(name):
            name = result.name + "_bis"
        result.name = name
        result.writers = {}
        result.is_post_initialized = False
        return result

    def set_dt(self, dt : float):
        self.dt.value = dt
        self.dt_func.x.array[:] = dt

    def set_initial_condition( self, expression ):
        try:
            constant = float(expression)
            self.u.x.array[:] = constant
        except TypeError:
            self.u.interpolate(expression)
        self.u_prev.x.array[:] = self.u.x.array[:]

    def set_rhs( self, rhs ):
        self.rhs = rhs

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
        self.phase_change = False
        for key in parameters.keys():
            if key.startswith("material"):
                material = Material(parameters[key])
                self.materials.append(material)
                self.phase_change = (self.phase_change or material.phase_change)

        assert len(self.materials) > 0, "No materials defined!"
        # All domain starts out covered by material #0
        self.material_id = fem.Function(self.dg0,name="material_id")
        self.material_id.x.array[:] = 0.0
        # Initialize material funcs
        self.k   = fem.Function(self.dg0,name="conductivity")
        self.cp  = fem.Function(self.dg0,name="specific_heat")
        self.rho = fem.Function(self.dg0,name="density")
        if self.phase_change:
            self.latent_heat   = fem.Function(self.dg0,name="latent_heat")
            self.solidus_temperature  = fem.Function(self.dg0,name="solidus_temperature")
            self.liquidus_temperature = fem.Function(self.dg0,name="liquidus_temperature")
            self.smoothing_cte_phase_change = fem.Constant(self.domain,
                                                           float(parameters["smoothing_cte_phase_change"]))
        self.set_material_funcs()
    
    def _update_mat_at_cells(self, mat_id, cells):
        material = self.materials[mat_id]
        for k, v in base_mat_to_problem.items():
            self.__dict__[v].x.array[cells] = material.__dict__[k]
        if self.phase_change:
            for k, v in phase_change_mat_to_problem.items():
                self.__dict__[v].x.array[cells] = material.__dict__[k]

    def update_material_funcs(self,cells,new_id):
        self.material_id.x.array[cells] = new_id
        self._update_mat_at_cells(new_id, cells)

    def set_material_funcs(self):
        for mat_id in range(len(self.materials)):
            cells = np.flatnonzero(abs(self.material_id.x.array-mat_id)<1e-7)
            self._update_mat_at_cells(mat_id,cells)

    def compute_advected_el_size(self):
        if self.advected_el_size is None:
            self.advected_el_size = fem.Function(self.dg0,name="supg_tau")
        mhs_fenicsx_cpp.compute_el_size_along_vector(self.advected_el_size._cpp_object,self.advection_speed._cpp_object)

    def pre_iterate(self,forced_time_derivative=False,verbose=True):
        # Pre-iterate source first, current track is tn's
        if rank==0 and verbose:
            print(f"\nProblem {self.name} about to solve for iter {self.iter+1}, time {self.time+self.dt.value}")
        self.source.pre_iterate(self.time,self.dt.value,verbose=verbose)
        # Mesh motion
        if self.domain_speed is not None and not(self.is_mesh_shared):
            dx = np.round(self.domain_speed*self.dt.value,7)
            self.domain.geometry.x[:] += dx
            self.bb_tree.bbox_coordinates[:] += dx
            self.bb_tree_nodes.bbox_coordinates[:] += dx
            self.dof_coords += dx
            self.clear_gamma_data()
            if self.is_supg:
                self.compute_advected_el_size()

        self.source.set_fem_function(self.dof_coords)
        self.iter += 1
        self.time += self.dt.value
        if not(forced_time_derivative):
            self.u_prev.x.array[:] = self.u.x.array

    def post_iterate(self):
        self._destroy()
        self.has_preassembled = False
        self.is_grad_computed = False

    def initialize_activation(self, finalize=True):
        self.active_els = np.arange(self.num_cells,dtype=np.int32)
        self.active_els_func= fem.Function(self.dg0,name="active_els")
        self.active_nodes_func = fem.Function(self.v,name="active_nodes")
        self.active_els_func.x.array[:] = 1.0
        self.local_active_els = np.arange(self.cell_map.size_local,dtype=np.int32)
        if finalize:
            self.finalize_activation()

    # TODO: Overload this function
    def set_activation(self, active_els=typing.Optional[list], finalize=True):
        if active_els is None:
            active_els = np.arange(self.num_cells,dtype=np.int32)
        self.active_els = active_els
        self.active_els_func.x.array.fill(0.0)
        self.active_els_func.x.array[active_els] = 1.0
        self.active_els_func.x.scatter_forward()
        self.local_active_els = self.active_els_func.x.array.nonzero()[0]
        self.local_active_els = self.local_active_els[:np.searchsorted(self.local_active_els, self.cell_map.size_local)]
        if finalize:
            self.finalize_activation()

    def finalize_activation(self):
        old_active_dofs_array  = self.active_nodes_func.x.array.copy()
        self.active_dofs = fem.locate_dofs_topological(self.v, self.dim, self.active_els, remote=True)
        self.active_nodes_func.x.array[:] = np.float64(0.0)
        self.active_nodes_func.x.array[self.active_dofs] = np.float64(1.0)
        #self.active_nodes_func.x.scatter_forward()
        just_active_dofs_array = self.active_nodes_func.x.array - old_active_dofs_array
        self.just_activated_nodes = np.flatnonzero(just_active_dofs_array)

        self.restriction = multiphenicsx.fem.DofMapRestriction(self.v.dofmap, self.active_dofs)
        self.update_boundary()
        self.set_form_subdomain_data()

    def set_form_subdomain_data(self):
        self.form_subdomain_data = {
                fem.IntegralType.cell : [(1, self.local_active_els)],
                fem.IntegralType.exterior_facet : [(1, self.get_facet_integrations_entities(self.bfacets_tag.find(1)))],
                }

    def update_boundary(self):
        bfacets_indices  = locate_active_boundary(self.domain, self.active_els_func)
        self.bfacets_tag  = mesh.meshtags(self.domain, self.dim-1,
                                         np.arange(self.num_facets, dtype=np.int32),
                                         get_mask(self.num_facets,
                                                  bfacets_indices, dtype=np.int32),
                                         )
        # If needed, we are creating bnodes_tag in find_interface with child problem

    def compute_gradient(self, cells: typing.Optional[npt.NDArray[np.int32]] = None):
        if not(self.is_grad_computed):
            gradient_expression = fem.Expression(ufl.grad(self.u),self.grad_u.function_space.element.interpolation_points) #TODO: compile this only one
            self.grad_u.interpolate(gradient_expression, cells0=cells)
            self.grad_u.x.scatter_forward()
            self.is_grad_computed = True

    def clear_gamma_data(self):
        self.gamma_nodes.clear()
        self.gamma_facets.clear()
        self.gamma_facets_index_map.clear()
        self.gamma_imap_to_global_imap.clear()
        self.gamma_integration_data.clear()

    def get_active_in_external(self, p_ext:Problem ):
        # Consider only asking for currently active nodes!
        owners = mhs_fenicsx_cpp.find_owner_rank(self.domain.geometry.x,
                                                 p_ext.bb_tree._cpp_object,
                                                 p_ext.active_els_func._cpp_object)
        return np.array(owners>=0,dtype=np.bool)

    def subtract_problem(self, p_ext : 'Problem', finalize=True):
        self.ext_nodal_activation[p_ext] = self.get_active_in_external(p_ext)
        new_active_els = mhs_fenicsx_cpp.deactivate_from_nodes(self.domain._cpp_object,
                                                               self.active_els_func._cpp_object,
                                                               self.ext_nodal_activation[p_ext])
        self.set_activation(new_active_els, finalize=finalize)
        if finalize:
            self.find_gamma(p_ext, self.ext_nodal_activation[p_ext])

    def intersect_problem(self, p_ext : 'Problem', finalize=True):
        self.ext_nodal_activation[p_ext] = self.get_active_in_external(p_ext)
        new_active_els = mhs_fenicsx_cpp.intersect_from_nodes(self.domain._cpp_object,
                                                              self.active_els_func._cpp_object,
                                                              self.ext_nodal_activation[p_ext])
        self.set_activation(new_active_els, finalize=finalize)
        if finalize:
            self.find_gamma(p_ext, self.ext_nodal_activation[p_ext])

    def find_gamma(self, p_ext : 'Problem', ext_active_dofs_array = None):
        # TODO: Add p_ext as argument
        if ext_active_dofs_array is None:
            ext_active_dofs_array = self.get_active_in_external(p_ext)
        self.ext_nodal_activation[p_ext] = ext_active_dofs_array
        self.gamma_facets[p_ext] = mesh.MeshTags(mhs_fenicsx_cpp.find_interface(
            self.domain._cpp_object,
            self.bfacets_tag._cpp_object,
            self.v.dofmap.index_map,
            ext_active_dofs_array))

        self.update_gamma_data(p_ext)

    def set_gamma(self, p_ext : 'Problem', gamma_facets_tag : mesh.MeshTags):
        dim = gamma_facets_tag.dim
        assert(dim==self.dim-1)
        im = gamma_facets_tag.topology.index_map(dim)
        assert((im is not None) and (im == self.domain.topology.index_map(dim)))
        self.gamma_facets[p_ext] = gamma_facets_tag
        self.update_gamma_data(p_ext)

    def update_gamma_data(self, p_ext : 'Problem'):
        # TODO: Add p_ext as argument
        self.gamma_nodes[p_ext] = indices_to_function(self.v,
                                                      self.gamma_facets[p_ext].find(1),
                                                      self.dim-1,
                                                      name=f"gamma_{p_ext.name}",)
        indices_all_gamma_facets = self.gamma_facets[p_ext].values.nonzero()[0]
        self.gamma_facets_index_map[p_ext], \
        self.gamma_imap_to_global_imap[p_ext] = cpp.common.create_sub_index_map(self.facet_map,
                                                    indices_all_gamma_facets,
                                                    False)
        self.gamma_integration_data[p_ext] = self.get_facet_integrations_entities(indices_all_gamma_facets)

    def add_dirichlet_bc(self, func, bdofs=None, bfacets_tag=None, marker=None, reset=False):
        if reset:
            self.dirichlet_bcs = []
        if bdofs is None:
            if (marker is not None):
                bdofs  = fem.locate_dofs_geometrical(self.v,marker)
            else:
                if bfacets_tag is None:
                    bfacets_tag = self.bfacets_tag
                bdofs = fem.locate_dofs_topological(self.v, self.dim-1, bfacets_tag.find(1),)

        u_bc = fem.Function(self.v)
        u_bc.interpolate(func)
        bc = fem.dirichletbc(u_bc, bdofs)
        self.dirichlet_bcs.append(bc)
        return bc

    def get_facet_integrations_entities(self, facet_indices):
        return get_facet_integration_entities(self.domain,facet_indices,self.active_els_func)

    def set_forms_domain(self, subdomain_idx=1, argument=None):
        dx = ufl.Measure("dx", metadata=self.quadrature_metadata)
        u = argument if argument is not None else self.u
        v = ufl.TestFunction(self.v)

        # Conduction
        self.a_ufl = self.k*ufl.dot(ufl.grad(u), ufl.grad(v))*dx(subdomain_idx)

        # Source term
        if self.rhs is not None:
            self.source.fem_function.interpolate(self.rhs)
        self.l_ufl = self.source.fem_function*v*dx(subdomain_idx)

        # Time derivative
        time_derivative_coefficient = self.rho * self.cp
        ## Phase change
        if self.phase_change:
            cte = (2.0*self.smoothing_cte_phase_change/(self.liquidus_temperature - self.solidus_temperature))
            melting_temperature = (self.liquidus_temperature + self.solidus_temperature) / 2.0
            fp  = lambda tem : cte/2.0*(1 - ufl.tanh(cte*(tem - melting_temperature))**2)
            time_derivative_coefficient += self.rho * self.latent_heat * fp(u)

        if not(self.is_steady):
            self.a_ufl += time_derivative_coefficient * \
                    (u - self.u_prev)/self.dt_func * \
                    v * dx(subdomain_idx)

        # Translational advection
        has_advection = np.linalg.norm(self.advection_speed.value)
        if has_advection:
            advection_norm = fem.Constant(self.domain, np.linalg.norm(self.advection_speed.value))
            self.a_ufl += time_derivative_coefficient*ufl.dot(self.advection_speed,ufl.grad(u))*v*dx(subdomain_idx)
            # Translational advection stabilization
            if self.is_supg:
                assert(self.advected_el_size is not None)
                supg_coeff = self.advected_el_size**2 / (2 * self.advected_el_size * \
                        time_derivative_coefficient * advection_norm + \
                         4 * self.k)
                if not(self.is_steady):
                    self.a_ufl += supg_coeff * time_derivative_coefficient * \
                            time_derivative_coefficient * (u - self.u_prev)/self.dt_func * \
                            ufl.dot(self.advection_speed,ufl.grad(v)) * dx(subdomain_idx)
                self.a_ufl += supg_coeff * time_derivative_coefficient * \
                        time_derivative_coefficient * ufl.dot(self.advection_speed,ufl.grad(u)) * \
                        ufl.dot(self.advection_speed,ufl.grad(v)) * dx(subdomain_idx)
                self.l_ufl += supg_coeff * time_derivative_coefficient * \
                        self.source.fem_function * \
                        ufl.dot(self.advection_speed,ufl.grad(v)) * dx(subdomain_idx)

        # Rotational advection
        # WARNING: Stabilziation not implemented
        has_rotational_advection = np.linalg.norm(self.angular_advection_speed.value)
        if has_rotational_advection:
            x = ufl.SpatialCoordinate(self.domain)
            if self.dim < 3:
                pointwise_advection_speed = (self.angular_advection_speed)*ufl.perp(x-self.rotation_center)
            else:
                pointwise_advection_speed = ufl.cross(self.angular_advection_speed,x-self.rotation_center)
            self.a_ufl += time_derivative_coefficient*ufl.dot(pointwise_advection_speed,ufl.grad(u))*v*dx(subdomain_idx)

    def set_forms_boundary(self, subdomain_idx=1, argument=None):
        '''
        rn must be called after set_forms_domain
        since a_ufl and l_ufl not initialized before
        '''
        #TODO: Exclude Gamma facets from this!
        ds = ufl.Measure('ds', metadata=self.quadrature_metadata)
        u = argument if argument is not None else self.u
        v = ufl.TestFunction(self.v)
        # CONVECTION
        if self.convection_coeff is not None:
            self.a_ufl += self.convection_coeff * \
                          u*v* \
                          ds(subdomain_idx)
            T_env   = fem.Constant(self.domain, PETSc.ScalarType(self.T_env))
            self.l_ufl += self.convection_coeff * \
                          T_env*v* \
                          ds(subdomain_idx)

    def compile_forms(self):
        self.r_ufl = self.a_ufl - self.l_ufl#residual
        self.j_ufl = ufl.derivative(self.r_ufl, self.u)#jacobian
        self.mr_ufl = -self.r_ufl#minus residual
        self.mr_compiled = fem.compile_form(self.domain.comm, self.mr_ufl,
                                            form_compiler_options={"scalar_type": np.float64})
        self.j_compiled = fem.compile_form(self.domain.comm, self.j_ufl,
                                            form_compiler_options={"scalar_type": np.float64})

    def instantiate_forms(self):
        rcoeffmap, rconstmap = get_identity_maps(self.mr_ufl)
        self.mr_instance = fem.create_form(self.mr_compiled,
                                           [self.v],
                                           msh=self.domain,
                                           subdomains=self.form_subdomain_data,
                                           coefficient_map=rcoeffmap,
                                           constant_map=rconstmap)
        lcoeffmap, lconstmap = get_identity_maps(self.j_ufl)
        self.j_instance = fem.create_form(self.j_compiled,
                                          [self.v, self.v],
                                          msh=self.domain,
                                          subdomains=self.form_subdomain_data,
                                          coefficient_map=lcoeffmap,
                                          constant_map=lconstmap)

    def compile_create_forms(self):
        self.compile_forms()
        self.instantiate_forms()

    def pre_assemble(self):
        self._destroy()
        self.A = multiphenicsx.fem.petsc.create_matrix(self.j_instance,
                                                  (self.restriction, self.restriction),
                                                  )
        self.L = multiphenicsx.fem.petsc.create_vector(self.mr_instance,
                                                       self.restriction)
        self.x = multiphenicsx.fem.petsc.create_vector(self.mr_instance, restriction=self.restriction)
        self._obj_vec = multiphenicsx.fem.petsc.create_vector(self.mr_instance, self.restriction)
        self.has_preassembled = True

    def assemble_jacobian(self,
                          J_mat: typing.Optional[PETSc.Mat] = None,
                          zero=True,
                          finalize=True):
        if J_mat is None:
            J_mat = self.A
        if zero:
            J_mat.zeroEntries()
        multiphenicsx.fem.petsc.assemble_matrix(J_mat,
                                                self.j_instance,
                                                bcs=self.dirichlet_bcs,
                                                restriction=(self.restriction, self.restriction))
        if finalize:
            J_mat.assemble()

    def assemble_residual(self,
                          F_vec: typing.Optional[PETSc.Vec] = None):
        if F_vec is None:
            F_vec = self.L
        with F_vec.localForm() as l_local:
            l_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector(F_vec,
                                                self.mr_instance,
                                                restriction=self.restriction,)
        # Dirichlet
        multiphenicsx.fem.petsc.apply_lifting(F_vec, [self.j_instance], [self.dirichlet_bcs], [self.u.x.petsc_vec], restriction=self.restriction)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        multiphenicsx.fem.petsc.set_bc(F_vec,self.dirichlet_bcs, self.u.x.petsc_vec, restriction=self.restriction)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def assemble(self):
        if not(self.has_preassembled):
            self.pre_assemble()
        self.assemble_jacobian()
        self.assemble_residual()

    def _update_solution(self, x = None):
        sol = self.u
        if x is None:
            x = self.x
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with sol.x.petsc_vec.localForm() as sol_sub_vector_local, \
                multiphenicsx.fem.petsc.VecSubVectorWrapper(x, self.v.dofmap, self.restriction) as x_wrapper:
                    sol_sub_vector_local[:] = x_wrapper

    def _destroy(self):
        for attr in ["x", "A", "L"]:
            try:
                self.__dict__[attr].destroy()
            except KeyError:
                pass
    
    def initialize_post(self):
        self.result_folder = f"post_{self.name}"
        shutil.rmtree(self.result_folder,ignore_errors=True)
        self.writers["vtk"] = io.VTKFile(self.domain.comm, f"{self.result_folder}/{self.name}.pvd", "wb")
        self.writers["vtx"] = io.VTXWriter(self.domain.comm, f"{self.result_folder}/{self.name}.bp",output=[self.u,self.source.fem_function,self.active_nodes_func])
        self.is_post_initialized = True

    def writepos(self,extra_funcs=[]):
        if not(self.is_post_initialized):
            self.initialize_post()
        funcs = [self.u,
                 #self.u_prev,
                 self.active_els_func,
                 self.active_nodes_func,
                 self.material_id,
                 self.source.fem_function,
                 #self.k,
                 ]
        for f in self.gamma_nodes.values():
            funcs.append(f)
        if self.is_grad_computed:
            funcs.append(self.grad_u)
        #BPARTITIONTAG
        partition = fem.Function(self.dg0,name="partition")
        partition.x.array[:] = rank
        funcs.append(partition)
        #EPARTITIONTAG

        bnodes = indices_to_function(self.v,self.bfacets_tag.find(1),self.dim-1,name="bnodes")
        funcs.append(bnodes)
        funcs.extend(extra_funcs)
        self.writers["vtk"].write_function(funcs,t=np.round(self.time,7))

    def writepos_vtx(self):
        self.writers["vtx"].write(self.time)

    def clear_dirchlet_bcs(self):
        self.dirichlet_bcs = []

    def write_bmesh(self):
        bmesh = dolfinx.mesh.create_submesh(self.domain,self.dim-1,self.bfacets_tag.find(1))[0]
        with io.VTKFile(bmesh.comm, f"out/bmesh_{self.name}.pvd", "w") as ofile:
            ofile.write_mesh(bmesh)

    def set_linear_solver(self, opts:typing.Optional[dict] = None):
        if opts is None:
            opts = {"pc_type" : "lu", "pc_factor_mat_solver_type" : "mumps",}
        self.linear_solver_opts = dict(opts)

    def set_snes_sol_vector(self) -> PETSc.Vec:  # type: ignore[no-any-unimported]
        """ Set PETSc.Vec to be passed to PETSc.SNES.solve to initial guess """
        
        #x = multiphenicsx.fem.petsc.create_vector(self.mr_instance, self.restriction)
        sol = self.u
        with multiphenicsx.fem.petsc.VecSubVectorWrapper(self.x, self.v.dofmap, self.restriction) as x_wrapper:
            with sol.x.petsc_vec.localForm() as solution_local:
                x_wrapper[:] = solution_local

    def obj(  # type: ignore[no-any-unimported]
        self, snes: PETSc.SNES, x: PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.R(snes, x, self._obj_vec)
        return self._obj_vec.norm()  # type: ignore[no-any-return]

    def R(  # type: ignore[no-any-unimported]
        self, snes: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec
    ) -> None:
        """Assemble the residual."""
        self._update_solution(x)
        self.assemble_residual(F_vec)
        F_vec.scale(-1)

    def J(  # type: ignore[no-any-unimported]
        self, snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat,
        P_mat: PETSc.Mat
    ) -> None:
        """Assemble the jacobian."""
        self.assemble_jacobian(J_mat)

    def non_linear_solve(self, max_iter=20):
        if not(self.has_preassembled):
            self.pre_assemble()
        # Solve
        snes = PETSc.SNES().create(self.domain.comm)
        snes.setTolerances(max_it=max_iter)
        ksp_opts = PETSc.Options()
        for k,v in self.linear_solver_opts.items():
            ksp_opts[k] = v
        snes.getKSP().setFromOptions()
        snes.setObjective(self.obj)
        snes.setFunction(self.R, self.L)
        snes.setJacobian(self.J, J=self.A, P=None)
        snes.setMonitor(lambda _, it, residual: print(it, residual))
        self.set_snes_sol_vector()
        snes.solve(None, self.x)
        self._update_solution(self.x)  # TODO can this be safely removed?
        snes.destroy()
        self.is_grad_computed = False#dubious line

class GammaL2Dotter:
    def __init__(self, p:Problem):
        self.f, self.g = (ufl.Coefficient(p.v), ufl.Coefficient(p.v))
        self.l_ufl = self.f * self.g * ufl.ds(1)
        self.l_com = fem.compile_form(comm, self.l_ufl,
                                      form_compiler_options={"scalar_type": np.float64})
        self.p = p

    def set_gamma(self, p_ext : Problem):
        self.integration_ents = self.p.gamma_integration_data[p_ext]

    def __call__(self,f:fem.Function,g:typing.Optional[fem.Function] = None):
        form_subdomain_data = {fem.IntegralType.exterior_facet : [(1, self.integration_ents)]}
        if g is None:
            g = f
        assert(f.function_space == g.function_space)
        coefficient_map = {self.f : f, self.g : g}
        self.l_instance = fem.create_form(self.l_com,
                                          [],
                                          msh=f.function_space.mesh,
                                          subdomains=form_subdomain_data,
                                          coefficient_map=coefficient_map,
                                          constant_map={})
        l2_norm = dolfinx.fem.assemble_scalar(self.l_instance)
        l2_norm = comm.allreduce(l2_norm, op=MPI.SUM)
        return l2_norm
