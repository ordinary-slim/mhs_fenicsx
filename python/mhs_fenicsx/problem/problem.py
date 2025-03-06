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
from mhs_fenicsx.problem.printer import createPrinter, DEDPrinter, LPBFPrinter
from mhs_fenicsx.problem.material import Material
import mhs_fenicsx_cpp
import typing
import functools
import copy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class Problem:
    def __init__(self, domain, parameters, finalize_activation=True, name="case"):
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
        # Material parameters
        self.define_materials(parameters)

        for dim in [self.dim, self.dim-1]:
            self.domain.topology.create_entities(dim)
        self.domain.topology.create_connectivity(self.dim,self.dim)
        self.domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
        self.domain.topology.create_connectivity(0, domain.topology.dim)
        # Set num cells per processor
        self.cell_map = self.domain.topology.index_map(self.dim)
        self.facet_map = self.domain.topology.index_map(self.dim-1)
        self.node_map = self.domain.topology.index_map(0)
        self.num_cells = self.cell_map.size_local + self.cell_map.num_ghosts
        self.num_facets = self.facet_map.size_local + self.facet_map.num_ghosts
        self.num_nodes = self.node_map.size_local + self.node_map.num_ghosts
        self.bb_tree = geometry.bb_tree(self.domain,self.dim,np.arange(self.num_cells,dtype=np.int32),padding=1e-7)
        self.restriction: typing.Optional[multiphenicsx.fem.DofMapRestriction] = None
        self.initialize_activation(finalize=finalize_activation)

        self.u   = fem.Function(self.v, name="uh")   # Solution
        self.u_prev = fem.Function(self.v, name="uh_n") # Previous solution
        self.u_av = fem.Function(self.dg0, name="u_av") # Average @ cells
        self.grad_u = fem.Function(self.dg0_vec,name="grad")
        self.is_grad_computed = False
        self.dirichlet_bcs = []

        # BCs / Interface
        self.ext_nodal_activation : dict['Problem', npt.NDArray[np.bool]] = {}
        self.ext_colliding_els : dict['Problem', npt.NDArray[np.int32]] = {}
        self.gamma_nodes : dict['Problem', fem.Function] = {}
        self.gamma_facets : dict['Problem', mesh.MeshTags] = {}
        self.gamma_facets_index_map : dict['Problem', dolfinx.common.IndexMap] = {}
        self.gamma_imap_to_global_imap : dict['Problem', npt.NDArray[np.int32]] = {}
        self.gamma_integration_data : dict['Problem', npt.NDArray[np.int32]] = {}

        # Time
        self.is_steady = parameters["isSteady"]
        self.iter     = 0
        self.time     = 0.0
        dt = parameters["dt"] if not(self.is_steady) else -1.0
        self.dt       = fem.Constant(self.domain, dt)
        self.dt_func  = fem.Function(self.v, name="dt")
        self.dt_func.x.array[:] = self.dt.value
                         
        # Source term
        self.sources = createHeatSources(self)
        self.source, self.current_source_term = self.sources[0], 0

        self.rhs = None # For python functions

        # 3D printing
        self.printer : typing.Optional[Printer] = createPrinter(self) if "printer" in parameters else None

        # Motion
        self.domain_speed     = np.array(parameters["domain_speed"]) if "domain_speed" in parameters else None
        self.attached_to_hs   = bool(parameters["attached_to_hs"]) if "attached_to_hs" in parameters else False
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
        # Integration
        self.quadrature_metadata = parameters["quadrature_metadata"] \
                if "quadrature_metadata" in parameters \
                else {"quadrature_rule":"vertex", "quadrature_degree":1, }
        self.is_post_initialized = False
        self.is_mesh_shared = False
        self.set_linear_solver(parameters["petsc_opts"] if "petsc_opts" in parameters else None)

    def __del__(self):
        try:
            for writer in self.writers.values():
                writer.close()
        except AttributeError:
            pass
        self._destroy()

    def copy(self,name=None):
        self.is_mesh_shared = True
        to_be_skipped = set([
            "restriction",
            "is_post_initialized",
            "form_subdomain_data",
            ])
        to_be_shallow_copied = set([
            "material_to_itag",
            ])
        to_be_deep_copied = set([
            "linear_solver_opts",
            "sources",
            "dirichlet_bcs",
            ])
        to_be_reset = set([
            "local_cells",
            "ext_nodal_activation",
            "ext_colliding_els",
            "gamma_nodes",
            "gamma_facets",
            "gamma_facets_index_map",
            "gamma_imap_to_global_imap",
            "gamma_integration_data",
            "writers",
            ])
        attributes = (set(self.__dict__.keys()) - to_be_skipped) - to_be_deep_copied - to_be_shallow_copied
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
        for k in to_be_shallow_copied:
            setattr(result, k, self.__dict__[k].copy())
        for k in to_be_deep_copied:
            setattr(result, k, copy.deepcopy(self.__dict__[k]))
        for k in to_be_reset:
            setattr(result, k, type(self.__dict__[k])())
        if not(name):
            name = result.name + "_bis"
        result.name = name
        result.is_post_initialized = False
        result.switch_source_term(self.current_source_term)
        result.form_subdomain_data = {fem.IntegralType.cell : [],
                                      fem.IntegralType.exterior_facet : []}
        return result

    def set_dt(self, dt : float):
        self.dt.value = dt
        self.dt_func.x.array[:] = dt

    def set_domain_speed(self, v : npt.NDArray[typing.Union[np.float32, np.float64]]):
        assert(v.size == 3)
        self.domain_speed = v
        self.advection_speed.value = -v[:self.dim]

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
        self.T_env = fem.Constant(self.domain, PETSc.ScalarType(parameters["environment_temperature"]))
        self.T_dep = parameters["deposition_temperature"] if "deposition_temperature" in parameters else None
        self.convection_coeff = fem.Constant(self.domain, PETSc.ScalarType(parameters["convection_coeff"])) if "convection_coeff" \
                in parameters else None
        self.radiation_coeff = fem.Constant(self.domain, PETSc.ScalarType(parameters["radiation_coeff"])) if "radiation_coeff" \
                in parameters else None

        self.materials = []
        self.phase_change, self.melting = False, False
        for key in parameters.keys():
            if key.startswith("material"):
                name = key.split('_')[-1]
                material = Material(parameters[key], name=name)
                self.materials.append(material)
                self.phase_change = (self.phase_change or material.phase_change)
                self.melting = (self.melting or (material.melts_to is not None))
        self.material_library = {mat.name : mat for mat in self.materials}
        self.local_cells = {mat: np.array([], dtype=np.int32) for mat in self.materials}
        self.material_library[None] = None
        if self.melting:
            for mat in self.materials:
                mat.melts_to = self.material_library[mat.melts_to]

        assert len(self.materials) > 0, "No materials defined!"
        self.material_to_tag  = {mat: (idx+1) for idx, mat in enumerate(self.materials)}
        self.material_to_itag = self.material_to_tag.copy()

        # All domain starts out covered by material #0
        self.material_id = fem.Function(self.dg0,name="material_id")
        self.material_id.x.array.fill(1.0)# Everything initialized to first material
        # Initialize material funcs
        smoothing_cte_phase_change = parameters["smoothing_cte_phase_change"] \
                if "smoothing_cte_phase_change" in parameters else 0.0
        self.smoothing_cte_phase_change = fem.Constant(self.domain, np.float64(smoothing_cte_phase_change))
    
    def update_material_at_cells(self, cells, mat : Material, finalize=True):
        self.material_id.x.array[cells] = self.material_to_tag[mat]
        if finalize:
            self.material_id.x.scatter_forward()

    def switch_source_term(self, idx : int):
        assert(0 <= idx < len(self.sources))
        self.current_source_term = idx
        self.source = self.sources[idx]
        for idx, source in enumerate(self.sources):
            if not(idx == self.current_source_term):
                source.fem_function.x.array.fill(0.0)

    def compute_advected_el_size(self):
        if self.advected_el_size is None:
            self.advected_el_size = fem.Function(self.dg0,name="supg_tau")
        mhs_fenicsx_cpp.compute_el_size_along_vector(self.advected_el_size._cpp_object,self.advection_speed._cpp_object)

    def pre_iterate(self, forced_time_derivative=False, verbose=True, finalize_activation=True):
        # Pre-iterate source first, current track is tn's
        if rank==0 and verbose:
            print(f"\nProblem {self.name} about to solve for iter {self.iter+1}, time {self.time+self.dt.value}")
        self.source.pre_iterate(self.time,self.dt.value,verbose=verbose)
        # If chimera, update domain speed here
        if self.attached_to_hs:
            self.set_domain_speed(self.source.speed)
        # Mesh motion
        if self.domain_speed is not None and not(self.is_mesh_shared):
            dx = np.round(self.domain_speed*self.dt.value,7)
            self.domain.geometry.x[:] += dx
            self.bb_tree.bbox_coordinates[:] += dx
            self.dof_coords += dx
            self.clear_gamma_data()

        if self.is_supg and (np.linalg.norm(self.advection_speed.value) > 1e-7):
            self.compute_advected_el_size()

        self.source.set_fem_function(self.dof_coords)
        # Print
        if self.printer:
            self.printer.deposit(self.time, self.dt.value, finalize=finalize_activation)

        self.iter += 1
        self.time += self.dt.value

        if not(forced_time_derivative):
            self.u_prev.x.array[:] = self.u.x.array

    def in_plane_rotation(self, center : npt.NDArray, radians : float):
        c, s = np.cos(radians, dtype=np.float64), np.sin(radians, dtype=np.float64)
        Rt = np.array([[ +c,  +s, 0.0],
                       [ -s,  +c, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        for array in [self.domain.geometry.x,
                      self.dof_coords]:
            array[:] -= center
            array[:]  = center + np.matmul(array, Rt)
        self.bb_tree = geometry.bb_tree(self.domain,self.dim,np.arange(self.num_cells,dtype=np.int32),padding=1e-7)

    def post_modify_solution(self, cells=None):
        self.is_grad_computed = False
        self.melt(cells=cells)

    def melt(self, cells=None):
        '''Post-iterate op'''
        self.melted = False
        if self.melting:
            if cells is None:
                cells = self.local_active_els
                mat_dic = self.local_cells
            else:
                mat_dic = {}
                self.divide_domain_by_materials(cells, mat_dic)
            self.u_av.interpolate(self.u, cells0=cells)
            self.u_av.x.scatter_forward()
            for mat in self.materials:
                if mat.melts_to is not None:
                    indices_cells_to_melt = (self.u_av.x.array[mat_dic[mat]] >= mat.T_m.value).nonzero()[0]
                    cells_to_melt = mat_dic[mat][indices_cells_to_melt]
                    if cells_to_melt.size:
                        self.melted = True
                        self.update_material_at_cells(cells_to_melt, mat.melts_to, finalize=False)
            self.material_id.x.scatter_forward()

    def post_iterate(self, finalize=True):
        self._destroy()
        self.has_preassembled = False

    def is_path_over(self):
        return (self.source.path.tracks[-1].t1 - self.time) < 1e-7

    def initialize_activation(self, finalize=True):
        self.active_els_func= fem.Function(self.dg0,name="active_els")
        self.active_nodes_func = fem.Function(self.v,name="active_nodes")
        self.form_subdomain_data = {fem.IntegralType.cell : [],
                                    fem.IntegralType.exterior_facet : []}
        self.reset_activation(finalize=finalize)

    def reset_activation(self, finalize=True):
        self.active_els = np.arange(self.num_cells,dtype=np.int32)
        self.local_active_els = np.arange(self.cell_map.size_local,dtype=np.int32)
        self.active_els_func.x.array[:] = 1.0
        self.active_nodes_func.x.array[:] = 1.0
        if finalize:
            self.finalize_activation()

    @functools.singledispatchmethod
    def set_activation(self, active_els, finalize=True):
        pass

    @set_activation.register
    def _(self, active_els: np.ndarray, finalize=True) -> typing.NoReturn:
        self.active_els = active_els
        self.active_els_func.x.array.fill(0.0)
        self.active_els_func.x.array[active_els] = 1.0
        self.active_els_func.x.scatter_forward()
        self.local_active_els = self.active_els_func.x.array.nonzero()[0]
        self.local_active_els = self.local_active_els[:np.searchsorted(self.local_active_els, self.cell_map.size_local)]
        if finalize:
            self.finalize_activation()

    @set_activation.register
    def _(self, active_els: fem.Function, finalize=True) -> typing.NoReturn:
        self.active_els_func = active_els
        self.active_els_func.x.scatter_forward()
        self.active_els = self.active_els_func.x.array.nonzero()[0]
        self.local_active_els = self.active_els[:np.searchsorted(self.active_els, self.cell_map.size_local)]
        if finalize:
            self.finalize_activation()

    def update_active_dofs(self):
        old_active_dofs_array  = self.active_nodes_func.x.array.copy()
        self.active_dofs = fem.locate_dofs_topological(self.v, self.dim, self.active_els, remote=True)
        self.active_nodes_func.x.array[:] = np.float64(0.0)
        self.active_nodes_func.x.array[self.active_dofs] = np.float64(1.0)
        #self.active_nodes_func.x.scatter_forward()
        just_active_dofs_array = self.active_nodes_func.x.array - old_active_dofs_array
        self.just_activated_nodes = np.flatnonzero(just_active_dofs_array)

    def finalize_activation(self):
        self.update_active_dofs()
        self.restriction = multiphenicsx.fem.DofMapRestriction(self.v.dofmap, self.active_dofs)
        self.update_boundary()
        for v in self.form_subdomain_data.values():
            v.clear()

    def get_facets_subdomain_data(self, facets_integration_data = None, mat2itag = None):
        facet_subdomain_data = []
        if facets_integration_data is None:
            # Subtract gamma facets to boundary facets mask
            for facets in self.gamma_imap_to_global_imap.values():
                self.bfacets_mask[facets] = 0.0
            facets = self.bfacets_mask.nonzero()[0] # gamma facets already subtracted
            facets_integration_data = self.get_facet_integrations_entities(facets)
        if mat2itag is None:
            mat2itag = self.material_to_itag
        facets_boun_els = facets_integration_data[::2]
        facets_mats = self.material_id.x.array[facets_boun_els]
        for mat in self.materials:
            tag = self.material_to_tag[mat]
            ifacets = (facets_mats == tag).nonzero()[0]
            indices_integration_data = np.vstack((2*ifacets, 2*ifacets+1)).reshape(-1, order='F')
            facet_subdomain_data.append((mat2itag[mat],
                                         facets_integration_data[indices_integration_data]))
        return facet_subdomain_data

    def divide_domain_by_materials(self, els=None, dic=None):
        if dic is None:
            dic = self.local_cells
        if els is None:
            els = self.local_active_els
        for mat, tag in self.material_to_tag.items():
            dic[mat] = els[(self.material_id.x.array[els] == tag).nonzero()[0]]
        return dic

    def set_form_subdomain_data(self):
        self.divide_domain_by_materials()
        cell_subdomain_data = [(self.material_to_itag[mat], self.local_cells[mat]) for mat in self.materials]
        self.form_subdomain_data[fem.IntegralType.cell].extend(cell_subdomain_data)
        self.form_subdomain_data[fem.IntegralType.exterior_facet].extend(self.get_facets_subdomain_data())

    def update_boundary(self):
        bfacets_indices  = locate_active_boundary(self.domain, self.active_els_func)
        self.bfacets_mask = get_mask(self.num_facets,
                                     bfacets_indices)
        self.bfacets_tag  = mesh.meshtags(self.domain, self.dim-1,
                                          np.arange(self.num_facets, dtype=np.int32),
                                          self.bfacets_mask)

    def compute_gradient(self, cells: typing.Optional[npt.NDArray[np.int32]] = None):
        if not(self.is_grad_computed):
            gradient_expression = fem.Expression(ufl.grad(self.u),self.grad_u.function_space.element.interpolation_points) #TODO: compile this only one
            self.grad_u.interpolate(gradient_expression, cells0=cells)
            self.grad_u.x.scatter_forward()
            self.is_grad_computed = True

    def clear_gamma_data(self):
        self.gamma_nodes.clear()
        for f in self.gamma_nodes.items():
            f.x.array.fill(0.0)
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
        active_els, self.ext_colliding_els[p_ext] = mhs_fenicsx_cpp.deactivate_from_nodes(self.domain._cpp_object,
                                                                                      self.active_els_func._cpp_object,
                                                                                      self.ext_nodal_activation[p_ext])
        self.ext_colliding_els[p_ext] = self.ext_colliding_els[p_ext][:np.searchsorted(self.ext_colliding_els[p_ext], self.cell_map.size_local)]
        self.set_activation(active_els, finalize=finalize)
        if finalize:
            self.find_gamma(p_ext, self.ext_nodal_activation[p_ext])

    def interpolate(self, p_ext : 'Problem', dofs_to_interpolate = None, cells1 = None):
        if dofs_to_interpolate is None:
            dofs_to_interpolate = self.ext_nodal_activation[p_ext].nonzero()[0]
            dofs_to_interpolate = dofs_to_interpolate[:np.searchsorted(dofs_to_interpolate, self.domain.topology.index_map(0).size_local)]
        if cells1 is None:
            cells1 = p_ext.local_active_els
        # BUG: Nothing is being done with cells1!
        interpolate_cg1(p_ext.u,
                        self.u,
                        cells1,
                        dofs_to_interpolate,
                        self.dof_coords[dofs_to_interpolate],
                        1e-6)

    def intersect_problem(self, p_ext : 'Problem', finalize=True):
        self.ext_nodal_activation[p_ext] = self.get_active_in_external(p_ext)
        active_els, self.ext_colliding_els[p_ext] = mhs_fenicsx_cpp.intersect_from_nodes(self.domain._cpp_object,
                                                                                         self.active_els_func._cpp_object,
                                                                                         self.ext_nodal_activation[p_ext])
        self.ext_colliding_els[p_ext] = self.ext_colliding_els[p_ext][:np.searchsorted(self.ext_colliding_els[p_ext], self.cell_map.size_local)]
        self.set_activation(active_els, finalize=finalize)
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
        f = self.gamma_nodes[p_ext] if p_ext in self.gamma_nodes else None
        self.gamma_nodes[p_ext] = indices_to_function(self.v,
                                                      self.gamma_facets[p_ext].find(1),
                                                      self.dim-1,
                                                      name=f"gamma_{p_ext.name}",
                                                      f=f)
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

    def ext_conductivity(self, ext_mat : fem.Function, ext_sol : fem.Function):
        mats = self.materials
        mat_to_tag = self.material_to_tag 
        if len(self.materials) == 1:
            return mats[0].k.ufl(ext_sol)
        else:
            k = ufl.conditional(ufl.eq(mat_to_tag[mats[1]], ext_mat), mats[1].k.ufl(ext_sol), mats[0].k.ufl(ext_sol))
            for i in range(1, len(mats) - 1):
                k = ufl.conditional(ufl.eq(mat_to_tag[mats[i+1]], ext_mat), mats[i+1].k.ufl(ext_sol), k)
            return k

    def set_forms(self, material_indices={}):
        dx = ufl.Measure("dx", metadata=self.quadrature_metadata)
        u = self.u
        v = ufl.TestFunction(self.v)

        if material_indices:
            self.material_to_itag = material_indices

        a_ufl = []
        l_ufl = []
        for mat, itag in self.material_to_itag.items():
            # Conduction
            a_ufl.append(mat.k.ufl(u)*ufl.dot(ufl.grad(u), ufl.grad(v))*dx(itag))

            # Source term
            if self.rhs is not None:
                self.source.fem_function.interpolate(self.rhs)

            for source in self.sources:
                l_ufl.append(source.fem_function*v*dx(itag))

            # Time derivative
            time_derivative_coefficient = mat.rho.ufl(u) * mat.cp.ufl(u)
            advection_coefficient = mat.rho.ufl(u) * mat.cp.ufl(u)
            liquid_fraction, dliquid_fraction = mat.get_handles_liquid_fraction(self.domain, self.smoothing_cte_phase_change)
            if self.phase_change:
                advection_coefficient += mat.rho.ufl(u) * mat.L.ufl(u) * dliquid_fraction(u)
            if not(self.is_steady):
                a_ufl.append(time_derivative_coefficient * \
                        (u - self.u_prev)/self.dt_func * \
                        v * dx(itag))
                if self.phase_change:
                    a_ufl.append(mat.rho.ufl(u) * mat.L.ufl(u) * \
                            (liquid_fraction(u) - liquid_fraction(self.u_prev))/self.dt_func * \
                            v * dx(itag))

            # Translational advection
            has_advection = (np.linalg.norm(self.advection_speed.value) > 1e-7)
            if has_advection:
                advection_norm = ufl.sqrt(ufl.dot(self.advection_speed, self.advection_speed))
                a_ufl.append(advection_coefficient*ufl.dot(self.advection_speed,ufl.grad(u))*v*dx(itag))
                # Translational advection stabilization
                if self.is_supg:
                    assert(self.advected_el_size is not None)
                    supg_coeff = self.advected_el_size**2 / (2 * self.advected_el_size * \
                            advection_coefficient * advection_norm + \
                             4 * mat.k.ufl(u))
                    if not(self.is_steady):
                        a_ufl.append(supg_coeff * time_derivative_coefficient * \
                                (u - self.u_prev)/self.dt_func * \
                                advection_coefficient * ufl.dot(self.advection_speed,ufl.grad(v)) * dx(itag))
                        if self.phase_change:
                            a_ufl.append(supg_coeff * mat.rho.ufl(u) * mat.L.ufl(u) * \
                                    (liquid_fraction(u) - liquid_fraction(self.u_prev))/self.dt_func * \
                                    advection_coefficient * ufl.dot(self.advection_speed,ufl.grad(v)) * dx(itag))
                    a_ufl.append(supg_coeff * advection_coefficient * \
                            ufl.dot(self.advection_speed,ufl.grad(u)) * \
                            advection_coefficient * \
                            ufl.dot(self.advection_speed,ufl.grad(v)) * dx(itag))
                    l_ufl.append(supg_coeff * \
                            self.source.fem_function * \
                            advection_coefficient * \
                            ufl.dot(self.advection_speed,ufl.grad(v)) * dx(itag))

            # Rotational advection
            # WARNING: Stabilziation not implemented
            has_rotational_advection = np.linalg.norm(self.angular_advection_speed.value)
            if has_rotational_advection:
                x = ufl.SpatialCoordinate(self.domain)
                if self.dim < 3:
                    pointwise_advection_speed = (self.angular_advection_speed)*ufl.perp(x-self.rotation_center)
                else:
                    pointwise_advection_speed = ufl.cross(self.angular_advection_speed,x-self.rotation_center)
                a_ufl.append(advection_coefficient*ufl.dot(pointwise_advection_speed,ufl.grad(u))*v*dx(itag))

            # BOUNDARY
            ds = ufl.Measure('ds', metadata=self.quadrature_metadata)
            # CONVECTION
            if self.convection_coeff is not None:
                a_ufl.append(self.convection_coeff * \
                             u*v* \
                             ds(itag))
                l_ufl.append(self.convection_coeff * \
                             self.T_env*v* \
                             ds(itag))
            # RADIATION
            if self.radiation_coeff is not None:
                a_ufl.append(self.radiation_coeff * \
                             u**4*v* \
                             ds(itag))
                l_ufl.append(self.radiation_coeff * \
                             self.T_env**4*v* \
                             ds(itag))

        self.a_ufl = sum(a_ufl)
        self.l_ufl = sum(l_ufl)

    def compile_forms(self):
        self.r_ufl = self.a_ufl - self.l_ufl#residual
        self.j_ufl = ufl.derivative(self.r_ufl, self.u)#jacobian
        self.r_compiled = fem.compile_form(self.domain.comm, self.r_ufl,
                                            form_compiler_options={"scalar_type": np.float64})
        self.j_compiled = fem.compile_form(self.domain.comm, self.j_ufl,
                                            form_compiler_options={"scalar_type": np.float64})

    def clear_subdomain_data(self):
        for v in self.form_subdomain_data.values():
            v.clear()

    def instantiate_forms(self, clear=False):
        if clear:
            self.clear_subdomain_data()
        self.set_form_subdomain_data()
        rcoeffmap, rconstmap = get_identity_maps(self.r_ufl)
        self.r_instance = fem.create_form(self.r_compiled,
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
        self.L = multiphenicsx.fem.petsc.create_vector(self.r_instance,
                                                       self.restriction)
        self.x = multiphenicsx.fem.petsc.create_vector(self.r_instance, restriction=self.restriction)
        self._obj_vec = multiphenicsx.fem.petsc.create_vector(self.r_instance, self.restriction)
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
                                                self.r_instance,
                                                restriction=self.restriction,)
        # Dirichlet
        multiphenicsx.fem.petsc.apply_lifting(F_vec, [self.j_instance], [self.dirichlet_bcs], [self.u.x.petsc_vec], restriction=self.restriction, alpha=-1.0)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        multiphenicsx.fem.petsc.set_bc(F_vec,self.dirichlet_bcs, self.u.x.petsc_vec, restriction=self.restriction, alpha=-1.0)
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
                 self.active_els_func,
                 self.active_nodes_func,
                 self.material_id,
                 self.source.fem_function,
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

        bnodes = indices_to_function(self.v,self.bfacets_mask.nonzero()[0], self.dim-1,name="bnodes")
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
        
        #x = multiphenicsx.fem.petsc.create_vector(self.r_instance, self.restriction)
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

    def J(  # type: ignore[no-any-unimported]
        self, snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat,
        P_mat: PETSc.Mat
    ) -> None:
        """Assemble the jacobian."""
        self.assemble_jacobian(J_mat)

    def non_linear_solve(self, max_iter=50, snes_opts = {}):
        if not(self.has_preassembled):
            self.pre_assemble()
        # Solve
        snes = PETSc.SNES().create(self.domain.comm)
        snes.setTolerances(max_it=max_iter)
        opts = PETSc.Options()
        for k,v in self.linear_solver_opts.items():
            opts[k] = v
        snes.getKSP().setFromOptions()
        opts.clear()
        for k,v in snes_opts.items():
            opts[k] = v
        snes.setFromOptions()
        snes.setObjective(self.obj)
        snes.setFunction(self.R, self.L)
        snes.setJacobian(self.J, J=self.A, P=None)
        snes.setMonitor(lambda _, it, residual: print(it, residual))
        self.set_snes_sol_vector()
        snes.solve(None, self.x)
        assert (snes.getConvergedReason() > 0), f"did not converge : {snes.getConvergedReason()}"
        self._update_solution(self.x)  # TODO can this be safely removed?
        snes.destroy()
        [opts.__delitem__(k) for k in opts.getAll().keys()] # Clear options data-base
        opts.destroy()
        self.post_modify_solution()

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
