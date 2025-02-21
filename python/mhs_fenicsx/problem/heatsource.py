from abc import ABC, abstractmethod
import numpy as np
from mhs_fenicsx.gcode import Path, Track, gcode_to_path, get_infinite_track
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl
from mhs_fenicsx.geometry import OBB
from mhs_fenicsx_cpp import mesh_collision
import copy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mhs_fenicsx.problem.Problem import Problem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def createHeatSources(p: 'Problem'):
    sources = []
    for hs in p.input_parameters["source_terms"]:
        try:
            hs_type = hs["type"]
            if   hs_type == 1:
                sources.append(Gaussian1D(p, hs))
            elif hs_type == 2:
                sources.append(Gaussian2D(p, hs))
            elif hs_type == 3:
                sources.append(Gaussian3D(p, hs))
            elif hs_type == 4:
                sources.append(LumpedHeatSource(p, hs))
            else:
                raise ValueError(f"Unknown heat source type {hs_type}.")
        except KeyError:
            if p.dim == 1:
                sources.append(Gaussian1D(p, hs))
            elif p.dim == 2:
                sources.append(Gaussian2D(p, hs))
            else:
                sources.append(Gaussian3D(p, hs))
    return sources

class HeatSource(ABC):
    def __init__(self, p: 'Problem', hs_params : dict):
        '''
        TODO: Unify interface so that only initialized with Path
        '''
        self.x      = np.array(hs_params["initial_position"],dtype=np.float64)
        self.R = hs_params["radius"]
        self.power = hs_params["power"]
        self.speed = np.array(hs_params["initial_speed"],dtype=np.float64)
        if "path" in hs_params:
            self.path  = gcode_to_path(hs_params["path"],default_power=self.power)
            self.x     = self.path.tracks[0].p0
            self.speed = self.path.tracks[0].get_speed()
        else:
            track = get_infinite_track(self.x, p.time, self.speed, self.power)
            self.path = Path([track])
        self.tn = self.path.tracks[0].t0
        self.initialize_fem_function(p)
        self.attributes_to_reference = set() # attributes to reference upon copy

    @abstractmethod
    def __call__(self,x):
        pass

    def set_fem_function(self, x):
        self.fem_function.x.array[:] = self(x.transpose())

    def initialize_fem_function(self,p:'Problem'):
        self.fem_function = fem.Function(p.v,name="source")

    def pre_iterate(self,tn,dt,verbose=True):
        self.tn = tn
        self.tnp1 = tn + dt
        self.path.update(tn)
        track_tnp1 = self.path.get_track(self.tnp1, pad=+1e-9)
        self.x      = track_tnp1.get_position(self.tnp1, False)
        self.speed  = track_tnp1.get_speed()
        self.power  = track_tnp1.power
        if rank==0 and verbose:
            print(f"Current track is {self.path.current_track}")

    def __deepcopy__(self,memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        all_attributes = set(self.__dict__.keys())
        attributes_to_copy = all_attributes - self.attributes_to_reference
        attributes_to_reference  = all_attributes.intersection(self.attributes_to_reference)
        for k in attributes_to_copy:
            v = self.__dict__[k]
            if isinstance(v, fem.Function):
                setattr(result, k, v.copy())
                result.__dict__[k].name = v.name
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        for k in attributes_to_reference:
            result.__dict__[k] = self.__dict__[k]
        return result

class Gaussian1D(HeatSource):
    def __call__(self,x):
        r2 = (x[0] - self.x[0])**2
        return 2 * self.power / np.pi / self.R**2 * \
            np.exp(-2*(r2)/self.R**2 )

class Gaussian2D(HeatSource):
    def __call__(self,x):
        r2 = (x[0] - self.x[0])**2 + (x[1] - self.x[1])**2
        return 2 * self.power / np.pi / self.R**2 * \
            np.exp(-2*(r2)/self.R**2 )

class Gaussian3D(HeatSource):
    def __call__(self,x):
        r2 = (x[0] - self.x[0])**2 + (x[1] - self.x[1])**2 + (x[2] - self.x[2])**2
        return 6*np.sqrt(3)*(self.power) / np.power(np.pi, 1.5) / np.power(self.R, 3) * \
            np.exp(-3*(r2)/self.R**2 )

class LumpedHeatSource(HeatSource):
    def __init__(self, p:'Problem', hs_params : dict):
        super().__init__(p, hs_params)
        self.mdwidth, self.mdheight, self.mddepth = 0.0, 0.0, 0.0
        for key in ["mdwidth", "mdheight", "mddepth"]: 
            if key in hs_params:
                self.__dict__[key] = hs_params[key]
        self.domain = p.domain
        self.bb_tree = p.bb_tree
        self.compile_volume_form()
        # attributes to reference upon copy
        self.attributes_to_reference = {"domain", "bb_tree", "volume_form_compiled"}

    def compile_volume_form(self):
        dV = ufl.Measure("dx", metadata={"quadrature_degree":1,})
        volume_form_ufl = ufl.TestFunction(self.fem_function.function_space)*dV(1)
        self.volume_form_compiled = fem.compile_form(self.domain.comm, volume_form_ufl,
                                                     form_compiler_options={"scalar_type": np.float64})
    def instantiate_volume_form(self):
        subdomain_data = {fem.IntegralType.cell : [(1, self.heated_els)]}
        coefficient_map = {self.fem_function:self.fem_function}
        return fem.create_form(self.volume_form_compiled,
                               [self.fem_function.function_space],
                               msh=self.domain,
                               subdomains=subdomain_data,
                               coefficient_map=coefficient_map,
                               constant_map={})

    def initialize_fem_function(self,p:'Problem'):
        self.fem_function = fem.Function(p.dg0,name="source")

    def set_fem_function(self, x):
        tracks = self.path.get_track_interval(self.tn, self.tnp1)
        tdim = self.domain.topology.dim
        cell_map = self.domain.topology.index_map(tdim)
        heated_els_mask = np.zeros((cell_map.size_local + cell_map.num_ghosts), dtype=np.bool_)
        for track in tracks:
            p0 = track.get_position(self.tn,   bound=True)
            p1 = track.get_position(self.tnp1, bound=True)
            obb = OBB(p0, p1, self.mdwidth, self.mdheight, self.mddepth, tdim)
            obb_mesh = obb.get_dolfinx_mesh()
            heated_els = mesh_collision(self.domain._cpp_object,obb_mesh._cpp_object,bb_tree_big=self.bb_tree._cpp_object)
            heated_els_mask[heated_els] = np.True_

        self.heated_els = heated_els_mask[:cell_map.size_local].nonzero()[0]
        heated_volume_form = self.instantiate_volume_form()
        heated_volume = fem.assemble_scalar(heated_volume_form)
        heated_volume = comm.allreduce(heated_volume, op=MPI.SUM)
        heated_volume = np.round(heated_volume,9)
        # Compute power density
        pd = self.power / heated_volume
        self.fem_function.x.array[:] = 0.0
        self.fem_function.x.array[self.heated_els] = pd

    def __call__(self,x):
        pass
