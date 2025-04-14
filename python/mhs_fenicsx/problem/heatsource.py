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

def create_heat_sources(p: 'Problem'):
    sources = []
    for hs in p.input_parameters["source_terms"]:
        try:
            hs_type = hs["type"]
            if   hs_type == "gaussian1d":
                sources.append(Gaussian1D(p, hs))
            elif hs_type == "gaussian2d":
                sources.append(Gaussian2D(p, hs))
            elif hs_type == "gaussian3d":
                sources.append(Gaussian3D(p, hs))
            elif hs_type == "lumped":
                sources.append(LumpedHeatSource(p, hs))
            elif hs_type == "gusarov":
                sources.append(Gusarov(p, hs))
            elif hs_type == "penetrating_gaussian":
                sources.append(PenetratingGaussian(p, hs))
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
        self.x      = np.array(hs_params["initial_position"], dtype=np.float64)
        self.R = hs_params["radius"]
        self.power = hs_params["power"]
        self.speed = np.array(hs_params["initial_speed"],dtype=np.float64)
        if "path" in hs_params:
            path = gcode_to_path(hs_params["path"],default_power=self.power)
        else:
            track = get_infinite_track(self.x, p.time, self.speed, self.power)
            path = Path([track])
        self.set_path(path)
        self.initialize_fem_function(p)
        self.attributes_to_reference = set() # attributes to reference upon copy

    @abstractmethod
    def __call__(self, x):
        pass

    def set_path(self, path: Path):
        self.path = path
        self.x = self.path.tracks[0].p0
        self.speed = self.path.tracks[0].get_speed()

    def set_fem_function(self, x):
        self.fem_function.x.array[:] = self(x.transpose())

    def initialize_fem_function(self,p:'Problem'):
        self.fem_function = fem.Function(p.v,name="source")

    def pre_iterate(self, tn, dt, verbose=True):
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
    def __call__(self, x):
        r2 = (x[0] - self.x[0])**2
        return 2 * self.power / np.pi / self.R**2 * \
            np.exp(-2*(r2)/self.R**2 )

class Gaussian2D(HeatSource):
    def __call__(self, x):
        r2 = (x[0] - self.x[0])**2 + (x[1] - self.x[1])**2
        return 2 * self.power / np.pi / self.R**2 * \
            np.exp(-2*(r2)/self.R**2 )

class Gaussian3D(HeatSource):
    def __call__(self, x):
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
        dV = ufl.Measure("dx", domain = self.domain, metadata={"quadrature_degree":1,})
        volume_form_ufl = 1*dV(1)
        self.volume_form_compiled = fem.compile_form(self.domain.comm, volume_form_ufl,
                                                     form_compiler_options={"scalar_type": np.float64})
    def instantiate_volume_form(self):
        subdomain_data = {fem.IntegralType.cell : [(1, self.heated_els)]}
        return fem.create_form(self.volume_form_compiled,
                               [],
                               msh=self.domain,
                               subdomains=subdomain_data,
                               coefficient_map={},
                               constant_map={})

    def initialize_fem_function(self,p:'Problem'):
        self.fem_function = fem.Function(p.dg0, name="source")

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
        # Compute power density
        pd = self.power / heated_volume
        self.fem_function.x.array[:] = 0.0
        self.fem_function.x.array[self.heated_els] = pd

    def __call__(self, x):
        pass

class Gusarov(HeatSource):
    def __init__(self, p:'Problem', hs_params : dict):
        assert(p.dim == 3)
        super().__init__(p, hs_params)
        self.beta = hs_params["extinction_coefficient"]
        self.rho = hs_params["hemispherical_reflectivity"]
        self.a = np.sqrt(1 - self.rho)
        self.layer_thickness = hs_params["layer_thickness"]
        self.optical_thickness = self.beta * self.layer_thickness
        a, rho, l = self.a, self.rho, self.optical_thickness
        self.D = + (1 - a) * (1 - a - rho * (1 + a)) * np.exp(-2*a*l) \
                 - (1 + a) * (1 + a - rho * (1 - a)) * np.exp(+2*a*l)

    def __call__(self, x):
        beta, a, rho = self.beta, self.a, self.rho
        l, D, R = self.optical_thickness, self.D, self.R
        r = np.sqrt((x[0] - self.x[0])**2 + (x[1] - self.x[1])**2, dtype=np.float64)
        xi = - beta * (x[2] - self.x[2])
        rfrac = r / R
        Q0 = 3 * self.power / np.pi / np.power(R, 2, dtype=np.float64) * \
                (1 - rfrac)**2 * (1 + rfrac)**2
        Q0[np.where(rfrac > 1.0)] = 0
        dqdxi1  = - (1 - a) * np.exp(-2*a*xi)
        dqdxi1 += + (1 + a) * np.exp(+2*a*xi)
        dqdxi1 *= (1 - rho**2) * np.exp(-l)
        dqdxi2  = - (1 + a - rho*(1 - a)) * np.exp(-2*a*(xi - l))
        dqdxi2 += + (1 - a - rho*(1 + a)) * np.exp(+2*a*(xi - l))
        dqdxi2 *= -(3 + rho * np.exp(-2*l))
        dqdxi3 = 3 * (1 - rho) * (np.exp(-xi) + rho * np.exp(xi - 2*l)) / (4 * rho - 3)
        dqdxi = 2*rho*np.power(a, 2) / ((4*rho - 3)*D) * (dqdxi1 + dqdxi2)
        dqdxi += dqdxi3
        dqdxi[np.where(xi > l)] = 0
        pd = (- beta * Q0 * dqdxi)
        return pd

class PenetratingGaussian(HeatSource):
    def __init__(self, p:'Problem', hs_params : dict):
        assert(p.dim == 3)
        super().__init__(p, hs_params)
        self.depth = hs_params["depth"]

    def pd_in_plane(self, x):
        r2_in_plane = (x[0] - self.x[0])**2 + (x[1] - self.x[1])**2
        return self.power / 2 / np.pi / self.R**2 * np.exp( - r2_in_plane / 2 / self.R**2 )

    def pd_depth(self, x):
        z = np.sqrt((x[2] - self.x[2])**2)
        return 2 / np.sqrt(2 * np.pi) / self.depth * np.exp( - z**2 / 2 / self.depth** 2)

    def __call__(self, x):
        return self.pd_depth(x) * self.pd_in_plane(x)
