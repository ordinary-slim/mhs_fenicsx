from abc import ABC, abstractmethod
import numpy as np
from mhs_fenicsx.gcode import Path, Track, gcode_to_path
from mpi4py import MPI
from dolfinx import fem, mesh
from typing import TYPE_CHECKING
import ufl
from mhs_fenicsx.geometry import OBB
from mhs_fenicsx_cpp import mesh_collision
import copy
if TYPE_CHECKING:
    from mhs_fenicsx.problem.Problem import Problem

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# TODO: Compiled expressions here? Faster rhs assembly

class HeatSource(ABC):
    def __init__(self,p:'Problem'):
        '''
        TODO: Unify interface so that only initialized with Path
        '''
        params = p.input_parameters
        self.x      = np.array(params["heat_source"]["initial_position"],dtype=np.float64)
        self.R = params["heat_source"]["radius"]
        self.power = params["heat_source"]["power"]
        self.speed = np.array(params["heat_source"]["initial_speed"],dtype=np.float64)
        self.path  = None
        if "path" in params:
            self.path  = gcode_to_path(params["path"],default_power=self.power)
            self.x     = self.path.tracks[0].p0
            self.speed = self.path.tracks[0].get_speed()
        self.x_prev = self.x.copy()
        self.initialize_fem_function(p)

    @abstractmethod
    def __call__(self,x):
        pass

    def set_fem_function(self, x):
        self.fem_function.x.array[:] = self(x.transpose())

    def initialize_fem_function(self,p:'Problem'):
        self.fem_function = fem.Function(p.v,name="source")

    def pre_iterate(self,tn,dt,verbose=True):
        try:
            self.x_prev = self.path.current_track.get_position(tn, False)
        except:
            self.x_prev[:] = self.x[:]

        if self.path is None:
            self.x += self.speed*dt
        else:
            tnp1 = tn + dt
            self.path.update(tn)
            track_tnp1 = self.path.get_track(tnp1)
            self.x      = track_tnp1.get_position(tnp1, False)
            self.speed  = track_tnp1.get_speed()
            self.power  = track_tnp1.power
            if rank==0 and verbose:
                print(f"Current track is {self.path.current_track}")

    def __deepcopy__(self,memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, fem.Function):
                setattr(result, k, v.copy())
                result.__dict__[k].name = v.name
            else:
                setattr(result, k, copy.deepcopy(v, memo))
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
    def __init__(self,p:'Problem'):
        super().__init__(p)
        params = p.input_parameters["heat_source"]
        self.mdwidth = params["mdwidth"]
        self.mdheight = params["mdheight"]
        self.domain = p.domain
        self.bb_tree = p.bb_tree

    def initialize_fem_function(self,p:'Problem'):
        self.fem_function = fem.Function(p.dg0,name="source")
    def set_fem_function(self, x):
        # Mark heated elements
        # Collision
        obb = OBB(self.x_prev,self.x,self.mdwidth,self.mdheight, 0.0, self.domain.topology.dim)
        obb_mesh = obb.get_dolfinx_mesh()
        self.heated_els = mesh_collision(self.domain._cpp_object,obb_mesh._cpp_object,bb_tree_big=self.bb_tree._cpp_object)
        # Compute volume of heated els
        dV = ufl.Measure("dx",
                         subdomain_data=
                             mesh.meshtags(self.domain,
                                           self.domain.topology.dim,
                                           self.heated_els,
                                           np.ones_like(self.heated_els,)),
                         metadata={"quadrature_degree":1,}
                        )
        heated_volume_form = fem.form(ufl.TestFunction(self.fem_function.function_space)*dV(1))
        heated_volume = fem.assemble_scalar(heated_volume_form)
        heated_volume = comm.allreduce(heated_volume, op=MPI.SUM)
        heated_volume = np.round(heated_volume,9)
        # Compute power density
        pd = self.power / heated_volume
        self.fem_function.x.array[:] = 0.0
        self.fem_function.x.array[self.heated_els] = pd

    def __call__(self,x):
        pass
