from abc import ABC, abstractmethod
import numpy as np
from mhs_fenicsx.gcode import Path, Track, gcode_to_path
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class HeatSource(ABC):
    def __init__(self,params:dict):
        '''
        TODO: Unify interface so that only initialized with Path
        '''
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

    @abstractmethod
    def __call__(self,x):
        pass
    def pre_iterate(self,tn,dt,verbose=True):
        self.x_prev[:] = self.x[:]
        if self.path is None:
            self.x += self.speed*dt
        else:
            tnp1 = tn + dt
            self.path.update(tn)
            self.x     = self.path.current_track.get_position(tnp1)
            self.speed = self.path.current_track.get_speed()
            self.power = self.path.current_track.power
            if rank==0 and verbose:
                print(f"Current track is {self.path.current_track}")

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
