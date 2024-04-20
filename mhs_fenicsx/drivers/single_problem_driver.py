import typing
from mhs_fenicsx.problem import Problem, HeatSource
from mhs_fenicsx.gcode import TrackType
from dolfinx import fem
import numpy as np

class SingleProblemDriver:
    '''
    Driver for heat equation defined on a single domain
    '''
    def __init__(self,p:Problem,params:dict):
        self.p = p
        self.printing_dt = params["dt"]
        if "cooling_dt" in params:
            self.cooling_dt  = params["cooling_dt"]
        else:
            self.cooling_dt  = self.printing_dt
        self.dt = self.printing_dt
        self.p.set_initial_condition(params["environment_temperature"])
        self.deactivate_below_surface()
        self.p.set_forms_domain()
        self.p.set_forms_boundary()
        p.compile_forms()

    def set_dt(self):
        next_track = self.p.source.path.get_track(self.p.time)
        if self.p.source.path is not None:
            max_dt = next_track.t1 - self.p.time
            if next_track.type is TrackType.PRINTING:
                dt = self.printing_dt
            else:
                dt = self.cooling_dt
            if max_dt > 1e-7:
                self.dt = min(dt,max_dt)
            self.p.dt = self.dt
    def pre_iterate(self):
        self.set_dt()
        self.p.pre_iterate()
    def iterate(self):
        self.p.assemble()
        self.p.solve()
    def post_iterate(self):
        self.p.post_iterate()
    def deactivate_below_surface(self):
        active_els = fem.locate_dofs_geometrical(self.p.dg0_bg, lambda x : x[self.p.domain.topology.dim-1] < 0.0 )
        self.p.set_activation(active_els)
