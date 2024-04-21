import typing
from mhs_fenicsx.problem import Problem, HeatSource
from mhs_fenicsx.gcode import TrackType
from dolfinx import fem, mesh
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
        if self.p.source.path is not None:
            self.next_track = self.p.source.path.get_track(self.p.time)
            max_dt = self.next_track.t1 - self.p.time
            if self.next_track.type is TrackType.PRINTING:
                dt = self.printing_dt
            else:
                dt = self.cooling_dt
            if max_dt > 1e-7:
                self.dt = min(dt,max_dt)
            self.p.dt.value = self.dt

    def on_new_track_operations(self):
        if self.p.source.path is not None:
            if self.next_track.type is TrackType.RECOATING:
                self.deposit_new_layer()

    def pre_iterate(self):
        self.set_dt()
        if self.p.source.path.is_new_track:
            self.on_new_track_operations()
        self.p.pre_iterate()

    def iterate(self):
        self.p.assemble()
        self.p.solve()

    def post_iterate(self):
        self.p.post_iterate()

    def deactivate_below_surface(self):
        active_els = fem.locate_dofs_geometrical(self.p.dg0_bg, lambda x : x[self.p.domain.topology.dim-1] < 0.0 )
        self.p.set_activation(active_els)

    def deposit_new_layer(self):
        dim = self.p.domain.topology.dim
        inactive_els_indices = self.p.active_els_tag.find(0)
        height_midpoints_inactive = mesh.compute_midpoints(self.p.domain,
                                                           dim,
                                                           inactive_els_indices)[:,dim-1]
        els_to_activate = inactive_els_indices[np.flatnonzero(height_midpoints_inactive<self.next_track.p1[dim-1])]
        self.p.set_activation(np.concatenate((self.p.active_els_tag.find(1), els_to_activate)))
        self.p.u.x.array[self.p.just_activated_nodes] = self.p.T_dep
        self.p.u_prev.x.array[self.p.just_activated_nodes] = self.p.T_dep
        # TODO: Move this
        self.p.set_forms_domain()
        self.p.set_forms_boundary()
        self.p.compile_forms()
