import typing
from mhs_fenicsx.problem import Problem, HeatSource
from mhs_fenicsx.gcode import TrackType
from mhs_fenicsx.geometry import OBB, mesh_collision
from dolfinx import fem, mesh
from dolfinx import io
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class SingleProblemDriver:
    '''
    Driver for heat equation defined on a single domain
    '''
    def __init__(self,p:Problem,params:dict):
        self.p = p
        # PRINT SETTINGS
        if "print" in params:
            self.print_type = params["print"]["type"]
            self.hatch_width = params["print"]["width"]
            self.hatch_height = params["print"]["height"]
            self.hatch_depth = params["print"]["depth"]
        else:
            self.print_type = "OFF"
        self.printing_dt = params["dt"]
        self.track_tn = None
        self.layer_counter = -1
        self.is_new_track = False
        if "cooling_dt" in params:
            self.cooling_dt  = params["cooling_dt"]
        else:
            self.cooling_dt  = self.printing_dt
        self.dt = self.printing_dt
        self.p.set_initial_condition(params["environment_temperature"])
        if not(self.print_type.startswith("OFF")):
            self.deactivate_below_surface(set_inactive_to_powder=True)
        self.p.set_forms_domain()
        self.p.set_forms_boundary()
        p.compile_create_forms()

    def set_dt(self):
        track_tn = self.p.source.path.get_track(self.p.time)
        if track_tn != self.track_tn:
            self.is_new_track = True
        self.track_tn = track_tn
        max_dt = self.track_tn.t1 - self.p.time
        if self.track_tn.type is TrackType.PRINTING:
            dt = self.printing_dt
        else:
            dt = self.cooling_dt
        if max_dt > 1e-7:
            self.dt = min(dt,max_dt)
        self.p.dt.value = self.dt

    def on_new_track_operations(self):
        if self.track_tn.type is TrackType.RECOATING:
            self.deposit_new_layer()
            self.layer_counter += 1

    def pre_iterate(self):
        self.is_new_track = False
        self.set_dt()
        if self.is_new_track:
            self.on_new_track_operations()
        if self.track_tn.type is TrackType.PRINTING:
            self.hatch_to_metal()
        self.p.pre_iterate()

    def hatch_to_metal(self):
        #TODO: Define hatch
        x0 = self.p.source.x
        x1 = self.p.source.x + self.track_tn.get_speed()*self.dt
        obb = OBB(x0,x1,self.hatch_width,self.hatch_height,
                  self.hatch_depth,self.p.dim)
        obb_mesh = obb.get_dolfinx_mesh()
        new_metal_els = mesh_collision(self.p.domain,obb_mesh,bb_tree_mesh_big=self.p.bb_tree)
        self.p.update_material_funcs(new_metal_els,0)

    def iterate(self):
        self.p.assemble()
        self.p.solve()

    def post_iterate(self):
        self.p.post_iterate()

    def deactivate_below_surface(self, set_inactive_to_powder=False):
        active_els = fem.locate_dofs_geometrical(self.p.dg0, lambda x : x[self.p.domain.topology.dim-1] < 0.0 )
        self.p.set_activation(active_els)
        if set_inactive_to_powder:
            assert len(self.p.materials)>1, "At least 2 materials for LPBF simulation."
            powder_els = np.flatnonzero(np.round(np.float64(1)-self.p.active_els_func.x.array))
            self.p.update_material_funcs(powder_els,1)


    def deposit_new_layer(self):
        dim = self.p.domain.topology.dim
        inactive_els_indices = self.p.active_els_tag.find(0)
        height_midpoints_inactive = mesh.compute_midpoints(self.p.domain,
                                                           dim,
                                                           inactive_els_indices)[:,dim-1]
        activation_height = self.track_tn.p1[dim-1] + self.hatch_height
        els_to_activate = inactive_els_indices[np.flatnonzero(height_midpoints_inactive<activation_height)]
        self.p.set_activation(np.concatenate((self.p.active_els_tag.find(1), els_to_activate)))
        self.p.u.x.array[self.p.just_activated_nodes] = self.p.T_dep
        self.p.u_prev.x.array[self.p.just_activated_nodes] = self.p.T_dep
        # TODO: Move this
        self.p.set_forms_domain()
        self.p.set_forms_boundary()
        self.p.compile_create_forms()
