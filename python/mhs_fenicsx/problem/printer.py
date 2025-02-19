from mhs_fenicsx.geometry import OBB
from mhs_fenicsx_cpp import mesh_collision
from typing import TYPE_CHECKING
from mhs_fenicsx.problem.heatsource import LumpedHeatSource
from abc import ABC, abstractmethod
from mhs_fenicsx.gcode import TrackType
from dolfinx import mesh
if TYPE_CHECKING:
    from mhs_fenicsx.problem.Problem import Problem

def createPrinter(p: 'Problem'):
    params = p.input_parameters["printer"]
    printer_type = params["type"]
    if printer_type == 'DED':
        return DEDPrinter(p)
    elif printer_type == 'LPBF':
        return LPBFPrinter(p)
    else:
        raise KeyError('Wrong printer type')

class Printer(ABC):
    def __init__(self, params):
        self.p : Problem
        self.T_dep = params["deposition_temperature"]

    def activate(self, finalize=True):
        p = self.p
        p.set_activation(p.active_els_func, finalize=finalize)
        if not(finalize):
            p.update_active_dofs()
        p.u.x.array[p.just_activated_nodes] = self.T_dep

class DEDPrinter(Printer):
    def __init__(self, p: 'Problem'):
        params = p.input_parameters["printer"]
        super().__init__(params)
        self.path = p.source.path
        self.mdwidth = params["mdwidth"]
        self.mdheight = params["mdheight"]
        self.reuse_heated_els = params["reuse_heated_els"]
        self.p = p

    def deposit(self, tn:float, dt:float, finalize=True):
        p = self.p
        if not(self.reuse_heated_els):
            tnp1 = tn + dt
            tracks = self.path.get_track_interval(tn, tnp1)
            tdim = p.domain.topology.dim
            for track in tracks:
                x = [track.get_position(t, bound=True) for t in [tn, tnp1]]
                obb = OBB(x[0], x[1], self.mdwidth, self.mdheight, 0.0, tdim)
                obb_mesh = obb.get_dolfinx_mesh()
                deposited_els = mesh_collision(p.domain._cpp_object,obb_mesh._cpp_object,bb_tree_big=p.bb_tree._cpp_object)
                p.active_els_func.x.array[deposited_els] = 1.0
        else:
            assert(type(p.source) == LumpedHeatSource)
            p.active_els_func.x.array[p.source.heated_els] = 1.0
        self.activate(finalize=finalize)

class LPBFPrinter(Printer):
    def __init__(self, p: 'Problem'):
        params = p.input_parameters["printer"]
        super().__init__(params)
        self.path = p.source.path
        self.p = p
        self.tdim = p.domain.topology.dim

    def deposit(self, tn:float, dt:float, finalize=True):
        p = self.p
        tnp1 = tn + dt
        tracks = self.path.get_track_interval(tn, tnp1)
        tdim = p.domain.topology.dim
        recoating_tracks = [track for track in tracks if track.type == TrackType.RECOATING]
        if recoating_tracks:
            inactive_els = (p.active_els_func.x.array == 0).nonzero()[0]
            midpoints_inactive_cells = mesh.compute_midpoints(p.domain, self.tdim, inactive_els)
            for track in recoating_tracks:
                height = track.get_position(tnp1,   bound=True)[self.tdim-1]
                deposited_els = inactive_els[(midpoints_inactive_cells[:, tdim-1] <= height).nonzero()[0]]
                p.active_els_func.x.array[deposited_els] = 1.0
            self.activate(finalize=finalize)
