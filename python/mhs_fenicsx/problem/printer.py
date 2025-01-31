from mhs_fenicsx.geometry import OBB
from mhs_fenicsx_cpp import mesh_collision
from typing import TYPE_CHECKING
from mhs_fenicsx.problem.heatsource import LumpedHeatSource
if TYPE_CHECKING:
    from mhs_fenicsx.problem.Problem import Problem

class Printer:
    def __init__(self, p: 'Problem'):
        params = p.input_parameters["printer"]
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
                p0 = track.get_position(tn,   bound=True)
                p1 = track.get_position(tnp1, bound=True)
                obb = OBB(p0, p1, self.mdwidth, self.mdheight, 0.0, tdim)
                obb_mesh = obb.get_dolfinx_mesh()
                deposited_els = mesh_collision(p.domain._cpp_object,obb_mesh._cpp_object,bb_tree_big=p.bb_tree._cpp_object)
                p.active_els_func.x.array[deposited_els] = 1.0
        else:
            assert(type(p.source) == LumpedHeatSource)
            p.active_els_func.x.array[p.source.heated_els] = 1.0
        p.set_activation(p.active_els_func, finalize=finalize)
        p.u.x.array[p.just_activated_nodes] = p.T_dep
