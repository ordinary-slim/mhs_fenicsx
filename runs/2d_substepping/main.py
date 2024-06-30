from line_profiler import LineProfiler
from mhs_fenicsx.problem.helpers import indices_to_function
from mpi4py import MPI
from dolfinx import mesh, fem, cpp
from mhs_fenicsx.problem import Problem, get_mask
import mhs_fenicsx.geometry
import yaml
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)
radius = params["heat_source"]["radius"]
speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))

def get_dt(adim_dt):
    return adim_dt * (radius / speed)

def write_gcode():
    L = params["L"]
    half_len = L / 2.0
    gcode_lines = []
    gcode_lines.append(f"G0 F{speed} X{0.5-half_len} Y1 Z0\n")
    gcode_lines.append(f"G1 X{0.5+half_len} E0.1\n")
    with open(params["path"],'w') as f:
        f.writelines(gcode_lines)

class MHSSubsteppingDriver:
    def __init__(self,macro_problem:Problem):
        self.slow_problem = macro_problem
        self.fraction_macro_step = 0

    def predictor_step(self):
        ps = self.slow_problem

        self.t0_macro_step = ps.time

        # MACRO-STEP
        ps.pre_iterate()
        ps.set_forms_domain()
        ps.compile_forms()
        ps.pre_assemble()
        ps.assemble()
        ps.solve()
        ps.post_iterate()

        ps.writepos()

        self.t1_macro_step = ps.time

    def extract_subproblem(self):
        ps = self.slow_problem
        # Determine geometry of subproblem
        # To do it properly, get initial time of macro step, final time of macro step
        # Do collision tests across track and accumulate elements to extract
        # Here we just do a single collision test
        initial_track_macro_step = ps.source.path.get_track(self.t0_macro_step)
        assert(initial_track_macro_step==ps.source.path.current_track)
        pad = 3*radius
        p0 = initial_track_macro_step.get_position(self.t0_macro_step)
        p1 = initial_track_macro_step.get_position(self.t1_macro_step)
        direction = initial_track_macro_step.get_direction()
        p0 -= direction*pad
        p1 += direction*pad
        obb = mhs_fenicsx.geometry.OBB(p0,p1,width=pad,height=pad,depth=pad,dim=ps.dim,
                                       shrink=False)
        obb_mesh = obb.get_dolfinx_mesh()
        subproblem_els = mhs_fenicsx.geometry.mesh_collision(ps.domain,obb_mesh,bb_tree_mesh_big=ps.bb_tree)
        # Extract subproblem: TODO: Extract necessary functions
        submesh, self.sub2parent_cell_map, self.vertex_map, self.xdof_map = mesh.create_submesh(ps.domain,ps.dim,subproblem_els)
        micro_params = ps.input_parameters.copy()
        micro_params["dt"] = get_dt(params["micro_adim_dt"])
        self.fast_problem = Problem(submesh,micro_params, name="small")
        pf = self.fast_problem
        self.bc_from_predictor = fem.Function(self.fast_problem.v,name="predictor_bc")
        self.inherit_functions([self.bc_from_predictor,pf.u],
                               [ps.u,ps.u_prev])
        self.u_prev_macro = pf.u.copy()

    def find_interface(self):
        (ps,pf) = self.slow_problem, self.fast_problem
        cdim = ps.domain.topology.dim
        bnodes_parent = mesh.compute_incident_entities(ps.domain.topology,
                                                       ps.bfacets_tag.values.nonzero()[0],
                                                       ps.domain.topology.dim-1,
                                                       0,)
        bnodes_tag  = mesh.meshtags(ps.domain, 0,
                      np.arange(ps.num_nodes, dtype=np.int32),
                      get_mask(ps.num_nodes,
                               bnodes_parent, dtype=np.int32),)

        mask = np.zeros(pf.num_facets, dtype=np.int32)

        pf.domain.topology.create_connectivity(cdim-1,0)
        con_facet_vertex_child = pf.domain.topology.connectivity(cdim-1,0)
        for ifacet in pf.bfacets_tag.find(1):
            incident_nodes = con_facet_vertex_child.links(ifacet)
            incident_nodes_par = self.vertex_map[incident_nodes]
            for inode in incident_nodes_par:
                if bnodes_tag.values[inode] == 0:
                    mask[ifacet] = 1
                    break

        self.gamma_fast =  mesh.meshtags(pf.domain, cdim-1,
                                         np.arange(pf.num_facets, dtype=np.int32),
                                         mask)
    def inherit_functions(self,fs_sub:list[fem.Function],fs_par:list[fem.Function]):
        (ps,pf) = (self.slow_problem,self.fast_problem)
        num_local_cells = pf.cell_map.size_local
        for cell in range(num_local_cells):
            sub_dofs = pf.v.dofmap.cell_dofs(cell)
            parent_dofs = ps.v.dofmap.cell_dofs(self.sub2parent_cell_map[cell])
            for parent, child in zip(parent_dofs,sub_dofs):
                for b in range(pf.v.dofmap.bs):
                    self.bc_from_predictor.x.array[child*pf.v.dofmap.bs+b] = ps.u.x.array[parent*ps.v.dofmap.bs+b]
        self.bc_from_predictor.x.scatter_forward()

    def set_dirichlet_interface(self):
        pf = self.fast_problem
        # Get Gamma DOFS right
        dofs_gamma_fast = fem.locate_dofs_topological(pf.v,pf.dim-1,self.gamma_fast.find(1))
        self.dirichlet_gamma = fem.Function(pf.v,name="dirichlet_gamma")
        # Set Gamma dirichlet
        self.dirichlet_micro_steps = pf.add_dirichlet_bc(self.dirichlet_gamma,bdofs=dofs_gamma_fast, reset=False)

    def update_dirichlet_interface(self):
        #  Update values of self.dirichlet_gamma depending on time of pf
        self.dirichlet_micro_steps.g.x.array[:] = (1-self.fraction_macro_step)*self.u_prev_macro.x.array[:] + self.fraction_macro_step*self.bc_from_predictor.x.array[:]

    def micro_steps(self):
        (ps,pf) = (self.slow_problem,self.fast_problem)
        # Time-loop subproblem
        self.set_dirichlet_interface()
        pf.set_forms_domain()
        pf.compile_forms()
        # MICRO-STEP
        while (ps.time - pf.time) > 1e-7:
            # TODO: Update it here
            pf.pre_iterate()
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            self.update_dirichlet_interface()
            pf.pre_assemble()
            pf.assemble()
            pf.solve()
            pf.post_iterate()
            print(f"Percentage of completion of micro-steps: {100*self.fraction_macro_step}%")
            pf.writepos(extra_funcs=[self.bc_from_predictor])


def main():
    write_gcode()
    els_per_radius = params["els_per_radius"]
    points_side = np.round(1.0 / radius * els_per_radius).astype(int) + 1
    big_mesh  = mesh.create_unit_square(MPI.COMM_WORLD,
                                        points_side,
                                        points_side,
                                        #mesh.CellType.quadrilateral,
                                        )

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"])
    big_p = Problem(big_mesh, macro_params, name="big")

    driver = MHSSubsteppingDriver(big_p)
    driver.predictor_step()
    driver.extract_subproblem() # generates driver.fast_problem
    driver.find_interface()
    driver.micro_steps()

    # TODO: Sketch out last corrector step

if __name__=="__main__":
    main()
