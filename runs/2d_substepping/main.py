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

def build_subentity_to_parent_mapping(edim:int,pdomain:mesh.Mesh,cdomain:mesh.Mesh,subcell_map,subvertex_map):
    tdim = cdomain.topology.dim
    centity_map = cdomain.topology.index_map(edim)
    num_cents = centity_map.size_local + centity_map.num_ghosts
    subentity_map = np.full(num_cents,-1,dtype=np.int32)

    cdomain.topology.create_connectivity(edim,tdim)
    ccon_e2c = cdomain.topology.connectivity(edim,tdim)
    ccon_e2v = cdomain.topology.connectivity(edim,0)

    pdomain.topology.create_connectivity(tdim,edim)
    pdomain.topology.create_connectivity(edim,tdim)
    pcon_e2v = pdomain.topology.connectivity(edim,0)
    pcon_c2e = pdomain.topology.connectivity(tdim,edim)

    for ient in range(num_cents):
        entity_found = False
        cicells = ccon_e2c.links(ient)#child incident cells
        picells = subcell_map[cicells]#parent incident cells
        pinodes = subvertex_map[ccon_e2v.links(ient)]#parent incident nodes
        pinodes.sort()
        for picell in picells:
            if not(entity_found):
                for pient in pcon_c2e.links(picell):
                    pinodes2compare = pcon_e2v.links(pient).copy()
                    pinodes2compare.sort()
                    entity_found = (pinodes==pinodes2compare).all()
                    if entity_found:
                        subentity_map[ient] = pient
                        break
    return subentity_map


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

        #ps.writepos()

        self.t1_macro_step = ps.time

    def extract_subproblem(self):
        ps = self.slow_problem
        cdim = ps.dim
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
        obb = mhs_fenicsx.geometry.OBB(p0,p1,width=pad,height=pad,depth=pad,dim=cdim,
                                       shrink=False)
        obb_mesh = obb.get_dolfinx_mesh()
        subproblem_els = mhs_fenicsx.geometry.mesh_collision(ps.domain,obb_mesh,bb_tree_mesh_big=ps.bb_tree)
        # Extract subproblem:
        submesh, self.subcell_map, self.subvertex_map, self.subgeom_map = mesh.create_submesh(ps.domain,cdim,subproblem_els)
        micro_params = ps.input_parameters.copy()
        micro_params["dt"] = get_dt(params["micro_adim_dt"])
        self.fast_problem = Problem(submesh,micro_params, name="small")
        pf = self.fast_problem
        self.subfacet_map = build_subentity_to_parent_mapping(cdim-1,
                                                               ps.domain,
                                                               submesh,
                                                               self.subcell_map,
                                                               self.subvertex_map)
        self.find_interface()
        self.bc_from_predictor = fem.Function(self.fast_problem.v,name="predictor_bc")
        # Extract necessary functions
        self.inherit_functions([self.bc_from_predictor,pf.u],[ps.u,ps.u_prev])
        self.u_prev_macro = pf.u.copy()

    def inherit_functions(self,fr_sub:list[fem.Function],fs_par:list[fem.Function]):
        for fr,fs in zip(fr_sub,fs_par):
            fr.interpolate(fs,cells0=self.subcell_map,cells1=np.arange(len(self.subcell_map)))

    def find_interface(self):
        (ps,pf) = self.slow_problem, self.fast_problem
        cdim = ps.domain.topology.dim
        mask = np.zeros(pf.num_facets, dtype=np.int32)
        fast_bfacet_indices = pf.bfacets_tag.values.nonzero()[0]
        bfacets_gamma_tag = np.int32(1) - ps.bfacets_tag.values[self.subfacet_map[fast_bfacet_indices]]
        mask[fast_bfacet_indices[bfacets_gamma_tag.nonzero()[0]]] = 1
        self.gamma_fast =  mesh.meshtags(pf.domain, cdim-1,
                                         np.arange(pf.num_facets, dtype=np.int32),
                                         mask)
        # Build interpolation data dg0 boundary
        tdim = self.fast_problem.domain.topology.dim
        ccon_f2c = self.fast_problem.domain.topology.connectivity(tdim-1,tdim)
        pcon_f2c = self.slow_problem.domain.topology.connectivity(tdim-1,tdim)
        indices_gamma_facets = self.gamma_fast.values.nonzero()[0]
        num_gamma_facets = len(indices_gamma_facets)
        self.cinterface_cells = np.full(num_gamma_facets,-1)
        self.pinterface_cells = np.full(num_gamma_facets,-1)
        for idx in range(num_gamma_facets):
            ifacet = indices_gamma_facets[idx]
            ccell = ccon_f2c.links(ifacet)
            assert(len(ccell)==1)
            ccell = ccell[0]
            if ccell >= self.fast_problem.cell_map.size_local:
                continue
            self.cinterface_cells[idx] = ccell
            pifacet = self.subfacet_map[ifacet]
            pcells = pcon_f2c.links(pifacet)
            if pcells[0] == self.subcell_map[ccell]:
                self.pinterface_cells[idx] = pcells[1]
            else:
                self.pinterface_cells[idx] = pcells[0]
        # TODO: Mark interface in parent mesh

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
        #BDEBUG
        #TODO: Driver writepos
        gamma_nodes = indices_to_function(pf.v,
                                    self.gamma_fast.find(1),
                                    1,
                                    name="gamma_nodes",)
        #EDEBUG
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
            pf.writepos(extra_funcs=[gamma_nodes,self.bc_from_predictor])

def interpolate_dg0_cells_to_cells(fs,fr,scells,rcells):
    block_size = fr.function_space.value_size
    for rcell,scell in zip(rcells,scells):
        fr.x.array[rcell*block_size:rcell*block_size+block_size] = fs.x.array[scell*block_size:scell*block_size+block_size]
    fr.x.scatter_forward()

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
    driver.micro_steps()
    big_p.writepos()


    # Interpolate
    big_p.u.interpolate(lambda x : np.power(x[0],2))
    grad_u_fast = driver.fast_problem.grad_u
    big_p.is_grad_computed = False
    big_p.compute_gradient()
    interpolate_dg0_cells_to_cells(big_p.grad_u,grad_u_fast,driver.pinterface_cells,driver.cinterface_cells)
    grad_u_fast.is_grad_computed = True

    driver.fast_problem.time = 999
    big_p.time = 999
    driver.fast_problem.writepos(extra_funcs=[grad_u_fast])
    big_p.writepos()


if __name__=="__main__":
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(MHSSubsteppingDriver)
        lp.add_function(build_subentity_to_parent_mapping)
        lp.add_function(mhs_fenicsx.geometry.mesh_collision)
        lp_wrapper = lp(main)
        lp_wrapper()
        with open(f"profiling_{rank}.txt", 'w') as pf:
            lp.print_stats(stream=pf)
