from line_profiler import LineProfiler
from mhs_fenicsx.problem.helpers import indices_to_function
from mpi4py import MPI
from dolfinx import mesh, fem, cpp
from mhs_fenicsx.problem import Problem, get_mask
from mhs_fenicsx.drivers.staggered_drivers import StaggeredRRDriver, interpolate_dg0_cells_to_cells
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

        # MACRO-STEP
        ps.pre_iterate()
        ps.set_forms_domain()
        ps.compile_forms()
        ps.pre_assemble()
        ps.assemble()
        ps.solve()
        ps.post_iterate()

    def micro_steps(self,rr_driver:StaggeredRRDriver):
        (ps,pf) = (self.slow_problem,self.fast_problem)
        self.micro_iter = 0
        while (self.t1_macro_step - pf.time) > 1e-7:
            forced_time_derivative = (self.micro_iter==0)
            print(f"forced_time_derivative = {forced_time_derivative}")
            pf.pre_iterate(forced_time_derivative=forced_time_derivative)
            self.micro_iter += 1
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            rr_driver.relaxation_coeff[pf].value = self.fraction_macro_step
            pf.assemble()
            pf.solve()
            pf.post_iterate()


            tmp = pf.time
            pf.time = self.macro_iter*100 + self.micro_iter
            pf.writepos()
            pf.time = tmp

    def extract_subproblem(self):
        ps = self.slow_problem
        self.t0_macro_step = ps.time
        self.t1_macro_step = ps.time + ps.dt.value
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
        self.submesh_data = {}
        submesh_data = mesh.create_submesh(ps.domain,cdim,subproblem_els)
        submesh = submesh_data[0]
        self.submesh_data["subcell_map"] = submesh_data[1]
        self.submesh_data["subvertex_map"] = submesh_data[2]
        self.submesh_data["subgeom_map"] = submesh_data[3]
        micro_params = ps.input_parameters.copy()
        micro_params["dt"] = get_dt(params["micro_adim_dt"])
        self.fast_problem = Problem(submesh_data[0],micro_params, name="small")
        pf = self.fast_problem
        self.submesh_data["parent"] = ps
        self.submesh_data["child"] = pf
        self.subfacet_map = build_subentity_to_parent_mapping(cdim-1,
                                                               ps.domain,
                                                               submesh,
                                                               self.submesh_data["subcell_map"],
                                                               self.submesh_data["subvertex_map"])
        self.find_interface()
        pf.u.interpolate(ps.u_prev,cells0=self.submesh_data["subcell_map"],cells1=np.arange(len(self.submesh_data["subcell_map"])))
        self.u_prev_macro = pf.u.copy()

    def subtract_child(self):
        # Subtract child from parent
        # TODO: Make this more efficient, redundancy in setting active_els_func
        # here and inside of set_activation
        (ps,pf) = self.slow_problem, self.fast_problem
        ps.active_els_func.x.array[self.submesh_data["subcell_map"]] = 0
        ps.active_els_func.x.scatter_forward()
        active_els = ps.active_els_func.x.array.nonzero()[0]
        active_els = [el for el in active_els if el < ps.cell_map.size_local]
        ps.set_activation(active_els)

    def find_interface(self):
        (ps,pf) = self.slow_problem, self.fast_problem
        cdim = ps.domain.topology.dim
        cmask = np.zeros(pf.num_facets, dtype=np.int32)
        fast_bfacet_indices = pf.bfacets_tag.values.nonzero()[0]
        bfacets_gamma_tag = np.int32(1) - ps.bfacets_tag.values[self.subfacet_map[fast_bfacet_indices]]
        fgamma_facets_indices = fast_bfacet_indices[bfacets_gamma_tag.nonzero()[0]]
        cmask[fgamma_facets_indices] = 1
        pf.set_gamma(mesh.meshtags(pf.domain, cdim-1,
                                   np.arange(pf.num_facets, dtype=np.int32),
                                   cmask))
        pmask = np.zeros(ps.num_facets, dtype=np.int32)
        pmask[self.subfacet_map[fgamma_facets_indices]] = 1
        ps.set_gamma(mesh.meshtags(ps.domain, cdim-1,
                                   np.arange(ps.num_facets, dtype=np.int32),
                                   pmask))
        # Build interpolation data dg0 boundary
        tdim = self.fast_problem.domain.topology.dim
        ccon_f2c = self.fast_problem.domain.topology.connectivity(tdim-1,tdim)
        pcon_f2c = self.slow_problem.domain.topology.connectivity(tdim-1,tdim)
        indices_gamma_facets = pf.gamma_facets.values.nonzero()[0]
        num_gamma_facets = len(indices_gamma_facets)
        self.submesh_data["cinterface_cells"] = np.full(num_gamma_facets,-1)
        self.submesh_data["pinterface_cells"] = np.full(num_gamma_facets,-1)
        for idx in range(num_gamma_facets):
            ifacet = indices_gamma_facets[idx]
            ccell = ccon_f2c.links(ifacet)
            assert(len(ccell)==1)
            ccell = ccell[0]
            if ccell >= self.fast_problem.cell_map.size_local:
                continue
            self.submesh_data["cinterface_cells"][idx] = ccell
            pifacet = self.subfacet_map[ifacet]
            pcells = pcon_f2c.links(pifacet)
            if pcells[0] == self.submesh_data["subcell_map"][ccell]:
                self.submesh_data["pinterface_cells"][idx] = pcells[1]
            else:
                self.submesh_data["pinterface_cells"][idx] = pcells[0]

    def iterate_substepped_rr(self,rr_driver:StaggeredRRDriver):
        (ps,pf) = (self.slow_problem,self.fast_problem)

        rr_driver.update_robin(pf)
        self.micro_steps(rr_driver)
        # have solution at tnp1
        # Updated sol, grad, conduc from n+1
        # I have to apply curr sol, grad, conduc from n+1 weighted with the one from n

        rr_driver.update_robin(ps)
        ps.pre_iterate()
        ps.assemble()
        ps.solve()

    def pre_iterate(self):
        (ps,pf) = (self.slow_problem,self.fast_problem)
        self.macro_iter += 1
        for p in [ps,pf]:
            p.time = self.t0_macro_step
            p.iter = self.prev_iter[p]
            p.u_prev.x.array[:] = self.u_prev[p].x.array[:]
        # TODO: Undo pre-iterate of source
        # TODO: Undo pre-iterate of domain
        # TODO: Undo post-iterate of problem

    def pre_loop(self,rr_driver:StaggeredRRDriver):
        (ps,pf) = (self.slow_problem,self.fast_problem)
        self.macro_iter = 0
        self.prev_iter = {ps:ps.iter,pf:pf.iter}
        self.u_prev = {ps:ps.u.copy(),pf:pf.u.copy()}
        for p,p_ext in zip([pf],[ps]):
            interpolate_dg0_cells_to_cells(p_ext.grad_u,rr_driver.prev_ext_grad[p],rr_driver.active_gamma_cells[p_ext],rr_driver.active_gamma_cells[p])
            interpolate_dg0_cells_to_cells(p_ext.k,rr_driver.prev_ext_conductivity[p],rr_driver.active_gamma_cells[p_ext],rr_driver.active_gamma_cells[p])
            rr_driver.prev_ext_sol[p].interpolate(p_ext.u,cells0=rr_driver.submesh_cells[p_ext],cells1=rr_driver.submesh_cells[p])

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

    substeppin_driver = MHSSubsteppingDriver(big_p)

    substeppin_driver.extract_subproblem() # generates driver.fast_problem
    (ps,pf) = (substeppin_driver.slow_problem,substeppin_driver.fast_problem)
    rr_driver = StaggeredRRDriver(ps,pf,
                                  submesh_data=substeppin_driver.submesh_data,
                                  max_staggered_iters=3,
                                  is_relaxed=[False,True],)
    # Move extra_subproblem here
    rr_driver.pre_loop()
    substeppin_driver.pre_loop(rr_driver)
    substeppin_driver.predictor_step()
    substeppin_driver.subtract_child()
    for _ in range(rr_driver.max_staggered_iters):
        substeppin_driver.pre_iterate()
        rr_driver.pre_iterate()
        substeppin_driver.iterate_substepped_rr(rr_driver)
        rr_driver.post_iterate(verbose=True)
        #rr_driver.writepos()
        if rr_driver.convergence_crit < rr_driver.convergence_threshold:
            break
    #TODO: Interpolate solution to inactive ps
    ps.u.interpolate(pf.u,
                     cells0=np.arange(pf.num_cells),
                     cells1=substeppin_driver.submesh_data["subcell_map"])
    ps.writepos()

if __name__=="__main__":
    profiling = True
    if profiling:
        lp = LineProfiler()
        lp.add_module(MHSSubsteppingDriver)
        lp.add_module(StaggeredRRDriver)
        lp.add_function(build_subentity_to_parent_mapping)
        lp.add_function(mhs_fenicsx.geometry.mesh_collision)
        lp_wrapper = lp(main)
        lp_wrapper()
        with open(f"profiling_{rank}.txt", 'w') as pf:
            lp.print_stats(stream=pf)
