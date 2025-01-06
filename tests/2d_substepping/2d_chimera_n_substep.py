import yaml
from mpi4py import MPI
from test_2d_substepping import get_initial_condition, get_dt, write_gcode, get_mesh
from mhs_fenicsx.problem import Problem
from mhs_fenicsx.drivers import MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper, NewtonRaphson, StaggeredRRDriver
from mhs_fenicsx.submesh import build_subentity_to_parent_mapping, find_submesh_interface, \
compute_dg0_interpolation_data
from dolfinx import mesh
import numpy as np

class MHSStaggeredChimeraSubstepper(MHSStaggeredSubstepper):
    def define_subproblem(self):
        ''' Build subproblem in submesh '''
        ps = self.ps
        cdim = ps.domain.topology.dim
        subproblem_els = self.find_subproblem_els()
        # Extract subproblem:
        self.submesh_data = {}
        submesh_data = mesh.create_submesh(ps.domain,cdim,subproblem_els)
        submesh = submesh_data[0]
        self.submesh_data["subcell_map"] = submesh_data[1]
        self.submesh_data["subvertex_map"] = submesh_data[2]
        self.submesh_data["subgeom_map"] = submesh_data[3]
        micro_params = ps.input_parameters.copy()
        hs_radius = ps.source.R
        track_t0 = ps.source.path.get_track(self.t0_macro_step)
        hs_speed  = track_t0.speed# TODO: Can't use this speed!
        micro_params["dt"] = micro_params["micro_adim_dt"] * (hs_radius / hs_speed)
        micro_params["petsc_opts"] = micro_params["petsc_opts_micro"]
        self.pf = Problem(submesh_data[0],micro_params, name="small")
        pf = self.pf
        self.submesh_data["parent"] = ps
        self.submesh_data["child"] = pf
        self.submesh_data["subfacet_map"] = build_subentity_to_parent_mapping(cdim-1,
                                                               ps.domain,
                                                               submesh,
                                                               self.submesh_data["subcell_map"],
                                                               self.submesh_data["subvertex_map"])
        find_submesh_interface(ps,pf,self.submesh_data)
        compute_dg0_interpolation_data(ps,pf,self.submesh_data)
        pf.u.interpolate(ps.u,cells0=self.submesh_data["subcell_map"],cells1=np.arange(len(self.submesh_data["subcell_map"])))

    def micro_steps(self):
        (ps,pf) = (self.ps,self.pf)
        sd = self.staggered_driver
        self.micro_iter = 0
        while (self.t1_macro_step - pf.time) > 1e-7:
            forced_time_derivative = (self.micro_iter==0)
            pf.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
            self.micro_iter += 1
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            f = self.fraction_macro_step
            sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                    f*self.ext_sol_array_tnp1[:]
            sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                    f*self.ext_flux_array_tnp1[:]
            if not(pf.phase_change):
                pf.assemble()
                pf.solve()
            else:
                nr_driver = NewtonRaphson(pf)
                nr_driver.solve()
            #pf.post_iterate()
            self.micro_post_iterate()
            self.writepos(case="micro")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run_staggered_RR(params, writepos=True):
    els_per_radius = params["els_per_radius"]
    radius = params["heat_source"]["radius"]
    speed = np.linalg.norm(np.array(params["heat_source"]["initial_speed"]))
    driver_constructor = StaggeredRRDriver
    initial_relaxation_factors=[1.0,1.0]
    big_mesh = get_mesh(params, els_per_radius, radius)

    macro_params = params.copy()
    macro_params["dt"] = get_dt(params["macro_adim_dt"], radius, speed)
    macro_params["petsc_opts"] = macro_params["petsc_opts_macro"]
    big_p = Problem(big_mesh, macro_params, name=f"big_chimera_ss_RR")
    initial_condition_fun = get_initial_condition(params)
    big_p.set_initial_condition(  initial_condition_fun )

    max_timesteps = params["max_timesteps"]
    for _ in range(max_timesteps):
        substeppin_driver = MHSStaggeredChimeraSubstepper(big_p,writepos=(params["substepper_writepos"] and writepos))

        substeppin_driver.define_subproblem() # generates driver.fast_problem
        (ps,pf) = (substeppin_driver.ps,substeppin_driver.pf)
        staggered_driver = driver_constructor(pf,ps,
                                       submesh_data=substeppin_driver.submesh_data,
                                       max_staggered_iters=params["max_staggered_iters"],
                                       initial_relaxation_factors=initial_relaxation_factors,)
        substeppin_driver.set_staggered_driver(staggered_driver)

        el_density = np.round((1.0 / radius) * els_per_radius).astype(np.int32)
        h = 1.0 / el_density
        k = float(params["material_metal"]["conductivity"])
        staggered_driver.dirichlet_coeff[staggered_driver.p1] = 1.0/4.0
        staggered_driver.dirichlet_coeff[staggered_driver.p2] =  k / (2 * h)
        staggered_driver.relaxation_coeff[staggered_driver.p1].value = 3.0 / 3.0

        staggered_driver.pre_loop(prepare_subproblems=False)
        substeppin_driver.pre_loop()
        if params["predictor_step"]:
            substeppin_driver.predictor_step()
            if (substeppin_driver.do_writepos and writepos):
                substeppin_driver.writepos("predictor")
        substeppin_driver.subtract_fast()
        staggered_driver.prepare_subproblems(finalize=False)
        for _ in range(staggered_driver.max_staggered_iters):
            substeppin_driver.pre_iterate()
            staggered_driver.pre_iterate()
            substeppin_driver.iterate()
            substeppin_driver.post_iterate()
            staggered_driver.post_iterate(verbose=True)
            if writepos:
                substeppin_driver.writepos(case="macro")

            if staggered_driver.convergence_crit < staggered_driver.convergence_threshold:
                break
        substeppin_driver.post_loop()
        #TODO: Interpolate solution to inactive ps
        ps.u.interpolate(pf.u,
                         cells0=np.arange(pf.num_cells),
                         cells1=substeppin_driver.submesh_data["subcell_map"])
        if writepos:
            ps.writepos()
    return big_p


if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    run_staggered_RR(params,True)
