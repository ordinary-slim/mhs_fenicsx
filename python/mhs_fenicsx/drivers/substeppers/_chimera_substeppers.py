from mhs_fenicsx.problem import Problem, L2Differ
from mhs_fenicsx.problem.helpers import interpolate_cg1, interpolate_dg0
from mhs_fenicsx.drivers.substeppers import MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper
from mhs_fenicsx.drivers._robin_drivers import MonolithicRRDriver, StaggeredRRDriver
from mhs_fenicsx.drivers._staggered_interp_drivers import StaggeredInterpDDDriver
import mhs_fenicsx_cpp
import ufl
import numpy as np
from dolfinx import fem
from mhs_fenicsx.chimera import interpolate_solution_to_inactive, shape_moving_problem
import multiphenicsx
from petsc4py import PETSc
import typing
import numpy.typing as npt
from abc import ABC, abstractmethod
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def check_assumptions(ps, pf, pm):
    return not(np.logical_and(pf.gamma_facets[pm].values, ps.bfacets_tag.values).all())

class ChimeraSubstepper(ABC):
    @abstractmethod
    def __init__(self):
        self.pf : Problem
        self.pm : Problem
        self.fast_subproblem_els : npt.NDArray[np.int32]
        self.params : dict
        self.t0_macro_step : float
        self.t1_macro_step : float
        self.fraction_macro_step : float

    def chimera_post_init(self, initial_orientation : npt.NDArray):
        driver_type = self.params["chimera_driver"].get("type", "monolithic")
        gamma_coeff1 = self.params["chimera_driver"].get("gamma_coeff1", 1.0)
        gamma_coeff2 = self.params["chimera_driver"].get("gamma_coeff2", 1.0)
        DriverClass = StaggeredRRDriver if driver_type == "staggered" else MonolithicRRDriver

        self.chimera_driver = DriverClass(self.pf, self.pm, gamma_coeff1, gamma_coeff2)
        self.chimera_always_on = self.params["chimera_always_on"]
        self.chimera_on = self.chimera_always_on
        self.current_orientation = initial_orientation.astype(np.float64)
        # Overwrite methods
        def post_step_wo_substepping():
            super(type(self), self).post_step_wo_substepping()
            self.chimera_post_step_without_substepping()
        def prepare_micro_step():
            self.chimera_prepare_micro_step()

        def micro_pre_iterate():
            pf = self.pf
            forced_time_derivative = (pf.time - self.t0_macro_step) < 1e-9
            self.chimera_micro_pre_iterate(forced_time_derivative=forced_time_derivative)
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)

        def micro_post_iterate():
            self.chimera_micro_post_iterate()
        self.prepare_micro_step = prepare_micro_step
        self.post_step_wo_substepping = post_step_wo_substepping
        self.micro_pre_iterate = micro_pre_iterate
        self.micro_post_iterate = micro_post_iterate
        # Steadiness workflow
        self.steadiness_workflow_params = self.params["chimera_steadiness_workflow"]
        self.steadiness_threshold = self.steadiness_workflow_params.get("threshold", 0.15)
        self.min_steps_at_dt = int(self.steadiness_workflow_params.get("min_steps_at_dt", 1))
        self.steadiness_metric = L2Differ(self.pm)
        self.steadiness_measurements = []
        # On new substepping cycle, reuse last dt of previous cycle
        self.reuse_previous_dt = self.steadiness_workflow_params.get(
            "reuse_last_dt", False)
        self.last_dt = np.float64(-1.0)

    def chimera_post_step_without_substepping(self):
        (pf, pm) = self.pf, self.pm
        pm.intersect_problem(pf, finalize=False)
        pm.interpolate(pf)

    def chimera_prepare_micro_step(self):
        (pf, pm) = self.pf, self.pm
        super(type(self), self).prepare_micro_step()
        next_track = self.pf.source.path.get_track(pf.time)
        self.direction_change = False
        self.unsteady_reset = False
        self.rotation_angle = 0.0
        if not(next_track == pf.source.path.current_track):
            d0 = self.current_orientation
            d1 = next_track.get_direction()
            self.rotation_angle = np.arccos(d0.dot(d1))
            if (abs(self.rotation_angle) > 1e-9):
                self.direction_change = True
            self.current_orientation = d1

        if self.direction_change:
            pm.in_plane_rotation(pf.source.x, self.rotation_angle)
            self.unsteady_reset = True

        if self.unsteady_reset:
            self.steadiness_measurements.clear()
            self.chimera_on = self.chimera_always_on
        else:
            self.is_steady = self.is_steady_enough()
            if (self.is_steady) and not(self.chimera_on):
                self.chimera_on = True
                self.is_steady = False

        self.chimera_set_timesteps()

    def chimera_set_timesteps(self):
        (pf, pm) = self.pf, self.pm
        next_track = self.pf.source.path.get_track(pf.time)

        dt = pf.dt.value
        if self.unsteady_reset:
            # Reset timestep to finest
            dt = pf.dimensionalize_mhs_timestep(next_track, self.params["micro_adim_dt"])
        else:
            increasing_dt = self.params["chimera_steadiness_workflow"]["enabled"]
            if increasing_dt:
                if np.isclose(self.fraction_macro_step, 0.0) and self.reuse_previous_dt \
                     and (self.last_dt > 0.0):
                    dt = self.last_dt
                if self.is_steady:
                    increment = pf.dimensionalize_mhs_timestep(next_track, self.params["chimera_steadiness_workflow"]["adim_dt_increment"])
                    dt = min(pf.dimensionalize_mhs_timestep(next_track, self.params["chimera_steadiness_workflow"]["max_adim_dt"]), dt + increment)

        # Compute max allowed dt
        max_dt_substep = self.t1_macro_step - pf.time
        if isinstance(self, MHSSemiMonolithicSubstepper):
            # Leave room for monolithic step
            max_dt_substep -= pf.dt.value
        max_dt_track = next_track.t1 - pf.time
        max_dt = min(max_dt_track, max_dt_substep)
        assert(max_dt > 0.0)

        if dt > max_dt:
            dt = max_dt
        else:
            # Don't save dt as last_dt if cut by max_dt
            self.last_dt = np.float64(dt)

        for p in [pf, pm]:
            p.set_dt(dt)

    def is_steady_enough(self):
        if self.params["chimera_steadiness_workflow"]["enabled"]:
            yes = (len(self.steadiness_measurements) > self.min_steps_at_dt) and \
                    (((self.steadiness_measurements[-1] - self.steadiness_measurements[-2]) / self.steadiness_measurements[-2]) < self.steadiness_threshold)
            if yes:
                self.steadiness_measurements.clear()
            return yes
        else:
            next_track = self.pf.source.path.get_track(self.pf.time)
            return (((self.pf.time - next_track.t0) / (next_track.t1 - next_track.t0)) >= self.steadiness_threshold)

    def chimera_micro_pre_iterate(self, forced_time_derivative=False, compute_robin_coupling_data=True):
        # TODO: Move this to Chimera driver?
        (pf, pm) = self.pf, self.pm
        pm_interp_done = False
        prev_pm_active_nodes_mask = pm.active_nodes_func.x.array.copy()
        shape_moving_problem(pm)
        pm.intersect_problem(pf, finalize=False)
        pm.update_active_dofs()

        # Advance pf in time, straightforward
        pf.pre_iterate(forced_time_derivative=forced_time_derivative, verbose=False)

        # If rotation, interpolate all possible DOFs of pm, otherwise only newly activated DOFs
        if self.direction_change:
            newly_activated_dofs = pm.ext_nodal_activation[pf].nonzero()[0]
            newly_activated_dofs = newly_activated_dofs[:np.searchsorted(newly_activated_dofs, pm.domain.topology.index_map(0).size_local)]
        else:
            newly_activated_dofs = np.logical_and(pm.active_nodes_func.x.array,
                                                  np.logical_not(prev_pm_active_nodes_mask)).nonzero()[0]

        num_dofs_to_interpolate = pm.domain.comm.allreduce(newly_activated_dofs.size)
        if num_dofs_to_interpolate > 0:
            pm.interpolate(pf, dofs_to_interpolate=newly_activated_dofs)
            pm_interp_done = True

        # Advance pm in time, updating u_prev only if interpolation was done
        # We could also always update u_prev
        pm.pre_iterate(forced_time_derivative=(forced_time_derivative and not(pm_interp_done)),
                       verbose=False)

        self.fast_subproblem_els = pf.active_els.copy()# This can be removed
        pm.intersect_problem(pf, finalize=False)

        if self.chimera_on:
            pf.subtract_problem(pm, finalize=False)# Can I finalize here?
            self.chimera_interpolate_material_id()

        for p in [pf, pm]:
            p.finalize_activation()
        for p, p_ext in ([pf, pm], [pm, pf]):
            p.find_gamma(p_ext, compute_robin_coupling_data=compute_robin_coupling_data)

    def chimera_micro_post_iterate(self):
        (pf, pm) = self.pf, self.pm
        pf.set_activation(self.fast_subproblem_els, finalize=False)
        max_ft = abs(pf.u.x.array - pf.u_prev.x.array / pf.dt.value).max()
        max_ft = comm.allreduce(max_ft, MPI.MAX)
        #self.steadiness_measurements.append(self.steadiness_metric.get_steadiness_metric())
        self.steadiness_measurements.append(max_ft)
        if rank==0:
            adim_dt = self.pm.adimensionalize_mhs_timestep(pm.source.path.current_track)
            print(f"is Chimera ON? {self.chimera_on}, adim dt = {adim_dt}, steadiness metric = {self.steadiness_measurements[-1]}, max ft = {max_ft}")

    def chimera_interpolate_material_id(self):
        (pf, pm) = (self.pf, self.pm)
        interpolate_dg0(pf.material_id, pm.material_id,
                        pf.ext_colliding_els[pm], pm.local_active_els)

class MHSStaggeredChimeraSubstepper(MHSStaggeredSubstepper, ChimeraSubstepper):
    def __init__(self,
                 staggered_driver_class : typing.Type[StaggeredInterpDDDriver],
                 slow_problem:Problem, moving_problem : Problem,
                 staggered_relaxation_factors : list[float] = [1.0, 1.0],
                 max_nr_iters=25, max_ls_iters=5,
                 compile_forms=True,
                 initial_orientation=np.array([1.0, 0.0, 0.0])
                 ):
        self.pm = moving_problem
        for mat in self.pm.material_to_itag:
            self.pm.material_to_itag[mat] += 2*len(slow_problem.materials)
        super().__init__(staggered_driver_class,
                         slow_problem,
                         staggered_relaxation_factors,
                         max_nr_iters,
                         max_ls_iters,
                         compile_forms)
        self.plist.append(self.pm)
        self.fplist.append(self.pm)
        self.chimera_post_init(initial_orientation)

    def compile_forms(self):
        ps, pm = (self.ps, self.pm)
        for p in [ps, pm]:
            p.set_forms()
            self.r_ufl[p] = p.a_ufl - p.l_ufl
            self.j_ufl[p]  = ufl.derivative(self.r_ufl[p], p.u)
            self.j_compiled[p]  = fem.compile_form(p.domain.comm, self.j_ufl[p],
                                          form_compiler_options={"scalar_type": np.float64})
            self.r_compiled[p] = fem.compile_form(p.domain.comm, self.r_ufl[p],
                                          form_compiler_options={"scalar_type": np.float64})

    def pre_loop(self):
        super().pre_loop()
        self.pm.set_dt(self.pf.dt.value)

    def post_iterate(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        self.staggered_driver.relaxation_coeff[pf].value = self.relaxation_coeff_pf
        for p in plist:
            p.post_iterate()

    def micro_step(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        sd = self.staggered_driver
        cd = self.chimera_driver

        f = self.fraction_macro_step
        sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                f*self.ext_sol_array_tnp1[:]
        sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                f*self.ext_flux_array_tnp1[:]

        sd.instantiate_forms(pf)
        # Check gamma integration data is present
        sd.assert_tag(pf)

        pf.pre_assemble()

        if self.chimera_on:
            self.instantiate_forms(pm)
            pm.pre_assemble()
            cd.non_linear_solve()
            interpolate_solution_to_inactive(pf, pm)
        else:
            pf.non_linear_solve()
            pm.interpolate(pf)

    def writepos(self,case="macro",extra_funs=[]):
        if not(self.do_writepos):
            return
        (ps,pf,pm) = (self.ps,self.pf,self.pm)
        sd = self.staggered_driver
        if not(self.writers):
            self.initialize_post()

        if case=="predictor":
            p, p_ext = (ps, pf)
            time = 0.0
        elif case=="micro":
            p, p_ext = (pf, ps)
            time = (self.macro_iter-1) + self.fraction_macro_step
        else:
            p, p_ext = (ps, pf)
            time = (self.macro_iter-1) + self.fraction_macro_step
        get_funs = lambda p : [p.u,p.source.fem_function,p.active_els_func,p.grad_u, p.u_prev,self.u_prev[p], p.material_id, p.u_av] + [gn for gn in p.gamma_nodes.values()]
        p_funs = get_funs(p)
        for fun_dic in [sd.ext_flux,sd.net_ext_flux,sd.ext_sol,
                        sd.prev_ext_flux,sd.prev_ext_sol]:
            try:
                p_funs.append(fun_dic[p])
            except (AttributeError,KeyError):
                continue
        p_funs += extra_funs
        p.compute_gradient()
        self.writers[p].write_function(p_funs,t=time)
        if case=="micro":
            self.writers[pm].write_function(get_funs(pm),t=time)

class MHSSemiMonolithicChimeraSubstepper(MHSSemiMonolithicSubstepper, ChimeraSubstepper):
    def __init__(self,slow_problem:Problem, moving_problem : Problem,
                 max_nr_iters=25,max_ls_iters=5,
                 initial_orientation=np.array([1.0, 0.0, 0.0])):
        self.pm = moving_problem
        super().__init__(slow_problem, max_nr_iters, max_ls_iters, compile_forms=False)
        self.plist.append(self.pm)
        self.fplist.append(self.pm)
        self.compile_forms()
        self.chimera_post_init(initial_orientation)

    def pre_loop(self, prepare_fast_problem=False):
        super().pre_loop(prepare_fast_problem)
        self.pm.set_dt(self.pf.dt.value)

    def micro_step(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        cd = self.chimera_driver

        assert(check_assumptions(ps, pf, pm))

        f = self.fraction_macro_step
        # Set Dirichlet condition
        self.fast_dirichlet_tcon.g.x.array[self.gamma_dofs_fast] = \
                (1-f)*ps.u_prev.x.array[self.gamma_dofs_fast] + \
                f*ps.u.x.array[self.gamma_dofs_fast]

        self.instantiate_forms(pf)
        pf.pre_assemble()
        if self.chimera_on:
            self.instantiate_forms(pm)
            pm.pre_assemble()
            cd.non_linear_solve()
            interpolate_solution_to_inactive(pf,pm)
        else:
            pf.non_linear_solve()
            pm.interpolate(pf)

    def update_solution(self, sol_vector, interpolate=False):
        (ps, pf, pm) = (self.ps, self.pf, self.pm)
        self.chimera_driver.update_solution(sol_vector)
        if interpolate:
            interpolate_solution_to_inactive(pf, pm)
        ps.u.x.array[:] = pf.u.x.array[:]

    def assemble_jacobian(
            self, snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P_mat: PETSc.Mat):
        (ps, pf, pm) = (self.ps, self.pf, self.pm)
        super(type(self), self).assemble_jacobian(snes, x, J_mat.getNestSubMatrix(0,0), P_mat)
        pm.assemble_jacobian(finalize=False)
        self.chimera_driver.assemble_robin_jacobian_p_p_ext(self.chimera_driver.p1, self.chimera_driver.p2)
        self.chimera_driver.assemble_robin_jacobian_p_p_ext(self.chimera_driver.p2, self.chimera_driver.p1)
        for p in [pf, pm]:
            self.chimera_driver.assemble_robin_jacobian_p_p(p)
        J_mat.assemble()

    def assemble_residual(self, snes: PETSc.SNES, x: PETSc.Vec, R_vec: PETSc.Vec): 
        (ps, pf, pm, cd) = (self.ps, self.pf, self.pm, self.chimera_driver)
        self.update_solution(x)
        Rf, Rm = R_vec.getNestSubVecs()
        with Rf.localForm() as R_local:
            R_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector(Rf,
                                                self.r_instance,
                                                restriction=pf.restriction)
        Rf.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        pm.assemble_residual(Rm)
        for p, F_vec in zip([pf, pm], [Rf, Rm]):
            cd.assemble_robin_residual(p, F_vec)

    def obj(  # type: ignore[no-any-unimported]
        self,
        snes: PETSc.SNES, x: PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.assemble_residual(snes, x, self.obj_vec)
        return self.obj_vec.norm()  # type: ignore[no-any-return]

    def monolithic_step(self):
        (ps, pf, pm) = (self.ps, self.pf, self.pm)
        ps.pre_iterate(forced_time_derivative=True)
        # PRE-ITERATE
        # NOTE: monolithic coupling data can't be computed yet because DOF numbering
        #      is not yet set up for pf
        self.chimera_micro_pre_iterate(forced_time_derivative=((pf.time - self.t0_macro_step) < 1e-7),
                                       compute_robin_coupling_data=False)

        assert(check_assumptions(ps, pf, pm))

        self.set_gamma_slow_to_fast()

        self.j_instance, self.r_instance = self.instantiate_monolithic_forms()

        # NEW RESTRICTION including both fast and slow DOFs
        dofs_big_mesh = np.hstack((pf.active_dofs, self.dofs_slow))
        dofs_big_mesh.sort()
        self.restriction = multiphenicsx.fem.DofMapRestriction(pf.v.dofmap, dofs_big_mesh)
        pf.j_instance = self.j_instance
        pf.r_instance = self.r_instance
        pf.restriction = self.restriction
        self.initial_restriction = self.restriction

        # NOTE: Now that all restrictions are set, final DOF numbering is set
        # and monolithic coupling data is computed
        for p, p_ext in [(pf, pm), (pm, pf)]:
            p.compute_robin_coupling_data(p_ext)

        pf.pre_assemble()

        self.instantiate_forms(pm)
        pm.pre_assemble()

        cd = self.chimera_driver
        cd.pre_assemble()
        self.L = cd.L
        self.A = cd.A
        self.x = cd.x
        self.obj_vec = cd.obj_vec

        # Solve
        if "petsc_opts_macro_chimera" in ps.input_parameters:
            solver_opts = dict(ps.input_parameters["petsc_opts_macro_chimera"])
        else:
            solver_opts = {"pc_type" : "lu", "pc_factor_mat_solver_type" : "mumps",}
        snes = PETSc.SNES().create(pf.domain.comm)

        opts = PETSc.Options()
        for k,v in solver_opts.items():
            opts[k] = v
        snes.setFromOptions()
        pc = snes.getKSP().getPC()
        if pc.getType() == "fieldsplit":
            index_sets = self.A.getNestISs()
            pc.setFieldSplitIS(["u1", index_sets[0][0]], ["u2", index_sets[1][1]])

        snes.setObjective(self.obj)
        snes.setFunction(self.assemble_residual, self.L)
        snes.setJacobian(self.assemble_jacobian, J=self.A, P=None)
        snes.setMonitor(lambda _, it, residual: print(it, residual, flush=True) if rank == 0 else None)
        cd.set_snes_sol_vector(self.x)
        snes.solve(None, self.x)
        self.update_solution(self.x, interpolate=True)
        assert (snes.getConvergedReason() > 0)
        snes.destroy()
        [opts.__delitem__(k) for k in opts.getAll().keys()] # Clear options data-base
        opts.destroy()

        ## POST-ITERATE
        self.chimera_micro_post_iterate()
        for p in self.plist:
            p.post_modify_solution()
            p.post_iterate()
        ps.set_dt(ps.dt.value)
        pf.dirichlet_bcs = [self.fast_dirichlet_tcon]

        self.micro_iter += 1
        self.fraction_macro_step = 1.0
        if self.do_writepos:
            self.writepos()

    def writepos(self,case="macro", extra_funs=[]):
        if not(self.do_writepos):
            return
        (pf, ps) = (self.pf, self.ps)
        if not(self.writers):
            self.initialize_post()
        if case=="predictor":
            Ps = [ps]
            time = 0.0
        elif case=="micro":
            Ps = [pf]
            time = (self.macro_iter-1) + self.fraction_macro_step
        else:
            Ps = [ps, pf]
            time = (self.macro_iter-1) + self.fraction_macro_step
        get_funs = lambda p : [p.u, p.source.fem_function,p.active_els_func,p.grad_u,
                p.u_prev,self.u_prev[p], p.material_id, p.u_av] + list(p.gamma_nodes.values())

        for p in Ps:
            p.compute_gradient()
            self.writers[p].write_function(get_funs(p),t=time)
        if not(case=="predictor"):
            self.writers[self.pm].write_function(get_funs(self.pm),t=time)
