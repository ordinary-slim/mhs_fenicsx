from mhs_fenicsx.problem import Problem, L2Differ
from mhs_fenicsx.problem.helpers import interpolate_cg1, interpolate_dg0
from mhs_fenicsx.drivers.substeppers import MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper
from mhs_fenicsx.drivers._monolithic_drivers import MonolithicRRDriver
from mhs_fenicsx.drivers._staggered_drivers import StaggeredDomainDecompositionDriver
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
        self.quadrature_degree : int
        self.fast_subproblem_els : npt.NDArray[np.int32]
        self.params : dict
        self.t0_macro_step : float
        self.t1_macro_step : float

    def chimera_post_init(self, initial_orientation : npt.NDArray):
        self.chimera_driver = MonolithicRRDriver(self.pf, self.pm,
                                                 1.0, 1.0,
                                                 quadrature_degree=self.quadrature_degree)
        self.chimera_always_on = self.params["chimera_always_on"]
        self.chimera_on = self.chimera_always_on
        self.current_orientation = initial_orientation.astype(np.float64)
        # Overwrite methods
        def post_step_wo_substepping():
            super(type(self), self).post_step_wo_substepping()
            self.chimera_post_step_without_substepping()
        def prepare_micro_step():
            self.chimera_prepare_micro_step()
        def micro_post_iterate():
            self.chimera_micro_post_iterate()
        self.prepare_micro_step = prepare_micro_step
        self.post_step_wo_substepping = post_step_wo_substepping
        self.micro_post_iterate = micro_post_iterate
        # Steadiness workflow
        self.steadiness_workflow_params = self.params["chimera_steadiness_workflow"]
        self.steadiness_metric = L2Differ(self.pm)
        self.steadiness_measurements = []

    def chimera_post_step_without_substepping(self):
        (pf, pm) = self.pf, self.pm
        pm.intersect_problem(pf, finalize=False)
        pm.interpolate(pf)

    def chimera_prepare_micro_step(self):
        (pf, pm) = self.pf, self.pm
        super(type(self), self).prepare_micro_step()
        next_track = self.pf.source.path.get_track(pf.time)
        self.direction_change = False
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

        max_dt_substep = self.t1_macro_step - pf.time

        def set_dt(dt : float):
            dt = min(dt, max_dt_substep)
            for p in [pf, pm]:
                p.set_dt(dt)

        increasing_dt = self.params["chimera_steadiness_workflow"]["enabled"]
        # STEADINESS WORKFLOW
        # Unsteady reset
        if self.direction_change:
            if not(self.chimera_always_on):
                self.chimera_on = False
            if increasing_dt:
                set_dt(pf.dimensionalize_mhs_timestep(next_track, self.params["micro_adim_dt"]))
        else:
            if self.is_steady_enough():
                if not(self.chimera_on):
                    self.chimera_on = True
                else:
                    if increasing_dt:
                        current_dt = pm.dt.value
                        increment = pf.dimensionalize_mhs_timestep(next_track, self.params["chimera_steadiness_workflow"]["adim_dt_increment"])
                        dt = min(pf.dimensionalize_mhs_timestep(next_track, self.params["chimera_steadiness_workflow"]["max_adim_dt"]), current_dt + increment)
                        set_dt(dt)

        if pf.dt.value > (max_dt_substep + 1e-7):
            set_dt(max_dt_substep)

    def is_steady_enough(self):
        next_track = self.pf.source.path.get_track(self.pf.time)
        return (((self.pf.time - next_track.t0) / (next_track.t1 - next_track.t0)) >= 0.15)

    def chimera_micro_pre_iterate(self, forced_time_derivative=False):
        (pf, pm) = self.pf, self.pm
        prev_pm_active_nodes_mask = pm.active_nodes_func.x.array.copy()
        shape_moving_problem(pm)
        pm.intersect_problem(pf, finalize=False)
        pm.update_active_dofs()
        newly_activated_dofs = np.logical_and(pm.active_nodes_func.x.array,
                                              np.logical_not(prev_pm_active_nodes_mask)).nonzero()[0]
        num_dofs_to_interpolate = pm.domain.comm.allreduce(newly_activated_dofs.size)
        if num_dofs_to_interpolate > 0:
            pm.interpolate(pf, dofs_to_interpolate=newly_activated_dofs)
        pf.pre_iterate(forced_time_derivative=forced_time_derivative, verbose=False)
        if self.direction_change:
            if self.chimera_on:
                pm.interpolate(pf, dofs_to_interpolate=newly_activated_dofs)
            pm.pre_iterate(forced_time_derivative=False, verbose=False)
        else:
            pm.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
        self.fast_subproblem_els = pf.active_els.copy()# This can be removed
        pm.intersect_problem(pf, finalize=False)

        if self.chimera_on:
            pf.subtract_problem(pm, finalize=False)# Can I finalize here?
            self.chimera_interpolate_material_id()

        for p, p_ext in [(pm, pf), (pf, pm)]:
            p.finalize_activation()
            p.find_gamma(p_ext)#TODO: Re-use previous data here

    def chimera_micro_post_iterate(self):
        (pf, pm) = self.pf, self.pm
        pf.set_activation(self.fast_subproblem_els, finalize=False)
        self.steadiness_measurements.append(self.steadiness_metric.get_steadiness_metric())
        if rank==0:
            adim_dt = self.pm.adimensionalize_mhs_timestep(pm.source.path.current_track)
            print(f"is Chimera ON? {self.chimera_on}, adim dt = {adim_dt}, steadiness metric = {self.steadiness_measurements[-1]}")

    def chimera_interpolate_material_id(self):
        (pf, pm) = (self.pf, self.pm)
        interpolate_dg0(pf.material_id, pm.material_id,
                        pf.ext_colliding_els[pm], pm.local_active_els)

class MHSStaggeredChimeraSubstepper(MHSStaggeredSubstepper, ChimeraSubstepper):
    def __init__(self,
                 staggered_driver_class : typing.Type[StaggeredDomainDecompositionDriver],
                 staggered_relaxation_factors : list[float],
                 slow_problem:Problem, moving_problem : Problem,
                 max_nr_iters=25, max_ls_iters=5,
                 compile_forms=True,
                 initial_orientation=np.array([1.0, 0.0, 0.0])
                 ):
        self.pm = moving_problem
        for mat in self.pm.material_to_itag:
            self.pm.material_to_itag[mat] += 2*len(slow_problem.materials)
        super().__init__(staggered_driver_class,
                         staggered_relaxation_factors,
                         slow_problem,
                         max_nr_iters,
                         max_ls_iters,
                         compile_forms)
        self.plist.append(self.pm)
        self.fplist.append(self.pm)
        self.quadrature_degree = 2 # Gamma Chimera
        self.name = "staggered_chimera_substepper"
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
        sd.assert_tag(pf)

        forced_time_derivative = (pf.time - self.t0_macro_step) < 1e-7
        self.chimera_micro_pre_iterate(forced_time_derivative=forced_time_derivative)

        self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
        f = self.fraction_macro_step
        sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                f*self.ext_sol_array_tnp1[:]
        sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                f*self.ext_flux_array_tnp1[:]

        sd.instantiate_forms(pf)

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
        self.quadrature_degree = 2 # Gamma Chimera
        self.name = "semi_monolithic_chimera_substepper"
        self.chimera_post_init(initial_orientation)

    def pre_loop(self, prepare_fast_problem=False):
        super().pre_loop(prepare_fast_problem)
        self.pm.set_dt(self.pf.dt.value)

    def micro_step(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        cd = self.chimera_driver

        forced_time_derivative = (pf.time - self.t0_macro_step) < 1e-7
        self.chimera_micro_pre_iterate(forced_time_derivative=forced_time_derivative)

        assert(check_assumptions(ps, pf, pm))

        self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
        f = self.fraction_macro_step
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

    def set_snes_sol_vector(self, x) -> PETSc.Vec:  # type: ignore[no-any-unimported]
        """
        Set PETSc.Vec to be passed to PETSc.SNES.solve to initial guess
        """
        (ps, pf, pm) = (self.ps, self.pf, self.pm)
        with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(
                x, self.chimera_driver.dofmaps, self.chimera_driver.restriction) as nest_sol:
            for sol_sub, u_sub in zip(nest_sol, [pf.u, pm.u]):
                with u_sub.x.petsc_vec.localForm() as u_sol_sub_vector_local:
                    sol_sub[:] = u_sol_sub_vector_local[:]

    def update_solution(self, sol_vector, interpolate=False):
        (ps, pf, pm) = (self.ps, self.pf, self.pm)
        with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(sol_vector, self.chimera_driver.dofmaps, self.chimera_driver.restriction) as nest_sol:
            for sol_sub, u_sub in zip(nest_sol, [pf.u, pm.u]):
                with u_sub.x.petsc_vec.localForm() as u_sub_vector_local:
                    u_sub_vector_local[:] = sol_sub[:]
        pm.u.x.scatter_forward()
        pf.u.x.scatter_forward()
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
        self.chimera_driver.assemble_robin_jacobian_p_p()
        J_mat.assemble()

    def assemble_residual(self, snes: PETSc.SNES, x: PETSc.Vec, R_vec: PETSc.Vec): 
        (ps, pf, pm) = (self.ps, self.pf, self.pm)
        self.update_solution(x)
        Rf, Rm = R_vec.getNestSubVecs()
        with Rf.localForm() as R_local:
            R_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector(Rf,
                                                self.r_instance,
                                                restriction=pf.restriction)
        Rf.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        pm.assemble_residual(Rm)
        self.chimera_driver.assemble_robin_residual(R_vec)

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
        self.chimera_micro_pre_iterate(forced_time_derivative=((pf.time - self.t0_macro_step) < 1e-7))

        assert(check_assumptions(ps, pf, pm))

        self.set_gamma_slow_to_fast()
        self.j_instance, self.r_instance = self.instantiate_monolithic_forms()

        # Set-up SNES solve
        pf.clear_dirchlet_bcs()
        pf.u.x.array[self.dofs_slow] = ps.u.x.array[self.dofs_slow]#useful before set_snes

        # Make a new restriction
        dofs_big_mesh = np.hstack((pf.active_dofs, self.dofs_slow))
        dofs_big_mesh.sort()# TODO: maybe REMOVABLE
        self.restriction = multiphenicsx.fem.DofMapRestriction(pf.v.dofmap, dofs_big_mesh)
        pf.j_instance = self.j_instance
        pf.r_instance = self.r_instance
        pf.restriction = self.restriction
        self.initial_restriction = self.restriction
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
        snes.setMonitor(lambda _, it, residual: print(it, residual))
        self.set_snes_sol_vector(self.x)
        snes.solve(None, self.x)
        self.update_solution(self.x, interpolate=True)
        assert (snes.getConvergedReason() > 0)
        snes.destroy()
        [opts.__delitem__(k) for k in opts.getAll().keys()] # Clear options data-base
        opts.destroy()

        ## POST-ITERATE
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
