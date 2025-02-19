from mhs_fenicsx.problem import Problem
from mhs_fenicsx.problem.helpers import interpolate_cg1, interpolate_dg0
from mhs_fenicsx.drivers.substeppers import MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper
from mhs_fenicsx.drivers._monolithic_drivers import MonolithicRRDriver
import mhs_fenicsx_cpp
import ufl
import numpy as np
from dolfinx import fem
from mhs_fenicsx.chimera import interpolate_solution_to_inactive
import multiphenicsx
from petsc4py import PETSc
import typing
import numpy.typing as npt
from abc import ABC, abstractmethod

def check_assumptions(ps, pf, pm):
    return not(np.logical_and(pf.gamma_facets[pm].values, ps.bfacets_tag.values).all())

class ChimeraSubstepper(ABC):
    @abstractmethod
    def __init__(self):
        self.pf : Problem
        self.pm : Problem
        self.quadrature_degree : int
        self.fast_subproblem_els : npt.NDArray[np.int32]

    @abstractmethod
    def micro_pre_iterate(self):
        pass

    def chimera_post_init(self, chimera_always_on):
        self.chimera_driver = MonolithicRRDriver(self.pf, self.pm,
                                                 quadrature_degree=self.quadrature_degree)
        self.chimera_off = False
        self.chimera_always_on = chimera_always_on

    def chimera_micro_pre_iterate(self):
        pf = self.pf
        super(type(self), self).micro_pre_iterate()
        if not(self.chimera_always_on):
            next_track = self.pf.source.path.get_track(pf.time)
            if ((pf.time - next_track.t0) / (next_track.t1 - next_track.t0)) < 0.15:
                self.chimera_off = True
            else:
                self.chimera_off = False

    def chimera_micro_post_iterate(self):
        '''
        post_iterate of micro_step
        TODO: Maybe refactor? Seems like ps needs the same thing
        '''
        (pf, pm) = (self.pf, self.pm)
        for p in (pf, pm):
            current_track = p.source.path.get_track(p.time)
            dt2track_end = current_track.t1 - p.time
            if abs(p.dt.value - dt2track_end) < 1e-9:
                p.set_dt(dt2track_end)
        self.pf.set_activation(self.fast_subproblem_els, finalize=False)

    def chimera_interpolate_material_id(self):
        (pf, pm) = (self.pf, self.pm)
        interpolate_dg0(pf.material_id, pm.material_id,
                        pf.ext_colliding_els[pm], pm.local_active_els)

    def chimera_interpolate_to_moving(self):
        (pf, pm) = (self.pf, self.pm)
        local_ext_dofs = pm.ext_nodal_activation[pf].nonzero()[0]
        local_ext_dofs = local_ext_dofs[:np.searchsorted(local_ext_dofs, pm.domain.topology.index_map(0).size_local)]
        interpolate_cg1(pf.u,
                        pm.u,
                        np.arange(pf.cell_map.size_local),
                        local_ext_dofs,
                        pm.dof_coords[local_ext_dofs],
                        1e-6)

class MHSStaggeredChimeraSubstepper(MHSStaggeredSubstepper, ChimeraSubstepper):
    def __init__(self,slow_problem:Problem, moving_problem : Problem,
                 writepos=True,
                 max_nr_iters=25,max_ls_iters=5,
                 do_predictor=False,
                 compile_forms=True,
                 chimera_always_on=True,
                 ):
        self.pm = moving_problem
        for mat in self.pm.material_to_itag:
            self.pm.material_to_itag[mat] += 2*len(slow_problem.materials)
        super().__init__(slow_problem,writepos,max_nr_iters,max_ls_iters, do_predictor,compile_forms)
        self.plist.append(self.pm)
        self.quadrature_degree = 2 # Gamma Chimera
        self.name = "staggered_chimera_substepper"
        self.chimera_post_init(chimera_always_on)

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

    def micro_pre_iterate(self):
        self.chimera_micro_pre_iterate()

    def micro_post_iterate(self):
        self.chimera_micro_post_iterate()

    def micro_step(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        sd = self.staggered_driver
        cd = self.chimera_driver
        forced_time_derivative = (pf.time - self.t0_macro_step) < 1e-7
        pm.intersect_problem(pf, finalize=False)
        for p in [pm, pf]:
            p.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
        self.fast_subproblem_els = pf.active_els_func.x.array.nonzero()[0]
        pm.intersect_problem(pf, finalize=False)
        sd.assert_tag(pf)

        if not(self.chimera_off):
            pf.subtract_problem(pm, finalize=False)# Can I finalize here?
            self.chimera_interpolate_material_id()

        for p, p_ext in [(pm, pf), (pf, pm)]:
            p.finalize_activation()
            p.find_gamma(p_ext)#TODO: Re-use previous data here

        self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
        f = self.fraction_macro_step
        sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                f*self.ext_sol_array_tnp1[:]
        sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                f*self.ext_flux_array_tnp1[:]

        pf.form_subdomain_data[fem.IntegralType.exterior_facet].extend(sd.gamma_subdomain_data[pf])
        assert(check_assumptions(ps, pf, pm))

        pf.instantiate_forms()
        pf.pre_assemble()
        if not(self.chimera_off):
            self.instantiate_forms(pm)
            pm.pre_assemble()
            cd.non_linear_solve()
            interpolate_solution_to_inactive(pf, pm)
        else:
            pf.non_linear_solve()
            self.chimera_interpolate_to_moving()

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
                 writepos=True,
                 max_staggered_iters=1,
                 max_nr_iters=25,max_ls_iters=5,
                 do_predictor=False,
                 chimera_always_on=True):
        self.pm = moving_problem
        super().__init__(slow_problem,writepos, max_staggered_iters, max_nr_iters,max_ls_iters, do_predictor, compile_forms=False)
        self.plist.append(self.pm)
        self.compile_forms()
        self.quadrature_degree = 2 # Gamma Chimera
        self.name = "semi_monolithic_chimera_substepper"
        self.chimera_post_init(chimera_always_on)

    def pre_loop(self, prepare_fast_problem=False):
        super().pre_loop(prepare_fast_problem)
        self.pm.set_dt(self.pf.dt.value)

    def micro_pre_iterate(self):
        self.chimera_micro_pre_iterate()

    def micro_post_iterate(self):
        self.chimera_micro_post_iterate()

    def micro_step(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        cd = self.chimera_driver
        forced_time_derivative = (pf.time - self.t0_macro_step) < 1e-7
        pm.intersect_problem(pf, finalize=False)
        for p in [pm, pf]:
            p.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
        pm.intersect_problem(pf, finalize=False)
        self.fast_subproblem_els = pf.active_els_func.x.array.nonzero()[0]

        if not(self.chimera_off):
            pf.subtract_problem(pm, finalize=False)# Can I finalize here?
            self.chimera_interpolate_material_id()

        for p, p_ext in [(pm, pf), (pf, pm)]:
            p.finalize_activation()
            p.find_gamma(p_ext)#TODO: Re-use previous data here
        assert(check_assumptions(ps, pf, pm))

        self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
        f = self.fraction_macro_step
        self.fast_dirichlet_tcon.g.x.array[self.gamma_dofs_fast] = \
                (1-f)*ps.u_prev.x.array[self.gamma_dofs_fast] + \
                f*ps.u.x.array[self.gamma_dofs_fast]

        self.instantiate_forms(pf)
        pf.pre_assemble()
        if not(self.chimera_off):
            self.instantiate_forms(pm)
            pm.pre_assemble()
            cd.non_linear_solve()
            interpolate_solution_to_inactive(pf,pm)
        else:
            pf.non_linear_solve()
            self.chimera_interpolate_to_moving()

    def monolithic_step(self):
        (ps, pf, pm) = (self.ps, self.pf, self.pm)
        pm.intersect_problem(pf, finalize=False)
        ps.pre_iterate(forced_time_derivative=True)
        pf.pre_iterate()
        pm.pre_iterate()
        pm.intersect_problem(pf, finalize=False)
        pf.subtract_problem(pm, finalize=False)# Can I finalize here?
        self.chimera_interpolate_material_id()
        for p, p_ext in [(pm, pf), (pf, pm)]:
            p.finalize_activation()
            p.find_gamma(p_ext)#TODO: Re-use previous data here
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

        def set_snes_sol_vector(self) -> PETSc.Vec:  # type: ignore[no-any-unimported]
            """
            Set PETSc.Vec to be passed to PETSc.SNES.solve to initial guess
            """
            sol_vector = self.x
            with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(
                    sol_vector, cd.dofmaps, cd.restriction) as nest_sol:
                for sol_sub, u_sub in zip(nest_sol, [pf.u, pm.u]):
                    with u_sub.x.petsc_vec.localForm() as u_sol_sub_vector_local:
                        sol_sub[:] = u_sol_sub_vector_local[:]

        def update_solution(sol_vector, interpolate=False):
            with multiphenicsx.fem.petsc.NestVecSubVectorWrapper(sol_vector, cd.dofmaps, cd.restriction) as nest_sol:
                for sol_sub, u_sub in zip(nest_sol, [pf.u, pm.u]):
                    with u_sub.x.petsc_vec.localForm() as u_sub_vector_local:
                        u_sub_vector_local[:] = sol_sub[:]
            pm.u.x.scatter_forward()
            pf.u.x.scatter_forward()
            if interpolate:
                interpolate_solution_to_inactive(pf, pm)
            ps.u.x.array[:] = pf.u.x.array[:]

        def J_snes(
                snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P_mat: PETSc.Mat):
            super(type(self), self).assemble_jacobian(snes, x, J_mat.getNestSubMatrix(0,0), P_mat)
            pm.assemble_jacobian(finalize=False)
            cd.assemble_robin_jacobian_p_p_ext(cd.p1, cd.p2)
            cd.assemble_robin_jacobian_p_p_ext(cd.p2, cd.p1)
            cd.assemble_robin_jacobian_p_p()
            J_mat.assemble()

        def R_snes(snes: PETSc.SNES, x: PETSc.Vec, R_vec: PETSc.Vec): 
            update_solution(x)
            Rf, Rm = R_vec.getNestSubVecs()
            with Rf.localForm() as R_local:
                R_local.set(0.0)
            multiphenicsx.fem.petsc.assemble_vector(Rf,
                                                    self.r_instance,
                                                    restriction=pf.restriction)
            Rf.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            pm.assemble_residual(Rm)
            cd.assemble_robin_residual(R_vec)

        def obj_snes(  # type: ignore[no-any-unimported]
            snes: PETSc.SNES, x: PETSc.Vec
        ) -> np.float64:
            """Compute the norm of the residual."""
            R_snes(snes, x, self.obj_vec)
            return self.obj_vec.norm()  # type: ignore[no-any-return]

        # Solve
        snes = PETSc.SNES().create(pf.domain.comm)
        snes.setTolerances(max_it=20)

        snes.getKSP().setType("preonly")
        opts = PETSc.Options()
        opts.setValue('-ksp_error_if_not_converged', 'true')
        opts.setValue('-snes_type', 'newtonls')
        #opts.setValue('-snes_line_search_type', 'l2')
        snes.setFromOptions()
        #snes.getKSP().setFromOptions()
        snes.getKSP().getPC().setType("lu")
        snes.getKSP().getPC().setFactorSolverType("mumps")
        #snes.getKSP().getPC().setType("fieldsplit")
        #nested_IS = self.A.getNestISs()
        #snes.getKSP().getPC().setFieldSplitIS(["u1", nested_IS[0][0]], ["u2", nested_IS[1][1]])

        snes.setObjective(obj_snes)
        snes.setFunction(R_snes, self.L)
        snes.setJacobian(J_snes, J=self.A, P=None)
        snes.setMonitor(lambda _, it, residual: print(it, residual))
        set_snes_sol_vector(self)
        snes.solve(None, self.x)
        update_solution(self.x, interpolate=True)
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
