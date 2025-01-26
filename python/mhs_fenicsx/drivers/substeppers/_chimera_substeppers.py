from mhs_fenicsx.problem import Problem
from mhs_fenicsx.drivers.substeppers import MHSStaggeredSubstepper, MHSSemiMonolithicSubstepper
from mhs_fenicsx.drivers._monolithic_drivers import MonolithicRRDriver
import ufl
import numpy as np
from dolfinx import fem
from mhs_fenicsx.chimera import interpolate_solution_to_inactive
import multiphenicsx

def check_assumptions(ps, pf, pm):
    return not(np.logical_and(pf.gamma_facets[pm].values, ps.bfacets_tag.values).all())

def chimera_micro_post_iterate(pf, pm):
    '''
    post_iterate of micro_step
    TODO: Maybe refactor? Seems like ps needs the same thing
    '''
    for p in (pf, pm):
        current_track = p.source.path.get_track(p.time)
        dt2track_end = current_track.t1 - p.time
        if abs(p.dt.value - dt2track_end) < 1e-9:
            p.set_dt(dt2track_end)


class MHSStaggeredChimeraSubstepper(MHSStaggeredSubstepper):
    def __init__(self,slow_problem:Problem, moving_problem : Problem,
                 writepos=True,
                 max_nr_iters=25,max_ls_iters=5,
                 do_predictor=False,
                 compile_forms=True):
        self.pm = moving_problem
        super().__init__(slow_problem,writepos,max_nr_iters,max_ls_iters, do_predictor,compile_forms)
        self.plist.append(self.pm)
        self.quadrature_degree = 2 # Gamma Chimera
        self.name = "staggered_chimera_substepper"
        self.chimera_driver = MonolithicRRDriver(self.pf, self.pm,
                                                 quadrature_degree=self.quadrature_degree)

    def compile_forms(self):
        ps, pm = (self.ps, self.pm)
        self.integration_tag[pm] = 3
        for p in [ps, pm]:
            p.set_forms_domain(subdomain_idx=self.integration_tag[p])
            p.set_forms_boundary(subdomain_idx=self.integration_tag[p])
            r_ufl = p.a_ufl - p.l_ufl
            self.mr_ufl[p] = -r_ufl
            self.j_ufl[p]  = ufl.derivative(r_ufl, p.u)
            self.j_compiled[p]  = fem.compile_form(p.domain.comm, self.j_ufl[p],
                                          form_compiler_options={"scalar_type": np.float64})
            self.mr_compiled[p] = fem.compile_form(p.domain.comm, self.mr_ufl[p],
                                          form_compiler_options={"scalar_type": np.float64})

    def pre_loop(self):
        super().pre_loop()
        self.pm.set_dt(self.pf.dt.value)

    def post_iterate(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        self.staggered_driver.relaxation_coeff[pf].value = self.relaxation_coeff_pf
        for p in plist:
            p.post_iterate()

    def micro_post_iterate(self):
        chimera_micro_post_iterate(self.pf, self.pm)

    def micro_steps(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        sd = self.staggered_driver
        cd = self.chimera_driver

        self.micro_iter = 0
        while (self.t1_macro_step - pf.time) > 1e-7:
            forced_time_derivative = (self.micro_iter==0)
            pm.intersect_problem(pf, finalize=False)
            for p in [pm, pf]:
                p.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
            fast_subproblem_els = pf.active_els_func.x.array.nonzero()[0]
            pm.intersect_problem(pf, finalize=False)
            ps_pf_integration_data = pf.form_subdomain_data[fem.IntegralType.exterior_facet][-1]
            assert(ps_pf_integration_data[0] == sd.gamma_integration_tag)
            pf.subtract_problem(pm, finalize=False)# Can I finalize here?
            for p, p_ext in [(pm, pf), (pf, pm)]:
                p.finalize_activation()
                p.find_gamma(p_ext)#TODO: Re-use previous data here
            pf.form_subdomain_data[fem.IntegralType.exterior_facet].append(ps_pf_integration_data)
            assert(check_assumptions(ps, pf, pm))

            self.micro_iter += 1
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            f = self.fraction_macro_step
            sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                    f*self.ext_sol_array_tnp1[:]
            sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                    f*self.ext_flux_array_tnp1[:]

            pf.instantiate_forms()
            self.instantiate_forms(pm)
            for p in [pm, pf]:
                p.pre_assemble()

            cd.non_linear_solve()

            self.micro_post_iterate()
            interpolate_solution_to_inactive(pf,pm)
            self.writepos(case="micro")
            pf.set_activation(fast_subproblem_els, finalize=False)

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
        get_funs = lambda p : [p.u,p.source.fem_function,p.active_els_func,p.grad_u, p.u_prev,self.u_prev[p]] + [gn for gn in p.gamma_nodes.values()]
        p_funs = get_funs(p)
        for fun_dic in [sd.ext_flux,sd.net_ext_flux,sd.ext_conductivity,sd.ext_sol,
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

class MHSSemiMonolithicChimeraSubstepper(MHSSemiMonolithicSubstepper):
    def __init__(self,slow_problem:Problem, moving_problem : Problem,
                 writepos=True,
                 max_nr_iters=25,max_ls_iters=5,
                 do_predictor=False):
        self.pm = moving_problem
        super().__init__(slow_problem,writepos,max_nr_iters,max_ls_iters, do_predictor, compile_forms=False)
        self.plist.append(self.pm)
        self.integration_tag[self.pm] = 3
        self.compile_forms()
        self.quadrature_degree = 2 # Gamma Chimera
        self.name = "semi_monolithic_chimera_substepper"
        self.chimera_driver = MonolithicRRDriver(self.pf, self.pm,
                                                 quadrature_degree=self.quadrature_degree)

    def pre_loop(self, prepare_fast_problem=False):
        super().pre_loop(prepare_fast_problem)
        self.pm.set_dt(self.pf.dt.value)

    def micro_post_iterate(self):
        chimera_micro_post_iterate(self.pf, self.pm)

    def micro_steps(self):
        (ps,pf,pm) = plist = (self.ps,self.pf,self.pm)
        cd = self.chimera_driver

        self.micro_iter = 0
        while self.micro_iter < (self.num_micro_steps - 1):
            forced_time_derivative = (self.micro_iter==0)
            pm.intersect_problem(pf, finalize=False)
            for p in [pm, pf]:
                p.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
            pm.intersect_problem(pf, finalize=False)
            fast_subproblem_els = pf.active_els_func.x.array.nonzero()[0]
            pf.subtract_problem(pm, finalize=False)# Can I finalize here?
            for p, p_ext in [(pm, pf), (pf, pm)]:
                p.finalize_activation()
                p.find_gamma(p_ext)#TODO: Re-use previous data here
            assert(check_assumptions(ps, pf, pm))

            self.micro_iter += 1
            self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
            f = self.fraction_macro_step
            self.fast_dirichlet_tcon.g.x.array[self.gamma_dofs_fast] = \
                    (1-f)*ps.u_prev.x.array[self.gamma_dofs_fast] + \
                    f*ps.u.x.array[self.gamma_dofs_fast]

            self.instantiate_forms(pf)
            self.instantiate_forms(pm)
            for p in [pm, pf]:
                p.pre_assemble()

            cd.non_linear_solve()

            self.micro_post_iterate()
            interpolate_solution_to_inactive(pf,pm)
            self.writepos(case="micro")
            pf.set_activation(fast_subproblem_els, finalize=False)

    def monolithic_step(self):
        (ps, pf, pm) = (self.ps, self.pf, self.pm)
        pf.clear_dirchlet_bcs()
        pm.intersect_problem(pf, finalize=False)
        ps.pre_iterate(forced_time_derivative=True)
        pf.pre_iterate()
        pm.pre_iterate()
        pm.intersect_problem(pf, finalize=False)
        pf.subtract_problem(pm, finalize=False)# Can I finalize here?
        for p, p_ext in [(pm, pf), (pf, pm)]:
            p.finalize_activation()
            p.find_gamma(p_ext)#TODO: Re-use previous data here
        assert(check_assumptions(ps, pf, pm))

        self.set_gamma_slow_to_fast()
        j_instance, mr_instance = self.instantiate_monolithic_forms()

        A = multiphenicsx.fem.petsc.create_matrix(j_instance, restriction=(self.initial_restriction, self.initial_restriction))
        R = multiphenicsx.fem.petsc.create_vector(mr_instance, restriction=self.initial_restriction)

        #lin_algebra_objects = [A, R, x, obj_vec]

        ## SOLVE
        #def set_snes_sol_vector() -> PETSc.Vec:  # type: ignore[no-any-unimported]
        #    """
        #    Set PETSc.Vec to be passed to PETSc.SNES.solve to initial guess
        #    """
        #    with multiphenicsx.fem.petsc.VecSubVectorWrapper(
        #            x, ps.v.dofmap, self.initial_restriction) as x_wrapper:
        #        with pf.u.x.petsc_vec.localForm() as uf_sub_vector_local, \
        #          ps.u.x.petsc_vec.localForm() as us_sub_vector_local:
        #              x_wrapper[:] = us_sub_vector_local
        #              x_wrapper[self.dofs_fast] = uf_sub_vector_local[self.dofs_fast]

        #def update_solution(sol_vector):
        #    with pf.u.x.petsc_vec.localForm() as uf_sub_vector_local, \
        #         ps.u.x.petsc_vec.localForm() as us_sub_vector_local, \
        #            multiphenicsx.fem.petsc.VecSubVectorWrapper(sol_vector, ps.v.dofmap, self.initial_restriction) as sol_vector_wrapper:
        #                us_sub_vector_local[:] = sol_vector_wrapper
        #                uf_sub_vector_local[:] = sol_vector_wrapper
        #    ps.u.x.scatter_forward()
        #    pf.u.x.scatter_forward()

        #def assemble_residual(snes: PETSc.SNES, x: PETSc.Vec, R_vec: PETSc.Vec): 
        #    update_solution(x)
        #    with R_vec.localForm() as R_local:
        #        R_local.set(0.0)
        #    multiphenicsx.fem.petsc.assemble_vector(R_vec,
        #                                            mr_instance,
        #                                            restriction=self.initial_restriction)
        #    # TODO: Dirichlet here?
        #    R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        #    R_vec.scale(-1)

        #def assemble_jacobian(
        #        snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P_mat: PETSc.Mat):
        #    J_mat.zeroEntries()
        #    multiphenicsx.fem.petsc.assemble_matrix(J_mat, j_instance, restriction=(self.initial_restriction, self.initial_restriction))
        #    J_mat.assemble()


        #def obj(  # type: ignore[no-any-unimported]
        #    snes: PETSc.SNES, x: PETSc.Vec
        #) -> np.float64:
        #    """Compute the norm of the residual."""
        #    assemble_residual(snes, x, obj_vec)
        #    return obj_vec.norm()  # type: ignore[no-any-return]

        ## Solve
        #snes = PETSc.SNES().create(ps.domain.comm)
        #snes.setTolerances(max_it=self.max_nr_iters)
        #ksp_opts = PETSc.Options()
        #for k,v in ps.linear_solver_opts.items():
        #    ksp_opts[k] = v
        #snes.getKSP().setFromOptions()
        #snes.setObjective(obj)
        #snes.setFunction(assemble_residual, R)
        #snes.setJacobian(assemble_jacobian, J=A, P=None)
        #snes.setMonitor(lambda _, it, residual: print(it, residual))
        #set_snes_sol_vector()
        #snes.solve(None, x)
        #update_solution(x)
        #snes.destroy()
        #for ds in lin_algebra_objects:
        #    ds.destroy()

        ## POST-ITERATE
        #for p in self.plist:
        #    p.post_iterate()
        #ps.set_dt(ps.dt.value)
        #pf.dirichlet_bcs = [self.fast_dirichlet_tcon]

        #self.micro_iter += 1
        #self.fraction_macro_step = 1.0
        #if self.do_writepos:
        #    self.writepos()
