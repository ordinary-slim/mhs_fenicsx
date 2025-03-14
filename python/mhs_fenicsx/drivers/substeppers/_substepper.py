from mhs_fenicsx.problem.helpers import propagate_dg0_at_facets_same_mesh, set_same_mesh_interface, get_identity_maps
from mpi4py import MPI
from dolfinx import fem, io
import ufl
from mhs_fenicsx.problem import Problem
from mhs_fenicsx_cpp import mesh_collision
from mhs_fenicsx.drivers._staggered_drivers import StaggeredRRDriver, StaggeredDNDriver, StaggeredDomainDecompositionDriver
from mhs_fenicsx.geometry import OBB
from mhs_fenicsx.gcode import TrackType
import numpy as np
import numpy.typing as npt
import shutil
import typing
from petsc4py import PETSc
from abc import ABC, abstractmethod
import multiphenicsx.fem.petsc

class MHSSubstepper(ABC):
    def __init__(self, slow_problem:Problem,
                 max_nr_iters=25, max_ls_iters=5,
                 compile_forms=True):
        self.ps = slow_problem
        self.params = self.ps.input_parameters["substepping_parameters"]
        self.pf = self.ps.copy(name=f"{self.ps.name}_micro_iters")
        self.plist = [self.ps, self.pf]
        self.fplist = [self.pf]
        ps, pf = self.ps, self.pf
        self.do_writepos = self.params["writepos"]
        self.writers = dict()
        self.max_nr_iters = max_nr_iters
        self.max_ls_iters = max_ls_iters
        self.do_predictor = self.params["predictor"]["enabled"]
        self.predictor_source_term_idx = self.params["predictor"]["source_term"] if "source_term" in self.params["predictor"] else 0
        self.r_ufl, self.j_ufl, self.r_compiled, self.j_compiled = {}, {}, {}, {}
        for mat in pf.material_to_itag:
            pf.material_to_itag[mat] += len(ps.materials)
        if compile_forms:
            self.compile_forms()

    def do_timestep(self):
        '''Decide type of time-step and macro time-step size'''
        ps, pf = self.ps, self.pf
        self.t0_macro_step = ps.time
        track = ps.source.path.get_track(self.t0_macro_step)
        if track.type == TrackType.PRINTING:
            print("Step WITH substepping STARTS...", flush=True)
            pf.set_dt(pf.dimensionalize_mhs_timestep(track, self.params["micro_adim_dt"]))
            ps.set_dt(ps.dimensionalize_mhs_timestep(track, self.params["macro_adim_dt"]))
            self.cap_timestep(ps)
            self.do_substepped_timestep()
        else:
            print("Step WITHOUT substepping STARTS...", flush=True)
            for p in self.plist:
                p.set_dt(p.dimensionalize_waiting_timestep(track, self.params["cooling_adim_dt"]))
            self.step_without_substepping()

    @abstractmethod
    def do_substepped_timestep(self):
        pass

    def cap_timestep(self, p):
        tracks = p.source.path.get_track_interval(p.time, p.time + p.dt.value)
        max_t1 = tracks[0].t1
        for track in tracks:
            if track.type is not(TrackType.PRINTING):
                break
            else:
                max_t1 = track.t1
        max_dt = max_t1 - p.time
        if (max_dt - p.dt.value) < 1e-9:
            p.set_dt(max_dt)
    
    @abstractmethod
    def compile_forms(self):
        pass

    def instantiate_forms(self, p):
        p.j_ufl, p.r_ufl = (self.j_ufl[p], self.r_ufl[p])
        p.j_compiled, p.r_compiled = (self.j_compiled[p], self.r_compiled[p])
        p.instantiate_forms()

    def __del__(self):
        for w in self.writers.values():
            w.close()

    def update_fast_problem(self):
        (ps, pf) = plist = (self.ps, self.pf)
        self.result_folder = f"post_{self.name}_tstep#{self.ps.iter}"
        # TODO: Complete. Data to set activation in predictor_step
        self.t1_macro_step = ps.time + ps.dt.value
        self.initial_active_els = self.ps.active_els_func.x.array.nonzero()[0]
        self.initial_restriction = self.ps.restriction
        self.fraction_macro_step = 0
        for p in plist:
            p.clear_gamma_data()
        # Store this and use it for deactivation?
        self.fast_els = self.find_subproblem_els()
        pf.set_activation(self.fast_els)
        pf.set_linear_solver(pf.input_parameters["petsc_opts_micro"])
        # Subtract fast
        ps.active_els_func.x.array[self.fast_els] = 0.0
        ps.set_activation(ps.active_els_func, finalize=not(self.do_predictor))
        self.slow_els = ps.active_els.copy()
        if self.do_predictor:
            ps.update_boundary()
        set_same_mesh_interface(ps, pf)
        self.dofs_fast = fem.locate_dofs_topological(pf.v, pf.dim, self.fast_els)
        mask_dofs_slow = np.logical_and(np.logical_not(pf.active_nodes_func.x.array),
                                        ps.active_nodes_func.x.array)
        self.dofs_slow = mask_dofs_slow.nonzero()[0]

    @abstractmethod
    def pre_loop(self):
        self.macro_iter = 0
        self.prev_iter = {p:p.iter for p in self.plist}
        self.u_prev = {p:p.u.copy() for p in self.plist}
        self.material_id_prev = {p:p.material_id.copy() for p in self.plist}
        for u in self.u_prev.values():
            u.name = "u_prev_driver"
        self.initialize_post()

    def initialize_post(self):
        if not(self.do_writepos):
            return
        for w in self.writers.values():
            w.close()
        shutil.rmtree(self.result_folder,ignore_errors=True)
        for p in self.plist:
            self.writers[p] = io.VTKFile(p.domain.comm, f"{self.result_folder}/{self.name}_{p.name}.pvd", "wb")

    @abstractmethod
    def writepos(self,case="macro",extra_funs=[]):
        pass

    def pre_iterate(self):
        self.macro_iter += 1
        for p in self.plist:
            p.time = self.t0_macro_step
            p.iter = self.prev_iter[p]
            p.u_prev.x.array[:] = self.u_prev[p].x.array[:]
            p.material_id.x.array[:] = self.material_id_prev[p].x.array[:]

    def find_subproblem_els(self):
        ps = self.ps
        cdim = ps.dim
        # Determine geometry of subproblem
        # To do it properly, get initial time of macro step, final time of macro step
        # Do collision tests across track and accumulate elements to extract
        # Here we just do a single collision test
        tracks = ps.source.path.get_track_interval(self.t0_macro_step, self.t1_macro_step)
        hs_radius = ps.source.R
        if "adim_lens" in self.params:
            adim_lens_substepper = self.params["adim_lens"]
            adim_back_pad = adim_lens_substepper["back_pad"]
            adim_front_pad = adim_lens_substepper["front_pad"]
            adim_side_pad = adim_lens_substepper["side_pad"]
        else:
            adim_back_pad = 5
            adim_front_pad = 5
            adim_side_pad = 3
        back_pad = adim_back_pad*hs_radius
        front_pad = adim_front_pad*hs_radius
        side_pad = adim_side_pad*hs_radius
        subproblem_els_mask = np.zeros((ps.cell_map.size_local + ps.cell_map.num_ghosts), dtype=np.bool_)
        for track in tracks:
            p0 = track.get_position(self.t0_macro_step, bound=True)
            p1 = track.get_position(self.t1_macro_step, bound=True)
            direction = track.get_direction()
            p0 -= direction*back_pad
            p1 += direction*front_pad
            obb = OBB(p0,p1,width=back_pad,height=side_pad,depth=side_pad,dim=cdim,
                                           shrink=False)
            obb_mesh = obb.get_dolfinx_mesh()
            #subproblem_els = mhs_fenicsx.geometry.mesh_collision(ps.domain,obb_mesh,bb_tree_mesh_big=ps.bb_tree)
            colliding_els = mesh_collision(ps.domain._cpp_object,obb_mesh._cpp_object,bb_tree_big=ps.bb_tree._cpp_object)
            subproblem_els_mask[colliding_els] = np.True_
            subproblem_els_mask = np.logical_and(ps.active_els_func.x.array[:], subproblem_els_mask)
        return subproblem_els_mask.nonzero()[0]

    def predictor_step(self, writepos=False):
        ''' Always linear '''
        ps = self.ps
        # Save current data
        slow_subdomain_data = ps.form_subdomain_data
        source_term_idx = ps.current_source_term
        ps.switch_source_term(self.predictor_source_term_idx)
        ps.set_activation(self.initial_active_els)
        # MACRO-STEP
        ps.pre_iterate()
        self.instantiate_forms(ps)
        ps.pre_assemble()
        ps.non_linear_solve(snes_opts={'-snes_type': 'ksponly'})# LINEAR SOLVE
        ps.post_iterate()

        # Reset iter to prev.
        # Useful for writepos
        ps.iter -= 1
        if writepos:
            self.writepos("predictor")
        ps.switch_source_term(source_term_idx)
        ps.set_activation(self.slow_els)
        ps.form_subdomain_data = slow_subdomain_data

    def step_without_substepping(self, writepos=False):
        # TODO: Advance in time pf and pm!
        ps = self.ps
        ps.pre_iterate()
        self.instantiate_forms(ps)
        ps.pre_assemble()
        ps.non_linear_solve()
        ps.post_iterate()
        if writepos:
            self.writepos("predictor")
        # Communicate 
        self.post_step_wo_substepping()

    def post_step_wo_substepping(self):
        '''Update other problems to current solution'''
        ps, pf = (self.ps, self.pf)
        for p in self.fplist:
            p.pre_iterate()
        # Set variables pf
        pf.u.x.array[:] = ps.u.x.array[:]
        pf.is_grad_computed = False
        pf.material_id.x.array[:] = pf.material_id.x.array[:]
        # TODO: Update pm

    def micro_steps(self):
        self.micro_iter = 0
        while self.is_substepping():
            self.prepare_micro_step()
            self.micro_step()
            self.writepos(case="micro")
            self.micro_post_iterate()

    @abstractmethod
    def is_substepping(self):
        raise NotImplementedError

    @abstractmethod
    def micro_step(self):
        raise NotImplementedError

    def prepare_micro_step(self):
        self.micro_iter += 1
        for p in self.fplist:
            self.cap_timestep(p)

    def micro_post_iterate(self):
        pass

    def post_loop(self):
        #TODO: Change this!
        (ps,pf) = (self.ps,self.pf)
        self.ps.set_activation(self.initial_active_els)
        # Update material composition
        ps.material_id.x.array[self.fast_els] = pf.material_id.x.array[self.fast_els]
        ps.material_id.x.scatter_forward()
        pf.material_id.x.array[:] = ps.material_id.x.array[:]

class MHSSemiMonolithicSubstepper(MHSSubstepper):
    def __init__(self, slow_problem: Problem,
                 max_nr_iters=25,max_ls_iters=5,
                 compile_forms=True):
        super().__init__(slow_problem, max_nr_iters, max_ls_iters, compile_forms)
        self.name = "semi_monolithic_substepper"
        self.max_staggered_iters = self.params["max_staggered_iters"]

    def do_substepped_timestep(self):
        self.update_fast_problem()
        self.pre_loop()
        if self.do_predictor:
            self.predictor_step(writepos=self.do_writepos)
        for _ in range(self.max_staggered_iters):
            self.pre_iterate()
            self.micro_steps()
            self.monolithic_step()
            self.post_iterate()
        self.post_loop()

    def compile_forms(self):
        (ps,pf) = (self.ps,self.pf)
        for p in self.plist:
            p.set_forms()
            self.r_ufl[p] = p.a_ufl - p.l_ufl
            self.j_ufl[p]  = ufl.derivative(self.r_ufl[p], p.u)
            self.j_compiled[p]  = fem.compile_form(p.domain.comm, self.j_ufl[p],
                                          form_compiler_options={"scalar_type": np.float64})
            self.r_compiled[p] = fem.compile_form(p.domain.comm, self.r_ufl[p],
                                          form_compiler_options={"scalar_type": np.float64})
        # Monolithic problem
        self.r_mono_ufl = (ps.a_ufl + pf.a_ufl) - (ps.l_ufl + pf.l_ufl)
        self.j_mono_ufl  = ufl.derivative(self.r_mono_ufl, ps.u) + ufl.derivative(self.r_mono_ufl, pf.u)
        self.j_mono_compiled  = fem.compile_form(ps.domain.comm, self.j_mono_ufl,
                                      form_compiler_options={"scalar_type": np.float64})
        self.r_mono_compiled = fem.compile_form(ps.domain.comm, self.r_mono_ufl,
                                      form_compiler_options={"scalar_type": np.float64})

    def pre_loop(self, prepare_fast_problem=True):
        super().pre_loop()
        (ps,pf) = (self.ps,self.pf)
        self.num_micro_steps = np.rint(ps.dt.value / pf.dt.value).astype(np.int32)
        # Prepare fast problem
        self.set_dirichlet_fast()
        self.initialize_post()
        if prepare_fast_problem:
            self.instantiate_forms(pf)
            pf.pre_assemble()

    def post_iterate(self):
        (ps,pf) = (self.ps,self.pf)
        for p in [ps,pf]:
            p.post_iterate()

    def set_dirichlet_fast(self):
        '''
        - Called right before micro-steps
        - Interpolate macro tnp1 sol to Gamma nodes
        '''
        (ps,pf) = (self.ps,self.pf)
        pf.clear_dirchlet_bcs()
        dirichlet_fun_fast = fem.Function(pf.v, name="dirichlet_con_fast")
        self.gamma_dofs_fast = fem.locate_dofs_topological(pf.v, pf.dim-1, pf.gamma_facets[ps].find(1))
        # Set Gamma dirichlet
        self.fast_dirichlet_tcon = fem.dirichletbc(dirichlet_fun_fast, self.gamma_dofs_fast)
        pf.dirichlet_bcs.append(self.fast_dirichlet_tcon)

    def is_substepping(self):
        return (self.t1_macro_step - (self.pf.time + self.pf.dt.value)) > 1e-7

    def micro_step(self):
        (ps,pf) = (self.ps,self.pf)
        forced_time_derivative = (pf.time - self.t0_macro_step) < 1e-7
        pf.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
        pf.clear_subdomain_data()
        pf.instantiate_forms()
        f = self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
        # Update Dirichlet BC pf here!
        self.fast_dirichlet_tcon.g.x.array[self.gamma_dofs_fast] = \
                (1-f)*ps.u_prev.x.array[self.gamma_dofs_fast] + \
                f*ps.u.x.array[self.gamma_dofs_fast]
        pf.non_linear_solve()

    def set_gamma_slow_to_fast(self):
        (ps, pf) = (self.ps, self.pf)
        ps.u.x.array[self.dofs_fast] = pf.u.x.array[self.dofs_fast]
        ps.dt_func.x.array[self.gamma_dofs_fast] = pf.dt.value
        ps.u_prev.x.array[self.gamma_dofs_fast] = pf.u_prev.x.array[self.gamma_dofs_fast]

    def instantiate_monolithic_forms(self):
        (ps, pf) = (self.ps, self.pf)
        for p in ps, pf:
            p.set_form_subdomain_data()
        cell_subdomain_data = [subdomain for p in [ps, pf] for subdomain in p.form_subdomain_data[fem.IntegralType.cell]]
        facet_subdomain_data = [subdomain for p in [ps, pf] for subdomain in p.form_subdomain_data[fem.IntegralType.exterior_facet]]
        form_subdomain_data = {fem.IntegralType.cell:cell_subdomain_data,
                               fem.IntegralType.exterior_facet:facet_subdomain_data}
        rcoeffmap, rconstmap = get_identity_maps(self.r_mono_ufl)
        r_instance = fem.create_form(self.r_mono_compiled,
                                     [ps.v],
                                     msh=ps.domain,
                                     subdomains=form_subdomain_data,
                                     coefficient_map=rcoeffmap,
                                     constant_map=rconstmap)
        lcoeffmap, lconstmap = get_identity_maps(self.j_mono_ufl)
        j_instance = fem.create_form(self.j_mono_compiled,
                                     [ps.v, ps.v],
                                     msh=ps.domain,
                                     subdomains=form_subdomain_data,
                                     coefficient_map=lcoeffmap,
                                     constant_map=lconstmap)
        return j_instance, r_instance

    def set_snes_sol_vector(self, x) -> PETSc.Vec:  # type: ignore[no-any-unimported]
        """
        Set PETSc.Vec to be passed to PETSc.SNES.solve to initial guess
        """
        (ps, pf) = (self.ps, self.pf)
        with multiphenicsx.fem.petsc.VecSubVectorWrapper(
                x, ps.v.dofmap, self.initial_restriction) as x_wrapper:
            with pf.u.x.petsc_vec.localForm() as uf_sub_vector_local, \
              ps.u.x.petsc_vec.localForm() as us_sub_vector_local:
                  x_wrapper[:] = us_sub_vector_local
                  x_wrapper[self.dofs_fast] = uf_sub_vector_local[self.dofs_fast]

    def update_solution(self, sol_vector):
        (ps, pf) = (self.ps, self.pf)
        with pf.u.x.petsc_vec.localForm() as uf_sub_vector_local, \
             ps.u.x.petsc_vec.localForm() as us_sub_vector_local, \
                multiphenicsx.fem.petsc.VecSubVectorWrapper(sol_vector, ps.v.dofmap, self.initial_restriction) as sol_vector_wrapper:
                    us_sub_vector_local[:] = sol_vector_wrapper
                    uf_sub_vector_local[:] = sol_vector_wrapper
        ps.u.x.scatter_forward()
        pf.u.x.scatter_forward()

    def assemble_residual(self, snes: PETSc.SNES, x: PETSc.Vec, R_vec: PETSc.Vec): 
        self.update_solution(x)
        (ps, pf) = (self.ps, self.pf)
        with R_vec.localForm() as R_local:
            R_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector(R_vec,
                                                self.r_instance,
                                                restriction=self.initial_restriction)
        # TODO: Dirichlet here?
        R_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    def assemble_jacobian(
            self, snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P_mat: PETSc.Mat):
        (ps, pf) = (self.ps, self.pf)
        J_mat.zeroEntries()
        multiphenicsx.fem.petsc.assemble_matrix(J_mat, self.j_instance, restriction=(self.initial_restriction, self.initial_restriction))
        J_mat.assemble()


    def obj(  # type: ignore[no-any-unimported]
        self, snes: PETSc.SNES, x: PETSc.Vec
    ) -> np.float64:
        (ps, pf) = (self.ps, self.pf)
        """Compute the norm of the residual."""
        self.assemble_residual(snes, x, self.obj_vec)
        return self.obj_vec.norm()  # type: ignore[no-any-return]

    def monolithic_step(self):
        (ps, pf) = (self.ps, self.pf)
        pf.clear_dirchlet_bcs()
        pf.pre_iterate()
        ps.pre_iterate(forced_time_derivative=True)

        self.set_gamma_slow_to_fast()
        self.j_instance, self.r_instance = self.instantiate_monolithic_forms()

        self.A = multiphenicsx.fem.petsc.create_matrix(self.j_instance, restriction=(self.initial_restriction, self.initial_restriction))
        self.R = multiphenicsx.fem.petsc.create_vector(self.r_instance, restriction=self.initial_restriction)
        self.x = multiphenicsx.fem.petsc.create_vector(self.r_instance, restriction=self.initial_restriction)
        self.obj_vec = multiphenicsx.fem.petsc.create_vector(self.r_instance, restriction=self.initial_restriction)
        lin_algebra_objects = [self.A, self.R, self.x, self.obj_vec]

        # SOLVE
        # Solve
        snes = PETSc.SNES().create(ps.domain.comm)
        snes.setTolerances(max_it=self.max_nr_iters)
        ksp_opts = PETSc.Options()
        for k,v in ps.linear_solver_opts.items():
            ksp_opts[k] = v
        snes.getKSP().setFromOptions()
        snes.setObjective(self.obj)
        snes.setFunction(self.assemble_residual, self.R)
        snes.setJacobian(self.assemble_jacobian, J=self.A, P=None)
        snes.setMonitor(lambda _, it, residual: print(it, residual))
        self.set_snes_sol_vector(self.x)
        snes.solve(None, self.x)
        self.update_solution(self.x)
        snes.destroy()
        for ds in lin_algebra_objects:
            ds.destroy()

        # POST-ITERATE
        ps.post_iterate()
        pf.post_iterate()
        ps.set_dt(ps.dt.value)
        pf.dirichlet_bcs = [self.fast_dirichlet_tcon]

        self.micro_iter += 1
        self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
        if self.do_writepos:
            self.writepos()

    def writepos(self,case="macro",extra_funs=[]):
        if not(self.do_writepos):
            return
        (pf, ps) = (self.pf, self.ps)
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
        funs = [p.u, p.gamma_nodes[p_ext],p.source.fem_function,p.active_els_func,p.grad_u,
                p.u_prev,self.u_prev[p], p.material_id]
        funs += extra_funs

        p.compute_gradient()
        #print(f"time = {time}, micro_iter = {self.micro_iter}, macro_iter = {self.macro_iter}")
        self.writers[pf].write_function(funs,t=time)


class MHSStaggeredSubstepper(MHSSubstepper):
    def __init__(self,
                 staggered_driver_class : typing.Type[StaggeredDomainDecompositionDriver],
                 staggered_relaxation_factors : list[float],
                 slow_problem : Problem,
                 max_nr_iters=25, max_ls_iters=5,
                 compile_forms=True):
        super().__init__(slow_problem, max_nr_iters, max_ls_iters, compile_forms)
        self.staggered_driver = staggered_driver_class(self.pf, self.ps,
                                                       max_staggered_iters=self.params["max_staggered_iters"],
                                                       initial_relaxation_factors=staggered_relaxation_factors)
        self.name = "staggered_substepper"

    def do_substepped_timestep(self):
        (ps, pf) = (self.ps, self.pf)
        staggered_driver = self.staggered_driver
        self.update_fast_problem()
        staggered_driver.pre_loop(prepare_subproblems=False)
        self.pre_loop()
        if self.do_predictor:
            self.predictor_step(writepos=self.do_writepos)
        for p in (ps, pf):
            staggered_driver.instantiate_forms(p)
            p.pre_assemble()
        for _ in range(staggered_driver.max_staggered_iters):
            self.pre_iterate()
            staggered_driver.pre_iterate()
            self.iterate()
            self.post_iterate()
            staggered_driver.post_iterate(verbose=True)
            if self.do_writepos:
                self.writepos(case="macro")
            if staggered_driver.convergence_crit < staggered_driver.convergence_threshold:
                break
        self.post_loop()

    def compile_forms(self):
        ps = self.ps
        ps.set_forms()
        self.r_ufl[ps] = ps.a_ufl - ps.l_ufl
        self.j_ufl[ps]  = ufl.derivative(self.r_ufl[ps], ps.u)
        self.j_compiled[ps]  = fem.compile_form(ps.domain.comm, self.j_ufl[ps],
                                      form_compiler_options={"scalar_type": np.float64})
        self.r_compiled[ps] = fem.compile_form(ps.domain.comm, self.r_ufl[ps],
                                      form_compiler_options={"scalar_type": np.float64})

    def writepos(self,case="macro",extra_funs=[]):
        if not(self.do_writepos):
            return
        (ps,pf) = (self.ps,self.pf)
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
        funs = [p.u,p.gamma_nodes[p_ext], p.source.fem_function, p.active_els_func, p.grad_u,
                p.u_prev, self.u_prev[p], p.material_id]
        for fun_dic in [sd.ext_flux,sd.net_ext_flux,sd.ext_sol,
                        sd.prev_ext_flux,sd.prev_ext_sol]:
            try:
                funs.append(fun_dic[p])
            except (AttributeError, KeyError):
                continue

        funs += extra_funs
        self.writers[p].write_function(funs,t=time)

    def is_substepping(self):
        return (self.t1_macro_step - self.pf.time) > 1e-7

    def micro_step(self):
        (ps,pf) = (self.ps,self.pf)
        sd = self.staggered_driver
        forced_time_derivative = (pf.time - self.t0_macro_step) < 1e-7
        pf.pre_iterate(forced_time_derivative=forced_time_derivative,verbose=False)
        pf.clear_subdomain_data()
        sd.instantiate_forms(pf)
        self.fraction_macro_step = (pf.time-self.t0_macro_step)/(self.t1_macro_step-self.t0_macro_step)
        if type(sd)==StaggeredRRDriver:
            f = self.fraction_macro_step
            sd.net_ext_sol[pf].x.array[:] = (1-f)*self.ext_sol_tn[pf].x.array[:] + \
                    f*self.ext_sol_array_tnp1[:]
            sd.net_ext_flux[pf].x.array[:] = (1-f)*self.ext_flux_tn[pf].x.array[:] + \
                    f*self.ext_flux_array_tnp1[:]
        elif type(sd)==StaggeredDNDriver and sd.p_dirichlet==pf:
            sd.dirichlet_tcon.g.x.array[:] = (1-sd.relaxation_coeff[pf].value)*pf.u.x.array[:] + \
                                             sd.relaxation_coeff[pf].value*((1-self.fraction_macro_step)*\
                                             sd.prev_ext_sol[pf].x.array[:] + self.fraction_macro_step*\
                                             sd.ext_sol[pf].x.array[:])
        elif type(sd)==StaggeredDNDriver and sd.p_neumann==pf:
            sd.relaxation_coeff[pf].value = self.fraction_macro_step
        pf.non_linear_solve()

    def iterate_substepped_rr(self):
        (ps,pf) = (self.ps,self.pf)
        rr_driver = self.staggered_driver
        assert(type(rr_driver)==StaggeredRRDriver)
        self.update_robin_fast()
        self.micro_steps()
        # have solution at tnp1
        # Updated sol, grad, conduc from n+1
        # I have to apply curr sol, grad, conduc from n+1 weighted with the one from n

        rr_driver.instantiate_forms(ps)
        rr_driver.update_robin(ps)
        ps.pre_iterate(forced_time_derivative=True)
        ps.non_linear_solve()

    def iterate_substepped_dn(self):
        dn_driver = self.staggered_driver
        assert(type(dn_driver)==StaggeredDNDriver)
        (pn,pd) = (self.ps,self.pf)

        # Solve fast/Dirichlet problem
        self.micro_steps()

        dn_driver.instantiate_forms(pn)
        dn_driver.update_neumann_interface()
        # Solve slow/Neumann problem
        pn.pre_iterate(forced_time_derivative=True)
        pn.non_linear_solve()
        dn_driver.update_relaxation_factor()
        dn_driver.update_dirichlet_interface()

    def pre_iterate(self):
        super().pre_iterate()
        self.relaxation_coeff_pf = self.staggered_driver.relaxation_coeff[self.pf].value
        # TODO: Undo pre-iterate of source
        # TODO: Undo pre-iterate of domain
        # TODO: Undo post-iterate of problem

    def update_robin_fast(self):
        '''
        Update Robin condition of fast problem before micro-steps
        '''
        sd = self.staggered_driver
        pf = self.pf
        assert(type(sd)==StaggeredRRDriver)
        sd.update_robin(pf)
        self.ext_sol_array_tnp1 = sd.net_ext_sol[pf].x.array.copy()
        self.ext_flux_array_tnp1 = sd.net_ext_flux[pf].x.array.copy()

    def post_iterate(self):
        (ps,pf) = (self.ps,self.pf)
        self.staggered_driver.relaxation_coeff[pf].value = self.relaxation_coeff_pf
        for p in [ps,pf]:
            p.post_iterate()


    def pre_loop(self):
        super().pre_loop()
        (ps,pf) = (self.ps,self.pf)
        sd = self.staggered_driver
        if type(sd)==StaggeredRRDriver:
            self.iterate = self.iterate_substepped_rr
        elif type(sd)==StaggeredDNDriver:
            self.iterate = self.iterate_substepped_dn
        else:
            raise ValueError("Unknown staggered driver type.")
        for u in self.u_prev.values():
            u.name = "u_prev_driver"
        (p,p_ext) = (pf,ps)
        self.ext_flux_tn = {p:fem.Function(p.dg0_vec,name="ext_flux_tn")}
        p_ext.compute_gradient()
        # TODO: Are these necessary? Can I get them from my own data?
        if type(sd)==StaggeredRRDriver:
            self.ext_sol_tn = {p:fem.Function(p.v,name="ext_sol_tn")}
            propagate_dg0_at_facets_same_mesh(p_ext, p_ext.grad_u, p, self.ext_flux_tn[p])
            self.ext_sol_tn[p].x.array[:] = p_ext.u.x.array[:]
        elif type(sd)==StaggeredDNDriver:
            if sd.p_dirichlet==pf:
                self.ext_sol_tn = {p:fem.Function(p.v,name="ext_sol_tn")}
                self.ext_sol_tn[p].x.array[:] = p_ext.u.x.array[:]
            else:
                propagate_dg0_at_facets_same_mesh(p_ext, p_ext.grad_u, p, self.ext_flux_tn[p])

    def post_loop(self):
        super().post_loop()
        (ps,pf) = (self.ps,self.pf)
        # Update solution
        ps.u.x.array[self.dofs_fast] = pf.u.x.array[self.dofs_fast]
        ps.u.x.scatter_forward()
        ps.is_grad_computed = False
        pf.u.x.array[:] = ps.u.x.array[:]
        pf.is_grad_computed = False
