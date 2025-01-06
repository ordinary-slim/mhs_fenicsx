import numpy as np
from mhs_fenicsx.problem import Problem

class NewtonRaphson:
    '''
    NR driver with basic backtracking for non-linear problems
    Temporary unoptimized solution
    Usage not recommended over PETSc SNES or dolfinx drivers
    '''
    def __init__(self,problem:Problem,max_nr_iters=25,max_ls_iters=5):
        self.p = problem
        self.max_nr_iters = max_nr_iters
        self.max_ls_iters = max_ls_iters

    def solve(self):
        p = self.p
        nr_iter = 0
        self.nr_converged = False
        if not(p.has_preassembled):
            p.pre_assemble()
        p.assemble_residual()
        while (nr_iter < self.max_nr_iters) and (not(self.nr_converged)):
            nr_iter += 1
            un = p.u.copy();un.name="un"
            p.assemble_jacobian()
            p.solve()

            residual_work_n   = p.x.dot(p.L)
            p.assemble_residual()
            residual_work_np1 = p.x.dot(p.L)
            # LINE-SEARCH
            ls_iter = 0
            relaxation_coeff = 1.0
            while (abs(residual_work_np1) >= 0.8*abs(residual_work_n)) \
                and (ls_iter < self.max_ls_iters):
                    ls_iter += 1
                    relaxation_coeff *= 0.5
                    p.u.x.array[:] = un.x.array[:] + relaxation_coeff*p.du.x.array[:]
                    p.assemble_residual()
                    residual_work_np1 = p.x.dot(p.L)
            correction_norm = relaxation_coeff*np.sqrt(p.du.x.petsc_vec.dot(p.du.x.petsc_vec))
            solution_norm = p.u.x.petsc_vec.norm(0)
            relative_change = correction_norm / solution_norm
            print(f"NR iter #{nr_iter}, residual_work = {residual_work_np1}, did {ls_iter} line search iterations, step norm = {correction_norm}, relative_change = {relative_change}.")
            if relative_change < 1e-7:
                self.nr_converged = True
        if not(self.nr_converged):
            exit("NR iters did not converge!")
