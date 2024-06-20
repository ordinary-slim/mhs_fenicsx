from line_profiler import LineProfiler
from mpi4py import MPI
from dolfinx import mesh, fem, cpp
from mhs_fenicsx.problem import Problem
import yaml
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

with open("input.yaml", 'r') as f:
    params = yaml.safe_load(f)

def exact_sol(x):
    return 2 -(x[0]**2 + x[1]**2)

class Rhs:
    def __init__(self,rho,cp,k,v):
        self.rho = rho
        self.cp = cp
        self.k = k
        self.v = v

    def __call__(self,x):
        return_val = -2*self.rho*self.cp*self.v[0]*x[0] + -2*self.rho*self.cp*self.v[1]*x[1] + 4*self.k
        return return_val

rhs = Rhs(params["material"]["density"],
          params["material"]["specific_heat"],
          params["material"]["conductivity"],
          params["advection_speed"])

def main():
    points_side = params["points_side"]
    big_mesh  = mesh.create_unit_square(MPI.COMM_WORLD, points_side, points_side, mesh.CellType.quadrilateral)
    # Problem: Laplace
    big_p = Problem(big_mesh, params, name="big")
    # Solve
    big_p.set_rhs(rhs)
    big_p.add_dirichlet_bc(exact_sol)
    big_p.set_forms_domain()
    big_p.compile_forms()
    big_p.pre_assemble()
    big_p.assemble()
    big_p.solve()
    big_p.writepos()
    # Submesh
    els_left = fem.locate_dofs_geometrical(big_p.dg0_bg, lambda x : x[0] <= 0.5 )
    submesh = mesh.create_submesh(big_mesh,big_p.dim,els_left)
    small_p = Problem(submesh[0],params, name="small")
    small_p.writepos()


if __name__=="__main__":
    lp = LineProfiler()
    lp_wrapper = lp(main)
    lp_wrapper()
    if rank==0:
        with open("profiling.txt", 'w') as pf:
            lp.print_stats(stream=pf)
