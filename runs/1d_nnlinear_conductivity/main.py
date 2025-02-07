from dolfinx import mesh, fem
from mhs_fenicsx.problem import Problem
from mpi4py import MPI
import yaml
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main(params):
    nelems = 20
    L = 2.0
    ul, ur = 2.0, 7.0
    k_0, beta = 1.0, 1.0
    k = lambda u : k_0 * ( 1 + beta * u )
    u_exact = lambda x : ((1 + beta * ul) * \
                          np.exp( x[0]/L * \
                          np.log((1 + beta*ur)/(1 + beta*ul))) -1.0) \
                          / beta
    c2 = ul*(ul+2)
    c1 = (ur*(ur+2) - ul*(ul+2)) / 2.0
    u_exact = lambda x : -1 + np.sqrt(1 + c2 + c1*x[0])
    domain = mesh.create_interval(comm,
                                  nelems,
                                  [0.0, L])
    p = Problem(domain, params, "nnlinearconduc")
    marker  = lambda x : np.logical_or(np.isclose(x[0],0),
                                       np.isclose(x[0],L))
    u_d = fem.Function(p.v,name="exact")
    u_d.interpolate(u_exact)
    bdofs_dir  = fem.locate_dofs_geometrical(p.v,marker)
    p.dirichlet_bcs = [fem.dirichletbc(u_d, bdofs_dir)]

    # Solve
    p.set_forms()
    p.compile_create_forms()
    p.pre_assemble()
    p.non_linear_solve()
    p.post_iterate()
    p.writepos(extra_funcs=[u_d])

if __name__=="__main__":
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    main(params)
