from dolfinx import mesh, fem
from mhs_fenicsx.problem import Problem
import numpy as np
from mpi4py import MPI
import yaml
from mhs_fenicsx.problem.helpers import assert_pointwise_vals

comm = MPI.COMM_WORLD

def mesh_box(box,el_size):
    nx = np.round((box[3]-box[0]) / el_size).astype(np.int32)
    ny = np.round((box[4]-box[1]) / el_size).astype(np.int32)
    nz = np.round((box[5]-box[2]) / el_size).astype(np.int32)
    return mesh.create_box(MPI.COMM_WORLD,
           [box[:3], box[3:]],
           [nx,ny,nz],
           mesh.CellType.hexahedron,
           )

points = np.array([
    [-0.005, 0.025, 0.0],
    [+0.005, -0.025, 0.02],
    [+0.005, +0.015, 0.02],
    ],dtype=np.float64)
target_vals = np.array([87.69508918580138, 784.1606374650604, 913.5767591386578],dtype=np.float64)

def run():
    with open("input.yaml", 'r') as f:
        params = yaml.safe_load(f)
    el_size = 0.01
    domain = mesh_box([-0.005, -0.025, 0.0, 0.005, +0.025, 0.02],el_size)
    problem = Problem(domain,params,name="5on5")
    
    problem.u.x.array[:] = problem.T_env
    active_els = fem.locate_dofs_geometrical(problem.dg0, lambda x : x[problem.domain.topology.dim-1] < 0.01 )
    problem.set_activation(active_els)

    problem.set_forms_domain()
    problem.set_forms_boundary()
    problem.compile_forms()

    for _ in range(5):
        problem.pre_iterate()
        problem.set_activation(np.hstack((problem.active_els_func.x.array.nonzero()[0], problem.source.heated_els)))
        problem.u_prev.x.array[problem.just_activated_nodes] = problem.T_dep
        problem.instantiate_forms()
        problem.pre_assemble()
        problem.non_linear_solve()
        problem.post_iterate()

    return problem

def test_3d_printing_5on5():
    p = run()
    assert_pointwise_vals(p,points,target_vals)

if __name__=="__main__":
    test_3d_printing_5on5()
