import numpy as np
from scipy.optimize import fsolve
from scipy.special import erf, erfc
from dolfinx import fem, mesh, io
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc
plot = True
if plot:
    import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def transcendental_fun(x):
    return np.sqrt(np.pi)*x - \
        (stl / np.exp(np.pow(x,2)) / erf(x) ) + \
        (sts / np.exp(np.pow(x,2)) / erfc(x) )

def locate_interface(t):
    return 2*lamma*np.sqrt(diffusivity*t);

def exact_sol(x,t):
    interface_pos = locate_interface(t);
    sol = np.zeros(x.shape[1],dtype=x.dtype)
    liquid_indices = x[0]<=interface_pos
    solid_indices  = x[0]>interface_pos
    sol[liquid_indices] = T_l - (T_l - T_m)*erf(x[0,liquid_indices]/(2*np.sqrt(diffusivity*t)))/erf(lamma)
    sol[solid_indices] = T_s + (T_m - T_s)*erfc(x[0,solid_indices]/(2*np.sqrt(diffusivity*t)))/erfc(lamma)
    return sol



## PARAMETERS
x_left = 0.0
x_right = 0.1
cross_section = 1.0
run_type = "Newton-Raphson"#Picard, Newton-Raphson
rho = 4510*cross_section
c_p = 520
k   = 16*cross_section
diffusivity = k / rho / c_p
l = 325000
T_m = 1750
T_l = 2000
T_s = 1500
T_0 = T_s
stl = c_p*(T_l - T_m) / l
sts = c_p*(T_m - T_s) / l
lamma = fsolve(transcendental_fun, 0.388150542167233)
time = 0.0
S   = 40
max_nr_iters = 20
dt = 1.0
num_time_steps = 20

Tsl_av = (T_s + T_l)/2.0;
cte = S*2/(T_l - T_s);
fl =   lambda tem : 1/2*(np.tanh(cte*(tem - Tsl_av))+1.0)
flp  = lambda tem : cte/2.0*(1 - np.pow(np.tanh(cte*(tem - Tsl_av)), 2))
flpp = lambda tem : -np.pow(cte,2)*(np.tanh(cte*(tem - Tsl_av))*np.pow(1.0/np.cosh(cte*(tem - Tsl_av)),2))
fl_ufl   = lambda tem : 1/2*(ufl.tanh(cte*(tem - Tsl_av))+1.0)
flp_ufl  = lambda tem : cte/2.0*(1 - ufl.tanh(cte*(tem - Tsl_av))**2)
flpp_ufl = lambda tem : -cte**2 * ufl.tanh(cte*(tem-Tsl_av)) / (ufl.cosh(cte*(tem-Tsl_av)))**2
#flp_ufl = lambda tem : tem**2
#flpp_ufl = lambda tem : 2*tem
nelems = 1000

def main():
    domain = mesh.create_interval(comm,nelems,(x_left,x_right))
    ufl_coeffs = dict()
    for var in ["rho", "c_p", "k", "l", "dt"]:
        ufl_coeffs[var] = fem.Constant(domain,float(globals()[var]))
    tdim = 1
    V = fem.functionspace(domain, ("Lagrange", 1))
    f_exact = fem.Function(V, name="exact")
    u_np1 = fem.Function(V, name="uh")# Solution
    u_n = fem.Function(V, name="uh_n")# Previous time iter
    u_i = fem.Function(V, name="uh_i")# Previous non-linear iter
    du  = fem.Function(V, name="delta_u")# Non-linear increment

    # IC
    u_np1.x.array[:] = T_s

    # Dirichlet BC
    bfacets = mesh.locate_entities_boundary(domain, tdim-1, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(f_exact, fem.locate_dofs_topological(V, tdim-1, bfacets))

    # Forms
    (delta_u, v) = (ufl.TrialFunction(V), ufl.TestFunction(V))
    dx = ufl.dx(metadata={"quadrature_degree": 5})
    F = ufl_coeffs["rho"]*ufl_coeffs["c_p"]/ufl_coeffs["dt"]*u_n*v*dx
    F += ufl_coeffs["rho"]*ufl_coeffs["l"]*flp_ufl(u_np1)/ufl_coeffs["dt"]*u_n*v*dx
    F += -ufl_coeffs["rho"]*ufl_coeffs["c_p"]/ufl_coeffs["dt"]*u_np1*v*dx
    F += -ufl_coeffs["rho"]*ufl_coeffs["l"]*flp_ufl(u_np1)/ufl_coeffs["dt"]*u_np1*v*dx
    F += -ufl_coeffs["k"]*ufl.dot(ufl.grad(u_np1),ufl.grad(v))*dx
    own_derivative = False
    J = ufl.derivative(F,u_np1)
    if own_derivative:
        J  = -ufl_coeffs["rho"]*ufl_coeffs["c_p"]/ufl_coeffs["dt"]*delta_u*v*dx
        J += -ufl_coeffs["rho"]*ufl_coeffs["l"]*flp_ufl(u_np1)/ufl_coeffs["dt"]*delta_u*v*dx
        J += -ufl_coeffs["rho"]*ufl_coeffs["l"]*flpp_ufl(u_np1)/ufl_coeffs["dt"]*(u_np1-u_n)*delta_u*v*dx
        J += -ufl_coeffs["k"]*ufl.dot(ufl.grad(delta_u),ufl.grad(v))*dx
    residual = fem.form(F)
    jacobian = fem.form(J)

    A = fem.petsc.create_matrix(jacobian)
    L = fem.petsc.create_vector(residual)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)

    # TIME-LOOP
    time = 0.0
    for titer in range(num_time_steps):
        time += dt
        u_n.x.array[:] = u_np1.x.array[:]
        lamma_exact_sol = lambda x : exact_sol(x,time)
        f_exact.interpolate(lamma_exact_sol)

        nriter = 0
        while nriter < max_nr_iters:
            nriter += 1
            u_i.x.array[:] = u_np1.x.array[:]
            with L.localForm() as loc_L:
                loc_L.set(0)
            A.zeroEntries()
            fem.petsc.assemble_matrix(A,jacobian, bcs=[bc])
            A.assemble()
            fem.petsc.assemble_vector(L, residual)
            L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            L.scale(-1)

            # Compute b - J(u_D-u_(i-1))
            fem.petsc.apply_lifting(L, [jacobian], [[bc]], x0=[u_i.x.petsc_vec])
            fem.petsc.set_bc(L, [bc], u_i.x.petsc_vec)
            L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

            # Solve
            solver.solve(L, du.x.petsc_vec)
            du.x.scatter_forward()

            u_np1.x.array[:] += du.x.array[:]

            correction_norm = du.x.petsc_vec.norm(0)
            print(f"Iteration {nriter}: Correction norm {correction_norm}")

            '''
            if plot:
                # PLOT
                plt.figure(figsize=(8, 6))
                plt.plot(domain.geometry.x[:,0], u_np1.x.array, label='uh', color='b')
                plt.plot(domain.geometry.x[:,0], f_exact.x.array, label='exact_sol', color='r')
                plt.title('T field')
                plt.xlabel('x')
                plt.ylabel('T')
                plt.legend()
                plt.savefig(f"nr_iter{nriter}.png",dpi=300)
                plt.close()
            '''

            if correction_norm < 1e-10:
                break

        if plot:
            # PLOT
            plt.figure(figsize=(8, 6))
            #plt.plot(domain.geometry.x[:,0], f_exact.x.array, label='exact_sol', color='r')
            plt.plot(domain.geometry.x[:,0], u_np1.x.array, label='uh', color='b')
            plt.title('T field')
            plt.xlabel('x')
            plt.ylabel('T')
            plt.legend()
            plt.savefig(f"time_iter{titer}.png",dpi=300)
            plt.close()

if __name__=="__main__":
    main()
