import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import numpy as np

import basix.ufl
from dolfinx import fem, mesh, io, plot
import dolfinx.fem.petsc
import ufl
from meshing import meshBox
import yaml

box = [-16, -5, 16, 5]
with open("input.yaml", "r") as inputFile:
    params = yaml.safe_load( inputFile )

class PowerDensity:
    def __init__(self, p0, 
                 power,
                 radius,
                 speed = np.array([0.0, 0.0]),):
        self.t = 0.0
        self.p0 = p0
        self.power = power
        self.radius = radius
        if speed is not None:
            self.speed = speed

    def __call__(self, x):
        self.pos = self.p0 + self.t*self.speed
        values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
        values = 2 * self.power / np.pi / self.radius**2 * \
            np.exp(  - 2*( (x[0] - self.pos[0])**2 + \
                           (x[1] - self.pos[1])**2) / \
                           self.radius**2  )
        return values

class Problem:
    def __init__(self,
                 domain,
                 params,
                 caseName="case",
                 ):

        self.domain = domain
        # PHYSICAL PARAMS
        self.rho	= fem.Constant( self.domain, PETSc.ScalarType( params["rho"] ) )
        self.k   = fem.Constant( self.domain, PETSc.ScalarType( params["k"] ) )
        self.cp  = fem.Constant( self.domain, PETSc.ScalarType( params["cp"] ) )
        # 2D moving Gaussian
        self.p0  = np.array(params["p0"])
        self.power = params["power"]
        self.radius = params["radius"]
        self.speedDomain = np.reshape(np.array(params["speedDomain"]), (-1, 1))
        self.speedAdvec = np.array(params["speedAdvec"])

        # Define solution and test space
        self.V = fem.functionspace(self.domain, ("CG", 1))
        # Define space for ALE displacement
        self.Vex = self.domain.ufl_domain().ufl_coordinate_element()
        self.u_n = fem.Function(self.V)
        self.uh  = fem.Function(self.V)
        self.uh.name = "uh"
        self.source = fem.Function(self.V)
        self.Vdisp = fem.functionspace(self.domain, self.Vex)
        self.disp = fem.Function(self.Vdisp)

        self.it = 0
        self.time = 0.0
        self.dt = params["dt"]

        speedHs = np.array(params["speed"])
        self.source = PowerDensity(self.p0, self.power, self.radius, speedHs)

        self.f = fem.Function(self.V)
        self.f.name = "Source"
        self.f.interpolate(self.source)

        # INITIAL CONDITION
        self.Tenv= params["Tenv"]
        self.u_n.interpolate(lambda x : self.Tenv*np.ones( x.shape[1] ) )

        # initialize io
        self.vtk = io.VTKFile( MPI.COMM_WORLD,
                              "post/{}.pvd".format( caseName),
                              "w" )


    def setForms(self):
        speedAdvecUfl = fem.Constant( self.domain, PETSc.ScalarType( tuple(self.speedAdvec) ) )
        # VARIATIONAL PROBLEM
        dx = ufl.Measure(
                "dx",
                metadata={"quadrature_rule":"vertex",
                          "quadrature_degree":1,
                          }
                )
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        a = self.rho * self.cp * ( 1/self.dt ) * u * v * dx + self.k * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx \
                + self.rho * self.cp * ufl.dot( speedAdvecUfl, ufl.grad(u) ) * v * dx
        L = (self.rho*self.cp* (1 / self.dt) * self.u_n  + self.f) * v * dx
        self.bilinear_form = fem.form( a )
        self.linear_form = fem.form( L )

    def assembleLhs(self):
        self.A = fem.petsc.assemble_matrix(self.bilinear_form)
        self.A.assemble()

    def createRhs(self):
        self.b = fem.petsc.create_vector(self.linear_form)
    
    def updateRhs(self):
        # Update RHS
        self.source.t = self.time
        self.f.interpolate( self.source )
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b, self.linear_form)


    def setSolver(self):
        # LINEAR SOLVER
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

    def solve(self):
        self.solver.solve( self.b, self.uh.vector )
        self.uh.x.scatter_forward()

    def preIterate(self):
        self.time += self.dt
        self.it += 1
        #Move self.domain
        self.disp.interpolate(lambda x: np.tile(self.dt*self.speedDomain, x.shape[1]))
        self.domain.geometry.x[:,:self.domain.geometry.dim] += self.disp.x.array.reshape((-1, self.domain.geometry.dim))

    def postIterate(self):
        # Update solution at previous self.time step (u_n)
        self.u_n.x.array[:] = self.uh.x.array


    def timestep(self):
        self.preIterate()
        self.updateRhs()
        self.solve()
        self.postIterate()
        self.writepos()

    def writepos(self):
        # Postprocess
        self.vtk.write_function( self.uh, self.time )
        self.vtk.write_function(  self.f, self.time )

def project(projectedFunction,
            targetSpace,
            projection,
            interpolation_degree=1 ):
    # Compute function for fine grid at quadrature points of the coarse grid
    Qe = basix.ufl.quadrature_element(
        targetSpace.mesh.topology.cell_name(), degree=interpolation_degree)
    targetQuadratureSpace = dolfinx.fem.functionspace(targetSpace.mesh, Qe)
    nmmid = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
                                 targetQuadratureSpace.mesh._cpp_object,
                                 targetQuadratureSpace.element,
                                 projectedFunction.ufl_function_space().mesh._cpp_object,
                                 padding=1e-5,)
    q_func = dolfinx.fem.Function(targetQuadratureSpace)
    q_func.interpolate(projectedFunction, nmm_interpolation_data=nmmid)
    # VARIATIONAL PROBLEM
    dx = ufl.Measure(
            "dx",
            )
    u, v = ufl.TrialFunction(targetSpace), ufl.TestFunction(targetSpace)
    a = ufl.inner( u, v ) * dx
    L = ufl.inner( q_func, v ) * dx
    problem = dolfinx.fem.petsc.LinearProblem(a, L, u = projection )
    problem.solve()

def main():
    domain = meshBox(box)
    p0 = params["p0"]
    Rback, Rside = 5.0, 1.0
    movingBox = [p0[0] - Rback, p0[1] - Rside, p0[0] + Rside, p0[1] + Rside]
    movingDomain = meshBox( movingBox )
    pFixed = Problem(domain, params)
    movParams = dict(params)
    movParams["speedDomain"] = movParams["speed"]
    pMoving = Problem(movingDomain, movParams, caseName="movingProblem")

    pFixed.setForms()
    pFixed.assembleLhs()
    pFixed.createRhs()
    pFixed.setSolver()

    # TIME LOOP
    maxIter = 20
    for _ in range(maxIter):
        pFixed.preIterate()
        pMoving.preIterate()
        pFixed.updateRhs()
        pFixed.solve()
        # PROJECTION
        project( pFixed.uh, pMoving.V, pMoving.uh )
        pFixed.postIterate()
        pFixed.writepos()
        pMoving.writepos()

if __name__=="__main__":
    main()
