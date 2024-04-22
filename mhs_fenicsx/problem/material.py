from dolfinx import default_scalar_type

class Material:
    def __init__(self,params):
        self.k   = default_scalar_type(params["conductivity"])
        self.rho = default_scalar_type(params["density"])
        self.cp  = default_scalar_type(params["specific_heat"])
