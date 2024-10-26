from dolfinx import default_scalar_type

base_mat_to_problem = {
            "k" : "k",
            "rho" : "rho",
            "cp" : "cp",
            }
phase_change_mat_to_problem = {
            "L" : "latent_heat",
            "T_s" : "solidus_temperature",
            "T_l" : "liquidus_temperature",
            }

class Material:
    def __init__(self,params):
        self.k   = default_scalar_type(params["conductivity"])
        self.rho = default_scalar_type(params["density"])
        self.cp  = default_scalar_type(params["specific_heat"])
        self.phase_change = False
        if "phase_change" in params:
            self.phase_change = True
            self.L   = default_scalar_type(params["phase_change"]["latent_heat"])
            self.T_s = default_scalar_type(params["phase_change"]["solidus_temperature"])
            self.T_l = default_scalar_type(params["phase_change"]["liquidus_temperature"])
