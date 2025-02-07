import numpy as np
from dolfinx import fem
import typing
import ufl

class Material:
    def __init__(self, params: typing.Dict):
        self.k   = to_piecewise_linear(params["conductivity"])
        self.rho = to_piecewise_linear(params["density"])
        self.cp  = to_piecewise_linear(params["specific_heat"])
        self.phase_change = False
        if "phase_change" in params:
            self.phase_change = True
            self.L   = to_piecewise_linear(params["phase_change"]["latent_heat"])
            self.T_s = Constant(params["phase_change"]["solidus_temperature"])
            self.T_l = Constant(params["phase_change"]["liquidus_temperature"])


    def get_handles_liquid_fraction(self, domain, smoothing_cte):
        liquid_fraction, dliquid_fraction = lambda tem : 0.0, lambda tem : 0.0
        if self.phase_change:
            solidus_temperature  = self.T_s.ufl(domain)
            liquidus_temperature  = self.T_l.ufl(domain)
            sigma_temperature   = (liquidus_temperature - solidus_temperature) / 2.0
            melting_temperature = (liquidus_temperature + solidus_temperature) / 2.0
            liquid_fraction   = lambda tem : 0.5*(ufl.tanh((smoothing_cte/sigma_temperature) * (tem - melting_temperature)) + 1)
            dliquid_fraction  = lambda tem : (smoothing_cte/sigma_temperature)/2.0*(1 - ufl.tanh((smoothing_cte/sigma_temperature)*(tem - melting_temperature))**2)
        return liquid_fraction, dliquid_fraction

def to_piecewise_linear(vals: typing.Union[typing.List[typing.Tuple[float, float]],
                                           float]):
    if isinstance(vals, np.ScalarType):
        values = [(-1e9, np.float64(vals)), (+1e9, np.float64(vals))]
    else:
        values = vals
    return PiecewiseLinearProperty(values)

class Constant:
    def __init__(self, value: np.float64):
        self.value = value

    def ufl(self, domain):
        return fem.Constant(domain, self.value)

class PiecewiseLinearProperty:
    def __init__(self, values: typing.List[typing.Tuple[float, float]]):
        self.Xs = np.zeros(len(values), dtype=np.float64)
        self.Ys = np.zeros(len(values), dtype=np.float64)
        for idx, val in enumerate(values):
            self.Xs[idx], self.Ys[idx] = val

    def ufl(self, u):
        """From https://github.com/jpdean/culham_thermomech/blob/dbcb95779dda4ab9185a35965e6bb3ced136b0c1/utils.py"""
        # Compute gradients and constants for each piece
        ms = []
        cs = []
        for i in range(len(self.Xs) - 1):
            ms.append((self.Ys[i + 1] - self.Ys[i]) / (self.Xs[i + 1] - self.Xs[i]))
            cs.append(self.Ys[i] - ms[i] * self.Xs[i])

        # Construct interpolant using UFL conditional
        conditions = [ufl.gt(u, x) for x in self.Xs]
        pieces = [m * u + c for (m, c) in zip(ms, cs)]
        # If u < self.Xs[-1], extrapolate using the last piece
        pieces.append(pieces[-1])

        interp = ufl.conditional(conditions[1], pieces[1], pieces[0])
        for i in range(1, len(conditions) - 1):
            interp = ufl.conditional(conditions[i + 1], pieces[i + 1], interp)
        return interp
