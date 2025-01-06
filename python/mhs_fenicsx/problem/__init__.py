from mhs_fenicsx.problem.heatsource import HeatSource, Gaussian1D, Gaussian2D, Gaussian3D, LumpedHeatSource
from mhs_fenicsx.problem.helpers import interpolate, l2_squared, locate_active_boundary, get_facet_integration_entities, get_mask, interpolate_dg_at_facets, inidices_to_nodal_meshtag, indices_to_function, assert_pointwise_vals
from mhs_fenicsx.problem.material import Material
from mhs_fenicsx.problem.problem import Problem, GammaL2Dotter
