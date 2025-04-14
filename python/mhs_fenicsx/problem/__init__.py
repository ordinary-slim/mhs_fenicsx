from mhs_fenicsx.problem.heatsource import HeatSource, Gaussian1D, Gaussian2D, Gaussian3D, LumpedHeatSource, create_heat_sources
from mhs_fenicsx.problem.helpers import interpolate_cg1, l2_squared, locate_active_boundary, get_facet_integration_entities, get_mask, interpolate_dg_at_facets, inidices_to_nodal_meshtag, indices_to_function, set_same_mesh_interface, propagate_dg0_at_facets_same_mesh, assert_pointwise_vals
from mhs_fenicsx.problem.material import Material
from mhs_fenicsx.problem.problem import Problem, L2Differ, GammaL2Dotter
