from mhs_fenicsx.problem import Problem
from dolfinx import fem, mesh
import basix.ufl
from mhs_fenicsx_cpp import cellwise_determine_point_ownership, scatter_cell_integration_data_po, \
                            create_robin_robin_monolithic, interpolate_dg0_at_facets
from ffcx.element_interface import map_facet_points
import numpy as np

class MonolithicDomainDecompositionDriver:
    def __init__(self, sub_problem_1:Problem,sub_problem_2:Problem, quadrature_degree):
        (p1,p2) = (sub_problem_1,sub_problem_2)
        self.p1 = p1
        self.p2 = p2
        self.quadrature_degree = quadrature_degree

    def pre_iterate(self):
        ''' Find interface '''
        (p1,p2) = (self.p1,self.p2)
        self.gamma_cells = {
                    p1 : p1.gamma_integration_data[::2],
                    p2 : p2.gamma_integration_data[::2],
                    }
        self.gamma_qpoints_po = {
                    p1 : { p2 : None },
                    p2 : { p1 : None },
                    }
        self.gamma_renumbered_cells_ext = { p1 : {}, p2 : {}}
        self.gamma_dofs_cells_ext = { p1 : {}, p2 : {}}
        self.gamma_geoms_cells_ext = { p1 : {}, p2 : {}}
        self.gamma_iid = {p1:{}, p2:{}}
        self.ext_conductivity = {}
        self.midpoints_gamma = {p1:None, p2:None}


    def iterate(self):
        ''' Solve '''
        pass

    def post_iterate(self):
        ''' Solve '''
        pass

class MonolithicRRDriver(MonolithicDomainDecompositionDriver):
    def __init__(self, sub_problem_1:Problem,sub_problem_2:Problem, quadrature_degree):
        (p1,p2) = (sub_problem_1,sub_problem_2)
        super().__init__(p1, p2, quadrature_degree)

    def pre_iterate(self):
        super().pre_iterate()
        (p1,p2) = (self.p1,self.p2)
        self.A12 = self.assemble_robin_matrix(p1, p2, self.quadrature_degree)
        self.A21 = self.assemble_robin_matrix(p2, p1, self.quadrature_degree)

    def assemble_robin_matrix(self, p:Problem, p_ext:Problem, quadrature_degree=2):
        cdim = p.domain.topology.dim
        # GENERATE QUADRATURE
        cell_type =  p.domain.topology.entity_types[-1][0].name
        facet_type = p.domain.topology.entity_types[-2][0].name
        Qe = basix.ufl.quadrature_element(facet_type,
                                          degree=quadrature_degree)

        num_gps_facet = Qe.num_entity_dofs[-1][0]
        num_facets_cell = p.domain.ufl_cell().num_facets()
        quadrature_points_cell  = np.zeros((num_gps_facet * num_facets_cell, cdim), dtype=Qe._points.dtype)
        for ifacet in range(num_facets_cell):
            quadrature_points_cell[ifacet*num_gps_facet : ifacet*num_gps_facet + num_gps_facet, :cdim] = map_facet_points(Qe._points, ifacet, cell_type)

        # Manually tabulate
        num_local_gamma_cells = p.gamma_integration_data.size // 2
        gamma_qpoints = np.zeros((num_local_gamma_cells * \
                num_gps_facet, 3), dtype=np.float64)
        pgeo = p.domain.geometry
        for idx in range(num_local_gamma_cells):
            icell   = p.gamma_integration_data[2*idx]
            lifacet = p.gamma_integration_data[2*idx+1]
            ref_points = quadrature_points_cell[lifacet*num_gps_facet:lifacet*num_gps_facet+num_gps_facet, :]
            # Push forward
            gamma_qpoints[idx*num_gps_facet:\
                idx*num_gps_facet + num_gps_facet, :] =  \
                pgeo.cmap.push_forward(
                        ref_points, pgeo.x[pgeo.dofmap[icell]])

        self.gamma_qpoints_po[p][p_ext] = \
                cellwise_determine_point_ownership(p_ext.domain._cpp_object,
                                                   gamma_qpoints,
                                                   self.gamma_cells[p_ext],
                                                   np.float64(1e-7))
        self.gamma_renumbered_cells_ext[p][p_ext], \
        self.gamma_dofs_cells_ext[p][p_ext], \
        self.gamma_geoms_cells_ext[p][p_ext] = \
                        scatter_cell_integration_data_po(self.gamma_qpoints_po[p][p_ext],
                                                          p_ext.v._cpp_object,
                                                          p_ext.restriction)
        self.midpoints_gamma[p] = mesh.compute_midpoints(p.domain,p.domain.topology.dim-1,p.gamma_facets.find(1))
        self.gamma_iid[p][p_ext] = cellwise_determine_point_ownership(
                                            p_ext.domain._cpp_object,
                                            self.midpoints_gamma[p],
                                            self.gamma_cells[p_ext],
                                            np.float64(1e-6))
        self.ext_conductivity[p] = fem.Function(p.dg0, name="ext_k")
        interpolate_dg0_at_facets([p_ext.k._cpp_object],
                                  [self.ext_conductivity[p]._cpp_object],
                                  p.active_els_func._cpp_object,
                                  p.gamma_facets._cpp_object,
                                  self.gamma_cells[p],
                                  self.gamma_iid[p][p_ext],
                                  p.gamma_facets_index_map,
                                  p.gamma_imap_to_global_imap)

        p.domain.topology.create_entity_permutations()
        A = create_robin_robin_monolithic(self.ext_conductivity[p]._cpp_object,
                                          gamma_qpoints,
                                          quadrature_points_cell,
                                          Qe._weights,
                                          p.v._cpp_object,
                                          p.restriction,
                                          p_ext.v._cpp_object,
                                          p_ext.restriction,
                                          p.gamma_integration_data,
                                          self.gamma_qpoints_po[p][p_ext],
                                          self.gamma_renumbered_cells_ext[p][p_ext],
                                          self.gamma_dofs_cells_ext[p][p_ext],
                                          self.gamma_geoms_cells_ext[p][p_ext],
                                          )
        A.assemble()
        return A
