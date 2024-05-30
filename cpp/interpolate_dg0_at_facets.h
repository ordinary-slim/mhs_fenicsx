#pragma once
#include <cassert>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <vector>

using int_vector = std::vector<int>;

template <std::floating_point T>
void interpolate_dg0_at_facets(const dolfinx::fem::Function<T> &sending_f,
                               dolfinx::fem::Function<T> &receiving_f,
                               int_vector &facets,
                               geometry::BoundingBoxTree<T> &bb_tree_sending_f
                               //const dolfinx::fem::Function<T> &activation_fun_receiving,
                               //const dolfinx::fem::Function<T> &activation_fun_sending
                               ) {
  auto smesh = sending_f.function_space()->mesh();//sending mesh
  auto rmesh = receiving_f.function_space()->mesh();//receiving mesh
  int cdim = rmesh->topology()->dim();
}
