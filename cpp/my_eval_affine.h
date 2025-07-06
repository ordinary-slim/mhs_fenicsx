#include <dolfinx/fem/Function.h>

template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_t<T>>
void eval_affine(const dolfinx::fem::Function<T, U> &f, std::span<const U> x, std::array<std::size_t, 2> xshape,
    std::span<const std::int32_t> cells, std::span<T> u,
    std::array<std::size_t, 2> ushape)
{
  if (cells.empty())
    return;

  assert(x.size() == xshape[0] * xshape[1]);
  assert(u.size() == ushape[0] * ushape[1]);

  // TODO: This could be easily made more efficient by exploiting
  // points being ordered by the cell to which they belong.

  if (xshape[0] != cells.size())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }

  if (xshape[0] != ushape[0])
  {
    throw std::runtime_error(
        "Length of array for Function values must be the "
        "same as the number of points.");
  }

  // Get mesh
  assert(f.function_space());
  auto mesh = f.function_space()->mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);

  // Get coordinate map
  const dolfinx::fem::CoordinateElement<U>& cmap = mesh->geometry().cmap();

  // Get geometry data
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  auto x_g = mesh->geometry().x();

  // Get element
  auto element = f.function_space()->element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size = element->reference_value_size();
  const std::size_t value_size
      = f.function_space()->element()->reference_value_size();
  const std::size_t space_dimension = element->space_dimension() / bs_element;

  // If the space has sub elements, concatenate the evaluations on the
  // sub elements
  const int num_sub_elements = element->num_sub_elements();
  if (num_sub_elements > 1 and num_sub_elements != bs_element)
  {
    throw std::runtime_error("Function::eval is not supported for mixed "
                             "elements. Extract subspaces.");
  }

  // Create work vector for expansion coefficients
  std::vector<T> coefficients(space_dimension * bs_element);

  // Get dofmap
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = f.function_space()->dofmap();
  assert(dofmap);
  const int bs_dof = dofmap->bs();

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  std::vector<U> coord_dofs_b(num_dofs_g * gdim);
  dolfinx::fem::impl::mdspan_t<U, 2> coord_dofs(coord_dofs_b.data(), num_dofs_g,
                                              gdim);
  std::vector<U> xp_b(1 * gdim);
  dolfinx::fem::impl::mdspan_t<U, 2> xp(xp_b.data(), 1, gdim);

  // Loop over points
  std::ranges::fill(u, 0.0);
  std::span<const T> _v = f.x()->array();

  // Evaluate geometry basis at point (0, 0, 0) on the reference cell.
  // Used in affine case.
  std::array<std::size_t, 4> phi0_shape = cmap.tabulate_shape(1, 1);
  std::vector<U> phi0_b(std::reduce(
      phi0_shape.begin(), phi0_shape.end(), 1, std::multiplies{}));
  dolfinx::fem::impl::mdspan_t<const U, 4> phi0(phi0_b.data(), phi0_shape);
  cmap.tabulate(1, std::vector<U>(tdim), {1, tdim}, phi0_b);
  auto dphi0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi0, std::pair(1, tdim + 1), 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  // Reference coordinates for each point
  std::vector<U> Xb(xshape[0] * tdim);
  dolfinx::fem::impl::mdspan_t<U, 2> X(Xb.data(), xshape[0], tdim);

  // Geometry data at each point
  std::vector<U> J_b(xshape[0] * gdim * tdim);
  dolfinx::fem::impl::mdspan_t<U, 3> J(J_b.data(), xshape[0], gdim, tdim);
  std::vector<U> K_b(xshape[0] * tdim * gdim);
  dolfinx::fem::impl::mdspan_t<U, 3> K(K_b.data(), xshape[0], tdim, gdim);
  std::vector<U> detJ(xshape[0]);
  std::vector<U> det_scratch(2 * gdim * tdim);

  // Prepare geometry data in each cell
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    assert(x_dofs.size() == num_dofs_g);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x[p * xshape[1] + j];

    auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

    std::array<U, 3> Xpb = {0, 0, 0};
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        U,
        MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
            std::size_t, 1, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
        Xp(Xpb.data(), 1, tdim);

    // Compute reference coordinates X, and J, detJ and K
    dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi0, coord_dofs,
                                                       _J);
    dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(_J, _K);
    std::array<U, 3> x0 = {0, 0, 0};
    for (std::size_t i = 0; i < coord_dofs.extent(1); ++i)
      x0[i] += coord_dofs(0, i);
    dolfinx::fem::CoordinateElement<U>::pull_back_affine(Xp, _K, x0, xp);
    detJ[p]
        = dolfinx::fem::CoordinateElement<U>::compute_jacobian_determinant(
            _J, det_scratch);

    for (std::size_t j = 0; j < X.extent(1); ++j)
      X(p, j) = Xpb[j];
  }

  // Prepare basis function data structures
  std::vector<U> basis_derivatives_reference_values_b(
      1 * xshape[0] * space_dimension * reference_value_size);
  dolfinx::fem::impl::mdspan_t<const U, 4> basis_derivatives_reference_values(
      basis_derivatives_reference_values_b.data(), 1, xshape[0],
      space_dimension, reference_value_size);
  std::vector<U> basis_values_b(space_dimension * value_size);
  dolfinx::fem::impl::mdspan_t<U, 2> basis_values(basis_values_b.data(),
                                                space_dimension, value_size);

  // Compute basis on reference element
  element->tabulate(basis_derivatives_reference_values_b, Xb,
                    {X.extent(0), X.extent(1)}, 0);

  using xu_t = dolfinx::fem::impl::mdspan_t<U, 2>;
  using xU_t = dolfinx::fem::impl::mdspan_t<const U, 2>;
  using xJ_t = dolfinx::fem::impl::mdspan_t<const U, 2>;
  using xK_t = dolfinx::fem::impl::mdspan_t<const U, 2>;
  auto push_forward_fn
      = element->basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();

  // Transformation function for basis function values
  auto apply_dof_transformation
      = element->template dof_transformation_fn<U>(
          dolfinx::fem::doftransform::standard);

  // Size of tensor for symmetric elements, unused in non-symmetric case, but
  // placed outside the loop for pre-computation.
  int matrix_size;
  if (element->symmetric())
  {
    matrix_size = 0;
    while (matrix_size * matrix_size < ushape[1])
      ++matrix_size;
  }

  const std::size_t num_basis_values = space_dimension * reference_value_size;
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];
    if (cell_index < 0) // Skip negative cell indices
      continue;

    // Permute the reference basis function values to account for the
    // cell's orientation
    apply_dof_transformation(
        std::span(basis_derivatives_reference_values_b.data()
                      + p * num_basis_values,
                  num_basis_values),
        cell_info, cell_index, reference_value_size);

    {
      auto _U = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis_derivatives_reference_values, 0, p,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      push_forward_fn(basis_values, _U, _J, detJ[p], _K);
    }

    // Get degrees of freedom for current cell
    std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell_index);
    for (std::size_t i = 0; i < dofs.size(); ++i)
      for (int k = 0; k < bs_dof; ++k)
        coefficients[bs_dof * i + k] = _v[bs_dof * dofs[i] + k];

    if (element->symmetric())
    {
      int row = 0;
      int rowstart = 0;
      // Compute expansion
      for (int k = 0; k < bs_element; ++k)
      {
        if (k - rowstart > row)
        {
          row++;
          rowstart = k;
        }
        for (std::size_t i = 0; i < space_dimension; ++i)
        {
          for (std::size_t j = 0; j < value_size; ++j)
          {
            u[p * ushape[1]
              + (j * bs_element + row * matrix_size + k - rowstart)]
                += coefficients[bs_element * i + k] * basis_values(i, j);
            if (k - rowstart != row)
            {
              u[p * ushape[1]
                + (j * bs_element + row + matrix_size * (k - rowstart))]
                  += coefficients[bs_element * i + k] * basis_values(i, j);
            }
          }
        }
      }
    }
    else
    {
      // Compute expansion
      for (int k = 0; k < bs_element; ++k)
      {
        for (std::size_t i = 0; i < space_dimension; ++i)
        {
          for (std::size_t j = 0; j < value_size; ++j)
          {
            u[p * ushape[1] + (j * bs_element + k)]
                += coefficients[bs_element * i + k] * basis_values(i, j);
          }
        }
      }
    }
  }
}
