#include "CustomSparsityPattern.h"

// Copy of original create_matrix to pass my own SparsityPattern
void custom_create_matrix(Mat& A, MPI_Comm comm,
    const CustomSparsityPattern& sp, std::string type)
{
  PetscErrorCode ierr;
  ierr = MatCreate(comm, &A);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatCreate");

  // Get IndexMaps from sparsity patterm, and block size
  std::array maps = {sp.index_map(0), sp.index_map(1)};
  const std::array bs = {sp.block_size(0), sp.block_size(1)};

  if (!type.empty())
    MatSetType(A, type.c_str());

  // Get global and local dimensions
  const std::int64_t M = bs[0] * maps[0]->size_global();
  const std::int64_t N = bs[1] * maps[1]->size_global();
  const std::int32_t m = bs[0] * maps[0]->size_local();
  const std::int32_t n = bs[1] * maps[1]->size_local();

  // Set matrix size
  ierr = MatSetSizes(A, m, n, M, N);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatSetSizes");

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(A);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatSetFromOptions");

  // Find a common block size across rows/columns
  const int _bs = (bs[0] == bs[1] ? bs[0] : 1);

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _nnz_diag, _nnz_offdiag;
  if (bs[0] == bs[1])
  {
    _nnz_diag.resize(maps[0]->size_local());
    _nnz_offdiag.resize(maps[0]->size_local());
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = sp.nnz_diag(i);
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = sp.nnz_off_diag(i);
  }
  else
  {
    // Expand for block size 1
    _nnz_diag.resize(maps[0]->size_local() * bs[0]);
    _nnz_offdiag.resize(maps[0]->size_local() * bs[0]);
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = bs[1] * sp.nnz_diag(i / bs[0]);
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = bs[1] * sp.nnz_off_diag(i / bs[0]);
  }

  // Allocate space for matrix
  ierr = MatXAIJSetPreallocation(A, _bs, _nnz_diag.data(), _nnz_offdiag.data(),
                                 nullptr, nullptr);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatXIJSetPreallocation");

  // Set block sizes
  ierr = MatSetBlockSizes(A, bs[0], bs[1]);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatSetBlockSizes");

  // Create PETSc local-to-global map/index sets
  ISLocalToGlobalMapping local_to_global0;
  const std::vector map0 = maps[0]->global_indices();
  const std::vector<PetscInt> _map0(map0.begin(), map0.end());
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[0], _map0.size(),
                                      _map0.data(), PETSC_COPY_VALUES,
                                      &local_to_global0);

  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Check for common index maps
  if (maps[0] == maps[1] and bs[0] == bs[1])
  {
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global0);
    if (ierr != 0)
      dolfinx::la::petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
  }
  else
  {
    ISLocalToGlobalMapping local_to_global1;
    const std::vector map1 = maps[1]->global_indices();
    const std::vector<PetscInt> _map1(map1.begin(), map1.end());
    ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[1], _map1.size(),
                                        _map1.data(), PETSC_COPY_VALUES,
                                        &local_to_global1);
    if (ierr != 0)
      dolfinx::la::petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global1);
    if (ierr != 0)
      dolfinx::la::petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
    ierr = ISLocalToGlobalMappingDestroy(&local_to_global1);
    if (ierr != 0)
      dolfinx::la::petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  }

  // Clean up local-to-global 0
  ierr = ISLocalToGlobalMappingDestroy(&local_to_global0);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if A is not of type MATIS)
  // ierr = MatISSetPreallocation(A, 0, _nnz_diag.data(), 0,
  // _nnz_offdiag.data()); if (ierr != 0)
  //   error(ierr, __FILE__, "MatISSetPreallocation");

  // Set some options on Mat object
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatSetOption");
  ierr = MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, "MatSetOption");
}
