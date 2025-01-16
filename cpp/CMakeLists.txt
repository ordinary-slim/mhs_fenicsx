cmake_minimum_required(VERSION 3.15...3.27)

project(mhs_fenicsx_cpp)
include(GNUInstallDirs)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build compiler_commands.json for LSPs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_custom_target(
  copy-compile-commands ALL
  ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR}
)

# Find python
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Check for DOLFINx C++ backend
find_package(DOLFINX REQUIRED CONFIG)
if (DOLFINX_FOUND)
  message(STATUS "Found DOLFINx C++ backend at ${DOLFINX_DIR}")
endif()

# Check for DOLFINx python wrappers
execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import os, sys, dolfinx; print(os.path.join(os.path.dirname(dolfinx.__file__), 'wrappers'))"
  OUTPUT_VARIABLE DOLFINX_PY_WRAPPERS_DIR
  RESULT_VARIABLE DOLFINX_PY_WRAPPERS_COMMAND_RESULT
  ERROR_VARIABLE DOLFINX_PY_WRAPPERS_COMMAND_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT DOLFINX_PY_WRAPPERS_COMMAND_RESULT)
  message(STATUS "Found DOLFINx python wrappers at ${DOLFINX_PY_WRAPPERS_DIR}")
else()
  message(FATAL_ERROR "DOLFINx python wrappers could not be found.")
endif()

# Check for multiphenicsx headers
find_package(MULTIPHENICSX REQUIRED CONFIG)
if (MULTIPHENICSX_FOUND)
  message(STATUS "Found multiphenicsx C++ backend at ${MULTIPHENICSX_DIR}")
endif()

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_INCLUDE_COMMAND_RESULT
  ERROR_VARIABLE PETSC4PY_INCLUDE_COMMAND_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_INCLUDE_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "petsc4py could not be found.")
endif()

nanobind_add_module(mhs_fenicsx_cpp
                    NOMINSIZE
                    wrapper.cpp
                    wrapper_get_facet_integration_entities.cpp
                    interpolate.cpp
                    activation_utils.cpp
                    geometry_utils.cpp
                    submesh_utils.cpp
                    diffmesh_utils.cpp
                    wrapper_create_robin_robin_monolithic.cpp
                    custom_create_matrix.cpp
                    )

# Check for mpi4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  RESULT_VARIABLE MPI4PY_INCLUDE_COMMAND_RESULT
  ERROR_VARIABLE MPI4PY_INCLUDE_COMMAND_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MPI4PY_INCLUDE_COMMAND_RESULT)
  message(STATUS "Found mpi4py include directory at ${MPI4PY_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "mpi4py could not be found.")
endif()

#target_sources(mhs_fenicsx_cpp PRIVATE locate_active_boundary.cpp)
# Add DOLFINx C++ libraries
target_link_libraries(mhs_fenicsx_cpp PRIVATE dolfinx)

# # Add DOLFINx python, petsc4py and mpi4py include directories (DOLFINx C++
# ones already being added by target_link_libraries)
target_include_directories(mhs_fenicsx_cpp PRIVATE ${DOLFINX_PY_WRAPPERS_DIR})
target_include_directories(mhs_fenicsx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
target_include_directories(mhs_fenicsx_cpp PRIVATE ${MPI4PY_INCLUDE_DIR})

# Define HAS_PETSC4PY for compatibility with DOLFINx python wrappers
target_compile_definitions(mhs_fenicsx_cpp PRIVATE HAS_PETSC4PY)

# Install the compiled library to the cpp subdirectory
set_target_properties(
  mhs_fenicsx_cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS mhs_fenicsx_cpp LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
