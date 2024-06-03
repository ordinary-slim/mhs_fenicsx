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

nanobind_add_module(mhs_fenicsx_cpp
                    NOMINSIZE
                    wrapper.cpp
                    my_determine_point_ownership.cpp
                    interpolate_dg0_at_facets.cpp
                    )
#target_sources(mhs_fenicsx_cpp PRIVATE locate_active_boundary.cpp)
# Add DOLFINx C++ libraries
target_link_libraries(mhs_fenicsx_cpp PRIVATE dolfinx)

# Install the compiled library to the cpp subdirectory
set_target_properties(
  mhs_fenicsx_cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS mhs_fenicsx_cpp LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
