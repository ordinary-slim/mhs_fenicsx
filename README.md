# mhs_fenicsx

`mhs_fenicsx` is a FEniCSx extension for thermal modeling of Moving Heat Source (MHS) problems,
with a particular focus on Laser Powder Bed Fusion (LPBF).

While support for higher level discretizations is reduced (default is P1 / Q1 elements, Backward-Euler),
this repository implements specialized functionality, including:

- **Multi-time-step methods** (*TODO: add preprint link here*)
- **Moving subdomain methods**, see [Slimani et al. 2024](https://www.sciencedirect.com/science/article/pii/S0168874X2400132X)
- **Monolithic and staggered multi-mesh domain decomposition**

Supported physics:
- Heat diffusion
- Latent heat effects (both enthalpy-based and apparent heat capacity treatments)
- Convective and radiative heat transfer
- Thermal dependency of all parameters, including absorptivity

Available heat source profiles:
- Gaussian
- Gusarov
- Lumped volumetric source

Additive Manufacturing (AM) simulations are performed by starting from a mesh
representing the final build and **restricting the computational domain** to a subset
of active elements at each time step. Degrees of freedom (DOF) deactivation is handled
through the [`multiphenicsx`](https://github.com/multiphenics/multiphenicsx) extension.

## Dependencies

- **dolfinx** (C++ and python modules)

- **multiphenicsx** (C++ and python modules)

**Note:** The C++ module of `multiphenicsx` is not installed when using the main branch.
You must install [this fork](https://github.com/ordinary-slim/multiphenicsx).

## Docker

*TODO: Add Docker instructions here*

## Build instructions

``` bash
# Nanobind (c++) module
cmake -B build-dir -S ./cpp/
cmake --build build-dir
cmake --install build-dir
# python module
python3 -m pip -v install -e python --config-settings=build-dir="build" --no-build-isolation
```
