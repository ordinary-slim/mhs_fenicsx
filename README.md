# mhs_fenicsx

`mhs_fenicsx` is a FEniCSx extension for thermal modeling of Moving Heat Source (MHS) problems,
with a particular focus on Laser Powder Bed Fusion (LPBF).

While support for higher level discretizations is reduced (default is P1 / Q1 elements, Backward-Euler),
this repository implements specialized functionality, including:

- **Multi-time-step methods** see [Slimani et al. 2025 (freely accessible preprint)](http://dx.doi.org/10.2139/ssrn.5529518)
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

Detailed build instructions are reproducible from the Dockerfile.

## Docker

A pre-built Docker image for linux/amd64 is available.
The recommended workflow is to clone this repository locally and launch a
container inside it:

``` bash
docker pull ordinaryslim/mhs_fenicsx:latest
git clone https://github.com/ordinary-slim/mhs_fenicsx
docker run --rm --network=host --shm-size=512m -ti -v $(realpath mhs_fenicsx):/root/shared/ -w /root/shared --entrypoint /bin/bash ordinaryslim/mhs_fenicsx:latest
```

Once inside the container, you can run the provided examples and tests immediately.
Result files will appear directly in your cloned repository folder on the host.

## Reproducibility

The numerical examples from
[Slimani et al. (2025) (*freely accessible preprint*)](http://dx.doi.org/10.2139/ssrn.5529518)
correspond to the following folders:

1. _2D square track_ → `tests/2d_substepping/`
2. _AMB2018-02_ → `examples/amb2018_02/`
3. _3D LPBF_ → `examples/3d_lpbf_cube/`

Each folder includes a dedicated `README` file with instructions
for running the simulation.

## Postprocessing

Simulations results are written to `vtx` or `vtk` files,
which can be opened in Paraview.
For `vtx` files, Paraview ≥ 5.13 is recommended to correctly visualize
element-wise fields (e.g. `active_elements`, `material_id`).

Postprocessing options can be controlled from the `input.yaml` file or directly in the Python
script. Available options include:

- Write output at each substep in `vtk` format. Enable via:
``` yaml
substepping_parameters:
  writepos: True
```
- Write additional solution fields. In the Python script, pass extra functions to writepos:
``` python3
p.writepos(extra_funcs=[additional_functions])
```

## Build instructions

``` bash
# Nanobind (c++) module
cmake -B build-dir -S ./cpp/
cmake --build build-dir
cmake --install build-dir
# python module
python3 -m pip -v install -e python --config-settings=build-dir="build" --no-build-isolation
```

## Citations

If you found this library useful in academic or industry work, we appreciate your support if you consider:

1. Starring the project on Github
2. Citing the relevant paper(s):

[Substepped and advected subdomain methods for part-scale LPBF modeling (*freely accessible preprint*)](http://dx.doi.org/10.2139/ssrn.5529518)

[A Chimera method for thermal part-scale metal additive manufacturing simulation](https://doi.org/10.1016/j.finel.2024.104238)

``` bibtex
@article{slimani2024,
title = {A {C}himera method for thermal part-scale metal additive manufacturing simulation},
journal = {Finite Elements in Analysis and Design},
volume = {241},
pages = {104238},
year = {2024},
issn = {0168-874X},
doi = {https://doi.org/10.1016/j.finel.2024.104238},
url = {https://www.sciencedirect.com/science/article/pii/S0168874X2400132X},
author = {Mehdi Slimani and Miguel Cervera and Michele Chiumenti}
}
```
