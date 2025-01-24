From this directory, run:

```
# Build nanobind module
cmake -B build-dir -S ./cpp/
cmake --build build-dir
cmake --install build-dir
python3 -m pip -v install -e python --config-settings=build-dir="build" --no-build-isolation
```

CHECKPOINTS
===========

- [x] Monolithic Chimera
- [x] Square track case
- [ ] 3D
- [ ] 2D AM
- [ ] 3D AM
- [ ] Lumped heat source PREDICTOR (high priority)
- [ ] 2 staggered iterations at most
- [x] Phase change
- [ ] Temperature dependent parameters
- [x] 5 tests
- [x] 10 tests
- [ ] Chimera inside of substepper (difficult)
- [x] Coupling of Hodge
- [x] Try out no submesh

NOTES
=====

- If collision is expensive in `extract_subproblem`, remove narrow check


TODO
====

- [x] Divide activation in two? expensive and cheap parts
- [ ] Stop storing ext_conductivity (low priority)
- [ ] Add test for get_active_dofs_external! (low priority)
- [ ] Unify extract_cell_geo of get_active_dofs_external and geometry_utils (low priority)
- [x] Interpolate list of dg0 funcs simulatenously
- [x] Bug fix parallel substepping case with certain numbers of els. Most likely empty partitions.
- [ ] h for Robin's gamma not computed explicitly
- [x] Profile 5on5 test
- [x] Add shifting of bbox tree at c++ level
- [x] Checkout `compute_integration_domain`
- [ ] (VERY IMPORTANT) Update SUPG with phase change
- [ ] (IMPORTANT) Substepper + Dirichlet conditions
- [ ] On delete of substepper, set activation to physical (high priority)
- [ ] In substepper, update physical domain restriction where relevant
- [x] (FIRST PRIORITY) Fix bug SMS
- [x] Compare SNES to custom NR solver
- [x] Assemble manually mass matrix
- [x] Assemble manually stiffness matrix
- [x] Assemble manually mass boundary
- [x] Assemble manually flux boundary
- [x] Same mesh blocked Poisson problem
- [x] Different mesh blocked Poisson problem
- [x] Interface data as dictionnary
- [x] Remove submesh support in favor of same mesh approach
- [ ] Material rework to subdomains
- [ ] Store material properties as functions of Temperature
- [ ] Move Chimera problem to initial position in pre_loop of substepper
- [ ] Mesh independent forms in monolithic robin driver
- [ ] Move monolithic driver to outside of time-stepping
