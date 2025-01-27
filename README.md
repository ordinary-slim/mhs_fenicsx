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
- [x] Chimera inside of substepper (difficult)
- [x] Coupling of Hodge
- [x] Try out no submesh

NOTES
=====

- If collision is expensive in `extract_subproblem`, remove narrow check


TODO
====

- [x] Move monolithic driver to outside of time-stepping
- [x] Add test for get_active_dofs_external! (low priority)
- [x] (VERY IMPORTANT) Update SUPG with phase change
- [ ] Stop storing ext_conductivity (low priority)
- [ ] Unify extract_cell_geo of get_active_dofs_external and geometry_utils (low priority)
- [ ] h for Robin's gamma not computed explicitly
- [ ] (IMPORTANT) Substepper + Dirichlet conditions
- [ ] In substepper, update physical domain restriction where relevant
- [ ] Material rework to subdomains
- [ ] Store material properties as functions of Temperature
- [ ] Move Chimera problem to initial position in pre_loop of substepper
- [ ] Mesh independent forms in monolithic robin driver
- [ ] Set explicit linear solve in SNES where it matters
- [ ] On delete of substepper, set activation to physical (high priority)
