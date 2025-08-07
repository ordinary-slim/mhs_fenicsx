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
- [x] 3D
- [x] 2D AM
- [x] 3D AM
- [x] Lumped heat source PREDICTOR (high priority)
- [ ] 2 staggered iterations at most
- [x] Phase change
- [x] Melting of powder into bulk
- [x] Temperature dependent parameters
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
- [x] On delete of substepper, set activation to physical (high priority)
- [x] Stop storing ext_conductivity (low priority)
- [ ] Unify extract_cell_geo of get_active_dofs_external and geometry_utils (low priority)
- [ ] h for Robin's gamma not computed explicitly
- [ ] (IMPORTANT) Substepper + Dirichlet conditions
- [ ] In substepper, update physical domain restriction where relevant
- [x] Material rework to subdomains
- [x] Store material properties as functions of Temperature
- [ ] Move Chimera problem to initial position in pre_loop of substepper
- [ ] Mesh independent forms in monolithic robin driver
- [x] Set explicit linear solve in SNES where it matters
- [ ] In micro_pre_iterate, abort if end of path and set t1_macro and ps.dt to diff values
- [ ] Set form subdomain data of staggered drivers together with one of Problem! Hook or something
- [ ] Set back pad of micro iters to Chimera back
