From this directory, run:

```
# Build nanobind module
cmake -B build-dir -S ./cpp/
cmake --build build-dir
cmake --install build-dir
python3 -m pip -v install --config-settings=build-dir="build" --no-build-isolation ./python
```

CHECKPOINTS
===========

- [x] Square track case
- [ ] 3D
- [ ] 2D AM
- [ ] 3D AM
- [ ] Lumped heat source predictor (high priority)
- [ ] 2 staggered iterations at most
- [ ] Phase change
- [ ] Temperature dependent parameters
- [ ] 5 tests
- [ ] 10 tests
- [ ] Chimera inside of substepper (difficult)
- [ ] Coupling of Hodge
- [ ] Try out no submesh

NOTES
=====

- If collision is expensive in `extract_subproblem`, remove narrow check


TODO
====

- [ ] Divide activation in two? expensive and cheap parts
- [ ] Stop storing ext_conductivity (low priority)
- [ ] Add test for get_active_dofs_external! (low priority)
- [ ] Unify extract_cell_geo of get_active_dofs_external and geometry_utils (low priority)
- [x] Interpolate list of dg0 funcs simulatenously
- [x] Bug fix parallel substepping case with certain numbers of els. Most likely empty partitions.
- [ ] On delete of substepper, set activation to physical (high priority)
- [ ] h for Robin's gamma not computed explicitly
