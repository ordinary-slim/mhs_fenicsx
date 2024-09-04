From this directory, run:

```
# Build nanobind module
cmake -B build-dir -S ./cpp/
cmake --build build-dir
cmake --install build-dir
python3 -m pip -v install --config-settings=build-dir="build" --no-build-isolation ./python
```

TODO
====

- [ ] Stop storing ext_conductivity
- [ ] Add test for get_active_dofs_external!
- [ ] Unify extract_cell_geo of get_active_dofs_external and geometry_utils
- [x] Interpolate list of dg0 funcs simulatenously
- [x] Bug fix parallel substepping case with certain numbers of els. Most likely empty partitions.
