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
- [ ] Bug fix parallel substepping case with certain numbers of els. Most likely empty partitions.
- [ ] Interpolate dg0 product directly
