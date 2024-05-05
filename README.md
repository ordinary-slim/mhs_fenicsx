From this directory, run:
    # Build nanobind module
    cmake -B build-dir -S ./cpp/
    cmake --build build-dir
    cmake --install build-dir
    python3 -m pip -v install --config-settings=build-dir="build" --no-build-isolation ./python
