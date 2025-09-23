# Tests

Each test folder contains a `test_*.py` file
which in turn contains test routines.

## Running tests

Tests are meant to be ran with `pytest`.
** Note :** Collecting more than ~10 tests at once
results in a crash due to `numba` cache bug.

### Option 1 -- Run a subset

Run `pytest` inside a specific test folder:

```bash
cd 2d_substepping/
python3 -m pytest .
```

### Option 2 -- Run a subset

Use the helper script to run tests one by one:

```bash
python3 run_alltests_1by1.py --help
python3 run_alltests_1by1.py -n 3 # 3 processors
```

This avoids the cache bug by running tests sequentially.
