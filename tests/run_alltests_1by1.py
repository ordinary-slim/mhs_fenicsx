import pytest
import io
import sys
import os
import argparse
import subprocess

def main(nprocs):
    ''' Run tests 1 by 1 with MPI.
    If ran all together, crashes because of numba cache issue.'''
    cwd = os.getcwd().split(r"/")[-1]
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        pytest.main(["--collect-only", "-q"])  # Collect test names
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout  # Restore stdout
    tests = [line.strip().replace(cwd, '.') for line in output.split("\n") if line.strip() and "::" in line]
    for test in tests:
        out = subprocess.run(["mpirun", "-n", f"{nprocs}", "pytest", test, '-q', '-s', '--verbose'])
        if out.returncode:
            raise Exception("A test FAILED")
    print(f"All {len(tests)} tests PASSED!")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--nprocs', default=1)
    args = parser.parse_args()
    main(nprocs=args.nprocs)
