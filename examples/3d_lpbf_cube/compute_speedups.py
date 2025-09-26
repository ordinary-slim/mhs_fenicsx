import argparse
import os
import re

def main(folder):
    # Profiling files end with .txt
    profiling_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    ref_profiling_file = None
    for idx, f in enumerate(profiling_files):
        if 'ref' in f:
            ref_profiling_file = profiling_files.pop(idx)
            break
    if ref_profiling_file is None:
        raise ValueError("No reference profiling file found in the folder.")
    # Profiling files are produced with python line_profiler
    with open(os.path.join(folder, ref_profiling_file), 'r') as f:
        lines = f.readlines()
        # Reference time is time of `do_printing_timestep`
        ref_time = None
        for idx, line in enumerate(lines):
            if re.match(r'^Function: do_printing_timestep', line):
                # Time is two lines above
                time_line = lines[idx - 2]
                time_string = re.search(r'(\d+(\.\d+)?)', time_line).group(1)
                ref_time = float(time_string)
                break
        if ref_time is None:
            raise ValueError("Could not find timing information for `do_printing_timestep` in the reference file.")
    print(f"Reference time (ref): {ref_time:.4f} seconds")
    # Get times from other profiling files
    # The name of the model is everything in between `profiling_` and `.txt`
    for f in profiling_files:
        model_name = re.search(r'profiling_(.*)\.txt$', f).group(1)
        with open(os.path.join(folder, f), 'r') as pf:
            plines = pf.readlines()
            model_time = None
            for idx, line in enumerate(plines):
                if re.match(r'^Function: do_substepped_timestep', line):
                    time_line = plines[idx - 2]
                    time_string = re.search(r'(\d+(\.\d+)?)', time_line).group(1)
                    model_time = float(time_string)
                    if model_time > 0:
                        break
            if model_time is None:
                raise ValueError(f"Could not find timing information for `do_printing_timestep` in the file {f}.")
        speedup = ref_time / model_time
        print(f"Model: {model_name}, Time: {model_time:.4f} seconds, Speedup: {speedup:.2f}x")

    # There is also a file that end either with `out` or `print`
    # that contains some logging information
    log_files = [f for f in os.listdir(folder) if f.endswith('.out') or f.endswith('.print')]
    assert len(log_files) == 1, "There should be 1 log file."
    log_file = log_files[0]
    # Let's print all the lines that start with "Logger"
    # Let's also keep track of the max adimensional time-step
    max_timestep = 0.0
    with open(os.path.join(folder, log_file), 'r') as lf:
        llines = lf.readlines()
        print("\nLogging information:")
        for line in llines:
            if line.startswith("Logger"):
                print(line.strip())
            if line.startswith("is Chimera ON? True"):
                timestep = re.search(r'adim dt = (\d+(\.\d+)?)', line).group(1)
                timestep = float(timestep)
                max_timestep = max(max_timestep, timestep)
    print(f"\nMax adimensional time-step: {max_timestep:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Path to the folder containing the profiling files.")
    args = parser.parse_args()
    main(args.folder)
