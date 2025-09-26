import numpy as np
import scipy.sparse as sp
import glob
import re
from collections import defaultdict

def extract_nr(fname):
    match = re.search(r"nr(\d+)", fname)
    return int(match.group(1)) if match else None

def extract_ls(fname):
    match = re.search(r"ls(\d+)", fname)
    return int(match.group(1)) if match else None

def group_residuals():
    files = glob.glob("R_*.npy")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))  # grouped[nr][ls][field] = file
    for f in files:
        match = re.match(r"R_([fm])_nr(\d+)_ls(\d+).npy", f)
        if match:
            field, nr, ls = match.groups()
            grouped[int(nr)][int(ls)][field] = f
    return dict(sorted(grouped.items()))  # sort by nr

def group_jacobians():
    files = glob.glob("J_*.npz")
    grouped = defaultdict(dict)  # grouped[nr][block] = file
    for f in files:
        match = re.match(r"J_([fm])_([fm])_nr(\d+).npz", f)
        if match:
            row, col, nr = match.groups()
            grouped[int(nr)][f"{row}_{col}"] = f
    return dict(sorted(grouped.items()))

def print_residual_diffs(grouped_res):
    print("=== 🔁 Residual Norm Diffs (Grouped by NR + LS) ===")
    for nr, ls_blocks in grouped_res.items():
        print(f"\n--- Newton Iteration nr{nr} ---")
        prev_vecs = {}
        for ls in sorted(ls_blocks.keys()):
            print(f"  Line Search Iteration ls{ls}:")
            fields = ls_blocks[ls]
            for field in ["f", "m"]:
                fpath = fields.get(field)
                if fpath:
                    vec = np.load(fpath)
                    current_norm = np.linalg.norm(vec)
                    string = f"||R_{field}|| = {current_norm:g}"
                    if field in prev_vecs:
                        current_norm = np.linalg.norm(vec)
                        delta = np.linalg.norm(vec - prev_vecs[field])
                        string += f", ||Delta R_{field}|| = {delta:g}"
                    print(string)
                    prev_vecs[field] = vec

def print_jacobian_diffs(grouped_jacs):
    print("\n=== 🧮 Jacobian Frobenius Norm Diffs (per NR) ===")
    prev = {}
    for nr, blocks in grouped_jacs.items():
        print(f"\n--- Newton Iteration nr{nr} ---")
        for block in ["f_f", "f_m", "m_f", "m_m"]:
            curr_file = blocks.get(block)
            if not curr_file:
                continue
            curr = sp.load_npz(curr_file)
            if block in prev:
                diff = curr - prev[block]
                norm = sp.linalg.norm(diff)
                print(f"  ΔJ_{block} = {norm:.9e}")
            prev[block] = curr

if __name__ == "__main__":
    residuals = group_residuals()
    jacobians = group_jacobians()
    
    print_residual_diffs(residuals)
    print_jacobian_diffs(jacobians)

