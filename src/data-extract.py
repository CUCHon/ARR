import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

MODES = [
    "quality",
    "complexity",
    "fsifd",
    "avg_comp_fsifd",
    "avg_qual_fsifd",
    "avg_comp_fsifd_plus_comp",
    "avg_qual_fsifd_plus_qual",
    "avg_comp_fsifd_mul_qual",
    "avg_qual_fsifd_mul_comp",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select sample subsets by score modes (single or all).")
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="JSON file of sample records (fields: quality_score, complexity_score, top5_similar_ids, few_shot_ifd, ifd)."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help=(
            "Output file or directory. "
            "If mode!='all', treated as JSON file path; "
            "if mode=='all', treated as output directory."
        )
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=MODES + ["all"],
        help="Selection mode or 'all' to run all modes (default: all)."
    )
    parser.add_argument(
        "--ratio", type=float, default=0.1,
        help="Fraction (0<ratio<=1) of top-ranked samples to select (default: 0.1)."
    )
    return parser.parse_args()


def compute_select_score(rec, data, mode):
    qs = rec.get("quality_score", 0.0)
    cs = rec.get("complexity_score", 0.0)
    sim_ids = rec.get("top5_similar_ids", [])
    fsifds = rec.get("few_shot_ifd", [])

    comp_fsifd_vals = []
    qual_fsifd_vals = []
    for idx, sid in enumerate(sim_ids):
        sim = data[sid]
        fs = fsifds[idx] if idx < len(fsifds) else 0.0
        comp_fsifd_vals.append(sim.get("complexity_score", 0.0) * fs)
        qual_fsifd_vals.append(sim.get("quality_score", 0.0) * fs)

    avg_comp_fsifd = float(np.mean(comp_fsifd_vals)) if comp_fsifd_vals else 0.0
    avg_qual_fsifd = float(np.mean(qual_fsifd_vals)) if qual_fsifd_vals else 0.0

    if mode == "quality":
        return qs
    if mode == "complexity":
        return cs
    if mode == "fsifd": #return average of fsifd
        return np.mean(fsifds) if fsifds else 0.0
        
    if mode == "avg_comp_fsifd":
        return avg_comp_fsifd
    if mode == "avg_qual_fsifd":
        return avg_qual_fsifd
    if mode == "avg_comp_fsifd_plus_comp":
        return avg_comp_fsifd + cs
    if mode == "avg_qual_fsifd_plus_qual":
        return avg_qual_fsifd + qs
    if mode == "avg_comp_fsifd_mul_qual":
        return avg_comp_fsifd * qs
    if mode == "avg_qual_fsifd_mul_comp":
        return avg_qual_fsifd * cs
    return 0.0


def process_mode(data, mode, ratio, output_path):
    print(f"\n=== Mode: {mode} ===")
    n = len(data)
    k = max(1, int(n * ratio))

    for rec in tqdm(data, desc=f"Scoring ({mode})", leave=False):
        rec["_select_score"] = compute_select_score(rec, data, mode)

    data_sorted = sorted(data, key=lambda x: x.get("_select_score", 0.0), reverse=True)

    selected = []
    for rec in data_sorted:
        if rec.get("ifd", 0.0) > 1.0:
            continue
        selected.append(rec)
        if len(selected) >= k:
            break

    if os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, f"selected_{mode}.json")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_file = output_path

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(selected)}/{n} items to {out_file}")


def main():
    args = parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {args.input_path}")

    if args.mode == "all":
        modes_to_run = MODES
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        data_copy = list(data)
        process_mode(data_copy, mode, args.ratio, args.output_path)

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
