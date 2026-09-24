"""Compare two results.jsonl runs event by event.

Aligns events on (event, task, member) and reports mean deltas. Used for
same-machine rerun validation and cross-machine comparison: scores should
agree to BLAS noise (third decimal); larger deltas merit investigation
against the per-fold arrays.

Usage:
    python scripts/compare_runs.py results/results.jsonl results/validation/results.jsonl
    python scripts/compare_runs.py results/results.jsonl results/results.jsonl --tol 0.002
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(path):
    events = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if "mean" not in e:
                continue
            key = (e.get("event"), e.get("task"), e.get("member"))
            events[key] = e["mean"]  # last occurrence wins
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--tol", type=float, default=0.002,
                    help="flag deltas beyond this (default 0.002)")
    args = ap.parse_args()

    ref = load(args.reference)
    cand = load(args.candidate)
    keys = sorted(set(ref) & set(cand), key=str)
    only_ref = sorted(set(ref) - set(cand), key=str)
    only_cand = sorted(set(cand) - set(ref), key=str)

    print(f"reference:  {args.reference} ({len(ref)} scored events)")
    print(f"candidate:  {args.candidate} ({len(cand)} scored events)")
    print(f"matched:    {len(keys)}")
    if only_ref:
        print(f"only in reference:  {len(only_ref)} (e.g. {only_ref[:3]})")
    if only_cand:
        print(f"only in candidate:  {len(only_cand)} (e.g. {only_cand[:3]})")
    print()

    flagged = 0
    max_abs = 0.0
    for key in keys:
        delta = cand[key] - ref[key]
        max_abs = max(max_abs, abs(delta))
        flag = " <<<" if abs(delta) > args.tol else ""
        if flag:
            flagged += 1
        print(f"{str(key):55s} ref={ref[key]:.4f} cand={cand[key]:.4f} "
              f"delta={delta:+.4f}{flag}")
    print()
    print(f"max |delta|: {max_abs:.4f}; flagged beyond {args.tol}: {flagged}")


if __name__ == "__main__":
    main()
