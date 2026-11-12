#!/usr/bin/env python3
"""Compare two training runs' ClearML scalar series (T0 direct vs T1 dcs).

For each watched metric, align the two runs by step and report max/mean abs
diff. Loss/KL/reward trajectories must track closely: any engine-weight
corruption shows up as immediate divergence because generation is on-policy.

Usage: python compare_runs_clearml.py TASK_ID_A TASK_ID_B [--json out.json]
"""

import argparse
import json

from clearml import Task


WATCH = [
    "train/loss",
    "train/pg_loss",
    "train/kl_loss",
    "train/grad_norm",
    "train/entropy_loss",
    "train/train_rollout_logprob_abs_diff",
    "rollout/rewards",
    "rollout/raw_reward",
    "rollout/log_probs",
    "rollout/ref_log_probs",
    "rollout/rollout_log_probs",
    "rollout/response_len/mean",
    "rollout/acc/mean",
]


def series(task_id):
    t = Task.get_task(task_id=task_id)
    out = {}
    for fam, variants in t.get_reported_scalars().items():
        for variant, s in variants.items():
            out[f"{fam}/{variant}"] = (s["x"], s["y"])
    return t.name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_a")
    ap.add_argument("task_b")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    name_a, sa = series(args.task_a)
    name_b, sb = series(args.task_b)
    print(f"A: {name_a}\nB: {name_b}")

    report = {}
    print(f"{'metric':45s} {'n':>4s} {'max|Δ|':>12s} {'mean|Δ|':>12s} {'rel_max':>10s}")
    for m in WATCH:
        if m not in sa or m not in sb:
            print(f"{m:45s} MISSING in {'A' if m not in sa else 'B'}")
            continue
        xa, ya = sa[m]
        xb, yb = sb[m]
        ya_by_x = dict(zip(xa, ya))
        yb_by_x = dict(zip(xb, yb))
        common = sorted(set(ya_by_x) & set(yb_by_x))
        if not common:
            print(f"{m:45s} no common steps")
            continue
        diffs = [abs(ya_by_x[x] - yb_by_x[x]) for x in common]
        scale = max(1e-12, max(abs(ya_by_x[x]) for x in common))
        max_d, mean_d = max(diffs), sum(diffs) / len(diffs)
        report[m] = {"n": len(common), "max_abs_diff": max_d, "mean_abs_diff": mean_d, "rel_max": max_d / scale}
        print(f"{m:45s} {len(common):4d} {max_d:12.6g} {mean_d:12.6g} {max_d / scale:10.3g}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"a": args.task_a, "b": args.task_b, "report": report}, f, indent=2)
        print(f"written: {args.json_out}")


if __name__ == "__main__":
    main()
