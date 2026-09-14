#!/usr/bin/env python3
"""Per-step readout for an OSWorld agentic run.

Reads reward_dump.jsonl (written by app/reward_post_process.py when
OSWORLD_REWARD_DUMP is set). Everything is derived from that one file:

  step      = group_index // groups_per_step   (group_index is a global
              monotonic counter, so this survives across the whole run)
  accuracy  = sessions with outcome >= 0.5, over sessions in the step
  group std = the per-group reward std the GRPO baseline divides by;
              std == 0 means every attempt in that group scored the same,
              so the group contributed no gradient
  duration  = gap between consecutive steps' dump timestamps, i.e. one full
              rollout + train cycle

Usage:
  python3 scripts/analyze_run.py --dump ../../reward_dump.jsonl --md
"""

import argparse
import json
import statistics as st
from datetime import datetime


p = argparse.ArgumentParser()
p.add_argument("--dump", default="reward_dump.jsonl")
p.add_argument("--groups-per-step", type=int, default=4, help="= rollout_batch_size")
p.add_argument("--md", action="store_true", help="emit a markdown table")
a = p.parse_args()

rows = [json.loads(line) for line in open(a.dump, encoding="utf-8") if line.strip()]
if not rows:
    raise SystemExit("empty dump")

steps: dict[int, list] = {}
for r in rows:
    steps.setdefault(r["group_index"] // a.groups_per_step, []).append(r)

recs, prev = [], None
for s in sorted(steps):
    rs = steps[s]
    stds = [v for _, v in sorted({(r["group_index"], r["group_std"]) for r in rs})]
    turns = sorted(r["turns"] for r in rs)
    succ = sum(1 for r in rs if r["outcome"] >= 0.5)
    ts = datetime.fromisoformat(rs[0]["ts"])
    recs.append(
        {
            "step": s,
            "n": len(rs),
            "succ": succ,
            "acc": succ / len(rs),
            "std_mean": st.mean(stds),
            "std_min": min(stds),
            "std_max": max(stds),
            "dead": sum(1 for v in stds if v == 0),
            "groups": len(stds),
            "turn_med": st.median(turns),
            "cap": sum(1 for t in turns if t >= 50),
            "shaped": st.mean(r["shaped"] for r in rs),
            "min": (ts - prev).total_seconds() / 60 if prev else None,
            "ts": rs[0]["ts"][11:19],
        }
    )
    prev = ts

if a.md:
    print("| step | 通过率 | shaped 均值 | group std (均值/最小/最大) | 无梯度组 | 轮数中位 | 撞顶 | 耗时(min) |")
    print("|---:|---:|---:|:---|---:|---:|---:|---:|")
    for r in recs:
        d = f"{r['min']:.1f}" if r["min"] else "—"
        print(
            f"| {r['step']} | {r['succ']}/{r['n']} ({r['acc'] * 100:.0f}%) | {r['shaped']:+.3f} | "
            f"{r['std_mean']:.3f} / {r['std_min']:.3f} / {r['std_max']:.3f} | "
            f"{r['dead']}/{r['groups']} | {r['turn_med']:.0f} | {r['cap']}/{r['n']} | {d} |"
        )
else:
    hdr = f"{'step':>4} {'acc':>9} {'shaped':>8} {'std_mean':>9} {'dead':>7} {'turn_med':>8} {'cap':>7} {'min':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        d = f"{r['min']:6.1f}" if r["min"] else "     —"
        print(
            f"{r['step']:>4} {r['succ']:>3}/{r['n']:<2}{r['acc'] * 100:>4.0f}% {r['shaped']:>+8.3f} "
            f"{r['std_mean']:>9.3f} {r['dead']:>3}/{r['groups']:<3} {r['turn_med']:>8.0f} "
            f"{r['cap']:>3}/{r['n']:<3} {d}"
        )

acc = [r["acc"] for r in recs]
mins = [r["min"] for r in recs if r["min"]]
print(
    f"\n步数 {len(recs)} | 总通过率 {sum(r['succ'] for r in recs)}/{sum(r['n'] for r in recs)} "
    f"= {sum(r['succ'] for r in recs) / sum(r['n'] for r in recs) * 100:.1f}%"
)
print(
    f"通过率 前1/3 {st.mean(acc[: len(acc) // 3 or 1]) * 100:.1f}%  后1/3 {st.mean(acc[-(len(acc) // 3 or 1) :]) * 100:.1f}%"
)
print(
    f"无梯度组 {sum(r['dead'] for r in recs)}/{sum(r['groups'] for r in recs)} "
    f"= {sum(r['dead'] for r in recs) / sum(r['groups'] for r in recs) * 100:.0f}%"
)
if mins:
    print(
        f"每步耗时 中位 {st.median(mins):.1f} min | 最快 {min(mins):.1f} | 最慢 {max(mins):.1f} "
        f"| 剩余 150 步预计 {st.median(mins) * (150 - len(recs)) / 60:.1f} h"
    )
