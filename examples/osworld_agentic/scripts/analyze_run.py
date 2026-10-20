#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Summarize logical Sessions from reward_dump.jsonl by rollout, not optimizer
step.

The dump's one-based ``step`` identifies a completed prompt Group. Its
``group_index`` is local to the custom advantage callback and can always be 0.
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def summarize_run(
    rows: list[dict[str, Any]],
    *,
    run_tag: str | None = None,
    groups_per_rollout: int = 12,
    samples_per_group: int = 8,
    max_turns: int = 30,
) -> list[dict[str, Any]]:
    """Return Session-weighted metrics, rejecting ambiguous mixed-run input."""
    if min(groups_per_rollout, samples_per_group, max_turns) <= 0:
        raise ValueError("group sizes and max_turns must be positive")
    tags = {row.get("run_tag", "") for row in rows}
    if run_tag is None:
        if len(tags) > 1:
            raise ValueError("multiple run tags found; select one with --run-tag")
        run_tag = next(iter(tags), "")
    selected = [row for row in rows if row.get("run_tag", "") == run_tag]
    if not selected:
        raise ValueError(f"no reward records for run tag {run_tag!r}")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        step = int(row["step"])
        if step < 1:
            raise ValueError("dump step must be a one-based Group counter")
        groups[step].append(row)
    rollouts: dict[int, list[int]] = defaultdict(list)
    for step in groups:
        rollouts[(step - 1) // groups_per_rollout].append(step)
    results = []
    previous_completion = None
    for rollout_id, steps in sorted(rollouts.items()):
        sessions = [row for step in sorted(steps) for row in groups[step]]
        expected = set(range(rollout_id * groups_per_rollout + 1, (rollout_id + 1) * groups_per_rollout + 1))
        complete = set(steps) == expected and all(len(groups[step]) == samples_per_group for step in steps)
        for step in steps:
            identities = [row["session"] for row in groups[step]]
            if len(set(identities)) != len(identities):
                raise ValueError(f"duplicate Session in dump Group {step}; possible concatenated/restarted run")
            if len({row["group_kept"] for row in groups[step]}) != 1:
                raise ValueError(f"inconsistent group_kept in dump Group {step}")
        turns = [row["turns"] for row in sessions]
        stds = [groups[step][0]["group_std"] for step in steps]
        completion = max(datetime.fromisoformat(row["ts"]) for row in sessions)
        explicit_caps = sum(row.get("stop_reason") == "max_turns" for row in sessions)
        approximate_caps = sum(row.get("stop_reason") != "max_turns" and row["turns"] >= max_turns for row in sessions)
        result = {
            "run_tag": run_tag,
            "rollout": rollout_id,
            "complete": complete,
            "groups": len(steps),
            "sessions": len(sessions),
            "successes": sum(row["outcome"] >= 0.5 for row in sessions),
            "success_rate": statistics.mean(row["outcome"] >= 0.5 for row in sessions),
            "turns": sum(turns),
            "turn_mean": statistics.mean(turns),
            "turn_median": statistics.median(turns),
            "records": sum(row["records"] for row in sessions),
            "extra_calls": sum(max(0, row["records"] - row["turns"]) for row in sessions),
            "filtered_groups": sum(not groups[step][0]["group_kept"] for step in steps),
            "std_mean": statistics.mean(stds),
            "shaped_mean": statistics.mean(row["shaped"] for row in sessions),
            "max_turns_explicit": explicit_caps,
            "budget_hits_approximate": approximate_caps,
            "completion_interval_min": (
                (completion - previous_completion).total_seconds() / 60 if previous_completion else None
            ),
        }
        results.append(result)
        previous_completion = completion
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("reward_dump.jsonl"))
    parser.add_argument("--run-tag", help="required when dump contains multiple runs")
    parser.add_argument("--groups-per-rollout", "--groups-per-step", type=int, default=12, dest="groups_per_rollout")
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--md", action="store_true", help="emit a markdown table instead of JSON")
    args = parser.parse_args()
    try:
        with args.dump.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        results = summarize_run(
            rows,
            run_tag=args.run_tag,
            groups_per_rollout=args.groups_per_rollout,
            samples_per_group=args.samples_per_group,
            max_turns=args.max_turns,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if not args.md:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    print(f"Run: {results[0]['run_tag']}")
    print(
        "\n| rollout | 状态 | groups | Sessions | 成功率 | 环境步数 | 导出记录 | 额外调用 | 过滤组 | 明确耗尽/近似触顶 |"
    )
    print("|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        state = "完整" if row["complete"] else "不完整"
        print(
            f"| {row['rollout']} | {state} | {row['groups']} | {row['sessions']} | "
            f"{row['successes']}/{row['sessions']} ({row['success_rate']:.2%}) | "
            f"{row['turns']} | {row['records']} | {row['extra_calls']} | "
            f"{row['filtered_groups']} | {row['max_turns_explicit']}/{row['budget_hits_approximate']} |"
        )
    print("\nrollout 不是 optimizer step。额外调用为 records−turns，不能仅据此判定重试原因。")
    print("旧 dump 未区分 max_turns；turns≥预算仅计为近似触顶，也可能恰好在最后一步主动终止。")
    print("过滤组按 group_kept 统计；其 policy advantage 为零，但 KL 项仍可能有梯度。")
    print("completion_interval_min 为相邻轮末条 Group dump 的时间间隔，并非单独训练耗时。")


if __name__ == "__main__":
    main()
