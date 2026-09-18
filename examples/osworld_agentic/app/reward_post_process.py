"""Outcome-only RL reward + groupwise advantage, aligned with upstream.

Mirrors hybrid-routing-agent's `outcome_only_rl` recipe:
  configs/experiments/outcome_only_rl.yaml  (length_penalty_coef 0.05,
  max_steps_ref 50, cap_penalty 0.2, normalize_mode groupwise)
  reward/osworld_rl_reward.py:891-900       (the two penalty formulas)
  reward/osworld_rl_reward.py:580-624       (_groupwise_standardize_inplace)

Session exports already carry the evaluator's outcome. This batch hook groups
turn records by session before shaping and standardizing rewards; the generic
per-record reward path cannot represent those trajectory groups. It is called
per transfer batch, which can be smaller than a full rollout.

Reward (per trajectory, upstream's `raw_reward`):
    outcome    = +1 if the VM evaluator passed else -1
    length_pen = coef * T / max_steps_ref        -> at most `coef` (0.05)
    cap_pen    = cap  if T >= max_steps_ref      -> punishes "ran out of steps"
    raw        = outcome - length_pen - cap_pen

Both penalties target the never-terminate failure mode; together they cap at
0.25 against an outcome span of 2.0 -- tie-breakers, per upstream's comment,
not a competing objective.

Advantage (upstream _groupwise_standardize_inplace):
    per group (= one task's N attempts): z-score the trajectory returns, then
    broadcast to every turn record of that trajectory. OSWORLD_PER_TRAJ_NORM
    optionally divides by the record count; the outcome launcher disables it
    and sets max_steps_ref to 30 to match the session config.

Knobs (env, so ablations need no code edit):
    OSWORLD_LENGTH_PENALTY_COEF  default 0.05   (0 disables)
    OSWORLD_CAP_PENALTY          default 0.2    (0 disables)
    OSWORLD_MAX_STEPS_REF        default 50
    OSWORLD_PER_TRAJ_NORM        default 1      (0 disables the / T_i)
    OSWORLD_REWARD_DUMP          default unset  (path -> per-group JSONL dump)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


_DUMP_STEP = 0

_CENTRED = ("grpo", "gspo", "sapo", "cispo")


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _session_key(sample, position: int):
    """Identify the attempt a record belongs to; fall back to standing
    alone."""
    session_id = getattr(sample, "session_id", None)
    if session_id:
        return session_id
    metadata = getattr(sample, "metadata", None) or {}
    for key in ("session_id", "task_id"):
        if metadata.get(key):
            return metadata[key]
    return f"__position_{position}"


def _turns(sample, n_records: int) -> int:
    """Policy steps for this trajectory.

    One export record per agent turn, so the record count is the natural
    fallback when metadata is unavailable.
    """
    metadata = getattr(sample, "metadata", None) or {}
    for key in ("agent_turns", "rollout_turns", "num_turns"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return max(n_records, 1)


def post_process_rewards(args, samples):
    coef = _f("OSWORLD_LENGTH_PENALTY_COEF", 0.05)
    cap = _f("OSWORLD_CAP_PENALTY", 0.2)
    max_steps_ref = max(1, int(_f("OSWORLD_MAX_STEPS_REF", 50)))
    per_traj_norm = _f("OSWORLD_PER_TRAJ_NORM", 1.0) != 0.0
    # upstream paper Table 6: group-level accuracy filter (acc_lo, acc_hi) = (0.05, 0.95)
    acc_lo = _f("OSWORLD_ACC_LO", 0.05)
    acc_hi = _f("OSWORLD_ACC_HI", 0.95)
    # |A| clamp after group z-scoring (0 = off). A 7-1 group gives the singleton
    # |A|=2.65; with ~8 kept groups per update that one trajectory can dominate
    # the whole step. Clamp keeps every trajectory <= adv_clip. (2026-09-05)
    adv_clip = _f("OSWORLD_ADV_CLIP", 0.0)

    outcomes = [sample.get_reward_value(args) for sample in samples]

    # group -> session -> record positions
    groups: dict = {}
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("Sample.group_index is required for group reward normalization.")
        groups.setdefault(sample.group_index, {}).setdefault(_session_key(sample, position), []).append(position)

    shaped = [0.0] * len(samples)
    traj_turns: dict = {}
    for sessions in groups.values():
        for key, positions in sessions.items():
            turns = _turns(samples[positions[0]], len(positions))
            traj_turns[key] = turns
            raw = outcomes[positions[0]]
            outcome = 1.0 if float(raw if raw is not None else 0.0) >= 0.5 else -1.0
            value = outcome - coef * (turns / max_steps_ref)
            if turns >= max_steps_ref:
                value -= cap
            for position in positions:
                shaped[position] = value

    estimator = getattr(args, "advantage_estimator", None)
    if not getattr(args, "rewards_normalization", True) or estimator not in _CENTRED:
        return shaped, shaped

    advantages = [0.0] * len(samples)
    dump_rows = []
    for group_index, sessions in groups.items():
        keys = list(sessions)
        returns = [shaped[sessions[k][0]] for k in keys]
        mu = sum(returns) / len(returns)
        var = sum((r - mu) ** 2 for r in returns) / len(returns)
        std = var**0.5

        # drop groups that are all-pass or all-fail (upstream Table 6)
        _succ = [1.0 if float(outcomes[sessions[k][0]] or 0.0) >= 0.5 else 0.0 for k in keys]
        mean_succ = sum(_succ) / max(len(_succ), 1)
        group_kept = bool(acc_lo < mean_succ < acc_hi)

        for key, ret in zip(keys, returns):
            if not group_kept:
                adv = 0.0
            else:
                adv = (ret - mu) / std if std >= 1e-8 else (ret - mu)
            if adv_clip > 0:
                adv = max(-adv_clip, min(adv_clip, adv))
            t_i = max(len(sessions[key]), 1)
            if per_traj_norm:
                adv = adv / t_i
            for position in sessions[key]:
                advantages[position] = adv
            _md = getattr(samples[sessions[key][0]], "metadata", None) or {}
            dump_rows.append(
                {
                    "group_index": int(group_index),
                    "session": str(key),
                    "task_id": _md.get("task_id"),
                    "data_source": _md.get("data_source"),
                    "session_wallclock_s": _md.get("session_wallclock_s"),
                    "agent_turns": _md.get("agent_turns"),
                    "stop_reason": _md.get("stop_reason"),
                    "parse_failed_steps": _md.get("parse_failed_steps"),
                    "last_error": _md.get("last_error"),
                    "action_types": _md.get("action_types"),
                    "outcome": float(outcomes[sessions[key][0]] or 0.0),
                    "turns": traj_turns[key],
                    "records": t_i,
                    "shaped": round(ret, 6),
                    "advantage": round(adv, 6),
                    "group_std": round(std, 6),
                    "mean_succ": round(mean_succ, 4),
                    "group_kept": group_kept,
                }
            )

    dump_path = os.environ.get("OSWORLD_REWARD_DUMP")
    if dump_path and dump_rows:
        try:
            global _DUMP_STEP
            _DUMP_STEP += 1
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with open(dump_path, "a", encoding="utf-8") as handle:
                for row in dump_rows:
                    handle.write(
                        json.dumps(
                            {"ts": stamp, "run_tag": os.environ.get("OSWORLD_RUN_TAG", ""), "step": _DUMP_STEP, **row},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except OSError:
            pass  # diagnostics must never break a training step

    return shaped, advantages
