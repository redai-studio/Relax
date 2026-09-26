# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dataset builder for ALFWorld agentic rollout (deterministic, reproducible).

Why enumerate game files instead of random seeds?
-------------------------------------------------
ALFWorld is an interactive simulator: a "task instance" is a ``game.tw-pddl``
file on disk (downloaded by ``alfworld-download`` into ``$ALFWORLD_DATA``).

Relax runs one agent process per session, which lets us do the correct thing:
enumerate a **fixed, sorted** list of game files and pin **one specific game per
row**. At rollout time the agent process registers exactly that single game
(``textworld.gym.register_games([game_file], batch_size=1)`` via the env wrapper),
so the eval set is fully deterministic and reproducible, and each game's
``task_type`` lets us report the paper's six ALFWorld categories.

We apply the same filtering as upstream ``AlfredTWEnv.collect_game_files`` (task
type in ``task_types``; skip ``movable`` / ``Sliced``; require ``solvable``) but
add a ``sorted()`` for a stable, host-independent order.

Data contract (verified against Relax agentic runtime)
------------------------------------------------------
Each row is a unified-schema record; Relax forwards ``extra_info`` verbatim into
the agent's ``session_input["metadata"]``:
    - prompt      (list) : a minimal conversation ``[{"role": "user", "content": ""}]``.
                           The agent ignores it (the first real observation comes from
                           ``env.reset()``); it exists only so the data loader accepts the
                           row. A *list* (not str) is required because multimodal-capable
                           checkpoints (e.g. Qwen3.5-*-A3B) load a processor, and
                           ``process_raw_sample`` asserts a conversation-format prompt.
    - label       (str)  : empty; ALFWorld reward is emitted by the environment.
    - data_source (str)  : e.g. "alfworld/AlfredTWEnv".
    - extra_info  (dict) : {
          "game_file":   relative path under $ALFWORLD_DATA (host-portable),
          "task_type":   one of the six ALFWorld categories (for metrics),
          "split":       "train" | "eval",
          "eval_dataset":"train" | "eval_in_distribution" | "eval_out_of_distribution",
          "env_name":    "alfworld/AlfredTWEnv",
          "index":       row index within the split,
      }

train/eval isolation
--------------------
train games come from ``$ALFWORLD_DATA/json_2.1.1/train``; eval games come from
``valid_seen`` (in-distribution) or ``valid_unseen`` (out-of-distribution) — three
physically disjoint pools, so there is no contamination.

Usage
-----
export ALFWORLD_DATA=$HOME/.cache/alfworld   # set by alfworld-download
python examples/on_policy_distillation/agentic_opd/prepare_data.py \
    --output-dir /root/data/agentic_opd/alfworld \
    --eval-dataset eval_out_of_distribution \
    --train-size 2048          # 0 or negative = use all train games \
    --eval-size -1             # -1 = all eval games (paper-style full eval)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Logging — project convention with a stdlib fallback.
# ---------------------------------------------------------------------------
try:
    _repo_root = str(Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
    logger = logging.getLogger(__name__)


# Mirror AlfredTWEnv.TASK_TYPES: task_type id -> canonical name.
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

_SPLIT_SUBDIR = {
    "train": "json_2.1.1/train",
    "eval_in_distribution": "json_2.1.1/valid_seen",
    "eval_out_of_distribution": "json_2.1.1/valid_unseen",
}


def _collect_solvable_game_files(alfworld_data: str, split_subdir: str, task_type_ids: list[int]) -> list[dict]:
    """Enumerate solvable game files under a split, deterministically sorted.

    Replicates the filtering in ``AlfredTWEnv.collect_game_files`` (task-type
    filter, skip movable/Sliced, require ``solvable``) and returns records with a
    **relative** game_file path (portable across mounts) plus the task_type name.
    """
    data_path = os.path.join(alfworld_data, split_subdir)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"ALFWorld data path not found: {data_path}. Did you run `alfworld-download -f` and export ALFWORLD_DATA?"
        )
    allowed_task_types = {TASK_TYPES[t] for t in task_type_ids if t in TASK_TYPES}

    records: list[dict] = []
    # sorted() over os.walk yields a stable, host-independent order — the key
    # difference from upstream collect_game_files (which does not sort).
    for root, _dirs, files in sorted(os.walk(data_path)):
        if "traj_data.json" not in files:
            continue
        if "movable" in root or "Sliced" in root:
            continue
        game_file_path = os.path.join(root, "game.tw-pddl")
        if not os.path.exists(game_file_path):
            continue
        with open(os.path.join(root, "traj_data.json"), encoding="utf-8") as f:
            traj_data = json.load(f)
        task_type = traj_data.get("task_type")
        if task_type not in allowed_task_types:
            continue
        try:
            with open(game_file_path, encoding="utf-8") as f:
                gamedata = json.load(f)
        except Exception:
            continue
        if not gamedata.get("solvable", False):
            continue
        records.append(
            {
                "game_file": os.path.relpath(game_file_path, alfworld_data),
                "task_type": task_type,
            }
        )

    records.sort(key=lambda r: r["game_file"])
    return records


def _to_rows(game_records: list[dict], *, split: str, eval_dataset: str, env_name: str) -> list[dict]:
    rows: list[dict] = []
    for i, rec in enumerate(game_records):
        rows.append(
            {
                # Placeholder conversation: unused by the agent (it resets the env for
                # the real first observation), but must be a list so the data loader
                # accepts it under processor-backed (multimodal) checkpoints.
                "prompt": [{"role": "user", "content": ""}],
                "label": "",
                "data_source": env_name,
                "extra_info": {
                    "game_file": rec["game_file"],
                    "task_type": rec["task_type"],
                    "split": split,
                    "eval_dataset": eval_dataset,
                    "env_name": env_name,
                    "index": i,
                },
            }
        )
    return rows


def _write_parquet(rows: list[dict], path: str) -> None:
    df = pd.DataFrame(rows)
    bad_prompt = sorted({type(x).__name__ for x in df["prompt"] if not isinstance(x, list)})
    assert not bad_prompt, f"'prompt' must be a conversation list for every row; found types: {bad_prompt}"
    bad_extra = sorted({type(x).__name__ for x in df["extra_info"] if not isinstance(x, dict)})
    assert not bad_extra, f"'extra_info' must be a dict for every row; found types: {bad_extra}"
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)


def _summarize(rows: list[dict], name: str) -> None:
    from collections import Counter

    counts = Counter(r["extra_info"]["task_type"] for r in rows)
    logger.info("%s: %d games", name, len(rows))
    for task_type in TASK_TYPES.values():
        if counts.get(task_type):
            logger.info("    %-32s %d", task_type, counts[task_type])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enumerate ALFWorld game files into reproducible train/eval parquets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for train/test parquets.")
    parser.add_argument(
        "--alfworld-data",
        type=str,
        default=os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")),
        help="Root of downloaded ALFWorld data (default: $ALFWORLD_DATA or ~/.cache/alfworld).",
    )
    parser.add_argument("--env-name", type=str, default="alfworld/AlfredTWEnv", help="Env name / data_source key.")
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="eval_out_of_distribution",
        choices=["eval_in_distribution", "eval_out_of_distribution"],
        help="Which held-out pool to enumerate for eval (default: eval_out_of_distribution).",
    )
    parser.add_argument(
        "--task-types",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="ALFWorld task-type ids to include (default: all six).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=2048,
        help="Cap on number of train games (<=0 = use all). Games are shuffled with --train-shuffle-seed "
        "before capping so the subset is a representative sample (default: 2048).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=-1,
        help="Cap on number of eval games (<=0 = use ALL, i.e. the full held-out set, paper-style). Default: -1.",
    )
    parser.add_argument(
        "--train-shuffle-seed",
        type=int,
        default=42,
        help="Seed for shuffling the train game list before capping (default: 42). Eval is never shuffled.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    alfworld_data = os.path.expandvars(os.path.expanduser(args.alfworld_data))
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Enumerating ALFWorld games under %s", alfworld_data)

    # --- Train: enumerate, shuffle deterministically, then cap. ---
    train_games = _collect_solvable_game_files(alfworld_data, _SPLIT_SUBDIR["train"], args.task_types)
    logger.info("Found %d solvable train games", len(train_games))
    if args.train_size and args.train_size > 0 and args.train_size < len(train_games):
        rng = random.Random(args.train_shuffle_seed)
        train_games = train_games[:]  # copy before shuffle
        rng.shuffle(train_games)
        train_games = train_games[: args.train_size]
        train_games.sort(key=lambda r: r["game_file"])  # keep on-disk order stable within the cap
    train_rows = _to_rows(train_games, split="train", eval_dataset="train", env_name=args.env_name)

    # --- Eval: enumerate the held-out pool in fixed order; cap only if asked. ---
    eval_games = _collect_solvable_game_files(alfworld_data, _SPLIT_SUBDIR[args.eval_dataset], args.task_types)
    logger.info("Found %d solvable eval games in %s", len(eval_games), args.eval_dataset)
    if args.eval_size and args.eval_size > 0 and args.eval_size < len(eval_games):
        eval_games = eval_games[: args.eval_size]  # deterministic prefix of the sorted list
    eval_rows = _to_rows(eval_games, split="eval", eval_dataset=args.eval_dataset, env_name=args.env_name)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    _write_parquet(train_rows, train_path)
    _write_parquet(eval_rows, test_path)

    logger.info("=== ALFWorld dataset summary ===")
    _summarize(train_rows, "train")
    _summarize(eval_rows, f"eval ({args.eval_dataset})")
    logger.info("Done. Output written to %s", args.output_dir)


if __name__ == "__main__":
    main()
