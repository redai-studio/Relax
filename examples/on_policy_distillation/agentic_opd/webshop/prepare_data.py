# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dataset builder for WebShop agentic rollout (deterministic, reproducible).

Why enumerate goal indices instead of random seeds?
---------------------------------------------------
WebShop is an interactive simulator: a "task instance" is a *goal* drawn from
``env.server.goals`` and selected by an integer ``session`` index
(``env.reset(session=idx)``). The upstream SDAR / verl-agent path draws goals
non-deterministically from one vectorized env with a single global seed — fine
for *training* diversity, but unusable for eval (every run scores a different,
unordered subset). Relax runs one agent process per session, which lets us pin
**one specific goal index per row** so the eval set is fully deterministic.

Split convention (mirrors SDAR env_package/webshop/envs.py:141-144):
    - eval  : goal indices ``range(0, 500)``
    - train : goal indices ``range(500, num_goals)``
These are physically disjoint index ranges, so there is no contamination.

Data contract (verified against Relax agentic runtime)
------------------------------------------------------
Each row is a unified-schema record; Relax forwards ``extra_info`` verbatim into
the agent's ``session_input["metadata"]``:
    - prompt      (list) : a minimal conversation ``[{"role": "user", "content": ""}]``.
                           The agent ignores it (the first real observation comes
                           from the env server's reset); a *list* is required
                           because multimodal-capable checkpoints load a processor
                           and ``process_raw_sample`` asserts a conversation prompt.
    - label       (str)  : empty; WebShop reward is emitted by the environment.
    - data_source (str)  : "webshop/WebAgentTextEnv".
    - extra_info  (dict) : {
          "goal_idx":    goal index (int) — selects the WebShop goal (DESIGN §7.1),
          "split":       "train" | "eval",
          "env_name":    "webshop/WebAgentTextEnv",
          "index":       row index within the split,
      }

Usage
-----
# Counts goals by loading the catalog once (needs $WEBSHOP_HOME + the data the
# README installs). WEBSHOP_HOME is the dir containing web_agent_site.
python examples/on_policy_distillation/agentic_opd/webshop/prepare_data.py \
    --output-dir /root/webshop-relax \
    --webshop-home /path/to/WebShop

# If you already know the goal count, pass --num-goals to skip loading the catalog.
python examples/on_policy_distillation/agentic_opd/webshop/prepare_data.py \
    --output-dir /root/webshop-relax --num-goals 1000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging — project convention with a stdlib fallback.
# ---------------------------------------------------------------------------
try:
    _repo_root = str(Path(__file__).resolve().parents[4])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
    logger = logging.getLogger(__name__)


# Eval uses the first EVAL_SIZE goal indices; train uses the rest.
EVAL_INDEX_END = 500


def _count_goals(webshop_home: str) -> int:
    """Instantiate a WebShop env once to read ``len(env.server.goals)``.

    Heavy (loads the product catalog under ``webshop_home``); pass ``--num-
    goals`` to skip when the count is known. Uses WebShop's DEFAULT_FILE_PATH
    (the 1000 subset the README installs) unless the full catalog is in place.
    """
    if webshop_home and webshop_home not in sys.path:
        sys.path.insert(0, webshop_home)
    from web_agent_site.envs import WebAgentTextEnv

    env = WebAgentTextEnv(observation_mode="text", num_products=None)
    try:
        return len(env.server.goals)
    finally:
        try:
            env.close()
        except Exception:
            pass


def _to_rows(goal_ids: list[int], *, split: str, env_name: str) -> list[dict]:
    rows: list[dict] = []
    for i, gid in enumerate(goal_ids):
        rows.append(
            {
                # Placeholder conversation: unused by the agent (the real first
                # observation comes from the env server's reset), but must be a
                # list so the data loader accepts it under processor-backed
                # (multimodal) checkpoints.
                "prompt": [{"role": "user", "content": ""}],
                "label": "",
                "data_source": env_name,
                "extra_info": {
                    "goal_idx": int(gid),
                    "split": split,
                    "env_name": env_name,
                    "index": i,
                },
            }
        )
    return rows


def _write_parquet(rows: list[dict], path: str) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    bad_prompt = sorted({type(x).__name__ for x in df["prompt"] if not isinstance(x, list)})
    assert not bad_prompt, f"'prompt' must be a conversation list for every row; found types: {bad_prompt}"
    bad_extra = sorted({type(x).__name__ for x in df["extra_info"] if not isinstance(x, dict)})
    assert not bad_extra, f"'extra_info' must be a dict for every row; found types: {bad_extra}"
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enumerate WebShop goal indices into reproducible train/eval parquets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for train/test parquets.")
    parser.add_argument(
        "--webshop-home",
        type=str,
        default=os.environ.get("WEBSHOP_HOME", ""),
        help="Path to the WebShop simulator source (the dir containing web_agent_site). "
        "Defaults to $WEBSHOP_HOME. Only needed when --num-goals is not given.",
    )
    parser.add_argument(
        "--num-goals",
        type=int,
        default=0,
        help="Total number of WebShop goals. If >0, the product catalog is NOT loaded "
        "(fast); otherwise a WebShop env is instantiated to read len(goals).",
    )
    parser.add_argument("--env-name", type=str, default="webshop/WebAgentTextEnv", help="Env name / data_source key.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=2048,
        help="Cap on number of train goals (<=0 = use all). Indices are shuffled with "
        "--train-shuffle-seed before capping so the subset is representative (default: 2048).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=-1,
        help="Cap on number of eval goals (<=0 = use the full held-out set of 500). Default: -1.",
    )
    parser.add_argument(
        "--train-shuffle-seed",
        type=int,
        default=42,
        help="Seed for shuffling the train index list before capping (default: 42). Eval is never shuffled.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_goals and args.num_goals > 0:
        num_goals = args.num_goals
        logger.info("Using provided --num-goals=%d (catalog not loaded)", num_goals)
    else:
        if not args.webshop_home:
            raise SystemExit("--num-goals not given; --webshop-home (or $WEBSHOP_HOME) is required to count goals.")
        num_goals = _count_goals(args.webshop_home)
        logger.info("Counted %d WebShop goals from the catalog", num_goals)

    if num_goals <= EVAL_INDEX_END:
        raise SystemExit(
            f"Too few goals ({num_goals}); need > {EVAL_INDEX_END} so that the eval "
            f"range [0, {EVAL_INDEX_END}) and a non-empty train range both exist."
        )

    # --- Eval: fixed prefix of the sorted index range. ---
    eval_ids = list(range(0, EVAL_INDEX_END))
    if args.eval_size and args.eval_size > 0 and args.eval_size < len(eval_ids):
        eval_ids = eval_ids[: args.eval_size]

    # --- Train: [500, num_goals); shuffle deterministically, then cap. ---
    train_ids = list(range(EVAL_INDEX_END, num_goals))
    if args.train_size and args.train_size > 0 and args.train_size < len(train_ids):
        rng = random.Random(args.train_shuffle_seed)
        rng.shuffle(train_ids)
        train_ids = train_ids[: args.train_size]
        train_ids.sort()  # keep index order stable within the cap

    train_rows = _to_rows(train_ids, split="train", env_name=args.env_name)
    eval_rows = _to_rows(eval_ids, split="eval", env_name=args.env_name)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    _write_parquet(train_rows, train_path)
    _write_parquet(eval_rows, test_path)

    logger.info("=== WebShop dataset summary ===")
    logger.info("train: %d goals (indices [%d, %d))", len(train_rows), EVAL_INDEX_END, num_goals)
    logger.info("eval : %d goals (indices [0, %d))", len(eval_rows), len(eval_ids))
    logger.info("Done. Output written to %s", args.output_dir)


if __name__ == "__main__":
    main()
