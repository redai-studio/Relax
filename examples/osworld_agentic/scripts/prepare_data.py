# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld task JSON → Relax parquet.

OSWorld distributes tasks as one JSON file per task, laid out under
``evaluation_examples/<app>/<task-id>.json`` (see the upstream repo at
https://github.com/xlang-ai/OSWorld). Each file is the ``task_config`` dict
consumed by ``DesktopEnv.reset()``: it carries ``id``, ``instruction``,
``config`` (VM setup steps), ``evaluator`` and optional ``snapshot`` /
``proxy`` flags.

Relax's rollout worker reads a flat parquet with these columns:

* ``prompt``    — chat messages the tokenizer sees (a placeholder for the
                  instruction; the real system prompt is built by
                  HybridAgentLocal on the session side)
* ``label``     — a plain string. VM evaluators produce the reward, so we
                  leave this empty; the ``--label-key label`` flag still
                  needs a column to exist.
* ``extra_info``— dict column, surfaces as ``sample.metadata``. We embed
                  the full ``task_config`` under ``metadata.task_config``
                  so ``app/agent.py`` can hand it back to ``DesktopEnv``.

Usage
-----
::

    python3 examples/osworld_agentic/scripts/prepare_data.py \
        --input-dir /path/to/evaluation_examples \
        --output-dir ${DATA_DIR}/osworld/data \
        --train-manifest examples/osworld_agentic/scripts/manifests/train_74.txt \
        --eval-manifest  examples/osworld_agentic/scripts/manifests/eval_309.txt

If neither manifest is provided, every discovered task is emitted to both
``train.parquet`` and ``eval.parquet`` — good enough for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd


try:
    _repo_root = str(Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
    logger = logging.getLogger(__name__)


def _discover_tasks(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Return {task_id: (path, task_config)} for every ``*.json`` under
    ``root``."""
    tasks: dict[str, tuple[Path, dict[str, Any]]] = {}
    for jf in root.rglob("*.json"):
        try:
            cfg = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("skip %s: %s", jf, e)
            continue
        if not isinstance(cfg, dict):
            continue
        tid = cfg.get("id")
        if not tid:
            # Fall back to filename stem — some upstream forks drop the id field.
            tid = jf.stem
            cfg["id"] = tid
        if tid in tasks:
            logger.warning("duplicate task_id %s at %s (kept %s)", tid, jf, tasks[tid][0])
            continue
        tasks[tid] = (jf, cfg)
    return tasks


def _read_manifest(path: Path) -> list[str]:
    """A manifest is a plain text file, one task_id per line (# comments
    allowed)."""
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def _build_row(task_id: str, task_config: dict[str, Any]) -> dict[str, Any]:
    instruction = task_config.get("instruction") or ""
    # The rollout worker embeds this prompt into the chat template; the vendor
    # HybridAgentLocal ignores it and builds its own system+user prompt from
    # ``instruction`` inside the session. Kept minimal to avoid double-prompting.
    prompt = [{"role": "user", "content": instruction}]

    extra_info = {
        "data_source": "osworld",
        "task_id": task_id,
        "instruction": instruction,
        # Arrow cannot infer one struct type across heterogeneous task configs,
        # so the config travels as a JSON string; app/agent.py parses it back.
        "task_config": json.dumps(task_config, ensure_ascii=False),
    }
    return {
        "prompt": prompt,
        "label": "",
        "extra_info": extra_info,
    }


def _select(
    tasks: dict[str, tuple[Path, dict[str, Any]]],
    manifest: Path | None,
) -> list[dict[str, Any]]:
    if manifest is None:
        return [_build_row(tid, cfg) for tid, (_, cfg) in tasks.items()]

    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    for tid in _read_manifest(manifest):
        entry = tasks.get(tid)
        if entry is None:
            missing.append(tid)
            continue
        rows.append(_build_row(tid, entry[1]))
    if missing:
        # Loud but non-fatal: recipes evolve and the manifest may lag.
        logger.warning("manifest %s references %d unknown task_id(s): %s", manifest, len(missing), missing[:5])
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input-dir", type=Path, required=True, help="Path to OSWorld evaluation_examples directory (recursive scan)."
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write train.parquet / eval.parquet.")
    p.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Optional text file listing training task_ids (one per line).",
    )
    p.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Optional text file listing evaluation task_ids (one per line).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _discover_tasks(args.input_dir)
    if not tasks:
        raise SystemExit(f"No JSON tasks found under {args.input_dir}")
    logger.info("Discovered %d tasks under %s", len(tasks), args.input_dir)

    train_rows = _select(tasks, args.train_manifest)
    eval_rows = _select(tasks, args.eval_manifest)

    train_path = args.output_dir / "train.parquet"
    eval_path = args.output_dir / "eval.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(eval_rows).to_parquet(eval_path, index=False)

    logger.info("Wrote %d rows → %s", len(train_rows), train_path)
    logger.info("Wrote %d rows → %s", len(eval_rows), eval_path)


if __name__ == "__main__":
    main()
