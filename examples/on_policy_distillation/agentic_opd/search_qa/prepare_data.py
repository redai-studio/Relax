# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dataset builder for Search-QA agentic rollout.

Reads the Search-R1 corpus ``PeterJinGo/nq_hotpotqa_train`` (NQ + HotpotQA) and
writes Relax unified-schema parquets. Each row carries a placeholder ``prompt``
(the agent renders the real first prompt from ``env.reset()``) and forwards the
``(question, golden_answers)`` pair through ``extra_info`` -- Relax passes
``extra_info`` verbatim into the agent's ``session_input["metadata"]``.

Usage:
    python examples/on_policy_distillation/agentic_opd/search_qa/prepare_data.py \
        --output-dir /root/data/agentic_opd/search_qa \
        --train-size 2048 --eval-size 512   # 0 or negative = use all rows
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path


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


def _extract_golds(row: dict) -> list[str]:
    """Return the golden-answers list for a source row.

    Prefer the flat ``golden_answers`` column; fall back to SDAR's nested
    ``reward_model.ground_truth.target`` shape if that is all that is present.
    """
    golds = row.get("golden_answers")
    if golds is None:
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        if isinstance(ground_truth, dict):
            golds = ground_truth.get("target")
        elif isinstance(ground_truth, (list, str)):
            golds = ground_truth
    if golds is None:
        return []
    if isinstance(golds, str):
        return [golds]
    return [str(g) for g in golds]


def _extract_question(row: dict) -> str:
    """Return the natural-language question for a source row."""
    question = row.get("question")
    if question:
        return str(question)
    # Fall back to the last user turn of a chat-format ``prompt`` column.
    prompt = row.get("prompt")
    if isinstance(prompt, (list, tuple)) and prompt:
        last = prompt[-1]
        if isinstance(last, dict) and last.get("content"):
            return str(last["content"])
    return ""


def _to_rows(source_rows: list[dict], *, split: str, env_name: str) -> list[dict]:
    rows: list[dict] = []
    dropped = 0
    for rec in source_rows:
        question = _extract_question(rec)
        golds = _extract_golds(rec)
        if not question or not golds:
            dropped += 1
            continue
        i = len(rows)
        rows.append(
            {
                # Placeholder: unused by the agent, but must be a conversation list
                # so processor-backed (multimodal) checkpoints accept it.
                "prompt": [{"role": "user", "content": ""}],
                "label": "",
                "data_source": env_name,
                "extra_info": {
                    "question": question,
                    "ground_truth": golds,
                    "data_source": rec.get("data_source", ""),
                    "split": split,
                    "env_name": env_name,
                    "index": i,
                },
            }
        )
    if dropped:
        logger.warning("Dropped %d %s rows missing a question or golden_answers", dropped, split)
    return rows


def _write_parquet(rows: list[dict], path: str) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    bad_prompt = sorted({type(x).__name__ for x in df["prompt"] if not isinstance(x, list)})
    assert not bad_prompt, f"'prompt' must be a conversation list for every row; found types: {bad_prompt}"
    bad_extra = sorted({type(x).__name__ for x in df["extra_info"] if not isinstance(x, dict)})
    assert not bad_extra, f"'extra_info' must be a dict for every row; found types: {bad_extra}"
    bad_gold = sorted(
        {type(x["ground_truth"]).__name__ for x in df["extra_info"] if not isinstance(x["ground_truth"], list)}
    )
    assert not bad_gold, f"'ground_truth' must be a golds list for every row; found types: {bad_gold}"
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)


def _write_eval_splits(eval_rows: list[dict], output_dir: str) -> dict[str, int]:
    """Write one eval parquet per original ``data_source`` so each dataset's EM
    can be reported on its own.

    Returns ``{path: n_rows}``; no-ops when fewer than two distinct source
    labels are present.
    """
    from collections import defaultdict

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in eval_rows:
        src = (row.get("extra_info") or {}).get("data_source") or ""
        by_source[src].append(row)

    labelled = {src: rows for src, rows in by_source.items() if src}
    if len(labelled) < 2:
        logger.warning(
            "Eval rows carry <2 distinct data_source labels (%s); skipping per-source split -- "
            "report the combined test.parquet, or check the upstream data_source column.",
            {src or "<empty>": len(rows) for src, rows in by_source.items()},
        )
        return {}

    written: dict[str, int] = {}
    for src, rows in sorted(labelled.items()):
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", src).strip("_").lower()
        path = os.path.join(output_dir, f"test_{slug}.parquet")
        _write_parquet(rows, path)
        written[path] = len(rows)
    logger.info("Per-source eval splits: %s", {os.path.basename(p): n for p, n in written.items()})
    return written


def _resolve_split_parquets(hf_path: str, split: str, cache_dir: str | None) -> list[str]:
    """Return local parquet path(s) for ``split``.

    Supports a local dir (``<split>*.parquet``, recursively), a local file, or
    a HuggingFace dataset id (downloads ``<split>.parquet`` from the hub).
    """
    import glob

    if os.path.isdir(hf_path):
        hits = sorted(glob.glob(os.path.join(hf_path, f"{split}*.parquet")))
        if not hits:
            hits = sorted(glob.glob(os.path.join(hf_path, "**", f"{split}*.parquet"), recursive=True))
        if not hits:
            raise FileNotFoundError(f"No {split}*.parquet found under {hf_path}")
        return hits
    if os.path.isfile(hf_path):
        return [hf_path]

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=hf_path, filename=f"{split}.parquet", repo_type="dataset", cache_dir=cache_dir)
    return [path]


def _load_split(hf_path: str, split: str, cache_dir: str | None) -> list[dict]:
    """Load one split as a list of source dicts, keeping only the flat columns
    we use.

    Reads parquet directly (not via ``datasets.load_dataset``) because the
    train/test splits carry different nested struct columns and load_dataset
    fails to strict-cast test to the train schema.
    """
    import pyarrow.parquet as pq

    needed = ["question", "golden_answers", "data_source"]
    rows: list[dict] = []
    for path in _resolve_split_parquets(hf_path, split, cache_dir):
        available = set(pq.read_schema(path).names)
        cols = [c for c in needed if c in available]
        rows.extend(pq.read_table(path, columns=cols).to_pylist())
    return rows


def _select_eval(eval_src: list[dict], eval_size: int) -> list[dict]:
    """Pick the eval subset with every ``data_source`` represented.

    The test split is ordered by source, so a flat prefix would be single-
    source. Round-robin across sources instead so the ``eval_size`` budget
    splits roughly evenly (deterministic: within-source order preserved, no
    RNG). Falls back to a plain prefix when the cap is not binding or fewer
    than two source labels are present.
    """
    from collections import OrderedDict

    if not eval_size or eval_size <= 0 or eval_size >= len(eval_src):
        return eval_src

    by_source: OrderedDict[str, list[dict]] = OrderedDict()
    for rec in eval_src:
        by_source.setdefault(rec.get("data_source", ""), []).append(rec)
    if len(by_source) < 2:
        return eval_src[:eval_size]

    selected: list[dict] = []
    cursors = {src: 0 for src in by_source}
    while len(selected) < eval_size:
        advanced = False
        for src, group in by_source.items():
            if cursors[src] < len(group):
                selected.append(group[cursors[src]])
                cursors[src] += 1
                advanced = True
                if len(selected) >= eval_size:
                    break
        if not advanced:  # every source exhausted before reaching eval_size
            break
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the Search-R1 nq_hotpotqa corpus into reproducible train/eval parquets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for train/test parquets.")
    parser.add_argument(
        "--hf-path",
        type=str,
        default="PeterJinGo/nq_hotpotqa_train",
        help="HuggingFace dataset id (or local path) for the QA corpus.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("HF_DATASETS_CACHE") or None,
        help="HuggingFace datasets cache dir (default: $HF_DATASETS_CACHE).",
    )
    parser.add_argument("--env-name", type=str, default="search_qa/nq_hotpotqa", help="Env name / data_source key.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=0,
        help="Cap on train rows (<=0 = use all; default: all). A capped smoke-test subset is shuffled "
        "with --train-shuffle-seed before selection.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=512,
        help="Cap on eval rows, round-robin across sources so every dataset is represented "
        "(default: 512, keeps the eval pass to minutes; pass <=0 for the full test sets, tens "
        "of thousands of rows / hours per eval).",
    )
    parser.add_argument(
        "--train-shuffle-seed",
        type=int,
        default=42,
        help="Seed for shuffling the train rows before capping (default: 42). Eval is never shuffled.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading %s (train/test splits)", args.hf_path)
    train_src = _load_split(args.hf_path, "train", args.cache_dir)
    eval_src = _load_split(args.hf_path, "test", args.cache_dir)
    logger.info("Loaded %d train / %d test source rows", len(train_src), len(eval_src))

    # Train: shuffle deterministically, then cap.
    if args.train_size and args.train_size > 0 and args.train_size < len(train_src):
        rng = random.Random(args.train_shuffle_seed)
        train_src = train_src[:]
        rng.shuffle(train_src)
        train_src = train_src[: args.train_size]

    # Eval: balanced per-source subset so both NQ and HotpotQA are present.
    eval_src = _select_eval(eval_src, args.eval_size)

    train_rows = _to_rows(train_src, split="train", env_name=args.env_name)
    eval_rows = _to_rows(eval_src, split="eval", env_name=args.env_name)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    _write_parquet(train_rows, train_path)
    _write_parquet(eval_rows, test_path)
    _write_eval_splits(eval_rows, args.output_dir)

    logger.info("=== Search-QA dataset summary ===")
    logger.info("train: %d questions", len(train_rows))
    logger.info("eval : %d questions", len(eval_rows))
    logger.info("Done. Output written to %s", args.output_dir)


if __name__ == "__main__":
    main()
