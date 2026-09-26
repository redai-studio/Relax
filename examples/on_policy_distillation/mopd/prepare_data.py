# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Data preprocessing tool for MOPD (Multi-Teacher On-Policy Distillation).

Merges multiple domain-specific datasets into a single shuffled training parquet,
stamping each row with a ``data_source`` column for per-sample teacher routing.

Output: ``<output-dir>/train.parquet``, and when a test split exists (either
``--test-ratio > 0`` or a manifest with ``split: "test"`` entries)
``<output-dir>/test.parquet`` plus a source-balanced ``test_small.parquet``
(``--small-eval-per-source`` rows per data_source, default 25) for fast
training-time eval.

Every row shares the same schema regardless of source:
    - prompt      (str)       : plain-string user prompt; VL prompts embed a
                                leading "<image>" placeholder
    - label       (str)       : ground-truth answer
    - data_source (str)       : routing key (e.g. "openai/gsm8k", "dapo-math-17k")
    - images      (list|None) : base64 ``data:`` image URIs (VL only); embedded
                                inline so the parquet is mount-portable
    - extra_info  (dict)      : always ``{"rm_type": "mopd", "source": ...}``,
                                stored as a dict (parquet struct), never a JSON
                                string — the Relax loader reads it directly and
                                never calls ``json.loads`` on it

Usage examples
--------------
# Quickstart — point at raw GSM8K / Geo3K / dapo-math / openr1mm sources
python examples/on_policy_distillation/mopd/prepare_data.py \
    --gsm8k-dir /data/gsm8k \
    --geo3k-dir /data/geo3k \
    --dapo-math-path /data/dapo-math-17k/dapo-math-17k.jsonl \
    --openr1mm-path /data/multimodal-open-r1-8k-verified/train-00000-of-00001.parquet \
    --output-dir /data/MOPD

# Manifest — arbitrary list of parquet files
python examples/on_policy_distillation/mopd/prepare_data.py \
    --manifest manifest.json \
    --output-dir /data/MOPD
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Logging — project convention: relax.utils.logging_utils.get_logger
# The script is self-contained (no relax core imports at the top), so we add
# the repo root to sys.path only for the logger, with a stdlib fallback.
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

# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------

_VERL_REQUIRED_COLS = {"prompt", "data_source"}
_RAW_GSM8K_COLS = {"question", "answer"}
_RAW_GEO3K_COLS = {"problem", "answer"}


def _detect_format(df: pd.DataFrame) -> str:
    """Return one of 'verl', 'gsm8k', 'geo3k', or raise."""
    cols = set(df.columns)
    if _VERL_REQUIRED_COLS.issubset(cols):
        return "verl"
    if _RAW_GSM8K_COLS.issubset(cols) and "problem" not in cols:
        return "gsm8k"
    if _RAW_GEO3K_COLS.issubset(cols) and "problem" in cols:
        return "geo3k"
    raise ValueError(
        f"Cannot auto-detect format. Columns present: {sorted(cols)}. "
        "Expected verl (prompt+data_source), raw GSM8K (question+answer), or raw Geo3K (problem+answer+images)."
    )


# ---------------------------------------------------------------------------
# Converters — raw format -> unified schema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_MATH = "Please reason step by step, and put your final answer within \\boxed{}."

# The multimodal-open-r1 VL teacher (Qwen3.5-9B-vl-teacher-hf) was RL-trained
# (GRPO) WITH this exact system prompt. Without it the model never emits
# <answer></answer> tags, so the openr1mm reward can't extract its answer and its
# accuracy collapses (~0.6 -> ~0.35) — which starves the OPD signal for VL. We
# prepend it to the VL prompt string (rather than a real system turn) because the
# MOPD parquet stores `prompt` as a plain string and a message-list column can't
# be mixed with the text rows' string prompts. Byte-identical to the string in
# scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-async.sh.
OPENR1MM_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
)

# Geo3K instruction, kept byte-identical to verl's data_preprocess/geo3k.py so the
# format_reward in relax/engine/rewards/geo3k.py (which requires <think></think>
# followed by \boxed{}) actually fires. Appended AFTER the problem, matching verl's
# `prompt = problem + " " + instruction_following`.
GEO3K_INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)


def _convert_gsm8k_row(row: pd.Series) -> dict:
    """Convert a raw GSM8K row to the unified MOPD schema.

    ``prompt`` is stored as a **plain string** (system instruction + question
    joined), not a chat-message list: the Relax loader (``build_messages``)
    treats a string prompt as the user turn and wraps it via ``--apply-chat-
    template``. A JSON-encoded list would be mistaken for a single user message
    whose content is the raw JSON text — see ``_load_dapo_math_jsonl`` below
    for the same rationale.
    """
    prompt = f"{SYSTEM_PROMPT_MATH}\n{row['question']}"
    # GSM8K answers contain '####' separator; extract the final numeric answer.
    raw_answer = str(row["answer"])
    if "####" in raw_answer:
        label = raw_answer.split("####")[-1].strip()
    else:
        label = raw_answer.strip()
    return {
        "prompt": prompt,
        "label": label,
        "data_source": "openai/gsm8k",
        "extra_info": {"rm_type": "mopd", "source": "openai/gsm8k"},
    }


def _encode_geo3k_images(images) -> list[str] | None:
    """Normalize a raw Geo3K ``images`` cell to Relax-loadable image refs.

    The HF ``Image`` feature type serializes to a ``{"bytes": ..., "path":
    ...}`` struct on ``to_parquet()`` (confirmed against the real multimodal-
    open-r1 ``image`` column, which uses the same encoding) — not a
    path/URL/``data:`` URI that the Relax image loader understands.
    Base64-encode any raw bytes inline (same rationale as
    ``_load_openr1mm_parquet``); pass through values that are already string
    refs (e.g. from a pre-cleaned verl-style source).
    """
    if images is None:
        return None
    out = []
    for img in images:
        if isinstance(img, str):
            out.append(img)
        elif isinstance(img, dict) and img.get("bytes"):
            b64 = base64.b64encode(img["bytes"]).decode("ascii")
            out.append(f"data:image/png;base64,{b64}")
    return out or None


def _convert_geo3k_row(row: pd.Series) -> dict:
    """Convert a raw Geo3K row to the unified MOPD schema.

    ``prompt`` is a **plain string** with the ``"<image>"`` placeholder (same
    convention as ``_convert_gsm8k_row`` above and ``_load_openr1mm_parquet``
    below): a JSON-encoded chat-message list would be mishandled by the Relax
    loader as a single user turn whose content is the raw JSON text.
    """
    raw_images = row["images"] if "images" in row.index else None
    # pandas fills a missing cell in a mixed column with NaN (a bare float),
    # not None, when some rows in the same DataFrame have images and others
    # don't — guard against iterating over it.
    images = None if raw_images is None or isinstance(raw_images, float) else _encode_geo3k_images(raw_images)
    has_image = images is not None
    # Some raw Geo3K variants already embed the "<image>" placeholder in ``problem``
    # (e.g. "<image>Find x."); only prepend one when it is absent, so we never emit
    # two placeholders for a single image (which desyncs the VL processor).
    body = row["problem"]
    if has_image and "<image>" not in str(body):
        body = f"<image>\n{body}"
    prompt = f"{body} {GEO3K_INSTRUCTION}"
    result: dict = {
        "prompt": prompt,
        "label": str(row["answer"]).strip(),
        "data_source": "hiyouga/geometry3k",
    }
    if has_image:
        result["images"] = images
    result["extra_info"] = {"rm_type": "mopd", "source": "hiyouga/geometry3k"}
    return result


def _convert_verl_row(row: pd.Series, data_source_override: str | None = None) -> dict:
    """Normalise a verl-format row to the unified MOPD schema."""
    result: dict = {
        "prompt": row["prompt"],
        "data_source": data_source_override or row["data_source"],
    }
    # Label: prefer reward_model.ground_truth, fall back to explicit label column.
    if "reward_model" in row.index and isinstance(row["reward_model"], dict):
        result["label"] = row["reward_model"].get("ground_truth", "")
    elif "label" in row.index:
        result["label"] = row["label"]
    else:
        result["label"] = ""
    # pandas fills a missing cell with NaN (a bare float), not None, when some
    # rows in the same column have a value and others don't.
    images = row["images"] if "images" in row.index else None
    if images is not None and not isinstance(images, float):
        result["images"] = images
    extra_info = row["extra_info"] if "extra_info" in row.index else None
    if extra_info is not None and not isinstance(extra_info, float):
        result["extra_info"] = extra_info
    return result


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_dapo_math_jsonl(jsonl_path: str) -> pd.DataFrame:
    """Load dapo-math-17k from JSONL.

    Each source row has ``prompt`` as a chat-message list (a single ``user``
    turn whose content already embeds the instruction). ``prompt`` is flattened
    to a **plain string** (join the message contents), not stored as a JSON-
    encoded list: the Relax loader (``build_messages``) treats a string prompt
    as the user turn and wraps it via ``--apply-chat-template``. A JSON-encoded
    list would be mistaken for a single user message whose content is the raw
    JSON text.
    """
    logger.info("Loading dapo-math-17k from %s", jsonl_path)
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt_msgs = row["prompt"]
            # Single user turn in dapo-math-17k; join defensively if more.
            prompt = "\n".join(m.get("content", "") for m in prompt_msgs)
            records.append(
                {
                    "prompt": prompt,
                    "label": str(row["label"]),
                    "data_source": "dapo-math-17k",
                    "images": None,
                    "extra_info": {"rm_type": "mopd", "source": "dapo-math-17k"},
                }
            )
    logger.info("  -> %d rows", len(records))
    return pd.DataFrame(records)


def _load_openr1mm_parquet(parquet_path: str) -> pd.DataFrame:
    """Load multimodal-open-r1-8k-verified from parquet.

    Images are stored as raw bytes in the source ``image`` column. We embed
    each image *inline* into the output ``images`` column as a base64 ``data:``
    URI instead of writing separate JPEG files with absolute paths: the Relax
    image loader only accepts an absolute path, an HTTP(S) URL, or a ``data:``
    URI (it does NOT resolve a relative path against the parquet directory), so
    an absolute file path breaks the moment the parquet is mounted at a
    different path on another host. Inlining the bytes makes the parquet fully
    self-contained and portable across mounts.
    """
    import pyarrow.parquet as pq

    logger.info("Loading multimodal-open-r1 from %s", parquet_path)
    table = pq.read_table(parquet_path)

    records = []
    for i in range(len(table)):
        row = {col: table[col][i].as_py() for col in table.column_names}

        problem = row.get("problem", "")
        solution = row.get("solution", "")

        # Embed image bytes inline as a base64 data URI (portable across mounts).
        img_data = row.get("image") or {}
        img_bytes = img_data.get("bytes") if isinstance(img_data, dict) else None
        img_uri = None
        if img_bytes:
            b64 = base64.b64encode(img_bytes).decode("ascii")
            img_uri = f"data:image/jpeg;base64,{b64}"

        # Store prompt as a plain string with the "<image>" placeholder (same
        # convention as _convert_geo3k_row above): the Relax loader wraps a
        # string prompt as the user turn and, when --multimodal-keys is set,
        # splits the content on "<image>" to inject one image per marker.
        # Prepend OPENR1MM_SYSTEM_PROMPT so the VL teacher runs in-distribution
        # during MOPD (see the constant's docstring for why).
        if img_uri:
            body = problem if "<image>" in problem else f"<image>\n{problem}"
            images = [img_uri]
        else:
            # No image bytes for this row: strip any stray "<image>" so the prompt
            # never carries a placeholder without a matching image (that desyncs
            # the VL processor when it splits on the marker to inject images).
            body = problem.replace("<image>", "").strip()
            images = None
        prompt = f"{OPENR1MM_SYSTEM_PROMPT}\n{body}"

        records.append(
            {
                "prompt": prompt,
                "label": solution,
                "data_source": "multimodal-open-r1",
                "images": images,
                "extra_info": {"rm_type": "mopd", "source": "multimodal-open-r1"},
            }
        )
    logger.info("  -> %d rows", len(records))
    return pd.DataFrame(records)


def _load_and_convert(
    parquet_path: str,
    data_source_override: str | None = None,
) -> pd.DataFrame:
    """Load a single parquet file, auto-detect format, and convert to unified
    schema."""
    logger.info("Loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("  -> %d rows, columns: %s", len(df), list(df.columns))

    fmt = _detect_format(df)
    logger.info("  -> Detected format: %s", fmt)

    if fmt == "gsm8k":
        records = [_convert_gsm8k_row(row) for _, row in df.iterrows()]
    elif fmt == "geo3k":
        records = [_convert_geo3k_row(row) for _, row in df.iterrows()]
    else:
        records = [_convert_verl_row(row, data_source_override) for _, row in df.iterrows()]

    return pd.DataFrame(records)


def _load_quickstart_dir(
    directory: str,
    data_source_override: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load train.parquet (required) and test.parquet (optional) from.

    *directory*.
    """
    train_path = os.path.join(directory, "train.parquet")
    test_path = os.path.join(directory, "test.parquet")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Expected train.parquet in {directory}, but file not found.")

    train_df = _load_and_convert(train_path, data_source_override)
    test_df = _load_and_convert(test_path, data_source_override) if os.path.isfile(test_path) else None
    return train_df, test_df


# ---------------------------------------------------------------------------
# Merge, shuffle, split, write
# ---------------------------------------------------------------------------


def _merge_and_shuffle(
    frames: list[pd.DataFrame],
    seed: int,
) -> pd.DataFrame:
    """Concatenate dataframes and interleave by ``data_source``.

    A plain global shuffle can still produce long same-source runs by chance
    (e.g. 64 consecutive ``dapo-math-17k`` rows). Since each DP rank in
    streaming/colocate training consumes a contiguous slice of the dataset
    order (see ``get_data_iterator`` in ``relax/backends/megatron/data.py``), a
    run that long can leave one rank's local training step with zero multimodal
    samples — the student model's vision-encoder parameters then get no
    gradient that step, crashing Megatron's grad-sync with "Communication call
    has not been issued for this bucket". Smooth weighted round-robin (shuffle
    within each source, then interleave sources proportionally to their size, à
    la nginx's SWRR load balancing) bounds the max run length of any single
    source to ``ceil(total / count[source])`` instead of leaving it to chance.
    """
    merged = pd.concat(frames, ignore_index=True)
    groups = {
        src: group.sample(frac=1, random_state=seed).reset_index(drop=True)
        for src, group in merged.groupby("data_source", sort=True)
    }
    counts = {src: len(group) for src, group in groups.items()}
    total = sum(counts.values())

    credits = dict.fromkeys(groups, 0.0)
    cursor = dict.fromkeys(groups, 0)
    rows = []
    for _ in range(total):
        for src in groups:
            credits[src] += counts[src] / total
        pick = max((src for src in groups if cursor[src] < counts[src]), key=lambda s: credits[s])
        credits[pick] -= 1.0
        rows.append(groups[pick].iloc[cursor[pick]])
        cursor[pick] += 1

    return pd.DataFrame(rows).reset_index(drop=True)


def _train_test_split(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified-ish split by data_source (falls back to random if too few
    samples)."""
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    test_frames = []
    train_frames = []
    for _, group in df.groupby("data_source"):
        n_test = max(1, int(len(group) * test_ratio))
        shuffled = group.sample(frac=1, random_state=seed)
        test_frames.append(shuffled.iloc[:n_test])
        train_frames.append(shuffled.iloc[n_test:])

    train = pd.concat(train_frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to parquet.

    ``prompt`` must be a plain string and ``extra_info`` a dict for every row
    (never a JSON-encoded list/string) — see the module docstring. Asserting
    here instead of silently ``json.dumps``-ing a stray list/dict makes a
    future converter bug fail loudly at prep time instead of producing data
    that silently corrupts prompts at train time.
    """
    logger.info("Writing %d rows to %s", len(df), path)
    if "prompt" in df.columns:
        bad = sorted({type(x).__name__ for x in df["prompt"] if not isinstance(x, str)})
        assert not bad, f"'prompt' must be a plain str for every row; found types: {bad}"
    if "extra_info" in df.columns:
        bad = sorted({type(x).__name__ for x in df["extra_info"] if not isinstance(x, dict)})
        assert not bad, f"'extra_info' must be a dict for every row; found types: {bad}"
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_test_small(test_df: pd.DataFrame, n_per_source: int, seed: int) -> pd.DataFrame:
    """Build a small, source-balanced eval subset from the test split.

    Takes up to ``n_per_source`` rows from *every* ``data_source`` present in
    ``test_df`` (not hardcoded to a fixed pair), so per-source eval metrics
    (e.g. ``compute_mopd_metrics``) stay meaningful even after adding new
    teachers. Used for fast training-time monitoring; point ``--eval-prompt-
    data`` at the full ``test.parquet`` for a complete eval.
    """
    parts = [group.head(n_per_source) for _, group in test_df.groupby("data_source")]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MOPD data preprocessor — merge domain datasets with data_source routing column.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Quickstart mode
    parser.add_argument("--gsm8k-dir", type=str, default=None, help="Path to directory with GSM8K train.parquet")
    parser.add_argument("--geo3k-dir", type=str, default=None, help="Path to directory with Geo3K train.parquet")
    parser.add_argument(
        "--dapo-math-path", type=str, default=None, help="Path to dapo-math-17k.jsonl (single JSONL file)"
    )
    parser.add_argument(
        "--openr1mm-path",
        type=str,
        default=None,
        help="Path to multimodal-open-r1-8k-verified parquet (single file, raw image bytes)",
    )

    # Manifest mode
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to JSON manifest file listing parquet files with data_source and split info",
    )

    # Output
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for merged parquets")

    # Options
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Fraction of data to split as test set (default: 0, use existing splits)",
    )
    parser.add_argument(
        "--small-eval-per-source",
        type=int,
        default=25,
        help=(
            "Rows per data_source to sample from the test split into test_small.parquet, "
            "for fast training-time eval (default: 25). Set to 0 to skip writing it."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    has_quickstart = any(
        [args.gsm8k_dir is not None, args.geo3k_dir is not None, args.dapo_math_path, args.openr1mm_path]
    )
    has_manifest = args.manifest is not None

    if not has_quickstart and not has_manifest:
        parser.error(
            "Provide at least one quickstart flag "
            "(--gsm8k-dir / --geo3k-dir / --dapo-math-path / --openr1mm-path) or --manifest."
        )
    if has_quickstart and has_manifest:
        parser.error("Cannot mix quickstart flags and --manifest. Choose one mode.")

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    # --- Quickstart mode ---
    if has_quickstart:
        if args.gsm8k_dir:
            train_df, test_df = _load_quickstart_dir(args.gsm8k_dir, data_source_override="openai/gsm8k")
            train_frames.append(train_df)
            if test_df is not None:
                test_frames.append(test_df)

        if args.geo3k_dir:
            train_df, test_df = _load_quickstart_dir(args.geo3k_dir, data_source_override="hiyouga/geometry3k")
            train_frames.append(train_df)
            if test_df is not None:
                test_frames.append(test_df)

        if args.dapo_math_path:
            train_frames.append(_load_dapo_math_jsonl(args.dapo_math_path))

        if args.openr1mm_path:
            train_frames.append(_load_openr1mm_parquet(args.openr1mm_path))

    # --- Manifest mode ---
    if has_manifest:
        manifest_path = args.manifest
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            entries = json.load(f)

        if not isinstance(entries, list):
            raise ValueError("Manifest JSON must be a list of objects.")

        for entry in entries:
            path = entry["path"]
            data_source = entry.get("data_source")
            split = entry.get("split", "train")

            converted = _load_and_convert(path, data_source_override=data_source)

            if split == "train":
                train_frames.append(converted)
            elif split == "test":
                test_frames.append(converted)
            else:
                logger.warning("Unknown split '%s' for %s — treating as train.", split, path)
                train_frames.append(converted)

    # --- Merge ---
    if not train_frames:
        raise ValueError("No training data loaded. Check your inputs.")

    merged_train = _merge_and_shuffle(train_frames, seed=args.seed)
    merged_test = _merge_and_shuffle(test_frames, seed=args.seed) if test_frames else None

    # --- Optional train/test split (overrides existing splits) ---
    if args.test_ratio > 0:
        all_data = pd.concat([merged_train] + ([merged_test] if merged_test is not None else []), ignore_index=True)
        merged_train, merged_test = _train_test_split(all_data, args.test_ratio, args.seed)

    # --- Summary ---
    logger.info("=== MOPD Dataset Summary ===")
    logger.info("Train: %d rows", len(merged_train))
    for src, count in merged_train["data_source"].value_counts().items():
        logger.info("  %s: %d", src, count)
    if merged_test is not None:
        logger.info("Test: %d rows", len(merged_test))
        for src, count in merged_test["data_source"].value_counts().items():
            logger.info("  %s: %d", src, count)

    # --- Write ---
    os.makedirs(args.output_dir, exist_ok=True)
    _write_parquet(merged_train, os.path.join(args.output_dir, "train.parquet"))
    if merged_test is not None:
        _write_parquet(merged_test, os.path.join(args.output_dir, "test.parquet"))
        if args.small_eval_per_source > 0:
            small_df = _build_test_small(merged_test, args.small_eval_per_source, args.seed)
            _write_parquet(small_df, os.path.join(args.output_dir, "test_small.parquet"))
            logger.info("Test (small): %d rows", len(small_df))
            for src, count in small_df["data_source"].value_counts().items():
                logger.info("  %s: %d", src, count)

    logger.info("Done. Output written to %s", args.output_dir)


if __name__ == "__main__":
    main()
