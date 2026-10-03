# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Create deterministic Task 31 train/eval preference subsets and a provenance
manifest."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from relax.engine.sft.dataset.sample import VALID_ROLES


DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
DATASET_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
ORDER_NAMESPACE = "task31-ultrafeedback-v1:"
REJECTION_REASON_CODES = (
    "schema",
    "identical",
    "post_truncation",
    "prompt_mismatch",
    "empty_completion",
    "oversize",
)


def _reason_code(error: BaseException) -> str:
    message = str(error).lower()
    if "identical" in message:
        return "identical"
    if "strict shared prompt" in message:
        return "prompt_mismatch"
    return "schema"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_row(row: dict[str, Any], *, split: str, index: int) -> str:
    prompt_id = row.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError(f"{split}[{index}] has no non-empty prompt_id")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list) or not chosen or not rejected:
        raise ValueError(f"{split}[{index}] {prompt_id} has invalid chosen/rejected messages")
    for branch_name, messages in (("chosen", chosen), ("rejected", rejected)):
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"{split}[{index}] {prompt_id} {branch_name}[{message_index}] must be a message object"
                )
            role = message.get("role")
            content = message.get("content")
            if role not in VALID_ROLES:
                raise ValueError(
                    f"{split}[{index}] {prompt_id} {branch_name}[{message_index}] has invalid role {role!r}"
                )
            if not isinstance(content, str):
                raise ValueError(
                    f"{split}[{index}] {prompt_id} {branch_name}[{message_index}] content must be a string"
                )
    if chosen == rejected:
        raise ValueError(f"{split}[{index}] {prompt_id} has identical chosen/rejected branches")
    if chosen[-1].get("role") != "assistant" or rejected[-1].get("role") != "assistant":
        raise ValueError(f"{split}[{index}] {prompt_id} must end both branches with assistant")
    if chosen[:-1] != rejected[:-1]:
        raise ValueError(f"{split}[{index}] {prompt_id} does not have a strict shared prompt")
    return prompt_id


def _select(dataset, *, split: str, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[str] = set()
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if not isinstance(row, dict):
            rejected.append(
                {
                    "prompt_id": None,
                    "source_index": index,
                    "reason_code": "schema",
                    "reason": f"{split}[{index}] must be an object",
                }
            )
            continue
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"{split}[{index}] has no non-empty prompt_id")
        if prompt_id in seen:
            rejected.append(
                {
                    "prompt_id": prompt_id,
                    "source_index": index,
                    "reason_code": "schema",
                    "reason": f"duplicate prompt_id in {split}; retained first source occurrence",
                }
            )
            continue
        seen.add(prompt_id)
        order_key = hashlib.sha256(f"{ORDER_NAMESPACE}{prompt_id}".encode()).hexdigest()
        candidates.append((order_key, prompt_id, {**row, "_source_index": index}))
    if len(candidates) < count:
        raise ValueError(f"{split} contains {len(candidates)} valid rows, need {count}")
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[dict[str, Any]] = []
    for _, prompt_id, row in candidates:
        source_index = row.pop("_source_index")
        try:
            _validate_row(row, split=split, index=source_index)
        except ValueError as exc:
            rejected.append(
                {
                    "prompt_id": prompt_id,
                    "source_index": source_index,
                    "reason_code": _reason_code(exc),
                    "reason": str(exc),
                }
            )
            continue
        selected.append(
            {
                "prompt_id": prompt_id,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "metadata": {
                    "source_split": split,
                    "score_chosen": row.get("score_chosen"),
                    "score_rejected": row.get("score_rejected"),
                },
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"{split} contains only {len(selected)} valid rows after deterministic validation")
    return selected, rejected


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=512)
    args = parser.parse_args()
    if args.train_size <= 0 or args.eval_size <= 0:
        parser.error("subset sizes must be positive")

    from datasets import Dataset, load_dataset

    train_source = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train_prefs")
    eval_source = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="test_prefs")
    train_rows, train_rejections = _select(train_source, split="train_prefs", count=args.train_size)
    eval_rows, eval_rejections = _select(eval_source, split="test_prefs", count=args.eval_size)
    train_ids = {row["prompt_id"] for row in train_rows}
    eval_ids = {row["prompt_id"] for row in eval_rows}
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(f"train/eval prompt_id overlap: {sorted(overlap)[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for name, rows in (("train", train_rows), ("eval", eval_rows)):
        jsonl_path = args.output_dir / f"ultrafeedback_{name}.jsonl"
        parquet_path = args.output_dir / f"ultrafeedback_{name}.parquet"
        _write_jsonl(jsonl_path, rows)
        Dataset.from_list(rows).to_parquet(parquet_path)
        outputs[name] = {
            "count": len(rows),
            "prompt_ids": [row["prompt_id"] for row in rows],
            "jsonl": {"path": jsonl_path.name, "sha256": _sha256(jsonl_path)},
            "parquet": {"path": parquet_path.name, "sha256": _sha256(parquet_path)},
        }

    manifest = {
        "schema_version": 1,
        "source": {"dataset": DATASET_ID, "revision": DATASET_REVISION},
        "selection": {
            "algorithm": f'sha256("{ORDER_NAMESPACE}" + prompt_id), then first valid rows',
            "overlap_count": 0,
            "rejections": {
                "train": train_rejections,
                "eval": eval_rejections,
            },
            "rejection_counts": {
                split: {
                    reason_code: sum(item["reason_code"] == reason_code for item in rejections)
                    for reason_code in REJECTION_REASON_CODES
                }
                for split, rejections in (("train", train_rejections), ("eval", eval_rejections))
            },
        },
        "schema": {
            "prompt_id": "string",
            "chosen": "list<{role:string,content:string}>",
            "rejected": "list<{role:string,content:string}>",
            "metadata": "object",
        },
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "sha256": _sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
