# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare pinned public datasets for the Relax seq-cls SFT example."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


DATASETS = {
    "sst2": {
        "repo": "stanfordnlp/sst2",
        "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
        "config": None,
        "train_split": "train",
        "eval_split": "validation",
        "text_key": "sentence",
        "label_key": "label",
        "num_labels": 2,
        "problem_type": "single_label_classification",
    },
    "ag_news": {
        "repo": "fancyzhx/ag_news",
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "config": None,
        "train_split": "train",
        "eval_split": "test",
        "text_key": "text",
        "label_key": "label",
        "num_labels": 4,
        "problem_type": "single_label_classification",
    },
    "go_emotions": {
        "repo": "google-research-datasets/go_emotions",
        "revision": "add492243ff905527e67aeb8b80c082af02207c3",
        "config": "simplified",
        "train_split": "train",
        "eval_split": "validation",
        "text_key": "text",
        "label_key": "labels",
        "num_labels": 28,
        "problem_type": "multi_label_classification",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset", choices=["smoke", "extended", "full"], default="full")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _single_label_indices(rows, label_key: str, num_labels: int, count: int, seed: int) -> list[int]:
    by_label = {label: [] for label in range(num_labels)}
    for idx, row in enumerate(rows):
        label = int(row[label_key])
        if label in by_label:
            by_label[label].append(idx)
    rng = random.Random(seed)
    for indices in by_label.values():
        rng.shuffle(indices)

    selected: list[int] = []
    offsets = {label: 0 for label in by_label}
    while len(selected) < count:
        made_progress = False
        for label in range(num_labels):
            offset = offsets[label]
            if offset < len(by_label[label]):
                selected.append(by_label[label][offset])
                offsets[label] += 1
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            break
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} examples, but only found {len(selected)} valid labeled examples")
    return selected


def _multi_label_indices(rows, label_key: str, num_labels: int, count: int, seed: int) -> list[int]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    uncovered = set(range(num_labels))
    selected: list[int] = []
    selected_set: set[int] = set()

    for idx in indices:
        if len(selected) >= count:
            break
        labels = {int(label) for label in rows[idx][label_key]}
        if labels & uncovered:
            selected.append(idx)
            selected_set.add(idx)
            uncovered.difference_update(labels)
            if not uncovered:
                break
    if uncovered:
        raise RuntimeError(
            f"Could not cover all {num_labels} labels within {count} examples; missing {sorted(uncovered)}"
        )
    for idx in indices:
        if len(selected) == count:
            break
        if idx not in selected_set:
            selected.append(idx)
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} examples, but only found {len(selected)} valid labeled examples")
    return selected


def _select_indices(rows, spec: dict[str, Any], count: int, seed: int) -> list[int]:
    count = min(count, len(rows))
    if spec["problem_type"] == "single_label_classification":
        return _single_label_indices(rows, spec["label_key"], spec["num_labels"], count, seed)
    return _multi_label_indices(rows, spec["label_key"], spec["num_labels"], count, seed)


def _iter_relax_rows(rows, indices: Iterable[int], spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for idx in indices:
        row = rows[idx]
        label = row[spec["label_key"]]
        if hasattr(label, "tolist"):
            label = label.tolist()
        yield {
            "messages": [{"role": "user", "content": str(row[spec["text_key"]])}],
            "label": label,
            "source_id": str(row.get("idx", row.get("id", row.get("comment_id", idx)))),
        }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_one(name: str, args: argparse.Namespace) -> None:
    from datasets import load_dataset

    spec = DATASETS[name]
    load_kwargs = {"revision": spec["revision"]}
    if spec["config"] is not None:
        load_kwargs["name"] = spec["config"]
    train_rows = load_dataset(spec["repo"], split=spec["train_split"], **load_kwargs)
    eval_rows = load_dataset(spec["repo"], split=spec["eval_split"], **load_kwargs)

    if args.subset == "smoke":
        train_count = 2 * args.global_batch_size
        eval_count = args.global_batch_size + 1
    elif args.subset == "extended":
        train_count, eval_count = 2048, 512
    else:
        train_count, eval_count = len(train_rows), len(eval_rows)

    train_indices = _select_indices(train_rows, spec, train_count, args.seed)
    eval_indices = _select_indices(eval_rows, spec, eval_count, args.seed + 1)
    output_dir = args.output_dir / name / args.subset
    train_written = _write_jsonl(output_dir / "train.jsonl", _iter_relax_rows(train_rows, train_indices, spec))
    eval_written = _write_jsonl(output_dir / "validation.jsonl", _iter_relax_rows(eval_rows, eval_indices, spec))
    metadata = {
        "dataset": name,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "problem_type": spec["problem_type"],
        "num_labels": spec["num_labels"],
        "seed": args.seed,
        "subset": args.subset,
        "train_samples": train_written,
        "eval_samples": eval_written,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.global_batch_size < 1:
        raise ValueError("--global-batch-size must be positive")
    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        prepare_one(name, args)


if __name__ == "__main__":
    main()
