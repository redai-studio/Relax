# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare author-protocol DAPO training JSONL without executing code."""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from examples.retool_agentic.protocol import PREFIX, VERSION
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
LEAD = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
TAIL = '\n\nRemember to put your answer on its own line after "Answer:".'


def make_case(question: str, label: str, **metadata: object) -> dict:
    if not question.strip() or not str(label).strip():
        raise ValueError("Require the original question and gold answer")
    return dict(
        prompt=[dict(role="user", content=PREFIX + question)],
        label=str(label),
        metadata=dict(protocol_id=VERSION, question=question, **metadata),
    )


def deduplicate(rows: Iterable[dict]) -> tuple[list[dict], dict]:
    unique, counts = {}, collections.Counter()
    for row in rows:
        uid = row["extra_info"]["index"]
        prompt, reward = row["prompt"], row["reward_model"]
        if row["data_source"] != "math_dapo" or len(prompt) != 1 or prompt[0]["role"] != "user":
            raise ValueError("Unexpected DAPO source schema")
        payload = (prompt[0]["content"], reward["ground_truth"], reward["style"])
        if uid in unique and unique[uid] != payload:
            raise ValueError(f"Conflicting repeated payload for native ID {uid}")
        unique[uid] = payload
        counts[uid] += 1
    by_question = collections.defaultdict(list)
    for uid, (question, gold, _) in unique.items():
        by_question[question].append((uid, gold))
    conflicts = {uid for group in by_question.values() if len({gold for _, gold in group}) > 1 for uid, _ in group}
    cases, quarantine = [], []
    for uid, (published, gold, style) in unique.items():
        if uid in conflicts:
            quarantine.append(dict(source_id=uid, published_user_prompt=published, original_gold=gold))
            continue
        if not published.startswith(LEAD) or not published.endswith(TAIL) or not re.fullmatch(r"-?\d+", gold):
            raise ValueError("Unexpected DAPO question envelope or gold")
        cases.append(
            make_case(
                published[len(LEAD) : -len(TAIL)],
                gold,
                source_id=uid,
                train_index=len(cases),
                published_user_prompt=published,
                reward_style=style,
            )
        )
    return cases, dict(
        published_rows=sum(counts.values()),
        original_unique_native_ids=len(unique),
        unique_native_ids=len(cases),
        repeat_histogram={str(k): v for k, v in collections.Counter(counts.values()).items()},
        quarantine_records=quarantine,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Never overwrite prepared data")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    parquet = pq.ParquetFile(args.input)
    rows = (row for batch in parquet.iter_batches(batch_size=8192) for row in batch.to_pylist())
    cases, report = deduplicate(rows)
    if (
        report["published_rows"],
        report["original_unique_native_ids"],
        len(cases),
        len(report["quarantine_records"]),
    ) != (1791700, 17917, 17903, 14) or report["repeat_histogram"] != {"100": 17917}:
        raise ValueError("Require the full pinned DAPO source and the same 14 conflicting native IDs")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for case in cases:
        ids = tokenizer.apply_chat_template(case["prompt"], tokenize=True, add_generation_prompt=True)
        if len(ids) > 2048:
            raise ValueError("Prompt exceeds 2048 tokens; no silent filtering or truncation")
    encoded = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(encoded)
    report.update(source=str(args.input.resolve()), sha256=hashlib.sha256(encoded.encode()).hexdigest())
    args.out.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    logger.info("Prepared %s cases at %s", len(cases), args.out)


if __name__ == "__main__":
    main()
