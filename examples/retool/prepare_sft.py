# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Preserve all author text; mask stored interpreter spans without executing
code."""

import argparse
import json
import os
import re
from pathlib import Path

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def observation_spans(text: str) -> tuple[list[tuple[int, int]], list[dict]]:
    spans, notes = [], []
    opened = None
    tags = list(re.finditer(r"</?(?:interpreter|code|answer)>", text))
    for tag in tags:
        value = tag.group()
        if opened is not None:
            if value == "</interpreter>":
                spans.append((opened, tag.end()))
                opened = None
                continue
            if value in {"<interpreter>", "<code>", "</code>", "<answer>"}:
                end = tag.end() if value == "</code>" else tag.start()
                spans.append((opened, end))
                notes.append(
                    {
                        "kind": "author_unclosed_interpreter_conservative_mask",
                        "start": opened,
                        "end": end,
                        "boundary": value,
                    }
                )
                opened = None
        if value == "<interpreter>":
            opened = tag.start()
        elif value == "</interpreter>":
            spans.append((tag.start(), tag.end()))
            notes.append({"kind": "author_stray_interpreter_delimiter_masked", "start": tag.start(), "end": tag.end()})
    if opened is not None:
        spans.append((opened, len(text)))
        notes.append({"kind": "author_unclosed_interpreter_at_end", "start": opened, "end": len(text)})
    # One original source typo closes a code cell with </interpreter>, then has
    # a numeric stored return and another close. Preserve text; mask that return.
    for match in re.finditer(r"</interpreter>\s*[+-]?\d+(?:\.\d+)?\s*</interpreter>", text):
        spans.append((match.start(), match.end()))
        notes.append(
            {"kind": "author_numeric_return_between_stray_closers", "start": match.start(), "end": match.end()}
        )
    return spans, notes


def verify_author_download(path: Path) -> None:
    metadata = path.parent / ".cache/huggingface/download" / (path.name + ".metadata")
    if not metadata.is_file() or metadata.read_text().splitlines()[0] != "74943ce52f389f16926702302fcff6255875cbb2":
        raise ValueError("Require the pinned author download with its HF revision metadata")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify_author_download(args.parquet)
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from relax.engine.sft.dataset.chat_template import render_to_text, render_with_loss_mask
    from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample

    source = Path(__file__).resolve().parents[2]
    output = args.out.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("Prepared data must stay outside the repository")
    contract = dict(student_path=str(args.model), sft_data_revision="74943ce52f389f16926702302fcff6255875cbb2")
    rows = pq.read_table(args.parquet).to_pylist()
    assert len(rows) == 2000
    tokenizer = AutoTokenizer.from_pretrained(contract["student_path"], local_files_only=True, trust_remote_code=False)
    assert len(tokenizer) == 151665 and tokenizer.eos_token_id == 151645
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + "_preparing")
    assert not output.exists() and not staging.exists(), "Never overwrite published or partial data"
    staging.mkdir()
    records, anomalies, lengths = [], [], []
    token_offset = mask_offset = loss_total = observation_tokens = 0
    duplicates = set()
    with (staging / "tokens.u32").open("xb") as token_stream, (staging / "mask.u8").open("xb") as mask_stream:
        for index, row in enumerate(rows):
            messages = row["messages"]
            assert [m["role"] for m in messages] == ["user", "assistant"]
            identity = tuple((m["role"], m["content"]) for m in messages)
            duplicates.add(identity)
            sample = CanonicalSample(
                [CanonicalMessage(m["role"], m["content"], m["role"] == "assistant") for m in messages],
                {"source_dataset": "JoeYing/ReTool-SFT", "row_index": index},
            )
            tokens, mask = render_with_loss_mask(sample, tokenizer=tokenizer)
            text = render_to_text(sample, tokenizer=tokenizer)
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            assert tokens.tolist() == encoded["input_ids"]
            body = messages[-1]["content"]
            start = text.find(body)
            assert start >= 0 and text[start : start + len(body)] == body
            spans, notes = observation_spans(body)
            for i, (a, b) in enumerate(encoded["offset_mapping"]):
                if b > a and any(a < start + end and b > start + begin for begin, end in spans):
                    if mask[i]:
                        observation_tokens += 1
                    mask[i] = 0
            assert len(tokens) <= 32768, f"Author row{index} exceeds model native context"
            assert 0 <= int(tokens.min()) and int(tokens.max()) < len(tokenizer)
            assert int(mask.sum()) > 0 and int(mask[0]) == 0
            assert int(mask[-2]) == 1 or int(mask[-1]) == 1
            tokens.numpy().astype("<u4").tofile(token_stream)
            mask.numpy().astype(np.uint8).tofile(mask_stream)
            eos_count = int(((tokens == tokenizer.eos_token_id) & (mask == 1)).sum())
            assert eos_count == 1
            records.append(
                {
                    "author_row_index": index,
                    "source_dataset": "JoeYing/ReTool-SFT",
                    "source_revision": contract["sft_data_revision"],
                    "tokens_path": str(output / "tokens.u32"),
                    "mask_path": str(output / "mask.u8"),
                    "token_offset": token_offset,
                    "mask_offset": mask_offset,
                    "positions": len(tokens),
                    "assistant_loss_tokens": int(mask.sum()),
                    "eos_token_id": tokenizer.eos_token_id,
                    "supervised_assistant_EOS": eos_count,
                }
            )
            token_offset += len(tokens) * 4
            mask_offset += len(tokens)
            loss_total += int(mask.sum())
            lengths.append(len(tokens))
            if notes:
                anomalies.append({"author_row_index": index, "notes": notes, "raw_text_modified": False})
    (staging / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    report = {
        "status": "AUTHOR_FULL2000_RAW_TEXT_PRESERVED_MASKED_NOT_TRAINED",
        "source": "JoeYing/ReTool-SFT",
        "revision": contract["sft_data_revision"],
        "rows": len(rows),
        "distinct_full_records": len(duplicates),
        "positions": mask_offset,
        "supervised_tokens": loss_total,
        "explicit_observation_tokens_excluded": observation_tokens,
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "over16384": sum(n > 16384 for n in lengths),
        "author_tag_anomaly_notes": anomalies,
        "author_questions_and_solutions_text_preserved": True,
        "self_synthesized_rows": 0,
        "code_or_environment_history_executions": 0,
        "math_rescoring": 0,
        "format_note": "Preserve author code/interpreter/answer syntax; original malformed delimiters kept, conservative observation masks noted, no textual repair or fabricated output",
        "schema": "immutable_u32_tokens_u8_lossmask_nativeRelaxSFT",
    }
    (staging / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    os.rename(staging, output)
    logger.info("%s", {k: v for k, v in report.items() if k != "author_tag_anomaly_notes"})
    logger.info("Author tag anomaly rows: %s", [x["author_row_index"] for x in anomalies])


if __name__ == "__main__":
    main()
