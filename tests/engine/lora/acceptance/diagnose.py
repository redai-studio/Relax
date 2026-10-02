# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay a failed Session on ordinary A-only engines, without Ray or
publication."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from uuid import uuid4

import httpx

from relax.engine.lora.snapshot import fingerprint_model, read_snapshot
from relax.utils.logging_utils import get_logger

from .assertions import compare
from .processes import baseline, inference_profile, select_gpus


logger = get_logger(__name__)


def failed_task(source: dict) -> dict:
    tasks = source.get("tasks", {"single": source})
    candidates = [task for task in tasks.values() if any(row["status"] == "FAIL" for row in task.get("numerical", []))]
    if len(candidates) != 1:
        raise ValueError("expected exactly one task with a failed numerical sample")
    return candidates[0]


def comparison(actual: dict, expected: dict, atol: float, rtol: float) -> dict:
    result = {"actual": actual, "expected": expected, "atol": atol, "rtol": rtol}
    try:
        result.update(status="PASS", max_error=compare(actual, expected, atol=atol, rtol=rtol))
    except AssertionError as error:
        result.update(status="FAIL", error=str(error))
    return result


def replay_batch_sizes(
    client: httpx.Client, adapter: str, sample: dict, recorded: dict, sizes: list[int], evidence: dict
) -> None:
    """Vary submitted concurrency on identical cold prefixes in ordinary A.

    This isolates request-count sensitivity, not the original mixed workload.
    Submitted concurrency is not evidence of the scheduler's actual batch size.
    """
    attempt = next(row for row in sample["metadata"]["lora_attempts"] if row["token_end"] > row["token_start"])
    start, end = attempt["token_start"], attempt["token_end"]
    tokens = sample["tokens"]
    if not 0 < start < end <= len(tokens):
        raise AssertionError("invalid recorded attempt span")
    reference = None
    evidence.update(concurrent_replays=[], scheduler_batch_size="NOT_OBSERVED")

    def compare_prefix(output: dict, expected_tokens: list[int], expected_scores: list[float]) -> dict:
        generated = output["output_ids"]
        common = next(
            (i for i, pair in enumerate(zip(generated, expected_tokens)) if pair[0] != pair[1]),
            min(len(generated), len(expected_tokens)),
        )
        return {
            "tokens_equal": generated == expected_tokens,
            "matching_prefix_tokens": common,
            "scores": comparison(
                {i: output["meta_info"]["output_token_logprobs"][i][0] for i in range(common)},
                {i: expected_scores[i] for i in range(common)},
                recorded["atol"],
                recorded["rtol"],
            ),
        }

    for size in dict.fromkeys([1, *sizes]):
        for repeat in range(2):
            body = {
                "input_ids": [tokens[:start] for _ in range(size)],
                "extra_key": ["no7-batch-" + uuid4().hex for _ in range(size)],
                "lora_path": adapter,
                "return_logprob": True,
                "logprob_start_len": -1,
                "sampling_params": {"temperature": 0, "max_new_tokens": end - start, "ignore_eos": True},
            }
            entry = {"requested_concurrency": size, "repeat": repeat, "request": body, "comparisons": []}
            evidence["concurrent_replays"].append(entry)
            response = client.post("/generate", json=body)
            response.raise_for_status()
            outputs = response.json()
            entry["responses"] = outputs
            if not isinstance(outputs, list) or len(outputs) != size:
                raise AssertionError("native batch response count differs from submitted concurrency")
            for output in outputs:
                scores = output["meta_info"]["output_token_logprobs"]
                if len(output["output_ids"]) != end - start or [row[1] for row in scores] != output["output_ids"]:
                    raise AssertionError("batch output token/logprob alignment changed")
                if output["meta_info"].get("cached_tokens") != 0:
                    raise AssertionError("concurrency diagnostic unexpectedly reused a cached prefix")
                if reference is None:
                    reference = output
                entry["comparisons"].append(
                    {
                        "vs_serial": compare_prefix(
                            output,
                            reference["output_ids"],
                            [row[0] for row in reference["meta_info"]["output_token_logprobs"]],
                        ),
                        "vs_recorded_session": compare_prefix(
                            output, tokens[start:end], [recorded["actual"][str(i)] for i in range(start, end)]
                        ),
                    }
                )
            entry["matches_serial"] = all(
                item["vs_serial"]["tokens_equal"] and item["vs_serial"]["scores"]["status"] == "PASS"
                for item in entry["comparisons"]
            )
            logger.info(
                "GPU %s, deterministic=%s, submitted=%s, repeat=%s: matches serial=%s",
                evidence["gpu"]["index"],
                evidence.get("deterministic", False),
                size,
                repeat,
                entry["matches_serial"],
            )


def replay(
    client: httpx.Client,
    adapter: str,
    sample: dict,
    recorded: dict,
    atol: float,
    rtol: float,
    evidence: dict,
    *,
    cold: bool = True,
) -> None:
    """Compare decode with decode and independently check the bulk-prefill
    oracle.

    Only score recorded output while the replay has the same conditioning
    prefix. Divergent continuations are retained and explicitly fail token
    equality. Warm replay sends only the original turns, without scoring probes
    that would populate the prefix cache ahead of those turns.
    """
    tokens = sample["tokens"]
    evidence["turns"] = []

    def generate(ids: list[int], count: int, *, prefill: bool = False) -> dict:
        response = client.post(
            "/generate",
            json={
                "input_ids": ids,
                "lora_path": adapter,
                "return_logprob": True,
                "logprob_start_len": 0 if prefill else -1,
                "sampling_params": {"temperature": 0, "max_new_tokens": count},
            },
        )
        response.raise_for_status()
        return response.json()

    if cold:
        bulk = generate(tokens, 1, prefill=True)
        evidence["original_sequence_prefill"] = bulk
        entries = bulk["meta_info"]["input_token_logprobs"]
        if len(entries) != len(tokens) or any(entry[1] != token for entry, token in zip(entries, tokens, strict=True)):
            raise AssertionError("baseline input score alignment changed")
        expected = {int(key): float(entries[int(key)][0]) for key in recorded["actual"]}
        evidence["original_baseline_reproduced"] = comparison(
            expected, {int(key): value for key, value in recorded["baseline"].items()}, atol, rtol
        )
    for attempt in sample["metadata"]["lora_attempts"]:
        start, end = attempt["token_start"], attempt["token_end"]
        if not 0 < start < end <= len(tokens):
            raise AssertionError("invalid recorded attempt span")
        prefix, continuation = tokens[:start], tokens[start:end]
        turn = {"attempt": attempt, "prefix": prefix, "recorded_tokens": continuation}
        evidence["turns"].append(turn)
        generated = generate(prefix, len(continuation))
        turn["decode"] = generated
        turn["cached_tokens"] = generated["meta_info"].get("cached_tokens")
        decoded_tokens = generated["output_ids"]
        output_scores = generated["meta_info"]["output_token_logprobs"]
        if not decoded_tokens or [entry[1] for entry in output_scores] != decoded_tokens:
            raise AssertionError("baseline decode score alignment changed")
        own = {i: float(entry[0]) for i, entry in enumerate(output_scores)}
        turn["tokens_equal"] = decoded_tokens == continuation
        common = 0
        for left, right in zip(decoded_tokens, continuation):
            if left != right:
                break
            common += 1
        turn["matching_prefix_tokens"] = common
        turn["recorded_vs_independent_decode"] = comparison(
            {i: recorded["actual"][str(start + i)] for i in range(common)},
            {i: own[i] for i in range(common)},
            atol,
            rtol,
        )
        if cold:
            rescored = generate(prefix + decoded_tokens, 1, prefill=True)
            turn["own_sequence_prefill"] = rescored
            inputs = rescored["meta_info"]["input_token_logprobs"]
            if (
                len(inputs) != len(prefix) + len(decoded_tokens)
                or [row[1] for row in inputs] != prefix + decoded_tokens
            ):
                raise AssertionError("self-prefill score alignment changed")
            turn["independent_decode_vs_own_prefill"] = comparison(
                own, {i: float(inputs[start + i][0]) for i in range(len(decoded_tokens))}, atol, rtol
            )
        logger.info(
            "GPU %s, tokens %s:%s: equal=%s; recorded/decode=%s; decode/prefill=%s; cached=%s",
            evidence["gpu"]["index"],
            start,
            end,
            turn["tokens_equal"],
            turn["recorded_vs_independent_decode"]["status"],
            turn.get("independent_decode_vs_own_prefill", {}).get("status", "NOT_RUN"),
            turn["cached_tokens"],
        )
    if not cold:
        evidence["cache_reuse_observed"] = any((turn["cached_tokens"] or 0) > 0 for turn in evidence["turns"][1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--gpus", required=True, nargs=2)
    parser.add_argument("--output", required=True, type=Path, help="New diagnostic output directory")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        help="Ordinary-engine concurrency sweep, each size twice; uses isolated prefixes with radix enabled",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Diagnostic only: use deterministic inference with Triton attention, preserving graphs and radix",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm"),
        default="cold",
        help="cold: decode versus prefill; warm: consecutive original turns with prefix caching",
    )
    args = parser.parse_args()
    if args.batch_sizes and any(size < 1 or size > 32 for size in args.batch_sizes):
        parser.error("batch sizes must be between 1 and the engine limit of 32")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "INCOMPLETE", "acceptance_status": "NOT_EVALUATED", "processes": [], "engines": []}

    def interrupted(signum: int, frame) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        source = json.loads(args.report.read_text())
        task = failed_task(source)
        failed = next(row for row in task["numerical"] if row["status"] == "FAIL")
        if not args.batch_sizes and args.cache_mode == "cold" and "baseline" not in failed:
            raise ValueError("no completed prefill reference in this report; use --cache-mode warm or --batch-sizes")
        sample = task["samples"][failed["sample_index"]]
        profile = task["profile"]
        artifact = json.loads(Path(profile["artifacts_file"]).read_text())["versions"][failed["version"]]
        snapshot = read_snapshot(artifact["path"])
        if snapshot.digest != sample["metadata"]["lora_adapter"]["digest"]:
            raise ValueError("diagnostic adapter differs from the failed Session")
        if fingerprint_model(profile["model_path"]) != snapshot.base_model_digest:
            raise ValueError("diagnostic base differs from the recorded adapter contract")
        result.update(
            source_report=str(args.report.resolve()),
            sample=sample,
            recorded=failed,
            model=profile["model_path"],
            digest=snapshot.digest,
        )
        for gpu in select_gpus(args.gpus):
            cold = args.cache_mode == "cold" and not args.batch_sizes
            evidence = {"gpu": gpu, "cache_mode": "cold" if cold else "warm", "deterministic": args.deterministic}
            result["engines"].append(evidence)
            with baseline(
                profile["model_path"],
                str(snapshot.path),
                failed["version"],
                gpu["uuid"],
                args.output / f"baseline-{gpu['index']}.log",
                cold,
                result["processes"],
                deterministic=args.deterministic,
            ) as engine:
                with httpx.Client(base_url=engine["url"], timeout=120, trust_env=False) as client:
                    response = client.get("/server_info")
                    response.raise_for_status()
                    evidence["server_info"] = response.json()
                    inference_profile(evidence["server_info"], deterministic=args.deterministic, cold=cold)
                    if args.batch_sizes:
                        replay_batch_sizes(client, engine["lora_path"], sample, failed, args.batch_sizes, evidence)
                        continue
                    replay(
                        client,
                        engine["lora_path"],
                        sample,
                        failed,
                        failed["atol"],
                        failed["rtol"],
                        evidence,
                        cold=cold,
                    )
        result["status"] = "COMPLETE"
    except BaseException as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        path = args.output / "report.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        logger.info("Diagnostic %s: %s; original acceptance result is unchanged", result["status"], path)


if __name__ == "__main__":
    main()
