# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Resumeable OpenR1-MM evaluation through OpenAI-compatible multimodal
servers.

Use the SFT test.parquet (messages, images, source_id). Each problem receives
independent seeded samples. Only final <answer> content is scored against the
reference <answer>, never the reference or generated reasoning. Results retain
all generations; pass@k uses the unbiased combinatorial estimator.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import multiprocessing
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Solve the problem using the image. Put only the final answer inside <answer>...</answer>. "
    "For multiple-choice questions, give the option letter."
)
SCORER_VERSION = "final_answer_rule_v2"


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if len(matches) != 1 or text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def canonical_answer(text: str) -> tuple[str, str]:
    """Normalize explicit choices and complete math; keep prose/multi-value
    answers intact."""
    text = " ".join(text.strip().split())
    choices = re.findall(r"(?:option|choice|answer)\s*(?:is\s+)?(?:option\s+)?\(?([A-Z])\b", text)
    choices += re.findall(r"(?:选项|答案)(?:是|为)?\s*([A-Z])", text)
    if len(set(choices)) == 1:
        return "letter", choices[0]
    leading = re.match(r"^([A-Z])(?:[.:]\s|$)", text)
    if leading:
        return "letter", leading[1]
    text = re.sub(r"^(?:Therefore,?\s*)?(?:So\s+)?(?:the\s+)?(?:correct\s+)?answer\s+is\s+", "", text, flags=re.I)
    text = text.rstrip(".")
    text = text.replace("\\\\", "\\")
    latex = re.findall(r"\\\((.*?)\\\)|\$(.*?)\$", text)
    if len(latex) == 1:
        expr = next((part.strip() for part in latex[0] if part.strip()), "")
        outside = re.sub(r"\\\(.*?\\\)|\$.*?\$", "", text)
        if (re.search(r"\d|\\pi", expr) and not re.search(r"\d", outside)) or not outside.strip():
            return "math", expr
    text = re.sub(r"(?<=\d)\s*(?:°|degrees?|cm|km|mm|meters?|metres?|units?)$", "", text, flags=re.I)
    # Do not feed arbitrary prose to math_verify: it can extract an unrelated number.
    residue = re.sub(r"\\(?:frac|sqrt|pi|times|cdot|left|right|boxed)|\bpi\b", "", text)
    if re.search(r"\d|π|\\pi", text) and re.fullmatch(r"[\d\s.{}()\[\]+*/^=_%\\π√−-]+", residue or " "):
        return "math", text.replace("π", "\\pi").replace("√", "\\sqrt").replace("−", "-")
    return "text", text.casefold()


def score_answer(content: str, gold: str) -> dict[str, Any]:
    """Executed in a process main thread: math_verify timeouts use signals."""
    pred = extract_answer(content)
    if pred is None:
        return {"correct": False, "answer": None, "method": "missing_or_ambiguous_answer"}
    if " ".join(pred.split()) == " ".join(gold.split()):
        return {"correct": True, "answer": pred, "method": "exact"}
    gold_kind, gold_value = canonical_answer(gold)
    pred_kind, pred_value = canonical_answer(pred)
    if gold_kind == pred_kind and gold_value == pred_value:
        return {"correct": True, "answer": pred, "method": "normalized_" + gold_kind}
    if gold_kind == pred_kind == "math":
        from math_verify import LatexExtractionConfig, parse, verify

        config = [LatexExtractionConfig()]
        gold_expr = parse("$" + gold_value + "$", extraction_config=config, fallback_mode="no_fallback")
        pred_expr = parse("$" + pred_value + "$", extraction_config=config, fallback_mode="no_fallback")
        correct = bool(gold_expr and pred_expr and verify(gold_expr, pred_expr))
        return {"correct": correct, "answer": pred, "method": "math_verify"}
    return {"correct": False, "answer": pred, "method": "mismatch_" + gold_kind}


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"Invalid pass@k inputs: n={n}, c={c}, k={k}")
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def make_messages(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError("Expected exactly [user, assistant] in SFT data")
    gold = extract_answer(messages[-1]["content"])
    if gold is None:
        raise ValueError("Reference must contain exactly one nonempty <answer> block")
    parts = messages[0]["content"].split("<image>")
    if len(parts) - 1 != len(row["images"]):
        raise ValueError("Image markers and image count differ")
    content = []
    for index, part in enumerate(parts):
        if part.strip():
            content.append({"type": "text", "text": part})
        if index < len(row["images"]):
            url = row["images"][index]
            if not url.startswith(("data:image/", "http://", "https://")):
                raise ValueError("Images must be data URLs or HTTP(S) URLs")
            content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}], gold


def prepare_images(row: dict[str, Any], max_tokens: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse the training fetch_image RGB conversion and 14*2 pixel
    alignment."""
    from relax.utils.multimodal.config import MultimodalConfig
    from relax.utils.multimodal.image_utils import fetch_image

    urls, metadata = [], []
    for url in row["images"]:
        image = fetch_image(
            {"image": url}, image_patch_size=14, config=MultimodalConfig(image_max_token_num=max_tokens)
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw = buffer.getvalue()
        urls.append("data:image/png;base64," + base64.b64encode(raw).decode())
        metadata.append({"width": image.width, "height": image.height, "sha256": hashlib.sha256(raw).hexdigest()})
    return {**row, "images": urls}, metadata


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def summarize(records: dict[tuple[int, int], dict[str, Any]], ids: list[int], n: int, ks: list[int]) -> dict[str, Any]:
    questions = []
    for source_id in ids:
        samples = [records[(source_id, j)] for j in range(n) if (source_id, j) in records]
        correct = sum(int(s["score"]["correct"]) for s in samples)
        questions.append({"source_id": source_id, "n": len(samples), "correct": correct})
    complete = [q for q in questions if q["n"] == n]
    values = list(records.values())
    return {
        "status": "complete" if len(complete) == len(ids) else "partial",
        "scorer": SCORER_VERSION,
        "label_source": "SFT reference <answer>; noisy labels, not human-adjudicated accuracy",
        "expected_requests": len(ids) * n,
        "completed_requests": len(values),
        "completed_questions": len(complete),
        "total_questions": len(ids),
        "pass_at_k": {
            str(k): sum(pass_at_k(n, q["correct"], k) for q in complete) / len(complete) if complete else None
            for k in ks
        },
        "pass_at_k_denominator": "questions with all n samples completed",
        "sample_accuracy": sum(int(r["score"]["correct"]) for r in values) / len(values) if values else None,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in values)),
        "scoring_methods": dict(Counter(r["score"]["method"] for r in values)),
        "empty_final_answers": sum(not (r["content"] or "").strip() for r in values),
        "total_completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in values),
        "total_reasoning_tokens": sum(r["usage"].get("reasoning_tokens", 0) or 0 for r in values),
        "questions": questions,
    }


async def evaluate(args: argparse.Namespace) -> None:
    import aiohttp
    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    rows = pq.read_table(args.data).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    ids = [int(r["source_id"]) for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Dataset must contain nonempty, unique source_ids")
    prepared, image_metadata = [], []
    for row in rows:
        resized, metadata = prepare_images(row, args.image_max_token_num)
        prepared.append((int(row["source_id"]), *make_messages(resized)))
        image_metadata.append({"source_id": int(row["source_id"]), "images": metadata})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    config = {
        "dataset": str(Path(args.data).resolve()),
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "source_ids": ids,
        "model": args.model,
        "model_path": args.expected_model_path,
        "n": args.n,
        "ks": args.ks,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt": SYSTEM_PROMPT,
        "scorer": SCORER_VERSION,
        "image_preprocessing": {
            "method": "relax.fetch_image",
            "patch_size": 14,
            "image_max_token_num": args.image_max_token_num,
            "image_min_token_num": 4,
        },
    }
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume configuration differs; select a new output directory")
    write_json(config_path, config)
    write_json(output / "image_metadata.json", image_metadata)
    write_json(
        output / "references.json",
        [
            {"source_id": source_id, "gold": gold, "canonical": canonical_answer(gold)}
            for source_id, _, gold in prepared
        ],
    )
    records = {}
    for source_id in ids:
        for j in range(args.n):
            path = output / "samples" / f"{source_id}-{j:02d}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if (record["source_id"], record["sample_index"]) != (source_id, j):
                    raise ValueError(f"Invalid saved sample: {path}")
                records[(source_id, j)] = record
    write_json(output / "summary.json", summarize(records, ids, args.n, args.ks))
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    # Long generations can outlive the server's keep-alive window for idle
    # connections in the pool. Avoid recycling those sockets between requests.
    connector = aiohttp.TCPConnector(force_close=True, limit=args.concurrency * len(args.base_urls))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        servers = []
        for endpoint in args.base_urls:
            async with session.get(endpoint.rstrip("/") + "/get_server_info") as response:
                response.raise_for_status()
                info = await response.json()
            if str(Path(info["model_path"]).resolve()) != str(Path(args.expected_model_path).resolve()):
                raise ValueError(f"Unexpected model at {endpoint}: {info['model_path']}")
            if info["context_length"] <= args.max_tokens:
                raise ValueError(f"Insufficient context at {endpoint}: {info['context_length']}")
            servers.append(
                {
                    "endpoint": endpoint,
                    **{
                        k: info.get(k)
                        for k in (
                            "model_path",
                            "version",
                            "context_length",
                            "tp_size",
                            "max_running_requests",
                            "mem_fraction_static",
                            "quantization",
                            "reasoning_parser",
                            "random_seed",
                        )
                    },
                }
            )
        write_json(output / "servers.json", servers)
        queue = asyncio.Queue()
        # Keep one question's samples adjacent to make image/prompt caching useful.
        for source_id, messages, gold in prepared:
            for j in range(args.n):
                if (source_id, j) not in records:
                    queue.put_nowait((source_id, j, messages, gold))
        started = time.monotonic()
        initial_count = len(records)
        errors = []
        logger.info(
            "EVAL_START questions=%d n=%d pending=%d endpoints=%d", len(rows), args.n, queue.qsize(), len(servers)
        )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=context) as pool:

            async def worker(endpoint: str) -> None:
                while not queue.empty():
                    source_id, j, messages, gold = queue.get_nowait()
                    seed = (args.seed + source_id * args.n + j) % 2147483647
                    body = {
                        "model": args.model,
                        "messages": messages,
                        "n": 1,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "reasoning_effort": args.reasoning_effort,
                        "seed": seed,
                    }
                    request_started = time.monotonic()
                    for attempt in range(args.retries + 1):
                        try:
                            url = endpoint.rstrip("/") + "/v1/chat/completions"
                            async with session.post(url, json=body) as response:
                                if response.status != 200:
                                    raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1000]}")
                                result = await response.json()
                            choice = result["choices"][0]
                            if choice["finish_reason"] not in ("stop", "length"):
                                raise ValueError(f"Unexpected finish reason: {choice['finish_reason']}")
                            content = choice["message"].get("content") or ""
                            score = await asyncio.get_running_loop().run_in_executor(pool, score_answer, content, gold)
                            record = {
                                "source_id": source_id,
                                "sample_index": j,
                                "seed": seed,
                                "gold": gold,
                                "content": content,
                                "reasoning_content": choice["message"].get("reasoning_content"),
                                "finish_reason": choice["finish_reason"],
                                "usage": result.get("usage", {}),
                                "score": score,
                                "endpoint": endpoint,
                                "response_id": result.get("id"),
                                "elapsed_seconds": time.monotonic() - request_started,
                                "attempts": attempt + 1,
                            }
                            write_json(output / "samples" / f"{source_id}-{j:02d}.json", record)
                            records[(source_id, j)] = record
                            if len(records) % 16 == 0 or queue.empty():
                                summary = summarize(records, ids, args.n, args.ks)
                                summary["run_elapsed_seconds"] = time.monotonic() - started
                                write_json(output / "summary.json", summary)
                                logger.info(
                                    "EVAL_PROGRESS completed=%d/%d questions=%d rate=%.2f/s finishes=%s",
                                    len(records),
                                    len(ids) * args.n,
                                    summary["completed_questions"],
                                    (len(records) - initial_count) / max(time.monotonic() - started, 1),
                                    summary["finish_reasons"],
                                )
                            break
                        except Exception as error:
                            logger.warning(
                                "REQUEST_FAILED id=%d sample=%d attempt=%d error=%s", source_id, j, attempt, error
                            )
                            if attempt == args.retries:
                                errors.append({"source_id": source_id, "sample_index": j, "error": str(error)})
                                break
                            await asyncio.sleep(min(2**attempt, 10))
                    queue.task_done()

            await asyncio.gather(*(worker(endpoint) for endpoint in args.base_urls for _ in range(args.concurrency)))
        summary = summarize(records, ids, args.n, args.ks)
        summary["run_elapsed_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", summary)
        write_json(output / "errors.json", errors)
        logger.info(
            "EVAL_FINISHED status=%s completed=%d pass_at_k=%s", summary["status"], len(records), summary["pass_at_k"]
        )
        if errors or summary["status"] != "complete":
            raise RuntimeError("Evaluation incomplete; inspect errors.json and rerun the same configuration to resume")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-urls", nargs="+", required=True, help="SGLang roots without /v1")
    parser.add_argument("--model", default="kimi-k3-export")
    parser.add_argument("--expected-model-path", required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"], default="high")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--image-max-token-num", type=int, default=1024, help="Matches the OpenR1-MM SFT script")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--concurrency", type=int, default=16, help="Requests per endpoint")
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete split")
    args = parser.parse_args()
    if args.n < max(args.ks) or min(args.ks) < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("Require n >= all positive k, temperature > 0 and 0 < top_p <= 1")
    if (
        min(args.concurrency, args.score_workers, args.max_tokens, args.request_timeout) < 1
        or args.limit < 0
        or args.retries < 0
        or args.image_max_token_num < 4
    ):
        parser.error("Invalid concurrency, worker count, token budget, timeout, limit or retries")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
