# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run the pinned OpenSearch tool agent against an OpenAI-compatible model."""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

from common import normalized_text, write_json


UPSTREAM_REVISION = "1ac26e2c35a3e813e022813c88888db457d88f7d"


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def question_hash(question: str) -> str:
    return sha256(normalized_text(question.replace("<image>", "")).encode())


def image_hash(payload: bytes) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
        return sha256(str(image.size).encode() + image.tobytes())


def read_cases(path: Path, dataset: str) -> list[dict]:
    import pyarrow.parquet as parquet

    rows = parquet.read_table(path).to_pylist()
    cases = []
    for row in rows:
        if dataset == "simplevqa":
            if row["language"] != "EN":
                continue
            identifier, images = row["data_id"], [base64.b64decode(row["image"])]
        else:
            identifier = row["id"]
            images = [base64.b64decode(item["data"]) for item in json.loads(row["image_packed"])]
        cases.append({"id": str(identifier), "question": row["question"], "answer": row["answer"], "images": images})
    expected = 1013 if dataset == "simplevqa" else 500
    if len(cases) != expected or len({case["id"] for case in cases}) != expected:
        raise ValueError(f"Expected {expected} unique {dataset} cases; found {len(cases)}")
    return cases


def missing_tools() -> list[str]:
    required = [
        "QWEN_API_BASE",
        "QWEN_MODEL_NAME",
        "SERPAPI_URL",
        "SERPAPI_API_KEY",
        "IMAGE_UPLOAD_URL",
        "LAYOUT_PARSING_API_URL",
        "SR_MODEL_PATH",
    ]
    if not all(os.environ.get(name) for name in ["API_HOST", "API_USER", "API_KEY"]):
        required.extend(["SERPER_API_KEY", "JINA_API_KEY"])
    missing = [name for name in required if not os.environ.get(name)]
    if os.environ.get("SR_MODEL_PATH") and not Path(os.environ["SR_MODEL_PATH"]).is_file():
        missing.append("SR_MODEL_PATH (file missing)")
    return missing


def preserve_history_template(processor) -> str:
    from relax.engine.sft.dataset.qwen_chat_template_patch import try_patch_qwen_chat_template
    from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample

    sample = CanonicalSample(
        messages=[CanonicalMessage(role="assistant", content="", learn=False)],
        metadata={"source_dataset": "benchmark", "row_index": 0},
    )
    patched = try_patch_qwen_chat_template(sample, processor.chat_template, {"preserve_thinking": True})
    if patched is None:
        raise ValueError("Expected a supported Qwen history template")
    processor.chat_template = patched.template
    processor.tokenizer.chat_template = patched.template
    return patched.template


class HTTPRunner:
    def __init__(self, processor_path: Path, client, model: str) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        self.template_sha256 = sha256(preserve_history_template(self.processor).encode())
        self.client, self.model = client, model
        self.requests: list[dict] = []
        self.error: str | None = None

    def infer(self, contents, system_instruction=None, cfg=None) -> dict:
        from PIL import Image

        hf_messages = [{"role": "system", "content": system_instruction}]
        messages = [{"role": "system", "content": system_instruction}]
        for item in contents:
            parts, hf_parts = [], []
            for part in item.get("parts", []):
                if "text" in part:
                    parts.append({"type": "text", "text": part["text"]})
                    hf_parts.append(parts[-1])
                else:
                    if "inline_data" in part:
                        encoded = part["inline_data"]["data"]
                        payload = base64.b64decode(encoded.split("base64,", 1)[-1], validate=True)
                    elif "image_url" in part:
                        payload = url_image_bytes(part["image_url"]["url"])
                    else:
                        raise ValueError("Unsupported model-visible content part")
                    with Image.open(io.BytesIO(payload)) as image:
                        mime = Image.MIME[image.format]
                        hf_parts.append({"type": "image", "image": image.convert("RGB")})
                    encoded = base64.b64encode(payload).decode()
                    parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            role = "assistant" if item["role"] == "model" else "user"
            hf_messages.append({"role": role, "content": hf_parts})
            messages.append({"role": role, "content": parts})
        inputs = self.processor.apply_chat_template(
            hf_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_thinking=True,
        )
        tokens = int(inputs["input_ids"].shape[-1])
        # Reserve one terminal token; SGLang rejects a request at the exact context boundary.
        remaining = 32768 - tokens - 1
        if remaining <= 0:
            self.error = "context_limit"
            raise ValueError("Complete history exceeds the 32K context budget; no history was truncated")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=remaining,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
        )
        if response.usage is not None and response.usage.prompt_tokens != tokens:
            self.error = "token_budget_mismatch"
            raise ValueError(
                "Local and server prompt token counts differ; verify the shared template and image processing"
            )
        choice = response.choices[0]
        text = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            text = "<think>" + reasoning + "</think>" + text
        self.requests.append(
            {
                "input_tokens": tokens,
                "max_tokens": remaining,
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
        if choice.finish_reason == "length":
            self.error = "context_limit"
            raise ValueError("Generation exhausted the 32K context budget")
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
            "modelVersion": response.model,
            "usageMetadata": response.usage.model_dump() if response.usage else {},
        }


@lru_cache(maxsize=128)
def url_image_bytes(url: str) -> bytes:
    import requests

    if url.startswith("data:"):
        return base64.b64decode(url.split("base64,", 1)[1], validate=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def connect_tools(cache: Path, tools, cos_upload) -> str:
    import requests

    original = tools.execute_tool
    tool_configuration = {
        name: os.environ.get(name, "")
        for name in [
            "API_HOST",
            "API_USER",
            "SERPER_SEARCH_URL",
            "JINA_READER_URL",
            "QWEN_API_BASE",
            "QWEN_MODEL_NAME",
            "SERPAPI_URL",
            "IMAGE_UPLOAD_URL",
            "LAYOUT_PARSING_API_URL",
            "SR_MODEL_PATH",
        ]
    }
    tool_configuration["upstream_revision"] = UPSTREAM_REVISION
    if os.environ.get("SR_MODEL_PATH"):
        tool_configuration["sr_weights_sha256"] = sha256(Path(os.environ["SR_MODEL_PATH"]).read_bytes())
    cache.mkdir(parents=True, exist_ok=True)

    def cached(call, images, case_id, case_idx, turn, intermediate, **kwargs):
        identities = {}
        for key, value in images.items():
            if isinstance(value, bytes):
                identities[key] = image_hash(value)
            elif isinstance(value, str) and not value.startswith(("https://", "http://")) and Path(value).is_file():
                identities[key] = image_hash(Path(value).read_bytes())
            elif isinstance(value, str) and value.startswith(("https://", "http://")):
                identities[key] = image_hash(url_image_bytes(value))
            else:
                identities[key] = value
        canonical_call = json.loads(call) if isinstance(call, str) else json.loads(json.dumps(call))
        if canonical_call["name"] in {"image_search", "local_image_search", "lens_scan"}:
            url = canonical_call.get("parameters", {}).get("url", "")
            if url.startswith(("https://", "http://")):
                canonical_call["parameters"]["url"] = "image-sha256:" + image_hash(url_image_bytes(url))
        payload = json.dumps(
            {"call": canonical_call, "images": identities, "tool_configuration": tool_configuration}, sort_keys=True
        )
        path = cache / (sha256(payload.encode()) + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if all(value.startswith(("https://", "http://")) or Path(value).is_file() for value in result[1].values()):
                return result[0], result[1]
        result = original(call, images, case_id, case_idx, turn, intermediate, **kwargs)
        if "Tool execution error" not in result[0]:
            write_json(path, result)
        return result

    tools.execute_tool = cached
    if os.environ.get("IMAGE_UPLOAD_URL"):

        def upload(image, *_args, **_kwargs):
            body = io.BytesIO()
            image.save(body, format="PNG")
            response = requests.post(
                os.environ["IMAGE_UPLOAD_URL"], files={"file": ("image.png", body.getvalue(), "image/png")}, timeout=60
            )
            response.raise_for_status()
            return response.json()["url"]

        cos_upload.upload_pil_image = upload
    return sha256(json.dumps(tool_configuration, sort_keys=True).encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["simplevqa", "vdr"], required=True)
    parser.add_argument("--label", choices=["base", "relax"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-cache", type=Path, required=True)
    parser.add_argument("--overlap-index", type=Path)
    parser.add_argument("--max-cases", type=int, default=0, help="Nonzero values are smoke tests, not full benchmarks")
    parser.add_argument("--allow-missing-tools", action="store_true", help="Smoke tests only")
    args = parser.parse_args()
    if args.max_cases < 0 or (args.allow_missing_tools and not args.max_cases):
        parser.error("Missing tool configuration is permitted only in an explicitly bounded smoke test")
    missing = missing_tools()
    if missing and not args.allow_missing_tools:
        parser.error("Configure tools before a formal benchmark: " + ", ".join(missing))
    if not args.max_cases and not args.overlap_index:
        parser.error("A full benchmark requires --overlap-index for SFT train deduplication")
    provenance = json.loads(Path(os.environ["AGENT_PROVENANCE_FILE"]).read_text())
    if provenance["served_model_name"] != os.environ["AGENT_MODEL"]:
        raise ValueError("Served model name differs from the verified model provenance")
    if args.label == "relax" and (provenance.get("kind") != "trained-final" or provenance.get("epochs", 0) < 2):
        raise ValueError("Relax evaluation requires the final trained artifact, not an upload-validation model")
    if not provenance.get("hashes_verified"):
        raise ValueError("Verify the downloaded model against its upload receipt before evaluation")
    root = Path(os.environ["OPENSEARCH_ROOT"])
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("OpenSearch tool and judge sources differ from the pinned revision")
    sys.path.insert(0, str(root / "opensearch_vl"))
    import httpx
    import requests
    from eval_with_gpt4o import JUDGE_PROMPT, parse_judge_response
    from openai import OpenAI
    from opensearch_infer import config, cos_upload, tools
    from opensearch_infer.pipeline import process_single_case
    from opensearch_infer.prompts import SYSTEM_PROMPT
    from opensearch_infer.runners import InferenceConfig

    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    config.MAX_TURNS = 50
    tool_configuration_sha256 = connect_tools(args.tool_cache, tools, cos_upload)

    def visual_lookup(image_url):
        response = requests.get(
            os.environ["SERPAPI_URL"],
            params={"engine": "google_lens", "url": image_url, "api_key": os.environ["SERPAPI_API_KEY"]},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    agent = OpenAI(
        api_key=os.environ.get("AGENT_API_KEY", "EMPTY"),
        base_url=os.environ["AGENT_BASE_URL"],
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=300),
    )
    judge = OpenAI(
        api_key=os.environ["JUDGE_API_KEY"],
        base_url=os.environ["JUDGE_BASE_URL"],
        max_retries=0,
        default_headers={os.environ.get("JUDGE_API_KEY_HEADER", "api-key"): os.environ["JUDGE_API_KEY"]},
        http_client=httpx.Client(trust_env=False, timeout=180),
    )
    runner = HTTPRunner(args.processor, agent, os.environ["AGENT_MODEL"])
    cases = read_cases(args.data, args.dataset)
    if args.max_cases:
        cases = cases[: args.max_cases]
    overlap = json.loads(args.overlap_index.read_text()) if args.overlap_index else None
    questions = set(overlap["questions"]) if overlap else set()
    images = set(overlap["images"]) if overlap else set()
    protocol = {
        "dataset": args.dataset,
        "upstream_revision": revision,
        "temperature": 0,
        "context_length": 32768,
        "max_turns": 50,
        "judge_model": os.environ["JUDGE_MODEL"],
        "tool_cache": str(args.tool_cache.resolve()),
        "tool_configuration_sha256": tool_configuration_sha256,
        "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode()),
        "chat_template_sha256": runner.template_sha256,
        "judge_prompt_sha256": sha256(JUDGE_PROMPT.encode()),
        "data_sha256": sha256(args.data.read_bytes()),
        "smoke": bool(args.max_cases),
        "missing_tools": missing,
        "overlap_index_sha256": sha256(args.overlap_index.read_bytes()) if args.overlap_index else None,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    records, last_judge = [], 0.0
    with (args.output / "trajectories.jsonl").open("w") as sink:
        for index, case in enumerate(cases):
            runner.requests, runner.error = [], None
            record = {"id": case["id"], "correct": False, "overlaps_sft_train": None, "tool_errors": 0}
            if overlap:
                record["overlaps_sft_train"] = question_hash(case["question"]) in questions or any(
                    image_hash(image) in images for image in case["images"]
                )
            try:
                row = {
                    "data_id": case["id"],
                    "prompt": [{"role": "user", "content": case["question"]}],
                    "images": [{"bytes": image} for image in case["images"]],
                }
                trajectory = process_single_case(
                    row,
                    runner,
                    str(args.output / "cases"),
                    index,
                    dataset_type="test",
                    visual_lookup=visual_lookup,
                    inference_cfg=InferenceConfig(temperature=0, enable_thinking=True),
                )
                record["trajectory"] = trajectory
                record["tool_errors"] = sum(
                    "Tool execution error" in turn.get("tool_output", "") for turn in trajectory["turns"]
                )
                text = trajectory["turns"][-1].get("response_text", "") if trajectory["turns"] else ""
                final = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
                if runner.error or not final:
                    record["error"] = runner.error or "no_final_answer"
                else:
                    record["final_answer"] = final[1].strip()
                    time.sleep(max(0, 1.05 - (time.monotonic() - last_judge)))
                    last_judge = time.monotonic()
                    verdict = judge.chat.completions.create(
                        model=os.environ["JUDGE_MODEL"],
                        temperature=0,
                        max_tokens=2048,
                        messages=[
                            {
                                "role": "user",
                                "content": JUDGE_PROMPT.format(
                                    question=case["question"],
                                    correct_answer=case["answer"],
                                    response=record["final_answer"],
                                ),
                            }
                        ],
                    )
                    raw = verdict.choices[0].message.content or ""
                    if not re.search(r"correct:\s*(yes|no)", raw, re.IGNORECASE):
                        raise ValueError("Judge did not produce the required correctness verdict")
                    record["judge"] = parse_judge_response(raw)
                    record["correct"] = bool(record["judge"]["acc"])
            except Exception as error:
                record["error"] = type(error).__name__
            record["requests"] = runner.requests
            records.append(record)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            logger.info(
                "Case %s/%s id=%s correct=%s error=%s",
                index + 1,
                len(cases),
                case["id"],
                record["correct"],
                record.get("error"),
            )
    clean = [row for row in records if row["overlaps_sft_train"] is False]
    summary = {
        "protocol": protocol,
        "model": os.environ["AGENT_MODEL"],
        "expected_cases": len(cases),
        "correct": sum(row["correct"] for row in records),
        "pass_at_1": sum(row["correct"] for row in records) / len(cases),
        "failures": sum("error" in row for row in records),
        "tool_errors": sum(row["tool_errors"] for row in records),
        "truncations": sum(row.get("error") == "context_limit" for row in records),
        "deduplicated_cases": len(clean) if overlap else None,
        "deduplicated_pass_at_1": sum(row["correct"] for row in clean) / len(clean) if clean else None,
    }
    summary["model_source"] = provenance
    summary["label"] = args.label
    write_json(args.output / "summary.json", summary)
    logger.info("Benchmark summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
