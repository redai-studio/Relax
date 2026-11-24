# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare the pinned V*Bench and evaluate real, local crop interactions."""

import argparse
import base64
import fcntl
import hashlib
import io
import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local

from benchmark import HTTPRunner, image_hash, question_hash
from common import extract_answer, extract_calls, write_json


DATASET = "craigwu/vstar_bench"
REVISION = "d9ae62c903da0c98336e85c5ee89cd863b04b4da"
SYSTEM_PROMPT = """You are a visual investigation assistant. Answer the multiple-choice question using the supplied images.
You may inspect small details with the following local tool. No external search is available.
<tools>
{"type":"function","function":{"name":"crop","description":"Crop and enlarge a region from an image. Coordinates are integer pixels in the displayed image, whose dimensions are provided with each image.","parameters":{"type":"object","properties":{"image":{"type":"string"},"x":{"type":"integer"},"y":{"type":"integer"},"width":{"type":"integer"},"height":{"type":"integer"}},"required":["image","x","y","width","height"]}}}
</tools>
Call it as <tool_call>{"name":"crop","arguments":{"image":"img_1","x":0,"y":0,"width":100,"height":100}}</tool_call>.
The tool returns a new image reference. You may crop any previously provided image. Crops are extracted from the original-resolution region before display resizing.
Use tools when helpful; do not invent tool results. Finish with exactly one option letter in <response>A</response>, replacing A with your answer. Do not put the final answer inside a reasoning block."""


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_option(text: str) -> str | None:
    candidate = extract_answer(text)
    if ("<response>" in text or "</response>" in text) and (
        candidate is None or text.count("<response>") != 1 or text.count("</response>") != 1
    ):
        return None
    candidate = re.sub(r"\*\*|__", "", text.strip() if candidate is None else candidate)
    last_line = candidate.splitlines()[-1].strip() if candidate else ""
    prefix = re.match(
        r"^(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is\s*)?[:：]?\s*", last_line, flags=re.IGNORECASE
    )
    if prefix is None and len(re.findall(r"\([A-D]\)|\b[A-D][.)]", candidate)) > 1:
        return None
    candidate = last_line[prefix.end() :] if prefix else last_line
    if len(set(re.findall(r"\b[A-D]\b", candidate))) > 1:
        return None
    boxed = re.fullmatch(r"\\boxed\{([A-D])\}[.!]?", candidate)
    if boxed:
        return boxed[1]
    if re.fullmatch(r"[A-D]", candidate):
        return candidate
    match = re.fullmatch(r"(?:\(([A-D])\)|([A-D])[.)])(?:[ \t]+[^\r\n]+)?", candidate)
    markers = re.findall(r"\([A-D]\)|\b[A-D][.)]", candidate)
    return (match[1] or match[2]) if match and len(markers) == 1 else None


def prepare(data: Path) -> dict:
    import requests

    data.mkdir(parents=True, exist_ok=True)
    with (data / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = data / "manifest.json"
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if result["revision"] != REVISION or any(
                file_hash(data / name) != value for name, value in result["files_sha256"].items()
            ):
                raise ValueError("V*Bench cache differs from its pinned manifest")
            return result
        response = requests.get(
            f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true&expand=false",
            timeout=120,
        )
        response.raise_for_status()
        listing = {item["path"]: item for item in response.json() if item["type"] == "file"}

        def download(name: str) -> tuple[str, str]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe dataset path")
            metadata = listing[name]
            path = data / relative
            path.parent.mkdir(parents=True, exist_ok=True)

            def valid(payload: bytes) -> bool:
                if len(payload) != metadata["size"]:
                    return False
                if metadata.get("lfs"):
                    return hashlib.sha256(payload).hexdigest() == metadata["lfs"]["oid"]
                header = f"blob {len(payload)}\0".encode()
                return hashlib.sha1(header + payload).hexdigest() == metadata["oid"]

            if not path.exists() or not valid(path.read_bytes()):
                response = requests.get(
                    f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}", timeout=120
                )
                response.raise_for_status()
                if not valid(response.content):
                    raise ValueError(f"Dataset object checksum mismatch: {name}")
                temporary = path.with_suffix(path.suffix + ".part")
                temporary.write_bytes(response.content)
                temporary.replace(path)
            return name, file_hash(path)

        first = download("test_questions.jsonl")
        cases = [json.loads(line) for line in (data / first[0]).read_text().splitlines()]
        assert len(cases) == len({case["question_id"] for case in cases}) == 191
        images = sorted({case["image"] for case in cases})
        with ThreadPoolExecutor(8) as pool:
            files = dict([first, *pool.map(download, images)])
        result = {"dataset": DATASET, "revision": REVISION, "cases": 191, "files_sha256": files}
        write_json(manifest, result)
        return result


def display_image(raw, factor: int, image_tokens: int) -> bytes:
    from PIL import Image

    width, height = raw.size
    if max(width, height) / min(width, height) > 100:
        raise ValueError("Image aspect ratio exceeds 100")
    area = min(max(width * height, 256 * 256), image_tokens * factor * factor)
    scale = math.sqrt(area / (width * height))
    size = tuple(max(factor, int(value * scale / factor) * factor) for value in (width, height))
    output = io.BytesIO()
    raw.resize(size, Image.Resampling.BICUBIC).save(output, format="PNG")
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--processor", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--overlap-index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", choices=["base", "relax"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.max_turns < 1 or args.image_tokens < 256 or args.max_cases < 0:
        parser.error("Invalid concurrency or generation budget")
    manifest = prepare(args.data)
    if args.prepare_only:
        print(json.dumps({"cases": manifest["cases"], "revision": REVISION}), flush=True)
        return
    if any(value is None for value in (args.processor, args.provenance, args.output, args.label, args.overlap_index)):
        parser.error("Evaluation requires processor, provenance, output, label and overlap-index")
    import httpx
    from openai import OpenAI
    from PIL import Image

    provenance = json.loads(args.provenance.read_text())
    if not provenance.get("hashes_verified") or provenance["served_model_name"] != os.environ["AGENT_MODEL"]:
        raise ValueError("Model service differs from verified model provenance")
    if (provenance["kind"] == "trained-final") != (args.label == "relax"):
        raise ValueError("Model provenance does not match evaluation label")
    client = OpenAI(
        base_url=os.environ["AGENT_BASE_URL"],
        api_key=os.environ.get("AGENT_API_KEY", "EMPTY"),
        http_client=httpx.Client(trust_env=False),
        timeout=600,
        max_retries=2,
    )
    runner = HTTPRunner(args.processor, client, os.environ["AGENT_MODEL"])
    factor = int(runner.processor.image_processor.patch_size) * int(runner.processor.image_processor.merge_size)
    overlap = json.loads(args.overlap_index.read_text())
    known_questions, known_images = set(overlap["questions"]), set(overlap["images"])
    cases = [json.loads(line) for line in (args.data / "test_questions.jsonl").read_text().splitlines()]
    if args.max_cases:
        cases = cases[: args.max_cases]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "workflow": "local-crop-mcq",
        "questions_sha256": manifest["files_sha256"]["test_questions.jsonl"],
        "manifest_sha256": file_hash(args.data / "manifest.json"),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "chat_template_sha256": runner.template_sha256,
        "temperature": 0,
        "context_length": 32768,
        "max_turns": args.max_turns,
        "image_tokens": args.image_tokens,
        "image_factor": factor,
        "crop_coordinates": "display-pixels-xywh-mapped-to-original-region",
        "options": "official-test_questions.jsonl-order",
        "score": "final-option-exact-match-with-terminal-statement-normalization-v2",
        "workers": args.workers,
        "smoke": bool(args.max_cases),
        "overlap_index_sha256": file_hash(args.overlap_index),
        "evaluator_sha256": file_hash(Path(__file__)),
        "http_runner_sha256": file_hash(Path(__file__).with_name("benchmark.py")),
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "model-provenance.json", provenance)
    write_json(
        args.output / "evaluator.json",
        {
            "git_commit": os.environ.get("RELAX_GIT_COMMIT")
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": protocol["evaluator_sha256"],
        },
    )
    state = local()

    def evaluate(case: dict) -> dict:
        if not hasattr(state, "runner"):
            state.runner = HTTPRunner(args.processor, client, os.environ["AGENT_MODEL"])
        current = state.runner
        current.requests, current.error, current.last_response = [], None, None
        record = {
            "id": str(case["question_id"]),
            "category": case["category"],
            "correct": False,
            "reference": case["label"],
            "tool_calls": [],
            "tool_errors": 0,
            "trajectory": [],
        }
        case_dir = args.output / "cases" / record["id"]
        case_dir.mkdir(parents=True)
        images = {}

        def add_image(raw) -> list[dict]:
            name = f"img_{len(images) + 1}"
            payload = display_image(raw, factor, args.image_tokens)
            with Image.open(io.BytesIO(payload)) as shown:
                width, height = shown.size
            path = case_dir / f"{name}.png"
            path.write_bytes(payload)
            images[name] = (raw, (width, height))
            description = f"Image {name}: displayed width={width}, height={height} pixels."
            record["trajectory"].append(
                {
                    "image": name,
                    "path": str(path),
                    "display_size": [width, height],
                    "source_size": list(raw.size),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            return [{"text": description}, {"inline_data": {"data": base64.b64encode(payload).decode()}}]

        try:
            payload = (args.data / case["image"]).read_bytes()
            record["overlaps_sft_train"] = (
                question_hash(case["text"].split("\n(A)")[0]) in known_questions or image_hash(payload) in known_images
            )
            with Image.open(io.BytesIO(payload)) as image:
                raw = image.convert("RGB")
            history = [{"role": "user", "parts": [*add_image(raw), {"text": case["text"]}]}]
            for turn in range(args.max_turns):
                result = current.infer(history, system_instruction=SYSTEM_PROMPT)
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                history.append({"role": "model", "parts": [{"text": text}]})
                record["trajectory"].append({"turn": turn, "role": "assistant", "text": text})
                action = text.rsplit("</think>", 1)[-1]
                if "<think>" in action:
                    record["error"] = "unfinished_reasoning"
                    break
                try:
                    calls = extract_calls(action)
                except (ValueError, TypeError):
                    record["error"] = "invalid_tool_call"
                    break
                if not calls:
                    answer = final_option(action)
                    record["final_answer"] = answer
                    if answer is None:
                        record["error"] = "invalid_final_option"
                    else:
                        record["correct"] = answer == case["label"]
                    break
                for call in calls:
                    entry = {"turn": turn, **call, "success": False}
                    record["tool_calls"].append(entry)
                    try:
                        if call["name"] != "crop":
                            raise ValueError("Only crop is available")
                        values = call["arguments"]
                        source, (width, height) = images[values["image"]]
                        x, y, w, h = (values[key] for key in ("x", "y", "width", "height"))
                        if any(type(value) is not int for value in (x, y, w, h)) or w <= 0 or h <= 0:
                            raise ValueError("Crop requires integer pixel coordinates and positive dimensions")
                        left, top, right, bottom = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
                        if right <= left or bottom <= top:
                            raise ValueError("Crop lies outside the displayed image")
                        box = (
                            math.floor(left * source.width / width),
                            math.floor(top * source.height / height),
                            math.ceil(right * source.width / width),
                            math.ceil(bottom * source.height / height),
                        )
                        cropped = source.crop(box)
                        observation = [
                            {"text": "<tool_response>Image cropped successfully."},
                            *add_image(cropped),
                            {"text": "</tool_response>"},
                        ]
                        entry.update(success=True, source_box=list(box), output_image=f"img_{len(images)}")
                    except (KeyError, TypeError, ValueError) as error:
                        record["tool_errors"] += 1
                        observation = [{"text": f"<tool_response>Tool error: {error}</tool_response>"}]
                    history.append({"role": "user", "parts": observation})
            else:
                record["error"] = "turn_limit"
        except Exception as error:
            record["error"] = current.error or type(error).__name__
            if current.last_response is not None:
                record["raw_model_output"] = current.last_response
        record["requests"] = current.requests
        for raw, _ in images.values():
            raw.close()
        return record

    records = []
    with (args.output / "trajectories.jsonl").open("w") as sink, ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(evaluate, case) for case in cases]):
            record = future.result()
            records.append(record)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                json.dumps(
                    {
                        "completed": len(records),
                        "total": len(cases),
                        "id": record["id"],
                        "correct": record["correct"],
                        "tool_calls": len(record["tool_calls"]),
                        "error": record.get("error"),
                    }
                ),
                flush=True,
            )
    clean = [record for record in records if record.get("overlaps_sft_train") is False]
    calls = [call for record in records for call in record["tool_calls"]]
    write_json(
        args.output / "summary.json",
        {
            "protocol": protocol,
            "model_source": provenance,
            "label": args.label,
            "expected_cases": len(cases),
            "correct": sum(record["correct"] for record in records),
            "pass_at_1": sum(record["correct"] for record in records) / len(cases),
            "failures": sum("error" in record for record in records),
            "errors": {
                error: sum(record.get("error") == error for record in records)
                for error in sorted({record["error"] for record in records if "error" in record})
            },
            "tool_calls": len(calls),
            "tool_successes": sum(call["success"] for call in calls),
            "tool_case_rate": sum(bool(record["tool_calls"]) for record in records) / len(cases),
            "deduplicated_cases": len(clean),
            "deduplicated_pass_at_1": sum(record["correct"] for record in clean) / len(clean) if clean else None,
        },
    )
    client.close()
    model_failures = {
        "context_limit",
        "turn_limit",
        "invalid_final_option",
        "unfinished_reasoning",
        "invalid_tool_call",
    }
    if any(record.get("error") not in model_failures | {None} for record in records):
        raise SystemExit("Infrastructure errors remain; preserve answers and resolve before reporting a score")


if __name__ == "__main__":
    main()
