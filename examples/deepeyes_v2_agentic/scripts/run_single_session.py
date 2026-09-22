# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Run ONE DeepEyes V2 agent session end-to-end without Ray or the Relax
launcher.

Two input modes:
  1. --synthetic  (default): generates a tiny image + canned question, no parquet needed
  2. --parquet PATH --row N: pulls a real sample from a converted V2 parquet

Required env:
  OPENAI_BASE_URL        OpenAI-compatible chat endpoint (e.g. http://node:30000/v1)
  OPENAI_API_KEY         any string (SGLang doesn't verify)
  DATA_DIR or APPTAINER_IMAGE_PATH:
    - DATA_DIR                root produced by scripts/prepare.sh; SIF auto-found at
                              ${DATA_DIR}/sif/deepeyes_v2_kernel.sif
    - APPTAINER_IMAGE_PATH    explicit override (takes precedence over DATA_DIR)

Auto-set (override with env if needed):
  SANDBOX_CONFIG_PATH    examples/deepeyes_v2_agentic/apptainer_env/apptainer_config.yaml
  RELAX_INPUT_JSON       /tmp/deepeyes_v2_single_input.json
  RELAX_OUTPUT_JSON      /tmp/deepeyes_v2_single_output.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent
APP_DIR = EXAMPLE_DIR / "app"
# examples/<name>/scripts/ → repo root is two parents up of EXAMPLE_DIR;
# the agent subprocess needs this on PYTHONPATH so `from relax.utils...`
# resolves without requiring `pip install -e .` of the worktree.
RELAX_REPO_ROOT = EXAMPLE_DIR.parent.parent

# Make `from app.prompt import ...` work when run as a standalone script.
sys.path.insert(0, str(EXAMPLE_DIR))

from app.prompt import UNIFIED_SYSTEM_PROMPT as SYSTEM_PROMPT  # noqa: E402


DEFAULT_INPUT = Path("/tmp/deepeyes_v2_single_input.json")
DEFAULT_OUTPUT = Path("/tmp/deepeyes_v2_single_output.json")


def encode_data_uri(image: Image.Image) -> str:
    buf = BytesIO()
    (image if image.mode == "RGB" else image.convert("RGB")).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def synthetic_sample() -> tuple[Image.Image, str, dict[str, Any]]:
    """Generate a tiny image with a red square + question about it.

    Designed to be answerable by the model with one round of `<code>` to crop
    the marked region, then `<answer>`.
    """
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((180, 180, 332, 332), fill="red", outline="black", width=3)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((220, 240), "42", fill="white", font=font)

    question = (
        "There is a red square in the centre of this image with a number written inside it. "
        "Crop the red square out of `image_1` to see the number clearly, then tell me what number it is."
    )
    metadata = {
        "data_index": "synthetic-0001",
        "data_source": "perception",
        "ground_truth": "42",
    }
    return img, question, metadata


def parquet_sample(parquet: Path, row: int) -> tuple[Image.Image, str, dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    if row >= table.num_rows:
        sys.exit(f"row {row} out of range (parquet has {table.num_rows} rows)")
    record = table.to_pylist()[row]

    # Schema is project-specific; we expect either `prompt`+`images` (V2 convention)
    # or `messages` already shaped. Try `images` first.
    images_field = record.get("images") or []
    if not images_field:
        sys.exit(f"row {row} has no images")
    image_bytes = images_field[0]
    if isinstance(image_bytes, dict) and "bytes" in image_bytes:
        image_bytes = image_bytes["bytes"]
    img = Image.open(BytesIO(image_bytes))
    img.load()

    prompt = record.get("prompt") or record.get("question") or ""
    if isinstance(prompt, list):
        # Sometimes prompts are a list of chat-completion content parts
        text_parts = [p.get("text") or p.get("content") or "" for p in prompt if isinstance(p, dict)]
        prompt = "\n".join(t for t in text_parts if t)

    extra_info = record.get("extra_info") or {}
    metadata = {
        "data_index": str(extra_info.get("index") or row),
        "data_source": extra_info.get("data_source") or "perception",
        "ground_truth": (record.get("reward_model") or {}).get("ground_truth"),
    }
    return img, prompt, metadata


def build_input_json(image: Image.Image, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_data_uri(image)}},
                    {"type": "text", "text": question},
                ],
            },
        ],
        "metadata": metadata,
    }


def _abbrev_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return repr(content)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(repr(item))
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(item.get("text", ""))
        elif kind == "image_url":
            url = (item.get("image_url") or {}).get("url", "")
            head, _, _ = url.partition(",")
            parts.append(f"<image {head},…{len(url)} chars>")
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(parts)


def print_transcript(messages: list[dict[str, Any]]) -> None:
    print("[harness] ===== transcript =====")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        print(f"[harness] --- turn {i} [{role}] ---")
        print(_abbrev_content(msg.get("content")))


def check_env() -> None:
    missing = [k for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"missing required env: {', '.join(missing)}")
    # Auto-derive APPTAINER_IMAGE_PATH from DATA_DIR if not explicitly set.
    if not os.environ.get("APPTAINER_IMAGE_PATH"):
        data_dir = os.environ.get("DATA_DIR")
        if not data_dir:
            sys.exit("set DATA_DIR (run scripts/prepare.sh first) or APPTAINER_IMAGE_PATH")
        derived = Path(data_dir) / "sif" / "deepeyes_v2_kernel.sif"
        if not derived.is_file():
            sys.exit(f"SIF not found at {derived} — run: DATA_DIR={data_dir} bash scripts/prepare.sh")
        os.environ["APPTAINER_IMAGE_PATH"] = str(derived)
    sif = Path(os.environ["APPTAINER_IMAGE_PATH"])
    if not sif.is_file():
        sys.exit(f"APPTAINER_IMAGE_PATH does not exist: {sif}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--synthetic", action="store_true", help="generate a tiny synthetic sample (default if no parquet)"
    )
    src.add_argument("--parquet", type=Path, help="path to a converted V2 parquet")
    p.add_argument("--row", type=int, default=0, help="row index when --parquet is set (default 0)")
    p.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--keep-input", action="store_true", help="don't delete input JSON on success")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_env()

    os.environ.setdefault("SANDBOX_CONFIG_PATH", str(EXAMPLE_DIR / "apptainer_env" / "apptainer_config.yaml"))
    os.environ.setdefault("RELAX_INPUT_JSON", str(args.input_json))
    os.environ.setdefault("RELAX_OUTPUT_JSON", str(args.output_json))

    if args.parquet:
        image, question, metadata = parquet_sample(args.parquet, args.row)
        source_desc = f"{args.parquet}[{args.row}]"
    else:
        image, question, metadata = synthetic_sample()
        source_desc = "synthetic"

    payload = build_input_json(image, question, metadata)
    args.input_json.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[harness] input  : {args.input_json}  ({source_desc}, {len(json.dumps(payload))} bytes)")
    print(f"[harness] output : {args.output_json}")
    print(f"[harness] image  : {image.size[0]}x{image.size[1]} {image.mode}")
    print(f"[harness] prompt : {question[:120]}{'...' if len(question) > 120 else ''}")
    print(f"[harness] data_source: {metadata.get('data_source')}")
    print("[harness] launching: python -m app.agent --input-json … --output-json …")
    print(f"[harness] cwd      : {EXAMPLE_DIR}")
    print()

    # Prepend EXAMPLE_DIR (for `app.*`) and RELAX_REPO_ROOT (for `relax.*`)
    # to any existing PYTHONPATH, instead of replacing it.
    sub_pythonpath = os.pathsep.join(
        p for p in (str(EXAMPLE_DIR), str(RELAX_REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.agent",
            "--input-json",
            str(args.input_json),
            "--output-json",
            str(args.output_json),
        ],
        cwd=str(EXAMPLE_DIR),
        env={**os.environ, "PYTHONPATH": sub_pythonpath},
    )
    print()
    if proc.returncode != 0:
        print(f"[harness] agent exited with code {proc.returncode}")
        return proc.returncode

    out = json.loads(args.output_json.read_text(encoding="utf-8"))
    print_transcript(out.get("messages") or [])

    md = out.get("metadata") or {}
    counts = md.get("branch_counts", {})
    print()
    print("[harness] ===== summary =====")
    print(f"[harness] stop_reason   : {md.get('stop_reason')}")
    print(f"[harness] turns         : {sum(counts.values()) if counts else '?'}")
    print(f"[harness] branch_counts : {counts}")
    print(f"[harness] final_answer  : {md.get('final_answer')}")
    print(f"[harness] last_error    : {md.get('last_error')}")
    print(f"[harness] data_source   : {md.get('data_source')}")

    if not args.keep_input:
        args.input_json.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
