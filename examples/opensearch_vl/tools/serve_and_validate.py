# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Serve one final HF export and check held-out decisions through SGLang
HTTP."""

import argparse
import base64
import subprocess
import sys
import time
from pathlib import Path

from common import canonical_sample, read_jsonl, write_json
from evaluate import score_completion, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()
    import requests
    from transformers import AutoProcessor

    from relax.engine.sft.dataset.chat_template import render_to_text

    args.output.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    base_url = f"http://127.0.0.1:{args.port}"
    session = requests.Session()
    session.trust_env = False
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                str(args.model),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--tp-size",
                str(args.tp_size),
                "--mem-fraction-static",
                "0.7",
                "--trust-remote-code",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"SGLang exited with {process.returncode}; see {args.output / 'server.log'}")
                try:
                    if session.get(f"{base_url}/health", timeout=5).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            else:
                raise TimeoutError("SGLang readiness timed out")
            records = []
            for case in read_jsonl(args.data)[: args.max_cases]:
                row = {**case, "messages": case["messages"] + [{"role": "assistant", "content": case["reference"]}]}
                text = render_to_text(
                    canonical_sample(row),
                    tokenizer=processor.tokenizer,
                    apply_chat_template_kwargs={"preserve_thinking": True},
                )
                marker = "<|im_start|>assistant\n"
                boundary = text.rfind(marker)
                if boundary < 0:
                    raise ValueError("Missing assistant boundary")
                response = session.post(
                    f"{base_url}/generate",
                    json={
                        "text": text[: boundary + len(marker)],
                        "image_data": [base64.b64encode(Path(path).read_bytes()).decode() for path in case["images"]],
                        "sampling_params": {"temperature": 0, "max_new_tokens": 1024, "skip_special_tokens": False},
                    },
                    timeout=300,
                )
                response.raise_for_status()
                completion = response.json()["text"]
                if not completion:
                    raise ValueError(f"Empty service completion for {case['id']}")
                records.append(
                    {
                        "id": case["id"],
                        "kind": case["kind"],
                        "completion": completion,
                        **score_completion(case["reference"], completion, case["kind"]),
                    }
                )
            if not records:
                raise ValueError("No held-out service cases")
            write_json(args.output / "predictions.json", records)
            write_json(args.output / "summary.json", summarize(records))
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
