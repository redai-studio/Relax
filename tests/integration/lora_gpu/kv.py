# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fresh-engine bidirectional KV isolation, using independent joint-run
baselines."""

import asyncio
import hashlib
import json
from typing import Any

import requests

from tests.integration.lora_gpu.run import EPS, PROMPTS, adapter, arguments, compare, measure
from tests.integration.lora_gpu.support import Engines, events


async def main(args: Any, report: Any) -> None:
    from transformers import AutoTokenizer

    from relax.utils.http_utils import init_http_client

    reference = json.loads(args.baseline_report.read_text())
    assert reference["passed"] and reference["plan"]["tolerance"] == EPS
    assert reference["plan"]["arguments"]["aot"] == args.aot
    assert reference["plan"]["arguments"].get("deterministic", False) == args.deterministic
    assert not reference.get("error")
    hashes = reference["plan"]["source_sha256"]
    # Accept the original experiment's mapping as well as explicit digest records.
    if isinstance(hashes, list):
        hashes = {entry["path"]: entry["sha256"] for entry in hashes}
    from pathlib import Path

    import sglang

    source_root = Path(sglang.__file__).parent
    for relative in (
        "srt/managers/tokenizer_manager.py",
        "srt/managers/tokenizer_control_mixin.py",
        "srt/model_executor/model_runner.py",
    ):
        expected = [value for key, value in hashes.items() if key.endswith(relative)]
        assert expected == [hashlib.sha256((source_root / relative).read_bytes()).hexdigest()]
    for directory in (args.adapter_a, args.adapter_b):
        from pathlib import Path

        for name in ("adapter_model.safetensors", "adapter_config.json"):
            path = Path(directory) / name
            assert hashes[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference["plan"]["arguments"]["model"] == args.model
    from types import SimpleNamespace

    init_http_client(
        SimpleNamespace(
            rollout_num_gpus=2, rollout_num_gpus_per_engine=1, sglang_server_concurrency=16, use_distributed_post=False
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engines = Engines(args, args.output)
    report["comparisons"] = []
    try:
        for index, (warm, cold) in enumerate((("A", "B"), ("B", "A"))):
            port = args.port + index
            label = f"{warm}-to-{cold}"
            proc = engines.launch(port, args.gpus[index], label)
            for name, path in (("A", args.adapter_a), ("B", args.adapter_b)):
                response = requests.post(
                    f"http://127.0.0.1:{port}/load_lora_adapter",
                    json={"lora_name": name, "lora_path": path},
                    timeout=120,
                )
                response.raise_for_status()
            backend = adapter(port, tokenizer)
            for case, text in PROMPTS.items():
                ids = tokenizer.encode(text)
                await measure(backend, ids, warm, label, f"{case}-warm0")
                heated = await measure(backend, ids, warm, label, f"{case}-warm1")
                assert heated["cached"] == len(ids) - 1
                measured = await measure(backend, ids, cold, label, f"{case}-cold")
                result = compare(measured, reference["baselines"][cold][case]["cold"])
                result.update(
                    direction=label,
                    case=case,
                    source_cached=heated["cached"],
                    target_cached=measured["cached"],
                    measured=measured,
                )
                report["comparisons"].append(result)
                assert measured["cached"] == 0 and result["passed"], result
                heated_target = await measure(backend, ids, cold, label, f"{case}-target-repeat")
                assert heated_target["cached"] == len(ids) - 1
                if args.deterministic:
                    result["target_repeat_comparison"] = compare(
                        heated_target, reference["baselines"][cold][case]["cold"]
                    )
                    assert result["target_repeat_comparison"]["passed"], result
            assert not any(
                "flush" in event["ev"] or event["ev"] == "instr.error"
                for event in events(args.output / f"{label}.events.jsonl")
            )
            engines.stop(proc)
        report["passed"] = True
    finally:
        engines.close()


if __name__ == "__main__":
    args = arguments(baseline=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "tolerance": EPS,
        "arguments": vars(args),
        "baseline_sha256": hashlib.sha256(args.baseline_report.read_bytes()).hexdigest(),
    }
    try:
        asyncio.run(main(args, report))
    except BaseException as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
