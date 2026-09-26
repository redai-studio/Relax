# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the real rollout metric function, dump writers and offline CLI on
CPU."""

import importlib.util
import json
import random
import string
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from relax.entrypoints.diagnose_repetition import diagnose_dump, main
from relax.utils.metrics.metric_utils import compression_ratio, has_repetition
from relax.utils.training.train_dump_utils import (
    save_debug_rollout_data,
    save_eval_summary_jsonl,
    save_rollout_result_jsonl,
)
from relax.utils.types import Sample


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rollout_metrics_module(monkeypatch):
    """Stub deployment-only imports; execute the complete real rollout module.

    Metrics, Sample, dump serialization and the repetition detector stay real.
    Load under a private name and restore imports to avoid polluting other
    tests.
    """
    before = set(sys.modules)

    def stub(name: str, **attributes: object) -> None:
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attribute = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], attribute, module, raising=False)

    def unused(*args, **kwargs):
        raise AssertionError("The CPU metrics test must not invoke deployment infrastructure")

    stub("transfer_queue")
    stub(
        "sglang.srt.constants",
        GPU_MEMORY_TYPE_CUDA_GRAPH="graph",
        GPU_MEMORY_TYPE_KV_CACHE="kv",
        GPU_MEMORY_TYPE_WEIGHTS="weights",
    )
    stub("relax.backends.sglang.sglang_engine", SGLangEngine=object)
    stub("relax.utils.tracking_utils", init_tracking=unused)
    stub("relax.utils.opd.opd_utils", compute_mopd_metrics=lambda *args: {})
    stub("relax.utils.utils", get_ray_accelerator_kwargs=unused)
    stub("relax.distributed.ray.utils", NOSET_VISIBLE_DEVICES_ENV_VARS_LIST=[], Lock=object)
    name = "relax.distributed.ray._repetition_test_rollout"
    spec = importlib.util.spec_from_file_location(name, ROOT / "relax/distributed/ray/rollout.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        monkeypatch.undo()
        # Newly imported helpers may retain references to the temporary stubs.
        for imported in set(sys.modules) - before:
            if imported.startswith("relax."):
                loaded = sys.modules.pop(imported)
                parent, _, attribute = imported.rpartition(".")
                package = sys.modules.get(parent)
                if package is not None and getattr(package, attribute, None) is loaded:
                    delattr(package, attribute)


def _samples() -> list[Sample]:
    noise = "".join(random.Random(42).choices(string.ascii_letters + string.digits, k=30_000))
    responses = [noise[:10_000] + "repeat! " * 1_250 + noise[20_000:], noise]
    return [
        Sample(
            index=100 + i,
            group_index=7,
            prompt="重复提示" * 5_000,
            response=response,
            response_length=1,
            reward=float(i),
            status=Sample.Status.COMPLETED,
        )
        for i, response in enumerate(responses)
    ]


@pytest.mark.parametrize("evaluation", [False, True])
def test_repetition_real_rollout_metrics_dump_writers_and_cli(tmp_path, rollout_metrics_module, evaluation) -> None:
    samples = _samples()
    before = [sample.to_dict() for sample in samples]
    args = SimpleNamespace(
        log_reward_category=None,
        reward_key=None,
        log_passrate=False,
        advantage_estimator="ppo",
        save_debug_rollout_data=str(tmp_path / "{rollout_id}.pt"),
        rollout_result_dir=str(tmp_path / "results"),
    )
    metric = rollout_metrics_module.compute_metrics_from_samples(args, samples)
    assert metric["repetition_frac"] == 0.5
    assert all(compression_ratio(sample.response[-10_000:])[0] < 10 for sample in samples)
    assert rollout_metrics_module.has_repetition is has_repetition
    data = {"fixture": {"samples": samples}} if evaluation else samples
    save_debug_rollout_data(args, data, 42, evaluation=evaluation)
    if evaluation:
        save_eval_summary_jsonl(args, 42, data)
    else:
        save_rollout_result_jsonl(args, 42, samples)

    pt_path = tmp_path / ("eval_42.pt" if evaluation else "42.pt")
    jsonl_path = tmp_path / "results" / ("eval" if evaluation else "train") / "42.jsonl"
    for path in (pt_path, jsonl_path):
        output = tmp_path / f"{path.suffix[1:]}-report.json"
        run = subprocess.run(
            [sys.executable, "-m", "relax.entrypoints.diagnose_repetition", str(path), "--output", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert run.stdout == ""
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["summary"] == {"sample_count": 2, "repetitive_sample_count": 1, "repetition_frac": 0.5}
        hit, control = report["samples"]
        assert hit["rollout_id"] == 42
        assert hit["group_index"] == 7
        assert hit["index"] == (100 if path.suffix == ".pt" else None)
        assert hit["sample_index"] == (0 if path.suffix == ".jsonl" else None)
        assert hit["dataset"] == ("fixture" if evaluation and path.suffix == ".jsonl" else None)
        assert [(w["start"], w["end"]) for w in hit["hit_windows"]] == [(10_000, 20_000)]
        assert hit["window_count"] == control["window_count"] == 5
        assert control["hit_windows"] == []
        assert "response" not in hit and "prompt" not in hit
    assert [sample.to_dict() for sample in samples] == before


def test_repetition_fixed_fixture_report_matches_stdout(tmp_path, capsys) -> None:
    save_rollout_result_jsonl(SimpleNamespace(rollout_result_dir=str(tmp_path)), 42, _samples())
    main([str(tmp_path / "train/42.jsonl")])
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["repetition_frac"] == 0.5
    assert report["detector"]["offset_unit"] == "unicode_code_point"
    assert report["samples"][0]["sample_index"] == 0
    assert report["samples"][0]["max_compression_ratio"] > 10


def test_repetition_empty_dump_and_response(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert diagnose_dump(path)["summary"] == {
        "sample_count": 0,
        "repetitive_sample_count": 0,
        "repetition_frac": 0.0,
    }
    path.write_text('{"response": ""}\n', encoding="utf-8")
    report = diagnose_dump(path)
    assert report["samples"][0]["sample_position"] == 0
    assert report["samples"][0]["max_compression_ratio"] == 0.0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("record", ['{"response": null}', "{}", "[]", "{invalid"])
def test_repetition_malformed_dump_is_not_silently_skipped(tmp_path, record) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(record + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid.jsonl"):
        diagnose_dump(path)


def test_repetition_cli_rejects_overwriting_input(tmp_path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text('{"response": ""}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--output", str(path)])
    assert exc.value.code == 2
    assert path.read_text(encoding="utf-8") == '{"response": ""}\n'
