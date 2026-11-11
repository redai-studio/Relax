# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.tools.flops_benchmark.capture import _expected_parameters, _optimizer_fingerprint
from scripts.tools.flops_benchmark.common import file_digest, write_json
from scripts.tools.flops_benchmark.compare import compare, load_capture
from scripts.tools.flops_benchmark.prepare import choose_batches, loader_order
from scripts.tools.flops_benchmark.summarize import summarize_run


TOLERANCES = Path(__file__).resolve().parents[3] / "scripts/tools/flops_benchmark/tolerances.json"


def _capture(root: Path, *, change_delta: bool = False) -> Path:
    parameter = {"optimizer": [0], "chunk": 0, "name": "weight", "shape": [4], "numel": 4, "dp_group_ranks": [0, 1]}
    for rank in range(2):
        directory = root / f"rank-{rank:05d}"
        directory.mkdir(parents=True)
        file = directory / "shard.pt"
        torch.save({"gradient": torch.ones(2), "delta": torch.full((2,), 0.02 if change_delta else 0.01)}, file)
        write_json(
            directory / "manifest.json",
            {
                "rank": rank,
                "world_size": 2,
                "prepare_calls": 1,
                "update_successful": True,
                "manifest_sha256": "workload",
                "tolerances_sha256": file_digest(TOLERANCES),
                "model": "fixture",
                "model_revision": "fixed",
                "seed": 16,
                "loss_type": "sft",
                "per_token": True,
                "global_batch_size": 4,
                "rollout_id": 20,
                "step_id": 0,
                "tp_rank": 0,
                "pp_rank": 0,
                "cp_rank": 0,
                "ep_rank": rank,
                "batch_source_ids_sha256": "same-batch",
                "initial_state_id": "same-checkpoint",
                "scheduler": {"num_steps": 320},
                "rng": {"torch_cpu": "same-state"},
                "dp_rank": rank,
                "dp_size": 2,
                "model_chunks": 1,
                "expected_samples": [{"tokens": "a", "mask": "a"}, {"tokens": "b", "mask": "b"}],
                "consumed_samples": [[{"tokens": "a" if rank == 0 else "b", "mask": "a" if rank == 0 else "b"}]],
                "expected_parameters": [parameter],
                "shards": [
                    {
                        **parameter,
                        "start": 2 * rank,
                        "end": 2 * rank + 2,
                        "file": file.name,
                        "sha256": file_digest(file),
                        "initial_master_sha256": "initial",
                        "initial_optimizer_state": {"step": 20, "exp_avg": "same", "exp_avg_sq": "same"},
                    }
                ],
            },
        )
    log = root / "train.log"
    log.write_text("(Actor pid=1) step 20: {'train/loss': 0.5}\n")
    return log


def test_flops_comparison_uses_updates_and_complete_dp_shards(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    left, right = _capture(baseline), _capture(candidate, change_delta=True)
    result = compare(baseline, candidate, left, right, TOLERANCES)
    assert not result["passed"]
    quantities = {entry["quantity"]: entry for entry in result["parameters"]}
    assert quantities["gradient"]["passed"]
    assert not quantities["delta"]["passed"]
    assert quantities["delta"]["numel"] == 4
    assert len(result["parameters"]) == 2


def test_flops_comparison_rejects_optimizer_reset(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    left, right = _capture(baseline), _capture(candidate)
    assert compare(baseline, candidate, left, right, TOLERANCES)["passed"]
    path = candidate / "rank-00001/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shards"][0]["initial_optimizer_state"]["exp_avg"] = "reset"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Initial optimizer state"):
        compare(baseline, candidate, left, right, TOLERANCES)
    state = {"step": torch.tensor(0.0), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2), "hyperparameters": {}}
    with pytest.raises(ValueError, match="Adam step"):
        _optimizer_fingerprint(state, 20)
    state["step"].fill_(20)
    first = _optimizer_fingerprint(state, 20)
    state["exp_avg"][0] = 1
    assert _optimizer_fingerprint(state, 20) != first


@pytest.mark.parametrize("corruption", ["whole-parameter", "rank", "overlap", "tensor", "mask"])
def test_flops_comparison_rejects_incomplete_capture(tmp_path: Path, corruption: str) -> None:
    _capture(tmp_path)
    path = tmp_path / "rank-00001/manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "whole-parameter":
        for file in tmp_path.glob("rank-*/manifest.json"):
            data = json.loads(file.read_text())
            missing = {**data["expected_parameters"][0], "name": "missing-weight"}
            data["expected_parameters"].append(missing)
            write_json(file, data)
    elif corruption == "rank":
        path.unlink()
    elif corruption == "overlap":
        manifest["shards"][0]["start"] = 1
        write_json(path, manifest)
    elif corruption == "mask":
        manifest["consumed_samples"][0][0]["mask"] = "wrong"
        write_json(path, manifest)
    else:
        (path.parent / "shard.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_capture(tmp_path)


def test_flops_expected_parameters_include_unowned_buffer_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    child = SimpleNamespace(
        buffers=[SimpleNamespace(params=parameters)],
        data_parallel_group=object(),
        model_param_gbuf_map={parameters[0]: None},
    )
    names = {id(parameter): (0, str(i), (2,), 2) for i, parameter in enumerate(parameters)}
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1])
    assert len(_expected_parameters(child, names)) == 2
    child.buffers[0].params = parameters[:1]
    with pytest.raises(ValueError, match="Trainable model parameters"):
        _expected_parameters(child, names)


def test_flops_workload_freezes_strata_without_replacement() -> None:
    rows = [{"length": 10500, "effective_tokens": 10000} for _ in range(3)]
    rows += [{"length": 2000, "effective_tokens": 1800} for _ in range(45)]
    batches = choose_batches(rows, "stratified", 3, 16, 24576, 16)
    assert sorted(i for batch in batches for i in batch) == list(range(48))
    assert all(sum(rows[i]["length"] > 10000 for i in batch) == 1 for batch in batches)
    order = loader_order(48, 16)
    physical = [None] * 48
    selected = [i for batch in batches for i in batch]
    for physical_id, original_id in zip(order, selected, strict=True):
        physical[physical_id] = original_id
    assert [physical[i] for i in order] == selected
    with pytest.raises(ValueError, match="Insufficient distinct"):
        choose_batches(rows, "stratified", 4, 16, 24576, 16)


def test_flops_summary_weights_loop_time_and_checks_supervised_tokens(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    manifest = {
        "batch_size": 2,
        "batches": [
            {"effective_tokens": 100, "input_tokens": 200},
            {"effective_tokens": 100, "input_tokens": 200},
            {"effective_tokens": 200, "input_tokens": 400},
        ],
    }
    lines = []
    for step, elapsed in enumerate([20, 1, 4]):
        batch = manifest["batches"][step]
        lines.append(f"step {step}: {{'train/loss': 0.5, 'train/global_batch_size': 2}}")
        lines.append(
            f"perf {step}: {{'perf/step_time': {elapsed}, "
            f"'perf/step_resp_token_per_s': {batch['effective_tokens'] / elapsed}, "
            f"'perf/actor_train_tokens': {batch['input_tokens']}}}"
        )
    log.write_text("\n".join(lines))
    result = summarize_run(manifest, log, start=0, steps=3, warmup=1)
    assert result["effective_token_per_s"] == 60
    assert result["input_token_per_s"] == 120
    wrong = copy.deepcopy(manifest)
    wrong["batches"][2]["effective_tokens"] += 1
    with pytest.raises(ValueError, match="Supervised token count"):
        summarize_run(wrong, log, start=0, steps=3, warmup=1)
