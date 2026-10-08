# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file


_PATH = Path(__file__).resolve().parents[2] / "examples/models/kimi-k3/tools/convert_kimi_k3_torch_dist_to_hf.py"
_SPEC = importlib.util.spec_from_file_location("native_k3_export", _PATH)
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def _source(path: Path, shards: dict[str, dict[str, torch.Tensor]]) -> None:
    path.mkdir()
    weight_map = {}
    total = 0
    for filename, tensors in shards.items():
        save_file(tensors, str(path / filename))
        weight_map.update({name: filename for name in tensors})
        total += sum(t.numel() * t.element_size() for t in tensors.values())
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    (path / "config.json").write_text('{"model_type": "kimi_k3"}\n')


def test_quantization_owner_and_native_schema(tmp_path: Path) -> None:
    pytest.importorskip("megatron.bridge.models.conversion.quantization_utils")
    from megatron.bridge.models.conversion.quantization_utils import quantize_mxfp4_e2m1_like_scale

    name = "model.layers.1.mlp.experts.0.w1.weight"
    source = tmp_path / "source"
    _source(
        source,
        {
            "a.safetensors": {"model.layers.0.self_attn.o_norm.weight": torch.zeros(4)},
            "b.safetensors": {
                name + "_packed": torch.zeros((4, 32), dtype=torch.int8),
                name + "_scale": torch.zeros((4, 2), dtype=torch.uint8),
            },
        },
    )
    weight = torch.linspace(-6, 6, 256).reshape(4, 64).to(torch.bfloat16)
    task = SimpleNamespace(weight_dtype=None)
    calls = []

    def quantize(*args, **kwargs):
        calls.append(1)
        return quantize_mxfp4_e2m1_like_scale(*args, **kwargs)

    # No source state is available: the exporter must only read file headers.
    nonowner = export.NativeK3Transform(export.SourceLayout(source, 0, 2), "cpu", quantize)
    assert nonowner(task, {name: weight}, None) == {}
    assert not calls
    owner = export.NativeK3Transform(export.SourceLayout(source, 1, 2), "cpu", quantize)
    actual = owner(task, {name: weight}, None)
    reference = quantize_mxfp4_e2m1_like_scale(weight, torch.empty((4, 2), dtype=torch.uint8))
    assert torch.equal(actual[name + "_packed"], reference[0])
    assert torch.equal(actual[name + "_scale"], reference[1])
    assert len(calls) == 1
    norm_name = "model.layers.0.self_attn.o_norm.weight"
    restored = nonowner(task, {norm_name: torch.ones(4, dtype=torch.bfloat16)}, None)
    assert restored[norm_name].dtype == torch.float32


def test_vision_uses_checkpoint_and_rejects_missing(tmp_path: Path) -> None:
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    name = "vision_tower.norm.weight"
    _source(source, {"a.safetensors": {name: torch.zeros(4)}})
    dcp.save({name: torch.full((4,), 7.0)}, checkpoint_id=checkpoint, no_dist=True)
    layout = export.SourceLayout(source, 0, 1)
    actual = dict(export._checkpoint_vision(str(checkpoint), layout))
    assert torch.equal(actual[name], torch.full((4,), 7.0))
    empty = tmp_path / "other_checkpoint"
    dcp.save({"unrelated": torch.ones(4)}, checkpoint_id=empty, no_dist=True)
    with pytest.raises(ValueError, match="missing vision tensor"):
        dict(export._checkpoint_vision(str(empty), layout))


def test_validate_and_publish_preserve_previous_output(tmp_path: Path) -> None:
    source, staging, output = (tmp_path / p for p in ("source", "staging", "output"))
    tensors = {"weight": torch.ones(4)}
    _source(source, {"a.safetensors": tensors})
    _source(staging, {"a.safetensors": tensors})
    (staging / "config.json").write_text("{}")
    output.mkdir()
    (output / "previous").write_text("keep")
    layout = export.SourceLayout(source, 0, 1)
    summary = export._validate_shards(staging, layout)
    args = SimpleNamespace(
        staging_dir=str(staging),
        output_dir=str(output),
        origin_hf_dir=str(source),
        input_dir="checkpoint",
        world_size=1,
        tp=1,
        pp=1,
        ep=1,
        expert_tp=1,
        replace_output=True,
    )
    export._publish(args, [summary])
    assert (output / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert next(tmp_path.glob("output.before-*/previous")).read_text() == "keep"
    save_file({"weight": torch.ones(4, dtype=torch.bfloat16)}, str(output / "a.safetensors"))
    with pytest.raises(ValueError, match="dtype/shape mismatch"):
        export._validate_shards(output, layout)


def test_waiter_does_not_start_after_failed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    import ray.job_submission

    client = SimpleNamespace(get_job_status=lambda _: ray.job_submission.JobStatus.FAILED)
    monkeypatch.setattr(ray.job_submission, "JobSubmissionClient", lambda _: client)
    with pytest.raises(RuntimeError, match="replacement export was not started"):
        export._wait_for_job("unused", "old-job")


def test_hooks_survive_bridge_factory_recreation(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("megatron.bridge.models.conversion.model_bridge")
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    class KimiK3Bridge:
        pass

    class Factory:
        @property
        def _model_bridge(self):
            return KimiK3Bridge()

    factory = Factory()
    transform = SimpleNamespace(layout="rank-local-layout")
    monkeypatch.setattr(
        MegatronModelBridge, "stream_weights_megatron_to_hf", lambda self, *a, **kw: iter(["language"])
    )
    monkeypatch.setattr(export, "_checkpoint_vision", lambda path, layout, **kwargs: iter([(path, layout)]))
    export._install_export_hooks(factory, "saved-checkpoint", transform)
    assert factory._model_bridge is not factory._model_bridge
    assert list(factory._model_bridge.stream_weights_megatron_to_hf()) == [
        "language",
        ("saved-checkpoint", "rank-local-layout"),
    ]


def test_lora_vision_fallback_is_explicit_and_saved_weights_win(tmp_path: Path) -> None:
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    norm, projector = "vision_tower.norm.weight", "mm_projector.weight"
    _source(source, {"a.safetensors": {norm: torch.ones(4), projector: torch.ones(4)}})
    dcp.save({projector: torch.full((4,), 3.0)}, checkpoint_id=checkpoint, no_dist=True)
    layout = export.SourceLayout(source, 0, 1)
    with pytest.raises(ValueError, match="missing vision tensor"):
        dict(export._checkpoint_vision(str(checkpoint), layout))
    actual = dict(export._checkpoint_vision(str(checkpoint), layout, allow_base_fallback=True))
    assert torch.equal(actual[norm], torch.ones(4))
    assert torch.equal(actual[projector], torch.full((4,), 3.0))


def test_lora_export_rejects_unmapped_adapters() -> None:
    tasks = [SimpleNamespace(global_param_name="decoder.layers.0.mlp.to_wrap.weight0")]
    export._validate_lora_export_tasks(tasks, {"decoder.layers.0.mlp": [object()]})
    with pytest.raises(ValueError, match="coverage mismatch"):
        export._validate_lora_export_tasks(tasks, {"decoder.layers.1.mlp": [object()]})
    with pytest.raises(ValueError, match="coverage mismatch"):
        export._validate_lora_export_tasks([], {})


def test_lora_merge_precedes_native_mxfp4_quantization(tmp_path: Path) -> None:
    pytest.importorskip("megatron.bridge.models.conversion.peft_bridge")
    from megatron.bridge.models.conversion.peft_bridge import AdapterWeight, MegatronPeftBridge
    from megatron.bridge.models.conversion.quantization_utils import quantize_mxfp4_e2m1_like_scale

    name = "language_model.model.layers.1.self_attn.q_proj.weight"
    source = tmp_path / "source"
    _source(
        source,
        {
            "a.safetensors": {
                name + "_packed": torch.zeros((4, 32), dtype=torch.int8),
                name + "_scale": torch.zeros((4, 2), dtype=torch.uint8),
            }
        },
    )
    base = torch.zeros((4, 64), dtype=torch.bfloat16)
    a = torch.ones((2, 64), dtype=torch.bfloat16)
    b = torch.ones((4, 2), dtype=torch.bfloat16)
    adapter = AdapterWeight(
        "decoder.layers.1.self_attention.q_proj", None, 4, 2, SimpleNamespace(weight=a), SimpleNamespace(weight=b)
    )
    model = SimpleNamespace(config=SimpleNamespace(num_moe_experts=0))
    merged = MegatronPeftBridge()._merge_lora_adapter_weights([model], {name: base}, [adapter])
    assert torch.equal(merged[name], base + 2 * (b @ a))
    seen = []

    def quantize(weight, scale, **kwargs):
        seen.append(weight.clone())
        return quantize_mxfp4_e2m1_like_scale(weight, scale)

    hook = export.NativeK3Transform(export.SourceLayout(source, 0, 1), "cpu", quantize)
    result = hook(SimpleNamespace(weight_dtype=None), merged, None)
    expected = quantize_mxfp4_e2m1_like_scale(base + 2 * (b @ a), torch.empty((4, 2), dtype=torch.uint8))
    assert torch.equal(seen[0], merged[name])
    assert torch.equal(result[name + "_packed"], expected[0])
    assert torch.equal(result[name + "_scale"], expected[1])


def test_unknown_vision_tensor_is_not_silently_dropped(tmp_path: Path) -> None:
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    name = "vision_tower.norm.weight"
    _source(source, {"a.safetensors": {name: torch.ones(4)}})
    dcp.save({"vision_tower.unknown.weight": torch.ones(4)}, checkpoint_id=checkpoint, no_dist=True)
    with pytest.raises(ValueError, match="Unmapped checkpoint vision"):
        dict(export._checkpoint_vision(str(checkpoint), export.SourceLayout(source, 0, 1), allow_base_fallback=True))


def test_per_expert_lora_merge_selects_each_expert_adapter(monkeypatch) -> None:
    pytest.importorskip("megatron.bridge.models.conversion.peft_bridge")
    from megatron.bridge.models.conversion.peft_bridge import AdapterWeight, MegatronPeftBridge, parallel_state

    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)
    a = torch.ones((2, 2, 4), dtype=torch.bfloat16)
    b = torch.stack([torch.ones((3, 2)), torch.full((3, 2), 3.0)]).to(torch.bfloat16)
    adapter = AdapterWeight(
        "decoder.layers.1.mlp.experts.linear_fc2", None, 4, 2, SimpleNamespace(weight=a), SimpleNamespace(weight=b)
    )
    names = [f"language_model.model.layers.1.block_sparse_moe.experts.{i}.w2.weight" for i in range(2)]
    weights = {name: torch.zeros((3, 4), dtype=torch.bfloat16) for name in names}
    model = SimpleNamespace(config=SimpleNamespace(num_moe_experts=2))
    actual = MegatronPeftBridge()._merge_lora_adapter_weights([model], weights, [adapter])
    for i, name in enumerate(names):
        torch.testing.assert_close(actual[name], 2 * (b[i] @ a[i]))
