# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
    apply_factories,
    apply_factory_merges,
)


_SPEC = importlib.util.spec_from_file_location(
    "kimi_k3_lora_export", Path(__file__).resolve().parents[2] / "examples/models/kimi-k3/tools/kimi_k3_lora_export.py"
)
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def tensor(key: str, value: torch.Tensor | None = None) -> ShardedTensor:
    return ShardedTensor.from_rank_offsets(key, torch.ones(2, 2) if value is None else value)


def test_overlay_preserves_base_and_resolves_wrapped_prefix() -> None:
    base = torch.full((2, 2), 17.0)
    adapter = "decoder.layers.0.linear.adapter.linear_in.weight"
    saved = "language_model." + adapter
    state = {"model": {"base": tensor("decoder.layers.0.linear.to_wrap.weight", base), "adapter": tensor(adapter)}}
    result = export.prepare_overlay(state, {saved})["model"]
    assert isinstance(result["base"], LocalNonpersistentObject)
    assert result["base"].unwrap() is base
    assert result["adapter"].key == saved
    full = export.prepare_overlay(state, {saved, "language_model.decoder.layers.0.linear.weight"})
    assert full["model"]["base"].key == "language_model.decoder.layers.0.linear.weight"


def test_overlay_rejects_unmatched_adapter_and_unexpected_saved_weight() -> None:
    with pytest.raises(ValueError, match="Missing reconstructed adapter"):
        export.prepare_overlay({"a": tensor("decoder.x.adapter.linear_in.weight")}, set())
    with pytest.raises(ValueError, match="not covered"):
        export.prepare_overlay({"a": tensor("decoder.x.weight")}, {"language_model.decoder.y.weight"})
    with pytest.raises(ValueError, match="Ambiguous"):
        export.prepare_overlay(
            {"a": tensor("decoder.x.weight")}, {"decoder.x.weight", "language_model.decoder.x.weight"}
        )


def test_factory_expansion_keeps_merge_and_preserves_missing_base() -> None:
    data = torch.arange(8.0).reshape(4, 2)

    def build(key, weight, replica, flattened):
        return {"left": tensor("decoder.left.weight", weight[:2]), "right": tensor("decoder.right.weight", weight[2:])}

    factory = ShardedTensorFactory("decoder.fused.weight", data, build, lambda state: torch.cat(list(state.values())))
    state = export.prepare_overlay({"fused": factory}, {"language_model.decoder.left.weight"})
    expanded = dict(state)
    apply_factories(expanded)
    assert expanded["fused"]["left"].key == "language_model.decoder.left.weight"
    assert isinstance(expanded["fused"]["right"], LocalNonpersistentObject)
    loaded = {"fused": {"left": torch.full((2, 2), 99.0), "right": expanded["fused"]["right"].unwrap()}}
    actual = apply_factory_merges(loaded, state)["fused"]
    torch.testing.assert_close(actual[:2], torch.full((2, 2), 99.0))
    torch.testing.assert_close(actual[2:], data[2:])


def test_reconstruction_normalizes_targets_and_adapter_coverage() -> None:
    seen = []
    patch = SimpleNamespace(_assert_adapter_coverage=lambda expected, actual: seen.append((expected, actual)))
    spec = {
        "target_modules": ["language_model.decoder.layers.0.linear"],
        "adapter_keys": {"language_model.decoder.x.adapter.linear_in.weight"},
    }
    export.configure_patch(patch, spec)
    assert patch._lora_checkpoint_spec["target_modules"] == ["decoder.layers.0.linear"]
    patch._assert_adapter_coverage(spec["adapter_keys"], {"decoder.x.adapter.linear_in.weight"})
    assert seen[0][0] == seen[0][1]
    assert spec["target_modules"][0].startswith("language_model.")


def test_vision_adapter_is_explicitly_rejected() -> None:
    patch = SimpleNamespace(
        _read_checkpoint_metadata=lambda _: SimpleNamespace(
            state_dict_metadata={"vision_model.x.adapter.linear_in.weight": object()}
        )
    )
    with pytest.raises(ValueError, match="language adapters only"):
        export.build_spec("unused", patch)


def test_real_dcp_overlay_keeps_base_under_strict_model_load(tmp_path: Path) -> None:
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config

    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        path = tmp_path / "checkpoint"
        path.mkdir()
        key = "language_model.decoder.linear.adapter.linear_in.weight"
        expected_adapter = torch.full((2, 2), 9.0)
        # Write a real DCP fixture on CPU; Core's save finalizer requires CUDA.
        common_state = io.BytesIO()
        torch.save([{}], common_state)
        common_state.seek(0)
        common_key = ShardedObject("common_state", None, (1,), (0,)).unique_key
        dcp.save({key: expected_adapter, common_key: common_state}, checkpoint_id=path)
        save_config(CheckpointingConfig("torch_dist", 1), str(path))
        base = torch.full((2, 2), 17.0)
        request = export.prepare_overlay(
            {"model": {"base": tensor("decoder.linear.weight", base), "adapter": tensor(key, torch.zeros(2, 2))}},
            {key},
        )
        loaded = dist_checkpointing.load(request, str(path))
        model = torch.nn.Module()
        model.register_parameter("base", torch.nn.Parameter(torch.zeros(2, 2)))
        model.register_parameter("adapter", torch.nn.Parameter(torch.zeros(2, 2)))
        model.load_state_dict(loaded["model"], strict=True)
        torch.testing.assert_close(model.base, base)
        torch.testing.assert_close(model.adapter, expected_adapter)
    finally:
        if owned_group:
            dist.destroy_process_group()


def test_unknown_saved_tensor_namespace_is_rejected() -> None:
    patch = SimpleNamespace(
        _read_checkpoint_metadata=lambda _: SimpleNamespace(
            state_dict_metadata={"unknown_model.weight": SimpleNamespace(size=(2, 2))}
        )
    )
    with pytest.raises(ValueError, match="Unsupported saved tensor namespaces"):
        export.build_spec("unused", patch)


@pytest.mark.parametrize("optimizer_prefix", [None, "optimizer", "chained_0.optimizer", "chained_1.optimizer"])
def test_lora_spec_ignores_optimizer_tensors_in_real_dcp(tmp_path: Path, optimizer_prefix: str | None) -> None:
    import torch.distributed.checkpoint as dcp

    path = Path(__file__).resolve().parents[2] / "scripts/tools/convert_torch_dist_to_hf_bridge.py"
    spec = importlib.util.spec_from_file_location("lora_checkpoint_reader", path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    target = "language_model.decoder.layers.self_attention.linear_qkv"
    adapters = {
        target + ".adapter.linear_in.weight": torch.ones(2, 4),
        target + ".adapter.linear_out.weight": torch.ones(4, 2),
    }
    saved = dict(adapters)
    if optimizer_prefix is not None:
        saved.update(
            {
                f"{optimizer_prefix}.state.{state}.{key}": torch.zeros_like(value)
                for state in ("exp_avg", "exp_avg_sq", "fp32_param")
                for key, value in adapters.items()
            }
        )
    dcp.save(saved, checkpoint_id=tmp_path, no_dist=True)
    metadata = patch._read_checkpoint_metadata(str(tmp_path))
    original_keys = set(metadata.state_dict_metadata)
    patch._read_checkpoint_metadata = lambda _: metadata
    result = export.build_spec(str(tmp_path), patch)
    assert result["rank"] == 2
    assert result["target_modules"] == [target]
    assert result["adapter_keys"] == set(adapters)
    assert set(metadata.state_dict_metadata) == original_keys


@pytest.mark.parametrize("key", ["vision_tower.x.adapter.linear_in.weight", "chained_bad.optimizer.x.weight"])
def test_optimizer_filter_does_not_hide_unsupported_model_tensors(key: str) -> None:
    patch = SimpleNamespace(
        _read_checkpoint_metadata=lambda _: SimpleNamespace(state_dict_metadata={key: SimpleNamespace(size=(2, 2))})
    )
    with pytest.raises(ValueError, match="language adapters only|Unsupported saved tensor namespaces"):
        export.build_spec("unused", patch)
