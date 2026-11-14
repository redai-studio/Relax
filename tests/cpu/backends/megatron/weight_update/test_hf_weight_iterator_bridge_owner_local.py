# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core", reason="Bridge iterator requires Megatron Core", exc_type=ImportError)

from relax.backends.megatron.weight_update import hf_weight_iterator_bridge as bridge_iterator  # noqa: E402
from relax.utils.types import ParamInfo  # noqa: E402


def _info(name: str, src_rank: int) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.bfloat16,
        shape=torch.Size([2, 3]),
        attrs={},
        size=12,
        src_rank=src_rank,
    )


def _iterator(infos: list[ParamInfo], convert) -> bridge_iterator.HfWeightIteratorBridge:
    iterator = bridge_iterator.HfWeightIteratorBridge.__new__(bridge_iterator.HfWeightIteratorBridge)
    iterator.args = SimpleNamespace(num_experts=128)
    iterator.quantization_config = None
    iterator.lora_merge_mode = False
    iterator.lora_adapter_mode = False
    iterator._expert_buckets = [infos]
    iterator._vanilla_key_map = {info.name: info.name for info in infos if info.src_rank == 1}
    iterator._bridge_converter = SimpleNamespace(convert=convert)
    return iterator


def test_owner_local_expert_iterator_converts_only_owned_experts_without_collectives(monkeypatch) -> None:
    infos = [
        _info("decoder.layers.0.mlp.experts.linear_fc1.weight37", src_rank=1),
        _info("decoder.layers.0.mlp.experts.linear_fc2.weight37", src_rank=1),
        _info("decoder.layers.0.mlp.experts.linear_fc1.weight2", src_rank=0),
    ]
    converted_sources = []

    def convert(name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        converted_sources.append(name)
        expert_id = int(name.rsplit("weight", 1)[1])
        prefix = f"model.language_model.layers.0.mlp.experts.{expert_id}"
        if "linear_fc1" in name:
            return [(f"{prefix}.gate_proj.weight", tensor), (f"{prefix}.up_proj.weight", tensor + 1)]
        return [(f"{prefix}.down_proj.weight", tensor)]

    iterator = _iterator(infos, convert)
    monkeypatch.setattr(bridge_iterator.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge_iterator.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_tensor_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(bridge_iterator.dist, "get_rank", lambda *, group: 1)
    monkeypatch.setattr(bridge_iterator.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("owner-local export must not perform collectives")

    for collective in ("all_gather", "all_gather_object", "broadcast"):
        monkeypatch.setattr(bridge_iterator.dist, collective, unexpected_collective)

    weights = {
        infos[0].name: torch.zeros((2, 3), dtype=torch.bfloat16),
        infos[1].name: torch.full((2, 3), 2, dtype=torch.bfloat16),
    }
    actual = list(iterator.iter_owner_local_expert_tensors(weights))

    assert converted_sources == [infos[0].name, infos[1].name]
    assert [(owner, expert, name) for owner, expert, name, _ in actual] == [
        (1, 37, "model.language_model.layers.0.mlp.experts.37.gate_proj.weight"),
        (1, 37, "model.language_model.layers.0.mlp.experts.37.up_proj.weight"),
        (1, 37, "model.language_model.layers.0.mlp.experts.37.down_proj.weight"),
    ]
    assert [tensor.dtype for _, _, _, tensor in actual] == [torch.bfloat16] * 3


def test_grouped_owner_iterator_uses_numeric_expert_order_and_complete_projection_metadata(monkeypatch):
    infos = []
    for fc in (1, 2):
        for expert in range(24):
            shape = torch.Size((4, 3) if fc == 1 else (3, 2))
            infos.append(
                ParamInfo(
                    name=f"decoder.layers.0.mlp.experts.linear_fc{fc}.weight{expert}",
                    dtype=torch.bfloat16,
                    shape=shape,
                    attrs={},
                    size=shape.numel() * 2,
                    src_rank=expert // 12,
                )
            )
    infos.sort(key=lambda info: info.name)
    calls = []

    def convert(name, tensor):
        calls.append(name)
        expert = int(name.rsplit("weight", 1)[1])
        prefix = f"model.layers.0.mlp.experts.{expert}"
        if "fc1" in name:
            return [(prefix + ".gate_proj.weight", tensor[:2]), (prefix + ".up_proj.weight", tensor[2:])]
        return [(prefix + ".down_proj.weight", tensor)]

    iterator = _iterator(infos, convert)
    iterator.args.num_experts = 24
    iterator._vanilla_key_map = {info.name: info.name for info in infos if info.src_rank == 0}
    weights = {info.name: torch.zeros(info.shape, dtype=info.dtype) for info in infos if info.src_rank == 0}
    monkeypatch.setattr(bridge_iterator.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge_iterator.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_tensor_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_model_parallel_group", lambda: object())
    monkeypatch.setattr(bridge_iterator.dist, "get_rank", lambda *, group: 0)
    monkeypatch.setattr(bridge_iterator.dist, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(bridge_iterator.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    list(iterator.iter_owner_local_expert_tensors(weights, grouped=True))
    assert calls == [
        f"decoder.layers.0.mlp.experts.linear_fc{fc}.weight{expert}" for fc in (1, 2) for expert in range(12)
    ]


@pytest.mark.parametrize(
    ("pp_size", "vpp_size", "etp_size", "ep_size"),
    [(2, 1, 1, 2), (1, 2, 1, 2), (1, 1, 2, 2), (1, 1, 1, 1)],
)
def test_owner_local_expert_iterator_rejects_unsupported_parallel_layout(
    monkeypatch, pp_size: int, vpp_size: int, etp_size: int, ep_size: int
) -> None:
    iterator = _iterator([], lambda *_: [])
    monkeypatch.setattr(bridge_iterator.mpu, "get_pipeline_model_parallel_world_size", lambda: pp_size)
    monkeypatch.setattr(bridge_iterator.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: vpp_size)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_tensor_parallel_world_size", lambda: etp_size)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_model_parallel_world_size", lambda: ep_size)

    with pytest.raises(ValueError, match="Owner-local expert export requires"):
        list(iterator.iter_owner_local_expert_tensors({}))


def test_owner_local_expert_iterator_rejects_bridge_expert_id_mismatch(monkeypatch) -> None:
    info = _info("decoder.layers.0.mlp.experts.linear_fc1.weight37", src_rank=1)
    iterator = _iterator(
        [info],
        lambda *_: [("model.layers.0.mlp.experts.38.gate_proj.weight", torch.zeros((2, 3), dtype=torch.bfloat16))],
    )
    monkeypatch.setattr(bridge_iterator.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge_iterator.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_tensor_parallel_world_size", lambda: 1)
    monkeypatch.setattr(bridge_iterator.mpu, "get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(bridge_iterator.dist, "get_rank", lambda *, group: 1)
    monkeypatch.setattr(bridge_iterator.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))

    with pytest.raises(ValueError, match="Bridge expert identity mismatch"):
        list(iterator.iter_owner_local_expert_tensors({info.name: torch.zeros((2, 3), dtype=torch.bfloat16)}))
