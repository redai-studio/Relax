import json

import pytest

from relax.distributed.weight_sync.model_config import configuration_signature
from relax.distributed.weight_sync.sglang_loader import CheckpointTensors, EmbeddedSGLangGroup, verify_qwen3_dense


torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file


def test_preflight_reads_one_index_without_materializing_untied_weights(tmp_path, monkeypatch):
    import safetensors

    config = {"model_type": "qwen3", "num_attention_heads": 1, "num_key_value_heads": 1}
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file({"model.norm.weight": torch.ones(4, dtype=torch.bfloat16)}, tmp_path / "model.safetensors")
    group = object.__new__(EmbeddedSGLangGroup)
    group.config_hash = configuration_signature(config)
    group.startup_schema = CheckpointTensors(tmp_path).schema
    group.tp_size = group.pp_size = 1
    group.worker_receipts = [
        {
            "rank": 0,
            "layout": {"moe_ep_rank": 0, "moe_ep_size": 1, "moe_tp_rank": 0, "moe_tp_size": 1},
            "physical_schema": {"model.norm.weight": [[4], "torch.bfloat16"]},
        }
    ]
    group._rpc = lambda *args, **kwargs: None
    original, opened = safetensors.safe_open, []

    def reader(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", reader)
    group.preflight(tmp_path)
    assert len(opened) == 1
    # A later call reads a fresh index and cannot hide replacement/corruption.
    save_file({"model.norm.weight": torch.ones(4, dtype=torch.float32)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="schema mismatch"):
        group.preflight(tmp_path)
    assert len(opened) == 2


@pytest.mark.parametrize("fault", [None, "parameter", "buffer"])
def test_mixed_dtype_checks_each_live_parameter_and_buffer_before_pause(tmp_path, fault):
    config = {"model_type": "qwen3", "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2}
    startup, candidate = tmp_path / "startup", tmp_path / "candidate"
    startup.mkdir()
    candidate.mkdir()
    weights = {
        "model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "model.counter": torch.tensor(7, dtype=torch.int64),
    }
    save_file(weights, startup / "model.safetensors")
    physical = {(0, name): value.clone() for name, value in weights.items()}
    layouts = {0: {"moe_ep_rank": 0, "moe_ep_size": 1, "moe_tp_rank": 0, "moe_tp_size": 1}}
    if fault == "parameter":
        weights["model.norm.weight"] = weights["model.norm.weight"].float()
    if fault == "buffer":
        weights["model.counter"] = weights["model.counter"].int()
    save_file(weights, candidate / "model.safetensors")
    (candidate / "config.json").write_text(json.dumps(config))
    group = object.__new__(EmbeddedSGLangGroup)
    group.config_hash = configuration_signature(config)
    group.startup_schema = CheckpointTensors(startup).schema
    group.tp_size = group.pp_size = 1
    group._serving = True
    calls = []

    def rpc(phase, **kwargs):
        calls.append(phase)
        group.worker_receipts = [
            {
                "rank": 0,
                "layout": layouts[0],
                "physical_schema": {
                    name: [list(value.shape), str(value.dtype)] for (_, name), value in physical.items()
                },
            }
        ]

    group._rpc = rpc
    if fault:
        with pytest.raises(ValueError, match="schema mismatch"):
            group.load(candidate, "v1")
        assert calls == ["identify"]
        assert group._serving is True
    else:
        group.load(candidate, "v1")
        assert calls == ["identify", "load"]
        result = verify_qwen3_dense(candidate, None, physical_tensors=physical, layouts=layouts)
        assert result["logical_parameters"] == 2


@pytest.mark.parametrize("startup_packed,candidate_packed", [(True, False), (False, True)])
@pytest.mark.parametrize("fault", [None, "missing", "shape", "dtype", "duplicate"])
@pytest.mark.parametrize("expert_field", ["num_experts", "num_local_experts"])
def test_packed_startup_and_bridge_expert_inventory_are_equivalent(
    tmp_path, startup_packed, candidate_packed, fault, expert_field
):
    config = {
        "model_type": "qwen3_vl_moe",
        "text_config": {expert_field: 4, "num_attention_heads": 2, "num_key_value_heads": 2},
    }
    prefix = "model.language_model.layers.0.mlp.experts"
    physical_prefix = "model.layers.0.mlp.experts"
    gate = torch.arange(192, dtype=torch.bfloat16).reshape(4, 12, 4)
    down = torch.arange(96, dtype=torch.bfloat16).reshape(4, 4, 6)
    packed = {
        f"{prefix}.gate_up_proj": gate.transpose(-1, -2).contiguous(),
        f"{prefix}.down_proj": down.transpose(-1, -2).contiguous(),
    }
    split = {}
    for expert in range(4):
        split[f"{prefix}.{expert}.gate_proj.weight"] = gate[expert, :6].clone()
        split[f"{prefix}.{expert}.up_proj.weight"] = gate[expert, 6:].clone()
        split[f"{prefix}.{expert}.down_proj.weight"] = down[expert].clone()
    startup, candidate = tmp_path / "startup", tmp_path / "candidate"
    startup.mkdir()
    candidate.mkdir()
    save_file(packed if startup_packed else split, startup / "model.safetensors")
    incoming = dict(packed if candidate_packed else split)
    key = f"{prefix}.gate_up_proj" if candidate_packed else f"{prefix}.0.gate_proj.weight"
    if fault == "missing":
        del incoming[key]
    elif fault == "shape":
        incoming[key] = incoming[key][:-1].clone()
    elif fault == "dtype":
        incoming[key] = incoming[key].float()
    elif fault == "duplicate":
        incoming.update(split if candidate_packed else packed)
    save_file(incoming, candidate / "model.safetensors")
    (candidate / "config.json").write_text(json.dumps(config))
    group = object.__new__(EmbeddedSGLangGroup)
    group.config_hash = configuration_signature(config)
    group.startup_schema = CheckpointTensors(startup).schema
    group.tp_size, group.pp_size = 2, 1
    group._serving = True
    calls, physical, layouts = [], {}, {}
    for rank in range(2):
        physical[rank, f"{physical_prefix}.w13_weight"] = gate[rank * 2 : (rank + 1) * 2]
        physical[rank, f"{physical_prefix}.w2_weight"] = down[rank * 2 : (rank + 1) * 2]
        layouts[rank] = {"moe_ep_rank": rank, "moe_ep_size": 2, "moe_tp_rank": 0, "moe_tp_size": 1}

    def rpc(phase, **kwargs):
        calls.append(phase)
        group.worker_receipts = [
            {
                "rank": rank,
                "layout": layouts[rank],
                "physical_schema": {
                    name: [list(value.shape), str(value.dtype)]
                    for (worker, name), value in physical.items()
                    if worker == rank
                },
            }
            for rank in range(2)
        ]

    group._rpc = rpc
    if fault:
        with pytest.raises(ValueError):
            group.load(candidate, "v1")
        assert "load" not in calls
        assert group._serving is True
    else:
        group.load(candidate, "v1")
        assert calls == ["identify", "load"]
        result = verify_qwen3_dense(candidate, None, tp_size=2, physical_tensors=physical, layouts=layouts)
        assert result["logical_parameters"] == len(incoming)
        assert result["bytes_checked"] == (gate.numel() + down.numel()) * 2
