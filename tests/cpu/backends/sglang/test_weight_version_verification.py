import json

import pytest

from relax.distributed.weight_sync.sglang_loader import verify_qwen3_dense


torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file


@pytest.mark.parametrize("fault", ["missing", "extra", "shape"])
def test_preflight_uses_startup_inventory_independent_of_candidate(tmp_path, fault):
    from relax.distributed.weight_sync.model_config import configuration_signature
    from relax.distributed.weight_sync.sglang_loader import CheckpointTensors, EmbeddedSGLangGroup

    startup, candidate = tmp_path / "startup", tmp_path / "candidate"
    startup.mkdir()
    candidate.mkdir()
    config = {"model_type": "qwen3", "num_hidden_layers": 1}
    (candidate / "config.json").write_text(json.dumps(config))
    weights = {
        "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
    }
    save_file(weights, startup / "model.safetensors")
    group = object.__new__(EmbeddedSGLangGroup)
    group.config_hash = configuration_signature(config)
    group.startup_schema = CheckpointTensors(startup).schema
    if fault == "missing":
        del weights["model.norm.weight"]
    elif fault == "extra":
        weights["unexpected.weight"] = torch.ones(4, dtype=torch.bfloat16)
    else:
        weights["model.norm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    save_file(weights, candidate / "model.safetensors")
    with pytest.raises(ValueError, match="inventory"):
        group.preflight(candidate)


def test_versioned_memory_restore_uses_remaining_operation_budget():
    from unittest.mock import Mock, patch

    from relax.backends.sglang.sglang_engine import SGLangEngine
    from relax.distributed.weight_sync.deadline import time_budget

    engine = object.__new__(SGLangEngine)
    engine.node_rank, engine.server_host, engine.server_port = 0, "localhost", 1
    engine._weight_consumer_enabled = True
    engine._get_version_consumer = lambda: Mock(session=Mock(engines=Mock(installation_identity={"epoch": 1})))
    clock = [100.0]
    with patch("relax.distributed.weight_sync.deadline.time.monotonic", side_effect=lambda: clock[0]):
        with time_budget(10), patch("requests.post", return_value=Mock()) as post:
            clock[0] += 7
            engine._make_request("resume_memory_occupation", {"tags": ["weights"]})
            assert post.call_args.kwargs["timeout"] == 3
    engine._weight_consumer_enabled = False
    with patch("requests.post", return_value=Mock()) as post:
        engine._make_request("resume_memory_occupation", {"tags": ["weights"]})
        assert post.call_args.kwargs["timeout"] is None


@pytest.mark.parametrize("vl", [False, True])
@pytest.mark.parametrize("tp", [1, 2])
def test_tied_logical_aliases_match_one_physical_parameter(tmp_path, vl, tp):
    checkpoint, dump = tmp_path / "checkpoint", tmp_path / "dump"
    checkpoint.mkdir()
    dump.mkdir()
    text = {"num_attention_heads": 2, "num_key_value_heads": 2, "tie_word_embeddings": True}
    config = {"model_type": "qwen3_vl", "text_config": text} if vl else {"model_type": "qwen3", **text}
    (checkpoint / "config.json").write_text(json.dumps(config))
    key = "model.language_model.embed_tokens.weight" if vl else "model.embed_tokens.weight"
    value = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    save_file({key: value.clone(), "lm_head.weight": value.clone()}, checkpoint / "model.safetensors")
    for rank in range(tp):
        save_file({"lm_head.weight": value.chunk(tp)[rank].clone()}, dump / f"model-rank-{rank}-part-0.safetensors")
    reference = verify_qwen3_dense(checkpoint, dump, tp_size=tp)
    assert reference["logical_parameters"] == 2
    results = []
    for rank in range(tp):
        result = verify_qwen3_dense(
            checkpoint,
            None,
            tp_size=tp,
            physical_tensors={(rank, "lm_head.weight"): value.chunk(tp)[rank]},
            worker_rank=rank,
        )
        results.append(result)
        assert set(result["covered_logical_names"]) == {key, "lm_head.weight"}
    assert sum(item["bytes_checked"] for item in results) == reference["bytes_checked"]
    # Two names may only share one physical proof when their bytes match.
    conflicting = value.clone()
    conflicting[0, 0] = -1
    save_file({key: conflicting, "lm_head.weight": value.clone()}, checkpoint / "model.safetensors")
    with pytest.raises(ValueError, match="tied embedding"):
        verify_qwen3_dense(checkpoint, dump, tp_size=tp)


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize("outer_tied", [False, True])
def test_vl_parent_config_controls_embedding_aliases(tmp_path, model_type, outer_tied):
    checkpoint, dump = tmp_path / "checkpoint", tmp_path / "dump"
    checkpoint.mkdir()
    dump.mkdir()
    config = {
        "model_type": model_type,
        "tie_word_embeddings": outer_tied,
        "text_config": {"num_attention_heads": 2, "num_key_value_heads": 2, "tie_word_embeddings": not outer_tied},
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    embedding = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    head = embedding.clone() if outer_tied else embedding + 100
    logical = {"model.language_model.embed_tokens.weight": embedding, "lm_head.weight": head}
    save_file(logical, checkpoint / "model.safetensors")
    physical = {"lm_head.weight": head}
    if not outer_tied:
        physical["model.embed_tokens.weight"] = embedding
    save_file(physical, dump / "model-rank-0-part-0.safetensors")
    result = verify_qwen3_dense(checkpoint, dump)
    assert result["physical_parameters"] == (1 if outer_tied else 2)
    assert result["logical_parameters"] == 2
    live = verify_qwen3_dense(
        checkpoint, None, physical_tensors={(0, key): value for key, value in physical.items()}, worker_rank=0
    )
    assert live["bytes_checked"] == result["bytes_checked"]


def test_consumer_only_resumes_released_memory_tags():
    from unittest.mock import Mock

    from sglang.srt.constants import GPU_MEMORY_ALL_TYPES

    from relax.backends.sglang.sglang_engine import SGLangEngine

    engine = object.__new__(SGLangEngine)
    engine._weight_consumer_enabled = True
    engine._make_request, engine.flush_cache = Mock(), Mock()
    engine.resume_memory_occupation()
    engine._make_request.assert_not_called()
    engine.release_memory_occupation()
    assert engine._version_offloaded_tags == set(GPU_MEMORY_ALL_TYPES)
    engine.resume_memory_occupation(tags=["weights"])
    assert "weights" not in engine._version_offloaded_tags
    remaining = sorted(engine._version_offloaded_tags)
    engine.resume_memory_occupation()
    engine._make_request.assert_called_with("resume_memory_occupation", {"tags": remaining})
    engine._make_request.reset_mock()
    engine.resume_memory_occupation()
    engine._make_request.assert_not_called()


@pytest.mark.parametrize("ep_size,moe_tp_size", [(1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("vl", [False, True])
def test_static_expert_layout_checks_every_local_expert(tmp_path, ep_size, moe_tp_size, packed, vl):
    checkpoint, dump = tmp_path / "checkpoint", tmp_path / "dump"
    checkpoint.mkdir()
    dump.mkdir()
    tp_size = ep_size * moe_tp_size
    config = {"model_type": "qwen3_moe", "num_experts": 4, "num_attention_heads": 4, "num_key_value_heads": 4}
    if vl:
        config = {"model_type": "qwen3_vl_moe", "text_config": config}
    (checkpoint / "config.json").write_text(json.dumps(config))
    prefix = "model.layers.0.mlp.experts"
    logical = {}
    for expert in range(4):
        for part in ("gate_proj", "up_proj", "down_proj"):
            logical[f"{prefix}.{expert}.{part}.weight"] = (
                torch.arange(32, dtype=torch.float32).reshape(8, 4) + expert * 100 + len(part)
            )
    published = logical
    if packed:
        published = {
            f"{prefix}.gate_up_proj": torch.stack(
                [
                    torch.cat([logical[f"{prefix}.{expert}.{part}.weight"] for part in ("gate_proj", "up_proj")])
                    for expert in range(4)
                ]
            ),
            f"{prefix}.down_proj": torch.stack(
                [logical[f"{prefix}.{expert}.down_proj.weight"] for expert in range(4)]
            ),
        }
        published = {name: value.transpose(-1, -2).contiguous() for name, value in published.items()}
    if vl:
        published = {name.replace("model.", "model.language_model.", 1): value for name, value in published.items()}
    save_file(published, checkpoint / "model.safetensors")
    layouts = {}
    for rank in range(tp_size):
        er, tr = divmod(rank, moe_tp_size)
        layouts[rank] = {"moe_ep_rank": er, "moe_ep_size": ep_size, "moe_tp_rank": tr, "moe_tp_size": moe_tp_size}
        experts = range(er * (4 // ep_size), (er + 1) * (4 // ep_size))
        weights = {
            f"{prefix}.w13_weight": torch.stack(
                [
                    torch.cat(
                        [
                            logical[f"{prefix}.{expert}.{part}.weight"].chunk(moe_tp_size, dim=0)[tr]
                            for part in ("gate_proj", "up_proj")
                        ]
                    )
                    for expert in experts
                ]
            ),
            f"{prefix}.w2_weight": torch.stack(
                [logical[f"{prefix}.{expert}.down_proj.weight"].chunk(moe_tp_size, dim=1)[tr] for expert in experts]
            ),
        }
        save_file(weights, dump / f"model-rank-{rank}-part-0.safetensors")
    reference = verify_qwen3_dense(checkpoint, dump, tp_size=tp_size, layouts=layouts)
    assert reference["logical_parameters"] == len(published)
    assert reference["bytes_checked"] == sum(t.numel() * t.element_size() for t in published.values())
    from safetensors.torch import load_file

    covered = set()
    for rank in range(tp_size):
        weights = load_file(dump / f"model-rank-{rank}-part-0.safetensors")
        result = verify_qwen3_dense(
            checkpoint,
            None,
            tp_size=tp_size,
            layouts=layouts,
            physical_tensors={(rank, name): value for name, value in weights.items()},
            worker_rank=rank,
        )
        covered.update(result["covered_logical_names"])
        schema_result = verify_qwen3_dense(
            checkpoint,
            None,
            tp_size=tp_size,
            layouts=layouts,
            physical_tensors={(rank, name): value.to("meta") for name, value in weights.items()},
            worker_rank=rank,
            schema_only=True,
        )
        assert schema_result == result
    assert covered == set(published)
    # A changed local expert must fail the same byte-level validation.
    path = dump / "model-rank-0-part-0.safetensors"
    weights = load_file(path)
    weights[f"{prefix}.w13_weight"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="Physical weight mismatch"):
        verify_qwen3_dense(
            checkpoint,
            None,
            tp_size=tp_size,
            layouts=layouts,
            physical_tensors={(0, name): value for name, value in weights.items()},
            worker_rank=0,
        )
    save_file(weights, path)
    with pytest.raises(ValueError, match="Physical weight mismatch"):
        verify_qwen3_dense(checkpoint, dump, tp_size=tp_size, layouts=layouts)


@pytest.mark.parametrize(
    "fault", ["expert_count", "fused_dimension", "tp_dimension", "dtype", "down_bytes", "ambiguous"]
)
def test_packed_experts_reject_invalid_schema_or_content(tmp_path, fault):
    prefix = "model.layers.0.mlp.experts"
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "num_experts": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
            }
        )
    )
    gate = torch.arange(128, dtype=torch.float32).reshape(4, 8, 4)
    down = (torch.arange(64, dtype=torch.float32) + 500).reshape(4, 4, 4)
    logical = {f"{prefix}.gate_up_proj": gate.clone(), f"{prefix}.down_proj": down.clone()}
    physical = {}
    for rank in range(2):
        physical[rank, f"{prefix}.w13_weight"] = torch.cat(
            [
                gate[:, :4, :][:, rank * 2 : (rank + 1) * 2, :],
                gate[:, 4:, :][:, rank * 2 : (rank + 1) * 2, :],
            ],
            dim=1,
        ).clone()
        physical[rank, f"{prefix}.w2_weight"] = down[:, :, rank * 2 : (rank + 1) * 2].clone()
    if fault == "expert_count":
        logical[f"{prefix}.gate_up_proj"] = gate[:3].clone()
    elif fault == "fused_dimension":
        logical[f"{prefix}.gate_up_proj"] = gate[:, :7, :].clone()
    elif fault == "tp_dimension":
        logical[f"{prefix}.down_proj"] = down[:, :, :3].clone()
    elif fault == "dtype":
        logical[f"{prefix}.gate_up_proj"] = gate.to(torch.bfloat16)
    elif fault == "down_bytes":
        physical[1, f"{prefix}.w2_weight"][-1, -1, -1] = -1
    else:
        logical[f"{prefix}.0.gate_proj.weight"] = gate[0, :4].clone()
    logical = {
        name: value.transpose(-1, -2).contiguous() if value.ndim == 3 else value for name, value in logical.items()
    }
    save_file(logical, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="packed expert|Physical|Uncovered"):
        verify_qwen3_dense(tmp_path, None, tp_size=2, physical_tensors=physical)


@pytest.mark.parametrize("tp", [1, 2])
def test_pipeline_ranges_cover_global_layers_and_boundary_weights(tmp_path, tp):
    checkpoint, dump = tmp_path / "checkpoint", tmp_path / "dump"
    checkpoint.mkdir()
    dump.mkdir()
    config = {"model_type": "qwen3", "num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 4}
    (checkpoint / "config.json").write_text(json.dumps(config))
    logical = {
        f"model.layers.{layer}.input_layernorm.weight": torch.arange(4, dtype=torch.float32) + layer
        for layer in range(4)
    }
    logical.update(
        {
            "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
            "lm_head.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4) + 1,
            "model.norm.weight": torch.arange(4, dtype=torch.float32),
        }
    )
    save_file(logical, checkpoint / "model.safetensors")
    layouts = {}
    for pp in range(2):
        for tr in range(tp):
            worker = pp * tp + tr
            layouts[worker] = {
                "tp_rank": tr,
                "pp_rank": pp,
                "start_layer": pp * 2,
                "end_layer": (pp + 1) * 2,
                "moe_ep_rank": 0,
                "moe_ep_size": 1,
                "moe_tp_rank": tr,
                "moe_tp_size": tp,
            }
            weights = {
                name: value
                for name, value in logical.items()
                if any(name.startswith(f"model.layers.{layer}.") for layer in range(pp * 2, (pp + 1) * 2))
            }
            key = "model.embed_tokens.weight" if pp == 0 else "lm_head.weight"
            weights[key] = logical[key].chunk(tp)[tr].clone()
            if pp == 1:
                weights["model.norm.weight"] = logical["model.norm.weight"]
            save_file(weights, dump / f"model-rank-{worker}-part-0.safetensors")
    assert verify_qwen3_dense(checkpoint, dump, tp_size=tp, pp_size=2, layouts=layouts)["logical_parameters"] == 7
    from safetensors.torch import load_file

    covered = set()
    for rank in range(tp * 2):
        weights = load_file(dump / f"model-rank-{rank}-part-0.safetensors")
        result = verify_qwen3_dense(
            checkpoint,
            None,
            tp_size=tp,
            pp_size=2,
            layouts=layouts,
            physical_tensors={(rank, name): value for name, value in weights.items()},
            worker_rank=rank,
        )
        covered.update(result["covered_logical_names"])
    assert covered == set(logical)
    for rank in range(tp, tp * 2):
        layouts[rank]["start_layer"] = 1
    with pytest.raises(ValueError, match="overlap or leave a gap"):
        verify_qwen3_dense(checkpoint, dump, tp_size=tp, pp_size=2, layouts=layouts)


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "qwen3_vl", "qwen3_vl_moe"])
def test_config_roundtrip_ignores_location_but_preserves_model_settings(model_type):
    from transformers import AutoConfig

    from relax.distributed.weight_sync.model_config import configuration_signature

    config = AutoConfig.for_model(model_type).to_dict()
    moved = json.loads(json.dumps(config))
    moved.update(_name_or_path="/another/checkpoint", _commit_hash="source-revision", transformers_version="old")
    assert configuration_signature(config) == configuration_signature(moved)
    text = moved.get("text_config", moved)
    text["hidden_size"] *= 2
    assert configuration_signature(config) != configuration_signature(moved)


@pytest.mark.parametrize("worker", [0, 1])
def test_pipeline_checks_every_vision_replica(tmp_path, worker):
    config = {
        "model_type": "qwen3_vl",
        "text_config": {"num_attention_heads": 2, "num_key_value_heads": 2, "num_hidden_layers": 2},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    vision = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    logical = {"model.visual.patch_embed.proj.weight": vision, "model.visual.pos_embed.weight": vision + 1}
    logical.update(
        {
            f"model.language_model.layers.{layer}.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16)
            for layer in range(2)
        }
    )
    save_file(logical, tmp_path / "model.safetensors")
    layouts = {
        rank: {
            "tp_rank": 0,
            "pp_rank": rank,
            "start_layer": rank,
            "end_layer": rank + 1,
            "moe_ep_rank": 0,
            "moe_ep_size": 1,
            "moe_tp_rank": 0,
            "moe_tp_size": 1,
        }
        for rank in range(2)
    }
    physical = {(0, "visual.pos_embed.weight"): (vision + 1).clone()}
    for rank in range(2):
        physical[rank, "visual.patch_embed.proj.weight"] = vision.clone()
        physical[rank, f"model.layers.{rank}.input_layernorm.weight"] = torch.ones(4, dtype=torch.bfloat16)
    result = verify_qwen3_dense(tmp_path, None, pp_size=2, layouts=layouts, physical_tensors=physical)
    assert result["physical_parameters"] == 5
    assert result["logical_parameters"] == 4
    physical[1, "visual.pos_embed.weight"] = (vision + 1).clone()
    with pytest.raises(ValueError, match="unexpected=.*pos_embed"):
        verify_qwen3_dense(tmp_path, None, pp_size=2, layouts=layouts, physical_tensors=physical)
    del physical[1, "visual.pos_embed.weight"]
    position = physical.pop((0, "visual.pos_embed.weight"))
    with pytest.raises(ValueError, match="missing=.*pos_embed"):
        verify_qwen3_dense(tmp_path, None, pp_size=2, layouts=layouts, physical_tensors=physical)
    physical[0, "visual.pos_embed.weight"] = position
    physical[worker, "visual.patch_embed.proj.weight"][0, 0] = -1
    with pytest.raises(ValueError, match="Physical weight mismatch"):
        verify_qwen3_dense(tmp_path, None, pp_size=2, layouts=layouts, physical_tensors=physical)
    del physical[worker, "visual.patch_embed.proj.weight"]
    with pytest.raises(ValueError, match="Incomplete vision parameter inventory"):
        verify_qwen3_dense(tmp_path, None, pp_size=2, layouts=layouts, physical_tensors=physical)
