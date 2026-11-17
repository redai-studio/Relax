# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run patched Bridge imports with real CPU tensors, without CUDA/TE imports.

Set QWEN38_FP8_MEGATRON_SOURCE to override the image-built Megatron source.
Only optional model construction and transport boundaries are stubbed. The
loader, safetensors reader, dequantization, mappings and capture execute
intact, including the VL registry's packed/per-expert MTP layout selection.
Small GPU fixtures use a reduced registry and miss that selection. These
fixtures add no GPU or multi-process CI cost.
"""

import importlib.util
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def patched_source():
    source = os.environ.get("QWEN38_FP8_MEGATRON_SOURCE")
    if source:
        return Path(source) / "megatron"
    spec = importlib.util.find_spec("megatron")
    locations = spec.submodule_search_locations if spec else None
    for path in locations or ():
        if (Path(path) / "bridge/models/qwen/qwen4_exp_bridge.py").is_file():
            return Path(path)
    pytest.skip("Requires the image-built Qwen3.8 Megatron Bridge")


@pytest.fixture
def api(monkeypatch, patched_source):
    owned = {}

    def install(name, **attributes):
        if name not in owned:
            module = ModuleType(name)
            module.__path__ = []
            owned[name] = module
            monkeypatch.setitem(sys.modules, name, module)
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(install(parent), child, module)
        owned[name].__dict__.update(attributes)
        return owned[name]

    def placeholder(name):
        return type(name, (), {})

    def load(relative):
        name = "megatron." + relative.replace("/", ".").removesuffix(".py")
        parent, child = name.rsplit(".", 1)
        parent_module = install(parent)
        spec = importlib.util.spec_from_file_location(name, patched_source / relative)
        module = importlib.util.module_from_spec(spec)
        owned[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        setattr(parent_module, child, module)
        spec.loader.exec_module(module)
        return module

    parallel = SimpleNamespace(is_initialized=lambda: False)
    install("megatron.core", mpu=parallel, parallel_state=parallel, DistributedDataParallel=torch.nn.Module)
    install("megatron.core.fp8_utils", FP8_TENSOR_CLASS=(), HAVE_TE_FP8_TENSOR_CLASS=False)
    install("megatron.core.transformer.module", MegatronModule=torch.nn.Module, Float16Module=torch.nn.Module)
    install("megatron.core.transformer.transformer_config", TransformerConfig=placeholder("TransformerConfig"))
    install(
        "megatron.core.utils",
        get_pg_rank=lambda group: 0 if group is None else group.rank(),
        get_pg_size=lambda group: 1 if group is None else group.size(),
        get_batch_on_this_cp_rank=lambda batch: batch,
    )
    install("megatron.core.pipeline_parallel.utils", is_pp_first_stage=lambda: True, is_pp_last_stage=lambda: True)
    install("megatron.training.models.base", ModelConfig=placeholder("ModelConfig"))
    install("transformers.configuration_utils", PretrainedConfig=placeholder("PretrainedConfig"))
    install("transformers.modeling_utils", PreTrainedModel=torch.nn.Module)
    install("megatron.bridge.models.common", ModelConfigOverrideMixin=placeholder("ModelConfigOverrideMixin"))
    install("megatron.core._rank_utils", safe_get_world_size=lambda: 1)
    install("megatron.core._slurm_utils", resolve_slurm_rank=lambda: None)
    install("megatron.training.utils.common_utils", get_local_rank_preinit=lambda: 0)
    install("megatron.bridge.training.utils.packed_seq_utils", get_packed_seq_cp_partition_indices=lambda *args: None)
    install(
        "megatron.bridge.utils.slurm_utils",
        resolve_slurm_master_addr=lambda: None,
        resolve_slurm_master_port=lambda: None,
    )
    common = load("bridge/utils/common_utils.py")
    monkeypatch.setattr(common, "print_rank_0", lambda *args: None)
    install(
        "megatron.bridge.models.conversion.peft_bridge",
        **{name: placeholder(name) for name in ("AdapterWeight", "AdapterWeightConversionTask", "MegatronPeftBridge")},
    )
    install("megatron.bridge.models.conversion.quant_bridge", MegatronQuantizationBridge=placeholder("QuantBridge"))
    install("megatron.bridge.models.conversion.transformers_compat", rope_theta_from_hf=lambda config: None)
    install("megatron.bridge.models.decorators.dispatch", dispatch=lambda function: function)
    install("megatron.bridge.models.gpt.model_config", BridgeGPTModelConfig=placeholder("BridgeGPTModelConfig"))
    install("megatron.bridge.models.model_provider", ModelProviderMixin=placeholder("ModelProviderMixin"))
    install("megatron.bridge.models.transformer_config", TransformerConfig=placeholder("BridgeTransformerConfig"))
    install("megatron.bridge.utils", fusions=SimpleNamespace())
    install("megatron.bridge.utils.activation_map", ACTIVATION_FUNC_MAP={})
    state = load("bridge/models/hf_pretrained/state.py")
    utils = load("bridge/models/conversion/utils.py")
    load("bridge/models/conversion/quantization_utils.py")
    mapping = load("bridge/models/conversion/param_mapping.py")
    load("bridge/models/conversion/quant_mapping.py")
    load("bridge/models/conversion/mapping_registry.py")
    bridge = load("bridge/models/conversion/model_bridge.py")
    # Registration/model construction and DDP wrappers require CUDA libraries.
    monkeypatch.setattr(
        bridge.MegatronModelBridge, "register_bridge", classmethod(lambda cls, **kw: lambda target: target)
    )
    monkeypatch.setattr(utils, "unwrap_model", lambda model, *args, **kw: model)
    monkeypatch.setattr(bridge, "unwrap_model", lambda model: model)
    monkeypatch.setattr(mapping, "is_modelopt_dynamic_module", lambda module: False)
    install(
        "megatron.core.models.gpt.experimental_attention_variant_module_specs",
        get_transformer_block_with_experimental_attention_variant_spec=lambda *args, **kw: None,
    )
    install("megatron.core.models.gpt.gpt_model", GPTModel=torch.nn.Module)
    install(
        "megatron.bridge.models.gpt_provider",
        GPTModelProvider=placeholder("GPTModelProvider"),
        mtp_block_spec=lambda *args, **kw: None,
    )
    install("megatron.bridge.models.qwen.qwen35_bridge", _moe_routed_expert_mappings=lambda *args, **kw: [])
    qwen = load("bridge/models/qwen/qwen4_exp_bridge.py")
    install("megatron.core.extensions.transformer_engine", TEColumnParallelLinear=torch.nn.Linear)
    install("megatron.core.models.hybrid.hybrid_block", HybridStackSubmodules=placeholder("HybridStackSubmodules"))
    install(
        "megatron.core.transformer.hyper_connection",
        GatedResidualMTPHiddenNorm=torch.nn.Module,
        GatedResidualOutputMixer=torch.nn.Module,
    )
    install("megatron.core.transformer.spec_utils", ModuleSpec=placeholder("ModuleSpec"))
    install(
        "megatron.core.transformer.transformer_block",
        TransformerBlockSubmodules=placeholder("TransformerBlockSubmodules"),
    )
    install("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model", Qwen3VLModel=torch.nn.Module)
    install("megatron.bridge.models.qwen_vl.qwen35_vl_bridge", _get_vision_mappings=lambda: [])
    install("megatron.bridge.models.qwen_vl.qwen35_vl_provider", Qwen35VLMoEModelProvider=placeholder("VLProvider"))
    qwen_vl = load("bridge/models/qwen/qwen4_exp_vl_bridge.py")
    return SimpleNamespace(state=state, mapping=mapping, bridge=bridge, qwen=qwen, qwen_vl=qwen_vl, parallel=parallel)


def _bridge(api):
    bridge = api.qwen.Qwen4ExpBridge()
    bridge.hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})
    )
    return bridge


def _quantized(shape=(256, 256)):
    # This product is exact in FP32 and cannot be represented exactly in BF16.
    weight = torch.full(shape, 1.125).to(torch.float8_e4m3fn)
    scale = torch.full(tuple((dim + 127) // 128 for dim in shape), 1.0078125, dtype=torch.bfloat16)
    scale[-1, -1] = 0.0030670166015625
    expected = (
        weight.float() * scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: shape[0], : shape[1]]
    )
    return weight, scale, expected


@pytest.mark.parametrize("prefix", ["model.language_model.layers.0", "mtp.layers.0"])
def test_fp8_experts_use_original_block_scales(api, prefix):
    gate_name = f"{prefix}.mlp.experts.0.gate_proj.weight"
    up_name = f"{prefix}.mlp.experts.0.up_proj.weight"
    weight, scale, expected = _quantized((130, 258))
    state = {gate_name: weight, gate_name + "_scale_inv": scale, up_name: weight, up_name + "_scale_inv": scale}
    bridge = _bridge(api)
    actual = bridge.maybe_modify_loaded_hf_weight({"gate": gate_name, "up": up_name}, state)
    for tensor in actual.values():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    assert torch.any(expected != expected.bfloat16().float())
    assert bridge.get_hf_import_param_names(gate_name, set(state)) == (gate_name, gate_name + "_scale_inv")


@pytest.mark.parametrize("invalid", ["missing", "small_grid", "large_grid", "nan", "zero", "negative", "block_size"])
def test_fp8_import_rejects_invalid_scales(api, invalid):
    weight, scale, _ = _quantized()
    bridge = _bridge(api)
    state = {"weight": weight, "weight_scale_inv": scale}
    if invalid == "missing":
        del state["weight_scale_inv"]
    elif invalid == "small_grid":
        state["weight_scale_inv"] = scale[:1]
    elif invalid == "large_grid":
        state["weight_scale_inv"] = torch.ones(3, 2)
    elif invalid == "block_size":
        bridge.hf_pretrained.config.quantization_config["weight_block_size"] = [32, 32]
    else:
        scale[0, 0] = {"nan": float("nan"), "zero": 0, "negative": -1}[invalid]
    with pytest.raises(ValueError, match="scale|block_size"):
        bridge.maybe_modify_loaded_hf_weight("weight", state)


def test_bf16_source_is_not_dequantized(api):
    weight = torch.randn(5, 7).bfloat16()
    bridge = _bridge(api)
    assert bridge.maybe_modify_loaded_hf_weight("weight", {"weight": weight}) is weight


@pytest.mark.parametrize("packed", [False, True], ids=["fp8-split", "bf16-packed"])
@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "local-experts"])
def test_vl_registry_imports_mtp_experts_independently_of_backbone(api, monkeypatch, tmp_path, packed, grouped):
    weight, scale, full = _quantized((128, 128))
    prefix = "mtp.layers.0.mlp.experts."
    if packed:
        gate, up, down = full.bfloat16(), (full * 2).bfloat16(), (full * 4).bfloat16()
        fused = torch.cat((gate, up))
        state = {
            prefix + "gate_up_proj": torch.stack((torch.zeros_like(fused), fused)),
            prefix + "down_proj": torch.stack((torch.zeros_like(down), down)),
        }
        backbone_key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    else:
        gate, up, down = full, full * 2, full * 4
        state = {}
        for projection, multiplier in (("gate", 1), ("up", 2), ("down", 4)):
            name = prefix + f"1.{projection}_proj.weight"
            state[name], state[name + "_scale_inv"] = weight.clone(), scale * multiplier
        backbone_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    state[backbone_key] = torch.zeros(1)
    save_file(state, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in state}})
    )
    source = api.state.StateDict(api.state.SafeTensorsStateSource(tmp_path))
    bridge = api.qwen_vl.Qwen4ExpVLBridge()
    bridge.hf_pretrained = SimpleNamespace(
        state=source,
        config=SimpleNamespace(
            text_config=SimpleNamespace(mtp_num_hidden_layers=1, split_ngram_parts=1),
            quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
        ),
    )
    registry = bridge.mapping_registry()
    for projection, expected in (("fc1", torch.cat((gate, up))), ("fc2", down)):
        expert_path = (
            f"experts.linear_{projection}.weight1"
            if grouped
            else f"experts.local_experts.1.linear_{projection}.weight"
        )
        global_name = "language_model.mtp.layers.0.mtp_model_layer.layers.0.mlp." + expert_path
        mapping = registry.megatron_to_hf_lookup(global_name)
        assert mapping is not None
        module = torch.nn.Module()
        local_name = "weight0" if grouped else "weight"
        parameter = torch.nn.Parameter(torch.zeros_like(expected, dtype=torch.bfloat16))
        module.register_parameter(local_name, parameter)
        module.tensor_model_parallel, module.partition_dim = True, 0 if projection == "fc1" else 1
        task = api.bridge.WeightConversionTask(
            local_name, global_name, mapping, megatron_module=module, param_weight=parameter
        )
        monkeypatch.setattr(bridge, "build_conversion_tasks", lambda *args: [task])
        monkeypatch.setattr(bridge, "_with_progress_tracking", lambda tasks, description: tasks)
        monkeypatch.setattr(bridge, "finalize_hf_import", lambda model: None)
        bridge.export_weight_dtype = "fp8"
        bridge.load_weights_hf_to_megatron(SimpleNamespace(state=source, model_name_or_path=str(tmp_path)), [module])
        captured = bridge.unquantized_state_dict["model"][local_name]
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        torch.testing.assert_close(parameter, expected.bfloat16(), rtol=0, atol=0)


def test_vl_registry_rejects_mixed_mtp_expert_layouts(api):
    bridge = api.qwen_vl.Qwen4ExpVLBridge()
    bridge.hf_pretrained = SimpleNamespace(
        state=SimpleNamespace(
            source=SimpleNamespace(
                get_all_keys=lambda: [
                    "mtp.layers.0.mlp.experts.gate_up_proj",
                    "mtp.layers.0.mlp.experts.0.gate_proj.weight",
                ]
            )
        ),
        config=SimpleNamespace(text_config=SimpleNamespace(mtp_num_hidden_layers=1, split_ngram_parts=1)),
    )
    with pytest.raises(ValueError, match="mixes fused and per-expert.*mtp"):
        bridge.mapping_registry()


@pytest.mark.parametrize(
    "mapping_type", ["ColumnParallelMapping", "RowParallelMapping", "GatedMLPMapping", "AutoMapping"]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_fp32_products_survive_tp_mapping_and_master_capture(api, monkeypatch, tmp_path, mapping_type, rank):
    weight, scale, full = _quantized()
    name = "mtp.layers.0.mlp.experts.0.down_proj.weight" if mapping_type == "RowParallelMapping" else "weight"
    state = {name: weight, name + "_scale_inv": scale}
    if mapping_type == "GatedMLPMapping":
        state.update(up=weight.clone(), up_scale_inv=scale * 2)
        hf_param = {"gate": name, "up": "up"}
        local = full[rank * 128 : (rank + 1) * 128]
        expected = torch.cat((local, local * 2), dim=0)
    else:
        hf_param = name
        expected = (
            full[:, rank * 128 : (rank + 1) * 128]
            if mapping_type == "RowParallelMapping"
            else full[rank * 128 : (rank + 1) * 128]
        )
    save_file(state, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in state}})
    )
    source = api.state.StateDict(api.state.SafeTensorsStateSource(tmp_path))
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.zeros_like(expected, dtype=torch.bfloat16))
    module.tensor_model_parallel, module.partition_dim = True, 0
    mapping_class = getattr(api.mapping, mapping_type)
    mapping = mapping_class("weight", **hf_param) if isinstance(hf_param, dict) else mapping_class("weight", hf_param)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    mapping.set_process_groups_from_pg_collection(SimpleNamespace(tp=group, expt_tp=group))

    def scatter(output, scatter_list, *, src, group):
        assert output.dtype == torch.float32
        if rank == 0:
            assert all(piece.dtype == output.dtype for piece in scatter_list)
            output.copy_(scatter_list[rank])
        else:
            assert scatter_list is None
            output.copy_(expected)

    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda **kw: 0)
    monkeypatch.setattr(torch.distributed, "scatter", scatter)
    bridge = _bridge(api)
    task = api.bridge.WeightConversionTask(
        "weight", "weight", mapping, megatron_module=module, param_weight=module.weight
    )
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda *args: [task])
    monkeypatch.setattr(bridge, "_with_progress_tracking", lambda tasks, description: tasks)
    monkeypatch.setattr(bridge, "finalize_hf_import", lambda model: None)
    bridge.export_weight_dtype = "fp8"
    bridge.load_weights_hf_to_megatron(SimpleNamespace(state=source, model_name_or_path=str(tmp_path)), [module])
    captured = bridge.unquantized_state_dict["model"]["weight"]
    assert captured.device.type == "cpu" and captured.dtype == torch.float32
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    torch.testing.assert_close(module.weight, expected.bfloat16(), rtol=0, atol=0)
    assert torch.any(captured != module.weight.float())
    with torch.no_grad():
        module.weight.zero_()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_import_dtype_is_scoped_and_restored_on_error(api):
    mapping = api.mapping.ColumnParallelMapping("weight", "weight")
    tensor = torch.tensor([1.1337890625])

    def convert():
        return mapping.scatter_to_tp_ranks([tensor], tensor.shape, torch.bfloat16, tensor.device)

    assert convert().dtype == torch.bfloat16
    with api.mapping.hf_import_dtype(torch.float32):
        torch.testing.assert_close(convert(), tensor, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="test failure"), api.mapping.hf_import_dtype(torch.float16):
            assert convert().dtype == torch.float16
            raise RuntimeError("test failure")
        assert convert().dtype == torch.float32
    assert convert().dtype == torch.bfloat16


def _ple_state():
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    shards = {f"shard_{i}": f"{prefix}.shard_{i}.weight" for i in range(3)}
    full = torch.arange(1, 11).reshape(-1, 1).expand(-1, 3).to(torch.float8_e4m3fn)
    state = {name: full[4 * i : 4 * i + 4] for i, name in enumerate(shards.values())}
    state[prefix + ".weight_scale"] = torch.tensor([1.0078125], dtype=torch.bfloat16)
    return shards, state, prefix + ".weight_scale"


@pytest.mark.parametrize("rank", [0, 1])
def test_ple_rank_local_import_preserves_fp32_and_final_partial_shard(api, monkeypatch, rank):
    shards, state, scale_name = _ple_state()
    mapping = api.qwen.PLENGramEmbeddingMapping("weight", shards)
    monkeypatch.setattr(api.parallel, "is_initialized", lambda: True)
    monkeypatch.setattr(api.parallel, "get_tensor_model_parallel_rank", lambda: rank, raising=False)
    monkeypatch.setattr(api.parallel, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    bridge = _bridge(api)
    rows = bridge.maybe_modify_loaded_hf_weight(shards, state)
    expected = torch.cat([state[name].float() for name in shards.values()]) * state[scale_name].float()
    module = torch.nn.Embedding(5, 3, dtype=torch.bfloat16)
    module._table_loaded = False
    actual = mapping.hf_to_megatron(rows, module)
    assert actual.device.type == "cpu" and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected[rank * 5 : (rank + 1) * 5], rtol=0, atol=0)
    assert module._table_loaded
    assert scale_name in bridge.get_hf_import_param_names(shards, set(state))


@pytest.mark.parametrize("invalid", [None, torch.ones(2), torch.tensor([float("nan")]), torch.tensor([0.0])])
def test_ple_invalid_scale_fails_before_model_copy(api, invalid):
    shards, state, scale_name = _ple_state()
    if invalid is None:
        del state[scale_name]
    else:
        state[scale_name] = invalid
    with pytest.raises(ValueError, match="scale"):
        _bridge(api).maybe_modify_loaded_hf_weight(shards, state)


@pytest.fixture
def precision():
    spec = importlib.util.spec_from_file_location("qwen38_precision_cpu", ROOT / "relax/models/qwen3_8/precision.py")
    precision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(precision)
    return precision


@pytest.mark.parametrize("source", ["bf16", "other_model", "fp8_patched", "fp8_unpatched"])
def test_fp8_source_guard_requires_scale_aware_bridge(precision, source):
    config = SimpleNamespace(
        model_type="unrelated" if source == "other_model" else "qwen4_exp",
        quantization_config=None if source == "bf16" else {"quant_method": "fp8"},
    )
    bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(config=config),
        _model_bridge=SimpleNamespace(supports_fp8_hf_import=source == "fp8_patched"),
    )
    if source == "fp8_unpatched":
        with pytest.raises(RuntimeError, match="scale-aware Megatron Bridge"):
            precision.validate_fp8_hf_import(bridge)
    else:
        assert precision.validate_fp8_hf_import(bridge) is (source == "fp8_patched")


@pytest.mark.parametrize(
    "case",
    ["supported", "weights_only", "non_distributed", "internal_master", "cpu_offload", "low_precision_master", "fsdp"],
)
def test_fp8_master_guard_rejects_optimizers_that_ignore_captured_values(precision, case):
    bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(
            config=SimpleNamespace(model_type="qwen4_exp", quantization_config={"quant_method": "fp8"})
        ),
        _model_bridge=SimpleNamespace(supports_fp8_hf_import=True),
    )
    config = SimpleNamespace(
        use_distributed_optimizer=case != "non_distributed",
        main_params_dtype=torch.bfloat16 if case == "low_precision_master" else torch.float32,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=case == "internal_master",
        optimizer_cpu_offload=case == "cpu_offload",
    )
    child = SimpleNamespace(config=config, ddp_config=SimpleNamespace(use_megatron_fsdp=case == "fsdp"))
    optimizer = None if case == "weights_only" else SimpleNamespace(chained_optimizers=[child])
    if case in {"supported", "weights_only"}:
        assert precision.validate_fp8_hf_import(bridge, optimizer)
    else:
        with pytest.raises(ValueError, match="distributed FP32 masters"):
            precision.validate_fp8_hf_import(bridge, optimizer)


@pytest.fixture
def checkpoint_api(monkeypatch):
    def stub(name, path=None, **attributes):
        module = ModuleType(name)
        module.__path__ = [] if path is None else [str(path)]
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub("megatron")
    stub("megatron.training")
    stub("megatron.training.checkpointing", load_checkpoint=lambda: None, save_checkpoint=lambda: None)
    stub("megatron.training.global_vars", get_args=lambda: None)
    stub("relax.backends.megatron", ROOT / "relax/backends/megatron")
    stub("relax.models", ROOT / "relax/models")
    stub(
        "relax.utils.training.ppo_utils",
        use_critic_lm_head_for_hf_load=lambda model: nullcontext(),
        use_sequence_classification_lm_head_for_hf_load=lambda model: nullcontext(),
    )
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    # Undo the module's existing global PyTorch patches when the test finishes.
    monkeypatch.setattr(EnumerableShardingSpec, "__post_init__", EnumerableShardingSpec.__dict__["__post_init__"])
    name = "_init_from_local_shards_and_global_metadata"
    monkeypatch.setattr(ShardedTensor, name, ShardedTensor.__dict__[name])
    spec = importlib.util.spec_from_file_location(
        "relax.backends.megatron.checkpoint_cpu_test", ROOT / "relax/backends/megatron/checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.megatron_bridge_utils, "patch_megatron_model", lambda model: nullcontext())
    return module


@pytest.mark.parametrize("case", ["fp8_source", "fp8_storage", "bf16", "weights_only", "capture_missing"])
def test_hf_loader_passes_original_values_to_optimizer_and_releases_capture(checkpoint_api, monkeypatch, case):
    source_fp8 = case in {"fp8_source", "weights_only", "capture_missing"}
    captured = {"model": {"weight": torch.tensor([1.1337890625])}}
    bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(
            config=SimpleNamespace(
                model_type="qwen4_exp", quantization_config={"quant_method": "fp8"} if source_fp8 else None
            )
        ),
        _model_bridge=SimpleNamespace(supports_fp8_hf_import=True),
        export_weight_dtype="bf16",
    )
    want_capture = case in {"fp8_source", "fp8_storage", "capture_missing"}

    def load_hf_weights(model):
        assert (bridge.export_weight_dtype == "fp8") == want_capture
        bridge.unquantized_state_dict = captured if want_capture and case != "capture_missing" else None

    bridge.load_hf_weights = load_hf_weights
    module = ModuleType("megatron.bridge")
    module.AutoBridge = SimpleNamespace(from_hf_pretrained=lambda *args, **kw: bridge)
    monkeypatch.setitem(sys.modules, "megatron.bridge", module)
    optimizer = SimpleNamespace(
        config=SimpleNamespace(
            use_distributed_optimizer=True,
            main_params_dtype=torch.float32,
            use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        ),
        reload_model_params=Mock(),
    )
    args = SimpleNamespace(
        megatron_to_hf_mode="bridge",
        hf_checkpoint="source",
        ref_load=None,
        fp8_param_gather=case == "fp8_storage",
        fp16=False,
        bf16=True,
        load_main_params_from_ckpt=False,
    )
    if case == "capture_missing":
        with pytest.raises(RuntimeError, match="unquantized HF weight capture"):
            checkpoint_api._load_checkpoint_hf([], optimizer, args, None)
        optimizer.reload_model_params.assert_not_called()
        return
    result = checkpoint_api._load_checkpoint_hf([], None if case == "weights_only" else optimizer, args, None)
    assert result == (0, 0)
    if want_capture:
        optimizer.reload_model_params.assert_called_once_with(state_dict=captured)
        assert optimizer.reload_model_params.call_args.kwargs["state_dict"] is captured
    elif case == "bf16":
        optimizer.reload_model_params.assert_called_once_with()
    else:
        optimizer.reload_model_params.assert_not_called()
    assert bridge.unquantized_state_dict is None
