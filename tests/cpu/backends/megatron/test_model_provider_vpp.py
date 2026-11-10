import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from relax.utils.misc import resolve_is_vl_model
from relax.utils.training import ppo_utils


class _FakeProvider:
    def __init__(self, model=None):
        self.calls = []
        self.distributed_calls = []
        self.finalized = False
        self.model = model or torch.nn.Module()
        self.pre_wrap_hooks = []
        self.attention_backend = None
        self.tensor_model_parallel_size = 1
        self.sequence_parallel = False
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.variable_seq_lengths = False
        self.num_layers = 8
        self.moe_layer_freq = None
        self.fp16 = False
        self.bf16 = False
        self.params_dtype = None
        self.vision_dp_when_cp = False
        self.qsa_indexer_loss_coeff = 0.0
        self.deallocate_pipeline_outputs = True

    def finalize(self):
        self.finalized = True

    def provide(self, pre_process=True, post_process=True, vp_stage=None):
        self.calls.append(
            {
                "pre_process": pre_process,
                "post_process": post_process,
                "vp_stage": vp_stage,
            }
        )
        return self.model

    def register_pre_wrap_hook(self, hook):
        self.pre_wrap_hooks.append(hook)

    def provide_distributed_model(self, **kwargs):
        self.distributed_calls.append(kwargs)
        # Megatron-Bridge hands pre-wrap hooks the list of model chunks, not a bare module.
        model_chunks = self.model if isinstance(self.model, list) else [self.model]
        for hook in self.pre_wrap_hooks:
            model_chunks = hook(model_chunks)
        return model_chunks


def _install_fake_megatron(monkeypatch, provider=None):
    provider = provider or _FakeProvider()

    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    optimizer = types.ModuleType("megatron.core.optimizer")
    mpu = types.ModuleType("megatron.core.mpu")
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    models = types.ModuleType("megatron.core.models")
    gpt = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    transformer = types.ModuleType("megatron.core.transformer")
    multi_token_prediction = types.ModuleType("megatron.core.transformer.multi_token_prediction")
    spec_utils = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_config = types.ModuleType("megatron.core.transformer.transformer_config")
    training = types.ModuleType("megatron.training")
    arguments = types.ModuleType("megatron.training.arguments")
    tokenizer_package = types.ModuleType("megatron.training.tokenizer")
    tokenizer = types.ModuleType("megatron.training.tokenizer.tokenizer")
    training_training = types.ModuleType("megatron.training.training")
    bridge = types.ModuleType("megatron.bridge")
    misc = types.ModuleType("relax.utils.misc")
    bridge_training = types.ModuleType("megatron.bridge.training")
    bridge_training_utils = types.ModuleType("megatron.bridge.training.utils")
    bridge_config_utils = types.ModuleType("megatron.bridge.training.utils.config_utils")
    ray = types.ModuleType("ray")

    class _FakeGPTModel:
        pass

    class _FakeTransformerConfig:
        pass

    class _FakeMultiTokenPredictionBlock(torch.nn.Module):
        pass

    class _FakeAutoBridge:
        @classmethod
        def from_hf_pretrained(cls, *args, **kwargs):
            return cls()

        def to_megatron_provider(self, load_weights=False):
            return provider

    mpu.get_virtual_pipeline_model_parallel_world_size = lambda: 2
    mpu.get_virtual_pipeline_model_parallel_rank = lambda: 1
    mpu.get_context_parallel_world_size = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.get_tensor_model_parallel_rank = lambda: 0
    core.mpu = mpu
    core.tensor_parallel = tensor_parallel
    optimizer.OptimizerConfig = SimpleNamespace
    gpt.GPTModel = _FakeGPTModel
    gpt_layer_specs.get_gpt_decoder_block_spec = lambda *args, **kwargs: object()
    gpt_layer_specs.get_gpt_layer_local_spec = lambda *args, **kwargs: object()
    gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec = lambda *args, **kwargs: object()
    spec_utils.import_module = lambda path: object()
    transformer_config.TransformerConfig = _FakeTransformerConfig
    multi_token_prediction.MultiTokenPredictionBlock = _FakeMultiTokenPredictionBlock
    arguments.core_transformer_config_from_args = lambda args: _FakeTransformerConfig()
    arguments.parse_args = lambda *args, **kwargs: None
    arguments.validate_args = lambda *args, **kwargs: None
    tokenizer._vocab_size_with_padding = lambda *args, **kwargs: None
    bridge.AutoBridge = _FakeAutoBridge
    misc.load_function = lambda path: None
    misc.resolve_is_vl_model = resolve_is_vl_model
    # Stub the real megatron.training.training.get_megatron_ddp_config: it reads a long tail
    # of Megatron-parser-only args off the namespace, which these fake args don't carry.
    training_training.get_megatron_ddp_config = lambda args: {"fake_ddp_config_for": args.megatron_to_hf_mode}

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.optimizer": optimizer,
        "megatron.core.mpu": mpu,
        "megatron.core.tensor_parallel": tensor_parallel,
        "megatron.core.models": models,
        "megatron.core.models.gpt": gpt,
        "megatron.core.models.gpt.gpt_layer_specs": gpt_layer_specs,
        "megatron.core.transformer": transformer,
        "megatron.core.transformer.multi_token_prediction": multi_token_prediction,
        "megatron.core.transformer.spec_utils": spec_utils,
        "megatron.core.transformer.transformer_config": transformer_config,
        "megatron.training": training,
        "megatron.training.arguments": arguments,
        "megatron.training.tokenizer": tokenizer_package,
        "megatron.training.tokenizer.tokenizer": tokenizer,
        "megatron.training.training": training_training,
        "megatron.bridge": bridge,
        "relax.utils.misc": misc,
        "megatron.bridge.training": bridge_training,
        "megatron.bridge.training.utils": bridge_training_utils,
        "megatron.bridge.training.utils.config_utils": bridge_config_utils,
        "ray": ray,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return provider


def _load_model_provider(monkeypatch, provider=None):
    provider = _install_fake_megatron(monkeypatch, provider=provider)
    sys.modules.pop("relax.backends.megatron.model_provider", None)
    module = importlib.import_module("relax.backends.megatron.model_provider")
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 1)
    return module, provider


def _bridge_args(**overrides):
    values = {
        "megatron_to_hf_mode": "bridge",
        "hf_checkpoint": "fake-hf",
        "attention_backend": "flash",
        "tensor_model_parallel_size": 2,
        "sequence_parallel": True,
        "pipeline_model_parallel_size": 4,
        "virtual_pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "variable_seq_lengths": True,
        "dsa_indexer_loss_coeff": None,
        "dsa_indexer_use_sparse_loss": None,
        "attention_softmax_in_fp32": True,
        "bias_dropout_fusion": True,
        "apply_rope_fusion": False,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "distribute_saved_activations": False,
        "moe_router_load_balancing_type": "none",
        "moe_router_dtype": None,
        "moe_aux_loss_coeff": None,
        "moe_token_dispatcher_type": "alltoall",
        "moe_shared_expert_overlap": False,
        "moe_enable_deepep": False,
        "moe_flex_dispatcher_backend": None,
        "use_audio_in_video": False,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "vision_dp_when_tp": False,
        "vision_dp_when_cp": False,
        "calculate_per_token_loss": False,
        "num_layers": 8,
        "moe_layer_freq": None,
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "fp16": False,
        "bf16": True,
        "save": None,
        "use_distributed_optimizer": True,
        "use_megatron_fsdp": False,
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bridge_provider_receives_virtual_pipeline_size(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    model_provider = module.get_model_provider_func(_bridge_args(), role="actor")
    model_provider(pre_process=True, post_process=False, vp_stage=1)

    assert provider.virtual_pipeline_model_parallel_size == 2
    assert provider.finalized
    assert provider.calls == [{"pre_process": True, "post_process": False, "vp_stage": 1}]


@pytest.mark.parametrize("inherited_backend", ["none", "cudnn"])
@pytest.mark.parametrize(
    "backend,legacy_fusion,expected_backend",
    [(None, None, "cudnn"), (None, False, "none"), ("none", None, "none"), ("cudnn", None, "cudnn")],
)
def test_dsv4_bridge_resolves_dsa_backend_before_finalize(
    monkeypatch, inherited_backend, backend, legacy_fusion, expected_backend
):
    provider = _FakeProvider()
    provider.experimental_attention_variant = "dsv4_hybrid"
    provider.dsa_kernel_backend = inherited_backend
    provider.apply_dsa_kernel_fusion = inherited_backend == "cudnn"
    module, provider = _load_model_provider(monkeypatch, provider)

    def finalize():
        assert provider.dsa_kernel_backend == expected_backend
        assert provider.apply_dsa_kernel_fusion is legacy_fusion

    provider.finalize = finalize
    module.get_model_provider_func(
        _bridge_args(pipeline_model_parallel_size=1, dsa_kernel_backend=backend, apply_dsa_kernel_fusion=legacy_fusion)
    )


@pytest.mark.parametrize("path", ["bridge", "fsdp", "custom", "raw"])
@pytest.mark.parametrize("processor", [False, True])
@pytest.mark.parametrize("declared", [None, False, True])
def test_model_routing_is_resolved_before_model_construction(monkeypatch, tmp_path, path, processor, declared):
    if processor:
        (tmp_path / "processor_config.json").write_text("{}")
    provider = _FakeProvider()
    provider.relax_is_multimodal = declared
    module, provider = _load_model_provider(monkeypatch, provider)
    args = _bridge_args(
        hf_checkpoint=str(tmp_path),
        megatron_to_hf_mode="raw" if path == "raw" else "bridge",
        custom_model_provider_path="custom.provider" if path == "custom" else None,
        is_vl_model=not processor,
        uses_unsplit_forward=True,
    )
    if path == "fsdp":
        module.get_bridge_distributed_model(args)
    else:
        module.get_model_provider_func(args)
    expected = declared if path in ("bridge", "fsdp") and declared is not None else processor
    assert args.is_vl_model is expected
    assert args.uses_unsplit_forward is True
    assert provider.calls == []


def test_bridge_provider_receives_vision_dp_when_cp(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    model_provider = module.get_model_provider_func(_bridge_args(vision_dp_when_cp=True), role="actor")
    model_provider(pre_process=True, post_process=True)

    assert provider.vision_dp_when_cp is True


def test_bridge_provider_maps_legacy_vision_dp_when_tp_to_cp(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    model_provider = module.get_model_provider_func(_bridge_args(vision_dp_when_tp=True), role="actor")
    model_provider(pre_process=True, post_process=True)

    assert provider.vision_dp_when_cp is True


def test_bridge_vision_cp_all_gather_compat_forwards_cp_group_positionally(monkeypatch):
    class _OriginalAllGatherVisionEmbeddings:
        calls = []

        @staticmethod
        def apply(*args):
            _OriginalAllGatherVisionEmbeddings.calls.append(args)
            return "gathered"

    bridge_model_module = SimpleNamespace(AllGatherVisionEmbeddings=_OriginalAllGatherVisionEmbeddings)
    module, _ = _load_model_provider(monkeypatch)

    assert module._patch_bridge_vision_cp_all_gather(bridge_model_module, _OriginalAllGatherVisionEmbeddings)
    assert bridge_model_module.AllGatherVisionEmbeddings.apply("input", "seqlens", cp_group="cp") == "gathered"
    assert _OriginalAllGatherVisionEmbeddings.calls == [("input", "seqlens", "cp")]
    assert not module._patch_bridge_vision_cp_all_gather(bridge_model_module, _OriginalAllGatherVisionEmbeddings)


def test_bridge_critic_provider_registers_value_head_before_ddp(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    bridge_model = _FakeBridgeModel()
    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(bridge_model))

    model_provider = module.get_model_provider_func(_bridge_args(), role="critic")
    model = model_provider(pre_process=True, post_process=True)

    assert isinstance(model.output_layer, ppo_utils.LinearForLastLayer)
    assert model.output_layer.out_features == 1
    assert ppo_utils._RELAX_HF_OUTPUT_LAYER_ATTR not in model._modules
    assert all("relax_hf_output_layer" not in name for name, _ in model.named_parameters())


def test_bridge_reward_model_provider_registers_biasless_scalar_head_and_restores_on_error(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    args = _bridge_args(loss_type="rm")
    model = module.get_model_provider_func(args, role="actor")(post_process=True)
    reward_head = model.output_layer
    parameter_ids = tuple(id(parameter) for parameter in reward_head.parameters())

    assert isinstance(reward_head, ppo_utils.LinearForLastLayer)
    assert tuple(reward_head.weight.shape) == (1, 4)
    assert reward_head.bias is None
    assert list(model.state_dict()) == ["output_layer.weight"]

    with pytest.raises(RuntimeError, match="bridge failed"):
        with ppo_utils.use_critic_lm_head_for_hf_load([model]):
            assert model.output_layer.out_features == 8
            raise RuntimeError("bridge failed")

    assert model.output_layer is reward_head
    assert tuple(id(parameter) for parameter in reward_head.parameters()) == parameter_ids
    assert not hasattr(model, ppo_utils._RELAX_HF_OUTPUT_LAYER_ATTR)

    reward_head.weight.main_grad = torch.zeros_like(reward_head.weight)
    assert ppo_utils.validate_reward_model_head_registration([model], object()) == parameter_ids


def test_hf_load_context_restores_same_value_head(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(_bridge_args(), role="critic")(post_process=True)
    value_head = model.output_layer
    value_param_ids = tuple(id(param) for param in value_head.parameters())
    lm_head = getattr(model, ppo_utils._RELAX_HF_OUTPUT_LAYER_ATTR)

    with ppo_utils.use_critic_lm_head_for_hf_load([model]):
        assert model.output_layer is lm_head

    assert model.output_layer is value_head
    assert tuple(id(param) for param in model.output_layer.parameters()) == value_param_ids
    assert not hasattr(model, ppo_utils._RELAX_HF_OUTPUT_LAYER_ATTR)


def test_hf_load_context_restores_value_head_on_error(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(_bridge_args(), role="critic")(post_process=True)
    value_head = model.output_layer

    with pytest.raises(RuntimeError, match="bridge failed"):
        with ppo_utils.use_critic_lm_head_for_hf_load([model]):
            raise RuntimeError("bridge failed")

    assert model.output_layer is value_head
    assert not hasattr(model, ppo_utils._RELAX_HF_OUTPUT_LAYER_ATTR)


def test_bridge_actor_provider_registers_sequence_classification_head(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(
        _bridge_args(task_type="seq_cls", num_labels=3),
        role="actor",
    )(post_process=True)

    assert isinstance(model.output_layer, ppo_utils.LinearForLastLayer)
    assert model.output_layer.weight.shape == (3, 4)
    assert model.output_layer.bias is None
    assert ppo_utils._RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR not in model._modules
    assert all("relax_seq_cls_hf_output_layer" not in name for name, _ in model.named_parameters())


def test_hf_load_context_restores_same_sequence_classification_head(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(
        _bridge_args(task_type="seq_cls", num_labels=3),
        role="actor",
    )(post_process=True)
    classification_head = model.output_layer
    classification_param_ids = tuple(id(param) for param in classification_head.parameters())
    lm_head = getattr(model, ppo_utils._RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR)

    with ppo_utils.use_sequence_classification_lm_head_for_hf_load([model]):
        assert model.output_layer is lm_head

    assert model.output_layer is classification_head
    assert tuple(id(param) for param in model.output_layer.parameters()) == classification_param_ids
    assert not hasattr(model, ppo_utils._RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR)


def test_critic_value_head_validation_accepts_ddp_and_optimizer_ownership(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    class _FakeDDP(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(_bridge_args(), role="critic")(post_process=True)
    ddp = _FakeDDP(model)
    head_params = list(model.output_layer.parameters())
    for param in head_params:
        param.main_grad = torch.zeros_like(param)

    plain_optimizer = SimpleNamespace(param_groups=[{"params": head_params}])
    float_optimizer = SimpleNamespace(float16_groups=[head_params], fp32_from_fp32_groups=[])
    mixed_optimizer = SimpleNamespace(model_float16_groups=[head_params], model_fp32_groups=[])
    chained_optimizer = SimpleNamespace(chained_optimizers=[mixed_optimizer])

    expected_ids = tuple(id(param) for param in head_params)
    assert ppo_utils.validate_critic_value_head_registration([ddp], plain_optimizer) == expected_ids
    assert ppo_utils.validate_critic_value_head_registration([ddp], float_optimizer) == expected_ids
    assert ppo_utils.validate_critic_value_head_registration([ddp], chained_optimizer) == expected_ids


def test_critic_value_head_validation_accepts_distributed_optimizer_remote_shard(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(_bridge_args(), role="critic")(post_process=True)
    head_params = list(model.output_layer.parameters())
    for param in head_params:
        param.main_grad = torch.zeros_like(param)

    optimizer = SimpleNamespace(
        model_param_gbuf_map={param: (0, param.dtype, 0) for param in head_params},
        model_float16_groups=[],
        model_fp32_groups=[],
    )

    expected_ids = tuple(id(param) for param in head_params)
    assert ppo_utils.validate_critic_value_head_registration([model], optimizer) == expected_ids


def test_critic_value_head_validation_accepts_optimizer_main_params(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(_bridge_args(), role="critic")(post_process=True)
    head_params = list(model.output_layer.parameters())
    for param in head_params:
        param.main_grad = torch.zeros_like(param)
        param.main_param = torch.nn.Parameter(param.detach().float())

    optimizer = SimpleNamespace(
        model_float16_groups=[],
        model_fp32_groups=[],
        param_groups=[{"params": [param.main_param for param in head_params]}],
    )

    expected_ids = tuple(id(param) for param in head_params)
    assert ppo_utils.validate_critic_value_head_registration([model], optimizer) == expected_ids


def test_critic_value_head_validation_rejects_missing_ddp_ownership(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(_bridge_args(), role="critic")(post_process=True)
    optimizer = SimpleNamespace(param_groups=[{"params": list(model.output_layer.parameters())}])

    with pytest.raises(AssertionError, match="DDP"):
        ppo_utils.validate_critic_value_head_registration([model], optimizer)


def test_critic_value_head_movement_detection(monkeypatch):
    class _FakeBridgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
            self.output_layer = torch.nn.Linear(4, 8)

    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(_FakeBridgeModel()))
    model = module.get_model_provider_func(_bridge_args(), role="critic")(post_process=True)

    init_stats = ppo_utils.snapshot_critic_value_head_state([model])
    assert init_stats, "snapshot must be non-empty for a post-process critic chunk"

    # Unchanged params → False (not moved).
    assert ppo_utils.has_critic_value_head_moved([model], init_stats) is False

    # Perturb weight → True (moved).
    with torch.no_grad():
        model.output_layer.weight.add_(0.1)
    assert ppo_utils.has_critic_value_head_moved([model], init_stats) is True

    # Empty stats (e.g., non-post-process rank) → None so caller can skip.
    assert ppo_utils.has_critic_value_head_moved([model], {}) is None


def test_wrapper_derives_vp_stage_from_parallel_state(monkeypatch):
    module, _ = _load_model_provider(monkeypatch)
    calls = []

    def original_provider(pre_process=True, post_process=True, vp_stage=None):
        calls.append(
            {
                "pre_process": pre_process,
                "post_process": post_process,
                "vp_stage": vp_stage,
            }
        )
        return SimpleNamespace(named_parameters=lambda: [])

    wrapped_provider = module.wrap_model_provider_with_freeze(
        original_provider,
        SimpleNamespace(only_train_params_name_list=None, freeze_params_name_list=None),
    )
    wrapped_provider(pre_process=True, post_process=False)

    assert calls == [{"pre_process": True, "post_process": False, "vp_stage": 1}]


def test_freeze_wrapper_forwards_post_process_to_classification_head(monkeypatch):
    module, _ = _load_model_provider(monkeypatch)
    post_process_values = []
    monkeypatch.setattr(
        module,
        "ensure_sequence_classification_head_trainable",
        lambda model, args, role, post_process: post_process_values.append(post_process),
    )

    wrapped_provider = module.wrap_model_provider_with_freeze(
        lambda **kwargs: SimpleNamespace(named_parameters=lambda: []),
        SimpleNamespace(only_train_params_name_list=None, freeze_params_name_list=None),
    )
    wrapped_provider(post_process=False)

    assert post_process_values == [False]


def test_wrapper_passes_vp_stage_through_bridge_provider(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    bridge_provider = module.get_model_provider_func(_bridge_args(), role="actor")
    wrapped_provider = module.wrap_model_provider_with_freeze(
        bridge_provider,
        SimpleNamespace(only_train_params_name_list=None, freeze_params_name_list=None),
    )
    wrapped_provider(pre_process=True, post_process=False)

    assert provider.calls == [{"pre_process": True, "post_process": False, "vp_stage": 1}]


@pytest.mark.parametrize("cp_comm_type", ["p2p", "a2a", ["p2p"], ["a2a"], ["p2p", "a2a"] * 30])
def test_bridge_provider_preserves_requested_cp_communication_after_finalize(monkeypatch, cp_comm_type):
    provider = _FakeProvider()
    provider.cp_comm_type = "p2p"
    provider.num_layers = 60
    original_finalize = provider.finalize

    def validate_and_finalize():
        if isinstance(provider.cp_comm_type, list):
            assert len(provider.cp_comm_type) == provider.num_layers
        original_finalize()

    provider.finalize = validate_and_finalize
    module, provider = _load_model_provider(monkeypatch, provider)
    build_model = module.get_model_provider_func(
        _bridge_args(cp_comm_type=cp_comm_type, context_parallel_size=4, num_layers=60), role="actor"
    )
    build_model(pre_process=True, post_process=True)
    assert provider.finalized
    expected = cp_comm_type[0] if isinstance(cp_comm_type, list) and len(cp_comm_type) == 1 else cp_comm_type
    assert provider.cp_comm_type == expected


def test_bridge_provider_keeps_cp_communication_default_when_argument_absent(monkeypatch):
    provider = _FakeProvider()
    provider.cp_comm_type = "p2p"
    module, provider = _load_model_provider(monkeypatch, provider)
    module.get_model_provider_func(_bridge_args(), role="actor")
    assert provider.finalized
    assert provider.cp_comm_type == "p2p"


@pytest.mark.parametrize("wrapper", ["freeze", "lora", "freeze_lora"])
@pytest.mark.parametrize("signature", ["modern", "kwargs", "legacy"])
def test_provider_wrappers_forward_supported_mcore_arguments(monkeypatch, wrapper, signature):
    module, _ = _load_model_provider(monkeypatch)
    model = torch.nn.Linear(2, 2)
    calls = []
    config = object()
    pg_collection = object()

    def modern_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
        calls.append((pre_process, post_process, vp_stage, config, pg_collection))
        return model

    def kwargs_provider(pre_process=True, post_process=True, **kwargs):
        calls.append((pre_process, post_process, kwargs["vp_stage"], kwargs["config"], kwargs["pg_collection"]))
        return model

    def legacy_provider(pre_process=True, post_process=True):
        calls.append((pre_process, post_process))
        return model

    monkeypatch.setattr(module, "is_lora_enabled", lambda args: True)
    monkeypatch.setattr(module, "build_lora_peft", lambda args: lambda model, training: model)
    args = SimpleNamespace(only_train_params_name_list=None, freeze_params_name_list=None)
    provider = {"modern": modern_provider, "kwargs": kwargs_provider, "legacy": legacy_provider}[signature]
    if "lora" in wrapper:
        provider = module.wrap_model_provider_with_lora(provider, args)
    if "freeze" in wrapper:
        provider = module.wrap_model_provider_with_freeze(provider, args)

    assert provider(pre_process=False, vp_stage=3, config=config, pg_collection=pg_collection) is model
    expected = (False, True) if signature == "legacy" else (False, True, 3, config, pg_collection)
    assert calls == [expected]


@pytest.mark.parametrize("accepts_mcore_arguments", [True, False])
def test_custom_provider_forwards_supported_mcore_arguments(monkeypatch, accepts_mcore_arguments):
    module, _ = _load_model_provider(monkeypatch)
    model = torch.nn.Module()
    calls = []
    config = object()
    pg_collection = object()

    def modern_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
        calls.append((pre_process, post_process, vp_stage, config, pg_collection))
        return model

    def legacy_provider(pre_process=True, post_process=True):
        calls.append((pre_process, post_process))
        return model

    original_provider = modern_provider if accepts_mcore_arguments else legacy_provider
    monkeypatch.setattr(module, "load_function", lambda path: original_provider)
    provider = module.get_model_provider_func(_bridge_args(custom_model_provider_path="custom.provider"))

    assert provider(post_process=False, vp_stage=3, config=config, pg_collection=pg_collection) is model
    expected = (True, False, 3, config, pg_collection) if accepts_mcore_arguments else (True, False)
    assert calls == [expected]


def test_bridge_provider_receives_mcore_process_groups_through_wrappers(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)
    monkeypatch.setattr(module, "is_lora_enabled", lambda args: True)
    monkeypatch.setattr(module, "build_lora_peft", lambda args: lambda model, training: model)
    args = _bridge_args(only_train_params_name_list=None, freeze_params_name_list=None)
    wrapped = module.wrap_model_provider_with_freeze(module.get_model_provider_func(args), args)
    config = object()
    pg_collection = object()

    wrapped(post_process=False, config=config, pg_collection=pg_collection)

    assert provider._pg_collection is pg_collection
    assert provider.calls == [{"pre_process": True, "post_process": False, "vp_stage": 1}]


@pytest.mark.parametrize("explicit_config", [True, False])
def test_raw_provider_uses_supplied_config_and_process_groups(monkeypatch, explicit_config):
    module, _ = _load_model_provider(monkeypatch)
    config = object()
    pg_collection = object()
    constructed = []
    config_calls = []

    def make_config(args):
        config_calls.append(args)
        return config

    def make_model(**kwargs):
        constructed.append(kwargs)
        return torch.nn.Module()

    monkeypatch.setattr(module, "core_transformer_config_from_args", make_config)
    monkeypatch.setattr(module, "GPTModel", make_model)
    args = _bridge_args(
        megatron_to_hf_mode="raw",
        transformer_impl="transformer_engine",
        spec=None,
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        multi_latent_attention=False,
        moe_use_legacy_grouped_gemm=False,
        fp8_param_gather=False,
        padded_vocab_size=128,
        max_position_embeddings=256,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        use_rope_scaling=False,
        mtp_num_layers=None,
    )
    provider = module.get_model_provider_func(args)

    provider(vp_stage=2, config=config if explicit_config else None, pg_collection=pg_collection)

    assert config_calls == ([] if explicit_config else [args])
    assert constructed[0]["config"] is config
    assert constructed[0]["pg_collection"] is pg_collection
    assert constructed[0]["vp_stage"] == 2


@pytest.mark.parametrize(
    "name,value",
    [
        ("moe_hybridep_pad_variable_tokens", True),
        ("gdn_pre_gated_delta_rule_fusion", True),
        ("gdn_conv_pad_alignment", 8192),
    ],
)
@pytest.mark.parametrize("distributed", [False, True], ids=["ddp", "mfsdp"])
def test_bridge_forwards_blackwell_runtime_configuration(monkeypatch, name, value, distributed):
    provider = _FakeProvider()
    setattr(provider, name, None)
    module, provider = _load_model_provider(monkeypatch, provider)
    if distributed:
        module.get_bridge_distributed_model(_bridge_args(**{name: value}))
    else:
        module.get_model_provider_func(_bridge_args(**{name: value}))(pre_process=True, post_process=True)
    assert getattr(provider, name) == value


@pytest.mark.parametrize("max_tokens,alignment", [(65536, 8), (4096, 16)])
@pytest.mark.parametrize("distributed", [False, True], ids=["ddp", "mfsdp"])
def test_bridge_provider_receives_packed_sequence_limits(monkeypatch, max_tokens, alignment, distributed):
    provider = _FakeProvider()
    provider.max_seqlen_per_dp_cp_rank = 2048
    provider.pad_packed_seq_alignment = 1
    module, provider = _load_model_provider(monkeypatch, provider=provider)
    build = module.get_bridge_distributed_model if distributed else module.get_model_provider_func
    build(_bridge_args(max_seqlen_per_dp_cp_rank=max_tokens, pad_packed_seq_alignment=alignment))
    assert provider.max_seqlen_per_dp_cp_rank == max_tokens
    assert provider.pad_packed_seq_alignment == alignment


@pytest.mark.parametrize("distributed", [False, True], ids=["ddp", "mfsdp"])
def test_bridge_provider_preserves_native_offload_policy(monkeypatch, distributed):
    policy = {
        "fine_grained_activation_offloading": True,
        "offload_modules": ["moe_act", "core_attn"],
        "min_offloaded_tensor_size": 4096,
        "activation_offload_fraction": 0.5,
        "delta_offload_bytes_across_pp_ranks": 1024,
        "delay_offload_until_cuda_graph": True,
        "fine_grained_offloading_max_inflight_offloads": 2,
    }
    provider = _FakeProvider()
    for name in policy:
        setattr(provider, name, None)
    module, provider = _load_model_provider(monkeypatch, provider)
    build = module.get_bridge_distributed_model if distributed else module.get_model_provider_func
    build(_bridge_args(**policy), role="actor")
    assert provider.finalized
    assert {name: getattr(provider, name) for name in policy} == policy


def test_bridge_distributed_model_uses_megatron_fsdp(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    model = module.get_bridge_distributed_model(_bridge_args(use_megatron_fsdp=True), role="actor")

    assert len(model) == 1
    assert len(provider.pre_wrap_hooks) == 1
    assert provider.distributed_calls == [
        {
            "wrap_with_ddp": True,
            "ddp_config": {"fake_ddp_config_for": "bridge"},
            "fp16": False,
            "bf16": True,
            "use_megatron_fsdp": True,
            "overlap_param_gather_with_optimizer_step": False,
            "data_parallel_random_init": False,
            "use_cpu_initialization": False,
            "init_model_with_meta_device": False,
        }
    ]


def test_bridge_distributed_model_disables_pipeline_output_dealloc_under_pp(monkeypatch):
    """#20: megatron_fsdp wraps every FSDP-unit module's forward output in a
    `t.view_as(t)` (see megatron_fsdp.py::create_custom_backward_hook), so the
    activation crossing a PP boundary is always a view. The pipelined schedules'
    `deallocate_output_tensor()` asserts the opposite ("counter-productive to free
    a view of another tensor") and crashes at PP>1. GPTModelProvider defaults
    `deallocate_pipeline_outputs=True`, so get_bridge_distributed_model must turn
    it off whenever PP>1 is in play (PP==1 never calls deallocate_output_tensor at
    all, so it's left alone there)."""
    module, provider = _load_model_provider(monkeypatch)

    module.get_bridge_distributed_model(
        _bridge_args(use_megatron_fsdp=True, pipeline_model_parallel_size=4), role="actor"
    )

    assert provider.deallocate_pipeline_outputs is False


def test_bridge_distributed_model_leaves_pipeline_output_dealloc_alone_at_pp1(monkeypatch):
    module, provider = _load_model_provider(monkeypatch)

    module.get_bridge_distributed_model(
        _bridge_args(use_megatron_fsdp=True, pipeline_model_parallel_size=1), role="actor"
    )

    assert provider.deallocate_pipeline_outputs is True


@pytest.mark.parametrize("distributed", [False, True], ids=["ddp", "mfsdp"])
@pytest.mark.parametrize("precision", ["fp16", "bf16", "provider_default"])
def test_bridge_provider_configures_overrides_before_finalize(monkeypatch, distributed, precision):
    module, provider = _load_model_provider(monkeypatch)
    provider.masked_softmax_fusion = False
    provider.recompute_modules = ["default"]
    provider.moe_router_bias_update_rate = 0.001
    provider.dynamic_context_parallel = True
    provider.params_dtype = torch.float32
    provider.unrelated_setting = "provider default"
    runtime_overrides = {
        "fp8": "hybrid",
        "fp8_recipe": "blockwise",
        "fp8_quantizer_factory": "custom.quantizer",
        "pipeline_model_parallel_layout": "Et|t|t|tL",
        "cp_partition_mode": "zigzag",
        "sequence_packing_scheduler": "dynamic",
        "mtp_use_repeated_layer": True,
        "linear_cp_mode": "all_gather",
        "activation_func_clamp_shared_expert": 7.0,
    }
    for key in runtime_overrides:
        setattr(provider, key, None)
    snapshots = []
    dumps = []
    monkeypatch.setattr(provider, "finalize", lambda: snapshots.append(vars(provider).copy()))
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module, "_dump_provider_config", lambda p, path: dumps.append((p, path)))
    args = _bridge_args(
        masked_softmax_fusion=True,
        recompute_modules=["core_attn", "moe"],
        moe_router_bias_update_rate=0.02,
        dynamic_context_parallel=True,
        decoder_first_pipeline_num_layers=3,
        decoder_last_pipeline_num_layers=1,
        gradient_accumulation_fusion=False,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        save="fake-save",
        unrelated_setting="must not override",
        **runtime_overrides,
    )

    if distributed:
        module.get_bridge_distributed_model(args)
    else:
        module.get_model_provider_func(args)

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert {key: snapshot[key] for key in runtime_overrides} == runtime_overrides
    assert snapshot["masked_softmax_fusion"] is True
    assert snapshot["recompute_modules"] == ["core_attn", "moe"]
    assert snapshot["moe_router_bias_update_rate"] == 0.02
    assert snapshot["dynamic_context_parallel"] is False
    assert args.dynamic_context_parallel is True
    assert snapshot["num_layers_in_first_pipeline_stage"] == 3
    assert snapshot["num_layers_in_last_pipeline_stage"] == 1
    assert snapshot["gradient_accumulation_fusion"] is False
    assert snapshot["virtual_pipeline_model_parallel_size"] == 2
    assert snapshot["unrelated_setting"] == "provider default"
    assert (
        snapshot["params_dtype"]
        == {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "provider_default": torch.float32,
        }[precision]
    )
    assert snapshot["fp16"] is (precision == "fp16")
    assert snapshot["bf16"] is (precision == "bf16")
    assert dumps == [(provider, "fake-save")]


class _TaskHeadModel(torch.nn.Module):
    def __init__(self, post_process=True):
        super().__init__()
        self.post_process = post_process
        self.config = SimpleNamespace(hidden_size=4, sequence_parallel=False)
        self.decoder = torch.nn.Linear(4, 4)
        self.decoder.config = SimpleNamespace()
        self.output_layer = torch.nn.Linear(4, 8)


@pytest.mark.parametrize("distributed", [False, True], ids=["ddp", "mfsdp"])
@pytest.mark.parametrize("role", ["actor", "critic", "reference"])
@pytest.mark.parametrize("post_process", [False, True])
def test_bridge_task_heads_and_detach_paths_match_across_wrappers(monkeypatch, distributed, role, post_process):
    model = _TaskHeadModel(post_process=post_process)
    lm_head = model.output_layer
    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(model))
    args = _bridge_args(
        task_type="seq_cls" if role == "actor" else "causal_lm",
        num_labels=3,
        mtp_detach_paths=["embedding", "lm-head"],
        freeze_params_name_list=[r".*"] if role == "actor" else None,
    )

    if distributed:
        assert module.get_bridge_distributed_model(args, role=role) == [model]
    else:
        provide = module.wrap_model_provider_with_freeze(module.get_model_provider_func(args, role=role), args)
        # Positional arguments must retain the same pre/post-process interpretation.
        assert provide(True, post_process, 1) is model

    for config in (model.config, model.decoder.config):
        assert config.mtp_detach_embedding is True
        assert config.mtp_detach_backbone is False
        assert config.mtp_detach_lm_head is True
    if post_process and role in ("actor", "critic"):
        assert isinstance(model.output_layer, ppo_utils.LinearForLastLayer)
        assert model.output_layer.out_features == (3 if role == "actor" else 1)
        stash = (
            ppo_utils._RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR if role == "actor" else ppo_utils._RELAX_HF_OUTPUT_LAYER_ATTR
        )
        assert getattr(model, stash) is lm_head
        assert all(param is not lm_head.weight for param in model.parameters())
        assert all(param.requires_grad for param in model.output_layer.parameters())
    else:
        assert model.output_layer is lm_head
    if role == "actor":
        assert all(not param.requires_grad for param in model.decoder.parameters())


def test_bridge_fsdp_configures_every_chunk_before_distributed_wrap(monkeypatch):
    chunks = [_TaskHeadModel(post_process=False), _TaskHeadModel(post_process=True)]
    module, provider = _load_model_provider(monkeypatch, provider=_FakeProvider(chunks))
    probes = []
    norm_patches = []
    monkeypatch.setattr(module, "_install_cp_probe", lambda models: probes.append(models))
    monkeypatch.setattr(module, "patch_gated_residual_norms_for_fsdp", lambda model: norm_patches.append(model))
    args = _bridge_args(task_type="seq_cls", num_labels=3, mtp_detach_paths=[], freeze_params_name_list=[r".*"])

    model = module.get_bridge_distributed_model(args)

    assert model is chunks
    assert probes == [chunks]
    assert norm_patches == chunks
    assert len(provider.pre_wrap_hooks) == 1
    assert chunks[0].output_layer.out_features == 8
    assert chunks[1].output_layer.out_features == 3
    assert not chunks[0].output_layer.weight.requires_grad
    assert chunks[1].output_layer.weight.requires_grad
    for chunk in chunks:
        assert chunk.config.mtp_detach_embedding is False
        assert chunk.config.mtp_detach_backbone is False
        assert chunk.config.mtp_detach_lm_head is False


def test_bridge_ddp_preserves_task_head_lora_freeze_order(monkeypatch):
    model = _TaskHeadModel()
    module, _ = _load_model_provider(monkeypatch, provider=_FakeProvider(model))
    calls = []

    def apply_peft(model, training):
        assert training is True
        assert model.output_layer.out_features == 3
        assert model.config.mtp_detach_backbone is False
        calls.append(model)
        model.requires_grad_(False)
        model.register_parameter("adapter", torch.nn.Parameter(torch.ones(4)))
        return model

    monkeypatch.setattr(module, "build_lora_peft", lambda args: apply_peft)
    args = _bridge_args(
        task_type="seq_cls",
        num_labels=3,
        mtp_detach_paths=[],
        lora_rank=4,
        only_train_params_name_list=[r"^adapter$"],
    )
    provide = module.wrap_model_provider_with_freeze(module.get_model_provider_func(args), args)

    assert provide(post_process=True) is model
    assert calls == [model]
    assert {name for name, param in model.named_parameters() if param.requires_grad} == {
        "adapter",
        "output_layer.weight",
    }


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"megatron_to_hf_mode": "raw"}, "bridge"),
        ({"custom_model_provider_path": "custom.provider"}, "custom-model-provider-path"),
        ({"lora_rank": 4}, "LoRA"),
    ],
)
def test_bridge_fsdp_rejects_ignored_options_before_provider_creation(monkeypatch, overrides, match):
    module, provider = _load_model_provider(monkeypatch)

    with pytest.raises(ValueError, match=match):
        module.get_bridge_distributed_model(_bridge_args(**overrides))

    assert not provider.finalized
    assert provider.pre_wrap_hooks == []
    assert provider.distributed_calls == []


def test_custom_provider_still_takes_precedence_in_ordinary_path(monkeypatch):
    model = _TaskHeadModel()
    module, provider = _load_model_provider(monkeypatch)
    calls = []

    def custom(pre_process, post_process, vp_stage):
        calls.append((pre_process, post_process, vp_stage))
        return model

    monkeypatch.setattr(module, "load_function", lambda path: custom)
    args = _bridge_args(custom_model_provider_path="custom.provider", mtp_detach_paths=[])

    assert module.get_model_provider_func(args)(pre_process=False, post_process=False, vp_stage=2) is model
    assert calls == [(False, False, 2)]
    assert model.config.mtp_detach_backbone is False
    assert not provider.finalized


@pytest.mark.parametrize("distributed", [False, True], ids=["ddp", "mfsdp"])
def test_bridge_preserves_gdn_precision_and_reward_head_before_wrapping(monkeypatch, distributed):
    class GatedDeltaNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
            self.A_log.tensor_model_parallel = True

    model = _TaskHeadModel()
    model.self_attention = GatedDeltaNet()
    model.uses_unsplit_forward = True
    original_a_log = model.self_attention.A_log
    lm_head = model.output_layer
    module, provider = _load_model_provider(monkeypatch, provider=_FakeProvider(model))
    monkeypatch.setitem(sys.modules, "megatron.core.ssm.gated_delta_net", SimpleNamespace(GatedDeltaNet=GatedDeltaNet))
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.module",
        SimpleNamespace(mark_keep_in_fp32=lambda param: setattr(param, "keep_in_fp32", True)),
    )
    args = _bridge_args(loss_type="rm", mtp_detach_paths=[])
    if distributed:
        assert module.get_bridge_distributed_model(args) == [model]
    else:
        assert module.get_model_provider_func(args)(post_process=True) is model

    assert model.self_attention.A_log is original_a_log
    assert original_a_log.dtype == torch.float32
    assert original_a_log.tensor_model_parallel and original_a_log.keep_in_fp32
    assert isinstance(model.output_layer, ppo_utils.LinearForLastLayer)
    assert model.output_layer.out_features == 1
    assert getattr(model, ppo_utils._RELAX_HF_OUTPUT_LAYER_ATTR) is lm_head
    assert all(param is not lm_head.weight for param in model.parameters())
    assert args.uses_unsplit_forward is True
    assert args._relax_qwen35_optimized is False
