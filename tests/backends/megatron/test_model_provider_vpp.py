import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from relax.utils.training import ppo_utils


class _FakeProvider:
    def __init__(self, model=None):
        self.calls = []
        self.finalized = False
        self.model = model or torch.nn.Module()
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
    spec_utils = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_config = types.ModuleType("megatron.core.transformer.transformer_config")
    training = types.ModuleType("megatron.training")
    arguments = types.ModuleType("megatron.training.arguments")
    tokenizer_package = types.ModuleType("megatron.training.tokenizer")
    tokenizer = types.ModuleType("megatron.training.tokenizer.tokenizer")
    bridge = types.ModuleType("megatron.bridge")
    misc = types.ModuleType("relax.utils.misc")

    class _FakeGPTModel:
        pass

    class _FakeTransformerConfig:
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
    arguments.core_transformer_config_from_args = lambda args: _FakeTransformerConfig()
    arguments.parse_args = lambda *args, **kwargs: None
    arguments.validate_args = lambda *args, **kwargs: None
    tokenizer._vocab_size_with_padding = lambda *args, **kwargs: None
    bridge.AutoBridge = _FakeAutoBridge
    misc.load_function = lambda path: None

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
        "megatron.core.transformer.spec_utils": spec_utils,
        "megatron.core.transformer.transformer_config": transformer_config,
        "megatron.training": training,
        "megatron.training.arguments": arguments,
        "megatron.training.tokenizer": tokenizer_package,
        "megatron.training.tokenizer.tokenizer": tokenizer,
        "megatron.bridge": bridge,
        "relax.utils.misc": misc,
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
