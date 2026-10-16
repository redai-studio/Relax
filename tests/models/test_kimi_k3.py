# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU behavior tests that do not require an installed K3 Bridge
implementation."""

import importlib.util
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class _Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(32, 3)
        self.embedding_dropout = torch.nn.Dropout(0)
        with torch.no_grad():
            self.word_embeddings.weight.copy_(torch.arange(96).reshape(32, 3) / 100)

    def forward(self, input_ids, position_ids=None):
        return self.embedding_dropout(self.word_embeddings(input_ids).transpose(0, 1))


class _LanguageModel(torch.nn.Module):
    def __init__(self, pre_process=True, post_process=True):
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.pg_collection = SimpleNamespace(tp=object())
        self.share_embeddings_and_output_weights = False
        self.embedding = _Embedding() if pre_process else None
        self.calls = []
        self.input_tensor = None

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["decoder_input"] if kwargs["decoder_input"] is not None else self.input_tensor

    def set_input_tensor(self, value):
        self.input_tensor = value

    def shared_embedding_or_output_weight(self):
        return self.embedding.word_embeddings.weight


class _Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3, bias=False)])
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(3) * 2)
        self.calls = []

    @property
    def proj(self):
        return self.encoder.blocks[0]

    def forward(self, pixels, grid):
        self.calls.append((pixels, grid))
        return list(self.proj(pixels.flatten(1)).split(1))


class _Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(3) * 3)
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        return tuple(self.proj(feature) for feature in features)


@pytest.fixture
def k3_modules(monkeypatch):
    def install(name, **attributes):
        module = ModuleType(name)
        module.__path__ = []
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class FakeMegatronModule(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    @dataclass
    class FakeK3Provider:
        tensor_model_parallel_size: int = 1
        context_parallel_size: int = 1
        cp_partition_mode: str = "zigzag"
        linear_cp_mode: str = "chunkwise"
        kimi_linear_num_heads: int = 96
        virtual_pipeline_model_parallel_size: int | None = None
        mtp_num_layers: int | None = None
        variable_seq_lengths: bool = True
        sequence_parallel: bool = False
        params_dtype: object = torch.float32
        kimi_kda_layers: tuple = (1,)

        def provide(self, pre_process=None, post_process=None, vp_stage=None):
            model = _LanguageModel(
                pre_process=True if pre_process is None else pre_process,
                post_process=True if post_process is None else post_process,
            )
            model.pg_collection.cp = SimpleNamespace(size=lambda: self.context_parallel_size, rank=lambda: 0)
            return model

    class Mapping:
        def __init__(self, megatron_param, hf_param):
            self.megatron_param = megatron_param
            self.hf_param = hf_param

    class Registry:
        def __init__(self, *mappings):
            self.mappings = list(mappings)

    class FakeK3Bridge:
        _HF_PASSTHROUGH_PREFIXES = ("vision_tower.", "mm_projector.")

        def provider_bridge(self, hf_pretrained):
            return self.PROVIDER_CLASS()

        def mapping_registry(self):
            self.language_mappings = [
                Mapping("embedding.word_embeddings.weight", "language_model.model.embed_tokens.weight"),
                Mapping(
                    "decoder.layers.*.mlp.linear_fc1.weight",
                    {"gate": "language_model.model.layers.*.mlp.gate_proj.weight", "up": "language_model.up"},
                ),
                Mapping("decoder.layers.92.output_attn_res_norm.weight", "language_model.model.output_attn_res_norm"),
            ]
            return Registry(*self.language_mappings)

    registrations = []

    def register_bridge(**registration):
        def decorator(cls):
            cls.PROVIDER_CLASS = registration["provider"]
            registrations.append(registration)
            return cls

        return decorator

    for package in (
        "megatron",
        "megatron.core",
        "megatron.core.models",
        "megatron.core.transformer",
        "megatron.bridge",
        "megatron.bridge.models",
        "megatron.bridge.models.conversion",
        "megatron.bridge.models.hf_pretrained",
        "megatron.bridge.models.kimi",
    ):
        install(package)
    install("megatron.core.transformer.module", MegatronModule=FakeMegatronModule)
    install("megatron.core.packed_seq_params", PackedSeqParams=SimpleNamespace)
    install("megatron.core.tensor_parallel", scatter_to_sequence_parallel_region=lambda value, group: value)
    install("megatron.core.models.gpt", GPTModel=_LanguageModel)
    install("megatron.bridge.models.kimi.kimi_k3_provider", KimiK3ModelProvider=FakeK3Provider)
    install("megatron.bridge.models.kimi.kimi_k3_bridge", KimiK3Bridge=FakeK3Bridge)
    install("megatron.bridge.models.conversion.mapping_registry", MegatronMappingRegistry=Registry)
    install(
        "megatron.bridge.models.conversion.model_bridge",
        MegatronModelBridge=SimpleNamespace(register_bridge=register_bridge),
    )
    install("megatron.bridge.models.conversion.param_mapping", ReplicatedMapping=Mapping)
    install("megatron.bridge.models.hf_pretrained.causal_lm", PreTrainedCausalLM=SimpleNamespace)

    target = Path(__file__).parents[2] / "relax/models/kimi_k3"
    package = install("_kimi_k3_unit_target")
    package.__path__ = [str(target)]
    loaded = {}
    for name in ("model", "provider", "bridge"):
        module_name = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(module_name, target / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    loaded["registrations"] = registrations
    return SimpleNamespace(**loaded)


def _model(monkeypatch, modules, pre_process=True, **overrides):
    vision = _Vision()
    modules.model._enable_vision_layer_checkpointing(vision)
    projector = _Projector()
    monkeypatch.setattr(modules.model, "_build_vision_modules", lambda config: (vision, projector))
    values = dict(
        image_token_id=31,
        hf_model_path="fake-k3",
        vision_config=SimpleNamespace(merge_kernel_size=(2, 2), patch_size=1, vt_hidden_size=3),
        trust_remote_code=True,
    )
    values.update(overrides)
    provider = modules.provider.KimiK3VLModelProvider(**values)
    return provider.provide(pre_process=pre_process), vision, projector


class _CheckpointVisionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(0.25)
        self.calls = 0

    def forward(self, hidden_states, *, scale):
        self.calls += 1
        return hidden_states + self.dropout(torch.sin(self.proj(hidden_states))) * scale


class _CheckpointVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.blocks = torch.nn.ModuleList([_CheckpointVisionBlock() for _ in range(3)])

    def forward(self, hidden_states):
        for index, block in enumerate(self.encoder.blocks):
            hidden_states = block(hidden_states, scale=(index + 1) / 3)
        return hidden_states


@pytest.mark.parametrize("input_requires_grad", [False, True])
def test_kimi_k3_vision_layer_checkpoint_preserves_outputs_gradients_and_state_dict(k3_modules, input_requires_grad):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        reference = _CheckpointVision()
        checkpointed = deepcopy(reference)
        state_keys = tuple(checkpointed.state_dict())
        parameters = dict(checkpointed.named_parameters())
        k3_modules.model._enable_vision_layer_checkpointing(checkpointed)
        assert tuple(checkpointed.state_dict()) == state_keys
        assert all(dict(checkpointed.named_parameters())[name] is param for name, param in parameters.items())

        reference_input = torch.randn(4, 3, requires_grad=input_requires_grad)
        checkpointed_input = reference_input.detach().clone().requires_grad_(input_requires_grad)
        torch.manual_seed(37)
        reference_output = reference(reference_input)
        reference_output.square().sum().backward()
        torch.manual_seed(37)
        checkpointed_output = checkpointed(checkpointed_input)
        assert [block.calls for block in checkpointed.encoder.blocks] == [1, 1, 1]
        checkpointed_output.square().sum().backward()

    torch.testing.assert_close(checkpointed_output, reference_output)
    assert [block.calls for block in checkpointed.encoder.blocks] == [2, 2, 2]
    for name, parameter in checkpointed.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad)
    if input_requires_grad:
        torch.testing.assert_close(checkpointed_input.grad, reference_input.grad)


@pytest.mark.parametrize("evaluation", [False, True])
def test_kimi_k3_vision_layer_checkpoint_skips_eval_and_no_grad(monkeypatch, k3_modules, evaluation):
    tower = _CheckpointVision()
    k3_modules.model._enable_vision_layer_checkpointing(tower)

    def unexpected_checkpoint(*args, **kwargs):
        pytest.fail("vision checkpoint must be disabled in eval mode or without gradients")

    monkeypatch.setattr(k3_modules.model, "checkpoint", unexpected_checkpoint)
    inputs = torch.ones(4, 3, requires_grad=True)
    if evaluation:
        tower.eval()
        tower(inputs).sum().backward()
        assert inputs.grad is not None
    else:
        with torch.no_grad():
            assert not tower(inputs).requires_grad
    assert [block.calls for block in tower.encoder.blocks] == [1, 1, 1]


@pytest.mark.parametrize("packed", [True, False])
def test_kimi_k3_merges_preexpanded_visual_tokens_in_batch_order_and_backpropagates(monkeypatch, k3_modules, packed):
    model, vision, projector = _model(monkeypatch, k3_modules)
    token_ids = torch.tensor([[2, 31, 3], [31, 4, 5]])
    if packed:
        token_ids = token_ids.reshape(1, -1)
    pixels = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(2, 3, 1, 1)
    packed_params = SimpleNamespace(qkv_format="thd") if packed else None
    expected = model.language_model.embedding(input_ids=token_ids).transpose(0, 1).detach().clone()
    expected[token_ids == 31] = pixels.flatten(1) * 6

    output = model(
        token_ids,
        pixel_values=pixels,
        grid_thws=torch.tensor([[1, 1, 1], [1, 1, 1]]),
        packed_seq_params=packed_params,
    )

    torch.testing.assert_close(output.transpose(0, 1), expected)
    assert model.language_model.calls[0]["packed_seq_params"] is packed_params
    output.sum().backward()
    assert torch.count_nonzero(vision.proj.weight.grad) > 0
    assert torch.count_nonzero(projector.proj.weight.grad) > 0
    assert torch.count_nonzero(model.language_model.embedding.word_embeddings.weight.grad[31]) == 0


@pytest.mark.parametrize("feature_count", [1, 3])
def test_kimi_k3_rejects_missing_or_surplus_visual_features(monkeypatch, k3_modules, feature_count):
    model, _, _ = _model(monkeypatch, k3_modules)

    with pytest.raises(RuntimeError, match="pre-expanded media placeholder count"):
        model(
            torch.tensor([[31, 1, 31]]),
            pixel_values=torch.ones(feature_count, 3, 1, 1),
            grid_thws=torch.ones(feature_count, 3, dtype=torch.long),
        )


def test_kimi_k3_rejects_visual_inputs_without_grid_and_placeholders_without_pixels(monkeypatch, k3_modules):
    model, _, _ = _model(monkeypatch, k3_modules)
    with pytest.raises(ValueError, match="both pixel_values and grid_thws"):
        model(torch.tensor([[31]]), pixel_values=torch.ones(1, 3, 1, 1))
    with pytest.raises(RuntimeError, match="placeholders require visual inputs"):
        model(torch.tensor([[31]]))


def test_kimi_k3_text_batch_runs_visual_hooks_and_produces_zero_visual_gradients(monkeypatch, k3_modules):
    model, vision, projector = _model(monkeypatch, k3_modules)
    token_ids = torch.tensor([[2, 3, 4]])
    expected = model.language_model.embedding(input_ids=token_ids).detach()

    output = model(token_ids)
    output.sum().backward()

    torch.testing.assert_close(output, expected)
    assert len(vision.calls) == 1  # only encoder blocks are recomputed, not the entire tower
    assert projector.calls == 2
    assert vision.calls[0][0].shape == (4, 3, 1, 1)
    assert vision.calls[0][1].device.type == "cpu"
    for module in (vision, projector):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) == 0


def test_kimi_k3_frozen_visual_modules_skip_dummy_forward(monkeypatch, k3_modules):
    model, vision, projector = _model(monkeypatch, k3_modules, freeze_vision_model=True, freeze_vision_projection=True)
    model(torch.tensor([[2, 3]])).sum().backward()

    assert not vision.calls
    assert projector.calls == 0
    assert all(not p.requires_grad for module in (vision, projector) for p in module.parameters())


def test_kimi_k3_vision_tp_splits_frozen_tower_and_keeps_projector_gradients(monkeypatch, k3_modules):
    model, vision, projector = _model(monkeypatch, k3_modules, freeze_vision_model=True, vision_dp_when_tp=True)
    model.tp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(k3_modules.model.dist, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(k3_modules.model.dist, "get_rank", lambda group: group.rank())
    model.config.vision_config.merge_kernel_size = (1, 1)
    pixels = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(2, 3, 1, 1)

    def all_reduce(buffer, group):
        assert group is model.tp_group
        buffer[1] = pixels[1].flatten() * 2

    monkeypatch.setattr(k3_modules.model.dist, "all_reduce", all_reduce)
    output = model(torch.tensor([[31, 31]]), pixel_values=pixels, grid_thws=torch.ones(2, 3, dtype=torch.long))
    torch.testing.assert_close(output.transpose(0, 1).squeeze(0), pixels.flatten(1) * 6)
    assert len(vision.calls) == 1
    assert vision.calls[0][0].shape[0] == 1
    output.sum().backward()
    assert vision.proj.weight.grad is None
    assert torch.count_nonzero(projector.proj.weight.grad) > 0


def test_kimi_k3_pipeline_stage_delegates_attnres_payload_without_constructing_vision(monkeypatch, k3_modules):
    model, vision, projector = _model(monkeypatch, k3_modules, pre_process=False)
    payload = [torch.randn(3, 1, 3), torch.randn(3, 1, 2, 3)]

    model.set_input_tensor(payload)

    assert model.language_model.input_tensor is payload
    assert model(torch.tensor([[2, 3, 4]])) is payload
    assert model.vision_tower is None
    assert model.mm_projector is None
    assert not vision.calls
    assert projector.calls == 0


def test_kimi_k3_sequence_parallel_scatter_happens_after_visual_merge_with_explicit_group(monkeypatch, k3_modules):
    model, _, _ = _model(monkeypatch, k3_modules, sequence_parallel=True)
    scatter_calls = []

    def scatter(value, group):
        scatter_calls.append((value.clone(), group))
        return value[: value.shape[0] // 2]

    monkeypatch.setattr(k3_modules.model, "scatter_to_sequence_parallel_region", scatter)
    token_ids = torch.tensor([[2, 31, 3, 4]])
    padding_mask = torch.tensor([[False, False, False, True]])
    loss_mask = torch.ones(1, 4)
    output = model(
        token_ids,
        pixel_values=torch.ones(1, 3, 1, 1),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        padding_mask=padding_mask,
        loss_mask=loss_mask,
    )

    assert len(scatter_calls) == 2
    assert all(group is model.tp_group for _, group in scatter_calls)
    torch.testing.assert_close(scatter_calls[0][0][1], torch.full((1, 3), 6.0))
    assert output.shape == (2, 1, 3)
    assert model.language_model.calls[0]["padding_mask"].shape == (1, 2)
    assert model.language_model.calls[0]["loss_mask"] is loss_mask


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"context_parallel_size": 2, "cp_partition_mode": "contiguous"}, "requires --cp-partition-mode zigzag"),
        ({"context_parallel_size": 2, "linear_cp_mode": "invalid"}, "requires --linear-cp-mode"),
        ({"context_parallel_size": 5, "linear_cp_mode": "headwise"}, "heads divisible by TP \\* CP"),
        ({"tensor_model_parallel_size": 2, "sequence_parallel": False}, "requires --sequence-parallel"),
        ({"tensor_model_parallel_size": 2, "vision_dp_when_tp": True}, "requires --freeze-vision-model"),
        ({"virtual_pipeline_model_parallel_size": 2}, "virtual pipeline"),
        ({"mtp_num_layers": 1}, "MTP training"),
    ],
)
def test_kimi_k3_provider_rejects_unsupported_parallelism_and_mtp(k3_modules, overrides, message):
    provider = k3_modules.provider.KimiK3VLModelProvider(**overrides)
    with pytest.raises(ValueError, match=message):
        provider.provide()


def test_kimi_k3_provider_accepts_tensor_parallel_with_sequence_parallel(monkeypatch, k3_modules):
    model, _, _ = _model(monkeypatch, k3_modules, tensor_model_parallel_size=2, sequence_parallel=True)

    assert model.config.tensor_model_parallel_size == 2
    assert model.config.sequence_parallel


@pytest.mark.parametrize("freeze_language_model", [False, True])
def test_kimi_k3_first_layer_unused_attnres_weights_are_frozen_and_remain_in_checkpoint(
    monkeypatch, k3_modules, freeze_language_model
):
    class AttnResLayer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number
            self.self_attention_res_norm = torch.nn.Linear(3, 3, bias=False)
            self.self_attention_res_proj = torch.nn.Linear(3, 1, bias=False)
            self.mlp_res_norm = torch.nn.Linear(3, 3, bias=False)
            self.mlp_res_proj = torch.nn.Linear(3, 1, bias=False)

    language_model = _LanguageModel(pre_process=False)
    language_model.layers = torch.nn.ModuleList([AttnResLayer(1), AttnResLayer(2), AttnResLayer(93)])
    keys_before = set(language_model.state_dict())
    provider = k3_modules.provider.KimiK3VLModelProvider(freeze_language_model=freeze_language_model)
    monkeypatch.setattr(provider, "provide_language_model", lambda **kwargs: language_model)

    model = provider.provide(pre_process=False)

    assert set(model.language_model.state_dict()) == keys_before
    for layer in language_model.layers:
        for name, parameter in layer.named_parameters():
            unused = layer.layer_number == 1 and name.startswith("self_attention_res_")
            assert parameter.requires_grad is (not unused and not freeze_language_model)


def test_kimi_k3_language_sum_markers_use_core_sequence_parallel_reduction(monkeypatch, k3_modules):
    language_model = _LanguageModel(pre_process=False)
    language_model.attn_res_norm = torch.nn.Linear(3, 3, bias=False)
    language_model.routed_expert_norm = torch.nn.Linear(3, 3, bias=False)
    language_model.o_norm = torch.nn.Linear(3, 3, bias=False)
    language_model.replicated_linear = torch.nn.Linear(3, 3, bias=False)
    language_model.tp_sharded_linear = torch.nn.Linear(3, 3, bias=False)
    sum_parameters = [
        language_model.attn_res_norm.weight,
        language_model.routed_expert_norm.weight,
        language_model.o_norm.weight,
    ]
    for parameter in sum_parameters:
        parameter.sum_gradients_across_tp_domain = True
    average_parameter = language_model.replicated_linear.weight
    average_parameter.average_gradients_across_tp_domain = True
    provider = k3_modules.provider.KimiK3VLModelProvider(tensor_model_parallel_size=2, sequence_parallel=True)
    monkeypatch.setattr(provider, "provide_language_model", lambda **kwargs: language_model)

    provider.provide(pre_process=False)

    assert all(parameter.sequence_parallel for parameter in sum_parameters)
    assert all(not getattr(parameter, "average_gradients_across_tp_domain", False) for parameter in sum_parameters)
    assert average_parameter.average_gradients_across_tp_domain
    assert not getattr(average_parameter, "sequence_parallel", False)
    assert not getattr(language_model.tp_sharded_linear.weight, "sequence_parallel", False)


def test_kimi_k3_preserves_kda_fp32_parameter_markers_for_core_mixed_precision(monkeypatch, k3_modules):
    language_model = _LanguageModel(pre_process=False)
    language_model.attention = torch.nn.Module()
    attention = language_model.attention
    attention.A_log = torch.nn.Parameter(torch.tensor([1.001]))
    attention.dt_bias = torch.nn.Parameter(torch.tensor([2.003]))
    attention.weight = torch.nn.Parameter(torch.tensor([3.007]))
    attention._keep_in_float32_parameter_names = ("A_log", "dt_bias")
    provider = k3_modules.provider.KimiK3VLModelProvider()
    monkeypatch.setattr(provider, "provide_language_model", lambda **kwargs: language_model)

    provider.provide(pre_process=False)

    assert attention.A_log.keep_in_fp32
    assert attention.dt_bias.keep_in_fp32
    assert not getattr(attention.weight, "keep_in_fp32", False)


def test_kimi_k3_bridge_registers_visual_provider_and_preserves_hf_metadata(k3_modules):
    bridge = k3_modules.bridge.KimiK3VLBridge()
    vision_config = object()
    hf = SimpleNamespace(
        config=SimpleNamespace(vision_config=vision_config, media_placeholder_token_id=123),
        model_name_or_path="test-k3-checkpoint",
        trust_remote_code=True,
    )

    provider = bridge.provider_bridge(hf)

    assert isinstance(provider, k3_modules.provider.KimiK3VLModelProvider)
    assert provider.vision_config is vision_config
    assert provider.hf_model_path == hf.model_name_or_path
    assert provider.image_token_id == 123
    assert provider.trust_remote_code
    assert not provider.scatter_embedding_sequence_parallel
    assert k3_modules.registrations == [
        dict(
            source="KimiK3ForConditionalGeneration",
            target=k3_modules.model.KimiK3VLModel,
            provider=k3_modules.provider.KimiK3VLModelProvider,
            model_type="kimi_k3",
        )
    ]


def test_kimi_k3_bridge_maps_current_visual_weights_without_source_passthrough(k3_modules):
    bridge = k3_modules.bridge.KimiK3VLBridge()

    registry = bridge.mapping_registry()

    assert bridge._HF_PASSTHROUGH_PREFIXES == ()
    for original, wrapped in zip(bridge.language_mappings, registry.mappings):
        assert not original.megatron_param.startswith("language_model.")
        assert wrapped.megatron_param == f"language_model.{original.megatron_param}"
        assert wrapped.hf_param == original.hf_param
    assert [(mapping.megatron_param, mapping.hf_param) for mapping in registry.mappings[-2:]] == [
        ("vision_tower.**", "vision_tower.**"),
        ("mm_projector.**", "mm_projector.**"),
    ]


@pytest.mark.parametrize("legacy_output_recorder", [True, False])
def test_kimi_k3_vision_builder_loads_only_vision_classes_and_marks_tp_gradients(
    monkeypatch, k3_modules, legacy_output_recorder
):
    loaded = []
    hooked = []
    recorder = object()
    generic = SimpleNamespace(OutputRecorder=recorder) if legacy_output_recorder else SimpleNamespace()

    class Vision(_Vision):
        _supports_flash_attn_2 = True

        def __init__(self, config):
            super().__init__()
            self.config = config

    def load_class(name, path, **kwargs):
        assert generic.OutputRecorder is recorder
        loaded.append((name, path, kwargs))
        return {
            "modeling_kimi_k3.MoonViT3dPretrainedModel": Vision,
            "modeling_kimi_k3.VisionTowerConfig": lambda cfg: SimpleNamespace(_attn_implementation="eager"),
            "modeling_kimi_k3.ProjectorConfig": lambda cfg: cfg,
            "modeling_kimi_k3.PatchMergerMLPV2": lambda cfg: _Projector(),
        }[name]

    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.utils.common_utils",
        SimpleNamespace(hook_hf_module_setattr_for_tp_grad_sync=hooked.append),
    )
    monkeypatch.setitem(
        sys.modules, "transformers.dynamic_module_utils", SimpleNamespace(get_class_from_dynamic_module=load_class)
    )
    monkeypatch.setitem(
        sys.modules, "transformers.utils", SimpleNamespace(is_flash_attn_2_available=lambda: True, generic=generic)
    )
    monkeypatch.setitem(sys.modules, "transformers.utils.output_capturing", SimpleNamespace(OutputRecorder=recorder))
    config = SimpleNamespace(
        hf_model_path="test-k3",
        trust_remote_code=True,
        vision_config=SimpleNamespace(mm_projector_type="patchmergerv2"),
        params_dtype=torch.float64,
    )

    vision, projector = k3_modules.model._build_vision_modules(config)

    assert {name for name, _, _ in loaded} == {
        "modeling_kimi_k3.MoonViT3dPretrainedModel",
        "modeling_kimi_k3.VisionTowerConfig",
        "modeling_kimi_k3.ProjectorConfig",
        "modeling_kimi_k3.PatchMergerMLPV2",
    }
    assert all(path == "test-k3" and kwargs == {"trust_remote_code": True} for _, path, kwargs in loaded)
    assert hooked == [vision, projector]
    assert vision._supports_flash_attn
    assert vision.config._attn_implementation == "flash_attention_2"
    assert all(parameter.dtype is torch.float64 for module in hooked for parameter in module.parameters())


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_kimi_k3_packed_cp_partitions_visual_features_before_sp(monkeypatch, k3_modules, cp_rank, sequence_parallel):
    model, vision, projector = _model(
        monkeypatch, k3_modules, context_parallel_size=2, sequence_parallel=sequence_parallel
    )
    model.cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank)
    token_ids = torch.tensor([[2, 31, 3, 4, 5, 6, 7, 8, 9] + [0] * 7, [3, 4, 31, 5, 6] + [0] * 11])
    valid = torch.arange(16).unsqueeze(0) < torch.tensor([[9], [5]])
    pixels = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(2, 3, 1, 1)
    packed = SimpleNamespace(qkv_format="thd", cu_seqlens_q_cpu=[0, 16, 24])
    full_embeddings = model.language_model.embedding(input_ids=token_ids).transpose(0, 1).detach().clone()
    full_embeddings[token_ids == 31] = pixels.flatten(1) * 6
    full_embeddings[~valid] = 0
    indices = (
        list(range(0, 4)) + list(range(12, 16)) + list(range(16, 18)) + list(range(22, 24))
        if cp_rank == 0
        else list(range(4, 12)) + list(range(18, 22))
    )
    expected = full_embeddings.flatten(0, 1)[indices].unsqueeze(1)
    scatter_calls = []

    def scatter(value, group):
        assert group is model.tp_group
        scatter_calls.append(value.detach().clone())
        return value[: value.shape[0] // 2]

    monkeypatch.setattr(k3_modules.model, "scatter_to_sequence_parallel_region", scatter)
    output = model(
        token_ids,
        attention_mask=valid,
        labels=token_ids,
        loss_mask=valid.to(torch.float32),
        packed_seq_params=packed,
        pixel_values=pixels,
        grid_thws=torch.tensor([[1, 1, 1], [1, 1, 1]]),
    )
    call = model.language_model.calls[-1]
    assert call["packed_seq_params"] is packed
    assert call["attention_mask"] is None
    torch.testing.assert_close(call["input_ids"], token_ids.flatten()[indices].unsqueeze(0))
    torch.testing.assert_close(call["labels"], call["input_ids"])
    torch.testing.assert_close(call["loss_mask"], valid.flatten()[indices].unsqueeze(0).float())
    if sequence_parallel:
        torch.testing.assert_close(scatter_calls[0], expected)
        expected = expected[:6]
    torch.testing.assert_close(output, expected)
    assert call["padding_mask"].shape == (1, expected.size(0))
    output.sum().backward()
    assert vision.proj.weight.grad is not None
    assert projector.proj.weight.grad is not None


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_kimi_k3_bshd_cp_visual_gradients_sum_to_full_sequence(monkeypatch, k3_modules, cp_rank):
    model, vision, projector = _model(monkeypatch, k3_modules, context_parallel_size=2)
    model.cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank)
    tokens = torch.tensor([[2, 31, 3, 4, 5, 6, 7, 8]])
    output = model(tokens, pixel_values=torch.ones(1, 3, 1, 1), grid_thws=torch.tensor([[1, 1, 1]]))
    output.sum().backward()
    # All image tokens reside on CP rank 0. Rank 1 must still produce zero
    # (rather than absent) visual grads for the normal DP*CP reduction.
    expected = 1 if cp_rank == 0 else 0
    assert int(torch.count_nonzero(vision.proj.weight.grad) > 0) == expected
    assert int(torch.count_nonzero(projector.proj.weight.grad) > 0) == expected
    if cp_rank == 0:
        full, full_vision, full_projector = _model(monkeypatch, k3_modules)
        full(tokens, pixel_values=torch.ones(1, 3, 1, 1), grid_thws=torch.tensor([[1, 1, 1]])).sum().backward()
        torch.testing.assert_close(vision.proj.weight.grad, full_vision.proj.weight.grad)
        torch.testing.assert_close(projector.proj.weight.grad, full_projector.proj.weight.grad)


def test_kimi_k3_later_pipeline_stage_partitions_metadata_and_preserves_attnres_payload(monkeypatch, k3_modules):
    model, _, _ = _model(monkeypatch, k3_modules, pre_process=False, context_parallel_size=2)
    payload = torch.randn(4, 1, 12)
    model.set_input_tensor([payload])
    tokens = torch.arange(8).unsqueeze(0)
    result = model(tokens, labels=tokens)
    assert result[0] is payload
    torch.testing.assert_close(model.language_model.calls[-1]["input_ids"], torch.tensor([[0, 1, 6, 7]]))
    torch.testing.assert_close(model.language_model.calls[-1]["labels"], torch.tensor([[0, 1, 6, 7]]))


def test_kimi_k3_later_pipeline_stage_keeps_ragged_cp_sp_padding_and_payload(monkeypatch, k3_modules):
    model, _, _ = _model(monkeypatch, k3_modules, pre_process=False, context_parallel_size=2, sequence_parallel=True)
    payload = [torch.randn(6, 1, 12)]
    model.set_input_tensor(payload)
    tokens = torch.ones(2, 16, dtype=torch.long)
    valid = torch.arange(16).unsqueeze(0) < torch.tensor([[9], [5]])
    packed = SimpleNamespace(qkv_format="thd", cu_seqlens_q_cpu=[0, 16, 24])
    monkeypatch.setattr(k3_modules.model, "scatter_to_sequence_parallel_region", lambda value, group: value[:6])

    result = model(tokens, attention_mask=valid, packed_seq_params=packed)

    assert result is payload
    assert result[0].shape == (6, 1, 12)
    call = model.language_model.calls[-1]
    assert call["decoder_input"] is None
    assert call["packed_seq_params"] is packed
    assert packed.cu_seqlens_q_cpu == [0, 16, 24]
    assert call["input_ids"].shape == (1, 12)
    torch.testing.assert_close(call["padding_mask"], torch.tensor([[False, False, False, False, True, True]]))


@pytest.mark.parametrize("cp_size", [2, 8])
@pytest.mark.parametrize("layout", ["bshd", "ragged", "packed"])
@pytest.mark.parametrize("frozen", ["none", "vision", "projector"])
def test_kimi_k3_local_embedding_matches_full_merge_outputs_and_gradients(
    monkeypatch, k3_modules, cp_size, layout, frozen
):
    full, _, _ = _model(monkeypatch, k3_modules)
    if frozen != "none":
        getattr(full, "vision_tower" if frozen == "vision" else "mm_projector").requires_grad_(False)
    width = 4 * cp_size
    tokens = (torch.arange(2 * width).view(2, width) % 29) + 1
    valid = torch.arange(width)[None, :] < torch.tensor([[width - 3], [2 * cp_size - 1]])
    tokens[~valid] = 0
    tokens[0, 1:4] = 31
    tokens[1, 1] = 31
    packed = None
    if layout != "bshd":
        packed = SimpleNamespace(qkv_format="thd", cu_seqlens_q_cpu=[0, width, width + 2 * cp_size])
    if layout == "packed":
        tokens = torch.cat((tokens[0], tokens[1, : 2 * cp_size]))[None, :]
        valid = torch.cat((valid[0], valid[1, : 2 * cp_size]))[None, :]
    positions = torch.arange(tokens.numel()).view_as(tokens)
    pixels = torch.arange(12, dtype=torch.float32).view(4, 3, 1, 1) / 10
    kwargs = dict(pixel_values=pixels, grid_thws=torch.ones(4, 3, dtype=torch.long))
    for rank in range(cp_size):
        local, _, _ = _model(monkeypatch, k3_modules)
        local.load_state_dict(full.state_dict())
        if frozen != "none":
            getattr(local, "vision_tower" if frozen == "vision" else "mm_projector").requires_grad_(False)
        local.cp_group = SimpleNamespace(size=lambda: cp_size, rank=lambda: rank)
        seen = []
        local.language_model.embedding.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True
        )
        full.zero_grad(set_to_none=True)
        reference = full(tokens, position_ids=positions, **kwargs).transpose(0, 1)
        reference = reference.masked_fill(~valid[..., None], 0)
        indices, shape = k3_modules.model._cp_token_indices(tokens, packed, cp_size, rank)
        reference = reference.flatten(0, 1).index_select(0, indices).view(*shape, 3).transpose(0, 1)
        output = local(tokens, position_ids=positions, attention_mask=valid, packed_seq_params=packed, **kwargs)
        torch.testing.assert_close(output, reference)
        assert seen[0]["input_ids"].shape == shape
        torch.testing.assert_close(seen[0]["position_ids"], positions.flatten()[indices].view(shape))
        assert seen[0]["input_ids"].numel() < tokens.numel()
        weights = torch.linspace(0.1, 1, output.numel()).view_as(output)
        (reference.square() * weights).sum().backward()
        (output.square() * weights).sum().backward()
        for (name, actual), expected in zip(local.named_parameters(), full.parameters(), strict=True):
            if actual.requires_grad:
                assert actual.grad is not None, name
                torch.testing.assert_close(actual.grad, expected.grad, msg=name)
            else:
                assert actual.grad is None


@pytest.mark.parametrize("training", [True, False])
def test_kimi_k3_embedding_dropout_preserves_full_rng_or_uses_local_eval(monkeypatch, k3_modules, training):
    model, _, _ = _model(monkeypatch, k3_modules, context_parallel_size=2)
    model.language_model.embedding.embedding_dropout.p = 0.5
    model.train(training)
    reference = deepcopy(model)
    reference.cp_group = None
    tokens = torch.arange(1, 17).view(1, 16)
    observed = []
    model.language_model.embedding.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs["input_ids"].shape), with_kwargs=True
    )
    torch.manual_seed(19)
    full = reference(tokens)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    actual = model(tokens)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual, full[[0, 1, 2, 3, 12, 13, 14, 15]])
    assert observed == [(1, 16) if training else (1, 8)]


@pytest.mark.parametrize("feature_count", [1, 3])
def test_kimi_k3_local_embedding_rejects_visual_feature_mismatch(monkeypatch, k3_modules, feature_count):
    model, _, _ = _model(monkeypatch, k3_modules, context_parallel_size=2)
    with pytest.raises(RuntimeError, match="pre-expanded media placeholder count"):
        model(
            torch.tensor([[1, 31, 2, 3, 4, 31, 5, 6]]),
            pixel_values=torch.ones(feature_count, 3, 1, 1),
            grid_thws=torch.ones(feature_count, 3, dtype=torch.long),
        )


def test_kimi_k3_local_embedding_handles_empty_visual_output_without_gather(monkeypatch, k3_modules):
    model, _, _ = _model(monkeypatch, k3_modules, context_parallel_size=2)
    features = torch.empty(0, 3, requires_grad=True)
    monkeypatch.setattr(model, "_image_features", lambda *args: features)
    output = model(
        torch.arange(1, 9)[None, :], pixel_values=torch.ones(1, 3, 1, 1), grid_thws=torch.ones(1, 3, dtype=torch.long)
    )
    output.sum().backward()
    assert features.grad is not None
    assert output.shape == (4, 1, 3)
