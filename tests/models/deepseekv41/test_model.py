# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only execution glue tests; no CUDA module construction or distributed
job."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge.models.conversion.model_bridge")

from megatron.core.models.gpt.gpt_model import GPTModel  # noqa: E402
from megatron.core.transformer.hyper_connection import HyperConnectionModule  # noqa: E402
from megatron.core.transformer.spec_utils import ModuleSpec  # noqa: E402
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules  # noqa: E402
from megatron.core.transformer.transformer_layer import (  # noqa: E402
    HyperConnectionTransformerLayer,
    TransformerLayerSubmodules,
)

from relax.models.deepseekv41 import model as implementation  # noqa: E402
from relax.models.deepseekv41.model import (  # noqa: E402
    DeepSeekV41GPTModel,
    V41HyperConnection,
    V41TransformerBlock,
    V41TransformerLayer,
    get_v41_spec,
)
from relax.models.deepseekv41.runtime import (  # noqa: E402
    V41CarryPlan,
    V41IdsOnlyPlan,
    V41Runtime,
    v41_pack_carry,
    v41_unpack_carry,
)


@pytest.fixture
def model_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_layers=3,
        hidden_size=2,
        num_residual_streams=4,
        csa_compress_ratios=[0, 0, 0],
        v41_kv_source_layer_ids=(),
        v41_index_source_layer_ids=(),
        v41_engram_layer_ids=(),
        v41_candidate_source_layer_id=-1,
        v41_candidate_block_size=2,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        experimental_attention_variant=None,
        recompute_granularity=None,
        enable_hyper_connections=True,
        kv_lora_rank=16,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=3,
        layernorm_epsilon=1e-20,
        v41_hc_eps=1e-6,
    )


@pytest.fixture
def core_hooks(monkeypatch) -> None:
    # Isolate the construction hooks from GPU model allocation.
    monkeypatch.setattr(GPTModel, "_get_transformer_block_class", lambda self: TransformerBlock, raising=False)
    monkeypatch.setattr(TransformerBlock, "_build_hyper_connection_output_head", lambda self: None, raising=False)


def _spec(runtime: V41Runtime) -> TransformerBlockSubmodules:
    return TransformerBlockSubmodules(
        layer_specs=[ModuleSpec(module=V41TransformerLayer, params={"runtime": runtime})],
        layer_norm=torch.nn.Identity,
    )


def _fake_layer_init(self, config, submodules, layer_number=1, **kwargs) -> None:
    torch.nn.Module.__init__(self)
    self.config = config
    self.layer_number = layer_number
    self.pg_collection = kwargs.get("pg_collection", SimpleNamespace(tp=None, cp=None))


def _fake_layer_forward(self, hidden_states, *args, **kwargs):
    self.v41_runtime.pre = torch.sigmoid(hidden_states.float().mean(-1, keepdim=True)).expand(
        *hidden_states.shape[:2], self.config.num_residual_streams
    )
    return hidden_states * 2, kwargs.get("context")


def test_model_hyper_connection_uses_previous_pre_and_updates_next(model_config, monkeypatch) -> None:
    def fake_init(self, config, layer_number):
        torch.nn.Module.__init__(self)
        self.config = config
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = 2
        self._fused_add_3_op = lambda a, b, c: a + b + c

    monkeypatch.setattr(HyperConnectionModule, "__init__", fake_init)
    old_pre = torch.tensor([[[1.0, 0.5, 0.0, 0.25]]])
    next_pre = torch.full((1, 1, 4), 0.75)
    runtime = V41Runtime(pre=old_pre)
    module = V41HyperConnection(model_config, 1, runtime)
    module._projection_and_get_norm = lambda value: (torch.zeros(1, 1, 24), torch.ones(1, 1, 1))
    module._compute_h = lambda projection, norm: (next_pre, torch.ones(1, 1, 4), torch.zeros(1, 1, 16))
    module._sinkhorn_op = lambda logits, iterations, epsilon: torch.eye(4).view(1, 1, 4, 4)
    hidden = torch.arange(8, dtype=torch.bfloat16).view(1, 1, 8)
    output, mixing, post, residual = module(hidden)
    expected = (hidden.view(1, 1, 4, 2).float() * old_pre.unsqueeze(-1)).sum(2).to(hidden.dtype)
    assert torch.equal(output, expected)
    assert runtime.pre is next_pre
    assert torch.equal(residual, hidden)
    assert mixing.shape == (1, 1, 4, 4) and post.shape == (1, 1, 4)
    assert module.norm_eps == model_config.layernorm_epsilon
    assert module.sinkhorn_eps == module.compute_h_eps == model_config.v41_hc_eps


def test_model_hyper_connection_fp32_mix_orientation_and_gradients() -> None:
    module = V41HyperConnection.__new__(V41HyperConnection)
    torch.nn.Module.__init__(module)
    module.n, module.hidden_size = 4, 2
    hidden = torch.arange(8, dtype=torch.bfloat16).view(1, 1, 8).requires_grad_()
    output = torch.tensor([[[2.0, 3.0]]], dtype=torch.bfloat16, requires_grad=True)
    mixing = torch.tensor([[[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]]], dtype=torch.float32)
    post = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]], requires_grad=True)
    result = module.fused_h_res_h_post_bda(mixing, hidden, post, (output, None), 0.0, True, False)
    streams = hidden.view(1, 1, 4, 2)
    expected = torch.stack([streams[:, :, 3], streams[:, :, 0], streams[:, :, 1], streams[:, :, 2]], dim=2)
    expected = expected.float() + post.unsqueeze(-1) * output.float().unsqueeze(2)
    assert torch.equal(result, expected.flatten(-2).to(torch.bfloat16))
    result.float().sum().backward()
    assert hidden.grad is not None and output.grad is not None and post.grad is not None


def test_model_spec_preserves_core_moe_and_binds_only_instance_state(model_config, monkeypatch) -> None:
    sentinel_mlp = ModuleSpec(module=torch.nn.Identity)
    sentinel_norm = torch.nn.Identity
    calls = []

    def native_spec(config, vp_stage=None, pp_rank=None):
        calls.append((config, vp_stage, pp_rank))
        return TransformerBlockSubmodules(
            layer_specs=[
                ModuleSpec(
                    module=HyperConnectionTransformerLayer,
                    params={"existing": 123},
                    submodules=TransformerLayerSubmodules(
                        input_layernorm=sentinel_norm,
                        self_attention=ModuleSpec(module=torch.nn.Identity),
                        pre_mlp_layernorm=sentinel_norm,
                        mlp=sentinel_mlp,
                    ),
                )
                for _ in range(2)
            ],
            layer_norm=sentinel_norm,
        )

    monkeypatch.setattr(implementation, "get_transformer_block_with_experimental_attention_variant_spec", native_spec)
    first, second = V41Runtime(), V41Runtime()
    original_forward, original_expand = GPTModel.forward, HyperConnectionModule.input_expand
    one = get_v41_spec(model_config, first, vp_stage=None, pp_rank=0)
    two = get_v41_spec(model_config, second, vp_stage=None, pp_rank=1)
    for block, runtime in ((one, first), (two, second)):
        for layer in block.layer_specs:
            assert layer.module is V41TransformerLayer
            assert layer.params == {"existing": 123, "runtime": runtime}
            assert layer.submodules.mlp is sentinel_mlp
            assert layer.submodules.pre_mlp_layernorm is sentinel_norm
            assert layer.submodules.input_layernorm is sentinel_norm
            assert layer.submodules.self_attention.params["runtime"] is runtime
            assert layer.submodules.self_attention_hyper_connection.params["runtime"] is runtime
            assert layer.submodules.mlp_hyper_connection.params["runtime"] is runtime
    assert len(calls) == 2 and calls[1][2] == 1
    assert GPTModel.forward is original_forward
    assert HyperConnectionModule.input_expand is original_expand
    assert not hasattr(model_config, "v41_runtime")


def test_model_block_keeps_native_expand_without_old_output_head(model_config, monkeypatch, core_hooks) -> None:
    def fake_init(self, config, spec, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.pre_process = kwargs.get("pre_process", True)
        self.num_residual_streams = config.num_residual_streams
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.final_layernorm = torch.nn.Identity()
        self._build_hyper_connection_output_head()

    monkeypatch.setattr(TransformerBlock, "__init__", fake_init)
    model_config.recompute_granularity = "full"
    model_config.v41_engram_layer_ids = (0,)
    runtime = V41Runtime(input_ids=torch.tensor([[10, 20, 30]]))
    block = V41TransformerBlock(model_config, _spec(runtime))
    hidden = torch.arange(6, dtype=torch.bfloat16).view(3, 1, 2)
    packed = block.preprocess_for_layer_schedule(hidden)
    restored = V41Runtime()
    expanded = v41_unpack_carry(packed, restored, V41IdsOnlyPlan(model_config))
    assert torch.equal(expanded, hidden.unsqueeze(2).expand(3, 1, 4, 2).reshape(3, 1, 8))
    assert torch.equal(restored.input_ids, runtime.input_ids)
    assert not any(name.startswith("hc_head") for name in block.state_dict())
    contracted = torch.randn(3, 1, 2, dtype=torch.bfloat16)
    result, multi = block.postprocess_for_layer_schedule(contracted, return_mhc_multistream=True)
    assert torch.equal(result, contracted) and multi is None
    with pytest.raises(ValueError, match="contract mHC"):
        block.postprocess_for_layer_schedule(expanded)


def test_model_last_layer_contracts_using_final_pre(model_config, monkeypatch) -> None:
    monkeypatch.setattr(HyperConnectionTransformerLayer, "__init__", _fake_layer_init)
    monkeypatch.setattr(HyperConnectionTransformerLayer, "forward", _fake_layer_forward)
    runtime = V41Runtime()
    layer = V41TransformerLayer(model_config, None, runtime, layer_number=3)
    hidden = torch.randn(3, 1, 8, dtype=torch.bfloat16)
    output, _ = layer.forward(hidden)
    expected = ((hidden * 2).view(3, 1, 4, 2).float() * runtime.pre.unsqueeze(-1)).sum(2).to(hidden.dtype)
    assert output.shape == (3, 1, 2) and torch.equal(output, expected)


def test_model_engram_receives_restored_ids_before_layer(model_config, monkeypatch) -> None:
    class FakeEngram(torch.nn.Module):
        def __init__(self, config, layer_id, tp_group, cp_group):
            super().__init__()
            self.seen = None

        def forward(self, hidden, ids):
            self.seen = ids
            return hidden + 1

    monkeypatch.setattr(HyperConnectionTransformerLayer, "__init__", _fake_layer_init)
    monkeypatch.setattr(HyperConnectionTransformerLayer, "forward", _fake_layer_forward)
    monkeypatch.setattr(implementation, "DeepSeekV41Engram", FakeEngram)
    model_config.recompute_granularity = "full"
    model_config.v41_engram_layer_ids = (0,)
    ids = torch.tensor([[10, 20, 30]])
    runtime = V41Runtime(input_ids=ids)
    layer = V41TransformerLayer(model_config, None, runtime, layer_number=1)
    hidden = torch.randn(3, 1, 8, dtype=torch.bfloat16)
    output, context = layer.forward(v41_pack_carry(hidden, runtime, V41IdsOnlyPlan(model_config)), context="context")
    restored = V41Runtime()
    result = v41_unpack_carry(output, restored, V41CarryPlan(model_config, 0))
    assert torch.equal(layer.v41_engram.seen, ids)
    assert torch.equal(result, (hidden + 1) * 2)
    assert context == "context" and restored.pre.shape == (3, 1, 4)
    assert restored.input_ids is None


def test_model_layer_recompute_restores_state_after_another_microbatch(model_config, monkeypatch) -> None:
    import torch.utils.checkpoint

    monkeypatch.setattr(HyperConnectionTransformerLayer, "__init__", _fake_layer_init)
    monkeypatch.setattr(HyperConnectionTransformerLayer, "forward", _fake_layer_forward)
    model_config.recompute_granularity = "full"
    runtime = V41Runtime()
    layer = V41TransformerLayer(model_config, None, runtime, layer_number=1)
    hidden = torch.randn(3, 1, 8, dtype=torch.bfloat16, requires_grad=True)
    output = torch.utils.checkpoint.checkpoint(lambda value: layer.forward(value)[0], hidden, use_reentrant=True)
    restored = V41Runtime()
    result = v41_unpack_carry(output, restored, V41CarryPlan(model_config, 0))
    loss = result.float().sum() + restored.pre.sum()
    runtime.reset()
    runtime.pre = torch.zeros(99)
    loss.backward()
    actual = hidden.grad.clone()
    reference = hidden.detach().clone().requires_grad_()
    raw, _ = layer.forward(reference)
    state = V41Runtime()
    unpacked = v41_unpack_carry(raw, state, V41CarryPlan(model_config, 0))
    (unpacked.float().sum() + state.pre.sum()).backward()
    assert torch.equal(actual, reference.grad)


def test_model_constructor_owns_runtime_and_freezes_only_router(model_config, monkeypatch, core_hooks) -> None:
    routers = []

    def fake_init(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        router = torch.nn.Linear(2, 3, bias=False)
        router.expert_bias = torch.nn.Parameter(torch.zeros(3))
        expert = torch.nn.Linear(2, 2, bias=False)
        self.decoder = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(router=router, expert=expert))])
        self.received = kwargs
        routers.append((router, expert))

    monkeypatch.setattr(GPTModel, "__init__", fake_init)
    monkeypatch.setattr(implementation, "get_v41_spec", lambda config, runtime, **kwargs: _spec(runtime))
    first = DeepSeekV41GPTModel(model_config, vocab_size=32, max_sequence_length=16, pg_collection=SimpleNamespace())
    second = DeepSeekV41GPTModel(model_config, vocab_size=32, max_sequence_length=16, pg_collection=SimpleNamespace())
    assert first.v41_runtime is not second.v41_runtime
    assert first.received["transformer_layer_spec"].layer_specs[0].params["runtime"] is first.v41_runtime
    assert first._get_transformer_block_class() is V41TransformerBlock
    assert not hasattr(model_config, "v41_runtime")
    for router, expert in routers:
        assert not router.weight.requires_grad and not router.expert_bias.requires_grad
        assert expert.weight.requires_grad
        value = torch.ones(1, 2, requires_grad=True)
        router(value).sum().backward()
        assert value.grad is not None


def test_model_forward_resets_state_and_preserves_standard_arguments(monkeypatch) -> None:
    model = DeepSeekV41GPTModel.__new__(DeepSeekV41GPTModel)
    torch.nn.Module.__init__(model)
    model.v41_runtime = V41Runtime(pre=torch.ones(99), latent={99: torch.ones(1)})
    captured = {}

    def native_forward(self, **kwargs):
        captured.update(kwargs)
        assert self.v41_runtime.pre is None and not self.v41_runtime.latent
        assert self.v41_runtime.input_ids is kwargs["input_ids"]
        return kwargs["decoder_input"]

    monkeypatch.setattr(GPTModel, "forward", native_forward)
    ids = torch.tensor([[1, 2]])
    decoder = torch.ones(2, 1, 2)
    processor, processor_context = object(), object()
    result = model(
        ids,
        None,
        None,
        decoder_input=decoder,
        output_processor=processor,
        output_processor_context=processor_context,
    )
    assert result is decoder
    assert captured["output_processor"] is processor and captured["output_processor_context"] is processor_context
    assert model.v41_runtime.input_ids is None
