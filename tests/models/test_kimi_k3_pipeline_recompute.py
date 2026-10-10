# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for K3 AttnRes, fixed CP, pipeline payloads, and full
recompute.

This uses the real K3 layer, Core uniform checkpoint loop and
CheckpointFunction. Only CUDA RNG access, TE projections and FLA kernels use
CPU substitutes. Pipeline send/recv is represented by a detached payload and an
explicit gradient handoff.
"""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from tests.models.test_kimi_k3_context_parallel import (
    _attention,
    _bridge_layers,
    _cpu_core_all_to_all,
    _partition_indices,
    _reference_conv,
    _reference_kda,
)


torch = pytest.importorskip("torch")
dist = torch.distributed


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(6, 9, bias=False)
        self.down = torch.nn.Linear(9, 6, bias=False)

    def forward(self, value, padding_mask=None):
        assert padding_mask is None
        return self.down(torch.nn.functional.silu(self.up(value))), None


def _layer(module, layer_number, cp_group, cp_mode):
    layer = module.KimiK3TransformerLayer.__new__(module.KimiK3TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(hidden_size=6, num_layers=4, cuda_graph_impl="none")
    layer.layer_number = layer_number
    layer.attn_res_block_size = 3
    layer.is_stage_entry = False
    layer.is_stage_exit = False
    layer.self_attention = _attention(module, cp_group, mode=cp_mode)
    layer.self_attention.is_kda = True
    layer.self_attention.sequence_parallel = False
    layer.mlp = _MLP()
    layer.input_layernorm = module.KimiRMSNorm(6, 1e-5)
    layer.pre_mlp_layernorm = module.KimiRMSNorm(6, 1e-5)
    for name in ("self_attention", "mlp"):
        setattr(layer, name + "_res_norm", module.KimiRMSNorm(6, 1e-5))
        setattr(layer, name + "_res_proj", torch.nn.Linear(6, 1, bias=False))
    if layer_number == 1:
        # The empty first-layer bank never uses these parameters in upstream K3.
        layer.self_attention_res_norm.requires_grad_(False)
        layer.self_attention_res_proj.requires_grad_(False)
    if layer_number == 4:
        layer.output_attn_res_norm = module.KimiRMSNorm(6, 1e-5)
        layer.output_attn_res_proj = torch.nn.Linear(6, 1, bias=False)
    return layer.double()


def _block(layers):
    from megatron.core.transformer.transformer_block import TransformerBlock

    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        fp8=None,
        fp4=None,
    )
    block.layers = torch.nn.ModuleList(layers)
    block.num_layers_per_pipeline_rank = len(layers)
    return block


def _checkpointed(block, hidden_states, params):
    return block._checkpointed_forward(
        hidden_states=hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        attention_bias=None,
        packed_seq_params=params,
        use_inner_quantization_context=False,
    )


def _plain(layers, hidden_states, params, context=None):
    for layer in layers:
        hidden_states, context = layer(hidden_states, context=context, packed_seq_params=params)
    return hidden_states, context


def _loss(value, microbatch):
    # Unequal microbatch weights catch accidental state/gradient reuse.
    return (value.square().sum() + value.sin().sum() * 0.1) * (microbatch + 1)


def _distributed_pipeline_recompute(rank, world_size, rendezvous, cp_mode):
    import megatron.bridge.models.kimi.kimi_k3_layers as module
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel import random as core_random

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    group = dist.group.WORLD
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            _cpu_core_all_to_all(monkeypatch)
            monkeypatch.setattr(module, "causal_conv1d", _reference_conv)
            monkeypatch.setattr(module, "kda", _reference_kda)
            monkeypatch.setattr(module, "build_cp_context", lambda **kwargs: SimpleNamespace(**kwargs))
            rng_snapshots = []

            def get_cpu_rng_state():
                rng_snapshots.append(True)
                return (torch.get_rng_state(),)

            monkeypatch.setattr(core_random, "_get_all_rng_states", get_cpu_rng_state)
            monkeypatch.setattr(core_random, "_set_all_rng_states", torch.set_rng_state)

            torch.manual_seed(73)
            baseline = torch.nn.ModuleList([_layer(module, index, None, cp_mode) for index in range(1, 5)])
            parallel = torch.nn.ModuleList([_layer(module, index, group, cp_mode) for index in range(1, 5)])
            parallel.load_state_dict(baseline.state_dict())
            # Layer 3 is still inside the first 3-layer AttnRes block.
            parallel[1].is_stage_exit = True
            parallel[2].is_stage_entry = True
            first_stage = _block(parallel[:2])
            last_stage = _block(parallel[2:])

            calls = [[] for _ in parallel]
            for index, layer in enumerate(parallel):
                layer.register_forward_hook(
                    lambda _module, _inputs, _output, index=index: calls[index].append(torch.is_grad_enabled())
                )

            microbatches = []
            for microbatch, boundaries in enumerate(([0, 8, 20], [0, 12, 24])):
                cu = torch.tensor(boundaries, dtype=torch.int32)
                params = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    cu_seqlens_q_padded=cu,
                    cu_seqlens_kv_padded=cu,
                    cp_partition_mode="zigzag",
                )
                params.cu_seqlens_q_cpu = list(boundaries)
                indices = _partition_indices(boundaries, rank, world_size, "zigzag")
                full_input = torch.randn(boundaries[-1], 1, 6, dtype=torch.float64, requires_grad=True)
                local_input = full_input.detach().index_select(0, indices).requires_grad_()
                prefix, bank = _plain(baseline[:2], full_input, copy.copy(params))
                expected, _ = _plain(baseline[2:], prefix, copy.copy(params), context=bank)

                # A separate uncheckpointed suffix gives prefix/bank input-gradient
                # oracles, excluding upstream uses of the same snapshot tensor.
                oracle_prefix = prefix.detach().clone().requires_grad_()
                oracle_bank = bank.detach().clone().requires_grad_()
                oracle_suffix = copy.deepcopy(baseline[2:])
                oracle_output, _ = _plain(oracle_suffix, oracle_prefix, copy.copy(params), context=oracle_bank)

                payload = _checkpointed(first_stage, local_input, params)
                assert type(payload.grad_fn).__name__ == "CheckpointFunctionBackward"
                assert payload.shape[-1] == 12  # Prefix plus one AttnRes snapshot.
                received = payload.detach().clone().requires_grad_()
                actual = _checkpointed(last_stage, received, params)
                assert type(actual.grad_fn).__name__ == "CheckpointFunctionBackward"
                assert params.cp_partition_mode == "zigzag"
                torch.testing.assert_close(actual, expected.index_select(0, indices), atol=2e-6, rtol=2e-5)
                microbatches.append(
                    SimpleNamespace(
                        index=microbatch,
                        indices=indices,
                        params=params,
                        full_input=full_input,
                        local_input=local_input,
                        expected=expected,
                        actual=actual,
                        payload=payload,
                        received=received,
                        oracle_output=oracle_output,
                        oracle_prefix=oracle_prefix,
                        oracle_bank=oracle_bank,
                    )
                )

            # Keep two distinct packed-metadata graphs live before any backward.
            assert all(values == [False, False] for values in calls)
            for batch in reversed(microbatches):
                _loss(batch.actual, batch.index).backward()
                assert batch.received.grad is not None
                batch.payload.backward(batch.received.grad)
                _loss(batch.expected, batch.index).backward()
                _loss(batch.oracle_output, batch.index).backward()
                assert batch.params.cp_partition_mode == "zigzag"
                torch.testing.assert_close(
                    batch.local_input.grad,
                    batch.full_input.grad.index_select(0, batch.indices),
                    atol=5e-6,
                    rtol=5e-5,
                )
                expected_payload_grad = torch.cat(
                    (batch.oracle_prefix.grad, batch.oracle_bank.grad.flatten(-2)), dim=-1
                )
                torch.testing.assert_close(
                    batch.received.grad,
                    expected_payload_grad.index_select(0, batch.indices),
                    atol=5e-6,
                    rtol=5e-5,
                )
                assert batch.received.grad[..., 6:].abs().sum() > 0

            assert all(values == [False, False, True, True] for values in calls)
            assert len(rng_snapshots) >= 16
            baseline_parameters = dict(baseline.named_parameters())
            for name, parameter in parallel.named_parameters():
                expected_parameter = baseline_parameters[name]
                if not parameter.requires_grad:
                    assert parameter.grad is None and expected_parameter.grad is None
                    continue
                assert parameter.grad is not None, name
                dist.all_reduce(parameter.grad, group=group)
                torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=8e-6, rtol=8e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_mode", ["headwise", "chunkwise"])
def test_kimi_k3_cp2_pp2_full_uniform_recompute_preserves_two_microbatch_gradients(tmp_path, cp_mode):
    _bridge_layers()
    pytest.importorskip("megatron.core.transformer.transformer_block")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("K3 CP/pipeline recompute CPU test requires PyTorch Gloo support.")
    torch.multiprocessing.spawn(
        _distributed_pipeline_recompute,
        args=(2, (tmp_path / "pipeline_init").as_uri(), cp_mode),
        nprocs=2,
        join=True,
    )
