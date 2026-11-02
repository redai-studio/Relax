# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch

from tests.backends.megatron.test_qsa_block_mask_patch import _block_presence, _selection


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QSA FlexAttention requires CUDA")
@pytest.mark.parametrize("lengths", [(512,), (12, 252, 248), (4, 124, 128, 256), (12, 508)])
@pytest.mark.parametrize("rank", [0, 1])
def test_local_mask_handles_unaligned_packed_cp_tiles(lengths, rank):
    from megatron.core.ssm.mamba_context_parallel import _thd_partitioned_indices
    from megatron.core.transformer.experimental_attention_variant import qsa

    from relax.models.qwen3_8.qsa_cp import _local_mask

    selection = _selection(lengths, "sparse")
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    ids = _thd_partitioned_indices(cu, sum(lengths), 2, rank)
    mask = _local_mask(selection, ids)
    dense = qsa.build_qsa_dense_mask(selection, sum(lengths))[0].index_select(0, ids.long()).cpu()
    partial = _block_presence(mask.kv_num_blocks, mask.kv_indices)
    full = _block_presence(mask.full_kv_num_blocks, mask.full_kv_indices)
    for q in range(partial.shape[0]):
        for k in range(partial.shape[1]):
            tile = dense[q * 128 : (q + 1) * 128, k * 128 : (k + 1) * 128]
            if tile.any():
                assert partial[q, k] or full[q, k]
            if full[q, k]:
                assert tile.all()

    torch.manual_seed(42)
    inputs = [torch.randn(1, 2, sum(lengths) // 2, 32, device="cuda", dtype=torch.bfloat16)]
    inputs += [torch.randn(1, 1, sum(lengths), 32, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    upstream = torch.randn_like(inputs[0])

    def run(sparse):
        query, key, value = [x.detach().requires_grad_() for x in inputs]
        if sparse:
            output = qsa._compiled_flex_attention()(query, key, value, block_mask=mask, enable_gqa=True)
        else:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                output = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=dense.to("cuda")[None, None], enable_gqa=True
                )
        output.backward(upstream)
        return output.detach(), query.grad, key.grad, value.grad

    for actual, expected in zip(run(True), run(False)):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MCore checkpoint saves CUDA RNG state")
def test_replay_isolates_interleaved_microbatches_and_repeated_backward(monkeypatch):
    from megatron.core import tensor_parallel

    from relax.models.qwen3_8 import qsa_cp as patch

    monkeypatch.setenv("RELAX_QWEN38_QSA_RECOMPUTE_REPLAY", "1")
    owner = object()
    builds = []

    def fn(value):
        def build():
            builds.append(1)
            return value.detach().clone()

        # Repeated calls of the same owner need distinct ordered entries.
        first = patch._cached("route", owner, build)
        second = patch._cached("route", owner, lambda: build() + 1)
        return value * (first + second)

    a = torch.tensor(2.0, device="cuda", requires_grad=True)
    b = torch.tensor(5.0, device="cuda", requires_grad=True)
    out_a = tensor_parallel.checkpoint(fn, False, a)
    out_b = tensor_parallel.checkpoint(fn, False, b)
    out_b.backward()
    out_a.backward(retain_graph=True)
    out_a.backward()
    assert len(builds) == 4
    torch.testing.assert_close(a.grad, torch.tensor(10.0, device="cuda"))
    torch.testing.assert_close(b.grad, torch.tensor(11.0, device="cuda"))
    assert patch._active_tape.get() is None


def test_replay_resets_context_after_exception(monkeypatch):
    import sys
    from types import ModuleType

    from tests.backends.megatron.test_qsa_flex_mask_replay import _load_functions

    patch = ModuleType("qsa_replay_exception_test")
    monkeypatch.setitem(sys.modules, patch.__name__, patch)
    _load_functions("qsa_cp.py", {"_ReplayTape", "_cached", "_with_replay"}, patch.__dict__)
    patch._active_tape = patch.ContextVar("test_exception_tape", default=None)

    def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        patch._with_replay(fail)()
    assert patch._active_tape.get() is None
