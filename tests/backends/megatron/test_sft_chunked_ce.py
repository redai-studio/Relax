# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the chunked-CE SFT loss optimization.

Covers two concerns:

1. **Math equivalence** — running the lm_head + per-token CE in chunks along
   the sequence dimension produces the same loss and the same gradients as
   running it on the full sequence in one shot. This is a pure-math invariant
   tested with plain torch primitives (no distributed init / no Megatron
   fused kernels needed).

2. **Bypass plumbing** — ``_bypass_output_layer`` correctly (a) finds the
   lm_head through DDP / VL wrappers, (b) keeps MTP head calls intact and
   replaces only the main ``output_layer.forward`` with a passthrough,
   (c) restores the original forward on exit, (d) handles missing output_layer
   as a no-op, and (e) passes ``tensor_parallel_output_grad=False`` to the SP
   gather so gradients are not silently scaled by TP (regression guard for the
   bug discovered 2026-06-17).
"""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from relax.engine.sft.runtime import should_use_sft_chunked


try:
    from relax.backends.megatron import loss as _loss_mod
    from relax.backends.megatron.loss import sft_loss_function_chunked
    from relax.backends.megatron.model import (
        _bypass_output_layer,
        _find_lm_output_layer,
    )
except Exception as exc:
    pytest.skip(f"relax.backends.megatron unavailable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# 1. Math equivalence: chunked lm_head + CE ≡ unchunked lm_head + CE
# ---------------------------------------------------------------------------


def _full_loss_and_grads(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor):
    """Reference: materialize logits, CE in one pass."""
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    logits = (h @ w.t()).float()  # [T, V] fp32 — same upcast pattern as the real path
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    loss.backward()
    return loss.detach(), h.grad.detach(), w.grad.detach()


def _chunked_loss_and_grads(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk_size: int):
    """Mirrors the chunking pattern in sft_loss_function_chunked:

    per-chunk matmul, per-chunk CE (no reduction), concat, mean.
    """
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    T = h.size(0)
    per_token_losses: list[torch.Tensor] = []
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        # cast to match weight dtype (matches the _chunked_call downcast)
        h_sub = h[s:e].to(w.dtype)
        logits_sub = (h_sub @ w.t()).float()
        # per-token CE — fused_vocab_parallel_cross_entropy is also per-token
        ce_sub = torch.nn.functional.cross_entropy(logits_sub, labels[s:e], reduction="none")
        per_token_losses.append(ce_sub)
    loss = torch.cat(per_token_losses, dim=0).mean()
    loss.backward()
    return loss.detach(), h.grad.detach(), w.grad.detach()


@pytest.mark.parametrize("chunk_size", [1, 13, 64, 128, 256])
def test_chunked_ce_loss_matches_unchunked(chunk_size: int):
    """Chunked CE must produce the same loss as one-shot CE, for any chunk
    size."""
    torch.manual_seed(0)
    T, H, V = 256, 32, 64
    hidden = torch.randn(T, H, dtype=torch.float32)
    weight = torch.randn(V, H, dtype=torch.float32) * 0.02
    labels = torch.randint(0, V, (T,))

    loss_ref, _, _ = _full_loss_and_grads(hidden, weight, labels)
    loss_chunk, _, _ = _chunked_loss_and_grads(hidden, weight, labels, chunk_size)

    torch.testing.assert_close(loss_ref, loss_chunk, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("chunk_size", [1, 13, 64, 128, 256])
def test_chunked_ce_gradients_match_unchunked(chunk_size: int):
    """Per-parameter gradients (hidden_states and lm_head.weight) must be
    identical between chunked and unchunked.

    This is the invariant that the `gather_from_sequence_parallel_region(...,
    tensor_parallel_output_grad=False)` fix protects in the real distributed
    path.
    """
    torch.manual_seed(0)
    T, H, V = 256, 32, 64
    hidden = torch.randn(T, H, dtype=torch.float32)
    weight = torch.randn(V, H, dtype=torch.float32) * 0.02
    labels = torch.randint(0, V, (T,))

    _, gh_ref, gw_ref = _full_loss_and_grads(hidden, weight, labels)
    _, gh_chunk, gw_chunk = _chunked_loss_and_grads(hidden, weight, labels, chunk_size)

    # fp32 → tight tolerance
    torch.testing.assert_close(gh_ref, gh_chunk, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gw_ref, gw_chunk, atol=1e-5, rtol=1e-5)


def test_chunked_ce_bf16_within_tolerance():
    """Same equivalence under bf16 (matches the actual runtime dtype), with a
    looser tolerance reflecting bf16 accumulation order sensitivity."""
    torch.manual_seed(0)
    T, H, V = 256, 32, 64
    hidden = torch.randn(T, H, dtype=torch.bfloat16)
    weight = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    labels = torch.randint(0, V, (T,))

    loss_ref, gh_ref, gw_ref = _full_loss_and_grads(hidden, weight, labels)
    loss_chunk, gh_chunk, gw_chunk = _chunked_loss_and_grads(hidden, weight, labels, chunk_size=64)

    # bf16 accumulation order varies between chunked and unchunked; widen tolerance.
    torch.testing.assert_close(loss_ref.float(), loss_chunk.float(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(gh_ref.float(), gh_chunk.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(gw_ref.float(), gw_chunk.float(), atol=5e-3, rtol=5e-3)


# ---------------------------------------------------------------------------
# 2. Bypass plumbing
# ---------------------------------------------------------------------------


class _FakeLmHead(torch.nn.Module):
    """ColumnParallelLinear-ish stand-in.

    ``sequence_parallel=False`` so the bypass skips the SP gather and we don't
    need ``mpu`` initialized.
    """

    def __init__(self, in_features: int = 4, out_features: int = 8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.sequence_parallel = False
        self.tp_group = object()  # truthy: short-circuits the `or mpu.get_...()` fallback

    def forward(self, input_, weight=None, runtime_gather_output=None):
        w = weight if weight is not None else self.weight
        return torch.nn.functional.linear(input_, w), None


class _FakeBridgeVL(torch.nn.Module):
    """language_model.output_layer — exercises the VL-wrapper walk."""

    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.output_layer = _FakeLmHead()


class _FakeDDP(torch.nn.Module):
    """`.module` unwrap step."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.module = inner


def test_find_lm_output_layer_plain():
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = _FakeLmHead()

    m = Plain()
    assert _find_lm_output_layer(m) is m.output_layer


def test_find_lm_output_layer_through_ddp():
    inner = _FakeBridgeVL()
    wrapped = _FakeDDP(inner)
    assert _find_lm_output_layer(wrapped) is inner.language_model.output_layer


def test_find_lm_output_layer_none_when_absent():
    class Empty(torch.nn.Module):
        pass

    assert _find_lm_output_layer(Empty()) is None


def test_bypass_returns_input_unchanged_during_model_forward():
    """Inside the bypass, output_layer.forward must be a passthrough."""
    head = _FakeLmHead(in_features=4, out_features=8)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    m = M()
    x = torch.randn(3, 4)
    with _bypass_output_layer(m) as lm_head_forward:
        y, bias = m.output_layer(x)
        # passthrough returns (input, None) — no matmul applied
        assert torch.equal(y, x)
        assert bias is None
        # the yielded callable still does the real matmul
        z, _ = lm_head_forward(x)
        assert z.shape == (3, 8)
        torch.testing.assert_close(z, x @ head.weight.t())


def test_bypass_preserves_mtp_heads_and_defers_only_main_head():
    """MTP depths use the real output layer; only the following main call is
    bypassed."""
    head = _FakeLmHead(in_features=4, out_features=8)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    m = M()
    x = torch.randn(3, 4)
    expected = x @ head.weight.t()

    with _bypass_output_layer(m, mtp_output_layer_calls=2) as lm_head_forward:
        mtp_1, _ = m.output_layer(x)
        mtp_2, _ = m.output_layer(x)
        main_hidden, main_bias = m.output_layer(x)

        torch.testing.assert_close(mtp_1, expected)
        torch.testing.assert_close(mtp_2, expected)
        assert torch.equal(main_hidden, x)
        assert main_bias is None

        main_logits, _ = lm_head_forward(main_hidden)
        torch.testing.assert_close(main_logits, expected)


def test_bypass_preserves_functional_mtp_head_call():
    """Relax's patched MTP path uses torch.func.functional_call for its
    detached head."""
    head = _FakeLmHead(in_features=4, out_features=8)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    m = M()
    x = torch.randn(3, 4)
    detached_params = {name: param.detach() for name, param in head.named_parameters()}

    with _bypass_output_layer(m, mtp_output_layer_calls=1) as lm_head_forward:
        mtp_logits, _ = torch.func.functional_call(
            head,
            detached_params,
            (x,),
            {"runtime_gather_output": None},
        )
        main_hidden, _ = m.output_layer(x)
        main_logits, _ = lm_head_forward(main_hidden)

    torch.testing.assert_close(mtp_logits, x @ head.weight.t())
    torch.testing.assert_close(main_logits, x @ head.weight.t())


def test_bypass_rejects_mtp_call_count_mismatch_and_restores_forward():
    """Fail clearly if MCore no longer calls the head once per configured MTP
    depth."""
    head = _FakeLmHead(in_features=4, out_features=8)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    m = M()
    x = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="main output_layer was not reached"):
        with _bypass_output_layer(m, mtp_output_layer_calls=1):
            m.output_layer(x)  # Consumed as the expected MTP call; main call is missing.

    logits, _ = m.output_layer(x)
    torch.testing.assert_close(logits, x @ head.weight.t())


def test_bypass_reuses_explicit_main_weight_for_tied_embeddings():
    """The deferred main call supplies the shared embedding weight for later
    chunks."""

    class TiedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = None
            self.sequence_parallel = False
            self.tp_group = object()

        def forward(self, input_, weight=None, runtime_gather_output=None):
            assert weight is not None
            return torch.nn.functional.linear(input_, weight), None

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = TiedHead()

    m = M()
    x = torch.randn(3, 4)
    shared_weight = torch.nn.Parameter(torch.randn(8, 4))

    with _bypass_output_layer(m) as lm_head_forward:
        main_hidden, _ = m.output_layer(x, weight=shared_weight)
        main_logits, _ = lm_head_forward(main_hidden)

    torch.testing.assert_close(main_logits, x @ shared_weight.t())


def test_bypass_restores_forward_on_exit():
    """After the context exits the layer must compute the real matmul again."""
    head = _FakeLmHead(in_features=4, out_features=8)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    m = M()
    x = torch.randn(3, 4)
    expected = x @ head.weight.t()

    with _bypass_output_layer(m) as _:
        hidden, _ = m.output_layer(x)
        assert torch.equal(hidden, x)

    y, _ = m.output_layer(x)
    torch.testing.assert_close(y, expected)


def test_bypass_restores_after_exception():
    """An exception inside the with-block must still restore forward."""
    head = _FakeLmHead(in_features=4, out_features=8)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    m = M()
    with pytest.raises(RuntimeError):
        with _bypass_output_layer(m):
            raise RuntimeError("boom")

    # forward must work normally after the exception
    x = torch.randn(2, 4)
    y, _ = m.output_layer(x)
    torch.testing.assert_close(y, x @ head.weight.t())


def test_bypass_noop_when_no_output_layer():
    class M(torch.nn.Module):
        pass

    with _bypass_output_layer(M()) as lm_head_forward:
        assert lm_head_forward is None


def test_bypass_chunked_call_handles_fp32_input_against_bf16_weight():
    """Regression: hidden_states arrives in fp32 from the bridge upstream,
    weight is bf16; _chunked_call must downcast input so matmul succeeds.
    Without the dtype align this raises ``expected mat1 and mat2 to have
    the same dtype``.
    """
    head = _FakeLmHead(in_features=4, out_features=8)
    head.weight.data = head.weight.data.to(torch.bfloat16)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    m = M()
    x_fp32 = torch.randn(3, 4, dtype=torch.float32)
    with _bypass_output_layer(m) as lm_head_forward:
        hidden, _ = m.output_layer(x_fp32)
        y, _ = lm_head_forward(hidden)  # would raise without the downcast
    assert y.dtype == torch.bfloat16
    assert y.shape == (3, 8)


def test_bypass_restores_sequence_parallel_flag_after_chunked_call():
    """``_chunked_call`` temporarily sets ``sequence_parallel=False`` and must
    restore the original value on exit (both normal and exceptional)."""
    head = _FakeLmHead(in_features=4, out_features=8)
    head.sequence_parallel = True  # pretend SP is on

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    m = M()
    # We can't actually call _chunked_call with SP=True without mpu; instead
    # patch out the SP gather by force-disabling sp_enabled at call time.
    # Just verify the flag is restored after a chunked_call.
    x = torch.randn(2, 4)
    head.sequence_parallel = False  # bypass's sp_enabled is captured here as False
    with _bypass_output_layer(m) as lm_head_forward:
        hidden, _ = m.output_layer(x)
        head.sequence_parallel = True  # simulate concurrent external flip
        lm_head_forward(hidden)
        assert head.sequence_parallel is True, "chunked_call must restore SP flag"


# ---------------------------------------------------------------------------
# 3. Functional checks for the SP gather kwarg, the chunked dispatch chain,
#    and the predicate gate.
# ---------------------------------------------------------------------------


def test_bypass_sp_gather_passes_output_grad_false(monkeypatch):
    """With SP-on, ``_passthrough`` must call
    ``gather_from_sequence_parallel_region`` with
    ``tensor_parallel_output_grad=False``.

    Default ``True`` triggers reduce-scatter backward which double-counts the
    chunked matmul's input-grad AllReduce, scaling every gradient by TP_size
    (observed as ~2× grad_norm on TP=2 — the 2026-06-17 bug). Functional check:
    monkeypatch the gather function with a kwarg-capturing spy and assert.
    """
    head = _FakeLmHead(in_features=4, out_features=8)
    head.sequence_parallel = True  # flip SP on so the gather branch fires

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    captured_kwargs: dict = {}

    def spy_gather(input_, **kwargs):
        captured_kwargs.update(kwargs)
        return input_

    # `_bypass_output_layer` does `from megatron.core.tensor_parallel.mappings
    # import gather_from_sequence_parallel_region` lazily inside the function;
    # patching the attribute on that module redirects the import.
    monkeypatch.setattr(
        "megatron.core.tensor_parallel.mappings.gather_from_sequence_parallel_region",
        spy_gather,
    )

    m = M()
    x = torch.randn(3, 4)
    with _bypass_output_layer(m):
        m.output_layer(x)  # routes to patched _passthrough → spy_gather

    assert captured_kwargs.get("tensor_parallel_output_grad") is False, (
        f"_passthrough must pass tensor_parallel_output_grad=False — got "
        f"kwargs={captured_kwargs}. Default True scales backward grads by TP_size."
    )


def test_bypass_can_skip_sp_gather_for_mtp_only_anchor(monkeypatch):
    head = _FakeLmHead(in_features=4, out_features=8)
    head.sequence_parallel = True

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = head

    def fail_gather(*_args, **_kwargs):
        raise AssertionError("MTP-only passthrough must not gather sequence-parallel hidden states")

    monkeypatch.setattr(
        "megatron.core.tensor_parallel.mappings.gather_from_sequence_parallel_region",
        fail_gather,
    )

    x = torch.randn(3, 4)
    with _bypass_output_layer(M(), gather_passthrough=False):
        hidden, _ = head(x)

    assert torch.equal(hidden, x)


def test_chunked_loss_delegates_to_get_log_probs_and_entropy(monkeypatch):
    """``sft_loss_function_chunked`` must call ``get_log_probs_and_entropy``
    passing ``lm_head_forward=``.

    Otherwise the chunked path forks from the legacy SFT layout machinery (per-
    sample slicing + CP redistribute) which lives entirely inside
    ``get_log_probs_and_entropy``. Functional check: spy on the dispatched
    helper and assert it receives our marker callable verbatim.
    """
    captured_kwargs: dict = {}

    def spy_glpae(logits, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.empty((0,), device=logits.device), {"log_probs": [torch.zeros(3)]}

    monkeypatch.setattr(_loss_mod, "get_log_probs_and_entropy", spy_glpae)

    lm_head_marker = object()
    args = Namespace(qkv_format="thd", calculate_per_token_loss=False)
    batch = {
        "unconcat_tokens": [torch.zeros(3, dtype=torch.long)],
        "total_lengths": [3],
        "response_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "max_seq_lens": None,
        "padded_total_lengths": None,
    }
    hidden_states = torch.randn(1, 3, 4)

    sft_loss_function_chunked(
        args,
        batch,
        hidden_states,
        lambda x: x.sum(),
        lm_head_forward=lm_head_marker,
    )

    assert captured_kwargs.get("lm_head_forward") is lm_head_marker, (
        f"sft_loss_function_chunked must thread lm_head_forward into "
        f"get_log_probs_and_entropy — got kwargs={captured_kwargs}."
    )


@pytest.mark.parametrize(
    ("configured_value", "expected"),
    [
        (True, True),
        (False, False),
        (None, True),
    ],
)
def test_loss_function_forwards_recompute_checkpoint_mode(monkeypatch, configured_value, expected):
    captured_kwargs: dict = {}

    def spy_checkpoint(_function, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.tensor(1.0), {"loss": torch.tensor(1.0)}

    monkeypatch.setattr(_loss_mod, "checkpoint", spy_checkpoint)
    monkeypatch.setattr(_loss_mod, "get_cp_local_num_tokens", lambda *args, **kwargs: torch.tensor(1))
    monkeypatch.setattr(_loss_mod, "get_sum_of_sample_mean", lambda *args, **kwargs: object())

    args = Namespace(
        loss_type="sft",
        recompute_loss_function=True,
        sft_chunked_logits=False,
        qkv_format="thd",
        calculate_per_token_loss=True,
        allgather_cp=False,
        global_batch_size=1,
    )
    if configured_value is not None:
        args.recompute_loss_function_use_reentrant = configured_value
    batch = {
        "total_lengths": [1],
        "response_lengths": [1],
        "loss_masks": [torch.ones(1)],
    }

    _loss_mod.loss_function(args, batch, num_microbatches=1, logits=torch.ones(1))

    assert captured_kwargs["use_reentrant"] is expected


@pytest.mark.parametrize(
    ("loss_type", "chunked_flag", "expected"),
    [
        ("sft", True, True),  # SFT + opt-in → chunked
        ("sft", False, False),  # SFT but didn't opt in → legacy
        ("policy_loss", True, False),  # Not SFT
        ("value_loss", True, False),  # Not SFT
    ],
)
def test_should_use_sft_chunked(loss_type, chunked_flag, expected):
    """The chunked-path gate predicate: SFT mode + explicit opt-in.

    Remaining incompatibilities (tied embeddings, combined-1f1b) are enforced
    as hard AssertionErrors in arguments.py.slime_validate_args, so by the time
    we reach this gate sft_chunked_logits=True is guaranteed compatible.
    Functional check on the helper extracted from ``forward_step``.
    """
    args = Namespace(
        loss_type=loss_type,
        sft_chunked_logits=chunked_flag,
    )
    assert should_use_sft_chunked(args) is expected
