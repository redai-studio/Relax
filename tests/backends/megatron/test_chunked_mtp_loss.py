# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the chunked-MTP-loss patch
(megatron_patch/chunked_mtp_loss_patch.py).

The patch computes each MTP prediction depth's head + cross-entropy in sequence
chunks so the full ``[S, V/TP]`` per-depth logits never materialize (fixes the
tail-stage OOM). Because ``compute_language_model_loss`` reduces per token, chunking
the sequence and concatenating the per-token ``[b, chunk]`` losses must reconstruct
the full ``[b, s]`` loss BIT-FOR-BIT.

These tests drive the patched ``process_mtp_loss`` against the ORIGINAL upstream one
with mocked head (a plain Linear) + stub per-token CE, and assert the per-depth loss
tensor fed to ``MTPLossAutoScaler.apply`` is identical. Pure CPU, no distributed init.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from megatron.core.transformer import multi_token_prediction as _mtp_mod

    from relax.backends.megatron.megatron_patch import chunked_mtp_loss_patch as _patch
except Exception as exc:  # pragma: no cover
    pytest.skip(f"megatron/relax unavailable: {exc}", allow_module_level=True)


S, B, H, V, LAYERS = 16, 2, 8, 32, 2


class _FakeHead(nn.Module):
    """Stand-in for ColumnParallelLinear output_layer.

    forward returns (logits, None) matching the (logits, bias) contract. The
    patch calls ``type(output_layer).forward(output_layer, x, weight=,
    runtime_gather_output=)``, so this must accept those kwargs.
    """

    def __init__(self, hidden, vocab):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden, dtype=torch.float32))

    def forward(self, input_, weight=None, runtime_gather_output=None):
        w = weight if weight is not None else self.weight
        return input_ @ w.t(), None  # [.., vocab]


def _stub_clml(labels, logits):
    """Per-token CE: labels [b, s], logits [s, b, V] -> loss [b, s]."""
    s, b, v = logits.shape
    lg = logits.transpose(0, 1).reshape(b * s, v)
    lb = labels.reshape(b * s)
    return F.cross_entropy(lg, lb, reduction="none").reshape(b, s)


def _make_config(calculate_per_token_loss=False, fuse_linear=False):
    return types.SimpleNamespace(
        mtp_num_layers=LAYERS,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=calculate_per_token_loss,
        cross_entropy_loss_fusion=fuse_linear,
        cross_entropy_fusion_impl="linear" if fuse_linear else "native",
    )


def _run_and_capture(fn, *, head, config, monkeypatch, chunk_size=None, scale_logits_fn=None, is_training=False):
    """Run a process_mtp_loss impl, spying on MTPLossAutoScaler.apply to
    capture the per-depth loss tensors it would send to backward.

    Returns (returned_hidden, [losses]).
    """
    torch.manual_seed(0)
    hidden = torch.randn(S * (1 + LAYERS), B, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, S))
    loss_mask = (torch.rand(B, S) > 0.3).to(torch.float32)

    captured = []

    class _SpyAutoScaler:
        @staticmethod
        def apply(hs, loss):
            captured.append(loss.detach().clone())
            return hs

    # Patch MTPLossAutoScaler in BOTH the source module and the patch module's binding.
    monkeypatch.setattr(_mtp_mod, "MTPLossAutoScaler", _SpyAutoScaler)
    monkeypatch.setattr(_patch, "MTPLossAutoScaler", _SpyAutoScaler, raising=False)
    if chunk_size is not None:
        monkeypatch.setattr(_patch, "_resolve_chunk_size", lambda: chunk_size)
    # force gate ON for the patched fn (no get_args() in-process)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)

    ret = fn(
        hidden,
        labels,
        loss_mask,
        head,
        None,
        None,
        is_training,
        _stub_clml,
        config,
        cp_group=None,
        packed_seq_params=None,
        scale_logits_fn=scale_logits_fn,
    )
    return ret, captured


@pytest.mark.parametrize("chunk_size", [S, 2, 3, 7])
def test_chunked_mtp_matches_original(chunk_size, monkeypatch):
    """Chunked head+CE == full head+CE, bit-identical (fp32), for every chunk
    size."""
    head = _FakeHead(H, V)
    cfg = _make_config()

    # original (unchunked): the captured upstream impl. Force chunk irrelevant.
    _, ref = _run_and_capture(_patch._ORIG_PROCESS_MTP_LOSS, head=head, config=cfg, monkeypatch=monkeypatch)
    # patched (chunked) at this chunk size.
    _, got = _run_and_capture(
        _patch.process_mtp_loss, head=head, config=cfg, monkeypatch=monkeypatch, chunk_size=chunk_size
    )

    assert len(got) == len(ref) == LAYERS
    for d, (g, r) in enumerate(zip(got, ref)):
        assert torch.equal(g, r), f"depth {d} chunk={chunk_size}: max|Δ|={(g - r).abs().max()}"


def test_scale_logits_fn_per_chunk(monkeypatch):
    """A per-token scalar scale_logits_fn is chunk-invariant."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    sfn = lambda x: 0.5 * x  # noqa: E731
    _, ref = _run_and_capture(
        _patch._ORIG_PROCESS_MTP_LOSS, head=head, config=cfg, monkeypatch=monkeypatch, scale_logits_fn=sfn
    )
    _, got = _run_and_capture(
        _patch.process_mtp_loss, head=head, config=cfg, monkeypatch=monkeypatch, chunk_size=3, scale_logits_fn=sfn
    )
    for g, r in zip(got, ref):
        assert torch.equal(g, r)


def test_calculate_per_token_loss_branch(monkeypatch):
    """The per-token-loss normalization branch is preserved under chunking."""
    head = _FakeHead(H, V)
    cfg = _make_config(calculate_per_token_loss=True)
    _, ref = _run_and_capture(_patch._ORIG_PROCESS_MTP_LOSS, head=head, config=cfg, monkeypatch=monkeypatch)
    _, got = _run_and_capture(_patch.process_mtp_loss, head=head, config=cfg, monkeypatch=monkeypatch, chunk_size=3)
    for g, r in zip(got, ref):
        assert torch.equal(g, r)


def test_is_training_logging_matches(monkeypatch):
    """is_training path: both the per-depth GRADIENT tensor
    (MTPLossAutoScaler.apply) AND the LOGGED per-depth loss
    (save_loss_to_tracker's first arg) must match upstream.

    Spies on save_loss_to_tracker so a divergence in mtp_loss_for_log (e.g.
    dropping the mtp_loss_scale factor from the log-only value) is caught — the
    gradient path alone would still match and hide it (regression: chunked
    patch logged the UNSCALED loss).
    """
    head = _FakeHead(H, V)
    cfg = _make_config()

    logged = {"ref": [], "got": []}

    def _spy_tracker(dst):
        return staticmethod(lambda loss, *a, **k: dst.append(loss.detach().clone()))

    monkeypatch.setattr(_mtp_mod.parallel_state, "get_data_parallel_group", lambda **k: None)
    monkeypatch.setattr(_patch.parallel_state, "get_data_parallel_group", lambda **k: None, raising=False)

    monkeypatch.setattr(_mtp_mod.MTPLossLoggingHelper, "save_loss_to_tracker", _spy_tracker(logged["ref"]))
    monkeypatch.setattr(
        _patch.MTPLossLoggingHelper, "save_loss_to_tracker", _spy_tracker(logged["ref"]), raising=False
    )
    _, ref = _run_and_capture(
        _patch._ORIG_PROCESS_MTP_LOSS, head=head, config=cfg, monkeypatch=monkeypatch, is_training=True
    )
    monkeypatch.setattr(_mtp_mod.MTPLossLoggingHelper, "save_loss_to_tracker", _spy_tracker(logged["got"]))
    monkeypatch.setattr(
        _patch.MTPLossLoggingHelper, "save_loss_to_tracker", _spy_tracker(logged["got"]), raising=False
    )
    _, got = _run_and_capture(
        _patch.process_mtp_loss, head=head, config=cfg, monkeypatch=monkeypatch, chunk_size=3, is_training=True
    )
    for g, r in zip(got, ref):
        assert torch.equal(g, r), "gradient (MTPLossAutoScaler.apply) tensor mismatch"
    assert len(logged["got"]) == len(logged["ref"]) == LAYERS
    for d, (g, r) in enumerate(zip(logged["got"], logged["ref"])):
        assert torch.equal(g, r), f"logged mtp_loss_for_log mismatch at depth {d}: {g} vs {r}"


def test_token_aware_tracker_api_receives_scaled_sum_and_count(monkeypatch):
    """New Megatron tracker APIs receive the raw scaled sum and token count."""
    seen = {}

    def tracker(loss_sum, num_tokens, layer_number, num_layers, **kwargs):
        seen.update(
            loss_sum=loss_sum,
            num_tokens=num_tokens,
            layer_number=layer_number,
            num_layers=num_layers,
            kwargs=kwargs,
        )

    monkeypatch.setattr(
        _patch,
        "_MTP_TRACKER_PARAMETERS",
        {"loss_sum", "num_tokens", "layer_number", "num_layers", "correct", "total"},
    )
    monkeypatch.setattr(_patch.MTPLossLoggingHelper, "save_loss_to_tracker", staticmethod(tracker))
    monkeypatch.setattr(_patch.parallel_state, "get_data_parallel_group", lambda **_kwargs: None)
    config = _make_config()
    loss = torch.tensor([[1.0, 2.0]])
    num_tokens = torch.tensor(2.0)
    correct = torch.tensor(1.0)
    total = torch.tensor(2.0)

    _patch._save_mtp_loss_to_tracker(loss, num_tokens, 0, config, correct, total)

    expected_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
    assert torch.equal(seen["loss_sum"], expected_scale * loss.sum())
    assert seen["num_tokens"] is num_tokens
    assert seen["layer_number"] == 0
    assert seen["num_layers"] == config.mtp_num_layers
    assert seen["kwargs"]["correct"] is correct
    assert seen["kwargs"]["total"] is total


def test_fuse_linear_delegates(monkeypatch):
    """fuse_linear_cross_entropy=True delegates to the original (never
    chunks)."""
    head = _FakeHead(H, V)
    cfg = _make_config(fuse_linear=True)
    called = {"orig": 0, "args": None}

    # Pure spy (does NOT call the real upstream body — the fused Blackwell path can't
    # run with a plain mock head). We only assert the patch DELEGATED without chunking.
    def spy_orig(*a, **k):
        called["orig"] += 1
        called["args"] = (a, k)
        return a[0]  # return hidden_states unchanged

    monkeypatch.setattr(_patch, "_ORIG_PROCESS_MTP_LOSS", spy_orig)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    labels = torch.randint(0, V, (B, S))
    loss_mask = torch.ones(B, S)
    _patch.process_mtp_loss(hidden, labels, loss_mask, head, None, None, False, _stub_clml, cfg)
    assert called["orig"] == 1, "fuse_linear=True must delegate to the original, not chunk"


def test_gate_off_delegates(monkeypatch):
    """When the chunked-MTP gate is off, the patched fn delegates to the
    original."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    called = {"orig": 0}
    orig = _patch._ORIG_PROCESS_MTP_LOSS

    def spy_orig(*a, **k):
        called["orig"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_patch, "_ORIG_PROCESS_MTP_LOSS", spy_orig)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: False)
    monkeypatch.setattr(_mtp_mod, "MTPLossAutoScaler", types.SimpleNamespace(apply=staticmethod(lambda hs, loss: hs)))
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    labels = torch.randint(0, V, (B, S))
    loss_mask = torch.ones(B, S)
    _patch.process_mtp_loss(hidden, labels, loss_mask, head, None, None, False, _stub_clml, cfg)
    assert called["orig"] == 1


def test_new_megatron_kwargs_are_accepted_and_version_filtered(monkeypatch):
    """Newer Megatron callers pass tp_group/input_ids; old originals must still
    work."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    called = {}

    def old_style_orig(*args, cp_group=None, packed_seq_params=None, scale_logits_fn=None):
        called.update(
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            scale_logits_fn=scale_logits_fn,
        )
        return args[0]

    monkeypatch.setattr(_patch, "_ORIG_PROCESS_MTP_LOSS", old_style_orig)
    monkeypatch.setattr(
        _patch,
        "_ORIG_PROCESS_MTP_LOSS_PARAMETERS",
        {"cp_group", "packed_seq_params", "scale_logits_fn"},
    )
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: False)
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    labels = torch.randint(0, V, (B, S))
    result = _patch.process_mtp_loss(
        hidden,
        labels,
        torch.ones(B, S),
        head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
        tp_group=object(),
        input_ids=torch.ones(B, S, dtype=torch.long),
    )
    assert result is hidden
    assert called == {"cp_group": None, "packed_seq_params": None, "scale_logits_fn": None}


def test_sequence_parallel_gathers_hidden_once(monkeypatch):
    """With sequence_parallel=True on the head, the patch must gather the SP-
    scattered hidden to the full sequence ONCE (so chunk boundaries align with
    the full-sequence labels), run the head with SP disabled per chunk, then
    restore the flag — mirroring relax model.py _bypass_output_layer.

    Regression for the on-cluster IndexError ('shapes [2048], [1024]'): before
    the fix the patch chunked the SP-scattered input and let the
    ColumnParallelLinear all-gather each chunk (×TP), so logits seq (chunk×TP)
    no longer matched the label slice (chunk). Here the head's own forward
    would NOT gather (it's a plain matmul), so we assert the behavioral
    contract: (1) gather_from_sequence_parallel_region is invoked exactly once
    per depth, (2) sequence_parallel is False during the head calls, (3) it is
    restored to True afterwards, and (4) the per-depth loss equals the
    unchunked reference.
    """
    torch.manual_seed(0)
    head = _FakeHead(H, V)
    head.sequence_parallel = True
    head.tp_group = object()  # truthy sentinel so tp_group resolution skips parallel_state

    cfg = _make_config()
    full_hidden = torch.randn(S * (1 + LAYERS), B, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, S))
    loss_mask = (torch.rand(B, S) > 0.3).to(torch.float32)

    # Reference: unchunked upstream on the FULL hidden with SP off (no gather, matmul only).
    ref_head = _FakeHead(H, V)
    ref_head.load_state_dict(head.state_dict())
    captured_ref = []
    monkeypatch.setattr(
        _mtp_mod,
        "MTPLossAutoScaler",
        types.SimpleNamespace(apply=staticmethod(lambda hs, loss: captured_ref.append(loss.detach().clone()) or hs)),
    )
    _patch._ORIG_PROCESS_MTP_LOSS(
        full_hidden,
        labels,
        loss_mask,
        ref_head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
    )

    # Stub the SP gather as identity (the test feeds full-seq hidden directly) and count calls
    # + observe the sequence_parallel flag at head-call time.
    gather_calls = {"n": 0}
    sp_during_head = []

    def _fake_gather(x, tensor_parallel_output_grad=False, group=None):
        gather_calls["n"] += 1
        return x

    monkeypatch.setattr(
        "megatron.core.tensor_parallel.mappings.gather_from_sequence_parallel_region", _fake_gather, raising=True
    )

    orig_fake_forward = _FakeHead.forward

    def _spy_forward(self, input_, weight=None, runtime_gather_output=None):
        sp_during_head.append(self.sequence_parallel)
        return orig_fake_forward(self, input_, weight=weight, runtime_gather_output=runtime_gather_output)

    monkeypatch.setattr(_FakeHead, "forward", _spy_forward)

    captured_got = []
    monkeypatch.setattr(
        _mtp_mod,
        "MTPLossAutoScaler",
        types.SimpleNamespace(apply=staticmethod(lambda hs, loss: captured_got.append(loss.detach().clone()) or hs)),
    )
    monkeypatch.setattr(_patch, "MTPLossAutoScaler", _mtp_mod.MTPLossAutoScaler, raising=False)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(_patch, "_resolve_chunk_size", lambda: 3)
    _patch.process_mtp_loss(
        full_hidden,
        labels,
        loss_mask,
        head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
    )

    assert gather_calls["n"] == LAYERS, f"expected one SP gather per depth, got {gather_calls['n']}"
    assert sp_during_head and all(sp is False for sp in sp_during_head), "head must run with sequence_parallel=False"
    assert head.sequence_parallel is True, "sequence_parallel must be restored to True after chunking"
    assert len(captured_got) == len(captured_ref) == LAYERS
    for d, (g, r) in enumerate(zip(captured_got, captured_ref)):
        assert torch.equal(g, r), f"SP depth {d}: chunked != unchunked (max|Δ|={(g - r).abs().max()})"


def test_patch_replaces_both_bindings():
    """The patch swapped process_mtp_loss in both the source and gpt_model
    modules."""
    import megatron.core.models.gpt.gpt_model as gpt_mod

    assert _mtp_mod.process_mtp_loss is _patch.process_mtp_loss
    assert gpt_mod.process_mtp_loss is _patch.process_mtp_loss
