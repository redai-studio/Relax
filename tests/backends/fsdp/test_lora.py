# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""LoRA on the FSDP generative backend: fold math, sync plan, key maps,
injection.

The tests split along an availability line. Everything that operates on the
*shape* of a PEFT-injected model (the sync plan, the fold, the key maps, the
manifest invariant) runs against a hand-built module that mimics PEFT's layout,
so it needs no ``peft`` and no GPU. Only the tests that exercise
``inject_lora_adapter`` itself require the real library.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from relax.backends.fsdp.lora import (
    build_lora_sync_plan,
    contract_mismatches,
    engine_adapter_state_dict,
    fold_lora_delta,
    is_lora_injected,
    lora_metadata_dict,
    lora_scaling,
    named_lora_params,
    peft_adapter_state_dict,
    strip_base_layer,
)
from relax.backends.fsdp.runtime import resolve_block_classes, upcast_trainable_params
from relax.backends.fsdp.weight_update import FullWeightChunkIterator, build_full_weight_manifest
from relax.models.generative import ordered_name_shape_hash


DIM, RANK, ALPHA = 8, 4, 8
SCALING = ALPHA / RANK


def _peft_available() -> bool:
    """``peft`` importable AND usable.

    A plain ``importorskip`` is not enough: peft 0.20's
    ``is_torchao_available`` RAISES (rather than returning False) on torchao <
    0.16, and peft pulls in ``transformers``, which itself imports torchao.
    Both surface as ImportError from an environment mismatch, not as a Relax
    bug.
    """
    try:
        from peft import LoraConfig, inject_adapter_in_model
        from peft.tuners.lora import LoraLayer

        class _Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(1, 1, bias=False)

        probe = _Probe()
        config = LoraConfig(
            r=1,
            lora_alpha=1,
            lora_dropout=0.0,
            target_modules=["proj"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        inject_adapter_in_model(config, probe, adapter_name="default")
        if not any(isinstance(module, LoraLayer) for module in probe.modules()):
            return False
    except Exception:
        return False
    return True


requires_peft = pytest.mark.skipif(not _peft_available(), reason="peft (or its torchao dependency) unavailable")


# ---------------------------------------------------------------------------
# fixtures: a plain model and its PEFT-shaped twin, holding identical weights
# ---------------------------------------------------------------------------


class _Wrapped(nn.Module):
    """The layout ``peft.inject_adapter_in_model`` leaves behind on a
    Linear."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.ModuleDict({"default": nn.Linear(DIM, RANK, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(RANK, DIM, bias=False)})
        nn.init.zeros_(self.lora_B["default"].weight)  # peft zero-inits B


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.ff = nn.Linear(DIM, DIM, bias=False)


class _LoradBlock(nn.Module):
    def __init__(self, src: _Block):
        super().__init__()
        self.to_q = _Wrapped(src.to_q)
        self.ff = src.ff


class _PeftConfig:
    def __init__(self, r: int, lora_alpha: int):
        self.r = r
        self.lora_alpha = lora_alpha


def _pair():
    """A plain model and a LoRA'd model sharing the same base weights."""
    torch.manual_seed(0)
    plain = _Block()
    lorad = _LoradBlock(plain)
    lorad.peft_config = {"default": _PeftConfig(RANK, ALPHA)}
    return plain, lorad


def _manifest(named):
    return build_full_weight_manifest(
        named,
        model_family="test",
        task="t2i",
        policy_version=1,
        base_model_sha256="base",
        wire_dtype="bf16",
        bucket_size_bytes=1 << 30,
    )


def _stream(named, **kw):
    it = FullWeightChunkIterator(named, wire_dtype="bf16", bucket_size_bytes=1 << 30, **kw)
    return [item for bucket in it for item in bucket]


# ---------------------------------------------------------------------------
# the four load-bearing invariants
# ---------------------------------------------------------------------------


def test_lora_merged_manifest_is_identical_to_full_ft():
    """The merged sync must be indistinguishable from full FT to the engine.

    Same tensor count, bytes and ordered name-shape hash — that is what lets
    the engine-side name routing and the manifest contract stay untouched.
    """
    plain, lorad = _pair()
    plan = build_lora_sync_plan(lorad)

    full, merged = _manifest(list(plain.named_parameters())), _manifest(plan.named_base_params)
    assert merged.tensor_count == full.tensor_count
    assert merged.total_bytes == full.total_bytes
    assert merged.ordered_name_shape_hash == full.ordered_name_shape_hash


def test_lora_merged_stream_satisfies_its_own_manifest():
    """Replicates the predicate that raises WeightSyncError in the actor."""
    _plain, lorad = _pair()
    plan = build_lora_sync_plan(lorad)
    manifest = _manifest(plan.named_base_params)

    streamed = _stream(plan.named_base_params, tensor_source=plan.materialize)
    assert len(streamed) == manifest.tensor_count
    assert sum(int(t.numel()) * t.element_size() for _, t in streamed) == manifest.total_bytes
    assert ordered_name_shape_hash([(n, tuple(t.shape)) for n, t in streamed]) == manifest.ordered_name_shape_hash


def test_zero_init_lora_streams_the_base_verbatim():
    """With B == 0 the merged stream must equal the full-FT stream, tensor for
    tensor and in the same order.

    One assertion covering name mapping, A/B pairing and ordering at once: any
    drift in the ``.base_layer`` strip or a mis-paired adapter shows up here.
    """
    plain, lorad = _pair()
    plan = build_lora_sync_plan(lorad)

    full = _stream(list(plain.named_parameters()))
    merged = _stream(plan.named_base_params, tensor_source=plan.materialize)
    assert [n for n, _ in full] == [n for n, _ in merged]
    for (name, a), (_, b) in zip(full, merged):
        assert torch.equal(a, b), name


def test_fold_is_the_correctly_rounded_result():
    """The fold must equal the exact result rounded once, not an approximation.

    ``B @ A`` contracts over the rank dimension, so a bf16 matmul accumulates
    that sum at bf16; rounding the delta before adding rounds twice. Either way
    thousands of elements land on the wrong bf16 value. Accumulating in fp32 and
    rounding once reproduces the reference exactly.
    """
    torch.manual_seed(0)
    size, rank = 256, 64
    base = (torch.randn(size, size) * 0.02).to(torch.bfloat16)
    lora_a = (torch.randn(rank, size) * 0.02).to(torch.bfloat16)
    lora_b = (torch.randn(size, rank) * 0.02).to(torch.bfloat16)

    reference = (base.float() + (lora_b.float() @ lora_a.float()) * SCALING).to(torch.bfloat16)
    good = fold_lora_delta(base, lora_a, lora_b, SCALING)
    bf16_matmul = (base + (lora_b @ lora_a) * SCALING).to(torch.bfloat16)
    rounded_delta = (base + ((lora_b.float() @ lora_a.float()) * SCALING).to(torch.bfloat16)).to(torch.bfloat16)

    assert good.dtype == base.dtype and good.shape == base.shape
    assert torch.equal(good, reference)
    assert (bf16_matmul != reference).sum() > 0
    assert (rounded_delta != reference).sum() > 0


def test_wire_cast_happens_after_the_fold():
    """The iterator owns the single rounding; the tensor_source must not pre-
    cast.

    Pre-casting the fold inputs to the wire dtype would round the LoRA update
    away before it is ever added.
    """
    _plain, lorad = _pair()
    with torch.no_grad():
        lorad.to_q.lora_B["default"].weight.normal_(0, 0.05)
        lorad.to_q.lora_A["default"].weight.normal_(0, 0.05)
    plan = build_lora_sync_plan(lorad)

    streamed = dict(_stream(plan.named_base_params, tensor_source=plan.materialize))
    expected = fold_lora_delta(
        lorad.to_q.base_layer.weight,
        lorad.to_q.lora_A["default"].weight,
        lorad.to_q.lora_B["default"].weight,
        SCALING,
    ).to(torch.bfloat16)
    assert torch.equal(streamed["to_q.weight"], expected)


# ---------------------------------------------------------------------------
# plan construction, naming, metadata
# ---------------------------------------------------------------------------


def test_full_ft_stream_order_is_pinned():
    """Golden hash: a future reordering must break a test, not a live sync."""
    named = list(_Block().named_parameters())
    assert [n for n, _ in named] == ["to_q.weight", "ff.weight"]
    assert (
        ordered_name_shape_hash(sorted((n, tuple(p.shape)) for n, p in named))
        == _manifest(named).ordered_name_shape_hash
    )


def test_strip_base_layer_is_segment_anchored():
    assert strip_base_layer("blk.0.attn.to_q.base_layer.weight") == "blk.0.attn.to_q.weight"
    assert strip_base_layer("blk.0.attn.to_q.weight") == "blk.0.attn.to_q.weight"
    # A module merely CONTAINING the substring must not be mangled.
    assert strip_base_layer("blk.0.base_layer_norm.weight") == "blk.0.base_layer_norm.weight"


def test_sync_plan_scaling_comes_from_the_injected_config():
    _plain, lorad = _pair()
    assert lora_scaling(lorad) == pytest.approx(SCALING)
    assert build_lora_sync_plan(lorad).scaling == pytest.approx(SCALING)
    # A resumed run must not be able to disagree with what is installed.
    lorad.peft_config["default"] = _PeftConfig(RANK, ALPHA * 2)
    assert build_lora_sync_plan(lorad).scaling == pytest.approx(SCALING * 2)


def test_sync_plan_rejects_a_half_injected_module():
    _plain, lorad = _pair()
    del lorad.to_q.lora_B
    with pytest.raises(ValueError, match="missing its lora_B"):
        build_lora_sync_plan(lorad, scaling=SCALING)


def test_is_lora_injected():
    plain, lorad = _pair()
    assert is_lora_injected(lorad)
    assert not is_lora_injected(plain)


def test_engine_state_dict_uses_bare_names_and_emits_alpha():
    """SGLang strips neither ``transformer.`` nor ``base_model.model.`` and
    only warns on an unmatched name, so a prefix here disables LoRA silently.

    It also reads alpha only from a per-layer tensor.
    """
    _plain, lorad = _pair()
    out = engine_adapter_state_dict(named_lora_params(lorad), alpha=ALPHA)
    assert sorted(out) == ["to_q.alpha", "to_q.lora_A.weight", "to_q.lora_B.weight"]
    assert float(out["to_q.alpha"]) == float(ALPHA)
    # Without an explicit alpha SGLang would infer alpha == rank, i.e. scale 1.0.
    assert "to_q.alpha" not in engine_adapter_state_dict(named_lora_params(lorad))


def test_engine_state_dict_strips_activation_checkpoint_wrapper():
    """Activation checkpointing inserts a train-only module segment that SGLang
    does not have in its DiT layer names."""
    a = torch.randn(RANK, DIM)
    b = torch.randn(DIM, RANK)
    named = [
        ("transformer_blocks.0._checkpoint_wrapped_module.attn.to_q.lora_A.default.weight", a),
        ("transformer_blocks.0._checkpoint_wrapped_module.attn.to_q.lora_B.default.weight", b),
    ]
    out = engine_adapter_state_dict(named, alpha=ALPHA)
    assert sorted(out) == [
        "transformer_blocks.0.attn.to_q.alpha",
        "transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer_blocks.0.attn.to_q.lora_B.weight",
    ]
    assert out["transformer_blocks.0.attn.to_q.lora_A.weight"] is a
    assert out["transformer_blocks.0.attn.to_q.lora_B.weight"] is b


def test_engine_state_dict_rejects_transport_key_collision():
    named = [
        ("block.to_q.lora_A.default.weight", torch.randn(RANK, DIM)),
        ("block._checkpoint_wrapped_module.to_q.lora_A.default.weight", torch.randn(RANK, DIM)),
    ]
    with pytest.raises(ValueError, match="same transport key"):
        engine_adapter_state_dict(named, alpha=ALPHA)


def test_peft_state_dict_uses_the_hf_envelope():
    _plain, lorad = _pair()
    out = peft_adapter_state_dict(named_lora_params(lorad))
    assert sorted(out) == [
        "base_model.model.to_q.lora_A.weight",
        "base_model.model.to_q.lora_B.weight",
    ]


def test_lora_metadata_records_alpha_and_derived_scaling():
    meta = lora_metadata_dict(rank=RANK, alpha=ALPHA, dropout=0.0, target_modules=["attn.to_q"])
    assert meta["scaling"] == pytest.approx(SCALING)
    assert meta["task_type"] == "FEATURE_EXTRACTION"


def test_contract_mismatches_catches_the_silent_ones():
    base = lora_metadata_dict(rank=RANK, alpha=ALPHA, dropout=0.0, target_modules=["attn.to_q"])
    assert contract_mismatches(base, dict(base)) == []
    # A wrong alpha raises nothing at load time; it just rescales the policy.
    assert any("alpha" in m for m in contract_mismatches(base, {**base, "alpha": ALPHA * 2}))
    assert any("rank" in m for m in contract_mismatches(base, {**base, "rank": RANK * 2}))
    assert any("target_modules" in m for m in contract_mismatches(base, {**base, "target_modules": ["attn.to_k"]}))


# ---------------------------------------------------------------------------
# fp32 optimizer masters
# ---------------------------------------------------------------------------


def test_upcast_promotes_only_trainable_float_params():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)).to(torch.bfloat16)
    model[0].weight.requires_grad_(False)
    model[0].bias.requires_grad_(False)
    model.register_buffer("counter", torch.zeros(2, dtype=torch.long))

    assert upcast_trainable_params(model, torch.float32) == 2  # weight + bias of layer 1
    assert model[0].weight.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float32
    assert model.counter.dtype == torch.long
    assert upcast_trainable_params(model, torch.float32) == 0  # idempotent


def test_bf16_master_loses_the_update_that_fp32_keeps():
    """Why --fsdp-master-dtype fp32 exists.

    A ~1e-4 relative step vanishes into a bf16 master's mantissa; the same step
    on an fp32 master lands. The failure mode is silent: grad norm and loss
    stay healthy while the policy never moves.
    """
    step = 1e-4
    bf16_param = torch.ones(4, dtype=torch.bfloat16)
    fp32_param = torch.ones(4, dtype=torch.float32)
    assert torch.equal(bf16_param + torch.full_like(bf16_param, step), bf16_param)
    assert not torch.equal(fp32_param + torch.full_like(fp32_param, step), fp32_param)


# ---------------------------------------------------------------------------
# real injection (needs peft)
# ---------------------------------------------------------------------------


class _FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(DIM, DIM, bias=False), nn.Dropout(0.0)])

    def forward(self, x):
        return self.to_out[0](self.to_q(x))


class _FakeDiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeAttn()

    def forward(self, x):
        return self.attn(x)


class _FakeDiT(nn.Module):
    _no_split_modules = ["_FakeDiTBlock"]

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_FakeDiTBlock() for _ in range(2)])
        self.proj_out = nn.Linear(DIM, DIM, bias=False)

    def forward(self, hidden_states=None, **_kw):
        x = hidden_states
        for block in self.transformer_blocks:
            x = block(x)
        return (self.proj_out(x),)


@requires_peft
def test_injection_preserves_the_module_class_and_forward():
    """``inject_adapter_in_model``, not ``get_peft_model``.

    A ``PeftModel`` wrapper would hide ``_no_split_modules`` from the FSDP
    shard policy and change the keyword forward signature the model adapters
    call.
    """
    from relax.backends.fsdp.lora import inject_lora_adapter

    torch.manual_seed(0)
    model = _FakeDiT()
    x = torch.randn(2, 3, DIM)
    before = model(hidden_states=x)[0].clone()

    count = inject_lora_adapter(
        model, rank=RANK, alpha=ALPHA, target_modules=["attn.to_q", "attn.to_out.0"], dropout=0.0
    )
    assert count == 4  # 2 blocks x 2 targets
    assert type(model.transformer_blocks[0]).__name__ == "_FakeDiTBlock"
    assert [c.__name__ for c in resolve_block_classes(model)] == ["_FakeDiTBlock"]
    # B is zero-initialized, so the adapter is a no-op at step 0.
    assert torch.equal(before, model(hidden_states=x)[0])


@requires_peft
def test_injection_freezes_the_base():
    from relax.backends.fsdp.lora import inject_lora_adapter

    model = _FakeDiT()
    inject_lora_adapter(model, rank=RANK, alpha=ALPHA, target_modules=["attn.to_q"], dropout=0.0)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n for n in trainable)


@requires_peft
def test_injection_rejects_a_target_that_matches_nothing():
    """PEFT is happy to match a subset, which silently halves the capacity."""
    from relax.backends.fsdp.lora import inject_lora_adapter

    with pytest.raises(ValueError, match="matched nothing"):
        inject_lora_adapter(
            _FakeDiT(),
            rank=RANK,
            alpha=ALPHA,
            # `attn.to_out` is the ModuleList, not the Linear inside it.
            target_modules=["attn.to_q", "attn.to_out"],
            dropout=0.0,
        )


@requires_peft
def test_real_injection_round_trips_through_the_sync_plan():
    """End-to-end on a real PEFT model: names, ordering and manifest all
    hold."""
    from relax.backends.fsdp.lora import inject_lora_adapter

    torch.manual_seed(0)
    plain = _FakeDiT()
    lorad = _FakeDiT()
    lorad.load_state_dict(plain.state_dict())
    inject_lora_adapter(lorad, rank=RANK, alpha=ALPHA, target_modules=["attn.to_q", "attn.to_out.0"], dropout=0.0)

    plan = build_lora_sync_plan(lorad)
    assert sorted(n for n, _ in plan.named_base_params) == sorted(n for n, _ in plain.named_parameters())
    assert len(plan.folds) == 4
    assert _manifest(plan.named_base_params).ordered_name_shape_hash == (
        _manifest(list(plain.named_parameters())).ordered_name_shape_hash
    )
    full = _stream(list(plain.named_parameters()))
    merged = _stream(plan.named_base_params, tensor_source=plan.materialize)
    for (name, a), (_, b) in zip(full, merged):
        assert torch.equal(a, b), name
