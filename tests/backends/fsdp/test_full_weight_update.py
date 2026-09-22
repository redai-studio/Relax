# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Full-weight iterator + manifest: deterministic order, bucketing, hash
parity."""

from __future__ import annotations

import torch

from relax.backends.fsdp.weight_update import FullWeightChunkIterator, build_full_weight_manifest
from relax.models.generative import ordered_name_shape_hash


def _model():
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))


def test_manifest_matches_streamed_order_hash():
    named = [(n, p) for n, p in _model().named_parameters()]
    name_map = lambda s: "transformer." + s  # noqa: E731
    manifest = build_full_weight_manifest(
        named,
        model_family="test",
        task="t2i",
        policy_version=1,
        base_model_sha256="base",
        wire_dtype="bf16",
        bucket_size_bytes=1024,
        name_map=name_map,
    )
    it = FullWeightChunkIterator(named, wire_dtype="bf16", bucket_size_bytes=200, name_map=name_map)
    streamed = [(n, tuple(t.shape)) for bucket in it for n, t in bucket]
    assert ordered_name_shape_hash(streamed) == manifest.ordered_name_shape_hash
    assert manifest.tensor_count == len(named)


def test_iterator_casts_to_wire_dtype_and_buckets():
    named = [(n, p) for n, p in _model().named_parameters()]
    it = FullWeightChunkIterator(named, wire_dtype="bf16", bucket_size_bytes=200)
    buckets = list(it)
    assert len(buckets) >= 2  # small bucket cap forces a split
    for bucket in buckets:
        for _n, t in bucket:
            assert t.dtype == torch.bfloat16


def test_manifest_total_bytes_matches_stream():
    named = [(n, p) for n, p in _model().named_parameters()]
    manifest = build_full_weight_manifest(
        named,
        model_family="test",
        task="t2i",
        policy_version=1,
        base_model_sha256="base",
        wire_dtype="bf16",
        bucket_size_bytes=10**9,
    )
    it = FullWeightChunkIterator(named, wire_dtype="bf16", bucket_size_bytes=10**9)
    streamed_bytes = sum(t.numel() * t.element_size() for bucket in it for _n, t in bucket)
    assert streamed_bytes == manifest.total_bytes


def test_activation_checkpoint_wrapper_is_stripped_from_wire_names():
    """Names on the wire must match the engine's, not the train module graph.

    ``apply_activation_checkpointing`` inserts a ``CheckpointWrapper`` whose
    child is ``_checkpoint_wrapped_module``, so it shows up in
    ``named_parameters()`` but not in ``state_dict()``. SGLang's loader skips
    unknown names silently and still reports success, so leaving the segment in
    drops every transformer-block weight while the sync claims to have worked.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, torch.nn.Linear),
    )
    named = [(n, p) for n, p in model.named_parameters()]
    assert any("_checkpoint_wrapped_module" in n for n, _ in named), "fixture did not wrap anything"

    state_dict_names = set(model.state_dict().keys())
    it = FullWeightChunkIterator(named, wire_dtype="bf16", bucket_size_bytes=10**9)
    streamed = [n for bucket in it for n, _t in bucket]

    assert not any("_checkpoint_wrapped_module" in n for n in streamed)
    assert set(streamed) == state_dict_names

    # The manifest must agree with the stream, or the receiver's fail-fast check
    # would reject a correct sync.
    manifest = build_full_weight_manifest(
        named,
        model_family="test",
        task="t2i",
        policy_version=1,
        base_model_sha256="base",
        wire_dtype="bf16",
        bucket_size_bytes=10**9,
    )
    it2 = FullWeightChunkIterator(named, wire_dtype="bf16", bucket_size_bytes=10**9)
    assert ordered_name_shape_hash([(n, tuple(t.shape)) for b in it2 for n, t in b]) == (
        manifest.ordered_name_shape_hash
    )


def test_strip_transport_wrappers_leaves_ordinary_names_alone():
    from relax.backends.fsdp.weight_update import strip_transport_wrappers

    assert strip_transport_wrappers("blocks.0.attn.to_q.weight") == "blocks.0.attn.to_q.weight"
    assert (
        strip_transport_wrappers("blocks.0._checkpoint_wrapped_module.attn.to_q.weight") == "blocks.0.attn.to_q.weight"
    )
