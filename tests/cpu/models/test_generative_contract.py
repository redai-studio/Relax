# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Generative contract: artifact manifest, full-weight manifest, adapter
protocol."""

from __future__ import annotations

import pytest
import torch

from relax.models.generative import (
    ArtifactTrack,
    FullWeightManifest,
    GenerativeModelAdapter,
    artifact_manifest_dict,
    ordered_name_shape_hash,
    resolve_sde_indices,
)


def test_resolve_sde_indices_explicit_wins():
    assert resolve_sde_indices({"sde_indices": [5, 1, 3]}) == [1, 3, 5]


def test_resolve_sde_indices_from_num_and_fraction():
    # First half of a 12-step schedule ([0, 6)), 3 stride-spaced picks. Must equal
    # the validated explicit sde_indices in the t2i launch script so the YAML
    # profile and the shell launcher train the same steps.
    assert resolve_sde_indices(
        {"num_inference_steps": 12, "num_sde_steps": 3, "sde_timestep_fraction": [0.0, 0.5]}
    ) == [0, 2, 4]


def test_resolve_sde_indices_valid_transitions_and_sorted_unique():
    idx = resolve_sde_indices({"num_inference_steps": 20, "num_sde_steps": 4, "sde_timestep_fraction": [0.0, 0.5]})
    assert idx == sorted(set(idx))
    assert all(0 <= i < 20 for i in idx)  # each i has a valid i+1 slot in a T+1 trajectory


def test_resolve_sde_indices_empty_without_config():
    assert resolve_sde_indices({"num_inference_steps": 12}) == []
    assert resolve_sde_indices({}) == []


def test_ordered_name_shape_hash_is_order_sensitive():
    a = ordered_name_shape_hash([("x", (2, 3)), ("y", (4,))])
    b = ordered_name_shape_hash([("y", (4,)), ("x", (2, 3))])
    assert a != b
    assert a == ordered_name_shape_hash([("x", (2, 3)), ("y", (4,))])


def test_full_weight_manifest_sha_is_stable():
    h = ordered_name_shape_hash([("a", (2, 2))])
    m1 = FullWeightManifest(1, "qwen_image", "t2i", 3, "base", 1, 16, "bf16", 512, h)
    m2 = FullWeightManifest(1, "qwen_image", "t2i", 3, "base", 1, 16, "bf16", 512, h)
    assert m1.sha256() == m2.sha256()
    m3 = FullWeightManifest(1, "qwen_image", "t2i", 4, "base", 1, 16, "bf16", 512, h)
    assert m1.sha256() != m3.sha256()


def test_artifact_track_and_manifest_dict():
    image_sha = "a" * 64
    weight_sha = "b" * 64
    track = ArtifactTrack(
        track="image", uri="/a/0.png", mime="image/png", sha256=image_sha, meta={"height": 384, "width": 512}
    )
    d = track.to_dict()
    assert d["height"] == 384 and d["track"] == "image"
    manifest = artifact_manifest_dict(
        task="t2i",
        sample_index=42,
        group_index=5,
        policy_version=17,
        outputs=[track],
        trajectory_uri="/a/g.safetensors",
        sampling_fingerprint="fp",
        weight_manifest_sha256=weight_sha,
    )
    assert manifest["schema_version"] == 1
    assert manifest["outputs"][0]["mime"] == "image/png"
    assert manifest["outputs"][0]["sha256"] == image_sha
    assert manifest["weight_manifest_sha256"] == weight_sha
    assert manifest["trajectory_uri"].endswith(".safetensors")


def test_artifact_track_meta_cannot_shadow_reserved_keys():
    """A meta key that rewrites ``uri`` would send a scorer to another file."""
    track = ArtifactTrack(track="image", uri="/a/0.png", mime="image/png", sha256="a" * 64, meta={"uri": "/evil.png"})
    with pytest.raises(ValueError):
        track.to_dict()


class _MiniAdapter:
    family = "mini"
    supported_tasks = ("t2i",)

    def load_train_model(self, config):
        return torch.nn.Linear(2, 2)

    def build_rollout_request(self, sample, sampling, seed):
        return {"seed": seed}

    def validate_rollout_response(self, response):
        return None

    def pack_trajectory(self, response):
        return {}

    def replay_transition(self, model, batch, step_index):
        return torch.zeros(1)

    def artifact_tracks(self, response):
        return []

    def weight_name_map(self, name):
        return name


def test_adapter_runtime_checkable():
    assert isinstance(_MiniAdapter(), GenerativeModelAdapter)
