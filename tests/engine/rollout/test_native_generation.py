# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native generation data contracts: sidecar, converter row, hydrate round-
trip."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from relax.engine.rollout import native_generation as ng
from relax.utils.types import Sample


def test_group_sidecar_write_and_hydrate_round_trip(tmp_path):
    path = ng.group_sidecar_path(str(tmp_path), "t2i", rollout_id=3, group_index=5)
    common = {"sigmas": torch.linspace(1, 0, 4), "sde_indices": torch.tensor([0, 1])}
    conditions = {"cond_prompt_embeds": torch.randn(2, 8)}
    tracks = {"image_x_t": torch.randn(2, 2, 3, 4), "image_x_next": torch.randn(2, 2, 3, 4)}
    ng.write_group_sidecar(path, common, conditions, tracks)

    batch = ng._load_group_batch(path, adapter=None)
    assert "sigmas" in batch and "image_x_t" in batch and "cond_prompt_embeds" in batch
    assert batch["sde_indices"].tolist() == [0, 1]


def test_load_group_batch_explains_a_non_shared_artifact_root(tmp_path):
    """A missing sidecar is almost always a non-shared --artifact-root.

    Every FSDP rank reads the sidecars the single rollout driver wrote, so a
    node-local path works on the driver's node and raises a bare
    FileNotFoundError on all the others. Say which knob is wrong instead.
    """
    import pytest

    path = ng.group_sidecar_path(str(tmp_path), "t2i", rollout_id=0, group_index=0)
    with pytest.raises(RuntimeError, match="artifact-root"):
        ng._load_group_batch(path, adapter=None)


def test_save_candidate_image_writes_png_and_returns_uri(tmp_path):
    from PIL import Image

    # CHW float in [0,1] (diffusers convention) → PNG on disk with a usable uri.
    img = torch.rand(3, 16, 24)
    uri, h, w, sha256 = ng.save_candidate_image(img, str(tmp_path), "t2i", rollout_id=0, group_index=1, slot=2)
    assert uri.endswith(".png") and os.path.exists(uri)
    assert (h, w) == (16, 24)
    assert len(sha256) == 64 and int(sha256, 16) >= 0
    assert Image.open(uri).size == (24, 16)  # PIL size is (W, H)
    _, _, _, changed_sha256 = ng.save_candidate_image(
        torch.zeros_like(img), str(tmp_path), "t2i", rollout_id=0, group_index=1, slot=2
    )
    assert changed_sha256 != sha256
    # No image → empty uri, no crash (reward simply has no image track).
    assert ng.save_candidate_image(None, str(tmp_path), "t2i", 0, 1, 2) == ("", None, None, "")


def test_save_candidate_image_keeps_rgb_for_engine_cthw_layout(tmp_path):
    from PIL import Image

    # SGLang's OutputBatch.output is [B, C, T, H, W] and rollout_api indexes the
    # batch, so what reaches us is [C, T=1, H, W]. Reading that as [B, C, H, W]
    # keeps only channel 0 and writes a grayscale PNG, which silently costs the
    # reward model all colour information.
    img = torch.zeros(3, 1, 8, 12)
    img[0], img[1], img[2] = 1.0, 0.5, 0.0  # distinct per-channel values
    uri, h, w, _sha256 = ng.save_candidate_image(img, str(tmp_path), "t2i", rollout_id=0, group_index=0, slot=0)
    opened = Image.open(uri)
    assert opened.mode == "RGB"
    assert (h, w) == (8, 12)
    assert opened.getpixel((0, 0)) == (255, 128, 0)  # round(), not truncate


def test_save_candidate_image_still_reads_bchw_as_batch(tmp_path):
    from PIL import Image

    # A genuine [B, C, H, W] batch must keep its batch reading (B=1, C=3).
    uri, h, w, _sha256 = ng.save_candidate_image(
        torch.rand(1, 3, 8, 12), str(tmp_path), "t2i", rollout_id=0, group_index=1, slot=0
    )
    assert Image.open(uri).mode == "RGB"
    assert (h, w) == (8, 12)


def test_cache_candidate_image_round_trips_without_a_file():
    uri, h, w, sha256 = ng.cache_candidate_image(torch.rand(3, 8, 12), "t2i", 3, 4, 5)
    assert uri == "memory://t2i/rollout_0000003/group_00000004_sample_0005"
    assert (h, w) == (8, 12)
    assert len(sha256) == 64
    assert ng.get_cached_candidate_image(uri).shape == (8, 12, 3)
    assert ng.get_cached_candidate_image(uri, remove=True).shape == (8, 12, 3)
    assert ng.get_cached_candidate_image(uri) is None


def test_prune_stale_artifacts_releases_cached_candidate_images():
    old_uri, *_ = ng.cache_candidate_image(torch.rand(3, 8, 12), "t2i", 1, 0, 0)
    recent_uri, *_ = ng.cache_candidate_image(torch.rand(3, 8, 12), "t2i", 4, 0, 0)

    ng.prune_stale_artifacts(
        SimpleNamespace(artifact_root=None, generation_task="t2i", artifact_retention_rollouts=2), rollout_id=5
    )

    assert ng.get_cached_candidate_image(old_uri) is None
    assert ng.get_cached_candidate_image(recent_uri, remove=True) is not None


def test_build_candidate_manifest_rejects_missing_weight_manifest_sha():
    import pytest

    with pytest.raises(ValueError, match="weight_manifest_sha256"):
        ng.build_candidate_manifest(
            task="t2i",
            sample_index=0,
            group_index=0,
            policy_version=1,
            artifact_tracks=[],
            trajectory_uri="/tmp/group.safetensors",
            sampling_fingerprint="fp",
            weight_manifest_sha256="",
        )


def _sample(group_index, slot, policy_version, reward):
    s = Sample(group_index=group_index, index=slot, prompt="p", reward=reward)
    s.train_metadata = {"group_index": group_index, "trajectory_slot": slot, "policy_version": policy_version}
    return s


def test_convert_samples_to_train_data_numeric_row_only():
    args = SimpleNamespace(
        debug_train_only=True,
        custom_reward_post_process_path=None,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
    )
    samples = [
        _sample(0, 0, 7, 1.0),
        _sample(0, 1, 7, 2.0),
        _sample(1, 0, 7, 3.0),
        _sample(1, 1, 7, 4.0),
    ]
    row = ng.convert_samples_to_train_data(args, samples)
    # Numeric-only fields, with sample_indices retained for TransferQueue's
    # per-row custom metadata. Media, prompt strings and policy versions stay in
    # the trajectory object/manifest path.
    assert set(row.keys()) == {
        "sample_indices",
        "group_indices",
        "trajectory_slots",
        "trajectory_refs",
        "advantages",
        "raw_reward",
        "total_lengths",
        "skip_optimizer_step",
    }
    assert row["sample_indices"] == [0, 1, 0, 1]
    assert row["group_indices"] == [0, 0, 1, 1]
    assert row["trajectory_slots"] == [0, 1, 0, 1]
    assert row["trajectory_refs"] == [[0] * 4] * 4
    assert row["advantages"] == [1.0, 2.0, 3.0, 4.0]
    assert row["raw_reward"] == [1.0, 2.0, 3.0, 4.0]
    assert row["skip_optimizer_step"] == [0, 0, 0, 0]


def test_convert_samples_to_train_data_preserves_raw_rewards_after_normalization():
    args = SimpleNamespace(
        debug_train_only=True,
        custom_reward_post_process_path=None,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
    )
    samples = [_sample(0, 0, 7, 1.0), _sample(0, 1, 7, 3.0)]

    row = ng.convert_samples_to_train_data(args, samples)

    assert row["advantages"] == [-1.0, 1.0]
    assert row["raw_reward"] == [1.0, 3.0]


def test_convert_samples_to_train_data_accepts_generative_reward_pair(monkeypatch):
    from relax.utils import utils

    monkeypatch.setattr(utils, "post_process_rewards", lambda _args, _samples: ([1.0, 3.0], [-1.0, 1.0]))
    args = SimpleNamespace(debug_train_only=True)
    samples = [_sample(0, 0, 7, 1.0), _sample(0, 1, 7, 3.0)]

    row = ng.convert_samples_to_train_data(args, samples)

    assert row["advantages"] == [-1.0, 1.0]
    assert row["raw_reward"] == [1.0, 3.0]


def test_adapter_pack_group_stacks_candidates():
    class _A:
        def pack_trajectory(self, resp):
            return {
                "sigmas": resp["sigmas"],
                "sde_indices": resp["sde_indices"],
                "image_x_t": resp["x_t"],
                "cond_embeds": resp["cond"],
            }

    sig = torch.linspace(1, 0, 4)
    sde = torch.tensor([0, 1])
    responses = [
        {"sigmas": sig, "sde_indices": sde, "x_t": torch.randn(2, 3), "cond": torch.randn(4)},
        {"sigmas": sig, "sde_indices": sde, "x_t": torch.randn(2, 3), "cond": torch.randn(4)},
    ]
    common, conditions, tracks = ng.adapter_pack_group(_A(), responses)
    # sigmas/sde_indices are shared (single copy); tracks stacked over candidates.
    assert common["sigmas"].shape == sig.shape
    assert tracks["image_x_t"].shape == (2, 2, 3)
    # These candidates' conditions genuinely DIFFER, so they must still stack.
    assert conditions["cond_embeds"].shape == (2, 4)


def test_adapter_pack_group_deduplicates_identical_conditions():
    """A group shares one prompt, so its condition tensors are bit-identical.

    Storing G copies of the same text embedding was ~58 MB of pure waste per
    16-candidate group, written to disk and re-read by every rank.
    """

    class _A:
        def pack_trajectory(self, resp):
            return {"sde_indices": resp["sde_indices"], "image_x_t": resp["x_t"], "cond_embeds": resp["cond"]}

    cond = torch.randn(4, 8)
    sde = torch.tensor([0, 1])
    responses = [{"sde_indices": sde, "x_t": torch.randn(2, 3), "cond": cond.clone()} for _ in range(4)]
    _common, conditions, tracks = ng.adapter_pack_group(_A(), responses)

    assert tracks["image_x_t"].shape == (4, 2, 3)  # per-candidate, still stacked
    assert conditions["cond_embeds"].shape == (1, 4, 8)  # one copy, dim kept
    assert torch.equal(conditions["cond_embeds"][0], cond)


@pytest.mark.parametrize("key", ["sigmas", "sde_indices"])
def test_adapter_pack_group_rejects_candidate_schedule_mismatch(key):
    class _A:
        def pack_trajectory(self, resp):
            return resp

    first = {"sigmas": torch.tensor([1.0, 0.5]), "sde_indices": torch.tensor([0]), "image_x_t": torch.ones(1)}
    second = {name: value.clone() for name, value in first.items()}
    second[key][-1] += 1

    with pytest.raises(ValueError, match=f"different {key} schedule"):
        ng.adapter_pack_group(_A(), [first, second])


def test_native_generation_sample_id_prefers_explicit_prompt_id():
    sample = Sample(prompt="p", metadata={"prompt_id": "custom:42", "sample_id": "pickapic_000042"})
    assert ng._native_generation_sample_id(sample, group_index=9, slot=1) == "prompt:custom:42:sample:1"


def test_native_generation_sample_id_uses_record_sample_id():
    train_sample = Sample(prompt="p", metadata={"sample_id": "pickapic_000123"})
    eval_sample = Sample(prompt="p", metadata={"sample_id": "pickapic_eval_000004"})

    assert ng._native_generation_sample_id(train_sample, group_index=0, slot=2) == "prompt:pickapic_000123:sample:2"
    assert (
        ng._native_generation_sample_id(eval_sample, group_index=0, slot=1) == "prompt:pickapic_eval_000004:sample:1"
    )


def test_native_generation_sample_id_keeps_positional_fallback():
    sample = Sample(prompt="p", metadata={})
    assert ng._native_generation_sample_id(sample, group_index=7, slot=3) == "prompt:7:sample:3"


def test_native_generation_sample_id_can_force_positional_mode():
    sample = Sample(prompt="p", metadata={"prompt_id": "custom:42", "sample_id": "pickapic_000004"})
    assert (
        ng._native_generation_sample_id(sample, group_index=7, slot=3, sampling={"sample_id_mode": "positional"})
        == "prompt:7:sample:3"
    )


def test_read_eval_prompts_uses_record_metadata_without_extra_nesting(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"prompt":"p","metadata":{"task":"t2i","sample_id":"pickapic_000001"}}\n',
        encoding="utf-8",
    )
    args = SimpleNamespace(input_key="prompt")
    cfg = SimpleNamespace(path=str(path))

    samples = ng._read_eval_prompts(args, cfg)

    assert samples[0].metadata == {"task": "t2i", "sample_id": "pickapic_000001"}
    assert ng._native_generation_sample_id(samples[0], group_index=0, slot=0) == "prompt:pickapic_000001:sample:0"


# ---------------------------------------------------------------------------
# DP work sharding (design doc 3.1)
# ---------------------------------------------------------------------------


def _write_two_groups(tmp_path, group_size=4, shared_conditions=False):
    """Two group sidecars with ``group_size`` candidates each + the numeric
    rows.

    Advantages encode (group*100 + slot) so a hydrated micro-batch's advantages
    reveal exactly which candidates the rank was assigned.
    """
    sigmas = torch.linspace(1, 0, 3)
    sde = torch.tensor([0, 1])
    for g in (0, 1):
        path = ng.group_sidecar_path(str(tmp_path), "t2i", rollout_id=0, group_index=g)
        common = {"sigmas": sigmas, "sde_indices": sde}
        # A deduplicated condition keeps a leading dim of 1 (see adapter_pack_group).
        n_cond = 1 if shared_conditions else group_size
        conditions = {"cond_prompt_embeds": torch.randn(n_cond, 8)}
        # candidate is dim 0; the per-candidate value equals its slot for asserts.
        x = torch.arange(group_size).float().reshape(group_size, 1, 1).repeat(1, 2, 4)
        tracks = {"image_x_t": x, "image_x_next": x + 100}
        ng.write_group_sidecar(path, common, conditions, tracks)
    rows = {
        "group_indices": torch.tensor([g for g in (0, 1) for _ in range(group_size)]),
        "trajectory_slots": torch.tensor([s for _ in (0, 1) for s in range(group_size)]),
        "advantages": torch.tensor([g * 100 + s for g in (0, 1) for s in range(group_size)]).float(),
        "skip_optimizer_step": torch.zeros(2 * group_size, dtype=torch.long),
    }
    args = SimpleNamespace(artifact_root=str(tmp_path), generation_task="t2i")
    return args, rows


def _permute_rows(rows, order):
    index = torch.as_tensor(order, dtype=torch.long)
    return {k: v[index] for k, v in rows.items()}


def test_hydrate_joins_advantages_by_slot_not_row_order(tmp_path):
    """The advantage↔candidate join must survive a reordered TQ read.

    ``by_group`` used to collect rows in arrival order and index them
    positionally, so any sampler / partition path that returned rows out of
    insertion order would attach every advantage to the wrong image — silently,
    with a plausible-looking loss curve. Shuffle the rows and demand the same
    binding.
    """
    args, rows = _write_two_groups(tmp_path, group_size=4)
    ordered = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=0, dp_world=1)
    # Interleave the two groups and reverse the slots within each.
    shuffled_rows = _permute_rows(rows, [7, 3, 6, 2, 5, 1, 4, 0])
    shuffled = ng.hydrate_micro_batches(args, None, 0, shuffled_rows, dp_rank=0, dp_world=1)

    assert [mb["advantages"].tolist() for mb in ordered] == [[0.0, 1.0, 2.0, 3.0], [100.0, 101.0, 102.0, 103.0]]
    assert [mb["advantages"].tolist() for mb in shuffled] == [mb["advantages"].tolist() for mb in ordered]
    # ...and the advantage at position j really is the advantage of sidecar
    # candidate j: image_x_t's per-candidate value equals its slot.
    for mb in shuffled:
        slots = mb["batch"]["image_x_t"][:, 0, 0].tolist()
        assert [a % 100 for a in mb["advantages"].tolist()] == slots


def test_hydrate_loads_a_group_from_serialized_ray_object_ref(monkeypatch, tmp_path):
    import ray

    args, rows = _write_two_groups(tmp_path, group_size=4)
    payload = b"serialized-object-ref"
    packed_ref = len(payload).to_bytes(4, "little") + payload
    rows["trajectory_refs"] = torch.tensor([list(packed_ref)] * 8)
    expected = {"sde_indices": torch.tensor([1]), "image_x_t": torch.arange(4).reshape(4, 1)}
    monkeypatch.setattr(ray.cloudpickle, "loads", lambda value: value)
    monkeypatch.setattr(ray, "get", lambda ref: expected if ref == payload else None)

    batches = ng.hydrate_micro_batches(args, None, 0, rows)

    assert torch.equal(batches[0]["batch"]["sde_indices"], expected["sde_indices"])
    assert torch.equal(batches[0]["batch"]["image_x_t"], expected["image_x_t"])


def test_hydrate_shuffled_rows_bind_correct_advantage_under_dp_shard(tmp_path):
    # Same guarantee once the group is split across ranks: rank r replays
    # candidate r and must get candidate r's advantage, whatever order the rows
    # arrived in.
    args, rows = _write_two_groups(tmp_path, group_size=4)
    shuffled_rows = _permute_rows(rows, [2, 5, 0, 7, 4, 1, 6, 3])
    for rank in range(4):
        mbs = ng.hydrate_micro_batches(args, None, 0, shuffled_rows, dp_rank=rank, dp_world=4)
        for gi, mb in enumerate(mbs):
            assert mb["advantages"].tolist() == [float(gi * 100 + rank)]
            assert mb["batch"]["image_x_t"][0, 0, 0].item() == float(rank)


def test_hydrate_rejects_a_duplicated_or_missing_slot(tmp_path):
    import pytest

    args, rows = _write_two_groups(tmp_path, group_size=4)
    dup = {k: v.clone() for k, v in rows.items()}
    dup["trajectory_slots"][1] = 0  # group 0 now has slot 0 twice and no slot 1
    with pytest.raises(ValueError, match="two rows for trajectory slot"):
        ng.hydrate_micro_batches(args, None, 0, dup, dp_rank=0, dp_world=1)

    gap = {k: v.clone() for k, v in rows.items()}
    gap["trajectory_slots"][1] = 9  # slot 1 missing, an out-of-range slot present
    with pytest.raises(ValueError, match="missing trajectory slots"):
        ng.hydrate_micro_batches(args, None, 0, gap, dp_rank=0, dp_world=1)


def test_hydrate_requires_the_slot_column(tmp_path):
    import pytest

    args, rows = _write_two_groups(tmp_path, group_size=4)
    rows.pop("trajectory_slots")
    with pytest.raises(ValueError, match="trajectory_slots"):
        ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=0, dp_world=1)


def test_hydrate_expands_deduplicated_conditions_to_the_candidate_count(tmp_path):
    """A group's single stored condition must reach the replay as ``[B, ...]``.

    The dedup is only safe because hydrate broadcasts it back; it is a view, so
    it costs no memory.
    """
    args, rows = _write_two_groups(tmp_path, group_size=4, shared_conditions=True)
    mbs = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=0, dp_world=1)
    cond = mbs[0]["batch"]["cond_prompt_embeds"]
    assert cond.shape == (4, 8)
    assert torch.equal(cond[0], cond[3])  # broadcast of the one stored copy

    # Sharded: each rank replays one candidate and sees a [1, 8] condition.
    sharded = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=1, dp_world=4)
    assert sharded[0]["batch"]["cond_prompt_embeds"].shape == (1, 8)


def test_hydrate_dp_shard_disjoint_and_covers_all(tmp_path):
    args, rows = _write_two_groups(tmp_path, group_size=4)
    world = 4
    seen_per_group = {0: [], 1: []}
    for rank in range(world):
        mbs = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=rank, dp_world=world)
        assert len(mbs) == 2  # one micro-batch (step) per group on every rank
        for gi, mb in enumerate(mbs):
            # group_size / world == 1 candidate per rank
            assert mb["batch"]["image_x_t"].shape[0] == 1
            assert mb["advantages"].numel() == 1
            slot = int(mb["advantages"].item()) - gi * 100
            seen_per_group[gi].append(slot)
    # union of all ranks' candidate slots == the full group, disjoint
    for gi in (0, 1):
        assert sorted(seen_per_group[gi]) == [0, 1, 2, 3]


def test_hydrate_dp_shard_selects_correct_candidate(tmp_path):
    args, rows = _write_two_groups(tmp_path, group_size=4)
    # rank 2 of 4 → candidate index 2 in each group ([dp_rank::dp_world]).
    mbs = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=2, dp_world=4)
    assert mbs[0]["advantages"].tolist() == [2.0]  # group 0, slot 2
    assert mbs[1]["advantages"].tolist() == [102.0]  # group 1, slot 2
    # the sharded batch tensor holds exactly that candidate (value == slot 2)
    assert torch.allclose(mbs[0]["batch"]["image_x_t"], torch.full((1, 2, 4), 2.0))
    # shared schedule tensors are NOT sharded
    assert mbs[0]["batch"]["sde_indices"].tolist() == [0, 1]


def test_hydrate_no_shard_when_group_not_divisible(tmp_path):
    args, rows = _write_two_groups(tmp_path, group_size=4)
    # 4 % 3 != 0 → fall back to full replay on every rank (identical batches).
    for rank in range(3):
        mbs = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=rank, dp_world=3)
        assert mbs[0]["batch"]["image_x_t"].shape[0] == 4
        assert mbs[0]["advantages"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_hydrate_single_rank_is_full_batch(tmp_path):
    args, rows = _write_two_groups(tmp_path, group_size=4)
    mbs = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=0, dp_world=1)
    assert mbs[0]["batch"]["image_x_t"].shape[0] == 4
    assert mbs[0]["advantages"].numel() == 4


def test_candidate_chunks_splits_only_on_even_division():
    # Disabled / no-op / short tail → one full-slice chunk (an unequal split would
    # silently reweight the actor's accumulated 1/N mean).
    assert ng._candidate_chunks(4, 0) == [[0, 1, 2, 3]]
    assert ng._candidate_chunks(4, 4) == [[0, 1, 2, 3]]
    assert ng._candidate_chunks(4, 8) == [[0, 1, 2, 3]]
    assert ng._candidate_chunks(3, 2) == [[0, 1, 2]]
    # Even division → one chunk per micro-batch, covering the slice exactly once.
    assert ng._candidate_chunks(4, 1) == [[0], [1], [2], [3]]
    assert ng._candidate_chunks(4, 2) == [[0, 1], [2, 3]]


def test_hydrate_splits_rank_slice_by_micro_batch_size(tmp_path):
    # group_size 4 over dp_world 2 → 2 candidates/rank; --micro-batch-size 1 splits
    # that into 2 single-candidate micro-batches so the replay peak stays at B=1
    # however large n_samples_per_prompt gets.
    args, rows = _write_two_groups(tmp_path, group_size=4)
    args.micro_batch_size = 1
    mbs = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=0, dp_world=2)
    assert len(mbs) == 4  # 2 groups x 2 chunks
    for mb in mbs:
        assert mb["batch"]["image_x_t"].shape[0] == 1
        assert mb["advantages"].numel() == 1
        assert mb["batch"]["sde_indices"].tolist() == [0, 1]  # shared, never sliced
    # rank 0 of 2 owns candidates [0, 2] of each group; the chunks cover them once.
    assert [mb["advantages"].item() for mb in mbs] == [0.0, 2.0, 100.0, 102.0]
    assert [mb["local_idx"] for mb in mbs] == [[0], [2], [0], [2]]
    # Unset / 0 → unchanged single micro-batch per group holding the whole slice.
    args.micro_batch_size = 0
    whole = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=0, dp_world=2)
    assert len(whole) == 2 and whole[0]["advantages"].tolist() == [0.0, 2.0]


# ---------------------------------------------------------------------------
# Group-batched dispatch (one request per group, n candidates per request)
# ---------------------------------------------------------------------------


class _FakeRef:
    def __init__(self, value):
        self.value = value


class _FakeEngine:
    """Engine stub whose ``generate_batch`` echoes back its request shard.

    Echoing is enough for ``_dispatch_groups``, which is only responsible for
    fan-out and for regrouping results in input order — it never looks inside a
    response.
    """

    def __init__(self, tag):
        self.tag = tag
        self.seen = []

    class _Remote:
        def __init__(self, outer):
            self.outer = outer

        def remote(self, requests):
            self.outer.seen.append(list(requests))
            return _FakeRef([dict(r) for r in requests])

    @property
    def generate_batch(self):
        return self._Remote(self)


def _patch_ray(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("ray")
    fake.get = lambda refs: [r.value for r in refs]
    fake.ObjectRef = _FakeRef
    monkeypatch.setitem(sys.modules, "ray", fake)


def test_dispatch_groups_spreads_candidates_and_regroups_in_order(monkeypatch):
    """Candidates -- not groups -- must be the unit of load balancing.

    Dispatching group-by-group idles engines whenever a group is smaller than the
    engine count (eval: 2 candidates over 8 engines = 1/4 occupancy). And the
    result must come back regrouped IN INPUT ORDER, because the caller zips it
    against its group list; shard-order would pair a group's samples with another
    group's trajectories.
    """
    _patch_ray(monkeypatch)
    engines = [_FakeEngine(i) for i in range(4)]
    # 3 groups of 2 candidates = 6 candidates over 4 engines.
    group_requests = [[{"id": f"g{g}c{c}"} for c in range(2)] for g in range(3)]

    out = ng._dispatch_groups(engines, group_requests)

    assert [[r["id"] for r in grp] for grp in out] == [
        ["g0c0", "g0c1"],
        ["g1c0", "g1c1"],
        ["g2c0", "g2c1"],
    ]
    # 6 candidates over 4 engines -> 2/2/1/1, i.e. every engine got work.
    assert sorted(len(e.seen[0]) for e in engines) == [1, 1, 2, 2]


def test_dispatch_groups_rejects_a_short_shard(monkeypatch):
    _patch_ray(monkeypatch)

    class _ShortEngine(_FakeEngine):
        class _Remote:
            def __init__(self, outer):
                self.outer = outer

            def remote(self, requests):
                return _FakeRef([])  # engine dropped every candidate

        @property
        def generate_batch(self):
            return self._Remote(self)

    import pytest

    with pytest.raises(RuntimeError, match="returned 0 responses"):
        ng._dispatch_groups([_ShortEngine(0)], [[{"id": "a"}]])


# The engine-side "a group that comes back short must fail loudly" check moved
# to its owner's suite when num_outputs_per_prompt was pinned to 1 — see
# tests/backends/sglang/test_diffusion_engine.py::
# test_map_responses_requires_exactly_one_candidate. Nothing to duplicate here.


# ---------------------------------------------------------------------------
# Artifact retention
# ---------------------------------------------------------------------------


def _touch_rollout_dir(root, task, rollout_id):
    path = os.path.join(root, task, f"rollout_{int(rollout_id):07d}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "group_00000000.safetensors"), "w", encoding="utf-8") as f:
        f.write("x")
    return path


def test_prune_stale_artifacts_keeps_the_retention_window(tmp_path):
    """Sidecars + PNGs are needed only until the group's train step ran.

    Nothing else deletes them, so without this a few hundred steps fill the
    shared volume and the run dies on an unrelated write error.
    """
    root = str(tmp_path)
    args = SimpleNamespace(artifact_root=root, generation_task="t2i", artifact_retention_rollouts=2)
    dirs = {r: _touch_rollout_dir(root, "t2i", r) for r in range(6)}
    eval_dir = _touch_rollout_dir(os.path.join(root, "eval"), "t2i", 1)

    ng.prune_stale_artifacts(args, rollout_id=5)

    # keep = 2 → rollouts 4 and 5 survive; 0..3 (and the stale eval dir) are gone.
    assert [r for r in range(6) if os.path.exists(dirs[r])] == [4, 5]
    assert not os.path.exists(eval_dir)


def test_prune_stale_artifacts_can_be_disabled_and_tolerates_a_missing_root(tmp_path):
    root = str(tmp_path)
    dirs = {r: _touch_rollout_dir(root, "t2i", r) for r in range(3)}
    ng.prune_stale_artifacts(
        SimpleNamespace(artifact_root=root, generation_task="t2i", artifact_retention_rollouts=0), rollout_id=99
    )
    assert all(os.path.exists(p) for p in dirs.values())
    # A root that was never written to must not raise on the first rollout.
    ng.prune_stale_artifacts(
        SimpleNamespace(
            artifact_root=os.path.join(root, "nope"), generation_task="t2i", artifact_retention_rollouts=1
        ),
        rollout_id=7,
    )
