# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU acquisition-to-two-update checks; no real Ray/FSDP collectives."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from relax.backends.fsdp.art import art_replay_update
from relax.engine.rewards import generative
from relax.engine.rewards.art import scalarize_art_rewards
from relax.engine.rollout import native_generation as ng
from relax.engine.rollout.art_cache import tensor_payload_bytes
from relax.utils.types import Sample


def _acquisition(monkeypatch, *, size=8, constant=False, invalid=()):
    args = SimpleNamespace(
        sampling_config={
            "algorithm": "art",
            "art": {"train_step_indices": [0, 2], "max_cache_bytes": 10**8, "max_replay_bytes": 10**8},
        },
        reward_component_weights={"ocr": 2.0, "lpips": -0.5},
        generative_advantage_std_mode="group",  # Must NOT normalize ART rewards.
        micro_batch_size=1,
        n_samples_per_prompt=size,
        global_batch_size=2 * size,
        debug_train_only=False,
        seed=7,
    )
    samples, caches = [], {}
    for group in (3, 9):
        values = torch.arange(size).float()
        endpoints = torch.stack([values, -values], 1).half()
        caches[group] = {
            "terminal_state": endpoints,
            "art_states": torch.arange(size).half()[:, None, None].expand(size, 2, 2).clone(),
            "art_old_velocities": torch.zeros(size, 2, 2),
            "art_step_indices": torch.tensor([0, 2]),
            "art_timesteps": torch.tensor([999.9999389648438, 300.0000305175781]),
            "art_sigmas": torch.tensor([1.0, 0.3]),
            "art_ode_dts": torch.tensor([-0.1, -0.3]),
            "valid_masks": torch.ones(size, dtype=torch.bool),
            "sampling_fingerprint": torch.arange(32, dtype=torch.uint8),
            "cond_prefix": torch.tensor([[float(group)]]),
            "policy_versions": torch.full((size,), 7),
            "weight_manifest_sha256": torch.full((size, 32), 0xAA, dtype=torch.uint8),
        }
        for slot in range(size):
            # Identical text must NOT merge distinct conditioning instances.
            samples.append(
                Sample(
                    prompt="same text",
                    index=len(samples),
                    group_index=group,
                    train_metadata={
                        "group_index": group,
                        "trajectory_slot": slot,
                        "art_cache_bytes": tensor_payload_bytes(caches[group]),
                        "art_guidance_bytes": 8,
                    },
                )
            )
    components = {
        "ocr": [
            1.0 if constant or s.group_index == 9 else float(s.train_metadata["trajectory_slot"]) for s in samples
        ],
        "lpips": [0.25] * len(samples),
    }
    components["__valid_mask__"] = [index not in invalid for index in range(len(samples))]
    monkeypatch.setattr(generative, "_consume_prefetched_rewards", lambda _samples: components.copy())
    monkeypatch.setattr(generative, "_log_reward_metrics", lambda *_args: None)
    monkeypatch.setattr("relax.utils.utils.post_process_rewards", generative.post_process)
    rows = ng.convert_samples_to_train_data(args, samples)
    monkeypatch.setattr(ng, "_load_group_batch_for_rows", lambda _a, _b, _r, group, *_rest: caches[group])
    return args, rows, caches, samples


def test_raw_reward_hook_converter_and_complete_group_join_survive_row_reordering(monkeypatch):
    args, rows, caches, samples = _acquisition(monkeypatch, size=32)
    assert rows["art_rewards"].dtype == torch.float32
    assert rows["art_rewards"][:32].tolist() == [2.0 * i - 0.125 for i in range(32)]
    assert not rows["advantages"].any()
    assert samples[1].reward == 1.0  # Raw primary logging stays separate.
    assert samples[1].train_metadata["art_reward"] == 1.875
    expected = ng.hydrate_micro_batches(args, None, 5, rows, dp_rank=0, dp_world=4)
    permutation = torch.randperm(len(samples), generator=torch.Generator().manual_seed(31))
    actual = ng.hydrate_micro_batches(args, None, 5, rows[permutation], dp_rank=0, dp_world=4)
    assert [(m["group_index"], m["local_idx"]) for m in actual] == [
        (m["group_index"], m["local_idx"]) for m in expected
    ]
    for mb in actual:
        assert "art_states" not in mb["batch"]
        batch = mb["art_load"]()
        group, slot = mb["group_index"], mb["local_idx"][0]
        torch.testing.assert_close(batch["art_states"], caches[group]["art_states"][slot : slot + 1])
        assert mb["batch"]["sampling_fingerprint"].shape == (32,)  # K=32 is NOT a candidate axis.
        assert mb["batch"]["sampling_fingerprint"].data_ptr() != caches[group]["sampling_fingerprint"].data_ptr()
        assert mb["batch"]["art_timesteps"].shape == (2,)
        assert mb["art_rewards"].item() == rows["art_rewards"][int(group == 9) * 32 + slot].item()
        direction = torch.tensor([[1.0, -1.0]]) if group == 3 else torch.zeros(1, 2)
        torch.testing.assert_close(batch["art_guidance_direction"], direction)
        assert not mb["skip_optimizer_step"]  # Mixed zero/active group is not globally degenerate.


class _Adapter:
    def __init__(self):
        self.prefix_calls, self.times = 0, []

    def prepare_replay_batch(self, model, batch):
        self.prefix_calls += 1
        return batch

    def predict_training_velocity(self, model, batch, state, sigma):
        self.times.append((state.clone(), batch["_replay_timestep"].clone(), sigma.clone()))
        return model.weight.expand_as(state)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("retain", [False, True])
def test_routed_replay_preserves_global_gradients_and_cached_inputs(monkeypatch, world, retain):
    args, rows, caches, _samples = _acquisition(monkeypatch)
    if not retain:
        args.sampling_config["art"]["max_replay_bytes"] = 8 * max(map(tensor_payload_bytes, caches.values())) + 16
    batches = [ng.hydrate_micro_batches(args, None, 2, rows, dp_rank=r, dp_world=world) for r in range(world)]
    assert {len(mbs) for mbs in batches} == {16 // world}
    model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, foreach=False)
    initial_cache = {g: b["art_old_velocities"].clone() for g, b in caches.items()}
    adapter = _Adapter()
    for update in range(2):
        rank_grads, seen, masked = [], [], 0
        for mbs in batches:
            model.zero_grad(set_to_none=True)
            selected = [mb for mb in mbs if mb["art_update_index"] == update]
            assert len(selected) == 8 // world
            for mb in selected:
                batch = mb["art_load"]()
                result = art_replay_update(
                    model,
                    adapter,
                    batch,
                    global_item_count=mb["art_global_item_count"],
                    data_parallel_world_size=world,
                )
                masked += int(result["masked_items"])
                seen.append((mb["group_index"], mb["local_idx"][0]))
                for state, time, sigma in adapter.times[-2:]:
                    slot = 0 if time.item() > 900 else 1
                    torch.testing.assert_close(time, batch["art_timesteps"][slot : slot + 1], rtol=0, atol=0)
                    torch.testing.assert_close(sigma, batch["art_sigmas"][slot : slot + 1], rtol=0, atol=0)
                    torch.testing.assert_close(state, batch["art_states"][:, slot], rtol=0, atol=0)
            rank_grads.append(model.weight.grad.clone())
        # Simulate FSDP's gradient average; this is not a distributed hardware test.
        model.weight.grad = torch.stack(rank_grads).mean(0)
        # Four active candidates * (sigma=1 + sigma=.3) / 16 original items.
        # The independent scalar Adam/moment oracle lives in test_art.py.
        expected = torch.tensor([[0.325, -0.325]]) if update == 0 else torch.zeros_like(model.weight)
        torch.testing.assert_close(model.weight.grad, expected)
        if update == 1:
            assert masked == 8  # Every active row/time is masked, but optimizer MUST still step.
            assert not model.weight.grad.any()
        before = model.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(model.weight, before)
        assert optimizer.state[model.weight]["step"] == update + 1
        assert sorted(g for g, _slot in seen) == [3] * 4 + [9] * 4
    assert adapter.prefix_calls == 16
    for group, cache in caches.items():
        torch.testing.assert_close(cache["art_old_velocities"], initial_cache[group], rtol=0, atol=0)


def test_zero_signal_is_global_and_invalid_acquisition_fails_before_replay(monkeypatch):
    args, rows, caches, _samples = _acquisition(monkeypatch, constant=True)
    for rank in range(4):
        mbs = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=rank, dp_world=4)
        assert mbs and all(mb["skip_optimizer_step"] for mb in mbs)
    with pytest.raises(ValueError, match="complete acquisition"):
        ng.hydrate_micro_batches(args, None, 0, rows[:8])
    with pytest.raises(ValueError, match="every configured"):
        ng.hydrate_micro_batches(args, None, 0, rows[:-1])
    invalid = rows.clone()
    invalid["valid_masks"][0] = 0
    with pytest.raises(ValueError, match="complete finite valid"):
        ng.hydrate_micro_batches(args, None, 0, invalid)
    caches[9]["valid_masks"] = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match="descriptor"):
        ng.hydrate_micro_batches(args, None, 0, rows)
    caches[9]["valid_masks"] = torch.ones(8, dtype=torch.bool)
    caches[9]["art_timesteps"] = caches[9]["art_timesteps"] + 1
    with pytest.raises(ValueError, match="identical selected"):
        ng.hydrate_micro_batches(args, None, 0, rows)


@pytest.mark.parametrize("invalid", [(0,), tuple(range(16))])
def test_judge_failure_skips_complete_acquisition_on_every_rank(monkeypatch, invalid):
    args, rows, _caches, samples = _acquisition(monkeypatch, invalid=invalid)
    assert rows["skip_optimizer_step"].bool().all()
    assert not samples[0].train_metadata["valid_mask"]
    for rank in range(8):
        batches = ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=rank, dp_world=8)
        assert len(batches) == 2 and all(mb["skip_optimizer_step"] for mb in batches)
        for mb in batches:
            assert not mb["art_load"]()["art_guidance_direction"].any()
    rows["skip_optimizer_step"][0] = 0
    with pytest.raises(ValueError, match="acquisition-wide skip"):
        ng.hydrate_micro_batches(args, None, 0, rows)


@pytest.mark.parametrize("case", ["invalid", "missing", "nonfinite", "weight", "overflow", "no_weights"])
def test_raw_scalarization_rejects_failed_reward_contract(case):
    components, weights, valid = {"ocr": [1.0, 2.0]}, {"ocr": 2.0}, [True, True]
    if case == "invalid":
        valid[0] = False
    elif case == "missing":
        weights["absent"] = 1.0
    elif case == "nonfinite":
        components["ocr"][0] = float("nan")
    elif case == "weight":
        weights["ocr"] = float("inf")
    elif case == "overflow":
        components["ocr"][0] = 3e38
    else:
        weights = {}
    with pytest.raises(ValueError, match="ART"):
        scalarize_art_rewards(components, weights, required=[], valid_mask=valid)


def test_raw_scalarization_preserves_upstream_component_order_and_rounding():
    components = {"c": [1e8, 0.187], "a": [-1e8, 1.1234567], "b": [1.0, -0.217]}
    weights = {"c": 1.0, "a": 1.0, "b": 0.713}
    result = scalarize_art_rewards(components, weights, required=[], valid_mask=[True, True])
    expected = sum(weights[key] * torch.tensor(components[key]) for key in sorted(components))
    torch.testing.assert_close(torch.tensor(result), expected, rtol=0, atol=0)
    assert result[0] == 0.0  # Unsorted c+a+b would incorrectly keep the small b term.


def test_cache_budget_rejects_before_tensor_loading_and_plan_releases_full_groups(monkeypatch):
    args, rows, caches, _ = _acquisition(monkeypatch)
    references = []

    def load(_args, _adapter, _rollout, group, *_rest):
        batch = {key: value.clone() for key, value in caches[group].items()}
        references.append(weakref.ref(batch["art_states"]))
        return batch

    monkeypatch.setattr(ng, "_load_group_batch_for_rows", load)
    args.sampling_config["art"]["max_replay_bytes"] = 1
    with pytest.raises(ValueError, match="max_replay_bytes"):
        ng.hydrate_micro_batches(args, None, 0, rows)
    assert references == []
    args.sampling_config["art"]["max_replay_bytes"] = 10**8
    args.sampling_config["art"]["max_cache_bytes"] = 1
    with pytest.raises(ValueError, match="max_cache_bytes"):
        ng.hydrate_micro_batches(args, None, 0, rows)
    assert references == []
    args.sampling_config["art"]["max_cache_bytes"] = 10**8
    plan = ng.hydrate_micro_batches(args, None, 0, rows)
    assert len(references) == 2 and all(reference() is None for reference in references)
    assert all("art_states" not in mb["batch"] for mb in plan)
    candidate = plan[0]["art_load"]()
    assert candidate["art_states"].shape == (1, 2, 2)
    assert all(reference() is None for reference in references)


def test_art_publisher_keeps_exact_byte_descriptors_and_rejects_before_publication(monkeypatch, tmp_path):
    from tests.cpu.backends.sglang.test_sensenova_art_rollout import SenseNovaU1ImageEditAdapter, _pack, _pair

    # Publishing retains reward images; keep them local to this test.
    monkeypatch.setattr(ng, "_CANDIDATE_IMAGE_CACHE", {})
    monkeypatch.setattr(ng, "_CANDIDATE_IMAGE_ROLLOUT_IDS", {})
    adapter = SenseNovaU1ImageEditAdapter()
    responses = [_pack(*_pair())[0] for _ in range(2)]
    samples = [Sample(prompt="edit", index=index, group_index=0) for index in range(2)]
    args = SimpleNamespace(generation_task="i2i", artifact_root=str(tmp_path), reward_runtime="cpu")
    sampling = {"algorithm": "art", "art": {"max_cache_bytes": 1, "max_replay_bytes": 10**6}}
    published = []

    def publish(_args, _sampling, _rollout, _group, common, conditions, tracks):
        published.append(tensor_payload_bytes({**common, **conditions, **tracks}))
        return b"owner-aware-ref", "ray://test"

    monkeypatch.setattr(ng, "_publish_group_trajectory", publish)
    budget = {"sizes": [], "guidance": []}
    with pytest.raises(ValueError, match="max_cache_bytes"):
        ng._prepare_generated_group(args, sampling, adapter, 0, "b" * 64, samples, responses, budget)
    assert published == [] and budget["sizes"] == []
    sampling["art"]["max_cache_bytes"] = 10**6
    ng._prepare_generated_group(args, sampling, adapter, 0, "b" * 64, samples, responses, budget)
    assert len(published) == 1 and budget["sizes"] == published
    assert all(sample.train_metadata["art_cache_bytes"] == published[0] for sample in samples)
    assert all(sample.train_metadata["art_guidance_bytes"] == 3 * 8 * 8 * 4 for sample in samples)


def test_art_dense_lower_bound_is_checked_before_engine_dispatch(monkeypatch):
    args = SimpleNamespace(
        rollout_batch_size=1,
        sampling_config={
            "algorithm": "art",
            "art": {"train_step_indices": [0, 2], "max_cache_bytes": 1, "max_replay_bytes": 10**6},
        },
    )
    monkeypatch.setattr(
        ng,
        "_load_adapter",
        lambda _args: SimpleNamespace(build_rollout_request=lambda *_args, **_kwargs: {"height": 1024, "width": 1024}),
    )
    monkeypatch.setattr(ng, "_get_engines", lambda *_args: [object()])
    monkeypatch.setattr(ng, "_get_samples", lambda *_args: [[Sample(prompt="edit", group_index=0)] * 8])
    monkeypatch.setattr(ng, "_dispatch_groups", lambda *_args, **_kwargs: pytest.fail("must reject before dispatch"))
    with pytest.raises(ValueError, match="before dispatch"):
        ng.generate_rollout(args, 0, None)


def test_owned_candidates_match_budget_fallback_without_reload_or_storage_alias(monkeypatch):
    from relax.engine.rollout import art_replay

    args, rows, caches, _ = _acquisition(monkeypatch)
    args.sampling_config["art"]["max_replay_bytes"] = 8 * max(map(tensor_payload_bytes, caches.values())) + 16
    loads, checks, references = [], [], []
    validate = art_replay._validate_art_group_cache

    def load(_args, _adapter, _rollout, group, *_rest):
        loads.append(group)
        batch = {key: value.clone() for key, value in caches[group].items()}
        references.extend(weakref.ref(value) for value in batch.values())
        return batch

    def check(batch, size):
        checks.append(size)
        validate(batch, size)

    monkeypatch.setattr(ng, "_load_group_batch_for_rows", load)
    monkeypatch.setattr(art_replay, "_validate_art_group_cache", check)
    expected = [mb["art_load"]() for mb in ng.hydrate_micro_batches(args, None, 2, rows, dp_rank=0, dp_world=4)]
    assert len(loads) == len(checks) == 2 + len(expected)  # Tight budgets retain the lazy path.
    loads.clear()
    checks.clear()
    references.clear()
    args.sampling_config["art"]["max_replay_bytes"] = 10**8
    plan = ng.hydrate_micro_batches(args, None, 2, rows, dp_rank=0, dp_world=4)
    assert loads == [3, 9] and checks == [8, 8]
    assert all(reference() is None for reference in references)
    for mb, wanted in zip(plan, expected):
        actual = mb["art_load"]()
        assert actual.keys() == wanted.keys()
        for key in wanted:
            torch.testing.assert_close(actual[key], wanted[key], rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="already consumed"):
            mb["art_load"]()
    assert loads == [3, 9] and checks == [8, 8]
    caches[3]["art_old_velocities"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        ng.hydrate_micro_batches(args, None, 0, rows, dp_rank=0, dp_world=4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_finite_validation_matches_full_scan_without_mutation(dtype):
    from relax.engine.rollout.art_replay import _finite_cpu_tensor

    cases = [
        torch.empty(0, dtype=dtype),
        torch.randn(7, 9).to(dtype).t(),
        torch.arange(30, dtype=dtype)[3::2],
        torch.ones(1, dtype=dtype).expand(20),
        torch.tensor([torch.finfo(dtype).tiny / 2, -torch.finfo(dtype).tiny / 2], dtype=dtype),
        torch.tensor([torch.finfo(dtype).min, torch.finfo(dtype).max, 0.0, -0.0], dtype=dtype),
    ]
    for bad in (float("nan"), float("inf"), -float("inf")):
        for offset in (0, 4, 8):
            value = torch.zeros(9, dtype=dtype)
            value[offset] = bad
            cases.append(value)
    for value in cases:
        before = value.clone()
        assert _finite_cpu_tensor(value) == bool(torch.isfinite(value).all())
        torch.testing.assert_close(value, before, rtol=0, atol=0, equal_nan=True)


def test_finite_validation_preserves_other_dtypes():
    from relax.engine.rollout.art_replay import _finite_cpu_tensor

    for value in (
        torch.tensor([True, False]),
        torch.tensor([1, 2]),
        torch.tensor([1 + 2j, complex(float("nan"), 0)]),
        torch.ones(4, dtype=torch.float64),
    ):
        assert _finite_cpu_tensor(value) == bool(torch.isfinite(value).all())
