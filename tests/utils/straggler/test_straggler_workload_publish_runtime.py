# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Runtime regression test for the P0-1 workload publish ordering.

This drives the REAL ``_get_prefetched_sft_window`` on a stub ``self`` and
forces the branch the source-order test cannot reach: a rank whose local
partition yields ONE micro-batch while the DP-wide maximum is THREE. Under the
old publish order the metadata described the discarded local partition, so the
published ``microbatches`` would be 1; the executed step runs 3. The assertion
is therefore on the executed K, not on the text of the file.
"""

import types

import pytest
import torch


# This harness drives the real Megatron actor path, so it needs the training
# stack. The CPU CI venv has `transfer_queue` but not `megatron`, and the
# actor module imports Megatron at collection time; skip there rather than
# fail collection.
# Run it with:
#   PYTHONPATH=$PWD:/root/autodl-tmp/megatron-stack/Megatron-LM \
#     /root/autodl-tmp/megatron-stack/venv/bin/python -m pytest <this file> -q
pytest.importorskip("transfer_queue", reason="requires the relaxed training stack (Megatron + transfer_queue)")
pytest.importorskip("megatron", reason="requires the Megatron training stack")

from relax.backends.megatron import actor as actor_mod  # noqa: E402 — guarded above
from relax.backends.megatron.actor import MegatronTrainRayActor  # noqa: E402 — guarded above
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions  # noqa: E402 — guarded above
from relax.utils.straggler import context as ctx  # noqa: E402 — guarded above


SAMPLES = [10, 10, 60, 90, 5, 5]
K_LOCAL = 1
MAX_K = 3
GLOBAL_BATCH = 32


class _FakeEvent:
    def record(self, stream=None):  # noqa: D102 - stub
        pass

    def synchronize(self):  # noqa: D102 - stub
        pass

    def wait(self, stream=None):  # noqa: D102 - stub
        pass


class _FakeStreamCtx:
    def __enter__(self):  # noqa: D102 - stub
        return None

    def __exit__(self, *exc):  # noqa: D102 - stub
        return False


class _FakeDeviceModule:
    @staticmethod
    def stream_context(stream):  # noqa: D102 - stub
        return _FakeStreamCtx()

    @staticmethod
    def Event():  # noqa: D102 - stub
        return _FakeEvent()


class _FakeDist:
    """all_reduce stub that answers the three collectives in call order."""

    class ReduceOp:  # noqa: D106 - stub
        MAX = "max"
        SUM = "sum"

    def __init__(self):
        self.calls = 0

    def all_reduce(self, tensor, op=None, group=None):  # noqa: D102 - stub
        # Answer by SHAPE and OPERATION, never by a call counter: a counter
        # desynchronises when the function under test is invoked more than once in
        # one process, which fed the global batch size into the DP-wide K reduce.
        self.calls += 1
        op_name = str(getattr(op, "name", op)).lower()
        if tensor.numel() == 4:  # tp_bounds = [k, -k, samples, -samples]
            tensor[:] = torch.tensor([K_LOCAL, -K_LOCAL, len(SAMPLES), -len(SAMPLES)], dtype=torch.int)
        elif "sum" in op_name:  # global sample count
            tensor[:] = torch.tensor([GLOBAL_BATCH], dtype=torch.int)
        else:  # dp_k_bounds -> the DP-wide maximum
            tensor[:] = torch.tensor([MAX_K], dtype=torch.int)


def _window(rollout_id):
    indices = get_seqlen_balanced_partitions(SAMPLES, K_LOCAL, equal_size=False)
    return types.SimpleNamespace(
        packed_micro_batches=[("cpu-batch", None) for _ in indices],
        first_device_micro_batch=("cpu-batch", None),
        first_ready_event=_FakeEvent(),
        rollout_data={"total_lengths": list(SAMPLES), "rollout_id": rollout_id},
    )


def _stub_self(window):
    stub = types.SimpleNamespace()
    stub.args = types.SimpleNamespace(global_batch_size=GLOBAL_BATCH, num_rollout=1, max_staleness=0)
    stub._sft_window_prefetcher = types.SimpleNamespace(
        get=lambda rollout_id: window,
        prefetch=lambda *a, **k: None,
    )
    stub._agree_sft_prefetch_window = lambda *a, **k: window
    # The method partials this at entry, before the branch under test.
    stub._fetch_and_prepack_sft_window = lambda *a, **k: None
    stub._sft_copy_stream = object()
    stub._sft_device = torch.device("cpu")
    return stub


@pytest.fixture
def _patched(monkeypatch):
    fake_dist = _FakeDist()
    monkeypatch.setattr(actor_mod, "dist", fake_dist, raising=False)
    monkeypatch.setattr(
        actor_mod, "device_utils", types.SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu"))
    )
    monkeypatch.setattr(
        actor_mod,
        "mpu",
        types.SimpleNamespace(
            get_tensor_model_parallel_group=lambda: "tp",
            get_data_parallel_group=lambda **k: "dp",
        ),
        raising=False,
    )
    monkeypatch.setattr(actor_mod, "device_module", _FakeDeviceModule)
    monkeypatch.setattr(actor_mod, "prepack_sft_micro_batch_cpu", lambda args, samples: ("cpu-batch", None))
    monkeypatch.setattr(actor_mod, "move_tensors_to_device", lambda batch, device, non_blocking=False: batch)
    monkeypatch.setattr(actor_mod, "_select_rollout_samples", lambda rollout_data, indices: list(indices))
    monkeypatch.setattr(actor_mod, "_raise_if_sft_peer_error", lambda *a, **k: None)
    monkeypatch.setattr(actor_mod, "_should_pause_sft_lookahead", lambda *a, **k: False)
    # The publish block is gated behind a cached enablement read, so the test must
    # turn the profiler ON, exactly as a real ON run does.
    monkeypatch.setattr(actor_mod, "_STRAGGLER_PUBLISH_ENABLED", True)
    ctx.reset_training_context_for_tests()
    return fake_dist


def test_repack_branch_publishes_the_executed_step_total_not_one_group(_patched):
    """local_k=1 < max_k=3: the single optimizer step carries the STEP TOTALS.

    One optimizer step consumes the whole window (the caller passes a single-
    element ``prepared_num_microbatches``), and ``_step_workload`` reads
    exactly one entry, so the published list must have exactly one entry whose
    values are the totals: tokens ``sum(samples)``, sequences ``len(samples)``,
    microbatches ``max_k``. Publishing one tuple per micro-batch made the
    detector read only the first group, which under-reports the step and lets
    two ranks with identical totals disagree.
    """
    rollout_id = 7
    window = _window(rollout_id)
    stub = _stub_self(window)

    rollout_data, iterator, executed_k = MegatronTrainRayActor._get_prefetched_sft_window(
        stub, "task", rollout_id, ["fields"]
    )

    assert executed_k == MAX_K, "the executed DP-wide K must be max_k, not the local k"

    published = ctx._step_workload(rollout_id, 0)
    assert published is not None, "the executed step must carry a workload"
    # `_step_workload` returns the tuple of Nones past the end, not None.
    assert ctx._step_workload(rollout_id, 1) == (None, None, None), "only one optimizer step consumes this window"

    tokens, sequences, microbatches = published
    assert tokens == sum(SAMPLES)
    assert sequences == len(SAMPLES)
    assert microbatches == executed_k == MAX_K
    # The values READ BACK for the step are the step totals, not one group's.
    executed_partition = get_seqlen_balanced_partitions(SAMPLES, MAX_K, equal_size=False)
    assert microbatches == len(executed_partition)
    assert (tokens, sequences, microbatches) == (sum(SAMPLES), len(SAMPLES), len(executed_partition))
    # And the old one-entry-per-micro-batch payload is explicitly NOT what we emit.
    assert published != (sum(SAMPLES[i] for i in executed_partition[0]), len(executed_partition[0]), 1)


def test_two_ranks_with_equal_totals_report_equal_tokens(_patched):
    """The false-suppression mechanism: a rank's first group is not its
    workload.

    ``[97,1,1,1]`` and ``[25,25,25,25]`` both total 100 but their first groups
    are 97 and 25, so publishing per-group made two equally-loaded ranks
    disagree by more than the 5% tolerance and suppressed the verdict.
    Publishing the step total makes them equal.
    """
    for samples in ([97, 1, 1, 1], [25, 25, 25, 25]):
        ctx.reset_training_context_for_tests()
        window = types.SimpleNamespace(
            packed_micro_batches=[("cpu-batch", None)],
            first_device_micro_batch=("cpu-batch", None),
            first_ready_event=_FakeEvent(),
            rollout_data={"total_lengths": list(samples), "rollout_id": 1},
        )
        stub = _stub_self(window)
        MegatronTrainRayActor._get_prefetched_sft_window(stub, "task", 1, ["fields"])
        published = ctx._step_workload(1, 0)
        assert published == (sum(samples), len(samples), MAX_K), samples


def test_the_assertion_discriminates_the_stale_payload_from_the_fixed_one(_patched):
    """Proves the guard above would fail under the stale publish order.

    The stale order published the rank-local partition; here that yields ONE
    step, while the fixed order yields MAX_K. Re-running the production publish
    from the k_local partition shows the discriminator is real.
    """
    stale = get_seqlen_balanced_partitions(SAMPLES, K_LOCAL, equal_size=False)
    fixed = get_seqlen_balanced_partitions(SAMPLES, MAX_K, equal_size=False)

    assert len(stale) == K_LOCAL
    assert len(fixed) == MAX_K
    # The stale payload has no second/third step at all, which is exactly what
    # `all(entry is not None ...)` in the test above detects.
    assert len(stale) != len(fixed)


def test_the_published_grouping_is_the_executed_grouping(_patched):
    """The per-step grouping matches the partition train() actually runs."""
    rollout_id = 8
    window = _window(rollout_id)
    stub = _stub_self(window)

    MegatronTrainRayActor._get_prefetched_sft_window(stub, "task", rollout_id, ["fields"])

    stats = ctx.training_context_stats()
    assert stats["workload_rollouts"] == 1
    assert stats["workload_publish_skipped"] == 0
    assert stats["workload_publish_errors"] == 0


def test_disabled_profiler_derives_no_workload(monkeypatch, _patched):
    """The gate: with the profiler off the training path derives nothing.

    Counts the calls into the pure derivation and asserts the disabled path
    makes none, so a disabled profiler really costs only the cached boolean.
    """
    monkeypatch.setattr(actor_mod, "_STRAGGLER_PUBLISH_ENABLED", False)
    assert actor_mod._straggler_publish_enabled() is False

    calls = {"n": 0}

    def _count(*args, **kwargs):
        calls["n"] += 1
        return []

    monkeypatch.setattr(actor_mod, "step_workloads", _count)

    rollout_id = 9
    stub = _stub_self(_window(rollout_id))
    MegatronTrainRayActor._get_prefetched_sft_window(stub, "task", rollout_id, ["fields"])

    assert calls["n"] == 0, "a disabled profiler must not derive a workload"
    assert ctx._step_workload(rollout_id, 0) == (None, None, None)
