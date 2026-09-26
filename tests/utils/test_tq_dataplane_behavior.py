# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real local SimpleStorage connection and data-plane contracts."""

from __future__ import annotations

import faulthandler
import importlib.util
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO

import pytest
import torch

from relax.utils.tq.correctness import diff_digests, leaf_digests, payload_rows


def _has_real_submodule(dotted: str) -> bool:
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not (_has_real_submodule("transfer_queue.storage") and importlib.util.find_spec("ray")),
    reason="requires real TransferQueue, Ray, and a startable local CPU cluster",
)
_TQ_ACTOR = "TransferQueueController"
_TQ_NS = "transfer_queue"
_PG_REAP_TIMEOUT_SECONDS = 15.0
# A wait that never returns must fail fast with thread stacks instead of burning
# the CI job's 60-minute timeout.
_TEST_DEADLINE_SECONDS = float(os.environ.get("RELAX_TQ_TEST_DEADLINE_SECONDS", "600"))
_STACK_DUMP_NAME = "hung_thread_stacks.log"


def _wait_controller_gone(timeout: float = 20.0) -> bool:
    import ray

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get_actor(_TQ_ACTOR, namespace=_TQ_NS)
        except ValueError:
            return True
        time.sleep(0.4)
    return False


def _force_kill_controller() -> None:
    import ray

    try:
        ray.kill(ray.get_actor(_TQ_ACTOR, namespace=_TQ_NS))
    except ValueError:
        pass


def _placement_group_states() -> dict[str, str]:
    import ray

    return {pg_id: str(info.get("state")) for pg_id, info in ray.util.placement_group_table().items()}


def _reserved_placement_groups(pg_ids: set[str]) -> set[str]:
    """Placement groups from ``pg_ids`` that still hold their CPU bundles."""
    return {pg_id for pg_id, state in _placement_group_states().items() if pg_id in pg_ids and state == "CREATED"}


def _reap_placement_groups(pg_ids: set[str]) -> None:
    """Remove placement groups ``tq.init`` created and wait for their CPUs.

    SimpleStorage's normal ``tq.close()`` path does not remove these groups.
    Reclaim them before reinitializing TQ to avoid exhausting cluster CPUs.
    """
    import ray
    from ray._raylet import PlacementGroupID
    from ray.util.placement_group import PlacementGroup

    for pg_id in sorted(pg_ids):
        ray.util.remove_placement_group(PlacementGroup(PlacementGroupID.from_hex(pg_id)))

    deadline = time.time() + _PG_REAP_TIMEOUT_SECONDS
    while time.time() < deadline and _reserved_placement_groups(pg_ids):
        time.sleep(0.2)
    pending = _reserved_placement_groups(pg_ids)
    if pending:
        raise RuntimeError(
            f"placement groups {sorted(pending)} still hold CPUs "
            f"{_PG_REAP_TIMEOUT_SECONDS:.0f}s after removal; the next tq.init would starve"
        )


def _stack_dump_stream(request: pytest.FixtureRequest) -> TextIO:
    """Choose a stack-dump destination that survives a hard exit in CI."""
    override = os.environ.get("RELAX_TEST_STACK_DUMP", "").strip()
    if override:
        return open(override, "a")
    xml_path = getattr(request.config.option, "xmlpath", None)
    if xml_path:
        return open(os.path.join(os.path.dirname(os.path.abspath(xml_path)), _STACK_DUMP_NAME), "a")
    return sys.stderr


@contextmanager
def _ray_wait_deadline(request: pytest.FixtureRequest) -> Iterator[None]:
    """Dump thread stacks and exit if a Ray operation hangs.

    In CI, write beside ``--junitxml`` so the artifact survives a hard exit
    without relying on pytest's captured stderr.
    """
    stream = _stack_dump_stream(request)
    faulthandler.dump_traceback_later(_TEST_DEADLINE_SECONDS, exit=True, file=stream)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        if stream is not sys.stderr:
            stream.close()


@pytest.fixture(autouse=True)
def _real_ray_wait_deadline(request: pytest.FixtureRequest) -> Iterator[None]:
    with _ray_wait_deadline(request):
        yield


def _payload(samples: int, fields: list[str], columns: int, seed: int = 0) -> Any:
    from tensordict import TensorDict

    generator = torch.Generator().manual_seed(seed)
    return TensorDict(
        {field: torch.randn(samples, columns, generator=generator) for field in fields},
        batch_size=[samples],
    )


def _multimodal_payload(samples: int) -> dict[str, Any]:
    grids = ((1, 58, 64), (1, 34, 64), (1, 64, 64), (1, 26, 40))
    generator = torch.Generator().manual_seed(20260813)
    multimodal, tokens = [], []
    for index in range(samples):
        temporal, height, width = grids[index % len(grids)]
        multimodal.append(
            {
                "pixel_values": torch.randn(temporal * height * width, 1536, generator=generator),
                "image_grid_thw": torch.tensor([[temporal, height, width]], dtype=torch.int64),
            }
        )
        tokens.append(torch.randint(0, 151_000, (512 + 173 * index,), generator=generator).tolist())
    return {"tokens": tokens, "multimodal_train_inputs": multimodal}


def _get(client: Any, partition: str, fields: list[str], size: int) -> Any:
    meta = client.get_meta(
        data_fields=fields,
        batch_size=size,
        partition_id=partition,
        mode="fetch",
        task_name=partition,
    )
    return meta, client.get_data(meta)


@pytest.fixture(scope="module")
def _ray_cluster(request: pytest.FixtureRequest) -> Iterator[None]:
    import ray

    # Module setup/teardown run outside the per-test watchdog. Arm separate
    # deadlines so cluster lifecycle calls are covered without timing the module.
    try:
        with _ray_wait_deadline(request):
            ray.init(ignore_reinit_error=True, logging_level="ERROR")
        yield
    finally:
        with _ray_wait_deadline(request):
            ray.shutdown()


@pytest.fixture
def tq_factory(_ray_cluster):
    import transfer_queue as tq
    from omegaconf import OmegaConf
    from transfer_queue import GRPOGroupNSampler

    leaked: set[str] = set()

    def reinit(capacity: int | None = 1024):
        tq.close()
        if not _wait_controller_gone():
            _force_kill_controller()
            assert _wait_controller_gone()
        # Each lifecycle owns a 1-CPU placement group that close() never releases,
        # so hand the previous one back before reserving a new one.
        _reap_placement_groups(leaked)
        leaked.clear()
        before = set(_placement_group_states())
        conf = OmegaConf.create(
            {
                "controller": {"sampler": GRPOGroupNSampler(n_samples_per_prompt=1), "polling_mode": True},
                "backend": {"SimpleStorage": {"total_storage_size": capacity, "num_data_storage_units": 1}},
            },
            flags={"allow_objects": True},
        )
        try:
            tq.init(conf=conf)
        finally:
            # Initialization can reserve CPUs before raising an exception.
            leaked.update(set(_placement_group_states()) - before)
        return tq.get_client()

    yield reinit
    try:
        tq.close()
    finally:
        try:
            _wait_controller_gone()
            _force_kill_controller()
            _wait_controller_gone()
        finally:
            _reap_placement_groups(leaked)


@pytest.mark.parametrize(
    ("fields", "columns", "samples", "seed"),
    [(["a", "b"], 8, 4, 0), (["img", "txt", "mask"], 16, 8, 42), (["pixel_values"], 1176, 4, 7)],
    ids=["connection", "multi-field", "multimodal-width"],
)
def test_dense_round_trip_is_byte_exact(tq_factory, fields: list[str], columns: int, samples: int, seed: int) -> None:
    client = tq_factory()
    payload = _payload(samples, fields, columns, seed)
    client.put(payload, partition_id="dense")
    _, received = _get(client, "dense", fields, samples)
    assert set(fields) <= set(received.keys())
    for field in fields:
        expected_rows = [leaf_digests(row) for row in payload_rows(payload[field])]
        actual_rows = [leaf_digests(row) for row in payload_rows(received[field])]
        assert len(actual_rows) == len(expected_rows)
        assert all(
            not diff_digests(expected, actual) for expected, actual in zip(expected_rows, actual_rows, strict=True)
        )


def test_backpressure_fails_without_publishing_data(tq_factory) -> None:
    client = tq_factory(capacity=4)
    with pytest.raises(RuntimeError, match="capacity"):
        client.put(_payload(8, ["a"], 4), partition_id="backpressure")
    meta, _ = _get(client, "backpressure", ["a"], 8)
    assert getattr(meta, "size", None) == 0


def test_simple_storage_unbounded_capacity_accepts_rows(tq_factory) -> None:
    """A None capacity permits writing and reading physical rows."""
    client = tq_factory(capacity=None)
    payload = _payload(4, ["a"], 4)
    client.put(payload, partition_id="unbounded")
    meta, received = _get(client, "unbounded", ["a"], 4)
    assert meta.size == 4
    expected_rows = payload_rows(payload["a"])
    actual_rows = payload_rows(received["a"])
    assert len(actual_rows) == len(expected_rows)
    for expected, actual in zip(expected_rows, actual_rows, strict=True):
        assert not diff_digests(leaf_digests(expected), leaf_digests(actual))


def test_empty_get_returns_without_hanging(tq_factory) -> None:
    meta, data = _get(tq_factory(), "empty", ["a"], 4)
    assert getattr(meta, "size", None) == 0 and list(data.keys()) == []


def test_repeat_put_stays_bounded_and_uncorrupted(tq_factory) -> None:
    client = tq_factory()
    first, second = _payload(4, ["a"], 4, 1), _payload(4, ["a"], 4, 2)
    client.put(first, partition_id="repeat")
    client.put(second, partition_id="repeat")
    meta, received = _get(client, "repeat", ["a"], 4)
    candidates = [leaf_digests(row) for row in [*payload_rows(first["a"]), *payload_rows(second["a"])]]
    assert getattr(meta, "size", None) == 4
    assert all(leaf_digests(row) in candidates for row in payload_rows(received["a"]))


def test_clear_partition_and_reinit_are_isolated(tq_factory) -> None:
    client = tq_factory()
    client.put(_payload(4, ["a"], 4), partition_id="cleanup")
    assert getattr(_get(client, "cleanup", ["a"], 4)[0], "size", None) == 4
    client.clear_partition("cleanup")
    assert getattr(_get(client, "cleanup", ["a"], 4)[0], "size", None) == 0
    assert getattr(_get(tq_factory(capacity=16), "cleanup", ["a"], 4)[0], "size", None) == 0


def test_multimodal_non_tensor_full_link_is_byte_exact(tq_factory, record_property) -> None:
    from relax.utils.utils import dict_to_tensordict

    samples = 4
    source = _multimodal_payload(samples)
    expected_mm = [leaf_digests(row) for row in source["multimodal_train_inputs"]]
    expected_tokens = [leaf_digests(torch.tensor(row, dtype=torch.int64)) for row in source["tokens"]]
    batch = dict_to_tensordict({**source, "sample_id": list(range(samples))}, batch_size=samples)
    assert type(batch.get("multimodal_train_inputs")).__name__ == "NonTensorStack"
    record_property("multimodal_payload_source", "synthetic")

    fields = ["sample_id", "tokens", "multimodal_train_inputs"]
    client = tq_factory()
    client.put(batch, partition_id="multimodal")
    _, received = _get(client, "multimodal", fields, samples)
    sample_ids = [int(value) for value in payload_rows(received["sample_id"])]
    assert sorted(sample_ids) == list(range(samples))
    for position, sample_id in enumerate(sample_ids):
        assert not diff_digests(expected_mm[sample_id], leaf_digests(payload_rows(received[fields[2]])[position]))
        assert not diff_digests(expected_tokens[sample_id], leaf_digests(payload_rows(received[fields[1]])[position]))
