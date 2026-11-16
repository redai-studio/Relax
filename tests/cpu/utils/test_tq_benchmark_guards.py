# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for the retained cross-node acceptance benchmark."""

from __future__ import annotations

import csv
import io
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmarks import tq_cross_node_bench as bench


def _argv(protocol: str, *extra: str) -> list[str]:
    return [
        "bench",
        "--protocol",
        protocol,
        "--consumer-node-id",
        "node",
        "--tcp-device",
        "eth0",
        "--csv",
        "x",
        *extra,
    ]


def _raise(error: BaseException) -> None:
    raise error


def test_simple_and_multimodal_profiles_use_the_expected_runtime_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    import transfer_queue

    monkeypatch.setattr(transfer_queue, "GRPOGroupNSampler", lambda **_kwargs: object())
    assert bench.build_conf("simple", master="", device="", segment_gib=1).backend.storage_backend == "SimpleStorage"
    payload = bench.make_multimodal_payload(num_samples=3, total_mib=1)
    column = payload.get("multimodal_train_inputs")
    assert type(column).__name__ == "NonTensorStack"
    rows = column.tolist()
    assert all(set(row) == {"pixel_values", "image_grid_thw"} for row in rows)
    assert len({tuple(row["pixel_values"].shape) for row in rows}) > 1
    digest = bench.field_byte_digests(payload, ["multimodal_train_inputs"])
    rows[1]["pixel_values"][0, 0] += 1
    assert bench.field_byte_digests(payload, ["multimodal_train_inputs"]) != digest


@pytest.mark.parametrize("num_samples", [1, 3])
def test_multimodal_scalar_columns_preserve_shapes_and_strict_digests(num_samples: int) -> None:
    import torch
    from tensordict import TensorDict

    payload = bench.make_multimodal_payload(num_samples=num_samples, total_mib=1)
    fields = ["sample_id", "rewards"]
    expected = bench.field_byte_digests(payload, fields)
    received = TensorDict({}, batch_size=[num_samples])
    for field, dtype in (("sample_id", torch.int64), ("rewards", torch.float32)):
        column = payload.get(field)
        assert column.shape == (num_samples, 1)
        assert column.dtype == dtype
        rows = [row.clone() for row in column.unbind()]
        received.set(field, torch.nested.as_nested_tensor(rows, layout=torch.jagged))
    assert bench.field_byte_digests(received, fields) == expected

    for field in fields:
        original = received.get(field)
        for changed in (
            payload.get(field) + 1,
            payload.get(field).to(torch.float64),
            payload.get(field).squeeze(-1),
        ):
            received.set(field, changed)
            assert bench.field_byte_digests(received, fields)[field] != expected[field]
        received.set(field, original)


@pytest.mark.parametrize("num_samples", [1, 3])
def test_multimodal_scalar_columns_round_trip_with_real_tq_metadata(num_samples: int) -> None:
    import torch
    import transfer_queue
    from tensordict import TensorDict

    # CI's module stub fabricates attributes, including __path__, via __getattr__.
    # Inspect the import spec without invoking that fallback before importing children.
    spec = vars(transfer_queue).get("__spec__")
    if spec is not None and spec.submodule_search_locations is None:
        pytest.skip("Real TransferQueue package required; CI provides a module stub")
    from transfer_queue.metadata import extract_field_schema

    payload = bench.make_multimodal_payload(num_samples=num_samples, total_mib=1)
    fields = ["sample_id", "rewards"]
    schema = extract_field_schema(payload.select(*fields))
    received = TensorDict({}, batch_size=[num_samples])
    for field in fields:
        column = payload.get(field)
        field_meta = schema[field]
        rows = [
            torch.frombuffer(bytearray(row.numpy().tobytes()), dtype=field_meta["dtype"]).reshape(field_meta["shape"])
            for row in column.unbind()
        ]
        received.set(field, torch.nested.as_nested_tensor(rows, layout=torch.jagged))
    assert bench.field_byte_digests(received, fields) == bench.field_byte_digests(payload, fields)


@pytest.mark.parametrize(
    ("protocol", "ib_bytes", "tcp_bytes", "payload_bytes", "expected"),
    [
        ("rdma", 800, 0, 1000, True),
        ("rdma", 799, 0, 1000, False),
        ("rdma", 900, 901, 1000, True),
        ("tcp", 0, 200, 1000, True),
        ("tcp", 0, 199, 1000, False),
        ("tcp", 201, 200, 1000, True),
        ("simple", 0, 1, 1000, True),
        ("simple", 1, 0, 1000, False),
        ("rdma", 1, 0, 4 * 1024**3, False),
        ("tcp", 0, 1, 4 * 1024**3, False),
    ],
)
def test_wire_proof_matrix(protocol: str, ib_bytes: int, tcp_bytes: int, payload_bytes: int, expected: bool) -> None:
    assert bench.wire_is_proven(protocol, ib_bytes, tcp_bytes, payload_bytes) is expected


def test_counter_scope_failure_modes_and_idle_subtraction(tmp_path) -> None:
    for device, port, value in (("rdma0", 1, 11), ("rdma1", 1, 17), ("rdma1", 2, 29)):
        directory = tmp_path / "infiniband" / device / "ports" / str(port) / "counters"
        directory.mkdir(parents=True)
        (directory / "port_rcv_data").write_text(str(value))
    tcp = tmp_path / "net" / "eth0" / "statistics"
    tcp.mkdir(parents=True)
    (tcp / "rx_bytes").write_text("101")
    assert bench.read_counters("eth0", "rdma1", 2, sysfs_root=tmp_path) == {
        "ib:rdma1:2": 29 * 4,
        "tcp:eth0": 101,
    }
    with pytest.raises(ValueError, match="safe device"):
        bench.read_counters("eth0", "*", 1, sysfs_root=tmp_path)
    with pytest.raises(RuntimeError, match="TCP receive counter"):
        bench.read_counters("missing", sysfs_root=tmp_path)
    with pytest.raises(RuntimeError, match="counter set changed"):
        bench._counter_delta({"tcp:x": 1}, {}, "tcp:")
    with pytest.raises(RuntimeError, match="reset or wrapped"):
        bench._counter_delta({"tcp:x": 2}, {"tcp:x": 1}, "tcp:")
    assert bench._subtract_idle_noise(1_100, 100, 1.0, 1.0) == 1_000
    assert bench._subtract_idle_noise(50, 100, 1.0, 1.0) == 0
    with pytest.raises(RuntimeError, match="duration must be positive"):
        bench._throughput_gbs(1_000, 0.0, "put")


@pytest.mark.parametrize(
    "argv",
    [
        _argv("rdma"),
        _argv("tcp", "--master", "master.example:50051", "--device", "rdma0"),
        *[
            _argv("rdma", "--master", "master.example:50051", "--device", device, "--rdma-port", "1")
            for device in ("../rdma0", "*", "rdma*", "rdma?", "[ab]")
        ],
        _argv("tcp", "--master", "master.example:50051", "--rdma-port", "1"),
        _argv("simple", "--master", "master.example:50051"),
    ],
)
def test_cli_rejects_ambiguous_or_unsafe_counter_configuration(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="2"):
        bench.parse_args()


@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["cleanup-ok", "cleanup-fails"])
@pytest.mark.parametrize(
    ("byte_exact", "tcp_bytes", "error_type", "match", "error_kind"),
    [
        (False, 1_000, AssertionError, "byte-exact", "ByteExactMismatch"),
        (True, 0, RuntimeError, "wire proof failed", "WireProofFailed"),
        (True, 1_000, RuntimeError, "partition cleanup failed", ""),
    ],
    ids=["byte-exact", "wire-proof", "valid-measurement"],
)
def test_round_is_flushed_before_cleanup_and_final_status_preserves_gate_errors(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
    byte_exact: bool,
    tcp_bytes: int,
    error_type: type[BaseException],
    match: str,
    error_kind: str,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(bench.ray, "get", lambda value: value)
    consumer = SimpleNamespace(
        sample_idle_counters=SimpleNamespace(remote=lambda _seconds: {"seconds": 1.0, "ib_bytes": 0, "tcp_bytes": 0}),
        begin_round=SimpleNamespace(remote=lambda: ({"tcp:x": 0}, 1.0)),
        fetch=SimpleNamespace(
            remote=lambda *_args: {
                "get_ms": 1.0,
                "round_seconds": 1.0,
                "ib_bytes": 0,
                "tcp_bytes": tcp_bytes,
                "byte_exact": byte_exact,
                "mismatch_fields": [] if byte_exact else ["field"],
            }
        ),
    )

    class Producer:
        def put(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("put")

        def clear_partition(self, _partition: str) -> None:
            events.append("clear")
            rows = list(csv.DictReader(io.StringIO(output.getvalue())))
            assert len(rows) == 1
            assert rows[0]["phase"] == "measurement"
            assert rows[0]["cleanup_status"] == "pending"
            assert rows[0]["status"] == ("fail" if error_kind else "pending")
            assert output.flushes == 1
            if cleanup_fails:
                raise RuntimeError("partition cleanup failed")

    class FlushedOutput(io.StringIO):
        flushes = 0

        def flush(self) -> None:
            self.flushes += 1
            super().flush()

    output = FlushedOutput()
    writer = csv.DictWriter(output, fieldnames=bench.CSV_COLUMNS)
    writer.writeheader()
    raises = bool(error_kind) or cleanup_fails
    with pytest.raises(error_type, match=match) if raises else nullcontext() as excinfo:
        bench._run_round(
            producer=Producer(),
            consumer=consumer,
            payload=SimpleNamespace(batch_size=[1]),
            fields=["field"],
            expected={"field": ()},
            nbytes=1_000,
            protocol="tcp",
            profile="synthetic",
            requested_mib=1,
            run=1,
            writer=writer,
            csv_handle=output,
            provenance={"relax_sha": "a" * 40, "tq_commit": "b" * 40, "mooncake_version": "test"},
        )
    _, row = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert row["phase"] == "final"
    assert row["cleanup_status"] == ("error" if cleanup_fails else "pass")
    assert row["cleanup_error_kind"] == ("RuntimeError" if cleanup_fails else "")
    expected_status = "fail" if error_kind else ("error" if cleanup_fails else "pass")
    expected_error = error_kind or ("RuntimeError" if cleanup_fails else "")
    assert (row["status"], row["error_kind"], row["byte_exact"]) == (expected_status, expected_error, str(byte_exact))
    assert output.flushes == 2
    assert row["mismatch_fields"] == ("" if byte_exact else "field")
    if error_kind:
        assert isinstance(excinfo.value.__cause__, RuntimeError) is cleanup_fails
    assert events == ["put", "clear"]


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("final_write_fails", [False, True])
def test_transfer_error_is_saved_before_cleanup(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool, final_write_fails: bool
) -> None:
    monkeypatch.setattr(bench.ray, "get", lambda value: value)
    write_record = bench._write_csv_record

    def write_or_fail(writer, handle, record) -> None:
        if final_write_fails and record["phase"] == "final":
            raise OSError("CSV write failed")
        write_record(writer, handle, record)

    monkeypatch.setattr(bench, "_write_csv_record", write_or_fail)
    consumer = SimpleNamespace(
        sample_idle_counters=SimpleNamespace(remote=lambda _seconds: {}),
        begin_round=SimpleNamespace(remote=lambda: ({}, 1.0)),
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=bench.CSV_COLUMNS)
    writer.writeheader()

    def clear_partition(_partition: str) -> None:
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        assert len(rows) == 1
        assert (rows[0]["phase"], rows[0]["status"], rows[0]["error_kind"]) == ("measurement", "error", "ValueError")
        if cleanup_fails:
            raise RuntimeError("cleanup failed")

    producer = SimpleNamespace(
        put=lambda *_args, **_kwargs: _raise(ValueError("put failed")),
        clear_partition=clear_partition,
    )
    with pytest.raises(ValueError, match="put failed") as excinfo:
        bench._run_round(
            producer=producer,
            consumer=consumer,
            payload=SimpleNamespace(batch_size=[1]),
            fields=["field"],
            expected={"field": ()},
            nbytes=1_000,
            protocol="tcp",
            profile="synthetic",
            requested_mib=1,
            run=1,
            writer=writer,
            csv_handle=output,
            provenance={},
        )
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    if final_write_fails:
        assert len(rows) == 1
        cause = excinfo.value.__cause__
        assert isinstance(cause.__cause__ if cleanup_fails else cause, OSError)
    else:
        _, final = rows
        assert (final["phase"], final["status"], final["error_kind"]) == ("final", "error", "ValueError")
        assert final["cleanup_status"] == ("error" if cleanup_fails else "pass")
    assert isinstance(excinfo.value.__cause__, RuntimeError) is cleanup_fails


def test_provenance_records_versions_and_rejects_dirty_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    results = iter([SimpleNamespace(returncode=0, stdout=""), SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")])
    monkeypatch.setattr(bench.subprocess, "run", lambda *_args, **_kwargs: next(results))
    direct_url = '{"vcs_info":{"commit_id":"' + "b" * 40 + '"}}'
    monkeypatch.setattr(
        bench.importlib_metadata,
        "distribution",
        lambda _name: SimpleNamespace(read_text=lambda _filename: direct_url),
    )
    monkeypatch.setattr(bench.importlib_metadata, "version", lambda _name: "0.3.test")
    assert bench.collect_provenance(tmp_path) == {
        "relax_sha": "a" * 40,
        "tq_commit": "b" * 40,
        "mooncake_version": "0.3.test",
    }
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=" M tracked-file\n"),
    )
    with pytest.raises(RuntimeError, match="clean tracked Relax checkout"):
        bench.collect_provenance(tmp_path)


@pytest.fixture
def teardown_events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from relax.utils.tq import lifecycle

    events: list[str] = []
    monkeypatch.setattr(bench.ray, "kill", lambda *_args, **_kwargs: events.append("consumer-killed"))
    monkeypatch.setattr(lifecycle, "detach_tq_client", lambda: events.append("producer-detached"))
    monkeypatch.setattr(lifecycle, "close_tq_owner", lambda _owner: events.append("owner-closed"))
    monkeypatch.setattr(bench.ray, "shutdown", lambda: events.append("ray-shutdown"))
    return events


@pytest.mark.parametrize("failure", ["consumer", "owner", "none"])
def test_teardown_is_ordered_and_does_not_touch_unowned_state(
    monkeypatch: pytest.MonkeyPatch, teardown_events: list[str], failure: str
) -> None:
    from relax.utils.tq import lifecycle

    consumer, attached, owner = None, False, None
    expected = ["ray-shutdown"]
    error_match = None
    if failure == "consumer":
        consumer, attached, owner = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: "ref")), True, "owner"
        monkeypatch.setattr(bench.ray, "get", lambda *_args, **_kwargs: _raise(RuntimeError("consumer failed")))

        def fail_owner_cleanup(_owner: Any) -> None:
            teardown_events.append("owner-closed")
            raise RuntimeError("owner failed")

        monkeypatch.setattr(lifecycle, "close_tq_owner", fail_owner_cleanup)
        expected = ["consumer-killed", "producer-detached", "owner-closed", "ray-shutdown"]
        error_match = "consumer failed"
    elif failure == "owner":
        owner = "owner"
        monkeypatch.setattr(lifecycle, "close_tq_owner", lambda _owner: _raise(RuntimeError("owner failed")))
        error_match = "owner failed"
    context = pytest.raises(RuntimeError, match=error_match) if error_match else nullcontext()
    with context:
        bench._teardown_benchmark(consumer, producer_attached=attached, owner=owner)
    assert teardown_events == expected
