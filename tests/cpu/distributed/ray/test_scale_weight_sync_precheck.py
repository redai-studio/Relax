# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the independent NCCL scale-out precheck."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


try:
    from relax.backends.sglang.sglang_engine import SGLangEngine
    from relax.utils.scale_utils import _scale_weight_sync_precheck_fingerprints_match

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import AwaitableValue, create_test_manager, make_engine_group, make_mock_engine, make_rollout_server


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _fingerprint(*, ib_disable="1", ifname="eth0", hca=None, gid=None):
    return {
        "nccl_ib_disable": ib_disable,
        "nccl_socket_ifname": ifname,
        "nccl_ib_hca": hca,
        "nccl_ib_gid_index": gid,
    }


def _fp_result(**kwargs):
    """A probe result wrapping just an env fingerprint (for the match
    helper)."""
    return {"fingerprint": _fingerprint(**kwargs)}


def _probe_result(*, success=True, category=None, error_type=None, error=None, ib_disable="1", ifname="eth0"):
    return {
        "success": success,
        "category": category,
        "error_type": error_type,
        "error": error,
        "fingerprint": _fingerprint(ib_disable=ib_disable, ifname=ifname),
        "results": [],
    }


def test_env_fingerprint_rejects_ib_disable_asymmetry():
    """Only NCCL_IB_DISABLE (IB vs socket) is a guaranteed transport-mode
    incompatibility -> the sole Stage-1 hard reject before the probe."""
    matches, category = _scale_weight_sync_precheck_fingerprints_match(
        _fp_result(ib_disable="0"), _fp_result(ib_disable="1")
    )
    assert matches is False
    assert category == "nccl_ib_disable_mismatch"


def test_env_fingerprint_matching_passes():
    matches, category = _scale_weight_sync_precheck_fingerprints_match(
        _fp_result(ib_disable="1", ifname="eth0"), _fp_result(ib_disable="1", ifname="eth0")
    )
    assert matches is True
    assert category is None


def test_env_fingerprint_ib_disable_normalized_no_false_mismatch():
    """NCCL disables IB only on a nonzero int, so unset / "" / "0" all mean IB
    enabled and must compare equal -- an unset seed vs an explicit "0" new node
    is NOT a transport mismatch (would otherwise hard-reject a compatible node
    before the probe)."""
    for seed_ib, new_ib in ((None, "0"), (None, ""), ("0", ""), ("0", "0")):
        matches, category = _scale_weight_sync_precheck_fingerprints_match(
            _fp_result(ib_disable=seed_ib), _fp_result(ib_disable=new_ib)
        )
        assert matches is True, f"{seed_ib!r} vs {new_ib!r} should not mismatch"
        assert category is None
    # A genuine IB-on vs IB-off difference is still a hard reject.
    matches, category = _scale_weight_sync_precheck_fingerprints_match(
        _fp_result(ib_disable=None), _fp_result(ib_disable="1")
    )
    assert matches is False
    assert category == "nccl_ib_disable_mismatch"


def test_non_mode_env_asymmetry_is_advisory_and_defers_to_probe():
    """HCA / GID index / socket ifname asymmetries are advisory (not hard
    rejected); the real NCCL probe decides connectivity in practice."""
    # IB HCA differs
    matches, category = _scale_weight_sync_precheck_fingerprints_match(
        _fp_result(ib_disable="0", hca="mlx5_1"), _fp_result(ib_disable="0", hca="mlx5_2")
    )
    assert matches is True and category is None
    # IB GID index differs
    matches, category = _scale_weight_sync_precheck_fingerprints_match(
        _fp_result(ib_disable="0", gid="3"), _fp_result(ib_disable="0", gid="5")
    )
    assert matches is True and category is None
    # socket ifname differs, even both on socket
    matches, category = _scale_weight_sync_precheck_fingerprints_match(
        _fp_result(ib_disable="1", ifname="eth0"), _fp_result(ib_disable="1", ifname="ens5")
    )
    assert matches is True and category is None


@pytest.mark.asyncio
async def test_unsupported_topology_skips_precheck():
    """P1-b: a multi-node TP topology (tp_size != local_gpu_count) makes the
    sglang-side probe return ``unsupported_topology``. The real direct sync
    handles multi-node, so the manager must SKIP the precheck (return
    ``(True, None)``) rather than fail-close the whole scale-out."""
    seed = make_mock_engine(url="http://seed:1", weight_version="v1")
    new = make_mock_engine()
    seed.run_scale_weight_sync_precheck.remote.return_value = AwaitableValue(
        _probe_result(success=False, category="unsupported_topology")
    )
    new.run_scale_weight_sync_precheck.remote.return_value = AwaitableValue(_probe_result())
    manager = create_test_manager()

    result = await manager._run_scale_weight_sync_precheck(
        seed,
        [new],
        master_address="10.0.0.1",
        tp_size=16,
        timeout=60,
    )

    assert result.success is True
    assert result.reason is None


def test_actor_probe_default_memory_floor_is_512mib():
    """P1-c: SGLang reserves ~85-90% VRAM, so a healthy engine often has <2 GiB
    free even though a probe only needs a CUDA context + tiny NCCL buffers. With
    the default floor lowered to 512 MiB, an engine with ~1 GiB free must launch
    probes instead of returning ``insufficient_gpu_memory``."""
    engine = object.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.base_gpu_id = 0
    engine.num_gpus_per_engine = 2
    engine.args = SimpleNamespace(num_gpus_per_node=8)
    created = []

    def _popen(command, **kwargs):
        process = _CompletedProcess(command, **kwargs)
        created.append(process)
        return process

    with (
        patch("relax.backends.sglang.sglang_engine.subprocess.Popen", side_effect=_popen),
        patch("torch.cuda.mem_get_info", return_value=(1 * 1024**3, 80 * 1024**3)),
        patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False),
    ):
        # ensure no lingering override from the environment
        import os as _os

        _os.environ.pop("RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MIN_FREE_BYTES", None)
        result = engine.run_scale_weight_sync_precheck("10.0.0.1", "18000,18001", 0, "mem-floor", 2, 10)

    assert result["success"] is True
    assert result["category"] is None
    assert len(created) == 2
    assert all(item["required_free_bytes"] == 512 * 1024**2 for item in result["memory"])


def test_actor_probe_reaps_and_cleans_up_on_unexpected_exception():
    """P1-d: an unexpected exception after launch (e.g. ``open`` failing during
    result collection) must not leak running child processes (CUDA context +
    NCCL group on the LIVE gpu) or the tempdir. The post-launch body is wrapped
    in try/finally so every process is terminated and the tempdir cleaned."""
    engine = object.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.base_gpu_id = 0
    engine.num_gpus_per_engine = 2
    engine.args = SimpleNamespace(num_gpus_per_node=8)
    created = []
    created_dirs = []

    def _popen(command, **kwargs):
        process = _CompletedProcess(command, **kwargs)
        created.append(process)
        return process

    real_open = open

    def _raising_open(path, *args, **kwargs):
        mode = kwargs.get("mode") or (args[0] if args else "r")
        # fail only the result-collection read of a rank log, not the write
        if "rank-" in str(path) and "w" not in mode:
            raise RuntimeError("injected collection failure")
        return real_open(path, *args, **kwargs)

    import tempfile as _tempfile

    real_tmpdir = _tempfile.TemporaryDirectory

    def _spy_tmpdir(*args, **kwargs):
        directory = real_tmpdir(*args, **kwargs)
        original_cleanup = directory.cleanup
        directory.cleanup = MagicMock(side_effect=original_cleanup)
        created_dirs.append(directory)
        return directory

    with (
        patch("relax.backends.sglang.sglang_engine.subprocess.Popen", side_effect=_popen),
        patch("relax.backends.sglang.sglang_engine.tempfile.TemporaryDirectory", side_effect=_spy_tmpdir),
        patch("relax.utils.scale_utils._terminate_probe_process") as terminate,
        patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 80 * 1024**3)),
        patch("builtins.open", side_effect=_raising_open),
        patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}),
    ):
        with pytest.raises(RuntimeError, match="injected collection failure"):
            engine.run_scale_weight_sync_precheck(
                master_address="10.0.0.1",
                ports="18000,18001",
                group_rank=0,
                run_token="reap",
                tp_size=2,
                timeout_secs=10,
            )

    # every started child is terminated on the failure path
    assert terminate.call_count == len(created) == 2
    # the tempdir is always cleaned up
    assert len(created_dirs) == 1
    created_dirs[0].cleanup.assert_called()


@pytest.mark.asyncio
async def test_env_mismatch_rejects_before_launching_probe():
    """Stage 1: a seed/new NCCL env asymmetry is rejected by the cheap
    fingerprint gate WITHOUT ever launching the NCCL probe."""
    seed = make_mock_engine(url="http://seed:1", weight_version="v1")
    new = make_mock_engine()
    seed.get_scale_weight_sync_transport_fingerprint.remote.return_value = AwaitableValue(_fingerprint(ib_disable="0"))
    new.get_scale_weight_sync_transport_fingerprint.remote.return_value = AwaitableValue(_fingerprint(ib_disable="1"))
    manager = create_test_manager()

    result = await manager._run_scale_weight_sync_precheck(
        seed, [new], master_address="10.0.0.1", tp_size=2, timeout=60
    )

    assert result.success is False
    assert "NCCL transport mismatch" in str(result.reason)
    # The heavy NCCL probe is never launched.
    seed.run_scale_weight_sync_precheck.remote.assert_not_called()
    new.run_scale_weight_sync_precheck.remote.assert_not_called()


@pytest.mark.asyncio
async def test_probe_failure_blocks_real_modelrunner_sync(patch_async_helpers):
    seed = make_mock_engine(url="http://seed:1", weight_version="v1")
    new = make_mock_engine()
    seed.run_scale_weight_sync_precheck.remote.return_value = AwaitableValue(
        _probe_result(success=False, category="probe_failed", error_type="DistBackendError", error="wrong type 3 != 4")
    )
    group = make_engine_group(engines=[seed])
    manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})

    ok = await manager._sync_weights_from_seed_engine(
        [new],
        timeout=60,
        model_name="default",
        run_precheck=True,
    )

    assert ok is False
    assert seed.run_scale_weight_sync_precheck.remote.call_count == 1
    assert new.run_scale_weight_sync_precheck.remote.call_count == 1
    seed.init_weights_send_group_for_remote_instance.remote.assert_not_called()
    new.init_weights_send_group_for_remote_instance.remote.assert_not_called()
    new.continue_generation.remote.assert_not_called()


@pytest.mark.asyncio
async def test_launch_transient_precheck_has_one_bounded_retry():
    seed = make_mock_engine(url="http://seed:1", weight_version="v1")
    new = make_mock_engine()
    seed.run_scale_weight_sync_precheck.remote.side_effect = [
        AwaitableValue(_probe_result(success=False, category="launch_transient")),
        AwaitableValue(_probe_result()),
    ]
    new.run_scale_weight_sync_precheck.remote.side_effect = [
        AwaitableValue(_probe_result(success=False, category="launch_transient")),
        AwaitableValue(_probe_result()),
    ]
    manager = create_test_manager()

    result = await manager._run_scale_weight_sync_precheck(
        seed,
        [new],
        master_address="10.0.0.1",
        tp_size=2,
        timeout=60,
    )

    assert result.success is True
    assert result.reason is None
    assert seed.run_scale_weight_sync_precheck.remote.call_count == 2
    assert new.run_scale_weight_sync_precheck.remote.call_count == 2


@pytest.mark.asyncio
async def test_probe_failure_surfaces_raw_error_and_does_not_retry():
    """A probe that ran and failed is not retried, and the raw NCCL exception
    is propagated so the user sees the real root cause."""
    seed = make_mock_engine(url="http://seed:1", weight_version="v1")
    new = make_mock_engine()
    seed.run_scale_weight_sync_precheck.remote.return_value = AwaitableValue(
        _probe_result(success=False, category="probe_failed", error_type="DistBackendError", error="wrong type 3 != 4")
    )
    manager = create_test_manager()

    result = await manager._run_scale_weight_sync_precheck(
        seed, [new], master_address="10.0.0.1", tp_size=2, timeout=60
    )

    assert result.success is False
    assert "wrong type 3 != 4" in str(result.reason)
    # Not retryable -> probed exactly once.
    assert seed.run_scale_weight_sync_precheck.remote.call_count == 1


@pytest.mark.asyncio
async def test_prelaunch_check_message_is_surfaced():
    """Pre-launch checks report their reason in ``message`` (not ``error``);
    that clear reason must reach the failure surface too.

    This is also how a non-NCCL accelerator surfaces (the GPU-memory query
    fails before launch).
    """
    seed = make_mock_engine(url="http://seed:1", weight_version="v1")
    new = make_mock_engine()
    seed.run_scale_weight_sync_precheck.remote.return_value = AwaitableValue(
        {
            "success": False,
            "category": "memory_check_failed",
            "message": "failed to query GPU memory: AssertionError: no CUDA device",
            "results": [],
        }
    )
    manager = create_test_manager()

    result = await manager._run_scale_weight_sync_precheck(
        seed, [new], master_address="10.0.0.1", tp_size=2, timeout=60
    )

    assert result.success is False
    assert "failed to query GPU memory" in str(result.reason)


@pytest.mark.asyncio
async def test_precheck_ports_use_independent_bounded_window():
    seed = make_mock_engine()
    manager = create_test_manager()

    def alloc(start_port=10000, consecutive=1, max_port=None):
        assert manager._SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE <= start_port < manager._WEIGHT_SYNC_PORT_BASE
        assert max_port == manager._SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX - 1
        return AwaitableValue(("10.0.0.1", start_port))

    seed._get_current_node_ip_and_free_port.remote.side_effect = alloc
    first = await manager._allocate_scale_weight_sync_precheck_ports(seed, tp_size=8)
    second = await manager._allocate_scale_weight_sync_precheck_ports(seed, tp_size=8)

    assert set(first).isdisjoint(second)
    assert all(
        manager._SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE <= int(port) < manager._WEIGHT_SYNC_PORT_BASE for port in first
    )
    assert all(
        manager._SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE <= int(port) < manager._WEIGHT_SYNC_PORT_BASE for port in second
    )


class _CompletedProcess:
    _next_pid = 100

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.returncode = 0
        rank = int(self.command[self.command.index("--rank") + 1])
        kwargs["stdout"].write("NCCL INFO NET/Socket : Using eth0\n")
        kwargs["stdout"].write(json.dumps({"success": True, "rank": rank}) + "\n")
        kwargs["stdout"].flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class _HangingProcess(_CompletedProcess):
    def __init__(self, command, **kwargs):
        super().__init__(command, **kwargs)
        self.returncode = None

    def poll(self):
        return self.returncode


def test_actor_probe_explicitly_limits_child_visible_devices():
    engine = object.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.base_gpu_id = 4
    engine.num_gpus_per_engine = 2
    engine.args = SimpleNamespace(num_gpus_per_node=8)
    created = []

    def _popen(command, **kwargs):
        process = _CompletedProcess(command, **kwargs)
        created.append(process)
        return process

    with (
        patch("relax.backends.sglang.sglang_engine.subprocess.Popen", side_effect=_popen),
        patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 80 * 1024**3)),
        patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}),
    ):
        result = engine.run_scale_weight_sync_precheck(
            master_address="10.0.0.1",
            ports="18000,18001",
            group_rank=0,
            run_token="run",
            tp_size=2,
            timeout_secs=10,
        )

    assert result["success"] is True
    assert all(
        process.command[:3] == [sys.executable, "-m", "relax.backends.sglang._scale_weight_sync_precheck"]
        for process in created
    )
    assert [process.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for process in created] == ["4,5", "4,5"]
    device_ids = [process.command[process.command.index("--device-id") + 1] for process in created]
    assert device_ids == ["0", "1"]
    assert all(process.kwargs["start_new_session"] is True for process in created)
    assert all(process.kwargs["stdout"] is not subprocess.PIPE for process in created)


def test_actor_probe_large_logs_do_not_use_pipe_or_deadlock():
    engine = object.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.base_gpu_id = 0
    engine.num_gpus_per_engine = 1
    engine.args = SimpleNamespace(num_gpus_per_node=8)

    class _LargeLogProcess(_CompletedProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            kwargs["stdout"].write("x" * (2 * 1024 * 1024))
            kwargs["stdout"].flush()

    with (
        patch("relax.backends.sglang.sglang_engine.subprocess.Popen", _LargeLogProcess),
        patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 80 * 1024**3)),
        patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}),
    ):
        result = engine.run_scale_weight_sync_precheck("10.0.0.1", "18000", 0, "large", 1, 10)

    assert result["success"] is True


def test_actor_probe_low_memory_does_not_launch_children():
    engine = object.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.base_gpu_id = 0
    engine.num_gpus_per_engine = 2
    engine.args = SimpleNamespace(num_gpus_per_node=8)

    with (
        patch("relax.backends.sglang.sglang_engine.subprocess.Popen") as popen,
        patch("torch.cuda.mem_get_info", return_value=(1024, 80 * 1024**3)),
        patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}),
    ):
        result = engine.run_scale_weight_sync_precheck("10.0.0.1", "18000,18001", 0, "low-memory", 2, 10)

    assert result["success"] is False
    assert result["category"] == "insufficient_gpu_memory"
    assert len(result["memory"]) == 2
    popen.assert_not_called()


def test_actor_probe_timeout_terminates_and_reaps_every_child():
    engine = object.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.base_gpu_id = 0
    engine.num_gpus_per_engine = 2
    engine.args = SimpleNamespace(num_gpus_per_node=8)
    created = []

    def _popen(command, **kwargs):
        process = _HangingProcess(command, **kwargs)
        created.append(process)
        return process

    def _terminate(process):
        process.returncode = -9

    with (
        patch("relax.backends.sglang.sglang_engine.subprocess.Popen", side_effect=_popen),
        patch("relax.utils.scale_utils._terminate_probe_process", side_effect=_terminate) as terminate,
        patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 80 * 1024**3)),
        patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}),
    ):
        result = engine.run_scale_weight_sync_precheck(
            master_address="10.0.0.1",
            ports="18000,18001",
            group_rank=0,
            run_token="timeout",
            tp_size=2,
            timeout_secs=0,
        )

    assert result["success"] is False
    assert result["category"] == "timeout"
    # Timed-out children are terminated in-loop (fast join) AND again in the
    # finally safety-net; both calls are idempotent on an already-dead process.
    assert terminate.call_count == 4
    assert all(process.returncode == -9 for process in created)


def test_actor_probe_unreapable_child_does_not_block_actor():
    """A probe wedged in an uninterruptible (D) state can survive SIGKILL, so
    the result-collection join must be bounded; otherwise it hangs the actor
    thread forever.

    The call must still return with a ``timeout`` verdict.
    """
    engine = object.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.base_gpu_id = 0
    engine.num_gpus_per_engine = 2
    engine.args = SimpleNamespace(num_gpus_per_node=8)
    created = []

    class _UnreapableProcess(_HangingProcess):
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd=self.command, timeout=timeout or 0)

    def _popen(command, **kwargs):
        process = _UnreapableProcess(command, **kwargs)
        created.append(process)
        return process

    with (
        patch("relax.backends.sglang.sglang_engine.subprocess.Popen", side_effect=_popen),
        patch("relax.utils.scale_utils._terminate_probe_process"),
        patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 80 * 1024**3)),
        patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}),
    ):
        result = engine.run_scale_weight_sync_precheck(
            master_address="10.0.0.1",
            ports="18000,18001",
            group_rank=0,
            run_token="unreapable",
            tp_size=2,
            timeout_secs=0,
        )

    assert result["success"] is False
    assert result["category"] == "timeout"
    assert len(created) == 2


@pytest.mark.asyncio
async def test_precheck_seed_failure_releases_lock(patch_async_helpers):
    seed = make_mock_engine(url="http://seed:1", weight_version="v1")
    new = make_mock_engine()

    async def _raise_seed_failure():
        raise TimeoutError("independent probe actor timed out")

    seed.run_scale_weight_sync_precheck.remote.return_value = _raise_seed_failure()
    group = make_engine_group(engines=[seed])
    manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})

    ok = await manager._sync_weights_from_seed_engine([new], timeout=60, run_precheck=True)

    assert ok is False
    assert manager._is_weight_updating is False
    manager._weight_sync_lock.release.remote.assert_called_once()
    new.continue_generation.remote.assert_not_called()


@pytest.mark.asyncio
async def test_probe_failure_logs_full_detail_to_stable_log():
    """The concise reason reaches the TUI, but the full NCCL detail
    (fingerprints.

    + per-rank log tails) is logged to the (stable) coordinator log —
    searchable by run_token — rather than dumped to a separate file.
    """
    seed = make_mock_engine(url="http://seed:1", weight_version="v1")
    new = make_mock_engine()
    seed.run_scale_weight_sync_precheck.remote.return_value = AwaitableValue(
        {
            "success": False,
            "category": "probe_failed",
            "error_type": "DistBackendError",
            "error": "wrong type 3 != 4",
            "fingerprint": _fingerprint(),
            "results": [
                {"local_rank": 0, "category": "probe_failed", "returncode": 1, "log_tail": "gpu-to-net path missing"}
            ],
        }
    )
    manager = create_test_manager()

    with patch("relax.distributed.ray.rollout.logger") as mock_logger:
        result = await manager._run_scale_weight_sync_precheck(
            seed, [new], master_address="10.0.0.1", tp_size=1, timeout=60
        )

    assert result.success is False
    # Concise reason to the TUI/error_message.
    assert "wrong type 3 != 4" in str(result.reason)
    # Full NCCL detail logged (searchable) to the stable coordinator log.
    logged = " ".join(str(call.args[0]) for call in mock_logger.error.call_args_list if call.args)
    assert "gpu-to-net path missing" in logged
