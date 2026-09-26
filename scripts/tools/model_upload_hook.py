# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""save-hf post-hooks: async push of HF checkpoints to a model registry.

Two backends live here, both conforming to the framework hook contract in
``relax.backends.megatron.model.save_hf_model``:

    hook(args, hf_path: str, rollout_id: int, *, dtype: str, is_lora: bool) -> None

Both backends enqueue work on a shared bounded ``ThreadPoolExecutor`` and
return immediately, so training is never blocked by upload.

* ``push_to_quicksilver`` — Xiaohongshu-internal Quicksilver model repo
  (``model_tools.push_model``). **Internal only.**
* ``push_to_huggingface`` — public HuggingFace Hub upload
  (``huggingface_hub.HfApi.upload_folder``).

Wire up via CLI:

    --save-hf /path/hf_output/iter_{rollout_id} \\
    --save-hf-post-hook-path scripts.tools.model_upload_hook.push_to_quicksilver
    # or:
    --save-hf-post-hook-path scripts.tools.model_upload_hook.push_to_huggingface

Quicksilver env vars (INTERNAL):

    QS_USER / QS_TOKEN               required
    RELAX_QS_MODEL_NAME              target model name (fallback: args.experiment_name)
    RELAX_QS_ENV                     "prod" (default) | "sandbox"
    RELAX_QS_ZONE                    "cn" (default) | "sg" | "ae"
    RELAX_QS_REGION                  storage region, e.g. "tencent-ap-shanghai"
    RELAX_QS_FRAMEWORK               framework tag (default: "huggingface")
    RELAX_QS_GPU_TYPE                gpu tag (default: "NoLimit")
    RELAX_QS_MAX_UPLOAD_WORKERS      concurrent uploads in flight (default: 2)
    RELAX_QS_UPLOAD_QUEUE_MAX        max in-flight + queued (default: 4); overflow drops
    RELAX_QS_UPLOAD_THREADS          per-upload parallel threads (default: 10)
    RELAX_QS_UPLOAD_INTERVAL         upload every N rollouts (default: 1)
    RELAX_QS_EXPORT_MODEL            "true" -> mark version ready-for-online

HuggingFace env vars:

    HF_TOKEN                         required (or ``HF_HUB_TOKEN``)
    RELAX_HF_REPO_ID                 target repo, e.g. "org/model" (fallback: args.experiment_name)
    RELAX_HF_REPO_PRIVATE            "true" for private repo (default: false)
    RELAX_HF_REVISION                branch/tag to push to (default: "main")
    RELAX_HF_UPLOAD_INTERVAL         upload every N rollouts (default: 1)
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_inflight_lock = threading.Lock()
_inflight_count = 0
_missing_toolkit_warned = False
_stale_cleanup_done: set[str] = set()


def _cleanup_stale_snapshots_once(snapshot_root: str) -> None:
    """Idempotent per-process wrapper around ``_cleanup_stale_snapshots``."""
    if snapshot_root in _stale_cleanup_done:
        return
    _stale_cleanup_done.add(snapshot_root)
    _cleanup_stale_snapshots(snapshot_root)


def _model_tools_available() -> bool:
    """True iff ``quicksilver-toolkit`` (``model_tools``) is importable."""
    return importlib.util.find_spec("model_tools") is not None


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[qs-upload] {key}={raw!r} is not an int; using default {default}")
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_executor() -> ThreadPoolExecutor:
    """Lazy-init a process-wide upload executor bounded by env var."""
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            max_workers = max(1, _env_int("RELAX_QS_MAX_UPLOAD_WORKERS", 2))
            _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="qs-upload")
            _silence_toolkit_log_noise()
            logger.info(f"[qs-upload] initialized executor with max_workers={max_workers}")
    return _executor


def flush(timeout_sec: float | None = None) -> None:
    """Block until in-flight uploads finish (or ``timeout_sec`` elapses).

    Default is 10 minutes; override via ``RELAX_QS_UPLOAD_FLUSH_TIMEOUT_SEC``.
    On timeout a warning is logged and any still-in-flight uploads are
    abandoned when the process exits.
    """
    import time

    if _executor is None:
        return
    if timeout_sec is None:
        timeout_sec = float(_env_int("RELAX_QS_UPLOAD_FLUSH_TIMEOUT_SEC", 600))
    start = time.monotonic()
    logger.info(
        f"[qs-upload] flush() waiting up to {timeout_sec:.0f}s for in-flight uploads (in-flight={_inflight_count})"
    )
    deadline = start + timeout_sec
    while _inflight_count > 0 and time.monotonic() < deadline:
        time.sleep(1.0)
    elapsed = time.monotonic() - start
    if _inflight_count > 0:
        logger.warning(
            f"[qs-upload] flush() giving up after {elapsed:.1f}s with in-flight={_inflight_count}; uploads may be lost"
        )
    else:
        logger.info(f"[qs-upload] flush() drained {elapsed:.1f}s")


def _silence_toolkit_log_noise() -> None:
    """Suppress model_tools / qcloud_cos / urllib3 chatter that's noise-only.

    - ``DefaultCryptoHandler`` prints ``[KMS] Unrecognized secretText ...`` when
      apollo config mixes plaintext AKID keys with ``[PRO]``-encrypted ones;
      the SDK falls back correctly, the ERROR is cosmetic.
    - ``qcloud_cos.cos_client`` logs 404 HEAD-before-PUT probes at WARNING; it
      is the normal upload path.
    - ``urllib3.connectionpool`` warns when concurrent multipart uploads
      overflow the default pool of 10 connections; the pool just recycles
      connections, slightly less efficient but harmless.
    """
    import logging

    for name in ("DefaultCryptoHandler", "qcloud_cos.cos_client", "urllib3.connectionpool"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _resolve_model_name(args: Any) -> str | None:
    """Env override wins over args.experiment_name."""
    name = os.environ.get("RELAX_QS_MODEL_NAME") or getattr(args, "experiment_name", None)
    if not name:
        return None
    return str(name)


def _precision_tag(dtype: str) -> str:
    return {"bf16": "bf16", "fp8": "fp8"}.get(dtype, "unknown")


def _do_push(
    hf_path: str,
    model_name: str,
    user: str,
    token: str,
    *,
    rollout_id: int,
    dtype: str,
    env: str,
    zone: str,
    region: str,
    framework: str,
    gpu_type: str,
    num_threads: int,
    export: bool,
) -> None:
    """Worker body: runs on the executor thread, catches everything."""
    global _inflight_count
    import time

    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    try:
        # Imported lazily so that missing quicksilver-toolkit in unit tests /
        # OSS installs never breaks import of this module.
        from model_tools import push_model

        n_files, total_bytes = _summarize_dir(hf_path)
        total_mib = total_bytes / (1024 * 1024)
        start = time.monotonic()

        model_config = {"rollout_id": rollout_id, "dtype": dtype}
        logger.info(
            f"[qs-upload] starting rollout={rollout_id} model={model_name!r} dtype={dtype} "
            f"files={n_files} size={total_mib:.1f}MiB src={hf_path} region={region or '<env>'} env={env}"
        )
        heartbeat_thread = _start_heartbeat(heartbeat_stop, rollout_id, model_name, total_mib, start)

        push_model(
            model_path=hf_path,
            model_name=model_name,
            user=user,
            token=token,
            env=env,
            zone=zone,
            framework=framework,
            gpu_type=gpu_type,
            region=region,
            precision=_precision_tag(dtype),
            model_format="huggingface",
            model_config=model_config,
            num_threads=num_threads,
            export=export,
        )
        elapsed = time.monotonic() - start
        rate = (total_mib / elapsed) if elapsed > 0 else 0.0
        logger.info(
            f"[qs-upload] finished rollout={rollout_id} model={model_name!r} "
            f"elapsed={elapsed:.1f}s size={total_mib:.1f}MiB avg={rate:.1f}MiB/s"
        )
    except Exception:
        logger.exception(f"[qs-upload] push failed model={model_name!r} rollout={rollout_id}")
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=5)
        shutil.rmtree(hf_path, ignore_errors=True)
        with _inflight_lock:
            _inflight_count -= 1


def _summarize_dir(path: str) -> tuple[int, int]:
    """Return (file_count, total_bytes) for an HF export directory."""
    n_files = 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
                n_files += 1
            except OSError:
                continue
    return n_files, total


_SNAPSHOT_ROOT_NAME = ".uploading"


def _link_snapshot(src: str, snapshot_root: str, rollout_id: int) -> str:
    """Recursively hardlink ``src`` into ``snapshot_root/iter_{rollout_id}``.

    The snapshot shares inodes with the source; the extra reference keeps the
    file bytes alive even if the caller later ``rmtree``s ``src`` (checkpoint
    rotation, cleanup script). Requires src and snapshot_root to sit on the
    same filesystem — enforced implicitly by ``os.link`` (raises OSError EXDEV
    on a cross-device attempt).
    """
    dst = os.path.join(snapshot_root, f"iter_{rollout_id}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_dir = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            os.link(os.path.join(root, name), os.path.join(target_dir, name))
    return dst


def _cleanup_stale_snapshots(snapshot_root: str) -> None:
    """Remove any leftover ``iter_*`` snapshots from a previous crashed run.

    Called once per process on the first upload; safe to be a no-op if the
    directory doesn't exist.
    """
    if not os.path.isdir(snapshot_root):
        return
    for entry in os.listdir(snapshot_root):
        full = os.path.join(snapshot_root, entry)
        if os.path.isdir(full):
            shutil.rmtree(full, ignore_errors=True)
            logger.info(f"[qs-upload] cleaned stale snapshot {full}")


def _start_heartbeat(
    stop_event: threading.Event,
    rollout_id: int,
    model_name: str,
    total_mib: float,
    start_monotonic: float,
) -> threading.Thread:
    """Daemon that logs an elapsed-only heartbeat every N seconds.

    model_tools' COS upload path exposes no byte-level callback (per-file
    ``upload_file`` is the only unit); the best we can do is show the caller
    that the upload is still alive.
    """
    import time

    interval = max(5, _env_int("RELAX_QS_UPLOAD_HEARTBEAT_SEC", 30))

    def _loop() -> None:
        while not stop_event.wait(interval):
            elapsed = time.monotonic() - start_monotonic
            logger.info(
                f"[qs-upload] progress rollout={rollout_id} model={model_name!r} "
                f"elapsed={elapsed:.0f}s size={total_mib:.1f}MiB (no byte-level cb from cos sdk)"
            )

    t = threading.Thread(target=_loop, name=f"qs-upload-hb-{rollout_id}", daemon=True)
    t.start()
    return t


def push_to_quicksilver(args, hf_path: str, rollout_id: int, *, dtype: str, is_lora: bool) -> None:
    """Framework-visible hook.

    Fast-return; heavy work runs in background.
    """
    global _inflight_count, _missing_toolkit_warned

    if not _model_tools_available():
        if not _missing_toolkit_warned:
            logger.warning(
                "[qs-upload] quicksilver-toolkit (model_tools) is not installed; "
                "skipping all uploads. Install it inside the training image if you want QS pushes."
            )
            _missing_toolkit_warned = True
        return

    interval = max(1, _env_int("RELAX_QS_UPLOAD_INTERVAL", 1))
    if rollout_id % interval != 0:
        logger.debug(f"[qs-upload] skip rollout={rollout_id} (interval={interval})")
        return

    user = os.environ.get("QS_USER")
    token = os.environ.get("QS_TOKEN")
    if not user or not token:
        logger.warning("[qs-upload] QS_USER / QS_TOKEN not set; skipping upload")
        return

    model_name = _resolve_model_name(args)
    if not model_name:
        logger.warning(
            "[qs-upload] cannot resolve model name (set RELAX_QS_MODEL_NAME or --experiment-name); skipping"
        )
        return

    if is_lora:
        logger.info(f"[qs-upload] rollout={rollout_id} is LoRA; uploading full hf dir (adapter included)")

    snapshot_root = os.path.join(os.path.dirname(hf_path) or ".", _SNAPSHOT_ROOT_NAME)
    _cleanup_stale_snapshots_once(snapshot_root)
    try:
        upload_path = _link_snapshot(hf_path, snapshot_root, rollout_id)
    except OSError as e:
        logger.warning(f"[qs-upload] failed to hardlink snapshot for rollout={rollout_id} ({e}); skipping this upload")
        return

    queue_max = max(1, _env_int("RELAX_QS_UPLOAD_QUEUE_MAX", 4))
    with _inflight_lock:
        if _inflight_count >= queue_max:
            logger.warning(
                f"[qs-upload] queue full ({_inflight_count}/{queue_max}); dropping rollout={rollout_id}. "
                "Increase RELAX_QS_UPLOAD_QUEUE_MAX or RELAX_QS_UPLOAD_INTERVAL if this happens often."
            )
            shutil.rmtree(upload_path, ignore_errors=True)
            return
        _inflight_count += 1

    executor = _get_executor()
    executor.submit(
        _do_push,
        upload_path,
        model_name,
        user,
        token,
        rollout_id=rollout_id,
        dtype=dtype,
        env=os.environ.get("RELAX_QS_ENV", "prod"),
        zone=os.environ.get("RELAX_QS_ZONE", "cn"),
        region=os.environ.get("RELAX_QS_REGION", ""),
        framework=os.environ.get("RELAX_QS_FRAMEWORK", "huggingface"),
        gpu_type=os.environ.get("RELAX_QS_GPU_TYPE", "NoLimit"),
        num_threads=max(1, _env_int("RELAX_QS_UPLOAD_THREADS", 10)),
        export=_env_bool("RELAX_QS_EXPORT_MODEL", default=False),
    )
    logger.info(f"[qs-upload] enqueued rollout={rollout_id} (in-flight+queued={_inflight_count}/{queue_max})")


# ---------------------------------------------------------------------------
# HuggingFace Hub backend
# ---------------------------------------------------------------------------


_missing_hf_hub_warned = False


def _huggingface_hub_available() -> bool:
    return importlib.util.find_spec("huggingface_hub") is not None


def _do_hf_push(
    hf_path: str,
    repo_id: str,
    token: str,
    *,
    rollout_id: int,
    dtype: str,
    revision: str,
    private: bool,
) -> None:
    """Worker: HfApi.upload_folder on a snapshot dir. Not tested — best-effort."""
    global _inflight_count
    import time

    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    try:
        from huggingface_hub import HfApi

        n_files, total_bytes = _summarize_dir(hf_path)
        total_mib = total_bytes / (1024 * 1024)
        start = time.monotonic()

        logger.info(
            f"[hf-upload] starting rollout={rollout_id} repo={repo_id!r} dtype={dtype} "
            f"files={n_files} size={total_mib:.1f}MiB revision={revision} private={private}"
        )
        heartbeat_thread = _start_heartbeat(heartbeat_stop, rollout_id, repo_id, total_mib, start)

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=hf_path,
            repo_id=repo_id,
            revision=revision,
            commit_message=f"rollout {rollout_id} ({dtype})",
        )
        elapsed = time.monotonic() - start
        rate = (total_mib / elapsed) if elapsed > 0 else 0.0
        logger.info(
            f"[hf-upload] finished rollout={rollout_id} repo={repo_id!r} "
            f"elapsed={elapsed:.1f}s size={total_mib:.1f}MiB avg={rate:.1f}MiB/s"
        )
    except Exception:
        logger.exception(f"[hf-upload] push failed repo={repo_id!r} rollout={rollout_id}")
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=5)
        shutil.rmtree(hf_path, ignore_errors=True)
        with _inflight_lock:
            _inflight_count -= 1


def push_to_huggingface(args, hf_path: str, rollout_id: int, *, dtype: str, is_lora: bool) -> None:
    """Framework-visible hook — async push to a HuggingFace Hub repo.

    Fast-return; heavy work runs in background. See module docstring for env
    vars. No unit-tested; treat as a starting point and adapt to your repo
    layout / branching strategy.
    """
    global _inflight_count, _missing_hf_hub_warned

    if not _huggingface_hub_available():
        if not _missing_hf_hub_warned:
            logger.warning(
                "[hf-upload] huggingface_hub is not installed; skipping all uploads. "
                "`pip install huggingface_hub` inside the training image if you want HF pushes."
            )
            _missing_hf_hub_warned = True
        return

    interval = max(1, _env_int("RELAX_HF_UPLOAD_INTERVAL", 1))
    if rollout_id % interval != 0:
        logger.debug(f"[hf-upload] skip rollout={rollout_id} (interval={interval})")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        logger.warning("[hf-upload] HF_TOKEN / HF_HUB_TOKEN not set; skipping upload")
        return

    repo_id = os.environ.get("RELAX_HF_REPO_ID") or getattr(args, "experiment_name", None)
    if not repo_id:
        logger.warning("[hf-upload] cannot resolve repo id (set RELAX_HF_REPO_ID or --experiment-name); skipping")
        return

    if is_lora:
        logger.info(f"[hf-upload] rollout={rollout_id} is LoRA; uploading full hf dir (adapter included)")

    snapshot_root = os.path.join(os.path.dirname(hf_path) or ".", _SNAPSHOT_ROOT_NAME)
    _cleanup_stale_snapshots_once(snapshot_root)
    try:
        upload_path = _link_snapshot(hf_path, snapshot_root, rollout_id)
    except OSError as e:
        logger.warning(f"[hf-upload] failed to hardlink snapshot for rollout={rollout_id} ({e}); skipping this upload")
        return

    queue_max = max(1, _env_int("RELAX_QS_UPLOAD_QUEUE_MAX", 4))
    with _inflight_lock:
        if _inflight_count >= queue_max:
            logger.warning(f"[hf-upload] queue full ({_inflight_count}/{queue_max}); dropping rollout={rollout_id}.")
            shutil.rmtree(upload_path, ignore_errors=True)
            return
        _inflight_count += 1

    executor = _get_executor()
    executor.submit(
        _do_hf_push,
        upload_path,
        str(repo_id),
        token,
        rollout_id=rollout_id,
        dtype=dtype,
        revision=os.environ.get("RELAX_HF_REVISION", "main"),
        private=_env_bool("RELAX_HF_REPO_PRIVATE", default=False),
    )
    logger.info(f"[hf-upload] enqueued rollout={rollout_id} (in-flight+queued={_inflight_count}/{queue_max})")
