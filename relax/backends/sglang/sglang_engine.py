# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
import ipaddress
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional
from urllib.parse import quote, urlsplit

import ray
import requests
import sglang_router
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree


try:
    from sglang.srt.server_args import LOAD_FORMAT_CHOICES as _SGLANG_LOAD_FORMAT_CHOICES
except ImportError:
    # Older SGLang releases do not expose the supported load formats. Keep
    # imports working, but fail closed if S3 streaming is requested.
    _SGLANG_LOAD_FORMAT_CHOICES = None

from relax.distributed.checkpoint_service.client.engine import create_client
from relax.distributed.ray.ray_actor import RayActor
from relax.utils import device as device_utils
from relax.utils import scale_utils
from relax.utils.async_utils import run
from relax.utils.env import Envs
from relax.utils.http_utils import get_host_info, router_worker_base_url
from relax.utils.logging_utils import get_logger
from relax.utils.megatron_peft_utils import convert_megatron_to_sglang_target_modules, is_lora_enabled
from relax.utils.model_source import ModelSource, SGLangLoadPlan
from relax.utils.s3_model_loader import (
    get_s3_model_cached_path,
    is_s3_uri,
    maybe_resolve_s3_model_to_shm,
    resolve_s3_model_metadata_to_shm,
)
from relax.utils.scale_utils import PrecheckProbeCategory


logger = get_logger(__name__)


# GenRM colocate offload drain: bound every HTTP round-trip and the whole drain
# by a wall-clock deadline so a wedged SGLang scheduler surfaces as a
# TimeoutError instead of blocking rank-0 in ray.get() (and every other rank at
# the downstream offload barrier) indefinitely.
_GENRM_OFFLOAD_DRAIN_TIMEOUT_S = 120.0
_GENRM_OFFLOAD_RELEASE_TIMEOUT_S = 120.0
_SGLANG_HTTP_ATTEMPT_TIMEOUT_S = 30.0
_MIN_HTTP_TIMEOUT_S = 1.0


def _preferred_s3_stream_load_format() -> str:
    if _SGLANG_LOAD_FORMAT_CHOICES is None:
        raise RuntimeError("The installed SGLang cannot report whether runai_streamer is supported")
    if "runai_streamer" in _SGLANG_LOAD_FORMAT_CHOICES:
        return "runai_streamer"
    raise RuntimeError("The installed SGLang does not support runai_streamer")


def _configure_runai_streamer_env(args) -> None:
    model_source = args.model_source
    if "RUNAI_STREAMER_S3_ENDPOINT" not in os.environ:
        endpoint = model_source.endpoint or os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = endpoint
    if "RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING" not in os.environ and model_source.addressing_style == "path":
        os.environ["RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING"] = "0"
    if model_source.credential_mode == "placeholder":
        os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
        os.environ["AWS_ACCESS_KEY_ID"] = "mock"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "mock"
        os.environ.pop("AWS_SESSION_TOKEN", None)


def build_sglang_load_plan(server_args: dict, args) -> SGLangLoadPlan:
    """Build one policy loading decision for an SGLang engine group."""
    model_source = getattr(args, "model_source", None)
    model_path = server_args["model_path"]
    load_format = server_args.get("load_format", "auto")
    if model_source is None:
        return SGLangLoadPlan(
            model_path=model_path,
            load_format=load_format,
            source=ModelSource(uri=model_path),
        )
    if model_path != model_source.uri or not is_s3_uri(model_source.uri):
        return SGLangLoadPlan(model_path=model_path, load_format=load_format, source=model_source)

    if load_format == "runai_streamer":
        _configure_runai_streamer_env(args)
        return SGLangLoadPlan(model_path=model_path, load_format=load_format, source=model_source)
    if load_format == "remote":
        raise ValueError("SGLang load_format='remote' is not an S3 model loader; use 'runai_streamer' instead")

    cached_path = get_s3_model_cached_path(model_source.uri, args)
    if load_format == "dummy":
        model_path = cached_path or resolve_s3_model_metadata_to_shm(model_source.uri, args)
        return SGLangLoadPlan(model_path=model_path, load_format=load_format, source=model_source)

    if load_format == "auto":
        # A multi-node engine group must not make node-local cache decisions
        # independently. Until group-wide readiness coordination is available,
        # all ranks use streaming. Single-node groups may reuse ready SHM.
        if server_args.get("nnodes", 1) == 1 and cached_path is not None:
            return SGLangLoadPlan(model_path=cached_path, load_format=load_format, source=model_source)
        load_format = _preferred_s3_stream_load_format()
        _configure_runai_streamer_env(args)
        return SGLangLoadPlan(model_path=model_path, load_format=load_format, source=model_source)

    model_path = maybe_resolve_s3_model_to_shm(model_source.uri, args)
    return SGLangLoadPlan(model_path=model_path, load_format=load_format, source=model_source)


def _apply_sglang_policy_load_plan(server_args: dict, args) -> dict:
    """Apply the policy model load plan to a copied ServerArgs dictionary."""
    plan = build_sglang_load_plan(server_args, args)
    resolved = dict(server_args)
    resolved["model_path"] = plan.model_path
    resolved["load_format"] = plan.load_format
    if (
        getattr(args, "model_source", None) is not None
        and plan.model_path == plan.source.uri
        and is_s3_uri(plan.source.uri)
        and plan.load_format == "runai_streamer"
        and not resolved.get("tokenizer_path")
    ):
        resolved["tokenizer_path"] = resolve_s3_model_metadata_to_shm(plan.source.uri, args)
    return resolved


# Consecutive connection failures that mean "the server process is gone" rather
# than "the server is briefly busy". Bail out instead of retrying until the
# drain deadline: a dead engine will never answer.
_MAX_CONSECUTIVE_CONNECT_ERRORS = 3


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    visible_env = device_utils.get_visible_devices_env_var()
    cvd = os.environ.get(visible_env)
    if not cvd:
        return physical_gpu_id  # no remapping
    # Visible devices can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"Device id {physical_gpu_id} is not valid under {visible_env}={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible) - 1} (local)."
    )


def _patched_run_scheduler_process(*args, **kwargs):
    """Scheduler-subprocess entry used for the routing-replay path.

    This wrapper is only installed when ``--optimize-routing-replay`` is
    enabled (see ``_launch_server_with_patches``), so the routing-replay async
    D→H patch is applied **unconditionally** here, preserving the original
    behavior.
    """
    from relax.backends.sglang.routing_replay_patch import apply_patch

    apply_patch()

    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def _launch_server_with_patches(server_args: ServerArgs):
    """Top-level picklable ``multiprocessing.Process`` target that applies the
    SGLang patches, each gated by its own env flag so any combination is valid:

    - main process: OPD pre-expanded multimodal patch
      (``RELAX_OPD_PREEXPANDED_PATCH=1``).
    - scheduler subprocess: routing-replay (``RELAX_OPTIMIZE_ROUTING_REPLAY=1``)
      installs ``_patched_run_scheduler_process``, which applies the
      routing-replay patch unconditionally.
    """
    from sglang.srt.entrypoints.http_server import launch_server

    if Envs.RELAX_OPD_PREEXPANDED_PATCH:
        from relax.utils.opd.opd_sglang_patch import apply_opd_preexpanded_patch

        apply_opd_preexpanded_patch()

    if Envs.RELAX_OPTIMIZE_ROUTING_REPLAY:
        launch_server(server_args, run_scheduler_process_func=_patched_run_scheduler_process)
    else:
        launch_server(server_args)


def _resolve_external_model_arch(package_name):
    """Scan an external model package for EntryClass and return architecture
    name.

    Mirrors SGLang's own import_model_classes() discovery logic: iterates over
    all non-package modules in the given package and looks for an
    ``EntryClass`` attribute.  Returns the ``__name__`` of the first discovered
    class, or ``None`` if nothing is found.
    """
    import importlib
    import pkgutil

    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(entry, list):
                    if entry:
                        return entry[0].__name__
                    continue
                return entry.__name__
    return None


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    multiprocessing.set_start_method("spawn", force=True)

    # Each SGLang patch is controlled by its own env flag and applied
    # independently (see ``_launch_server_with_patches`` and
    # ``_patched_run_scheduler_process``); any combination is valid:
    #   - RELAX_OPTIMIZE_ROUTING_REPLAY : async D→H routing-replay patch (runtime)
    #   - RELAX_OPD_PREEXPANDED_PATCH   : OPD pre-expanded multimodal patch (runtime)
    #   - RELAX_OPD_PER_POS_TOKEN_IDS   : OPD per-position token_ids logprob;
    optimize = Envs.RELAX_OPTIMIZE_ROUTING_REPLAY
    opd_patch = Envs.RELAX_OPD_PREEXPANDED_PATCH
    per_pos = Envs.RELAX_OPD_PER_POS_TOKEN_IDS
    logger.info(
        "Launching SGLang server with independently-gated patches: "
        f"routing_replay={optimize}, opd_preexpanded={opd_patch}, per_pos_token_ids={per_pos}"
    )

    p = multiprocessing.Process(target=_launch_server_with_patches, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _wait_server_healthy(base_url, api_key, is_process_alive, timeout=None):
    """Wait until the server at *base_url* is healthy.

    Args:
        base_url: Base URL of the engine (e.g. ``http://host:port``).
        api_key: Bearer token for the engine API (may be ``None``).
        is_process_alive: Callable returning ``False`` when the server
            process has exited.
        timeout: Maximum wall-clock seconds to wait.  ``None`` means no
            limit (backward-compatible default).  A ``TimeoutError`` is
            raised when the deadline is exceeded.
    """
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }
    # Per-request timeout for individual HTTP calls so that a single
    # ``requests.get`` does not block indefinitely on a black-holed host.
    _REQUEST_TIMEOUT = 10  # seconds (connect + read)

    deadline = (time.monotonic() + timeout) if timeout is not None else None

    def _check_deadline(phase: str):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for server to become healthy at {base_url} (phase={phase}, timeout={timeout}s)"
            )

    with requests.Session() as session:
        while True:
            _check_deadline("health_generate")
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers, timeout=_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            _check_deadline("flush_cache")
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers, timeout=_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)


class SGLangEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
        register_sigterm_handler: bool = False,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self._evicted = threading.Event()
        self._router_worker_id: str | None = None
        self._router_unregister_submitted = False
        if register_sigterm_handler:
            self._register_sigterm_handler()

    def _register_sigterm_handler(self):
        """Register SIGTERM handler for platform-initiated pod eviction.

        The signal handler only publishes an intent. RolloutManager owns the
        weight-update fence and the live-actor removal sequence; doing I/O or
        waiting here could block the actor RPC that releases an active update.
        """
        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(_signum, _frame):
            self._evicted.set()

        signal.signal(signal.SIGTERM, _handle_sigterm)

    def is_evicted(self) -> bool:
        """Check whether this engine has received a SIGTERM eviction signal."""
        return self._evicted.is_set()

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
        init_external_kwargs: Optional[dict] = None,
        skip_dcs_registration: bool = False,
        skip_router_registration: bool = False,
    ):
        """Initialize the SGLang engine.

        Args:
            skip_router_registration: If True, do not register to router during init.
                This is used during scale-out to ensure the engine receives weights
                before accepting requests. The caller must call register_to_router()
                after weight sync completes.
        """
        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port
        self._skip_router_registration = skip_router_registration

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )
        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        # Start the engine first so the server is healthy before we create the
        # DCS client.  Creating the client before the server is ready can cause
        # Actor's weight-update path to reach a DCS endpoint that does not yet
        # exist, especially for scaled-out engines whose init() runs concurrently
        # with the training loop.
        self.checkpoint_engine_client = None
        if self.args.rollout_external or init_external_kwargs:
            if not init_external_kwargs:
                init_external_kwargs = {"external_engine_need_check_fields": external_engine_need_check_fields}
            self._init_external(server_args_dict, **init_external_kwargs)
        else:
            self._init_normal(server_args_dict)

        # Register to DCS coordinator only if not skipped (e.g., for scaled-out engines)
        # Scaled-out engines use direct weight sync from seed engine instead of DCS.
        # Done after engine startup so the coordinator can immediately reach the server.
        if not skip_dcs_registration:
            self.register_dcs()

    def register_dcs(self):
        if self.node_rank == 0 and self.args.fully_async:
            # Resolve effective num_gpus_per_engine for this engine
            effective_num_gpus = self.num_gpus_per_engine or self.args.rollout_num_gpus_per_engine
            self.checkpoint_engine_client = run(
                create_client(
                    args=self.args,
                    coordinator_url=self.args.coordinator_url,
                    role="rollout",
                    ip=self.server_host,
                    port=self.server_port,
                    rank=self.rank,
                    metadata={"num_gpus_per_engine": effective_num_gpus},
                )
            )

    def _init_external(self, expect_server_args, external_engine_need_check_fields, timeout: float = 300):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _get_actual_server_args():
            response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info", timeout=10)
            response.raise_for_status()
            return response.json()

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert actual_value == expect_value, (
                    f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"
                )

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
            timeout=timeout,
        )
        actual_server_args = _get_actual_server_args()
        _sanity_check_server_args(actual_server_args, expect_server_args)

        if not self._skip_router_registration:
            self.register_to_router()

    def _init_normal(self, server_args_dict, *, apply_policy_load_plan: bool = True):
        if apply_policy_load_plan:
            server_args_dict = _apply_sglang_policy_load_plan(server_args_dict, self.args)

        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        if getattr(self.args, "optimize_routing_replay", False):
            os.environ["RELAX_OPTIMIZE_ROUTING_REPLAY"] = "1"

        if Envs.RELAX_OPD_PER_POS_TOKEN_IDS:
            os.environ["RELAX_FORCE_LOGPROBS_BASE64"] = "1"
            top_k = getattr(self.args, "opd_log_prob_top_k", 0)
            if top_k:
                os.environ["RELAX_OPD_TOKEN_IDS_LOGPROB_K"] = str(top_k)
            logger.info(
                "Set RELAX_FORCE_LOGPROBS_BASE64=1, RELAX_OPD_TOKEN_IDS_LOGPROB_K=%s (topk enabled)",
                Envs.RELAX_OPD_TOKEN_IDS_LOGPROB_K,
            )

        # Set SGLang external model/processor package env vars so the spawned
        # SGLang subprocess discovers and registers custom model implementations.
        # Must be set before launch_server_process() spawns child process
        # (multiprocessing start_method='spawn'), because the child inherits
        # the parent's os.environ at spawn time.
        external_pkg = getattr(self.args, "sglang_external_model_package", None)
        if external_pkg:
            os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = external_pkg
            os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] = external_pkg
            arch = _resolve_external_model_arch(external_pkg)
            if arch:
                os.environ["SGLANG_EXTERNAL_MM_MODEL_ARCH"] = arch
            logger.info(f"Set SGLANG_EXTERNAL_MODEL_PACKAGE={external_pkg}, SGLANG_EXTERNAL_MM_MODEL_ARCH={arch}")

        # Warm the OS page cache for this engine's HF checkpoint before the SGLang
        # subprocess mmaps the safetensors. Applies uniformly to rollout / genrm /
        # teacher because they all take this path; the shared marker keyed by
        # abs_path collapses repeated calls on the same node to a single read.
        if getattr(self.args, "warm_hf_checkpoint_page_cache", False):
            from relax.utils.hf_page_cache import warm_hf_checkpoint_page_cache

            warm_hf_checkpoint_page_cache(server_args_dict.get("model_path"))

        server_args_dict = {**server_args_dict, "host": server_args_dict["host"].strip("[]")}
        self.process = launch_server_process(ServerArgs(**server_args_dict))

        bootstrap_port = (
            server_args_dict.get("disaggregation_bootstrap_port") if self.worker_type == "prefill" else None
        )
        # Only register to router if skip_router_registration=False
        if not self._skip_router_registration:
            self.register_to_router(bootstrap_port=bootstrap_port)

    def _make_request(self, endpoint: str, payload: dict | None = None, timeout: float | None = None):
        """Make a POST request to the specified endpoint with the given
        payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)
            timeout: Optional per-request timeout in seconds. Defaults to None
                (no timeout) to preserve behaviour for existing callers; pass a
                bound on paths that must not hang (e.g. colocate offload).

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {}, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """

        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """Update model weights from tensor data. The HTTP server will only
        post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        serialized_tensors: str,
        config_dict: dict,
        load_format: str | None = None,
        pinned: bool = False,
    ) -> dict | None:
        """Load/refresh a LoRA adapter directly from serialized tensors — no
        disk IO.

        Wraps SGLang's ``/load_lora_adapter_from_tensors`` (available since 0.5.12), the transport
        the colocate backend uses. ``config_dict`` is the adapter config (the same content that
        would go into ``adapter_config.json``); ``serialized_tensors`` carries the full
        (TP-gathered, PP-merged) adapter tensors serialized with SGLang's ``MultiprocessingSerializer``.

        Unlike ``update_weights_from_tensor`` (which fans out one shard per TP worker), SGLang
        broadcasts this single blob to every TP worker, which each deserialize it and slice their
        own shard internally (``slice_lora_a/b_weights``). The caller must therefore serialize
        **host** tensors (CUDA-IPC handles would not survive the fan-out to GPU-isolated workers).

        Requires the server launched with ``--enable-lora`` and ``dp_size == 1``. Re-registers the
        adapter when called again with the same ``lora_name``.

        Args:
            lora_name: Adapter name; rollout requests pass ``lora_path=lora_name``.
            serialized_tensors: Adapter tensors serialized via ``MultiprocessingSerializer.serialize(..., output_str=True)``.
            config_dict: HF-PEFT config dict (see ``build_hf_peft_config_dict``).
            load_format: Optional SGLang load format (e.g. ``"flattened_bucket"``); ``None`` for a plain tensor dict.
            pinned: Pin the adapter against LRU eviction (kept False; one self-managed adapter).

        Returns:
            Response dict from the server (``{"success": bool, ...}``), or None on non-lead node.
        """
        return self._make_request(
            "load_lora_adapter_from_tensors",
            {
                "lora_name": lora_name,
                "config_dict": config_dict,
                "serialized_tensors": serialized_tensors,
                "load_format": load_format,
                "pinned": pinned,
            },
        )

    def update_lora_from_distributed(
        self,
        lora_name: str,
        names: list[str],
        dtypes: list,
        shapes: list,
        config_dict: dict,
        group_name: str,
        pinned: bool = False,
    ) -> dict | None:
        """Load/refresh a LoRA adapter whose tensors arrive over an NCCL group.

        — no disk IO.

        NCCL counterpart to :meth:`load_lora_adapter_from_tensors`. The HTTP call
        only carries metadata (``names``/``dtypes``/``shapes``/``config_dict``); the
        actual adapter tensors are broadcast ``src=0`` on ``group_name`` (the same
        weight-update group base weights use) and received on the SGLang side by
        ``/update_lora_from_distributed``. Fully-async uses this instead of a shared-
        directory handoff, so no adapter ever touches the network FS.

        Same-name replacement is handled server-side, so no separate
        ``/unload_lora_adapter`` call is needed. Requires ``--enable-lora`` and
        ``dp_size == 1``.

        Args:
            lora_name: Adapter name; rollout requests pass ``lora_path=lora_name``.
            names: Adapter tensor names (HF-PEFT layout), defining the broadcast order.
            dtypes: Per-tensor dtypes (``torch.dtype`` or str), serialized as bare names.
            shapes: Per-tensor shapes.
            config_dict: HF-PEFT config dict (see ``build_hf_peft_config_dict``).
            group_name: NCCL group to receive the tensors on.
            pinned: Pin the adapter against LRU eviction (kept False; one self-managed adapter).

        Returns:
            Response dict from the server (``{"success": bool, ...}``), or None on non-lead node.
        """
        return self._make_request(
            "update_lora_from_distributed",
            {
                "lora_name": lora_name,
                "config_dict": config_dict,
                "names": names,
                "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
                "shapes": shapes,
                "group_name": group_name,
                "pinned": pinned,
            },
        )

    def unload_lora_adapter(self, lora_name: str) -> dict | None:
        """Unload a previously registered LoRA adapter by name.

        Used in adapter mode to drop the prior adapter version before registering a refreshed
        one under the same name, so adapters do not accumulate / collide in the engine's LoRA
        registry.

        Args:
            lora_name: Adapter name to unload.

        Returns:
            Response dict from the server (``{"success": bool, ...}``), or None on non-lead node.
        """
        return self._make_request("unload_lora_adapter", {"lora_name": lora_name})

    def abort_requests(self, timeout: float = _SGLANG_HTTP_ATTEMPT_TIMEOUT_S):
        """Best-effort abort of all in-flight requests on the engine.

        Called while draining before offload (``release_memory_occupation``) so
        lingering requests do not block ``flush_cache``: SGLang returns HTTP
        400 from ``/flush_cache`` while the scheduler still has pending/running
        requests. A GenRM judge/summary ``/generate`` left over from a partial
        rollout can keep decoding for minutes, which otherwise times out the
        offload and kills the job. Aborting drains the scheduler so the
        subsequent flush + memory release proceed. Never raises — abort failure
        must not block the offload path.
        """
        if self.node_rank != 0:
            return
        try:
            requests.post(
                f"http://{self.server_host}:{self.server_port}/abort_request",
                json={"abort_all": True},
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001 — best-effort, keep offloading
            logger.info(f"abort_requests failed (continuing to flush): {e}")

    def flush_cache(self, timeout_s: float = 120.0, max_connect_errors: int = _MAX_CONSECUTIVE_CONNECT_ERRORS):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        deadline = time.monotonic() + timeout_s
        connect_errors = 0
        # flush cache will not return status_code 200 when there are pending requests
        while True:
            try:
                response = requests.get(
                    f"http://{self.server_host}:{self.server_port}/flush_cache",
                    timeout=_SGLANG_HTTP_ATTEMPT_TIMEOUT_S,
                )
                if response.status_code == 200:
                    break
                # 400 = running/waiting requests present; wait and retry below.
                connect_errors = 0
            except requests.exceptions.ConnectionError as e:
                connect_errors += 1
                logger.warning(
                    f"Cannot reach {self.server_host}:{self.server_port}/flush_cache "
                    f"({connect_errors}/{max_connect_errors}): {e}"
                )
                if connect_errors >= max_connect_errors:
                    raise ConnectionError(
                        f"Engine {self.server_host}:{self.server_port} unreachable while "
                        f"flushing cache ({connect_errors} consecutive connection errors) "
                        f"— the server process is most likely dead."
                    ) from e
            except Exception as e:
                connect_errors = 0
                logger.info(f"Error flushing cache: {e}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timeout while flushing cache.")
            time.sleep(1)

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        self.unregister_from_router()
        # external rollout has no process
        if hasattr(self, "process"):
            kill_process_tree(self.process.pid)

    def __del__(self):
        """Safety net: kill SGLang child processes when the actor is garbage-
        collected.

        This prevents orphaned sglang::scheduler / sglang::detokenizer
        processes from lingering after training completes, in case shutdown()
        was not called explicitly (e.g. Ray actor GC without explicit cleanup).
        """
        process = getattr(self, "process", None)
        if process is not None and process.is_alive():
            try:
                kill_process_tree(process.pid)
            except Exception:
                pass

    def get_url(self) -> str | None:
        """Return the HTTP URL of this engine, or None for non-node-0
        engines."""
        if self.node_rank != 0:
            return None
        return f"http://{self.server_host}:{self.server_port}"

    def get_pid_and_node_id(self) -> dict:
        """Return the PID and Ray node ID of this engine.

        Returns:
            dict with 'pid' (int) and 'node_id' (str) keys.
        """
        node_id = ""
        try:
            node_id = ray.get_runtime_context().get_node_id()
        except Exception:
            pass
        return {"pid": os.getpid(), "node_id": node_id}

    def register_to_router(self, bootstrap_port: int | None = None, strict: bool = True) -> bool:
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return True

        worker_url = f"http://{self.server_host}:{self.server_port}"
        try:
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                if self.worker_type != "regular":
                    msg = "pd disaggregation is not supported in old router or slime router."
                    if strict:
                        raise ValueError(msg)
                    logger.warning(msg)
                    return False
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}",
                    timeout=30,
                )
            else:
                payload = {
                    "url": worker_url,
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill" and bootstrap_port is not None:
                    payload["bootstrap_port"] = bootstrap_port
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                    timeout=30,
                )
            response.raise_for_status()
            self._router_worker_id = None
            if parse(sglang_router.__version__) > parse("0.2.1") and not self.args.use_slime_router:
                try:
                    response_payload = response.json()
                except ValueError:
                    response_payload = {}
                if isinstance(response_payload, dict):
                    worker_id = response_payload.get("worker_id")
                    if worker_id is not None and str(worker_id):
                        self._router_worker_id = str(worker_id)
                if self._router_worker_id is None:
                    location = response.headers.get("Location")
                    if not location and isinstance(response_payload, dict):
                        location = response_payload.get("location")
                    try:
                        location_path = urlsplit(location).path.rstrip("/")
                    except (TypeError, ValueError):
                        location_path = ""
                    parent_path, separator, worker_id = location_path.rpartition("/")
                    if separator and parent_path.endswith("/workers") and worker_id:
                        self._router_worker_id = worker_id
                if self._router_worker_id is None:
                    logger.warning(f"Router did not return a worker_id while registering engine {worker_url}.")
            self._router_unregister_submitted = False
            logger.info(f"Registered engine {worker_url} to router {self.router_ip}:{self.router_port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to register engine to router: {e}")
            return False

    def _wait_for_router_removal(self, worker_url: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        last_error = None
        while True:
            try:
                response = requests.get(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    timeout=min(5.0, max(1.0, deadline - time.monotonic())),
                )
                response.raise_for_status()
                workers = response.json().get("workers", [])
                if not any(
                    isinstance(worker, dict) and router_worker_base_url(worker.get("url", "")) == worker_url
                    for worker in workers
                ):
                    return True
                last_error = None
            except Exception as e:
                last_error = e

            if time.monotonic() >= deadline:
                error_suffix = f": {last_error}" if last_error is not None else ""
                logger.warning(f"Timed out waiting for worker {worker_url} to leave the router{error_suffix}")
                return False
            time.sleep(0.5)

    def unregister_from_router(self, wait_for_removal: bool = False, timeout: float = 30.0) -> bool:
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return True
        worker_url = f"http://{self.server_host}:{self.server_port}"
        router_version = parse(sglang_router.__version__)
        if self._router_unregister_submitted:
            if not wait_for_removal or router_version < parse("0.3.0"):
                return True
            removed = self._wait_for_router_removal(worker_url, timeout)
            if not removed:
                self._router_unregister_submitted = False
            return removed

        try:
            if router_version <= parse("0.2.1") or self.args.use_slime_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url={worker_url}",
                    timeout=30,
                )
            elif router_version < parse("0.3.0"):
                response = requests.delete(
                    f"http://{self.router_ip}:{self.router_port}/workers/{quote(worker_url, safe='')}",
                    timeout=30,
                )
            elif self._router_worker_id is not None:
                response = requests.delete(
                    f"http://{self.router_ip}:{self.router_port}/workers/{quote(self._router_worker_id, safe='')}",
                    timeout=30,
                )
            else:
                workers_response = requests.get(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    timeout=30,
                )
                workers_response.raise_for_status()
                all_workers = workers_response.json().get("workers", [])
                exact_worker = None
                dp_worker_found = False
                for worker in all_workers:
                    if not isinstance(worker, dict):
                        continue
                    listed_worker_url = worker.get("url")
                    if listed_worker_url == worker_url:
                        exact_worker = worker
                    elif (
                        isinstance(listed_worker_url, str) and router_worker_base_url(listed_worker_url) == worker_url
                    ):
                        dp_worker_found = True

                if dp_worker_found:
                    logger.warning(
                        f"Cannot unregister DP-aware engine {worker_url} without its registration worker ID."
                    )
                    return False
                if exact_worker is None:
                    logger.warning(f"Worker {worker_url} not found in router during unregister.")
                    return False

                worker_id = exact_worker.get("id")
                if worker_id is None or str(worker_id) == "":
                    logger.warning(f"Router worker {worker_url} did not have a worker ID during unregister.")
                    return False
                response = requests.delete(
                    f"http://{self.router_ip}:{self.router_port}/workers/{quote(str(worker_id), safe='')}",
                    timeout=30,
                )
            if response.status_code != 404:
                response.raise_for_status()
            self._router_unregister_submitted = True
            if wait_for_removal and router_version >= parse("0.3.0"):
                removed = self._wait_for_router_removal(worker_url, timeout)
                if not removed:
                    self._router_unregister_submitted = False
                    return False
            logger.info(f"Unregistered engine {worker_url} from router {self.router_ip}:{self.router_port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to unregister engine from router: {e}")
            return False

    def get_rank(self) -> int:
        """Return the engine rank assigned during __init__."""
        return self.rank

    def get_weight_version(self) -> Optional[str]:
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] = None):
        """Available tags for multi-stage resume: weights, kv_cache."""
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def init_weights_send_group_for_remote_instance(
        self, master_address, ports, group_rank, world_size, group_name="weight_send_group", backend="nccl"
    ):
        return self._make_request(
            "init_weights_send_group_for_remote_instance",
            {
                "master_address": master_address,
                "ports": ports,
                "group_rank": group_rank,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def send_weights_to_remote_instance(self, master_address, ports, group_name="weight_send_group"):
        return self._make_request(
            "send_weights_to_remote_instance",
            {
                "master_address": master_address,
                "ports": ports,
                "group_name": group_name,
            },
        )

    def get_scale_weight_sync_transport_fingerprint(self) -> dict:
        """The engine's NCCL transport env — a cheap fingerprint (no NCCL, no
        subprocess) for the Stage-1 seed-vs-new compatibility gate before the
        real probe."""
        return {
            "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
            "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
            "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
            "nccl_ib_gid_index": os.environ.get("NCCL_IB_GID_INDEX"),
        }

    def run_scale_weight_sync_precheck(
        self,
        master_address: str,
        ports: str,
        group_rank: int,
        run_token: str,
        tp_size: int,
        timeout_secs: float,
    ) -> dict:
        """Run an independent NCCL probe without touching ModelRunner groups.

        The probe reuses this actor's node, GPU mapping, Python environment,
        and NCCL environment. It launches local subprocesses only; no new Ray
        actor or GPU allocation is created. Transport variables are inherited
        verbatim, so socket-only mode is accepted only when both engine actors
        already received the same job-start environment.
        """
        if self.node_rank != 0:
            return {"success": False, "category": PrecheckProbeCategory.UNSUPPORTED_NODE_RANK.value, "results": []}
        local_gpu_count = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)
        if tp_size != local_gpu_count:
            return {
                "success": False,
                "category": PrecheckProbeCategory.UNSUPPORTED_TOPOLOGY.value,
                "message": f"precheck requires one local process per TP rank: {tp_size=} {local_gpu_count=}",
                "results": [],
            }

        port_list = ports.split(",")
        if len(port_list) != tp_size:
            return {
                "success": False,
                "category": PrecheckProbeCategory.INVALID_PORTS.value,
                "message": f"expected {tp_size} ports, got {len(port_list)}",
                "results": [],
            }

        env = os.environ.copy()
        visible_env = device_utils.get_visible_devices_env_var()
        inherited_visible = [item.strip() for item in env.get(visible_env, "").split(",") if item.strip()]
        physical_gpu_ids = [self.base_gpu_id + local_rank for local_rank in range(tp_size)]
        if inherited_visible:
            physical_tokens = [str(gpu_id) for gpu_id in physical_gpu_ids]
            if all(token in inherited_visible for token in physical_tokens):
                selected_visible = physical_tokens
                parent_device_ids = [inherited_visible.index(token) for token in physical_tokens]
            elif all(0 <= gpu_id < len(inherited_visible) for gpu_id in physical_gpu_ids):
                selected_visible = [inherited_visible[gpu_id] for gpu_id in physical_gpu_ids]
                parent_device_ids = physical_gpu_ids
            else:
                return {
                    "success": False,
                    "category": PrecheckProbeCategory.GPU_MAPPING_MISMATCH.value,
                    "message": f"cannot map {physical_gpu_ids=} under {visible_env}={inherited_visible}",
                    "results": [],
                }
        else:
            selected_visible = [str(gpu_id) for gpu_id in physical_gpu_ids]
            parent_device_ids = physical_gpu_ids
        # Ray intentionally does not isolate this actor's CVD. Restrict each
        # probe process tree to exactly the GPUs already assigned to the engine.
        env[visible_env] = ",".join(selected_visible)
        # Diagnostic verbosity is local to the probe subprocess and does not
        # alter the seed ModelRunner environment or transport selection.
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
        env.pop("NCCL_DEBUG_FILE", None)

        # Each probe creates a CUDA context next to the live ModelRunner. Check
        # every assigned device before launching any child so low headroom
        # fails closed without partially starting a probe. SGLang reserves
        # ~85-90% VRAM via mem-fraction-static, so a healthy engine often has
        # <2 GiB free even though a probe only needs a CUDA context + tiny NCCL
        # buffers (~few hundred MB); default to a realistic 512 MiB floor.
        min_free_bytes = Envs.RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MIN_FREE_BYTES
        memory_results = []
        try:
            import torch

            for local_rank, (physical_gpu_id, parent_device_id) in enumerate(zip(physical_gpu_ids, parent_device_ids)):
                free_bytes, total_bytes = torch.cuda.mem_get_info(parent_device_id)
                memory_results.append(
                    {
                        "local_rank": local_rank,
                        "physical_gpu_id": physical_gpu_id,
                        "parent_device_id": parent_device_id,
                        "free_bytes": free_bytes,
                        "total_bytes": total_bytes,
                        "required_free_bytes": min_free_bytes,
                    }
                )
        except Exception as exc:
            return {
                "success": False,
                "category": PrecheckProbeCategory.MEMORY_CHECK_FAILED.value,
                "message": f"failed to query GPU memory: {type(exc).__name__}: {exc}",
                "memory": memory_results,
                "results": [],
            }
        if any(item["free_bytes"] < min_free_bytes for item in memory_results):
            return {
                "success": False,
                "category": PrecheckProbeCategory.INSUFFICIENT_GPU_MEMORY.value,
                "memory": memory_results,
                "results": [],
            }

        processes = []
        log_directory = tempfile.TemporaryDirectory(prefix=f"relax-nccl-precheck-{run_token}-")
        started_at = time.monotonic()
        for local_rank, port in enumerate(port_list):
            command = [
                sys.executable,
                "-m",
                "relax.backends.sglang._scale_weight_sync_precheck",
                "--master-address",
                master_address,
                "--master-port",
                port,
                "--rank",
                str(group_rank),
                "--device-id",
                str(local_rank),
                "--timeout-secs",
                str(timeout_secs),
                "--run-token",
                f"{run_token}-{local_rank}",
            ]
            try:
                log_path = os.path.join(log_directory.name, f"rank-{local_rank}.log")
                log_file = open(log_path, "w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                processes.append((local_rank, process, log_file, log_path))
            except Exception as exc:
                if "log_file" in locals() and not log_file.closed:
                    log_file.close()
                for _, process, process_log, _ in processes:
                    scale_utils._terminate_probe_process(process)
                    process_log.close()
                log_directory.cleanup()
                return {
                    "success": False,
                    "category": PrecheckProbeCategory.LAUNCH_TRANSIENT.value,
                    "message": f"failed to launch rank {local_rank}: {type(exc).__name__}: {exc}",
                    "results": [],
                }

        deadline = started_at + timeout_secs
        # Wrap the entire post-launch body so an unexpected exception (e.g.
        # ``open`` failing during result collection) cannot leak running child
        # processes (CUDA context + NCCL group on the LIVE gpu) or the tempdir.
        # The finally is idempotent: terminating an already-dead process is a
        # no-op (poll() guards it) and closing a closed file is harmless.
        try:
            while any(process.poll() is None for _, process, _, _ in processes) and time.monotonic() < deadline:
                time.sleep(0.05)

            timed_out = {local_rank for local_rank, process, _, _ in processes if process.poll() is None}
            for local_rank, process, _, _ in processes:
                if local_rank in timed_out:
                    scale_utils._terminate_probe_process(process)

            results = []
            for local_rank, process, log_file, log_path in processes:
                # A probe wedged in an uninterruptible (D) state can survive SIGKILL,
                # so bound the join instead of blocking this actor thread forever.
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.error(
                        f"NCCL precheck rank {local_rank} (pid {process.pid}) unresponsive after "
                        "SIGKILL; abandoning join to avoid blocking the actor"
                    )
                log_file.close()
                with open(log_path, encoding="utf-8", errors="replace") as probe_log:
                    combined_output = probe_log.read()
                parsed = None
                for line in reversed(combined_output.splitlines()):
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                # Category is derived from structured signals only: the manager
                # deadline, the subprocess exit code, and its JSON — never by
                # scanning the NCCL log text.
                if local_rank in timed_out:
                    category, success = PrecheckProbeCategory.TIMEOUT.value, False
                elif process.returncode == 0 and bool(parsed and parsed.get("success")):
                    category, success = None, True
                elif parsed is None:
                    # No structured output → the subprocess never really ran.
                    category, success = PrecheckProbeCategory.LAUNCH_TRANSIENT.value, False
                else:
                    # Ran but NCCL init/collective raised; carry the raw exception.
                    category, success = PrecheckProbeCategory.PROBE_FAILED.value, False
                parsed = parsed or {}
                results.append(
                    {
                        "local_rank": local_rank,
                        "success": success,
                        "category": category,
                        "returncode": process.returncode,
                        "error_type": parsed.get("error_type"),
                        "error": parsed.get("error"),
                        "result": parsed or None,
                        "log_tail": combined_output[-4000:],  # surfaced in the coordinator's failure log
                    }
                )

            failed = [result for result in results if not result["success"]]
            return {
                "success": not failed,
                "category": failed[0]["category"] if failed else None,
                "error_type": failed[0].get("error_type") if failed else None,
                "error": failed[0].get("error") if failed else None,
                "run_token": run_token,
                "memory": memory_results,
                "fingerprint": {
                    "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
                    "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
                    "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
                    "nccl_ib_gid_index": os.environ.get("NCCL_IB_GID_INDEX"),
                    "visible_devices": selected_visible,
                },
                "results": results,
            }
        finally:
            for _, process, log_file, _ in processes:
                scale_utils._terminate_probe_process(process)
                try:
                    log_file.close()
                except Exception:
                    pass
            log_directory.cleanup()

    def pause_generation(self, timeout: float | None = None):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/pause_generation", json={}, timeout=timeout
        )
        response.raise_for_status()
        return response

    def continue_generation(self, timeout: float | None = None):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/continue_generation", json={}, timeout=timeout
        )
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """Update model weights from tensor data.

        The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()

    def unregister_dcs(self):
        if self.node_rank == 0 and self.checkpoint_engine_client is not None:
            logger.info(f"Unregistering checkpoint engine client for engine {self.server_host}:{self.server_port}...")
            run(self.checkpoint_engine_client.unregister())


class GenRMEngine(SGLangEngine):
    """GenRM Engine for Generative Reward Model.

    Inherits from SGLangEngine and overrides initialization to use genrm-
    specific arguments (model path, GPU count, sampling parameters, etc.).
    """

    def init(self, dist_init_addr, port, nccl_port, host=None, disaggregation_bootstrap_port=None):
        """Initialize the genRM engine with genrm-specific arguments."""
        self.router_ip = ""
        self.router_port = 0
        self._skip_router_registration = True

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_genrm_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict, apply_policy_load_plan=False)

    def release_memory_occupation(self):
        # GenRM is colocated on the training GPUs, so it must offload at the
        # rollout->train transition. Two failure modes are defended against here:
        #
        # 1. Admission race. SGLang's release_memory_occupation asserts the
        #    scheduler is idle (``_is_no_request``); a straggler agentic
        #    /generate admitted between our flush and the release crashes the
        #    scheduler. relax has no hard barrier guaranteeing all agentic
        #    sessions are quiesced before offload, so /pause_generation
        #    (mode="abort", the default) is issued first: it stops the scheduler
        #    from admitting new requests for the whole offloaded window AND
        #    aborts everything in flight. Admission is re-opened by
        #    continue_generation in resume_memory_occupation, after weights + KV
        #    cache are back. We still abort on each retry as a fallback in case
        #    the pause did not take (best-effort). Safe because the batch's
        #    reward/judge is already computed by offload time — no in-flight
        #    GenRM request needs to survive.
        #
        # 2. Unbounded hang. Every HTTP call must have a timeout and the whole
        #    drain must be bounded by a wall-clock deadline. Otherwise a wedged
        #    scheduler blocks rank-0 in ray.get() forever and every other rank
        #    stalls at the downstream offload barrier — a silent training hang.
        if self.node_rank == 0:
            deadline = time.monotonic() + _GENRM_OFFLOAD_DRAIN_TIMEOUT_S
            self._pause_generation_for_offload(deadline)
            connect_errors = 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timeout while draining GenRM before release.")
                self.abort_requests(timeout=max(_MIN_HTTP_TIMEOUT_S, deadline - time.monotonic()))
                try:
                    resp = requests.get(
                        f"http://{self.server_host}:{self.server_port}/flush_cache",
                        timeout=max(_MIN_HTTP_TIMEOUT_S, deadline - time.monotonic()),
                    )
                    if resp.status_code == 200:
                        break
                    connect_errors = 0
                except requests.exceptions.ConnectionError as e:
                    # requests wraps urllib3's NewConnectionError, so catching the
                    # latter here would never fire and a dead engine would be
                    # retried until the drain deadline.
                    connect_errors += 1
                    logger.warning(
                        f"Cannot reach {self.server_host}:{self.server_port}/flush_cache while "
                        f"draining GenRM ({connect_errors}/{_MAX_CONSECUTIVE_CONNECT_ERRORS}): {e}"
                    )
                    if connect_errors >= _MAX_CONSECUTIVE_CONNECT_ERRORS:
                        raise ConnectionError(
                            f"GenRM engine {self.server_host}:{self.server_port} unreachable while "
                            f"draining before release ({connect_errors} consecutive connection "
                            f"errors) — the server process is most likely dead."
                        ) from e
                except Exception as e:  # noqa: BLE001
                    connect_errors = 0
                    logger.info(f"Error flushing GenRM cache: {e}")
                time.sleep(1)
        return self._make_request("release_memory_occupation", timeout=_GENRM_OFFLOAD_RELEASE_TIMEOUT_S)

    def resume_memory_occupation(self, tags: list[str] = None):
        result = super().resume_memory_occupation(tags=tags)
        # Re-open admission that release_memory_occupation closed via
        # /pause_generation. Only after a full resume (weights + KV cache back):
        # GenRM always full-resumes, but the ``not tags`` guard prevents
        # re-enabling generation before KV cache exists if a partial
        # (weights-only) resume is ever introduced. Not swallowed — if the
        # engine stays paused, GenRM silently stops serving, so fail loudly.
        if self.node_rank == 0 and not tags:
            self.continue_generation(timeout=_SGLANG_HTTP_ATTEMPT_TIMEOUT_S)
        return result

    def _pause_generation_for_offload(self, deadline: float) -> None:
        """Best-effort /pause_generation (abort mode) before draining for
        offload.

        Never raises: if the pause does not take, the per-retry abort in
        release_memory_occupation still drains the engine — the pause only
        additionally closes the flush->release admission window.
        """
        try:
            self.pause_generation(timeout=max(_MIN_HTTP_TIMEOUT_S, deadline - time.monotonic()))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"GenRM pause_generation before offload failed (continuing to drain): {e}")


def _enable_draft_weights_cpu_backup(args, sglang_overrides: dict | None = None) -> bool:
    if getattr(args, "enable_mtp_training", False):
        return False

    speculative_algorithm = getattr(args, "sglang_speculative_algorithm", None)
    if sglang_overrides and "speculative_algorithm" in sglang_overrides:
        speculative_algorithm = sglang_overrides["speculative_algorithm"]
    return speculative_algorithm is not None


def _compute_genrm_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
):
    """Compute server arguments for genRM engine.

    This is similar to _compute_server_args but uses genrm-specific arguments:
    - model_path from genrm_model_path
    - tp_size from genrm_num_gpus_per_engine
    - max_total_tokens from genrm_max_context_len
    - max_decode_steps from genrm_max_response_len
    - sampling parameters from genrm_temperature, genrm_top_p, genrm_top_k
    """
    nnodes = max(1, args.genrm_num_gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)

    hf_model_path = args.genrm_model_path
    kwargs = {
        "model_path": hf_model_path if is_s3_uri(hf_model_path) else os.path.normpath(hf_model_path),
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": args.genrm_num_gpus_per_engine,
        "dp_size": args.genrm_engine_config.get("dp_size", 1),
        "pp_size": args.genrm_engine_config.get("pp_size", 1),
        "ep_size": args.genrm_engine_config.get("ep_size", 1),
        # Pin moe_dense_tp_size for the genRM so it does NOT inherit the global
        # --sglang-moe-dense-tp-size (set to 1 for the MoE actor/rollout). The genRM
        # is a standalone dense model at tp_size>1; inheriting moe_dense_tp_size=1
        # shards its dense layers as TP=1 while it runs tp_size=2 → corrupted weights
        # → garbage judgements (all rewards 0). None = use tp_size (correct default).
        "moe_dense_tp_size": args.genrm_engine_config.get("moe_dense_tp_size", None),
        # # context and response length
        # "max_total_tokens": args.genrm_engine_config['max_total_tokens'],
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": False,
        "enable_draft_weights_cpu_backup": _enable_draft_weights_cpu_backup(args, args.genrm_engine_config),
        # GenRM Only
        "enable_weights_cpu_backup": True,
        # The global load format belongs to policy rollout engines. GenRM must
        # load real weights unless its own engine config explicitly overrides it.
        "load_format": "auto",
    }
    # Allow per-genrm SGLang mem_fraction_static via --genrm-engine-config; this overrides
    # the global --sglang-mem-fraction-static below so rollout and genrm can share GPUs.
    if "mem_fraction_static" in args.genrm_engine_config:
        kwargs["mem_fraction_static"] = args.genrm_engine_config["mem_fraction_static"]

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert disaggregation_bootstrap_port is not None, (
            "disaggregation_bootstrap_port must be set for prefill worker"
        )
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # Per-genrm overrides from --genrm-engine-config. Applied after base args
    # and sglang_* defaults so user-supplied keys take highest priority. Keys
    # not recognized by the installed SGLang ServerArgs are dropped with a
    # warning rather than causing a TypeError at ServerArgs(**kwargs).
    server_arg_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    for key, value in (args.genrm_engine_config or {}).items():
        if key not in server_arg_fields:
            logger.info(
                f"Warning: --genrm-engine-config key {key!r} is not a ServerArgs field in the "
                f"installed SGLang; dropping."
            )
            continue
        if key in kwargs and kwargs[key] != value:
            logger.info(f"genrm_engine_config: overriding {key}={kwargs[key]} -> {value} (rank={rank})")
        kwargs[key] = value
        unused_keys.discard(key)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)

    hf_model_path = args.hf_checkpoint
    kwargs = {
        "model_path": hf_model_path if is_s3_uri(hf_model_path) else os.path.normpath(hf_model_path),
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": _gpus_per_engine // args.sglang_pp_size,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # MTP training syncs draft weights from the actor; otherwise only speculative
        # rollout needs draft weights backup for checkpoints without MTP weights.
        "enable_draft_weights_cpu_backup": _enable_draft_weights_cpu_backup(args, sglang_overrides),
        "enable_metrics": True,
    }

    # LoRA adapter mode: launch the engine with SGLang's runtime LoRA serving so the
    # trained adapter can be pushed each step via load_lora_adapter (see
    # UpdateWeightFromTensor._push_lora_adapter) and selected at generation via lora_path.
    if is_lora_enabled(args) and getattr(args, "lora_adapter_mode", False):
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = args.lora_rank
        # SGLang matches modules by leaf name; CLI holds canonical Megatron names. Fused
        # projections that SGLang groups differently from HF (GDN in_proj -> in_proj_qkvz)
        # must use the engine flavor, otherwise the module is never wrapped.
        kwargs["lora_target_modules"] = convert_megatron_to_sglang_target_modules(args.lora_target_modules)
        # We serve exactly one policy adapter. max_loaded_loras >= max_loras_per_batch is a
        # SGLang startup requirement; 2 leaves room for the unload->reload overlap.
        kwargs["max_loras_per_batch"] = 1
        kwargs["max_loaded_loras"] = 2
        # Mandatory: base is synced once, so it must survive colocate sleep/wake. Without CPU
        # backup, release_memory_occupation drops the GPU pages and base becomes garbage after
        # the first wake (other modes re-push full base every step and never notice).
        kwargs["enable_weights_cpu_backup"] = True

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert disaggregation_bootstrap_port is not None, (
            "disaggregation_bootstrap_port must be set for prefill worker"
        )
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    server_arg_fields = dataclasses.fields(ServerArgs)
    server_arg_field_names = {attr.name for attr in server_arg_fields}
    unused_keys = set(kwargs.keys())
    for attr in server_arg_fields:
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # Per-engine-group overrides from --sglang-config YAML.
    # Applied after base args so they take highest priority.
    if sglang_overrides:
        for key, value in sglang_overrides.items():
            if key in kwargs:
                logger.info(f"sglang_overrides: overriding {key}={kwargs[key]} -> {value} (rank={rank})")
            kwargs[key] = value
            unused_keys.discard(key)

    if (
        "cuda_graph_backend_prefill" in server_arg_field_names
        and kwargs.get("enable_memory_saver")
        and kwargs.get("cuda_graph_backend_prefill") is None
    ):
        # Breakable is SGLang's default prefill backend on CUDA, but it is incompatible with memory saver mode.
        kwargs["cuda_graph_backend_prefill"] = "disabled"

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "mem_fraction_static",
]
