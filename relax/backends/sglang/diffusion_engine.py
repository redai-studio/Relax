# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SGLang native-diffusion generation engine (RL rollout + weight update).

Drives SGLang multimodal-gen's RL API. **Rollout is fully upstreamed** in
v0.5.12.post1: the diffusion HTTP server already exposes ``POST /rollout/generate``
(returning the DiT trajectory, per-step log-probs and frozen conditions) — no
patch needed there. **Weight update is the one gap**: upstream only ships disk
reload; the fast in-memory CUDA-IPC path (``/update_weights_from_tensor``,
``/set_lora_from_tensor``) is added to the diffusion scheduler by a focused Relax
patch (design doc §10 / §9). This engine drives both so ``RolloutManager``
creates it like the text ``SGLangEngine``.

The server always runs as a framework-managed ``multiprocessing.Process``
executing SGLang's own diffusion ``launch_server`` (mirroring the text
``SGLangEngine``'s ``launch_server_process``). The child owns the scheduler
workers, so ``kill_process_tree`` reaps the whole tree cleanly on teardown
(single- and multi-GPU alike).

Tensors on the wire are SGLang's base64 envelope (``{"__tensor__": True, ...}``);
we reuse SGLang's own ``_maybe_deserialize`` so decoding is byte-identical. The
response is mapped to the flat dict the model adapter's ``pack_trajectory``
consumes: ``trajectory_latents`` / ``timesteps`` / ``rollout_log_probs`` /
``denoising_env`` / ``sde_indices`` / ``policy_version`` / ``height`` / ``width``.
"""

from __future__ import annotations

import os
import time
from dataclasses import fields
from typing import Any, Dict, List, Mapping, Optional

from relax.distributed.ray.ray_actor import RayActor
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

__all__ = ["SGLangNativeGenerationEngine"]

# Non-tunable transport bounds. Neither is a CLI argument: the health wait covers
# a cold DiT + VAE + text-encoder load from a network filesystem, and the request
# timeout has to outlast a full multi-step rollout batch.
_HEALTH_TIMEOUT_S = 1800.0
_REQUEST_TIMEOUT_S = 3600.0
_SGLANG_DIFFUSION_PATCH_PATH = "docker/patch/sglang/v0.5.15.post1.patch"
_REQUIRED_DIFFUSION_SAMPLING_FIELDS = frozenset(
    {
        "initial_noise_group_ids",
        "initial_noise_latent_shape",
        "initial_noise_seed",
        "denoise_seeds",
        "sigmas",
        "timesteps",
    }
)


def _verify_diffusion_patch_contract() -> None:
    """Fail fast when the pinned SGLang tree is missing Relax's static
    patch."""
    import inspect

    try:
        import sglang.multimodal_gen.configs.pipeline_configs.qwen_image as qwen_image
        import sglang.multimodal_gen.configs.post_training.rl_rollout as rl_rollout
        import sglang.multimodal_gen.configs.sample.sampling_params as sp_mod
        import sglang.multimodal_gen.runtime.entrypoints.utils as utils_mod
        import sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete as scheduler_mod
        import sglang.multimodal_gen.runtime.pipelines_core.stages.denoising as denoising_mod
        import sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation as latent_mod
        import sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin as scheduler_rl_mod
    except ImportError as exc:
        raise RuntimeError(
            "SGLang diffusion post-training modules are not importable. "
            f"Re-check that {_SGLANG_DIFFUSION_PATCH_PATH} is applied to the pinned SGLang image."
        ) from exc

    failures: list[str] = []

    sampling_fields = {field.name for field in fields(sp_mod.SamplingParams)}
    missing_fields = sorted(_REQUIRED_DIFFUSION_SAMPLING_FIELDS - sampling_fields)
    if missing_fields:
        failures.append(f"SamplingParams missing fields: {missing_fields}")

    if "dance" not in tuple(getattr(rl_rollout, "_VALID_ROLLOUT_SDE_TYPES", ())):
        failures.append("rollout_sde_type='dance' is not accepted by rl_rollout")

    if not hasattr(denoising_mod, "_make_step_generators"):
        failures.append("denoising stage missing driver denoise seed support")
    if not hasattr(latent_mod, "_driver_xt_recipe"):
        failures.append("latent preparation stage missing driver x_T support")

    flow_sde_sampling = getattr(scheduler_rl_mod.SchedulerRLMixin, "flow_sde_sampling", None)
    flow_consts = getattr(getattr(flow_sde_sampling, "__code__", None), "co_consts", ())
    if flow_sde_sampling is None or "dance" not in flow_consts:
        failures.append("SchedulerRLMixin.flow_sde_sampling missing dance SDE support")

    set_timesteps = getattr(scheduler_mod.FlowMatchEulerDiscreteScheduler, "set_timesteps", None)
    try:
        set_timesteps_source = inspect.getsource(set_timesteps)
    except (OSError, TypeError):
        set_timesteps_source = ""
    if "Driver-supplied sigmas" not in set_timesteps_source:
        failures.append("FlowMatchEulerDiscreteScheduler.set_timesteps still shifts driver sigmas")

    if getattr(qwen_image, "_QWEN_MAX_SEQUENCE_LENGTH", None) != 512:
        failures.append("Qwen-Image text encoder max sequence length patch missing")

    try:
        prepare_source = inspect.getsource(utils_mod.prepare_request)
    except (OSError, TypeError):
        prepare_source = ""
    if "initial_noise_group_ids" not in prepare_source or "torch.as_tensor" not in prepare_source:
        failures.append("prepare_request does not preserve driver rollout sampling fields")

    if failures:
        raise RuntimeError(
            "SGLang diffusion static patch contract is incomplete: "
            + "; ".join(failures)
            + f". Apply {_SGLANG_DIFFUSION_PATCH_PATH} to the pinned SGLang image before launching rollout."
        )


def _launch_diffusion_server(server_args_dict: Dict[str, Any], cuda_visible_devices: Optional[str]) -> None:
    """``multiprocessing.Process`` target: run SGLang's own diffusion
    ``launch_server`` in a framework-managed child.

    Mirrors the text ``SGLangEngine``'s ``launch_server_process`` /
    ``_launch_server_with_patches``: SGLang's ``launch_server`` spawns its
    ``sgl_diffusion::scheduler_*`` workers as children of THIS process and then
    blocks in uvicorn, so a single ``kill_process_tree`` on the parent pid reaps
    the whole tree (single- and multi-GPU alike).

    Runs in a freshly spawned interpreter. The diffusion server has no
    ``base_gpu_id`` knob (workers call ``set_device(local_rank)``), so the
    engine's physical GPU block is pinned here via ``CUDA_VISIBLE_DEVICES`` before
    any torch import.
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # Best-effort defense for the ABNORMAL teardown path only (engine Ray actor
    # crashes / is ray.kill'd without a graceful dispose): ask the kernel to
    # SIGTERM us when the parent dies so uvicorn can try to shut down and the
    # ``finally`` below can try to reap the workers. This is NOT the primary
    # reaper — the reliable one is the engine's ``_kill_server`` (graceful dispose
    # path), which SIGKILLs the whole tree by pid while the workers are still
    # attached. Under SIGTERM the reap is not guaranteed (uvicorn may exit before
    # the finally runs), so we never rely on it alone.
    try:
        import ctypes
        import signal as _signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, _signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass

    # Static patch contract check: a miss raises, this process dies, and
    # ``_wait_healthy`` reports "server exited early" instead of the job quietly
    # training on a stock rollout.
    _verify_diffusion_patch_contract()

    from sglang.multimodal_gen.runtime.launch_server import launch_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    # Build via ServerArgs.from_dict — the same path prepare_server_args →
    # from_cli_args uses — so the model-specific pipeline_config is resolved from
    # model_index.json (e.g. QwenImageVAEConfig, which carries the VAE arch fields
    # like input_channels). Plain ``ServerArgs(**dict)`` leaves the base
    # PipelineConfig, and the VAE loader then fails on the missing arch fields.
    server_args = (
        ServerArgs.from_dict(server_args_dict) if hasattr(ServerArgs, "from_dict") else ServerArgs(**server_args_dict)
    )
    try:
        launch_server(server_args)
    finally:
        # Reap the spawned ``sgl_diffusion::scheduler_*`` workers when the server
        # returns (SIGTERM / crash / normal exit). The upstream ``launch_server``
        # __main__ does this in its own ``finally``; running ``launch_server``
        # directly as an mp target skips that, so orphaned workers would keep the
        # GPU pinned. include_parent=False: we ARE the parent here.
        try:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(os.getpid(), include_parent=False)
        except Exception:
            pass


_SGLANG_DESERIALIZER: Any = None
_SGLANG_DESERIALIZER_RESOLVED = False


def _sglang_deserializer() -> Any:
    """Resolve SGLang's ``_maybe_deserialize`` once; ``None`` when unavailable.

    Resolution is cached because a partially-failed import of the
    ``multimodal_gen`` namespace package does not reliably raise
    ``ImportError`` on a retry: the first failure can drop
    ``sglang.multimodal_gen`` from ``sys.modules`` while leaving ``…runtime``
    behind, and the next attempt then dies with a ``KeyError`` out of
    ``_NamespacePath._recalculate``. Hence the broad except *around the import
    only* — see :func:`_deserialize` for why the decode call itself is
    deliberately not guarded.
    """
    global _SGLANG_DESERIALIZER, _SGLANG_DESERIALIZER_RESOLVED

    if not _SGLANG_DESERIALIZER_RESOLVED:
        try:
            from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import _maybe_deserialize

            _SGLANG_DESERIALIZER = _maybe_deserialize
        except Exception as exc:  # noqa: BLE001 - import machinery, not a decode failure
            logger.info(f"SGLang tensor decoder unavailable ({exc}); using the local base64 decoder.")
            _SGLANG_DESERIALIZER = None
        _SGLANG_DESERIALIZER_RESOLVED = True
    return _SGLANG_DESERIALIZER


def _deserialize(obj: Any) -> Any:
    """Decode SGLang's base64 tensor envelopes recursively.

    Uses SGLang's own helper when importable (production / node); falls back to
    a byte-compatible local decoder so unit tests run without SGLang installed.
    Only the *import* is tolerated: a decode failure means the wire format
    diverged (corrupt payload, unknown dtype, shape mismatch after an sglang
    bump) and must surface, not be papered over by the local decoder silently
    producing something else.
    """
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            if isinstance(obj.get("data"), str):
                return _local_maybe_deserialize(obj)
            decoder = _sglang_deserializer()
            if decoder is None:
                return _local_maybe_deserialize(obj)
            return decoder(obj)
        return {key: _deserialize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_deserialize(value) for value in obj]
    return obj


def _local_maybe_deserialize(obj: Any) -> Any:
    import base64

    import torch

    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            if isinstance(obj.get("data"), bytes):
                from safetensors.torch import load

                return load(obj["data"])["t"]
            dtype = getattr(torch, str(obj["dtype"]).replace("torch.", ""))
            raw = base64.b64decode(obj["data"])
            t = torch.frombuffer(bytearray(raw), dtype=dtype)
            return t.reshape(obj["shape"]) if obj.get("shape") else t
        return {k: _local_maybe_deserialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_local_maybe_deserialize(v) for v in obj]
    return obj


class SGLangNativeGenerationEngine(RayActor):
    """Ray actor driving SGLang's diffusion rollout + weight-update HTTP
    API."""

    def __init__(
        self,
        args,
        rank,
        worker_type: str = "regular",
        base_gpu_id: Optional[int] = None,
        sglang_overrides: Optional[dict] = None,
        num_gpus_per_engine: Optional[int] = None,
        register_sigterm_handler: bool = False,
    ) -> None:
        # Signature mirrors SGLangEngine so RolloutManager builds both the same way.
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self._url: Optional[str] = None
        self.process: Optional[Any] = None  # multiprocessing.Process running launch_server
        self._active_version = 0
        self._active_weight_manifest_sha256 = ""
        self._weight_updating = False

    # -- lifecycle ------------------------------------------------------------

    def init(self, host: Optional[str] = None, port: Optional[int] = None, **engine_args) -> Dict[str, Any]:
        """Launch the diffusion HTTP server; wait until healthy.

        ``engine_args`` absorbs the rest of ``RolloutManager``'s uniform init
        payload (router/DCS registration flags, ``nccl_port``, ...); only
        ``nccl_port`` is consumed, by ``_compute_server_args``.
        """
        self._url = self._launch_server(host, port, engine_args)
        self._wait_healthy(timeout=_HEALTH_TIMEOUT_S)
        return {"rank": self.rank, "url": self._url, "active_version": self._active_version}

    def get_base_gpu_id(self) -> Optional[int]:
        """Physical GPU this engine is pinned to (``CUDA_VISIBLE_DEVICES``
        base).

        The colocate CUDA-IPC weight sync uses this to pair each FSDP rank with
        the engine on the SAME physical GPU: the RolloutManager assigns
        ``base_gpu_id`` from the placement group's *reordered* GPU ids (a
        permutation of 0..N-1), so an engine's list position does NOT equal its
        physical GPU. Selecting the engine by matching ``base_gpu_id`` to the
        rank's device is what keeps the IPC handle's device_uuid valid.
        """
        return None if self.base_gpu_id is None else int(self.base_gpu_id)

    def _launch_server(self, host: Optional[str], port: Optional[int], engine_args: Mapping[str, Any]) -> str:
        """Launch SGLang's diffusion ``launch_server`` in a managed mp.Process.

        Mirrors the text ``SGLangEngine._init_normal`` →
        ``launch_server_process``: a framework-owned
        ``multiprocessing.Process`` (spawn — the Ray actor may already have
        CUDA initialized) runs SGLang's own ``launch_server``. The scheduler
        workers are children of that process, so ``_kill_server``'s
        ``kill_process_tree`` reaps the whole tree on teardown — no orphaned
        ``sgl_diffusion::scheduler_*`` processes wedging the node.
        """
        import multiprocessing as mp

        host = host or "127.0.0.1"
        if port is None:
            _ip, port = self._get_current_node_ip_and_free_port(start_port=21000)
        num_gpus = int(self.num_gpus_per_engine or getattr(self.args, "rollout_num_gpus_per_engine", 1))

        server_args_dict = self._compute_server_args(host, int(port), num_gpus, engine_args)

        # Pin this engine's physical GPU block by REORDERING CUDA_VISIBLE_DEVICES
        # so the block comes first, while keeping every GPU visible. The diffusion
        # server's (non-disagg) worker calls ``set_device(local_rank)`` with
        # local_rank = 0..num_gpus-1, so putting our block at CVD positions
        # 0..num_gpus-1 lands the model on the right physical GPU(s) — and no CVD
        # would collapse every single-GPU engine onto physical GPU 0 (set_device(0)).
        # Keeping the REST of the GPUs visible (not a 1-GPU scope) is what makes the
        # colocate CUDA-IPC weight sync work: the receiver's
        # ``_device_from_maybe_uuid`` iterates the visible devices to map the
        # sender's GPU uuid back to a local index. A 1-GPU scope only sees its own
        # uuid and fails deserialization with "Invalid device_uuid".
        #
        # ``--num-gpus-per-node`` (default 8) is the node width. Do NOT fall back
        # to a literal 8 when it is unset/zero: on a 4-GPU node that silently emits
        # device ids 4..7 and only works by accident.
        total_gpus = int(getattr(self.args, "num_gpus_per_node", 0) or 0)
        if total_gpus <= 0:
            raise RuntimeError(
                f"diffusion engine[{self.rank}] cannot pin CUDA_VISIBLE_DEVICES: args.num_gpus_per_node is "
                f"{getattr(self.args, 'num_gpus_per_node', None)!r}. Set --num-gpus-per-node to this node's GPU count."
            )
        cuda_visible_devices: Optional[str] = self._cuda_visible_devices(self.base_gpu_id, num_gpus, total_gpus)

        logger.info(
            f"SGLangNativeGenerationEngine[{self.rank}] launching diffusion server "
            f"(num_gpus={num_gpus}, base_gpu_id={self.base_gpu_id}, CUDA_VISIBLE_DEVICES={cuda_visible_devices}, "
            f"args={server_args_dict})"
        )
        # spawn (not fork): the Ray actor may have CUDA initialized, and forking a
        # CUDA context is unsafe — same reason the text launch_server_process spawns.
        ctx = mp.get_context("spawn")
        self.process = ctx.Process(
            target=_launch_diffusion_server,
            args=(server_args_dict, cuda_visible_devices),
            name=f"sglang-diffusion-engine-{self.rank}",
        )
        self.process.start()
        return f"http://{host}:{port}"

    @staticmethod
    def _cuda_visible_devices(base_gpu_id: Optional[int], num_gpus: int, total_gpus: int) -> Optional[str]:
        """Reordered CUDA_VISIBLE_DEVICES: this engine's physical GPU block
        first, then every other GPU — so ALL GPUs stay visible.

        The diffusion server's non-disagg worker calls
        ``set_device(local_rank)`` with ``local_rank`` in ``0..num_gpus-1``, so
        placing the block at CVD positions ``0..num_gpus-1`` pins the model to
        physical ``base..base+num_gpus-1``. The remaining GPUs are appended
        (not dropped) so the colocate CUDA-IPC weight sync can still resolve
        any actor rank's source-tensor uuid on the receiver
        (``_device_from_maybe_uuid`` iterates the visible devices). Returns
        ``None`` when unpinned.
        """
        if base_gpu_id is None:
            return None
        base = int(base_gpu_id)
        if base < 0 or base + num_gpus > total_gpus:
            raise RuntimeError(
                f"diffusion engine GPU block [{base}, {base + num_gpus}) does not fit a node of {total_gpus} GPUs; "
                "check --num-gpus-per-node against the node's actual GPU count."
            )
        block = list(range(base, base + num_gpus))
        rest = [g for g in range(total_gpus) if g not in block]
        return ",".join(str(g) for g in block + rest)

    def _compute_server_args(
        self, host: str, port: int, num_gpus: int, engine_args: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Build the diffusion ``ServerArgs`` kwargs (mirrors
        ``_compute_server_args``).

        Only the fields the diffusion ``ServerArgs`` needs: model + ports +
        parallelism. Sampling (height / steps / guidance_scale) is per-request
        (the RolloutRequest body), not a server arg. Parallelism beyond
        ``num_gpus`` is derived by ``ServerArgs.__post_init__``
        (tp/dp/sp/ulysses); raw overrides can be passed via
        ``sglang_overrides``.
        """
        kwargs: Dict[str, Any] = {
            "model_path": os.path.normpath(self._resolve_model_path()),
            "trust_remote_code": True,
            "host": host,
            "port": port,
            "num_gpus": num_gpus,
        }
        # Distinct torch-distributed master port per engine. The diffusion launcher
        # defaults master_port to a fixed value, so N single-GPU servers on one node
        # would all collide on it. RolloutManager already allocated a distinct
        # ``nccl_port`` per engine in addr_and_ports — reuse it as master_port.
        master_port = engine_args.get("nccl_port")
        if master_port is not None:
            kwargs["master_port"] = int(master_port)
        # SGLang otherwise discovers scheduler_port inside each concurrently
        # launched server. That check-then-bind races on a colocated node: two
        # servers can select the same port and one scheduler then executes both
        # engines' requests. The rollout allocator already reserves a distinct
        # dist_init_addr port (or port range) for every engine. Native diffusion
        # does not consume that text-engine rendezvous address, so reuse its port
        # for the internal scheduler instead of performing another racy lookup.
        dist_init_addr = engine_args.get("dist_init_addr")
        if dist_init_addr:
            try:
                kwargs["scheduler_port"] = int(str(dist_init_addr).rsplit(":", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid diffusion engine dist_init_addr: {dist_init_addr!r}") from exc
        revision = getattr(self.args, "model_revision", None)
        if revision:
            kwargs["revision"] = str(revision)
        # Multi-GPU diffusion server: force sequence parallelism (ulysses), NOT the
        # auto-enabled CFG-parallel. A >1-GPU server auto-enables CFG-parallel
        # (splits cond/uncond across ranks), which 500s a no-CFG request — and the
        # rollout is always no-CFG (the adapter rejects guidance_scale != 1.0 at
        # request-build time), so it would always fail. An explicit ulysses policy
        # makes SGLang's auto-tuner skip cfg-parallel; ulysses (sequence parallel)
        # REPLICATES the transformer across the engine's GPU block, so the
        # per-rank→per-worker full-tensor CUDA-IPC weight sync (one blob per engine
        # GPU) stays valid. Overridable via sglang_overrides below.
        #
        # NOTE: >1 GPU per engine is currently rejected at rollout time by the
        # SGLang diffusion patch's dance SDE path (the per-step log-prob is
        # computed over the unsharded noise).
        if int(num_gpus) > 1:
            kwargs["ulysses_degree"] = int(num_gpus)
        # LoRA adapter-mode rollout: tell the server which Linears to wrap. Left
        # unset, LoRAPipeline.is_target_layer() returns True for EVERY module and
        # each wrapper clones its base weight to host RAM — a second full CPU copy
        # of the DiT. `lora_path` is deliberately NOT set: the pipeline converts
        # lazily on the first set_lora_from_tensors, so the server boots (and
        # serves the base) exactly as it does for full FT.
        if getattr(self.args, "lora_adapter_mode", False):
            lora_targets = self._resolve_lora_target_modules()
            # SGLang matches by substring against its module names. Keep the
            # same suffixes PEFT used train-side; shortening "attn.to_out.0" to
            # a leaf would lose that target entirely.
            if lora_targets:
                kwargs["lora_target_modules"] = sorted({str(t) for t in lora_targets})
        # Per-engine raw ServerArgs overrides (dict) — mirrors SGLangEngine's
        # sglang_overrides so future flags need no code change here. model_path is
        # resolved above; skip it and any None-valued override so they cannot clobber
        # a real value.
        for key, value in (self.sglang_overrides or {}).items():
            if key == "model_path" or value is None:
                continue
            kwargs[key] = value
        return kwargs

    def _resolve_lora_target_modules(self) -> List[str]:
        explicit = getattr(self.args, "lora_target_modules", None)
        if explicit:
            return [str(t) for t in explicit]

        adapter_path = getattr(self.args, "model_adapter_path", None)
        if not adapter_path:
            return []

        try:
            from relax.utils.misc import load_function

            adapter_cls = load_function(adapter_path)
            adapter = adapter_cls()
        except Exception as exc:  # noqa: BLE001 - best-effort server memory hint
            logger.warning(f"Could not resolve LoRA target modules from adapter {adapter_path!r}: {exc}")
            return []

        targets = getattr(adapter, "lora_target_modules", None)
        return [str(t) for t in targets] if targets else []

    def _resolve_model_path(self) -> str:
        """Resolve the diffusion model directory across the framework's
        sources.

        The colocate framework injects ``sglang_overrides["model_path"]`` from
        ``args.hf_checkpoint`` (ModelConfig.resolve), which is None for a
        diffusion run — the DiT path uses ``args.model_path``, not
        hf_checkpoint. Fall through a precedence chain so a None override never
        clobbers the real path.
        """
        model_path = (
            (self.sglang_overrides or {}).get("model_path")
            or getattr(self.args, "model_path", None)
            or getattr(self.args, "sglang_hf_checkpoint", None)
            or getattr(self.args, "hf_checkpoint", None)
        )
        if not model_path:
            raise RuntimeError("diffusion engine has no model path — set --model-path (fsdp) or --hf-checkpoint.")
        return model_path

    def _wait_healthy(self, timeout: float) -> None:
        import requests

        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if self.process is not None and not self.process.is_alive():
                raise RuntimeError(f"diffusion server exited early (exitcode {self.process.exitcode}).")
            try:
                r = requests.get(f"{self._url}/health", timeout=5)
                if r.status_code == 200:
                    logger.info(f"SGLangNativeGenerationEngine[{self.rank}] healthy at {self._url}")
                    return
            except Exception as e:  # not up yet
                last_err = e
            time.sleep(3)
        raise TimeoutError(f"diffusion server not healthy within {timeout}s ({last_err}).")

    # -- generation -----------------------------------------------------------

    def generate_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """POST each RolloutRequest to /rollout/generate; return mapped
        trajectories.

        ``requests`` are ``RolloutRequest``-shaped dicts built by the model
        adapter's ``build_rollout_request``. SGLang has no cross-request
        dynamic batching for every diffusion pipeline, so requests are issued
        sequentially (design doc §9) and the caller shards across engines. The
        result is one trajectory per request, in request order.
        """
        return [self._map_responses(self._post("/rollout/generate", req), req) for req in requests]

    def _map_responses(self, resp: Any, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Map the single RolloutResponse in a server reply to an adapter
        sidecar dict.

        ``POST /rollout/generate`` is declared
        ``response_model=list[RolloutResponse]`` — a JSON *list* with one entry
        per generated candidate. The Qwen-Image adapter always sends
        ``num_outputs_per_prompt = 1`` (a batched group would collapse the
        group's samples onto one shared noise buffer upstream), so anything but
        exactly one entry means the server disagrees with the request and the
        group would be silently short.
        """
        entries = resp if isinstance(resp, list) else [resp]
        if not entries:
            raise RuntimeError("empty /rollout/generate response list.")
        if len(entries) != 1:
            raise RuntimeError(
                f"/rollout/generate returned {len(entries)} responses; the diffusion adapter always requests "
                "num_outputs_per_prompt=1, so exactly one was expected."
            )
        return self._map_response(entries[0], request)

    def _map_response(self, resp: Mapping[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
        """Map one (base64-decoded) RolloutResponse to the adapter sidecar
        dict."""
        resp = _deserialize(dict(resp))
        dit = resp.get("dit_trajectory") or {}
        rollout_log_probs = resp.get("rollout_log_probs")
        if isinstance(rollout_log_probs, Mapping):
            if "log_probs" not in rollout_log_probs:
                raise ValueError("rollout_log_probs mapping is missing required 'log_probs'.")
            rollout_log_probs = rollout_log_probs["log_probs"]
        out: Dict[str, Any] = {
            "request_id": resp.get("request_id"),
            "seed": resp.get("seed"),
            "policy_version": self._active_version,
            "weight_manifest_sha256": self._active_weight_manifest_sha256,
            "trajectory_latents": dit.get("latents"),  # [T+1, ...] (per-sample)
            "timesteps": dit.get("timesteps"),  # [T]
            "rollout_log_probs": rollout_log_probs,
            "denoising_env": resp.get("denoising_env"),
            # Only populated when the request sets SGLang's ``rollout_debug_mode``;
            # this passthrough is what makes the dance step's debug branch reach
            # the driver at all.
            "rollout_debug_tensors": resp.get("rollout_debug_tensors"),
            "generated_output": resp.get("generated_output"),
            # SDE step indices: prefer what the server actually used (the rollout
            # response returns them per the native-diffusion patch contract, either
            # top-level or nested in dit_trajectory); fall back to the requested
            # indices so a server that omits them still aligns the stored slots.
            "sde_indices": resp.get("sde_indices")
            or dit.get("sde_indices")
            or request.get("rollout_sde_step_indices"),
            "task": (request.get("_task") or None),
            # Echo the requested geometry. The trainer needs the true latent grid
            # to rebuild RoPE, and a packed sequence length cannot recover a
            # non-square grid on its own (h*w is not h and w) — the request is the
            # only source. ``validate_rollout_response`` fails fast without these.
            "height": request.get("height"),
            "width": request.get("width"),
        }
        return out

    # -- weight update --------------------------------------------------------
    #
    # Colocate hot path: in-memory tensor transport (CUDA IPC), mirroring SGLang
    # text `srt`'s update_weights_from_tensor. The endpoint is NOT in SGLang's
    # diffusion scheduler upstream (multimodal_gen only upstreams disk update) —
    # it is added by the focused Relax patch (design doc §10; see
    # examples/diffusion/readme.md). The NCCL-broadcast variant the text/Megatron
    # path uses has no diffusion caller and is deliberately not exposed here.

    def update_weights_from_tensor(
        self, serialized_named_tensors: Any, flush_cache: bool = False, target_modules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Receive a bucket of full tensors via CUDA IPC (colocated engine).

        ``target_modules`` MUST be set to the trainable module (e.g.
        ``["transformer"]``): without it the server collects every pipeline module
        and routes tensors by a ``"<module>."`` name prefix — the actor sends
        module-local names (``transformer_blocks.0.…``) which match no prefix, so
        every tensor is silently dropped and the weight update is a no-op.
        """
        body: Dict[str, Any] = {"serialized_named_tensors": serialized_named_tensors, "flush_cache": flush_cache}
        if target_modules is not None:
            body["target_modules"] = target_modules
        return self._post("/update_weights_from_tensor", body)

    def set_lora_from_tensors(
        self,
        named_tensors: Dict[str, Any],
        *,
        lora_name: str = "default",
        target: str = "transformer",
        strength: float = 1.0,
    ) -> Dict[str, Any]:
        """Install a freshly-trained LoRA adapter from in-memory tensors.

        Requires the combined SGLang patch at
        ``docker/patch/sglang/v0.5.15.post1.patch``. Stock SGLang's
        ``/v1/set_lora`` only reads a *file path*, and it short-circuits when the
        path is unchanged (``loaded_adapter_paths[nickname] != path``), so
        rewriting the same directory every step is a complete no-op after step 1.
        The patched endpoint takes tensors and always reapplies.

        The adapter is merged into the DiT weights server-side, from a pristine
        base snapshot taken at wrap time — so repeated merge/unmerge does not
        drift. That snapshot is also why adapter mode must never be combined with
        a full-weight sync: the sync would overwrite the live weights while the
        snapshot still holds the launch-time ones.
        """
        import base64
        import io

        import torch

        buffer = io.BytesIO()
        torch.save(named_tensors, buffer)
        body = {
            "lora_name": lora_name,
            "target": target,
            "strength": float(strength),
            "serialized_tensors": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        result = self._post("/set_lora_from_tensor", body)
        if not result.get("success", False):
            raise RuntimeError(f"set_lora_from_tensor failed: {result.get('message')}")
        return result

    def get_weights_checksum(self, module_names: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._post("/get_weights_checksum", {"module_names": module_names})

    def commit_weight_version(self, version: int, weight_manifest_sha256: str) -> Dict[str, Any]:
        """Record the active policy version after a successful weight
        update."""
        digest = str(weight_manifest_sha256)
        try:
            valid = len(digest) == 64 and int(digest, 16) >= 0
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("weight_manifest_sha256 must be a 64-character hexadecimal digest.")
        self._active_version = int(version)
        self._active_weight_manifest_sha256 = digest.lower()
        return {
            "active_version": self._active_version,
            "weight_manifest_sha256": self._active_weight_manifest_sha256,
        }

    def get_weight_version(self) -> int:
        return self._active_version

    # -- memory + health ------------------------------------------------------

    def release_memory_occupation(self, tags=None) -> Dict[str, Any]:
        """Offload the diffusion pipeline to CPU so the colocated actor can
        train.

        Added by the pinned Relax sglang patch. Failure is FATAL, not a
        warning: under ``--offload-rollout`` a swallowed 404/500 leaves the
        full pipeline resident and the colocated FSDP actor OOMs many steps
        later with nothing pointing back here.
        """
        result = self._post("/release_memory_occupation", {})
        self._raise_on_failure("release_memory_occupation", result)
        return result

    def resume_memory_occupation(self, tags=None) -> Dict[str, Any]:
        """Reload the pipeline to GPU.

        ``tags==[WEIGHTS]`` reloads only the DiT (used before the CUDA-IPC
        weight sync so actor + transformer fit). Fatal on failure for the same
        reason as :meth:`release_memory_occupation`: generating against a half-
        offloaded pipeline is not a recoverable state.
        """
        weights_only = False
        if tags:
            try:
                from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

                weights_only = all(t == GPU_MEMORY_TYPE_WEIGHTS for t in tags)
            except ImportError:
                weights_only = False
        result = self._post("/resume_memory_occupation", {"weights_only": weights_only})
        self._raise_on_failure("resume_memory_occupation", result)
        return result

    def _raise_on_failure(self, what: str, result: Mapping[str, Any]) -> None:
        """Turn an explicit ``{"success": False}`` body into an exception.

        HTTP-level failures already raise inside ``_post``; this covers the
        endpoints that answer 200 with a failure flag.
        """
        if isinstance(result, Mapping) and result.get("success") is False:
            raise RuntimeError(f"engine[{self.rank}] {what} failed: {result.get('message')}")

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the diffusion HTTP server.

        Returns True on HTTP 200 and RAISES otherwise — the same contract as
        the text ``SGLangEngine.health_generate``. Both shared callers rely on
        it: ``RolloutManager.healthcheck_engines`` only marks an engine failed
        when the Ray call raises, and ``_health_check_engines`` does
        ``all(results)``, which any dict would satisfy. Returning ``{"ok":
        False}`` made a hung or dead server pass the check forever.
        """
        import requests

        response = requests.get(f"{self._url}/health_generate", timeout=timeout)
        response.raise_for_status()
        return True

    # -- engine-pool contract -------------------------------------------------
    #
    # The text SGLangEngine exposes a wider surface consumed by the router,
    # elastic scale-out, the eviction monitor and CI fault injection. The
    # synchronous-colocate generative path never calls these, but leaving them
    # undefined turns any accidental use into a bare AttributeError deep inside a
    # Ray call. The cheap accessors are implemented for real; the router/elastic
    # ones fail loudly with an actionable message. `--autoscaler-config` is
    # additionally rejected at preflight (see fsdp/arguments.py).

    def get_url(self) -> Optional[str]:
        return self._url

    def get_rank(self) -> int:
        return self.rank

    def get_pid_and_node_id(self) -> Dict[str, Any]:
        import os

        import ray

        return {"pid": os.getpid(), "node_id": ray.get_runtime_context().get_node_id()}

    def flush_cache(self) -> Dict[str, Any]:
        """No-op: the diffusion server has no prefix/KV cache to flush."""
        return {"success": True, "message": "no cache on the diffusion server"}

    def pause_generation(self) -> Dict[str, Any]:
        """No-op: generation is driven request-by-request, never streamed."""
        return {"success": True}

    def continue_generation(self) -> Dict[str, Any]:
        return {"success": True}

    def set_weight_updating(self, updating: bool) -> Dict[str, Any]:
        """Record the weight-update flag.

        The colocate weight transaction already serializes against generation
        (the actor owns the engine while streaming), so this is bookkeeping
        only.
        """
        self._weight_updating = bool(updating)
        return {"success": True, "weight_updating": self._weight_updating}

    def is_evicted(self) -> bool:
        return False

    def _unsupported(self, name: str):
        raise NotImplementedError(
            f"{type(self).__name__}.{name}() is not implemented: the native diffusion engine supports only "
            "synchronous colocate rollout. Router registration, elastic scale-out and DCS weight transport "
            "are text-path features; do not enable --autoscaler-config or elastic rollout with "
            "--train-backend fsdp."
        )

    def register_to_router(self, *args, **kwargs):
        self._unsupported("register_to_router")

    def unregister_from_router(self, *args, **kwargs):
        self._unsupported("unregister_from_router")

    def register_dcs(self, *args, **kwargs):
        self._unsupported("register_dcs")

    def abort_requests(self, *args, **kwargs):
        self._unsupported("abort_requests")

    def check_weights(self, *args, **kwargs):
        self._unsupported("check_weights")

    def shutdown(self) -> None:
        self._kill_server()

    def __del__(self):  # best-effort reap if Serve tears the actor down abruptly
        try:
            self._kill_server()
        except Exception:
            pass

    def _kill_server(self) -> None:
        proc = getattr(self, "process", None)
        if proc is None:
            return
        pid = getattr(proc, "pid", None)
        if not proc.is_alive() or pid is None:
            self.process = None
            return
        # Reap the WHOLE tree by pid RIGHT NOW, while the parent (the uvicorn
        # mp.Process) is still alive and its ``sgl_diffusion::scheduler_*`` workers
        # are still attached as its children. ``kill_process_tree`` collects the
        # descendant pids first and then SIGKILLs each, so reparenting during the
        # kill is harmless — the same helper the text SGLangEngine relies on.
        #
        # Do NOT SIGTERM-and-wait first: if the parent exits before we reap, the
        # workers reparent to init (ppid=1) and, having renamed their proctitle to
        # ``sgl_diffusion::``, evade both a follow-up ``is_alive()`` check and the
        # launcher's ``pkill python``/``pkill sglang`` cleanup — orphaning a full
        # ~44GB CUDA process per GPU (observed, and the cause of the node-wedge /
        # reboot cycle). CUDA state is reclaimed cleanly by the OS on SIGKILL.
        try:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(pid)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        # Reap the mp.Process entry itself. Without the join it stays a zombie in
        # the Ray actor's process table until the actor exits — mp only clears the
        # child on join/exitcode.
        try:
            proc.join(timeout=10)
        except Exception:
            pass
        self.process = None

    # -- http -----------------------------------------------------------------

    def _post(self, path: str, body: Mapping[str, Any]) -> Any:
        import requests

        if self._url is None:
            raise RuntimeError(f"engine not initialized before POST {path}.")
        r = requests.post(f"{self._url}{path}", json=body, timeout=_REQUEST_TIMEOUT_S)
        if r.status_code >= 400:
            # Surface the server-side error body — the diffusion server returns
            # ``{"error": "<traceback>"}`` with the 500, which raise_for_status
            # would otherwise discard, leaving only an opaque "500 Server Error".
            logger.error(f"engine[{self.rank}] POST {path} -> {r.status_code}: {r.text[:2000]}")
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").partition(";")[0].strip().lower()
        if content_type == "application/msgpack":
            import msgspec

            return msgspec.msgpack.decode(r.content)
        return r.json()
