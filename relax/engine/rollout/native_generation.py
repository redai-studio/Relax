# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native generation rollout driver + data contracts.

This module owns the rollout side of native generative RL (design doc 8.2):

* :func:`generate_rollout` — the unified ``--rollout-function-path`` driver. It
  expands prompt groups from the standard ``RolloutDataSource``, asks the
  adapter to build DiffGenerator requests, dispatches them to the native
  generation engine, and keeps training media and trajectories in process/Ray
  memory while recording only lightweight references on each ``Sample``.
* :func:`convert_samples_to_train_data` — the ``--custom-convert-…`` hook. It
  emits the numeric-only TransferQueue row of design doc 5.4 (group index +
  trajectory slot + advantage + raw reward), so the TQ never carries pixels,
  latents or prompt strings.
* :func:`hydrate_micro_batches` — the actor-side inverse: it reads those numeric
  rows and rehydrates the referenced Ray objects (or legacy sidecars) into the tensor batches the
  FlowGRPO update consumes, joining each row's advantage to its candidate by
  ``trajectory_slot`` (never by row order — see the function docstring).

The converter / packing / hydrate functions are pure and CPU-testable, with one
deliberate exception: :func:`save_candidate_image` keeps a module-level flag so
the engine's decoded-image layout is logged exactly once per worker. The driver
orchestrates them against the live engines.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from relax.engine.rollout.base_types import RolloutFnTrainOutput
from relax.models.generative import ArtifactTrack, artifact_manifest_dict, resolve_sde_indices
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)

__all__ = [
    "generate_rollout",
    "evaluate_rollout",
    "convert_samples_to_train_data",
    "hydrate_micro_batches",
    "write_group_sidecar",
    "build_candidate_manifest",
    "save_candidate_image",
    "cache_candidate_image",
    "get_cached_candidate_image",
    "group_sidecar_path",
    "prune_stale_artifacts",
]


# ---------------------------------------------------------------------------
# Trajectory transport + manifest packing (design doc 5.2 / 5.3)
# ---------------------------------------------------------------------------


def group_sidecar_path(artifact_root: str, task: str, rollout_id: int, group_index: int) -> str:
    return os.path.join(
        artifact_root, task, f"rollout_{int(rollout_id):07d}", f"group_{int(group_index):08d}.safetensors"
    )


# Keep the current rollout plus this many previous ones. 2 leaves rollout N-1
# on disk while rollout N generates, so an actor still replaying the previous
# step's sidecars (fully-async mode) never has a file pulled out from under it.
_DEFAULT_ARTIFACT_RETENTION_ROLLOUTS = 2
_TRAJECTORY_REF_CACHE: Dict[Tuple[int, int], Any] = {}
_CANDIDATE_IMAGE_CACHE: Dict[str, Any] = {}
_CANDIDATE_IMAGE_ROLLOUT_IDS: Dict[str, int] = {}


def prune_stale_artifacts(args, rollout_id: int) -> None:
    """Release stale in-memory trajectories and delete old artifact
    directories.

    Current local-reward training keeps trajectories in Ray's object store and
    images in the reward worker's process. Files remain possible for eval,
    remote rewards, and legacy sidecar rows; they are needed only until that
    rollout's train/reward step has consumed them. The rollout driver owns both
    caches and directories, so it is also their safe reaper.

    ``--artifact-retention-rollouts <= 0`` disables pruning. Best-effort: a
    failed unlink must never abort a rollout.
    """
    keep = int(getattr(args, "artifact_retention_rollouts", _DEFAULT_ARTIFACT_RETENTION_ROLLOUTS))
    root = getattr(args, "artifact_root", None)
    if keep <= 0:
        return
    memory_cutoff = int(rollout_id) - max(keep, _DEFAULT_ARTIFACT_RETENTION_ROLLOUTS)
    for key in [key for key in _TRAJECTORY_REF_CACHE if key[0] <= memory_cutoff]:
        del _TRAJECTORY_REF_CACHE[key]
    for uri in [
        uri for uri, image_rollout_id in _CANDIDATE_IMAGE_ROLLOUT_IDS.items() if image_rollout_id <= memory_cutoff
    ]:
        _CANDIDATE_IMAGE_ROLLOUT_IDS.pop(uri, None)
        _CANDIDATE_IMAGE_CACHE.pop(uri, None)
    if not root:
        return
    cutoff = int(rollout_id) - keep
    if cutoff < 0:
        return
    import shutil

    task = str(getattr(args, "generation_task", "") or "")
    for base in (os.path.join(root, task), os.path.join(root, "eval", task)):
        try:
            names = os.listdir(base)
        except OSError:
            continue  # directory not created yet (first rollout) or not ours
        for name in names:
            if not name.startswith("rollout_") or not name[len("rollout_") :].isdigit():
                continue
            if int(name[len("rollout_") :]) > cutoff:
                continue
            shutil.rmtree(os.path.join(base, name), ignore_errors=True)


def _sampling_fingerprint(sampling: Mapping[str, Any], seed: int) -> str:
    """Provenance digest of the sampling geometry a rollout was drawn with.

    Constant for a whole rollout (the only per-rollout term is the resampled
    ``sde_indices``), so the driver computes it ONCE per rollout and reuses it
    for every group — a JSON dump + SHA256 per group was pure overhead on the
    driver's serial path.
    """
    payload = json.dumps({"sampling": dict(sampling), "seed": int(seed)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_group_sidecar(
    path: str,
    common: Mapping[str, torch.Tensor],
    conditions: Mapping[str, torch.Tensor],
    tracks: Mapping[str, torch.Tensor],
) -> str:
    """Write a flattened group trajectory sidecar (design doc 5.3).

    The nested ``common/`` / ``conditions/`` / ``tracks/`` namespaces are
    encoded as prefixed keys (safetensors is flat). Uses the runtime's atomic
    write order. ``conditions`` entries may carry a leading dim of 1 instead of
    the group size when they are identical across the group's candidates — see
    :func:`adapter_pack_group`.
    """
    from relax.backends.fsdp.runtime import write_trajectory_sidecar

    flat: Dict[str, torch.Tensor] = {}
    for k, v in common.items():
        flat[f"common.{k}"] = v
    for k, v in conditions.items():
        flat[f"conditions.{k}"] = v
    for k, v in tracks.items():
        flat[f"tracks.{k}"] = v
    return write_trajectory_sidecar(path, flat)


def _put_group_batch(
    rollout_id: int,
    group_index: int,
    common: Mapping[str, torch.Tensor],
    conditions: Mapping[str, torch.Tensor],
    tracks: Mapping[str, torch.Tensor],
) -> bytes:
    """Publish one deduplicated trajectory group through Ray shared memory.

    The serialized reference carries Ray's owner address.
    ``ObjectRef.binary()`` carries only the object ID, so reconstructing it in
    a trainer actor can wait forever because that actor does not know which
    worker owns the object.
    """
    import ray

    batch = {**common, **conditions, **tracks}
    ref = ray.put(batch)
    _TRAJECTORY_REF_CACHE[(int(rollout_id), int(group_index))] = ref
    return ray.cloudpickle.dumps(ref)


_LOGGED_CANDIDATE_IMAGE_SHAPE = False


def _log_candidate_image_shape(t: torch.Tensor) -> None:
    """Log the engine's decoded-image layout once per worker.

    The layout is an engine contract, not something the response declares, and
    getting it wrong is silent: a mis-read axis writes a valid-looking PNG that
    the reward model happily scores. One line per process makes the contract
    visible in the run log instead of inferable only from the artifacts.
    """
    global _LOGGED_CANDIDATE_IMAGE_SHAPE
    if not _LOGGED_CANDIDATE_IMAGE_SHAPE:
        _LOGGED_CANDIDATE_IMAGE_SHAPE = True
        logger.info(f"native generation: decoded image tensor shape from the engine = {tuple(t.shape)}")


def _candidate_image_array(image: Any):
    if image is None:
        return None
    import numpy as np

    t = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    t = t.detach().float().cpu()
    _log_candidate_image_shape(t)
    if t.ndim == 5:  # [B, C, T, H, W] -> first sample
        t = t[0]
    if t.ndim == 4:
        # SGLang's OutputBatch.output is [B, C, T, H, W] and rollout_api's
        # _build_response already indexes the batch, so a 4-D tensor arriving here
        # is [C, T, H, W] with T == 1 for images -- NOT [B, C, H, W]. Taking t[0]
        # on that keeps only the RED channel and the frame axis then reads as the
        # channel axis, so the artifact is written as a grayscale PNG and every
        # image-space reward scores a colourless image. Disambiguate on the frame
        # axis before falling back to the batch reading.
        if t.shape[0] in (1, 3) and t.shape[1] == 1:
            t = t[:, 0]  # [C, T=1, H, W] -> [C, H, W]
        else:
            t = t[0]  # [B, ...] -> first sample
    if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW -> HWC
        t = t.permute(1, 2, 0)
    arr = t.numpy()
    if float(arr.min()) < -0.01:  # [-1, 1] -> [0, 1]
        arr = (arr + 1.0) / 2.0
    if float(arr.max()) <= 1.01:  # [0, 1] -> [0, 255]
        arr = arr * 255.0
    # Round, don't truncate: diffusers' image_processor uses (x * 255).round(),
    # and a bare astype(uint8) floors every pixel (a systematic -0.5 LSB bias).
    arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:  # gray
        arr = arr[..., 0]
    return arr


def cache_candidate_image(
    image: Any, task: str, rollout_id: int, group_index: int, slot: int
) -> Tuple[str, Optional[int], Optional[int], str]:
    """Keep a training image in the rollout process until reward scoring."""
    arr = _candidate_image_array(image)
    if arr is None:
        return "", None, None, ""
    height, width = int(arr.shape[0]), int(arr.shape[1])
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    uri = f"memory://{task}/rollout_{int(rollout_id):07d}/group_{int(group_index):08d}_sample_{int(slot):04d}"
    _CANDIDATE_IMAGE_CACHE[uri] = arr
    _CANDIDATE_IMAGE_ROLLOUT_IDS[uri] = int(rollout_id)
    return uri, height, width, digest


def get_cached_candidate_image(uri: str, *, remove: bool = False):
    """Resolve a process-local image URI used by an in-process reward
    scorer."""
    if remove:
        _CANDIDATE_IMAGE_ROLLOUT_IDS.pop(uri, None)
        return _CANDIDATE_IMAGE_CACHE.pop(uri, None)
    return _CANDIDATE_IMAGE_CACHE.get(uri)


def save_candidate_image(
    image: Any, artifact_root: str, task: str, rollout_id: int, group_index: int, slot: int
) -> Tuple[str, Optional[int], Optional[int], str]:
    """Persist an eval/debug image to PNG; training uses
    :func:`cache_candidate_image`."""
    arr = _candidate_image_array(image)
    if arr is None:
        return "", None, None, ""
    height, width = int(arr.shape[0]), int(arr.shape[1])
    path = os.path.join(
        artifact_root,
        task,
        f"rollout_{int(rollout_id):07d}",
        f"group_{int(group_index):08d}_sample_{int(slot):04d}.png",
    )
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    import tempfile

    from PIL import Image

    fd, tmp_path = tempfile.mkstemp(prefix=".candidate-", suffix=".png", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            Image.fromarray(arr).save(f, format="PNG")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return path, height, width, digest


def build_candidate_manifest(
    *,
    task: str,
    sample_index: int,
    group_index: int,
    policy_version: int,
    artifact_tracks: List[ArtifactTrack],
    trajectory_uri: str,
    sampling_fingerprint: str,
    weight_manifest_sha256: str,
) -> Dict[str, Any]:
    return artifact_manifest_dict(
        task=task,
        sample_index=sample_index,
        group_index=group_index,
        policy_version=policy_version,
        outputs=artifact_tracks,
        trajectory_uri=trajectory_uri,
        sampling_fingerprint=sampling_fingerprint,
        weight_manifest_sha256=weight_manifest_sha256,
    )


# ---------------------------------------------------------------------------
# Rollout driver
# ---------------------------------------------------------------------------


def generate_rollout(args, rollout_id, data_source, data_system_client=None, evaluation=False):
    """Unified native generation driver (``--rollout-function-path``).

    Returns a :class:`RolloutFnTrainOutput` of ``Sample`` groups. Each sample
    carries in ``train_metadata`` the group index, trajectory slot, policy
    version and manifest the converter and actor need. Local rewards consume
    images from process memory and trainers hydrate trajectories from Ray's
    object store; durable artifacts are used only when the consumer is remote.
    """
    if evaluation:
        return evaluate_rollout(args, rollout_id)

    adapter = _load_adapter(args)
    engines = _get_engines(args, data_system_client)
    sampling = getattr(args, "sampling_config", None) or {}

    # Pin the trained/stochastic SDE steps ONCE per rollout and hand every request
    # the same explicit list. With --sampling-config sde_resample_per_rollout this
    # is a fresh draw each rollout (reference AllSDEScheduler), and resolving it here
    # rather than per request is what makes every candidate and every engine agree
    # on one set -- a per-request draw would put different candidates of the same
    # group on different trajectories and break the group's advantage baseline.
    rollout_sde = resolve_sde_indices(sampling, rollout_id)
    if rollout_sde:
        sampling = {**sampling, "sde_indices": rollout_sde}
        logger.info(f"[rollout {rollout_id}] trained SDE steps: {rollout_sde}")

    groups = _get_samples(data_source, args.rollout_batch_size)
    out_groups: List[List[Sample]] = []
    base_seed = int(getattr(args, "generation_seed", 1234))
    request_sampling_base = {**sampling, "_rollout_id": int(rollout_id), "_base_seed": base_seed}
    driver_xt = bool(sampling.get("driver_xt", True))
    # Constant for the whole rollout (see _sampling_fingerprint) — hoisted out of
    # the per-group loop.
    fp = _sampling_fingerprint(sampling, base_seed)
    prune_stale_artifacts(args, rollout_id)
    import time

    gen_t0 = time.time()
    # Dispatch a WAVE of groups at a time, with every candidate of the wave
    # spread across all engines.
    #
    # The old shape dispatched one group, waited for it, then did that group's
    # PNG encoding / sidecar write / manifest build with all 8 GPUs idle, then
    # dispatched the next. Batching a wave amortizes that driver-side serial work
    # over len(engines) groups instead of one. Waves also bound how many groups'
    # trajectories are in-flight in driver memory (all 48 groups at 16 candidates
    # would be ~700MB of latents).
    #
    # NOT done here: asking the server for a whole group in one batched
    # generation (num_outputs_per_prompt = n). It is the bigger win and the
    # plumbing exists, but Qwen-Image's denoising path does not repeat the text
    # conditioning to the expanded batch -- see build_rollout_request.
    for wave_start in range(0, len(groups), len(engines)):
        wave = groups[wave_start : wave_start + len(engines)]
        requests = []
        for group in wave:
            group_index = group[0].group_index
            group_requests = []
            for slot, sample in enumerate(group):
                sample_id = _native_generation_sample_id(sample, group_index, slot, sampling)
                request_sampling = {**request_sampling_base, "_sample_id": sample_id}
                # Each candidate in the group gets a DISTINCT seed so the G samples
                # per prompt are diverse (different initial + SDE noise). Identical
                # seeds → identical generations → zero intra-group reward variance →
                # GRPO advantages collapse and the optimizer step is skipped every
                # step (the reference recipe disables same-noise initialization too).
                # Under driver_xt (the default) the driver supplies x_T itself, so
                # the engine seed is only a fallback and stays constant.
                engine_seed = base_seed if driver_xt else base_seed + int(group_index or 0) * len(group) + slot
                group_requests.append(adapter.build_rollout_request(sample, request_sampling, engine_seed))
            requests.append(group_requests)
        wave_responses = _dispatch_groups(engines, requests)

        for group, responses in zip(wave, wave_responses):
            group_index = group[0].group_index
            if len(responses) != len(group):
                raise RuntimeError(
                    f"group {group_index}: engine returned {len(responses)} candidates for a group of {len(group)}."
                )

            # Fail fast on a malformed / short engine response (missing trajectory,
            # timesteps or sde_indices) with a clear per-field error, instead of a deep
            # opaque tensor error later in pack_trajectory.
            for resp in responses:
                adapter.validate_rollout_response(resp)

            common, conditions, tracks = adapter_pack_group(adapter, responses)
            trajectory_ref = _put_group_batch(rollout_id, group_index, common, conditions, tracks)
            trajectory_uri = f"ray://{trajectory_ref.hex()}"

            for slot, (sample, resp) in enumerate(zip(group, responses)):
                # Keep the decoded image in process for local rewards. Only an
                # external scorer needs a durable URI it can open.
                if getattr(args, "reward_runtime", "cpu") == "remote":
                    # External scorers receive JSON and cannot resolve this
                    # process's memory:// image cache.
                    img_uri, img_h, img_w, img_sha256 = save_candidate_image(
                        resp.get("generated_output"),
                        args.artifact_root,
                        args.generation_task,
                        rollout_id,
                        group_index,
                        slot,
                    )
                else:
                    img_uri, img_h, img_w, img_sha256 = cache_candidate_image(
                        resp.get("generated_output"),
                        args.generation_task,
                        rollout_id,
                        group_index,
                        slot,
                    )
                if img_uri:
                    resp["output"] = {
                        "uri": img_uri,
                        "mime": "image/png",
                        "sha256": img_sha256,
                        "height": img_h,
                        "width": img_w,
                    }
                manifest = build_candidate_manifest(
                    task=args.generation_task,
                    # `or slot` would rewrite a legitimate index 0 into the slot.
                    sample_index=sample.index if sample.index is not None else slot,
                    group_index=group_index,
                    policy_version=int(resp.get("policy_version", 0)),
                    artifact_tracks=adapter.artifact_tracks(resp),
                    trajectory_uri=trajectory_uri,
                    sampling_fingerprint=fp,
                    weight_manifest_sha256=str(resp.get("weight_manifest_sha256", "")),
                )
                sample.train_metadata = {
                    "group_index": group_index,
                    "trajectory_slot": slot,
                    "policy_version": int(resp.get("policy_version", 0)),
                    # Carry the rollout_id so the reward post-process can log its metrics
                    # against the same compute_rollout_step x-axis as train/* and perf/*
                    # (policy_version diverges from rollout_id after resume / early steps).
                    "rollout_id": int(rollout_id),
                    "trajectory_ref": trajectory_ref,
                    "manifest": manifest,
                }
                sample.status = Sample.Status.COMPLETED
            out_groups.append(group)
    gen_time = time.time() - gen_t0

    # Transfer the generated groups to the data system: convert each group to the
    # numeric TransferQueue train rows (reward + advantages via
    # ``convert_samples_to_train_data``) and ``async_put`` them into the
    # ``train_{rollout_id}`` partition the actor consumes. Mirrors the text
    # ``sglang_rollout.generate_rollout``, which transfers batches itself — without
    # this the actor blocks forever polling an empty partition.
    reward_time = 0.0
    if data_system_client is not None and out_groups:
        from relax.utils.async_utils import run
        from relax.utils.utils import transfer_batch_to_data_system

        reward_t0 = time.time()
        run(
            transfer_batch_to_data_system(
                args,
                out_groups,
                len(out_groups),
                rollout_id,
                data_system_client,
                is_last=True,
            )
        )
        reward_time = time.time() - reward_t0

    _log_rollout_perf(args, rollout_id, gen_time, reward_time, sum(len(g) for g in out_groups))
    return RolloutFnTrainOutput(samples=out_groups)


def evaluate_rollout(args, rollout_id: int):
    """Held-out evaluation for the generative path (``evaluation=True``).

    Generates ``n_samples_per_eval_prompt`` candidates for every prompt in each
    configured ``--eval-prompt-data`` dataset, scores them with the same reward
    scorer used in training, and returns the framework's eval contract
    ``{dataset_name: {"rewards": [...]}}`` — which the RolloutManager turns into
    ``eval/<dataset>`` dashboard series and an eval-summary jsonl.

    Unlike the training rollout this asks the engine for the final image only
    (no DiT trajectory, no denoising env) and writes no group sidecar: nothing
    is replayed, so the expensive trajectory capture would be pure waste.
    """
    import time

    from relax.engine.rewards.generative import score_samples
    from relax.engine.rollout.base_types import RolloutFnEvalOutput

    eval_t0 = time.time()
    eval_metrics: Dict[str, float] = {}
    datasets = getattr(args, "eval_datasets", None) or []
    if not datasets:
        logger.warning("native evaluate_rollout: no eval_datasets configured; nothing to evaluate.")
        return RolloutFnEvalOutput(data={})

    adapter = _load_adapter(args)
    engines = _get_engines(args, None)
    sampling = dict(getattr(args, "sampling_config", None) or {})
    base_seed = int(getattr(args, "generation_seed", 1234))
    request_sampling_base = {**sampling, "_rollout_id": int(rollout_id), "_base_seed": base_seed}
    driver_xt = bool(sampling.get("driver_xt", True))
    artifact_root = os.path.join(args.artifact_root, "eval")

    results: Dict[str, Dict[str, Any]] = {}
    for cfg in datasets:
        prompts = _read_eval_prompts(args, cfg)
        if not prompts:
            logger.warning(f"native evaluate_rollout: dataset {cfg.name!r} at {cfg.path!r} yielded no prompts.")
            continue
        n = int(getattr(cfg, "n_samples_per_eval_prompt", None) or getattr(args, "n_samples_per_eval_prompt", 1) or 1)

        scored: List[Sample] = []
        # n single-candidate requests per prompt, with every candidate of a wave
        # of len(engines) prompts spread across ALL engines by _dispatch_groups.
        # The previous shape dispatched one prompt's n requests at a time — with
        # n=2 over 8 engines that left 6 engines idle and serialized 256 prompts
        # into 256 round-trips, which is where the ~700 s eval pass went.
        # The candidate is the unit of parallelism rather than the prompt because
        # `build_rollout_request` pins `num_outputs_per_prompt: 1`: Qwen-Image's
        # denoising path does not repeat the text conditioning to a batch-expanded
        # generation (see that docstring), so a server-side group batch dies in
        # the AdaLN modulation.
        eval_groups = [
            [_clone_eval_sample(proto, group_index, slot, n) for slot in range(n)]
            for group_index, proto in enumerate(prompts)
        ]
        for wave_start in range(0, len(eval_groups), len(engines)):
            wave = eval_groups[wave_start : wave_start + len(engines)]
            requests = []
            for group in wave:
                group_index = group[0].group_index
                group_reqs = []
                for slot, sample in enumerate(group):
                    # Deterministic per-(dataset, prompt, candidate) seed so eval is
                    # reproducible across steps and comparable between runs.
                    request_sampling = {
                        **request_sampling_base,
                        "_sample_id": _native_generation_sample_id(sample, group_index, slot, sampling),
                    }
                    seed = base_seed if driver_xt else base_seed + (int(group_index) * n) + slot
                    req = adapter.build_rollout_request(sample, request_sampling, seed)
                    req["rollout_return_dit_trajectory"] = False
                    req["rollout_return_denoising_env"] = False
                    group_reqs.append(req)
                requests.append(group_reqs)
            for group, responses in zip(wave, _dispatch_groups(engines, requests)):
                group_index = group[0].group_index
                for slot, (sample, resp) in enumerate(zip(group, responses)):
                    uri, height, width, image_sha256 = save_candidate_image(
                        resp.get("generated_output"),
                        artifact_root,
                        args.generation_task,
                        rollout_id,
                        group_index,
                        slot,
                    )
                    if uri:
                        resp["output"] = {
                            "uri": uri,
                            "mime": "image/png",
                            "sha256": image_sha256,
                            "height": height,
                            "width": width,
                        }
                    sample.train_metadata = {
                        "group_index": group_index,
                        "trajectory_slot": slot,
                        "manifest": {"outputs": [t.to_dict() for t in adapter.artifact_tracks(resp)]},
                    }
                    scored.append(sample)

        component_rewards = score_samples(args, scored)
        if not component_rewards:
            continue
        weights = getattr(args, "reward_component_weights", None) or {}
        primary = max(weights, key=weights.get) if weights else next(iter(component_rewards))
        entry: Dict[str, Any] = {"rewards": [float(v) for v in component_rewards[primary]]}
        # Surface every component so a multi-component reward is fully visible.
        for comp, vals in component_rewards.items():
            if comp != primary:
                entry[f"reward_{comp}"] = [float(v) for v in vals]
        results[cfg.name] = entry
        logger.info(
            f"[eval {rollout_id}] {cfg.name}: n_prompts={len(prompts)} n_samples={n} "
            f"{primary}_mean={sum(entry['rewards']) / max(1, len(entry['rewards'])):.4f}"
        )
        # `_log_eval_rollout_data` only reads `rewards`, so anything else in the
        # entry (extra reward components) would be silently discarded. Emit the
        # eval shape + per-component means ourselves, on the same x-axis.
        eval_metrics[f"eval/{cfg.name}_num_prompts"] = float(len(prompts))
        eval_metrics[f"eval/{cfg.name}_num_candidates"] = float(len(entry["rewards"]))
        for comp, vals in component_rewards.items():
            floats = [float(v) for v in vals]
            if floats:
                eval_metrics[f"eval/{cfg.name}/{comp}_mean"] = sum(floats) / len(floats)

    _log_eval_perf(args, rollout_id, time.time() - eval_t0, eval_metrics)
    return RolloutFnEvalOutput(data=results)


def _log_eval_perf(args, rollout_id: int, eval_time: float, eval_metrics: Dict[str, float]) -> None:
    """Emit eval duration + shape/component metrics.

    Eval generation runs a full denoising pass per candidate and can take
    minutes; without this it is invisible on the dashboard.
    """
    from relax.utils.metrics.metric_utils import compute_rollout_step

    metrics = dict(eval_metrics)
    metrics["perf/eval_time"] = float(eval_time)
    step = int(compute_rollout_step(args, rollout_id))
    metrics["rollout/step"] = step
    logger.info(
        f"[eval {rollout_id}] eval_time={eval_time:.2f}s " + " ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
    )
    _emit_tracking_metrics(args, metrics, step, "native evaluate_rollout")


def _clone_eval_sample(proto: Sample, group_index: int, slot: int, n_per_prompt: int) -> Sample:
    """Clone an eval prompt prototype into one candidate.

    ``index`` encodes (prompt, candidate) as ``group_index * n + slot``, which is
    collision-proof for any ``n``; the previous ``* 1000 + slot`` silently
    aliased two prompts' candidates once ``n_samples_per_eval_prompt`` reached
    1000.
    """
    import copy

    sample = copy.deepcopy(proto)
    sample.index = group_index * max(1, int(n_per_prompt)) + slot
    sample.group_index = group_index
    return sample


def _sample_metadata(sample: Sample) -> Mapping[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
    nested = metadata.get("metadata")
    if isinstance(nested, Mapping):
        return {**nested, **{k: v for k, v in metadata.items() if k != "metadata"}}
    return metadata


def _sample_id_mode(sampling: Optional[Mapping[str, Any]] = None) -> str:
    if sampling and sampling.get("sample_id_mode") is not None:
        return str(sampling["sample_id_mode"]).strip().lower()
    return os.environ.get("RELAX_NATIVE_GENERATION_SAMPLE_ID_MODE", "metadata").strip().lower()


def _native_generation_sample_id(
    sample: Sample,
    group_index: Optional[int],
    slot: int,
    sampling: Optional[Mapping[str, Any]] = None,
) -> str:
    if _sample_id_mode(sampling) in {"positional", "group_index"}:
        return f"prompt:{int(group_index or 0)}:sample:{int(slot)}"
    metadata = _sample_metadata(sample)
    prompt_id = metadata.get("prompt_id")
    if prompt_id is None or not str(prompt_id).strip():
        prompt_id = metadata.get("sample_id")
    if prompt_id is None or not str(prompt_id).strip():
        prompt_id = str(int(group_index or 0))
    return f"prompt:{prompt_id}:sample:{int(slot)}"


def _read_eval_prompts(args, cfg) -> List[Sample]:
    """Read an eval prompt jsonl into ``Sample`` prototypes (one per prompt).

    Diffusion prompts are plain strings, so this deliberately bypasses the text
    ``Dataset`` (tokenizer / chat-template machinery) and reads the same jsonl
    schema that ``examples/diffusion/prepare_data.py`` emits.
    """
    path = getattr(cfg, "path", None)
    if not path or not os.path.exists(path):
        logger.warning(f"native evaluate_rollout: eval dataset path {path!r} does not exist.")
        return []
    prompt_key = getattr(cfg, "input_key", None) or getattr(args, "input_key", None) or "prompt"
    samples: List[Sample] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get(prompt_key)
            if not prompt:
                continue
            sample = Sample(prompt=prompt)
            record_metadata = record.get("metadata")
            if isinstance(record_metadata, Mapping):
                metadata = dict(record_metadata)
                for key in ("prompt_id", "task", "sample_id", "negative_prompt"):
                    if key in record and key not in metadata:
                        metadata[key] = record[key]
            else:
                metadata = {k: v for k, v in record.items() if k not in (prompt_key, "multimodal_inputs")}
            sample.metadata = metadata
            mm = record.get("multimodal_inputs") or {}
            if mm:
                sample.multimodal_inputs = mm
            samples.append(sample)
    return samples


def _log_rollout_perf(args, rollout_id: int, gen_time: float, reward_time: float, num_samples: int) -> None:
    """Emit the diffusion rollout-stage perf metrics (design doc 11.2).

    ``perf/rollout_time`` (generation dispatch + image save) and
    ``perf/reward_time`` (convert + reward scoring + TQ transfer), logged via
    the framework tracking sink from the RolloutManager worker.
    """
    from relax.utils.metrics.metric_utils import compute_rollout_step

    metrics = {
        "perf/rollout_time": float(gen_time),
        "perf/reward_time": float(reward_time),
        "rollout/step": compute_rollout_step(args, rollout_id),
    }
    if gen_time > 0 and num_samples:
        metrics["perf/rollout_samples_per_s"] = float(num_samples) / gen_time
    # Print to the run log too — every other emission site does, and without it
    # "which stage is slow" cannot be triaged from the log alone. Note
    # reward_time covers convert + scoring + TQ put, not scoring alone.
    logger.info(
        f"[rollout {rollout_id}] samples={num_samples} gen_time={gen_time:.2f}s "
        f"reward+convert+tq_time={reward_time:.2f}s "
        f"samples_per_s={metrics.get('perf/rollout_samples_per_s', 0.0):.3f}"
    )
    _emit_tracking_metrics(args, metrics, int(metrics["rollout/step"]), "native generate_rollout")


def _emit_tracking_metrics(args, metrics: Dict[str, float], step: int, context: str) -> None:
    """Push a metric dict to the tracking backends; never fail the caller.

    Only the SINK is guarded, and deliberately so: the backends are wandb /
    TensorBoard / ClearML / an HTTP metrics service, i.e. optional, remote and
    allowed to be absent or flaky, and losing a dashboard point must not abort
    a rollout. Everything that computes the metrics runs OUTSIDE this try, so a
    bug there still raises instead of hiding behind one warning line — that
    swallow-everything shape was how a real defect stayed invisible before.
    ``exc_info`` keeps the traceback when a backend does fail.
    """
    from relax.utils import tracking_utils

    try:
        tracking_utils.log(args, metrics, step_key="rollout/step")
        # Buffered adapter → commit the step to the backends (ClearML/TB/wandb).
        tracking_utils.flush_metrics(args, step)
    except Exception as e:
        logger.warning(f"{context}: metric sink failed ({type(e).__name__}: {e}); metrics dropped.", exc_info=True)


def adapter_pack_group(adapter, responses: List[Mapping[str, Any]]):
    """Stack per-candidate packed trajectories into group
    ``common/cond/tracks`` dicts.

    Conditions are DEDUPLICATED: every candidate of a group denoises the same
    prompt, so all G candidates carry a bit-identical text embedding. Stacking
    them wrote G copies of the same ~3.6 MB tensor into every sidecar (~58 MB
    per 16-candidate group, all but 3.6 MB of it waste, paid again on every
    rank's hydrate). When the per-candidate condition tensors compare equal we
    keep a single copy with a leading dim of 1; :func:`hydrate_micro_batches`
    expands it back to the candidate count as a view (zero copy). Candidates
    whose conditions genuinely differ still stack, so this stays correct if a
    future task varies conditioning within a group.
    """
    packed = [adapter.pack_trajectory(r) for r in responses]
    if not packed:
        return {}, {}, {}
    common: Dict[str, torch.Tensor] = {}
    conditions: Dict[str, torch.Tensor] = {}
    tracks: Dict[str, torch.Tensor] = {}
    keys = packed[0].keys()
    for key in keys:
        if key.startswith("cond_"):
            first = packed[0][key]
            # torch.equal over G-1 small tensors is microseconds; the stack it
            # avoids is tens of MB of allocation + disk.
            if all(torch.equal(first, p[key]) for p in packed[1:]):
                conditions[key] = first.unsqueeze(0)
            else:
                conditions[key] = torch.stack([p[key] for p in packed], dim=0)
            continue
        values = [p[key] for p in packed]
        if key in ("sigmas", "sde_indices"):
            first = values[0]
            for candidate_index, value in enumerate(values[1:], start=1):
                if not torch.equal(first, value):
                    raise ValueError(
                        f"Candidate {candidate_index} has a different {key} schedule from candidate 0; "
                        "refusing to replay a trajectory under another candidate's timesteps."
                    )
            common[key] = first
            continue
        stacked = torch.stack(values, dim=0)
        if key in ("sample_indices", "policy_versions", "seed_hashes"):
            common[key] = stacked
        else:
            tracks[key] = stacked
    return common, conditions, tracks


# ---------------------------------------------------------------------------
# TransferQueue converter (design doc 5.4)
# ---------------------------------------------------------------------------


def convert_samples_to_train_data(args, samples):
    """Emit the numeric-only diffusion TQ row (``--custom-convert-…`` hook).

    Deliberately narrow: only the columns something actually reads.
    ``group_indices`` + ``trajectory_slots`` are the JOIN KEY that binds an
    advantage to its candidate in the group sidecar (see
    :func:`hydrate_micro_batches`); ``raw_reward`` is the un-normalized scorer
    output, kept for logging because ``advantages`` has already been
    group-centered by the reward post-process and can no longer be read as a
    reward. ``total_lengths`` and ``sample_indices`` are required by the
    TransferQueue metadata contract.
    """
    from relax.utils.utils import dict_to_tensordict, post_process_rewards

    flat: List[Sample] = _flatten(samples)
    processed_rewards = post_process_rewards(args, flat)
    if isinstance(processed_rewards, tuple):
        raw_rewards, rewards = processed_rewards
    else:
        raw_rewards = [sample.get_reward_value(args) for sample in flat]
        rewards = processed_rewards

    group_indices, trajectory_slots = [], []
    for s in flat:
        tm = s.train_metadata or {}
        # `or 0` would rewrite a legitimate group_index 0 — use an explicit None test.
        group_index = tm.get("group_index")
        if group_index is None:
            group_index = s.group_index if s.group_index is not None else 0
        group_indices.append(int(group_index))
        trajectory_slots.append(int(tm.get("trajectory_slot", 0)))

    trajectory_ref_payloads = []
    for s in flat:
        ref = (s.train_metadata or {}).get("trajectory_ref")
        trajectory_ref_payloads.append(bytes(ref) if ref is not None else b"")
    # TransferQueue needs a rectangular numeric column. Prefix the meaningful
    # byte count so future Ray versions may change the serialized-ref size
    # without making padding ambiguous.
    trajectory_ref_width = max((len(ref) for ref in trajectory_ref_payloads), default=0)
    trajectory_refs = [
        list(len(ref).to_bytes(4, "little") + ref + bytes(trajectory_ref_width - len(ref)))
        for ref in trajectory_ref_payloads
    ]

    train_data = {
        "sample_indices": [int(s.index) for s in flat],
        "group_indices": group_indices,
        "trajectory_slots": trajectory_slots,
        # A compact serialized Ray ObjectRef, repeated per row. The referenced
        # group tensor is stored once in Plasma and shared by all trainer ranks
        # without serializing it through the TransferQueue.
        "trajectory_refs": trajectory_refs,
        "advantages": [float(r) for r in rewards],
        "raw_reward": [float(r) for r in raw_rewards],
        "total_lengths": [1 for _ in flat],
        # Degenerate-round flag (post_process sets it when the group advantage
        # variance collapses below the floor): the actor skips optimizer.step so a
        # zero-gradient round is an explicit skip, not a silent no-op.
        "skip_optimizer_step": [int((s.train_metadata or {}).get("skip_optimizer_step", 0)) for s in flat],
    }
    if args.debug_train_only:
        return train_data
    return dict_to_tensordict(train_data, len(flat))


# ---------------------------------------------------------------------------
# Actor-side hydration
# ---------------------------------------------------------------------------


def hydrate_micro_batches(args, adapter, rollout_id, rollout_data_ref, *, dp_rank: int = 0, dp_world: int = 1):
    """Rehydrate numeric TQ rows into FlowGRPO micro-batches (actor side).

    ``rollout_data_ref`` is the TensorDict of numeric rows written by
    :func:`convert_samples_to_train_data`. Rows are grouped by
    ``(rollout_id, group_index)``; each group's sidecar is loaded once and split
    into the per-candidate tensors the FlowGRPO update consumes.

    **DP work sharding (design doc 3.1).** Every rank builds the SAME ordered
    group list (one micro-batch = one optimizer step per group), so the step
    count — and therefore the FSDP all-gather / reduce-scatter collective sequence
    — is identical on every rank (mismatched step counts would deadlock). Within
    each group, when ``group_size`` divides ``dp_world`` the ``group_size /
    dp_world`` candidates ``[dp_rank::dp_world]`` are the only ones this rank
    replays: each rank's ``mean`` loss over an equal-size disjoint candidate
    subset, averaged across ranks by FSDP's gradient reduce, equals the global
    group-mean gradient exactly (mean-of-means identity — no ``loss_scale``
    needed). When ``group_size`` does NOT divide ``dp_world`` the group falls back
    to full replay on every rank (correct, just redundant); this also removes the
    empty-subset case, so every rank always runs the model forward for every step.

    **Replay sub-batching (``--micro-batch-size``).** The replay activations scale
    with the number of candidates forwarded at once, so a rank's slice is further
    split into chunks of ``micro_batch_size`` candidates, each emitted as its own
    micro-batch. The actor already scales every micro-batch's loss by ``1/N`` and
    accumulates, so equal-size chunks leave the gradient identical (mean-of-means)
    while making the memory peak independent of ``n_samples_per_prompt`` — the
    knob that otherwise caps ``n`` at ``dp_world`` on a 96 GB card. Only splits
    when the chunk size divides the slice evenly (unequal chunks would silently
    reweight the mean); ``0``/unset means no split.

    **Advantage ↔ candidate join.** A row's advantage is bound to its candidate
    by the row's ``trajectory_slot``, which is the dim-0 position that candidate
    occupies in the group sidecar. This used to rely on TQ rows coming back in
    insertion order: the rows for a group were collected in arrival order and
    indexed positionally. Any sampler / partition path that reordered rows would
    then have attached every advantage to the wrong image — silently, with a
    perfectly plausible loss curve. The join below is by slot, and asserts that
    each of a group's ``0..group_size-1`` slots appears exactly once, so a
    reordered, duplicated or truncated read fails loudly instead.
    """
    rows = _resolve_rows(rollout_data_ref)
    if rows is None:
        return []
    if "trajectory_slots" not in rows:
        raise ValueError(
            "hydrate_micro_batches: rows have no 'trajectory_slots' column, which is the only safe key for "
            "binding an advantage to its candidate in the group sidecar. Ensure the actor requests it in "
            "_read_train_partition's data_fields and that convert_samples_to_train_data wrote it."
        )

    # group_index -> {trajectory_slot: row position}. Keyed, not positional.
    by_group: Dict[int, Dict[int, int]] = {}
    group_ids = rows["group_indices"].tolist()
    slots = rows["trajectory_slots"].tolist()
    for i, (g, slot) in enumerate(zip(group_ids, slots)):
        group_slots = by_group.setdefault(int(g), {})
        if int(slot) in group_slots:
            raise ValueError(
                f"hydrate_micro_batches: group {int(g)} has two rows for trajectory slot {int(slot)} "
                f"(rows {group_slots[int(slot)]} and {i}); the advantage↔candidate join is ambiguous."
            )
        group_slots[int(slot)] = i

    _log_raw_reward(rows, dp_rank)

    micro_batches: List[Dict[str, Any]] = []
    # Deterministic group order so every rank iterates identically (lockstep).
    for group_index in sorted(by_group):
        group_slots = by_group[group_index]
        group_size = len(group_slots)
        missing = sorted(set(range(group_size)) - set(group_slots))
        if missing:
            raise ValueError(
                f"hydrate_micro_batches: group {group_index} has {group_size} rows but is missing trajectory "
                f"slots {missing}; the sidecar's candidates cannot be matched to their advantages."
            )
        # Row positions in SLOT order, so index j of `advantages` is sidecar
        # candidate j regardless of the order the TQ handed the rows back.
        positions = [group_slots[slot] for slot in range(group_size)]
        batch = _load_group_batch_for_rows(args, adapter, rollout_id, group_index, rows, positions)
        # Even split → shard candidates across ranks; else full replay on all ranks.
        if dp_world > 1 and group_size % dp_world == 0:
            local_idx = list(range(group_size))[dp_rank::dp_world]
        else:
            local_idx = list(range(group_size))
        batch = _shard_group_batch(batch, group_size, local_idx)
        step_indices = batch["sde_indices"].tolist()
        advantages = rows["advantages"][positions].float()[local_idx]
        # Degenerate round → skip the optimizer step (flag is uniform across the
        # round; read any of this group's rows). Absent column → never skip.
        skip = False
        if "skip_optimizer_step" in rows:
            skip = bool(int(rows["skip_optimizer_step"][positions].max()))
        # old_logp is recomputed by the actor (replay anchor); pass a batch view.
        local_size = len(local_idx)
        for chunk in _candidate_chunks(local_size, int(getattr(args, "micro_batch_size", 0) or 0)):
            chunk_batch = _shard_group_batch(batch, local_size, chunk) if len(chunk) < local_size else batch
            micro_batches.append(
                {
                    "batch": _expand_shared_conditions(chunk_batch, len(chunk)),
                    "advantages": advantages[chunk],
                    "step_indices": [int(s) for s in step_indices],
                    "local_idx": [local_idx[c] for c in chunk],
                    "skip_optimizer_step": skip,
                }
            )
    return micro_batches


def _log_raw_reward(rows, dp_rank: int) -> None:
    """Log the round's un-normalized reward once per step (rank 0 only).

    ``advantages`` has already been group-centered by the reward post-process,
    so its mean is ~0 by construction and says nothing about whether the policy
    is improving. ``raw_reward`` is the scorer's actual output and is the only
    column on the actor side that answers that — it exists purely for this
    line, so if this ever goes away, drop the column too.
    """
    if dp_rank != 0 or "raw_reward" not in rows:
        return
    raw = rows["raw_reward"].float()
    logger.info(
        f"native hydrate: n={raw.numel()} raw_reward mean={float(raw.mean()):.4f} "
        f"min={float(raw.min()):.4f} max={float(raw.max()):.4f}"
    )


def _expand_shared_conditions(batch: Dict[str, torch.Tensor], num_candidates: int) -> Dict[str, torch.Tensor]:
    """Broadcast deduplicated group conditions back to the candidate count.

    :func:`adapter_pack_group` stores a group's identical condition tensors once
    with a leading dim of 1. ``expand`` is a view, so the replay sees exactly the
    ``[B, ...]`` shapes it saw before dedup at zero memory cost. Conditions that
    genuinely differ per candidate already have a leading dim of ``B`` and are
    left alone.
    """
    if num_candidates <= 1:
        return batch
    out: Dict[str, torch.Tensor] = {}
    for key, val in batch.items():
        if key.startswith("cond_") and isinstance(val, torch.Tensor) and val.shape[:1] == (1,):
            out[key] = val.expand(num_candidates, *val.shape[1:])
        else:
            out[key] = val
    return out


def _candidate_chunks(local_size: int, micro_batch_size: int) -> List[List[int]]:
    """Split ``range(local_size)`` into equal chunks of ``micro_batch_size``.

    Returns a single full-slice chunk when the split is disabled (``<= 0``), a
    no-op (``>= local_size``), or would leave a short tail — unequal chunks
    would silently reweight the accumulated ``1/N`` mean.
    """
    if micro_batch_size <= 0 or micro_batch_size >= local_size or local_size % micro_batch_size:
        return [list(range(local_size))]
    return [list(range(s, s + micro_batch_size)) for s in range(0, local_size, micro_batch_size)]


# Keys that are shared across the group (not per-candidate) and must NOT be sliced
# along the candidate dimension when DP-sharding a group batch.
_GROUP_SHARED_KEYS = frozenset({"sigmas", "sde_indices"})


def _shard_group_batch(
    batch: Dict[str, torch.Tensor], group_size: int, local_idx: List[int]
) -> Dict[str, torch.Tensor]:
    """Select this rank's candidate subset from a group batch (design doc 3.1).

    Per-candidate tensors are stacked with the candidate as dim 0 (see
    :func:`adapter_pack_group`), so slice any tensor whose leading dim equals
    ``group_size`` to ``local_idx`` — except the group-shared schedule tensors
    (``sigmas`` / ``sde_indices``). Deduplicated conditions have a leading dim of
    1 and are therefore skipped too; :func:`_expand_shared_conditions` broadcasts
    them after the split. A no-op when ``local_idx`` covers the whole group (the
    non-sharded fallback).
    """
    if len(local_idx) == group_size:
        return batch
    index = torch.as_tensor(local_idx, dtype=torch.long)
    sharded: Dict[str, torch.Tensor] = {}
    for key, val in batch.items():
        if key not in _GROUP_SHARED_KEYS and isinstance(val, torch.Tensor) and val.shape[:1] == (group_size,):
            sharded[key] = val.index_select(0, index)
        else:
            sharded[key] = val
    return sharded


def _load_group_batch(sidecar_path: str, adapter) -> Dict[str, torch.Tensor]:
    """Load one group's trajectory sidecar into a flat tensor batch.

    ``--artifact-root`` must be on storage that every trainer rank can read:
    the rollout driver writes each sidecar from ONE process and all FSDP ranks
    read it back here. On a node-local path that works on the driver's node and
    fails on every other one, so the bare ``FileNotFoundError`` from
    safetensors would surface as a confusing missing-file traceback on some
    ranks only. Say what the actual requirement is instead.
    """
    from relax.backends.fsdp.runtime import hydrate_trajectory_sidecar

    if not os.path.exists(sidecar_path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(sidecar_path)))
        raise RuntimeError(
            f"native hydrate: trajectory sidecar {sidecar_path!r} is not readable from this rank. "
            f"--artifact-root ({root!r}) must point at a filesystem shared by the rollout driver and every "
            "trainer rank (the driver writes the sidecars, all ranks read them back); a node-local path "
            "silently works only on the driver's node. If the path IS shared, the rollout for this step "
            "never wrote the group — check for a failed rollout or an over-aggressive "
            "--artifact-retention-rollouts."
        )
    tensors, _meta = hydrate_trajectory_sidecar(sidecar_path)
    batch: Dict[str, torch.Tensor] = {}
    for key, val in tensors.items():
        _ns, _, name = key.partition(".")
        batch[name] = val
    return batch


def _load_group_batch_for_rows(args, adapter, rollout_id: int, group_index: int, rows, positions: List[int]):
    """Load a group from Ray shared memory, with sidecars as a legacy
    fallback."""
    if "trajectory_refs" in rows:
        packed_refs = [
            bytes(int(v) for v in rows["trajectory_refs"][position].detach().cpu().reshape(-1).tolist())
            for position in positions
        ]
        payloads = []
        for packed in packed_refs:
            if len(packed) < 4:
                raise ValueError(f"group {group_index}: trajectory reference is shorter than its length prefix.")
            size = int.from_bytes(packed[:4], "little")
            if size > len(packed) - 4:
                raise ValueError(f"group {group_index}: trajectory reference length {size} exceeds its payload.")
            payloads.append(packed[4 : 4 + size])
        if len(set(payloads)) != 1:
            raise ValueError(f"group {group_index}: candidates reference different trajectory objects.")
        payload = payloads[0]
        if payload:
            import ray

            return ray.get(ray.cloudpickle.loads(payload))
    sidecar = group_sidecar_path(args.artifact_root, args.generation_task, rollout_id, group_index)
    return _load_group_batch(sidecar, adapter)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _flatten(samples) -> List[Sample]:
    if samples and isinstance(samples[0], list):
        return [s for group in samples for s in group]
    return list(samples)


def _get_samples(data_source, num_samples):
    """Fetch prompt groups from the data source (Ray actor handle or local
    object)."""
    get_samples = data_source.get_samples
    if hasattr(get_samples, "remote"):  # Ray actor handle (RolloutManager path)
        import ray

        return ray.get(get_samples.remote(num_samples))
    return get_samples(num_samples)


def _load_adapter(args):
    from relax.utils.utils import load_function

    return load_function(args.model_adapter_path)()


def _get_engines(args, data_system_client):
    """Resolve rollout engine handles for direct generate_batch dispatch.

    The driver dispatches to the engines' ``generate_batch`` (a Ray method that
    POSTs to the diffusion HTTP server). ``generate_rollout`` runs inside the
    RolloutManager actor's worker thread (``generate`` →
    ``asyncio.to_thread``), so the manager handle is
    ``ray.get_runtime_context().current_actor``; its
    ``get_rollout_engines_and_lock`` returns the node-0 engine handles. Falls
    back to a ``data_system_client`` getter (unit tests), then empty (CPU
    tests).

    Only the "we are not running inside a Ray actor" probe is caught. A failure
    of the manager CALL is a real distributed fault and propagates: swallowing it
    used to downgrade e.g. a dead RolloutManager to a debug line and then report
    the generic "no generation engines available", which points the investigation
    at configuration instead of at the actor that actually died.
    """
    manager = None
    try:
        import ray

        manager = ray.get_runtime_context().current_actor
    except (ImportError, RuntimeError, AssertionError, AttributeError) as e:
        # No ray installed, or no current actor (driver process / unit tests).
        logger.debug(f"native generate_rollout: not inside a Ray actor ({e}); trying the data client.")
    if manager is not None:
        engines, *_rest = ray.get(manager.get_rollout_engines_and_lock.remote())
        if engines:
            return list(engines)

    if data_system_client is not None:
        getter = getattr(data_system_client, "get_generation_engines", None)
        if callable(getter):
            return list(getter())
    return []


def _dispatch_groups(engines, group_requests: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """Spread every candidate of every group across all engines; regroup on
    return.

    ``group_requests[g]`` is one group's per-candidate requests. They are
    FLATTENED before round-robin so the unit of load balancing is a candidate,
    not a group: dispatching group-by-group left engines idle whenever a group
    had fewer candidates than there are engines (the eval path, with 2 candidates
    over 8 engines, ran at 1/4 occupancy and that is where its ~700s went).

    Results come back keyed by flat index and are regrouped **in input order** —
    the caller zips this against its group list, so returning them in shard order
    (concatenating each engine's results) would pair a group's samples with
    another group's trajectories.
    """
    if not engines:
        raise RuntimeError("native generate_rollout: no generation engines available.")
    import ray

    flat: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for reqs in group_requests:
        spans.append((len(flat), len(flat) + len(reqs)))
        flat.extend(reqs)

    shards: List[List[int]] = [[] for _ in engines]
    for i in range(len(flat)):
        shards[i % len(engines)].append(i)
    refs, order = [], []
    for engine, idxs in zip(engines, shards):
        if not idxs:
            continue
        refs.append(engine.generate_batch.remote([flat[i] for i in idxs]))
        order.append(idxs)

    out: List[Optional[Dict[str, Any]]] = [None] * len(flat)
    for idxs, shard_result in zip(order, ray.get(refs)):
        if len(shard_result) != len(idxs):
            raise RuntimeError(f"engine returned {len(shard_result)} responses for {len(idxs)} requests.")
        for i, resp in zip(idxs, shard_result):
            out[i] = resp
    missing = [i for i, r in enumerate(out) if r is None]
    if missing:
        raise RuntimeError(f"no response for request indices {missing}.")
    return [[out[i] for i in range(lo, hi)] for lo, hi in spans]  # type: ignore[misc]


def _resolve_rows(rollout_data_ref):
    if rollout_data_ref is None:
        return None
    import ray

    if isinstance(rollout_data_ref, ray.ObjectRef):
        return ray.get(rollout_data_ref)
    return rollout_data_ref
