# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Generative reward manager: deferred group barrier + component normalization.

:func:`post_process` is the ``--custom-reward-post-process-path`` hook for native
generative RL. It fires once per rollout after a full group of candidates has
been generated (design doc 11.2): it reads each candidate's artifact manifest,
batch-scores the media through the configured scorer(s), centers every reward
component within its prompt group, applies the configured advantage std divisor,
weight-combines the components into one advantage, and returns
``(raw_rewards, advantages)`` in the ``(list, list)`` shape the Relax reward
pipeline expects.

A missing required component or a non-finite reward fails the whole group
(design doc 6.3), and if the final advantage's global variance collapses below
``1e-6`` the round is flagged so the actor can skip the optimizer step.

The :class:`GenerativeRewardScorer` protocol (design doc 7.3) is the contract
every scorer file in this package implements; the actual model loading lives in
the per-scorer modules (currently only pickscore).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple, runtime_checkable

import torch

from relax.models import flow_grpo
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)

__all__ = [
    "GenerativeRewardScorer",
    "BaseGenerativeScorer",
    "RewardRequest",
    "post_process",
    "combine_group_advantages",
    "compute_reward_metrics",
    "load_track_uris",
    "score_samples",
]

_ADV_VARIANCE_FLOOR = 1e-6


class RewardRequest(dict):
    """A scorer input: prompt + artifact descriptors for one candidate.

    A thin ``dict`` subclass so scorers can carry arbitrary scorer-specific
    fields alongside the manifest's ``outputs`` while staying trivially
    serializable (the ``remote`` runtime POSTs these as JSON).
    """


@runtime_checkable
class GenerativeRewardScorer(Protocol):
    """Media-tensor batch scorer (design doc 7.3)."""

    required_tracks: Tuple[str, ...]

    def onload(self) -> None: ...

    def score_batch(self, requests: List[RewardRequest]) -> Dict[str, List[float]]:
        """Return ``{component_name: [score per request]}``."""
        ...

    def offload(self) -> None: ...


def load_track_uris(request: Mapping[str, Any], track: str) -> List[str]:
    """Return the artifact URIs of a given track in a reward request.

    Reads the candidate's ``outputs`` manifest entries (as attached by the
    native generation driver) and filters by ``ArtifactTrack.track``. The only
    task on this branch is t2i, so the only track emitted today is ``image``.
    """
    uris: List[str] = []
    for out in request.get("outputs", []) or []:
        if out.get("track") == track and out.get("uri"):
            uris.append(out["uri"])
    return uris


class BaseGenerativeScorer:
    """Shared onload/offload device lifecycle for GPU/CPU media scorers.

    Subclasses set ``required_tracks`` and implement ``_load_model()`` (returns
    a ``torch.nn.Module`` or ``None`` for CPU-only scorers) and
    ``score_batch()``. ``reward_runtime=cpu`` forces ``allow_cuda=False`` so
    the scorer never initializes CUDA and never competes with the colocated
    diffusion pipeline for GPU memory (design doc 7.3 / 11.2). ``cpu`` is also
    the default when the attribute is missing: silently grabbing the rollout
    GPU is the worse failure.
    """

    required_tracks: Tuple[str, ...] = ()

    def __init__(self, args, *, allow_cuda: bool = True) -> None:
        self.args = args
        runtime = getattr(args, "reward_runtime", "cpu") if args is not None else "cpu"
        self._allow_cuda = allow_cuda and runtime != "cpu"
        self._model = None

    @property
    def device(self) -> torch.device:
        if self._allow_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):  # pragma: no cover - model-specific, overridden
        return None

    def onload(self) -> None:
        if self._model is None:
            self._model = self._load_model()
        if self._model is not None:
            self._model.to(self.device)

    def offload(self) -> None:
        if self._model is not None:
            self._model.to("cpu")
        if self._allow_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_batch(self, requests: List[RewardRequest]) -> Dict[str, List[float]]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Advantage combination (group barrier)
# ---------------------------------------------------------------------------


def combine_group_advantages(
    component_rewards: Mapping[str, Sequence[float]],
    group_indices: List[List[int]],
    component_weights: Mapping[str, float],
    *,
    required_components: Sequence[str],
    std_normalization: bool = True,
    advantage_std_mode: str | None = None,
) -> Tuple[List[float], bool]:
    """Normalize + weight-combine reward components into per-sample advantages.

    ``advantage_std_mode`` is explicit when set:

    - ``group`` divides each group-centered component by that group's own std.
    - ``batch`` divides every group-centered component by one batch-wide std.
    - ``none`` does not divide at all.

    When it is unset, ``std_normalization`` keeps the legacy Relax meaning of
    ``--disable-grpo-std-normalization``: True -> ``group``, False -> ``none``.

    Returns ``(advantages, degenerate)`` where ``degenerate`` is True when the
    combined advantage variance is below ``1e-6`` (the caller skips the step).
    Fails fast on a missing required component or any non-finite reward value.
    """
    for comp in required_components:
        if comp not in component_rewards:
            raise ValueError(f"generative reward: required component {comp!r} missing from scorer output.")
    tensors: Dict[str, torch.Tensor] = {}
    for name, values in component_rewards.items():
        t = torch.tensor(values, dtype=torch.float32)
        if not torch.isfinite(t).all():
            raise ValueError(f"generative reward: component {name!r} has non-finite values.")
        tensors[name] = t

    std_mode = advantage_std_mode if advantage_std_mode is not None else ("group" if std_normalization else "none")
    advantages = flow_grpo.combine_component_advantages(tensors, component_weights, group_indices, std_mode=std_mode)
    degenerate = bool(advantages.var(unbiased=False).item() < _ADV_VARIANCE_FLOOR)
    if degenerate:
        logger.warning("generative reward: advantage variance below floor; optimizer step should be skipped.")
    return advantages.tolist(), degenerate


# ---------------------------------------------------------------------------
# post-process hook
# ---------------------------------------------------------------------------


def compute_reward_metrics(
    component_rewards: Mapping[str, Sequence[float]],
    advantages: Sequence[float],
    group_indices: Sequence[Sequence[int]],
    degenerate: bool,
    primary: str | None = None,
) -> Dict[str, float]:
    """Pure reward/advantage statistics for observability (design doc 11.2).

    Emits per-component ``reward/<c>_{mean,std,min,max}``, the combined
    ``reward/advantage_{mean,std}``, ``reward/degenerate`` (0/1),
    ``reward/num_groups`` and ``reward/group_std_mean`` — the mean per-group
    std of the primary component, i.e. the exact quantity whose collapse below
    ``_ADV_VARIANCE_FLOOR`` makes a round degenerate (``loss=0`` no-op step).
    """
    metrics: Dict[str, float] = {}
    for comp, vals in component_rewards.items():
        t = torch.tensor([float(v) for v in vals], dtype=torch.float32)
        if t.numel() == 0:
            continue
        metrics[f"reward/{comp}_mean"] = float(t.mean())
        metrics[f"reward/{comp}_std"] = float(t.std(unbiased=False))
        metrics[f"reward/{comp}_min"] = float(t.min())
        metrics[f"reward/{comp}_max"] = float(t.max())
    adv = torch.tensor([float(a) for a in advantages], dtype=torch.float32)
    if adv.numel():
        metrics["reward/advantage_mean"] = float(adv.mean())
        metrics["reward/advantage_std"] = float(adv.std(unbiased=False))
    metrics["reward/degenerate"] = 1.0 if degenerate else 0.0
    metrics["reward/num_groups"] = float(len(group_indices))
    primary = primary or (next(iter(component_rewards), None))
    if primary is not None and primary in component_rewards:
        pv = [float(v) for v in component_rewards[primary]]
        group_stds = [
            float(torch.tensor([pv[i] for i in idxs], dtype=torch.float32).std(unbiased=False))
            for idxs in group_indices
            if len(idxs) > 1
        ]
        if group_stds:
            metrics["reward/group_std_mean"] = sum(group_stds) / len(group_stds)
    return metrics


def score_samples(args, samples) -> Dict[str, List[float]]:
    """Batch-score candidates and return raw per-component rewards.

    The training hook (:func:`post_process`) additionally group-normalizes and
    weight-combines into advantages; held-out evaluation wants only the raw
    component scores, so this exposes the scoring half on its own. Reuses the
    same cached reward manager, so eval does not reload the scorer.
    """
    flat: List[Sample] = _flatten(samples)
    if not flat:
        return {}
    manager = _get_manager(args)
    requests = [_build_reward_request(args, s) for s in flat]
    try:
        component_rewards = manager.score(requests)
    finally:
        _release_memory_images(requests)
    _validate_reward_lengths(component_rewards, len(flat))
    return component_rewards


def post_process(args, samples) -> Tuple[List[float], List[float]]:
    """``custom_reward_post_process_path`` entry point (design doc 11.2)."""
    import time

    flat: List[Sample] = _flatten(samples)
    manager = _get_manager(args)

    requests = [_build_reward_request(args, s) for s in flat]
    # Time the scorer on its own: the rollout driver's `perf/reward_time` lumps
    # convert + scoring + the TransferQueue put together (~104 s/rollout at 256
    # samples), so it cannot say whether a CPU-pinned scorer is the bottleneck.
    score_t0 = time.time()
    try:
        component_rewards = manager.score(requests)
    finally:
        _release_memory_images(requests)
    score_time = time.time() - score_t0
    _validate_reward_lengths(component_rewards, len(flat))

    group_indices = _group_index_lists(flat)
    component_weights = getattr(args, "reward_component_weights", None) or {}
    configured_required = getattr(args, "reward_required_components", None)
    required = list(component_weights.keys()) if configured_required is None else list(configured_required)
    std_mode = _advantage_std_mode(args)

    advantages, degenerate = combine_group_advantages(
        component_rewards,
        group_indices,
        component_weights,
        required_components=required,
        advantage_std_mode=std_mode,
        # Backward-compatible fallback when the explicit generative mode is not
        # present on older args objects: the shared flag is on by default and
        # `--disable-grpo-std-normalization` turns the std divisor OFF.
        std_normalization=bool(getattr(args, "grpo_std_normalization", True)),
    )

    # Raw reward = the primary (highest-weight) component, for logging.
    primary = max(component_weights, key=component_weights.get) if component_weights else next(iter(component_rewards))
    raw_rewards = [float(v) for v in component_rewards[primary]]
    if len(advantages) != len(flat):
        raise ValueError(f"generative reward length mismatch: samples={len(flat)} advantages={len(advantages)}")

    # Keep the RAW reward on the Sample. Overwriting it with the advantage made
    # every downstream `get_reward_value` consumer (rollout dumps, filters, the
    # sample-level metrics) report a group-centered number whose mean is ~0 by
    # construction as if it were a reward. The advantage is returned to the
    # caller and rides the TQ's `advantages` column; it does not belong on the
    # Sample's reward field. Stash it in train_metadata for anyone who wants
    # both.
    for s, raw, adv in zip(flat, raw_rewards, advantages):
        s.reward = float(raw)
        if s.train_metadata is None:
            s.train_metadata = {}
        s.train_metadata["advantage"] = float(adv)
    if degenerate:
        for s in flat:
            s.train_metadata["skip_optimizer_step"] = True

    # Observability: reward/advantage distribution (design doc 11.2). This is the
    # signal that diagnoses a degenerate (loss=0) round — reward/group_std_mean is
    # the per-group std whose collapse trips the degenerate floor.
    metrics = compute_reward_metrics(component_rewards, advantages, group_indices, degenerate, primary)
    metrics.update(_advantage_std_mode_metrics(std_mode))
    metrics["reward/score_time"] = float(score_time)
    metrics["reward/score_samples_per_s"] = float(len(flat)) / score_time if score_time > 0 else 0.0
    _log_reward_metrics(args, flat, metrics)
    return raw_rewards, advantages


def _log_reward_metrics(args, flat: List[Sample], metrics: Dict[str, float]) -> None:
    """Emit reward metrics via the framework tracking sink.

    ``post_process`` has no ``rollout_id`` argument, but the native generation
    driver stamps it into each sample's ``train_metadata`` — use it with
    ``compute_rollout_step`` so the reward metrics share the SAME x-axis as the
    actor's ``train/*`` and ``perf/*`` metrics. (Using ``policy_version``
    instead diverges after a resume and collides at step 0 before the first
    commit.) Falls back to ``policy_version`` only if ``rollout_id`` is absent
    (older rows).

    Only the tracking SINK is guarded (see below): it talks to optional, remote
    backends and losing a dashboard point must not fail a rollout. Everything
    that computes the metrics runs outside the guard, so a bug there raises
    instead of hiding behind one warning line.
    """
    from relax.utils import tracking_utils
    from relax.utils.metrics.metric_utils import compute_rollout_step

    rollout_id = None
    for s in flat:
        rid = (s.train_metadata or {}).get("rollout_id")
        if rid is not None:
            rollout_id = int(rid)
            break
    if rollout_id is not None:
        step = int(compute_rollout_step(args, rollout_id))
    else:
        # Fallback for rows written before the driver stamped rollout_id.
        # policy_version diverges from rollout_id after a resume, so this puts
        # reward/* on a DIFFERENT x-axis than train/* — warn rather than
        # silently switching axes.
        step = max((int((s.train_metadata or {}).get("policy_version", 0)) for s in flat), default=0)
        logger.warning(
            "generative reward: no rollout_id on the samples; falling back to policy_version "
            f"({step}) as the metric step. reward/* may not align with train/* on the dashboard."
        )
    metrics = {**metrics, "rollout/step": step}
    # Print to the run log too (the tracking sink goes to wandb/TB/metrics-service,
    # not stdout) — this is the reward-distribution line that diagnoses loss=0.
    logger.info(
        f"[reward step {step}] "
        + " ".join(f"{k.split('/', 1)[-1]}={v:.4f}" for k, v in metrics.items() if k != "rollout/step")
    )
    try:
        tracking_utils.log(args, metrics, step_key="rollout/step")
        # Buffered adapter → commit the step to the backends (ClearML/TB/wandb).
        tracking_utils.flush_metrics(args, int(step))
    except Exception as e:
        logger.warning(
            f"generative reward: metric sink failed ({type(e).__name__}: {e}); metrics dropped.", exc_info=True
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _advantage_std_mode(args) -> str:
    mode = getattr(args, "generative_advantage_std_mode", None)
    if mode is not None:
        return str(mode)
    return "group" if bool(getattr(args, "grpo_std_normalization", True)) else "none"


def _advantage_std_mode_metrics(mode: str) -> Dict[str, float]:
    return {
        f"reward/advantage_std_mode_{candidate}": 1.0 if mode == candidate else 0.0
        for candidate in flow_grpo.ADVANTAGE_STD_MODES
    }


def _validate_reward_lengths(component_rewards: Mapping[str, Sequence[float]], expected: int) -> None:
    for name, values in component_rewards.items():
        if len(values) != expected:
            raise ValueError(
                f"generative reward length mismatch: samples={expected} component={name!r} rewards={len(values)}"
            )


def _flatten(samples) -> List[Sample]:
    if samples and isinstance(samples[0], list):
        return [s for group in samples for s in group]
    return list(samples)


def _group_index_lists(flat: List[Sample]) -> List[List[int]]:
    groups: Dict[int, List[int]] = {}
    for pos, s in enumerate(flat):
        tm = s.train_metadata or {}
        gi = int(tm.get("group_index", s.group_index if s.group_index is not None else 0))
        groups.setdefault(gi, []).append(pos)
    return list(groups.values())


def _build_reward_request(args, sample: Sample) -> RewardRequest:
    tm = sample.train_metadata or {}
    manifest = tm.get("manifest", {})
    request = RewardRequest(
        prompt=sample.prompt,
        outputs=manifest.get("outputs", []),
        multimodal_inputs=sample.multimodal_inputs,
        metadata=sample.metadata,
    )
    memory_images = {}
    for output in request["outputs"]:
        uri = output.get("uri", "")
        if output.get("track") == "image" and uri.startswith("memory://"):
            from relax.engine.rollout.native_generation import get_cached_candidate_image

            image = get_cached_candidate_image(uri)
            if image is None:
                raise RuntimeError(f"generative reward: in-memory image {uri!r} was released before scoring.")
            memory_images[uri] = image
    if memory_images:
        request["memory_images"] = memory_images
    return request


def _release_memory_images(requests: List[RewardRequest]) -> None:
    from relax.engine.rollout.native_generation import get_cached_candidate_image

    for request in requests:
        for uri in request.get("memory_images", {}):
            get_cached_candidate_image(uri, remove=True)


# One manager per distinct scorer configuration. Keyed on the config itself, not
# on ``id(args)``: CPython reuses the id of a collected object, so an args
# namespace that went away could hand its scorer to an unrelated config.
_MANAGER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _manager_cache_key(args) -> Tuple[Any, ...]:
    return tuple(
        str(getattr(args, name, None))
        for name in ("reward_runtime", "reward_scorer_path", "reward_model_path", "reward_endpoint")
    )


def _get_manager(args):
    """Return the reward manager for this run's ``reward_runtime``.

    ``post_process`` runs inside the rollout worker, so there are exactly two
    things that can happen: score in this process (``cpu`` with CUDA disabled,
    or ``colocate`` sharing the rollout GPU), or POST to an external service
    (``remote``). The manager is cached per scorer configuration so the model
    is loaded once, not once per rollout.
    """
    from relax.distributed.ray.generative_reward import GenerativeRewardManager

    key = _manager_cache_key(args)
    if key not in _MANAGER_CACHE:
        runtime = getattr(args, "reward_runtime", "cpu")
        if runtime == "remote":
            _MANAGER_CACHE[key] = GenerativeRewardManager(args)
        else:
            _MANAGER_CACHE[key] = GenerativeRewardManager.local(args)
    return _MANAGER_CACHE[key]
