# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Generic contracts for native generative RL (diffusion / flow models).

Model families plug into a single FSDP runtime, rollout driver, reward barrier
and weight-transport path via the :class:`GenerativeModelAdapter` protocol.
Everything model-specific — latent geometry, trajectory packing, replay,
artifact tracks and the Megatron/HF-style weight name map — lives behind this
protocol so the runtime depends only on it (design doc section 4.1). This
branch ships the Qwen-Image adapter (t2i); the Wan 2.2 and LTX-2.3 adapters
live on ``backup/diffusion-generative-rl-full``, and any other family can be
supplied via ``--model-adapter-path``.

This module also defines the serialization-side contracts shared by the rollout
driver and the FSDP weight sync:

* :class:`ArtifactTrack` — one generated media stream (image / video / audio).
* :class:`FullWeightManifest` — the deterministic full-transformer weight
  descriptor validated by every rollout engine during a weight update
  (design doc 10.2).

The protocol is intentionally import-light: it references only ``torch`` and
stdlib types, and heavy per-family model code is imported lazily inside each
concrete adapter.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

import torch


__all__ = [
    "ArtifactTrack",
    "FullWeightManifest",
    "GenerativeModelAdapter",
    "ordered_name_shape_hash",
    "artifact_manifest_dict",
    "resolve_sde_indices",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1


def _require_sha256(value: str, field_name: str) -> str:
    digest = str(value)
    try:
        valid = len(digest) == 64 and int(digest, 16) >= 0
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{field_name} must be a non-empty 64-character hexadecimal SHA-256 digest.")
    return digest.lower()


def resolve_sde_indices(sampling: Mapping[str, Any], rollout_id: Optional[int] = None) -> List[int]:
    """Resolve the trained SDE step indices from a sampling config.

    FlowGRPO trains a subset of the denoising trajectory's steps (the stochastic
    "SDE" steps). These indices do double duty: they select which steps get SDE
    noise during GENERATION and which steps are replayed for the gradient.

    Resolution order:

    * ``sde_resample_per_rollout`` + a ``rollout_id`` ⇒ a fresh draw of
      ``num_sde_steps`` from the candidate pool, seeded by ``rollout_id``
      (reproducible and identical on every engine and rank without any
      communication). Pinned indices would confine exploration to one subspace
      and never give the other steps a gradient, however long the run
      (reference ``AllSDEScheduler.get_sde_indices``). The pool is ``sde_pool``
      when given, else the ``sde_timestep_fraction`` window. ``rollout_id=None``
      always takes the deterministic path — held-out eval must keep one fixed
      set of steps or its number is not comparable across steps.
    * explicit ``sde_indices`` list ⇒ used verbatim.
    * else ``num_sde_steps`` stride-spaced within ``[frac_lo * N, frac_hi * N)``
      where ``N = num_inference_steps`` and ``frac = sde_timestep_fraction``
      (default the whole schedule). Stride-based
      (``floor(k * len(window) / num_sde)``), not endpoint-inclusive: a 12-step
      schedule with ``num_sde_steps=3`` over the first half yields ``[0, 2, 4]``.
    * else ``[]`` — with no SDE config there is nothing to train (the pipeline
      then fails when packing the trajectory).

    Under ``sde_type="sde"`` step 0 must be excluded (``sigma == 1`` makes the
    diffusion coefficient singular; see
    :func:`~relax.models.flow_grpo.flow_sde_transition_std_dev_t`) — use
    ``sde_pool`` to drop it from the draw while keeping the rest.
    """
    explicit = sampling.get("sde_indices")
    if rollout_id is not None and sampling.get("sde_resample_per_rollout"):
        pool = _sde_pool(sampling, explicit)
        num_sde = int(sampling.get("num_sde_steps", 0) or 0) or (len(explicit) if explicit else 0)
        if pool and num_sde > 0:
            import numpy as np

            num_sde = min(num_sde, len(pool))
            chosen = np.random.default_rng(int(rollout_id)).choice(len(pool), size=num_sde, replace=False)
            return sorted(int(pool[c]) for c in chosen)
    if explicit:
        return sorted({int(i) for i in explicit})

    window = _sde_window(sampling)
    num_sde = int(sampling.get("num_sde_steps", 0) or 0)
    if not window or num_sde <= 0:
        return []
    num_sde = min(num_sde, len(window))
    return sorted({window[(k * len(window)) // num_sde] for k in range(num_sde)})


def _sde_window(sampling: Mapping[str, Any]) -> List[int]:
    """The ``sde_timestep_fraction`` slice of the denoising schedule."""
    n = int(sampling.get("num_inference_steps", 0) or 0)
    if n <= 0:
        return []
    frac = sampling.get("sde_timestep_fraction") or [0.0, 1.0]
    lo = max(0, int(round(float(frac[0]) * n)))
    hi = min(n, int(round(float(frac[1]) * n)))
    if hi <= lo:
        hi = n
    return list(range(lo, hi))


def _sde_pool(sampling: Mapping[str, Any], explicit) -> List[int]:
    """Candidate steps for a per-rollout draw: ``sde_pool``, else the
    window."""
    pool = sampling.get("sde_pool")
    if pool:
        return sorted({int(i) for i in pool})
    window = _sde_window(sampling)
    if window:
        return window
    return sorted({int(i) for i in explicit}) if explicit else []


# ---------------------------------------------------------------------------
# Artifact / trajectory contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactTrack:
    """One generated media stream produced by a candidate.

    ``track`` is one of ``image`` / ``video`` / ``audio``. ``meta`` carries
    modality-specific descriptors (fps, frames, height, width, sample_rate,
    samples, codec, …) that end up verbatim in the artifact manifest. Large
    media itself is written to ``uri`` and never enters the TransferQueue.
    """

    track: str
    uri: str
    mime: str
    sha256: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    #: Manifest keys owned by the track itself; ``meta`` may not shadow them.
    _RESERVED_KEYS = ("track", "uri", "mime", "sha256")

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to one manifest entry (``meta`` merged at the top level).

        Readers index the entry directly (``out["track"]`` / ``out["uri"]`` in
        the reward barrier), so the merge is part of the wire format. A
        ``meta`` key that shadows a reserved one would silently rewrite the
        artifact's identity — e.g. a scorer would open the wrong ``uri`` — so
        it is a hard error instead.
        """
        d: Dict[str, Any] = {
            "track": self.track,
            "uri": self.uri,
            "mime": self.mime,
            "sha256": _require_sha256(self.sha256, "ArtifactTrack.sha256"),
        }
        clash = sorted(k for k in self.meta if k in self._RESERVED_KEYS)
        if clash:
            raise ValueError(f"ArtifactTrack.meta may not override reserved manifest keys {clash}.")
        d.update(self.meta)
        return d


def artifact_manifest_dict(
    *,
    task: str,
    sample_index: int,
    group_index: int,
    policy_version: int,
    outputs: List[ArtifactTrack],
    trajectory_uri: str,
    sampling_fingerprint: str,
    weight_manifest_sha256: str,
    conditions: List[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build the lightweight per-candidate artifact manifest (design doc 5.2).

    Only URIs, digests and numeric descriptors go here — never pixel/latent
    tensors — so the manifest can be embedded in the rollout JSONL row.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "sample_index": int(sample_index),
        "group_index": int(group_index),
        "policy_version": int(policy_version),
        "conditions": list(conditions or []),
        "outputs": [t.to_dict() for t in outputs],
        "trajectory_uri": trajectory_uri,
        "sampling_fingerprint": sampling_fingerprint,
        "weight_manifest_sha256": _require_sha256(weight_manifest_sha256, "weight_manifest_sha256"),
    }


# ---------------------------------------------------------------------------
# Full weight manifest (design doc 10.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FullWeightManifest:
    """Deterministic descriptor of a full-transformer weight snapshot.

    Emitted by the FSDP weight-update iterator and validated by every SGLang
    engine before a version is committed. ``ordered_name_shape_hash`` pins the
    exact parameter order + shapes so a mismatched engine fails fast rather
    than silently loading a truncated state.
    """

    schema_version: int
    model_family: str
    task: str
    policy_version: int
    base_model_sha256: str
    tensor_count: int
    total_bytes: int
    wire_dtype: str
    bucket_size_bytes: int
    ordered_name_shape_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        """Stable content digest over the manifest fields (order-
        independent)."""
        data = self.to_dict()
        payload = "|".join(f"{k}={data[k]}" for k in sorted(data))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ordered_name_shape_hash(named_shapes: List[Tuple[str, Tuple[int, ...]]]) -> str:
    """Deterministic hash of an ordered ``[(name, shape), ...]`` sequence.

    The order is significant: it is the exact sequence the DTensor full-gather
    iterator streams parameters in, so both sender and receiver reconstruct the
    identical hash iff they agree on the full parameter set and its order.
    """
    hasher = hashlib.sha256()
    for name, shape in named_shapes:
        hasher.update(name.encode("utf-8"))
        hasher.update(b":")
        hasher.update(",".join(str(int(s)) for s in shape).encode("utf-8"))
        hasher.update(b";")
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GenerativeModelAdapter(Protocol):
    """Per-family boundary for condition, trajectory, artifact and weight
    logic.

    A run fixes exactly one adapter (``model_adapter_path`` in the task YAML).
    The FSDP runtime, rollout driver, reward barrier and weight transport
    depend only on this protocol — never on a concrete model class. See design
    doc 4.1.

    **Optional attributes** (read with ``getattr``, deliberately not declared
    as members so adding them never invalidates an existing adapter — this
    Protocol is ``runtime_checkable``, which the adapter unit tests assert
    against):

    ``lora_target_modules: Sequence[str]``
        Default LoRA targets for ``--fsdp-trainable-mode lora``, as module-name
        suffixes of the trainable transformer (e.g. ``"attn.to_q"``,
        ``"attn.to_out.0"``). Overridden by ``--lora-target-modules``; if
        neither is set, a LoRA run fails at init.
    ``lora_task_type: str``
        PEFT task type, default ``"FEATURE_EXTRACTION"`` (diffusion DiTs).
    """

    family: str
    supported_tasks: Tuple[str, ...]

    def load_train_model(self, config: Any) -> torch.nn.Module:
        """Instantiate the trainable transformer, already correctly frozen.

        The returned module is handed straight to FSDP, so whatever must not
        receive a gradient has to have ``requires_grad=False`` set here — there
        is no second freezing hook. (VAE / text encoders are not part of this
        module at all: they live frozen inside the rollout engine.)
        """
        ...

    def build_rollout_request(self, sample: Any, sampling: Mapping[str, Any], seed: Any) -> Dict[str, Any]:
        """Translate a Relax ``Sample`` into a DiffGenerator request dict.

        ``seed`` is an int for a single candidate, or a sequence of ints to ask
        the engine for one batched generation of that many candidates (one
        response per candidate, one seed each).
        """
        ...

    def validate_rollout_response(self, response: Mapping[str, Any]) -> None:
        """Fail fast if a generation response is missing required
        tracks/fields."""
        ...

    def pack_trajectory(self, response: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Pack a response into the sidecar tensor dict (``x_t`` / ``x_next`` /
        schedule / frozen conditions)."""
        ...

    def replay_transition(self, model: torch.nn.Module, batch: Mapping[str, Any], step_index: int) -> torch.Tensor:
        """Recompute the velocity prediction for one stored transition."""
        ...

    def artifact_tracks(self, response: Mapping[str, Any]) -> List[ArtifactTrack]:
        """List the media artifacts a response produced."""
        ...

    def weight_name_map(self, name: str) -> str:
        """Map an FSDP parameter name to the engine-side (HF) weight name."""
        ...
