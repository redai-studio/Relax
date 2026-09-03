# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""LoRA for the FSDP2 generative backend: injection, merge-fold and key maps.

Relax's other LoRA implementation (:mod:`relax.utils.megatron_peft_utils`) is
Megatron-Bridge shaped -- its adapters are ``...adapter.linear_in/linear_out``
and its target-module vocabulary is LLM specific (``linear_qkv`` ...), so none
of the injection or sync machinery transfers to a diffusers DiT. This module is
the FSDP/diffusers counterpart. Only the truly backend-neutral helpers are
reused from there (:func:`~relax.utils.megatron_peft_utils.write_hf_peft_adapter`
and friends, at export time).

Three responsibilities:

* **Injection** -- :func:`inject_lora_adapter` installs a single PEFT adapter in
  place via ``peft.inject_adapter_in_model``. Deliberately *not*
  ``get_peft_model``: that returns a ``PeftModel`` wrapper, which would hide
  ``_no_split_modules`` from :func:`~relax.backends.fsdp.runtime.resolve_block_classes`
  (wrong shard granularity), change the forward signature the model adapters
  call with keywords, and prefix every state-dict key with ``base_model.model.``.
* **Merged sync** -- :class:`LoraSyncPlan` presents the LoRA'd model as if it
  were the original full-FT model: the same parameter names, in the same order,
  with ``B @ A`` folded into each wrapped base weight on the fly. That keeps the
  weight-sync manifest byte-identical to full FT, so the engine side needs no
  changes at all.
* **Key maps** -- :func:`engine_adapter_state_dict` (adapter-mode sync, SGLang
  naming) and :func:`peft_adapter_state_dict` (on-disk HF-PEFT export).

``peft`` is imported lazily inside the functions that need it so this module
stays importable -- and the pure-tensor helpers stay testable -- without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from relax.backends.fsdp.weight_update import strip_transport_wrappers
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

__all__ = [
    "LORA_ADAPTER_NAME",
    "LoraSyncPlan",
    "build_lora_sync_plan",
    "engine_adapter_state_dict",
    "fold_lora_delta",
    "inject_lora_adapter",
    "is_lora_injected",
    "lora_metadata_dict",
    "lora_scaling",
    "named_lora_params",
    "peft_adapter_state_dict",
    "strip_base_layer",
]

# Single adapter per run. PEFT's default name; also what SGLang calls the
# "nickname" of the loaded adapter.
LORA_ADAPTER_NAME = "default"

# PEFT renames a wrapped ``X.weight`` to ``X.base_layer.weight``.
_BASE_LAYER_SEGMENT = ".base_layer."

_LORA_A_SEGMENT = ".lora_A."
_LORA_B_SEGMENT = ".lora_B."


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------


def inject_lora_adapter(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
    adapter_name: str = LORA_ADAPTER_NAME,
) -> int:
    """Install one LoRA adapter into ``model`` in place; return the layer
    count.

    Must run *before* ``fsdp2_wrap``: once ``fully_shard`` has run every
    parameter is a DTensor and PEFT's ``nn.Linear`` -> ``lora.Linear`` surgery is
    not supported on those. It must also run before the optimizer is built,
    which snapshots the trainable parameter list.

    PEFT matches ``target_modules`` by name suffix and is happy to match a
    *subset* -- a typo in one entry silently halves the adapter capacity with no
    error and no log line. So every target is required to match at least one
    module; anything unmatched raises.

    Args:
        model: The trainable transformer (not the pipeline bundle).
        rank: LoRA rank ``r``.
        alpha: LoRA alpha; the applied scale is ``alpha / rank``.
        target_modules: Module-name suffixes, e.g. ``("attn.to_q", "attn.to_out.0")``.
        dropout: LoRA dropout. Must be 0 for the FlowGRPO replay (a stochastic
            forward would desynchronize the pi_old anchor from the update).
        bias: PEFT ``bias`` mode.
        task_type: PEFT task type; ``FEATURE_EXTRACTION`` for diffusion DiTs.
        adapter_name: PEFT adapter name.

    Returns:
        The number of injected ``LoraLayer`` modules.
    """
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer

    targets = [str(t) for t in target_modules]
    if not targets:
        raise ValueError("inject_lora_adapter requires a non-empty target_modules list.")
    if int(rank) <= 0:
        raise ValueError(f"inject_lora_adapter requires rank > 0, got {rank}.")

    peft_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        bias=str(bias),
        task_type=str(task_type),
    )
    try:
        inject_adapter_in_model(peft_config, model, adapter_name=adapter_name)
    except ValueError as exc:
        raise ValueError(
            f"LoRA target module(s) matched nothing injectable. Requested suffixes: {targets}. PEFT reported: {exc}"
        ) from exc

    # Per-target match counts: PEFT reports nothing when only some targets hit.
    injected = [name for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    per_target = {t: sum(1 for name in injected if name.endswith(t)) for t in targets}
    unmatched = sorted(t for t, count in per_target.items() if count == 0)
    if unmatched:
        raise ValueError(
            f"LoRA target module(s) matched nothing: {unmatched}. "
            f"Matched targets: { {t: c for t, c in per_target.items() if c} }. "
            "Target names are module-name suffixes of the trainable transformer "
            "(e.g. 'attn.to_q', 'attn.to_out.0')."
        )
    logger.info(
        f"Injected LoRA adapter {adapter_name!r}: r={rank}, alpha={alpha}, dropout={dropout}, "
        f"{len(injected)} layers, per-target counts={per_target}"
    )
    return len(injected)


def is_lora_injected(model: torch.nn.Module) -> bool:
    """True if ``model`` carries PEFT LoRA parameters."""
    return any(_LORA_A_SEGMENT in name or _LORA_B_SEGMENT in name for name, _ in model.named_parameters())


# ---------------------------------------------------------------------------
# name helpers
# ---------------------------------------------------------------------------


def strip_base_layer(name: str) -> str:
    """``blk.0.attn.to_q.base_layer.weight`` -> ``blk.0.attn.to_q.weight``.

    Anchored on the dotted segment so a module legitimately named e.g.
    ``base_layer_norm`` is not mangled -- the same defensive reasoning as
    :func:`relax.utils.megatron_peft_utils.is_lora_adapter_param`.
    """
    return name.replace(_BASE_LAYER_SEGMENT, ".") if _BASE_LAYER_SEGMENT in name else name


def _split_lora_param(name: str, adapter_name: str) -> Optional[Tuple[str, str]]:
    """Return ``(module_stem, "lora_A"|"lora_B")`` for an adapter parameter.

    PEFT names them ``<stem>.lora_A.<adapter>.weight``. Returns ``None`` for a
    base parameter or for another adapter's tensors.
    """
    for segment, kind in ((_LORA_A_SEGMENT, "lora_A"), (_LORA_B_SEGMENT, "lora_B")):
        if segment not in name:
            continue
        stem, _, tail = name.partition(segment)
        # tail is "<adapter>.weight"
        adapter, _, leaf = tail.partition(".")
        if adapter != adapter_name or leaf != "weight":
            return None
        return stem, kind
    return None


def named_lora_params(
    model: torch.nn.Module,
    *,
    adapter_name: str = LORA_ADAPTER_NAME,
) -> List[Tuple[str, torch.Tensor]]:
    """``[(name, param)]`` for ``adapter_name``'s tensors, name-sorted."""
    out = [(name, p) for name, p in model.named_parameters() if _split_lora_param(name, adapter_name) is not None]
    return sorted(out, key=lambda kv: kv[0])


def lora_scaling(model: torch.nn.Module, *, adapter_name: str = LORA_ADAPTER_NAME) -> float:
    """The applied LoRA scale ``alpha / r``, read off the injected config.

    Read from ``model.peft_config`` rather than from ``args`` so a resumed run
    cannot silently disagree with the adapter that is actually installed.
    """
    peft_config = getattr(model, "peft_config", None)
    if not peft_config or adapter_name not in peft_config:
        raise ValueError(
            f"Model carries no PEFT config for adapter {adapter_name!r}; lora_scaling requires an injected adapter."
        )
    config = peft_config[adapter_name]
    rank = int(config.r)
    if rank <= 0:
        raise ValueError(f"PEFT config for {adapter_name!r} has non-positive rank {rank}.")
    return float(config.lora_alpha) / float(rank)


# ---------------------------------------------------------------------------
# merged weight sync
# ---------------------------------------------------------------------------


def fold_lora_delta(
    base: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """``base + (B @ A) * scaling``, accumulated in fp32, returned at
    ``base.dtype``.

    Two things go wrong if the arithmetic is left at bf16, and neither raises:

    * ``B @ A`` contracts over the rank dimension (64 terms at r=64). A bf16
      matmul accumulates that sum at bf16, which measurably degrades the delta.
    * Rounding the delta to bf16 *before* adding it rounds twice. Adding in fp32
      and rounding once is strictly closer to the exact result.

    So callers must pass all three tensors at master width; pre-casting any of
    them to the wire dtype defeats the function. The single unavoidable rounding
    happens on the way out, to ``base.dtype``.

    Note what this does *not* buy: the result is stored at ``base.dtype``, so a
    delta below that dtype's resolution at the base's magnitude is still lost.
    That is fine here because a trained adapter's accumulated delta is far above
    it -- keeping the *per-step* ~1e-4 updates is the optimizer master's job, see
    :func:`relax.backends.fsdp.runtime.upcast_trainable_params`.
    """
    merged = base.float() + (lora_b.float() @ lora_a.float()) * float(scaling)
    return merged.to(base.dtype)


@dataclass
class LoraSyncPlan:
    """A full-FT-shaped view of a LoRA'd model, for the weight-sync path.

    ``named_base_params`` carries the *original* (pre-injection) parameter
    names -- ``.base_layer`` already stripped -- so sorting, the manifest and
    the engine-side name routing all behave exactly as they do under full FT.
    ``folds`` maps those same names to the ``(A, B)`` pair that must be folded
    in when the tensor is materialized.

    Stripping here rather than inside the model adapter's ``weight_name_map``
    is deliberate: the iterator sorts on the *pre-map* name, so a late strip
    would reorder ``X.base_layer.weight`` relative to its neighbours and produce
    a self-consistent but silently different ordered-name-shape hash than full
    FT.
    """

    named_base_params: List[Tuple[str, torch.Tensor]]
    folds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    scaling: float = 1.0

    def materialize(
        self,
        name: str,
        param: torch.Tensor,
        gather_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """``tensor_source`` hook: full-gather ``param`` and fold its adapter
        in.

        Issues one ``full_tensor()`` collective for a plain parameter and three
        for a LoRA'd one. Collective safety comes from the caller walking
        ``named_base_params`` in a deterministic sorted order that is identical
        on every rank.
        """
        from relax.backends.fsdp.weight_update import materialize_full_tensor

        base = materialize_full_tensor(param, gather_device)
        pair = self.folds.get(name)
        if pair is None:
            return base
        lora_a = materialize_full_tensor(pair[0], gather_device)
        lora_b = materialize_full_tensor(pair[1], gather_device)
        return fold_lora_delta(base, lora_a, lora_b, self.scaling)


def build_lora_sync_plan(
    model: torch.nn.Module,
    *,
    adapter_name: str = LORA_ADAPTER_NAME,
    scaling: Optional[float] = None,
) -> LoraSyncPlan:
    """Build the merged-sync view of an injected ``model``.

    Walks ``named_parameters()`` once. Base parameters (trainable or frozen) are
    emitted under their pre-injection name; the adapter's own tensors are
    withheld and recorded as folds instead.

    Build this *after* ``fsdp2_wrap`` and cache it: it holds parameter objects,
    whose ``.data`` the optimizer and the offload/onload helpers mutate in
    place, so one build survives the whole run.
    """
    if scaling is None:
        scaling = lora_scaling(model, adapter_name=adapter_name)

    base_params: List[Tuple[str, torch.Tensor]] = []
    adapters: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, param in model.named_parameters():
        split = _split_lora_param(name, adapter_name)
        if split is not None:
            stem, kind = split
            adapters.setdefault(stem, {})[kind] = param
            continue
        if _LORA_A_SEGMENT in name or _LORA_B_SEGMENT in name:
            raise ValueError(
                f"Parameter {name!r} belongs to a LoRA adapter other than {adapter_name!r}; "
                "the FSDP generative backend supports exactly one adapter."
            )
        base_params.append((strip_base_layer(name), param))

    folds: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    base_names = {name for name, _ in base_params}
    for stem, pair in adapters.items():
        if "lora_A" not in pair or "lora_B" not in pair:
            raise ValueError(f"LoRA module {stem!r} is missing its {'lora_B' if 'lora_A' in pair else 'lora_A'} half.")
        target = f"{stem}.weight"
        if target not in base_names:
            raise ValueError(
                f"LoRA module {stem!r} has no matching base weight {target!r}; "
                "the base/adapter name mapping is out of sync."
            )
        folds[target] = (pair["lora_A"], pair["lora_B"])

    logger.info(f"Built LoRA sync plan: {len(base_params)} base tensors, {len(folds)} folded, scaling={scaling:.4f}")
    return LoraSyncPlan(named_base_params=base_params, folds=folds, scaling=float(scaling))


# ---------------------------------------------------------------------------
# adapter export / transport key maps
# ---------------------------------------------------------------------------


def engine_adapter_state_dict(
    named_params: Sequence[Tuple[str, torch.Tensor]],
    *,
    adapter_name: str = LORA_ADAPTER_NAME,
    alpha: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Rename adapter tensors for SGLang's diffusion LoRA loader.

    SGLang strips a ``diffusion_model.`` prefix and a trailing ``.weight``, then
    maps ``transformer_blocks.N.attn.*.lora_[AB].default`` -> the same without
    ``.default``. It does *not* strip a ``transformer.`` or ``base_model.model.``
    prefix, and an unmatched name is a warning rather than an error -- so a
    wrong prefix disables LoRA on 100% of layers while the load reports success.
    We therefore emit bare ``<module>.lora_A.weight`` names.

    ``alpha`` additionally emits a per-layer ``<module>.alpha`` scalar. SGLang
    derives the rank from ``lora_A.shape[0]`` but reads alpha *only* from such a
    tensor (it never opens ``adapter_config.json``), defaulting it to the rank
    -- i.e. silently applying scale 1.0 whenever ``alpha != rank``.
    """
    out: Dict[str, torch.Tensor] = {}
    stems: List[str] = []
    for name, param in named_params:
        split = _split_lora_param(name, adapter_name)
        if split is None:
            continue
        stem, kind = split
        stem = strip_transport_wrappers(stem)
        transport_key = f"{stem}.{kind}.weight"
        if transport_key in out:
            raise ValueError(f"Multiple LoRA parameters normalize to the same transport key {transport_key!r}.")
        out[transport_key] = param
        if kind == "lora_A":
            stems.append(stem)
    if alpha is not None:
        for stem in stems:
            out[f"{stem}.alpha"] = torch.tensor(float(alpha))
    return out


def peft_adapter_state_dict(
    named_params: Sequence[Tuple[str, torch.Tensor]],
    *,
    adapter_name: str = LORA_ADAPTER_NAME,
) -> Dict[str, torch.Tensor]:
    """Rename adapter tensors to the on-disk HF-PEFT layout.

    ``<stem>.lora_A.<adapter>.weight`` ->
    ``base_model.model.<stem>.lora_A.weight``, which is what
    ``peft.PeftModel.from_pretrained`` expects to find in
    ``adapter_model.safetensors``.
    """
    out: Dict[str, torch.Tensor] = {}
    for name, param in named_params:
        split = _split_lora_param(name, adapter_name)
        if split is None:
            continue
        stem, kind = split
        stem = strip_transport_wrappers(stem)
        out[f"base_model.model.{stem}.{kind}.weight"] = param
    return out


def lora_metadata_dict(
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
    adapter_name: str = LORA_ADAPTER_NAME,
) -> Dict[str, Any]:
    """The LoRA block recorded in ``adapter_contract.json``.

    ``alpha`` in particular cannot be recovered from the weights, so a resume
    or an offline merge that guesses it produces a silently rescaled policy
    rather than an error. ``scaling`` is derived here so an offline tool needs
    nothing else.
    """
    return {
        "rank": int(rank),
        "alpha": int(alpha),
        "dropout": float(dropout),
        "target_modules": [str(t) for t in target_modules],
        "bias": str(bias),
        "task_type": str(task_type),
        "adapter_name": str(adapter_name),
        "scaling": float(alpha) / float(rank) if rank else 0.0,
    }


def contract_mismatches(recorded: Mapping[str, Any], live: Mapping[str, Any]) -> List[str]:
    """Fields where a checkpoint's LoRA config disagrees with the live one.

    A rank mismatch would surface as an opaque shape error deep inside DCP; an
    alpha or target-module mismatch produces no error at all, just a
    differently scaled or differently placed adapter. Both are caught here
    before the load.
    """
    keys = ("rank", "alpha", "dropout", "target_modules", "bias", "task_type", "adapter_name")
    out: List[str] = []
    for key in keys:
        want, got = recorded.get(key), live.get(key)
        if key == "target_modules":
            want, got = list(want or []), list(got or [])
        if want != got:
            out.append(f"{key}: checkpoint={want!r} != current={got!r}")
    return out
