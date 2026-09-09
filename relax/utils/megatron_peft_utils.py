# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Utilities for PEFT (Parameter-Efficient Fine-Tuning) of Megatron in
Relax."""

import re
from fnmatch import fnmatchcase
from typing import Iterable, Tuple

import torch

from relax.utils.env import Envs, is_env_set


# Fixed name under which the trained policy LoRA adapter is registered on the rollout
# engines in adapter mode. Generation requests must pass ``lora_path=LORA_ADAPTER_NAME``
# for the adapter to take effect (see sglang_rollout.generate and _push_lora_adapter).
LORA_ADAPTER_NAME = "relax_policy_lora"
VISION_REGION_TOKENS = ("vision_model", "visual", "vit", "image_encoder", "projector", "audio", "vision_tower")
_LORA_NO_MATCH_SENTINEL = "__relax_lora_no_match__"


def count_adapter_parameters(model) -> Tuple[int, int, float]:
    """Count the number of trainable adapter parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (adapter_params, total_params, percentage)
    """
    from megatron.core.utils import unwrap_model

    unwrapped = unwrap_model(model)
    if not isinstance(unwrapped, list):
        unwrapped = [unwrapped]

    adapter_params = 0
    total_params = 0

    # Sum across every virtual-pipeline chunk; with VPP `model` is a list of chunks.
    for chunk in unwrapped:
        for name, param in chunk.named_parameters():
            total_params += param.numel()
            if is_lora_adapter_param(name) and param.requires_grad:
                adapter_params += param.numel()

    percentage = 100 * adapter_params / total_params if total_params > 0 else 0

    return adapter_params, total_params, percentage


GDN_MEGATRON_MODULE = "in_proj"
GDN_SGLANG_FUSED_MODULE = "in_proj_qkvz"
GDN_QKVZ_PARTS = ("in_proj_qkv", "in_proj_z")
GDN_GATE_PARTS = ("in_proj_b", "in_proj_a")


# Fused Megatron module name -> the HF-style projections it expands to. Megatron is the
# canonical form used everywhere internally (CLI, injection, weight sync); the mapping is
# one-to-many because a single fused Megatron linear covers several HF projections
# (linear_qkv -> q/k/v_proj, linear_fc1 -> gate/up_proj). Expansion is therefore lossless,
# unlike the reverse direction which could not be expressed uniquely.
MEGATRON_TO_HF_MODULES = {
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    "router": ["gate"],
    # Split-QKV / split-gate-up variants
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
    # MLA projections
    "linear_kv_down_proj": ["kv_a_proj_with_mqa"],
    "linear_kv_up_proj": ["kv_b_proj"],
    "linear_q_down_proj": ["q_a_proj"],
    "linear_q_up_proj": ["q_b_proj"],
    "linear_q_proj": ["q_proj"],
    GDN_MEGATRON_MODULE: [*GDN_QKVZ_PARTS, *GDN_GATE_PARTS],
    "out_proj": ["out_proj"],
}

# Overrides applied on top of MEGATRON_TO_HF_MODULES for the SGLang flavor.
MEGATRON_TO_SGLANG_MODULES = {
    GDN_MEGATRON_MODULE: [GDN_SGLANG_FUSED_MODULE],
}


def _expand_target_modules(megatron_modules: list[str], overrides: dict[str, list[str]]) -> list[str]:
    """Expand Megatron names via ``overrides`` first, then
    ``MEGATRON_TO_HF_MODULES``.

    Unknown names pass through unchanged (already HF-style or custom).
    Duplicates are removed while preserving order.
    """
    expanded: list[str] = []
    for module in megatron_modules:
        expanded.extend(overrides.get(module) or MEGATRON_TO_HF_MODULES.get(module, [module]))
    return list(dict.fromkeys(expanded))


def convert_megatron_to_hf_target_modules(megatron_modules: list[str]) -> list[str]:
    """Expand Megatron-style LoRA target module names to HF-style names.

    Relax uses Megatron-style names (``linear_qkv``/``linear_proj``/...) as the
    canonical form: the CLI accepts them and Megatron-Bridge's LoRA matcher walks
    the Megatron module tree directly. HF-style names are only needed when writing a
    standard HF-PEFT ``adapter_config.json`` (or building an inference PEFT config),
    so translation happens at export time via this one-to-many expansion.

    Matches the names Megatron-Bridge's ``export_adapter_weights`` emits, so this is the
    flavor for the checkpoint adapter — the engine push repacks GDN and needs
    :func:`convert_megatron_to_sglang_target_modules` instead.

    Args:
        megatron_modules: List of Megatron-style module names
            (e.g. ``["linear_qkv", "linear_proj"]``).

    Returns:
        List of HF-style module names with duplicates removed. Unknown names are
        passed through unchanged (already HF-style or custom).
    """
    return _expand_target_modules(megatron_modules, {})


def convert_megatron_to_sglang_target_modules(megatron_modules: list[str]) -> list[str]:
    """Expand Megatron-style LoRA target module names to the names SGLang
    matches by.

    Same expansion as :func:`convert_megatron_to_hf_target_modules` except where SGLang's
    module tree fuses differently from HF's (``MEGATRON_TO_SGLANG_MODULES``). SGLang selects
    modules by *leaf name* (``LoRAManager.init_lora_modules``) and validates that an adapter's
    ``target_modules`` are a subset of the server's ``--lora-target-modules``, so both the
    engine launch flag and the pushed adapter config must use this flavor — an HF-only name
    such as ``in_proj_qkv`` matches no SGLang module: on the launch flag it fails buffer
    allocation outright, and in a pushed adapter config the weights are merely skipped.

    Args:
        megatron_modules: List of Megatron-style module names.

    Returns:
        List of SGLang-style module names with duplicates removed.
    """
    return _expand_target_modules(megatron_modules, MEGATRON_TO_SGLANG_MODULES)


def is_lora_adapter_param(name: str) -> bool:
    """Return True if ``name`` is a LoRA adapter parameter.

    Matches both naming conventions, anchored on the dotted module segment so a
    base weight that merely *contains* the substring (e.g. a module named
    ``lora_gate``) is not misclassified — a false positive here would silently
    drop a base weight from weight-sync buckets:
    - Megatron-Bridge: ``...adapter.linear_in.weight`` / ``...adapter.linear_out.weight``
    - Standard PEFT:   ``...lora_A.weight`` / ``...lora_B.weight``
    """
    return ".lora_A." in name or ".lora_B." in name or ".adapter.linear_in." in name or ".adapter.linear_out." in name


_LORA_TENSOR_SUFFIXES = (
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_A.default.weight",
    ".lora_B.default.weight",
)


def _split_lora_tensor_name(name: str) -> Tuple[str, str] | None:
    """Split ``<module path>.lora_A|B[.default].weight`` into ``(module_path,
    suffix)``.

    Returns ``None`` for anything that is not an HF-PEFT adapter tensor name.
    """
    for suffix in _LORA_TENSOR_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return None


def repack_gdn_adapter_for_sglang(adapter: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite Bridge's 4-way GDN ``in_proj`` adapter into SGLang's fused
    ``in_proj_qkvz``.

    Megatron trains ONE adapter on the fused ``in_proj`` ([q, k, v, z, b, a]); Megatron-Bridge
    exports it as four HF tensors sharing a single ``lora_A``. SGLang instead expects the
    ``[q, k, v, z]`` block as one ``in_proj_qkvz`` adapter (``lora_A`` is replicated 4x by its
    own ``_normalize_in_proj_qkvz``, which is exactly the fused semantics) and has no LoRA
    support at all for the ``b``/``a`` gate slices. So:

    * ``lora_B`` of ``in_proj_qkv`` and ``in_proj_z`` are concatenated along the output dim;
    * the shared ``lora_A`` is emitted once, unchanged;
    * ``in_proj_b`` / ``in_proj_a`` are dropped — and asserted to be exactly zero, which
      ``install_gdn_gate_mask_hooks`` guarantees by giving those rows no gradient. A nonzero
      value here would mean the trained policy and the rollout policy disagree on the GDN
      gates, so it is a hard error rather than a silent truncation.

    Non-GDN entries and adapters from models without GDN layers pass through untouched.

    Args:
        adapter: HF-named adapter tensors as returned by ``bridge.export_adapter_weights``.

    Returns:
        The adapter with every GDN ``in_proj`` group replaced by its fused counterpart.
    """
    repacked: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    for name in adapter:
        parsed = _split_lora_tensor_name(name)
        if parsed is None:
            continue
        module_path, suffix = parsed
        if not module_path.endswith("." + GDN_QKVZ_PARTS[0]):
            continue
        # Keep the trailing dot so ``prefix + part`` rebuilds a sibling module path.
        prefix = module_path[: -len(GDN_QKVZ_PARTS[0])]

        parts = {}
        for part in GDN_QKVZ_PARTS + GDN_GATE_PARTS:
            tensor = adapter.get(f"{prefix}{part}{suffix}")
            if tensor is None:
                raise ValueError(
                    f"Incomplete GDN in_proj adapter group for '{prefix}*{suffix}': "
                    f"'{part}' is missing. Megatron-Bridge emits all four of "
                    f"{GDN_QKVZ_PARTS + GDN_GATE_PARTS} together, so this points at an "
                    "unexpected export layout rather than a configuration problem."
                )
            parts[part] = tensor
            consumed.add(f"{prefix}{part}{suffix}")

        if ".lora_B." in suffix:
            for gate_part in GDN_GATE_PARTS:
                if bool(parts[gate_part].any()):
                    raise ValueError(
                        f"GDN gate adapter '{prefix}{gate_part}{suffix}' is nonzero, but SGLang "
                        "cannot host LoRA on the GDN b/a slices, so it would be dropped and the "
                        "rollout policy would silently diverge from the trained policy. Ensure "
                        "install_gdn_gate_mask_hooks() ran after LoRA injection (adapter mode), "
                        "or use --lora-merge-mode, which folds these slices into the base weights."
                    )
            repacked[f"{prefix}{GDN_SGLANG_FUSED_MODULE}{suffix}"] = torch.cat(
                [parts[part] for part in GDN_QKVZ_PARTS], dim=0
            )
        else:
            # lora_A is shared across all four projections; emit one copy.
            repacked[f"{prefix}{GDN_SGLANG_FUSED_MODULE}{suffix}"] = parts[GDN_QKVZ_PARTS[0]]

    if not consumed:
        return adapter

    fused = {name: tensor for name, tensor in adapter.items() if name not in consumed}
    fused.update(repacked)
    return fused


def _gdn_gate_mask_hook(num_gate_rows: int):
    """Build a forward hook that zeroes the trailing ``num_gate_rows`` of an
    adapter output."""

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            return output
        # In-place index_put_ on the adapter's own (freshly allocated) output. Autograd records
        # it, so the corresponding rows of linear_out receive exactly zero gradient and stay at
        # their zero init forever — the gates are frozen, not merely hidden at export time.
        output[..., -num_gate_rows:] = 0
        return output

    return hook


def install_gdn_gate_mask_hooks(model) -> int:
    """Pin the GDN ``in_proj`` LoRA delta to zero on the b/a (beta/alpha gate)
    rows.

    Only meaningful in LoRA *adapter mode*: SGLang can carry an adapter on the fused
    ``in_proj_qkvz`` ([q, k, v, z]) but has none for the gate slices, so any LoRA delta learned
    on ``b``/``a`` could never be replayed at rollout time. Masking the adapter's *output* (not
    the weight) means those rows get zero gradient and remain at their zero init, which keeps
    training and rollout numerically identical instead of merely truncating at export.

    Not applied in merge mode, where the gate slices fold into the base weights losslessly.

    Args:
        model: A single model chunk that has already been through ``peft(model)``.

    Returns:
        The number of GDN modules whose input projection was masked (0 when the model has no
        GDN layers, or when ``in_proj`` is not among the LoRA target modules).
    """
    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except ImportError:
        return 0

    masked = 0
    for module in model.modules():
        if not isinstance(module, GatedDeltaNet):
            continue
        in_proj = getattr(module, "in_proj", None)
        adapter = getattr(in_proj, "adapter", None)
        if adapter is None:
            continue  # in_proj was not wrapped (not a LoRA target)
        if getattr(adapter, "_relax_gdn_gate_masked", False):
            continue
        adapter.register_forward_hook(_gdn_gate_mask_hook(2 * module.num_v_heads_local_tp))
        adapter._relax_gdn_gate_masked = True
        masked += 1
    return masked


def scope_target_modules_to_region(model, target_modules: list[str], scope: str) -> list[str]:
    """Narrow LoRA target modules to a model region (language / vision / all).

    Megatron-Bridge's LoRA matcher matches ``target_modules`` by *leaf name*, so a bare
    ``linear_qkv`` is injected into BOTH the language backbone and the vision tower of a
    VL model (the "vision tower mistakenly gets LoRA" issue). This resolves the requested
    ``scope`` against the concrete module tree of ``model`` and returns the *full-path*
    names to wrap; the matcher then matches each exactly (an exact string is a wildcard
    pattern with no ``*``). ``named_modules()`` names match the matcher's ``full_name``
    because both derive from the same recursive walk.

    Args:
        model: The single, already-built model chunk about to be wrapped. Names from
            ``named_modules()`` are relative to this chunk root and stay consistent within
            one ``peft(model)`` call.
        target_modules: The user's Megatron-style target module names (may contain ``*``).
        scope: ``"all"`` (passthrough, no walk), ``"language"`` (exclude vision), or
            ``"vision"`` (only vision).

    Returns:
        For ``scope == "all"`` the original ``target_modules`` unchanged. Otherwise the
        full module paths in the requested region; if that set is empty (e.g. a PP stage
        or VP chunk with no module of that region), a single non-matching sentinel so the
        matcher wraps nothing instead of every linear layer.
    """
    if scope == "all":
        return target_modules

    from megatron.bridge.peft.utils import wildcard_match

    want_vision = scope == "vision"
    scoped: list[str] = []
    for full_name, _module in model.named_modules():
        if not full_name:
            continue  # skip the unnamed root module
        leaf = full_name.rsplit(".", 1)[-1]
        if not any(leaf == t or wildcard_match(t, full_name) for t in target_modules):
            continue
        is_vision = any(tok in full_name for tok in VISION_REGION_TOKENS)
        if is_vision == want_vision:
            scoped.append(full_name)

    return scoped or [_LORA_NO_MATCH_SENTINEL]


def exclude_frozen_lora_target_modules(model, target_modules: list[str], freeze_patterns: Iterable[str]) -> list[str]:
    """Resolve LoRA targets to full module paths and drop frozen regions.

    PEFT adapters still execute their dropout path after their parameters are
    frozen.  For strict numerical comparisons this advances the RNG even when
    the zero-initialized adapter contributes nothing to the forward output.
    Filtering before injection avoids that side effect while preserving any
    explicitly unfrozen subregion, such as a vision merger excluded by a
    negative-lookahead freeze regex.
    """
    patterns = tuple(freeze_patterns)
    filtered: list[str] = []
    for full_name, _module in model.named_modules():
        if not full_name:
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if not any(
            leaf == target or full_name == target or fnmatchcase(full_name, target) for target in target_modules
        ):
            continue
        if any(re.search(pattern, full_name) for pattern in patterns):
            continue
        filtered.append(full_name)

    return filtered or [_LORA_NO_MATCH_SENTINEL]


def lora_module_category(name: str) -> str:
    """Bucket a LoRA-wrapped module (or adapter param) by architectural role.

    Used only for human-readable logging summaries. Rules are ordered so the
    more specific match wins: ``shared_experts`` is checked before the generic
    ``experts`` (routed) because the former's name contains ``_experts`` not
    ``.experts.``. Vision is separated so the language-only intent is visible
    at a glance (see the vision-exclusion gotcha in the LoRA MoE design doc).

    Vision tokens are matched WITHOUT requiring surrounding dots: at model-build
    (injection) time a VL model's params are named relative to the sub-model, so
    the token appears at the *start* (``vision_model.blocks...``), while later
    (checkpoint / weight-sync) the same param carries a ``module.module.`` prefix
    (``...module.vision_model....``). A leading-dot match would classify the same
    module differently at the two sites. Expert tokens keep their dots because
    they are always mid-path (``.mlp.experts.`` / ``.mlp.shared_experts.``) and
    the dots are what separate routed from shared.
    """
    if "vision_model" in name or "visual" in name or "vision_tower" in name:
        return "vision"
    if ".shared_experts." in name:
        return "shared_expert"
    if ".experts." in name:
        return "routed_expert"
    if "router" in name:
        return "router"
    if "linear_fc" in name:
        return "mlp"
    if ".in_proj." in name or ".out_proj." in name:
        return "gdn"
    return "attention"


def _lora_module_key(adapter_param_name: str) -> str:
    """Collapse a LoRA adapter param name to its wrapped-module key.

    Drops the adapter suffix (``.adapter.linear_in|out.weight`` or
    ``.lora_A|B.weight``) and any grouped-expert ``weight{N}`` index, so
    ``linear_in``/``linear_out`` (and every packed expert) map to a single
    module. Used to count *modules* rather than *params*.
    """
    name = adapter_param_name.replace(".adapter.linear_in.weight", "").replace(".adapter.linear_out.weight", "")
    name = re.sub(r"\.lora_[AB]\.weight\d*$", "", name)
    return name


def summarize_lora_modules(adapter_param_names: Iterable[str]) -> dict[str, int]:
    """Count unique LoRA-wrapped modules per architectural role.

    Takes an iterable of adapter param names and returns e.g.
    ``{"attention": 160, "routed_expert": 80, "shared_expert": 40, "vision": 0}``.
    ``linear_in``/``linear_out`` of the same module are collapsed to one count.
    """
    buckets: dict[str, set] = {}
    for name in adapter_param_names:
        buckets.setdefault(lora_module_category(name), set()).add(_lora_module_key(name))
    return {cat: len(keys) for cat, keys in sorted(buckets.items())}


def build_hf_peft_config_dict(
    *,
    lora_rank: int,
    lora_alpha: int,
    target_modules,
    lora_dropout: float,
    task_type: str = "CAUSAL_LM",
) -> dict:
    """Build the HF-PEFT adapter config dict (the content of
    ``adapter_config.json``).

    Single source of truth for both transports: the disk path serializes this
    to ``adapter_config.json`` and the from-tensors path passes it straight to
    SGLang's ``LoRAConfig.from_dict``. JSON-safe (no enums). ``target_modules``
    must already be HF-style names (see
    ``convert_megatron_to_hf_target_modules``).

    ``task_type`` defaults to the LLM value; diffusion DiTs must pass
    ``FEATURE_EXTRACTION`` or a PEFT loader will try to attach a causal-LM head.
    """
    return {
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),
        "lora_dropout": lora_dropout,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": task_type,
    }


def write_hf_peft_adapter(
    merged: dict[str, torch.Tensor],
    adapter_dir,
    *,
    lora_rank: int,
    lora_alpha: int,
    target_modules,
    lora_dropout: float,
    task_type: str = "CAUSAL_LM",
) -> str:
    """Write a merged LoRA adapter as a standard HF-PEFT directory.

    Produces ``adapter_config.json`` + ``adapter_model.safetensors`` — the on-disk
    format used by the checkpoint save, which Megatron-Bridge's ``load_peft_adapter``
    can read back.

    Args:
        merged: Full (TP-gathered, PP-merged) adapter tensors keyed by HF name.
        adapter_dir: Target directory (created if missing).
        lora_rank/lora_alpha/target_modules/lora_dropout: PEFT config written to
            ``adapter_config.json`` (HF-style target module names).
        task_type: PEFT task type; ``FEATURE_EXTRACTION`` for diffusion DiTs.

    Returns:
        The adapter directory path as a string.
    """
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # HF PEFT adapter config (JSON-safe: no enums).
    config_dict = build_hf_peft_config_dict(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type=task_type,
    )
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config_dict, f)

    # safetensors requires contiguous CPU tensors.
    state = {name: t.contiguous() for name, t in merged.items()}
    save_file(state, str(adapter_dir / "adapter_model.safetensors"))
    return str(adapter_dir)


def is_lora_enabled(args) -> bool:
    """Check if LoRA is enabled in training arguments."""
    return hasattr(args, "lora_rank") and args.lora_rank > 0


def is_lora_merge_mode(args) -> bool:
    """Check if LoRA merge mode is enabled.

    In merge mode, LoRA adapters are merged into base weights before sync.

    Args:
        args: Training arguments

    Returns:
        True if merge mode, False otherwise
    """
    return hasattr(args, "lora_merge_mode") and args.lora_merge_mode


def is_lora_adapter_mode(args) -> bool:
    """Check if LoRA adapter mode is enabled.

    In adapter mode the base model is synced once and each step only the trained LoRA
    adapter is pushed to the rollout engine via SGLang's runtime LoRA API. Mutually
    exclusive with merge mode (enforced at arg-validation time).

    Args:
        args: Training arguments

    Returns:
        True if adapter mode, False otherwise
    """
    return hasattr(args, "lora_adapter_mode") and args.lora_adapter_mode


def extract_lora_delta(
    new_lora_state: dict[str, torch.Tensor],
    old_lora_state: dict[str, torch.Tensor] | None = None,
    threshold: float = 1e-6,
) -> Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute the change (delta) of LoRA parameters between two syncs.

    Used by adapter mode's incremental sync: the first sync returns the full state,
    subsequent syncs return only parameters that changed by more than ``threshold``.
    The delta is used ONLY to decide whether a sync is needed (delta-skip); the tensors
    actually pushed to the engine are the absolute adapter values, not this delta.

    Args:
        new_lora_state: Current LoRA parameter state.
        old_lora_state: Previous state (None means first sync -> full state returned).
        threshold: Ignore changes below this value (float noise).

    Returns:
        ``(delta_dict, new_state_dict)``:
        - delta_dict: parameters that changed and would need syncing.
        - new_state_dict: a clone of the current state, saved for the next comparison.
    """
    if old_lora_state is None:
        # First sync: everything is "new"; clone twice so caller's saved state is
        # decoupled from the returned delta.
        return (
            {k: v.clone() for k, v in new_lora_state.items()},
            {k: v.clone() for k, v in new_lora_state.items()},
        )

    delta: dict[str, torch.Tensor] = {}
    for name, new_tensor in new_lora_state.items():
        if name not in old_lora_state:
            # Brand-new parameter: include its full value.
            delta[name] = new_tensor.clone()
            continue
        param_delta = new_tensor - old_lora_state[name]
        # .abs().max().item() forces a GPU->CPU sync; acceptable here because this runs
        # on the (infrequent) weight-update path, not a training/rollout hot path.
        if param_delta.abs().max().item() > threshold:
            delta[name] = param_delta

    new_state = {k: v.clone() for k, v in new_lora_state.items()}
    return delta, new_state


def build_lora_peft(args):
    """Build a Megatron-Bridge PEFT (LoRA) object from training arguments.

    Single source of truth for the LoRA config: the model provider applies this to
    the model, and adapter (re)loading reuses it so the adapter-key filter matches
    exactly what was applied. CLI exposes Megatron-style module names (``linear_qkv``,
    ...) which is what Megatron-Bridge's LoRA matcher walks, so they pass through
    unchanged.

    Args:
        args: Training arguments carrying ``lora_rank`` / ``lora_alpha`` /
            ``lora_target_modules`` / ``lora_dropout``.

    Returns:
        The Megatron-Bridge PEFT object (not yet applied to a model).
    """
    try:
        from megatron.bridge.peft.utils import create_peft
    except ImportError as e:
        raise RuntimeError(
            "LoRA training requires a newer training image with LoRA-enabled Megatron-Bridge. "
            "Please upgrade the training image."
        ) from e

    peft_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "target_modules": list(args.lora_target_modules),
        "dropout": args.lora_dropout,
        "bias": "none",
    }
    # Megatron-Bridge defaults to Xavier A initialization and one adapter
    # shared by all local experts.  Some external PEFT baselines use Kaiming
    # initialization and one adapter per expert instead.  Keep Relax defaults
    # unchanged, but allow launchers doing strict numerical comparisons to
    # select those Bridge-native options without adding framework CLI flags.
    lora_a_init_method = Envs.RELAX_LORA_A_INIT_METHOD.strip()
    if lora_a_init_method:
        peft_config["lora_A_init_method"] = lora_a_init_method
    if is_env_set("RELAX_LORA_SHARE_EXPERT_ADAPTERS"):
        peft_config["share_expert_adapters"] = Envs.RELAX_LORA_SHARE_EXPERT_ADAPTERS
    return create_peft(peft_config)


__all__ = [
    "LORA_ADAPTER_NAME",
    "count_adapter_parameters",
    "convert_megatron_to_hf_target_modules",
    "convert_megatron_to_sglang_target_modules",
    "MEGATRON_TO_HF_MODULES",
    "MEGATRON_TO_SGLANG_MODULES",
    "write_hf_peft_adapter",
    "extract_lora_delta",
    "install_gdn_gate_mask_hooks",
    "is_lora_enabled",
    "is_lora_merge_mode",
    "is_lora_adapter_mode",
    "build_lora_peft",
    "lora_module_category",
    "repack_gdn_adapter_for_sglang",
    "summarize_lora_modules",
    "scope_target_modules_to_region",
    "exclude_frozen_lora_target_modules",
    "VISION_REGION_TOKENS",
]
