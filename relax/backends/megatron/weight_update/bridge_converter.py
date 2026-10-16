# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
import re
from argparse import Namespace
from collections.abc import Collection, Sequence
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.backends.megatron.weight_conversion.processors import quantize_params, remove_padding
from relax.backends.megatron.weight_update.common import named_params_and_buffers
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# DeepSeek-V4 mHC stores one scalar nn.Parameter per alpha; the checkpoint stores
# them as a single 3-element `hc_*_scale`. Bridge ships _HCAlphaMapping for this,
# but its megatron_to_hf() reads alpha_post/alpha_res straight off the live module
# instead of using the tensors Relax hands it. During colocate weight sync the
# module's parameter storage has already been released by torch_memory_saver, so
# those reads return dangling device pointers and fault (observed: alpha_pre at
# 0x7f21... copies fine, alpha_post/alpha_res at 0xe91cc... raise CUDA "invalid
# argument", then torch.cat escalates to an illegal memory access).
# Relax converts every parameter separately and always passes a valid tensor, so
# buffer the three alphas here and emit the concatenation ourselves.
_HC_ALPHA_RE = re.compile(r"^(?P<group>.*)\.alpha_(?P<which>pre|post|res)$")
_HC_ALPHA_ORDER = ("pre", "post", "res")

# Schema aliases accepted by released checkpoints.  The emitted name is
# selected against the checkpoint key set instead of pinning the fast path to
# whichever spelling a particular Bridge version happens to use.
_HF_NAME_ALIAS_PAIRS = ((".indexer.scorer.weights_proj.", ".indexer.weights_proj."),)

# Kimi K3 routed experts (latent MoE): Bridge maps them 1:1 onto the native
# HF names ``...block_sparse_moe.experts.<id>.w[123].weight``. The hf_param
# spelling varies with the bridge flavor (VL wrapping adds a ``language_model.``
# megatron-side prefix but leaves the HF name unprefixed), so accept both.
_MXFP4_ROUTED_EXPERT_HF_NAME = re.compile(
    r"(?:language_model\.)?(?:model\.)?layers\.\d+\.(?:block_sparse_moe|mlp)\.experts\.\d+\.w[123]\.weight"
)


def _is_mxfp4_routed_expert(mapping) -> bool:
    """Whether ``mapping`` converts a Kimi K3 routed-expert w1/w2/w3 weight."""
    if not getattr(mapping, "is_expert", False) or getattr(mapping, "is_adapter", False):
        return False
    names = list(mapping.hf_param.values()) if isinstance(mapping.hf_param, dict) else [mapping.hf_param]
    return bool(names) and all(isinstance(n, str) and _MXFP4_ROUTED_EXPERT_HF_NAME.fullmatch(n) for n in names)


class _Mxfp4OnlineQuantizer:
    """Quantizer callable for Bridge's ``megatron_to_hf_quant`` protocol.

    Input: a 2-D BF16 expert weight (already TP/EP-gathered) and the block
    size ``(1, 32)``. Output: ``(packed, scale)`` using the bridge's reference
    MXFP4 arithmetic — the same kernel as the offline export
    (``examples/models/kimi-k3/tools/convert_kimi_k3_torch_dist_to_hf_parallel.py``),
    so online pushes stay byte-compatible with the original release.
    """

    def __call__(self, weight, block_size):
        from megatron.bridge.models.conversion.quantization_utils import quantize_mxfp4_e2m1_like_scale

        if block_size != (1, 32) or weight.ndim != 2 or weight.shape[1] % 32:
            raise ValueError(f"Unsupported online MXFP4 geometry: {tuple(weight.shape)}, {block_size}")
        template = torch.empty(
            (weight.shape[0], weight.shape[1] // block_size[1]), dtype=torch.uint8, device=weight.device
        )
        packed, scale = quantize_mxfp4_e2m1_like_scale(weight, template, block_size=block_size[1])
        # Same output contract as the offline export's LocalExpertQuantizer:
        # fail loud here, because a numeric cast downstream (e.g. float8 scale
        # -> uint8) would silently corrupt the exponent grid.
        if packed.dtype != torch.int8 or tuple(packed.shape) != (weight.shape[0], weight.shape[1] // 2):
            raise ValueError(f"Invalid packed MXFP4 output from quantizer: {packed.dtype} {tuple(packed.shape)}")
        if scale.dtype != torch.uint8 or tuple(scale.shape) != tuple(template.shape):
            raise ValueError(f"Invalid E8M0 scale output from quantizer: {scale.dtype} {tuple(scale.shape)}")
        return packed, scale


def _resolve_hf_name_against_checkpoint(hf_name: str, checkpoint_keys: Collection[str]) -> str:
    """Resolve a known bidirectional alias against the source checkpoint
    schema."""
    if hf_name in checkpoint_keys:
        return hf_name

    for first, second in _HF_NAME_ALIAS_PAIRS:
        if first in hf_name:
            candidate = hf_name.replace(first, second, 1)
        elif second in hf_name:
            candidate = hf_name.replace(second, first, 1)
        else:
            continue
        if candidate in checkpoint_keys:
            return candidate

    return hf_name


def _get_hf_checkpoint_keys(bridge: Any) -> frozenset[str] | None:
    """Read checkpoint keys without materializing tensors.

    ``None`` means that this Bridge version exposes neither the current public
    key API nor the older source API.  An empty set means key inspection worked
    and the checkpoint is empty.
    """
    state = getattr(getattr(bridge, "hf_pretrained", None), "state", None)
    if state is None:
        return None

    keys = getattr(state, "keys", None)
    if callable(keys):
        try:
            return frozenset(keys())
        except (AttributeError, NotImplementedError):
            # Older/lazy StateDict implementations expose ``keys`` but defer
            # enumeration to the source object below.  Do not swallow I/O or
            # checkpoint corruption errors.
            pass

    source = getattr(state, "source", None)
    get_all_keys = getattr(source, "get_all_keys", None)
    if callable(get_all_keys):
        return frozenset(get_all_keys())

    return None


def _normalize_global_name(name: str) -> str:
    global_name = strip_param_name_prefix(name)
    if global_name.startswith("vp_stages."):
        parts = global_name.split(".", 2)
        if len(parts) >= 3:
            global_name = parts[2]
    return global_name


def _noop_gather_from_ep_ranks(self_m, megatron_weights, megatron_module, hf_param_name):
    return {str(hf_param_name): megatron_weights}


class BridgeConverter:
    """Per-parameter megatron-to-HF conversion using megatron-bridge.

    All collective communication (PP broadcast, TP gather, EP gather) is
    disabled by temporarily setting the bridge mapping process groups to
    ``None``.  The caller is responsible for TP gather and EP gather
    *before* calling :meth:`convert`.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self._args = args
        self._model = model
        self._quantization_config = quantization_config
        self._bridge_task_map: dict[str, Any] | None = None
        self._bridge_mapping_registry: Any = None
        self._bridge_expert_transposes_down: bool = True
        self._configs_broadcast_done: bool = False
        self._hf_alias_checkpoint_keys: frozenset[str] | None = None
        # DeepSeek-V4 mHC alpha buffering, see _convert_hc_alpha.
        self._hc_alpha_buf: dict[str, dict[str, torch.Tensor]] = {}
        self._hc_alpha_hf_name: dict[str, str] = {}
        # Kimi K3 mxfp4-pack-quantized releases: routed experts are quantized
        # inside Bridge's quantized mappings, everything else passes through.
        self._mxfp4_pack_format = bool(
            quantization_config and quantization_config.get("format") == "mxfp4-pack-quantized"
        )
        self._mxfp4_quantizer = _Mxfp4OnlineQuantizer() if self._mxfp4_pack_format else None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def init_tasks(self) -> None:
        """Build the bridge task map on first use.

        Builds a mapping from ``global_param_name`` (e.g.
        ``decoder.layers.0.self_attention.linear_qkv.weight``) to the
        corresponding ``WeightConversionTask``.  Only tasks whose
        ``param_weight is not None`` (i.e. belonging to the current PP
        rank) are indexed.

        Also eagerly initialises any lazily-created inner mappings
        (``AutoMapping._mapping``) so that :meth:`collect_all_mappings`
        can discover and patch them later.
        """
        if self._bridge_task_map is not None:
            return

        from megatron.bridge import AutoBridge
        from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        from relax.utils.megatron_bridge_utils import patch_megatron_model

        bridge = AutoBridge.from_hf_pretrained(self._args.hf_checkpoint, trust_remote_code=True)
        checkpoint_keys = _get_hf_checkpoint_keys(bridge)
        if checkpoint_keys is not None:
            self._hf_alias_checkpoint_keys = frozenset(
                key
                for key in checkpoint_keys
                if any(first in key or second in key for first, second in _HF_NAME_ALIAS_PAIRS)
            )
        with patch_megatron_model(self._model):
            tasks = bridge.get_conversion_tasks(self._model)

        self._bridge_task_map = {}
        for task in tasks:
            # Bridge's task builder may leave a ``None`` entry when a mapping
            # does not apply to the source checkpoint on this rank.
            if task is not None and task.param_weight is not None:
                self._bridge_task_map[task.global_param_name] = task

        self._bridge_mapping_registry = bridge._model_bridge.mapping_registry()
        mapping_registry = self._bridge_mapping_registry
        for name, _param in named_params_and_buffers(self._args, self._model, include_persistent_buffers=True):
            global_name = _normalize_global_name(name)
            if global_name in self._bridge_task_map:
                continue
            # Keep the existing registry fallback for parameters. It also lets
            # older Bridge releases add persistent buffers they did not
            # enumerate. For buffers, repeat Bridge's HF-schema check so a
            # registry entry alone cannot revive an inapplicable mapping.
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if mapping is not None:
                if (
                    not isinstance(_param, torch.nn.Parameter)
                    and checkpoint_keys is not None
                    and not mapping.allow_hf_name_mismatch
                ):
                    hf_param_names = (
                        [mapping.hf_param] if isinstance(mapping.hf_param, str) else mapping.hf_param.values()
                    )
                    if any(hf_name not in checkpoint_keys for hf_name in hf_param_names):
                        continue
                self._bridge_task_map[global_name] = WeightConversionTask(
                    param_name=global_name,
                    global_param_name=global_name,
                    mapping=mapping,
                    megatron_module=None,
                    param_weight=_param,
                )

        for task in self._bridge_task_map.values():
            mapping = task.mapping
            if isinstance(mapping, AutoMapping) and mapping._mapping is None:
                if task.megatron_module is not None:
                    mapping._detected_type = mapping._detect_parallelism_type(task.megatron_module)
                    mapping._mapping = mapping._get_or_create_mapping(mapping._detected_type)
                else:
                    mapping._detected_type = "replicated"
                    mapping._mapping = mapping._get_or_create_mapping("replicated")
            inner_tp = getattr(mapping, "_tp_mapping", None)
            if isinstance(inner_tp, AutoMapping) and inner_tp._mapping is None:
                if task.megatron_module is not None:
                    inner_tp._detected_type = inner_tp._detect_parallelism_type(task.megatron_module)
                    inner_tp._mapping = inner_tp._get_or_create_mapping(inner_tp._detected_type)

        self._config_map: dict[str, Any] = {}
        for task in self._bridge_task_map.values():
            if task.megatron_module is not None:
                prefix = task.global_param_name.split(".")[0]
                if prefix not in self._config_map:
                    self._config_map[prefix] = task.megatron_module.config

        # Patch local tasks that have megatron_module=None (Phase 2 tasks
        # from named_params_and_buffers that AutoBridge didn't produce).
        for name, task in list(self._bridge_task_map.items()):
            if task.megatron_module is None:
                prefix = name.split(".")[0]
                config = self._config_map.get(prefix)
                if config is not None:
                    self._bridge_task_map[name] = dataclasses.replace(
                        task, megatron_module=SimpleNamespace(config=config)
                    )

        self._bridge_expert_transposes_down = False
        for task in self._bridge_task_map.values():
            cls = type(task.mapping)
            if cls.__name__ == "ExpertMLPDownProjMapping":
                self._bridge_expert_transposes_down = "megatron_to_hf" in cls.__dict__
                break

        logger.info("Bridge task map initialized with %d local tasks", len(self._bridge_task_map))

    def can_convert(self, name: str) -> bool:
        """Return whether Bridge has a mapping for a parameter or persistent
        buffer."""
        self.init_tasks()
        assert self._bridge_task_map is not None
        global_name = _normalize_global_name(name)
        return global_name in self._bridge_task_map

    def broadcast_and_apply_configs(self) -> None:
        """Broadcast ``_config_map`` across PP ranks and patch remaining tasks.

        Must be called by all PP ranks after :meth:`init_tasks`.  After this
        call every task in ``_bridge_task_map`` has a non-None
        ``megatron_module`` with the correct ``.config`` for QKV split.

        Safe to call multiple times; the broadcast only runs once.
        """
        if self._configs_broadcast_done:
            return
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size > 1:
            from megatron.bridge.models.conversion.utils import remove_non_pickleables

            # Keep live configs local: they can contain process groups that
            # all_gather_object cannot pickle. Bridge cleans a copy for PP transfer.
            local_config_map = {prefix: remove_non_pickleables(config) for prefix, config in self._config_map.items()}
            all_config_maps: list[dict[str, Any] | None] = [None] * pp_size
            dist.all_gather_object(
                obj=local_config_map,
                object_list=all_config_maps,
                group=mpu.get_pipeline_model_parallel_group(),
            )
            for remote_map in all_config_maps:
                for prefix, cfg in remote_map.items():
                    if prefix not in self._config_map:
                        self._config_map[prefix] = cfg

        for name, task in list(self._bridge_task_map.items()):
            if task.megatron_module is None:
                prefix = name.split(".")[0]
                config = self._config_map.get(prefix)
                if config is not None:
                    self._bridge_task_map[name] = dataclasses.replace(
                        task, megatron_module=SimpleNamespace(config=config)
                    )
        self._configs_broadcast_done = True

    # ------------------------------------------------------------------
    # Mapping collection
    # ------------------------------------------------------------------

    @staticmethod
    def collect_all_mappings(mapping) -> list:
        """Recursively collect a mapping and all its inner sub-mappings."""
        from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

        result: list = []
        visited: set = set()
        stack = [mapping]
        while stack:
            m = stack.pop()
            if id(m) in visited:
                continue
            visited.add(id(m))
            if isinstance(m, MegatronParamMapping):
                result.append(m)
                for attr_val in vars(m).values():
                    if isinstance(attr_val, MegatronParamMapping):
                        stack.append(attr_val)
        return result

    @contextmanager
    def _disable_mapping_collectives(self, mapping):
        """Null every process group on ``mapping`` (and inner sub-mappings) and
        replace ``gather_from_ep_ranks`` with a local passthrough.

        ``convert`` runs only on the src rank of each PP/EP group (expert
        weights) or after the caller has already gathered (TP), so any
        collective a mapping tries internally would silently run on the WORLD
        group once its group handle is None — deadlocking or gathering garbage.
        Group nulling alone is therefore not enough; the EP gather must be
        patched out as well. Patch only the participating instances so other
        mappings of the same class retain their collective implementations.
        """
        all_mappings = self.collect_all_mappings(mapping)
        saved_groups: list[tuple] = [(m.pp_group, m._tp_group, m._etp_group, m.ep_group) for m in all_mappings]
        missing = object()
        saved_gathers = [vars(m).get("gather_from_ep_ranks", missing) for m in all_mappings]
        try:
            for m in all_mappings:
                m.pp_group = None
                m._tp_group = None
                m._etp_group = None
                m.ep_group = None
                m.gather_from_ep_ranks = MethodType(_noop_gather_from_ep_ranks, m)
            yield
        finally:
            for m, (pp, tp, etp, ep) in zip(all_mappings, saved_groups):
                m.pp_group = pp
                m._tp_group = tp
                m._etp_group = etp
                m.ep_group = ep
            for m, original in zip(all_mappings, saved_gathers):
                if original is missing:
                    vars(m).pop("gather_from_ep_ranks", None)
                else:
                    m.gather_from_ep_ranks = original

    def _lookup_task(self, global_name: str) -> tuple[str, Any]:
        """Find the bridge's own ``WeightConversionTask`` for ``global_name``.

        Prefers the bridge's task over a rebuilt one because it carries the real
        ``megatron_module`` (and hence the parallelism actually detected for the module)
        rather than a config-only donor.

        Returns ``(resolved_name, task)``; ``task`` is ``None`` when neither spelling matched,
        in which case ``resolved_name`` is returned unchanged.
        """
        task = self._bridge_task_map.get(global_name)
        if task is not None:
            return global_name, task

        # Property 2 holds for params the bridge saw; a plain key can still exist for one it
        # skipped (e.g. a PP rank that does not own the layer, where ``init_tasks`` drops the
        # task via its ``param_weight is not None`` filter).
        if ".to_wrap." in global_name:
            plain_name = global_name.replace(".to_wrap.", ".")
            task = self._bridge_task_map.get(plain_name)
            if task is not None:
                return plain_name, task

        return global_name, None

    def _lookup_mapping(self, global_name: str) -> tuple[str, Any]:
        """Resolve ``global_name`` straight from the mapping registry.

        Used only when :meth:`_lookup_task` came up empty. Per property 1 the registry speaks
        no LoRA, so a wrapped name is retried plain.

        Returns ``(resolved_name, mapping)``; ``mapping`` is ``None`` when nothing matched.
        """
        mapping = self._bridge_mapping_registry.megatron_to_hf_lookup(global_name)
        if mapping is None and ".to_wrap." in global_name:
            global_name = global_name.replace(".to_wrap.", ".")
            mapping = self._bridge_mapping_registry.megatron_to_hf_lookup(global_name)
        return global_name, mapping

    # ------------------------------------------------------------------
    # Per-parameter conversion
    # ------------------------------------------------------------------

    def _convert_hc_alpha(self, name, global_name, param, task, match) -> list[tuple[str, torch.Tensor]]:
        """Buffer DSv4 mHC alphas and emit `hc_*_scale` once all three arrived.

        Returns an empty list for the first two alphas of a group; the third
        one emits ``{hf_param: cat([pre, post, res])}``. Only the ``alpha_pre``
        task carries the resolved HF name (the other two map to no-op
        secondaries), so the name is remembered independently of arrival order.
        """
        group = match.group("group")
        which = match.group("which")

        hf_param = getattr(task.mapping, "hf_param", None)
        if isinstance(hf_param, str) and hf_param:
            self._hc_alpha_hf_name[group] = hf_param

        buf = self._hc_alpha_buf.setdefault(group, {})
        buf[which] = param.detach().reshape(-1).float()

        if len(buf) < len(_HC_ALPHA_ORDER):
            return []

        hf_name = self._hc_alpha_hf_name.get(group)
        self._hc_alpha_buf.pop(group, None)
        if hf_name is None:
            logger.warning(
                "mHC alpha group %s completed but no HF name was seen; dropping (mapping=%s)",
                group,
                type(task.mapping).__name__,
            )
            return []
        self._hc_alpha_hf_name.pop(group, None)
        scale = torch.cat([buf[k] for k in _HC_ALPHA_ORDER])
        return quantize_params(self._args, name, [(hf_name, scale)], self._quantization_config)

    def _convert_mxfp4_expert(
        self, global_name: str, name: str, param: torch.Tensor, task: Any, mapping: Any
    ) -> list[tuple[str, torch.Tensor]]:
        """Routed experts of an mxfp4-pack-quantized release (Kimi K3).

        Bridge's quantized mapping converts the gathered BF16 expert weight
        into the native packed pair — ``w*.weight_packed`` plus E8M0
        ``w*.weight_scale`` — via the reference
        ``quantize_mxfp4_e2m1_like_scale`` kernel, so the rollout engine
        receives tensors byte-compatible with the original release. All other
        namespaces of such a release are ignored per config and flow through
        the plain path below.
        """
        with self._disable_mapping_collectives(mapping):
            try:
                converted_dict = mapping.megatron_to_hf_quant(
                    param, task.megatron_module, lambda _hf_name: True, self._mxfp4_quantizer, (1, 32)
                )
            except Exception:
                logger.error(
                    "megatron_to_hf_quant failed: name=%s mapping=%s param.shape=%s module=%s",
                    global_name,
                    type(mapping).__name__,
                    tuple(param.shape),
                    type(task.megatron_module).__name__ if task.megatron_module else "None",
                )
                raise

        weights = {hf_name for hf_name in converted_dict if not hf_name.endswith("_scale_inv")}
        scales = {hf_name[: -len("_scale_inv")] for hf_name in converted_dict if hf_name.endswith("_scale_inv")}
        if weights != scales:
            raise ValueError(
                f"Quantized mapping returned unpaired weights/scales for {global_name}: {sorted(weights ^ scales)}"
            )

        named_tensors = []
        for hf_name, tensor in converted_dict.items():
            if hf_name.endswith("_scale_inv"):
                if tensor.dtype != torch.uint8:
                    raise ValueError(
                        f"Expected uint8 E8M0 scale for {hf_name}, got {tensor.dtype}; "
                        "refusing a numeric cast that would corrupt the exponent grid"
                    )
                named_tensors.append((hf_name[: -len("_scale_inv")] + "_scale", tensor))
            else:
                # Bridge returns the packed code stream as int8; the release
                # schema (and SGLang's loader) store it as uint8.
                named_tensors.append((hf_name + "_packed", tensor.view(torch.uint8)))
        return named_tensors

    def convert(self, name: str, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Convert a single TP/EP-gathered parameter to HF format.

        Args:
            name: Global parameter name with ``module.module.`` prefix
                  (as yielded by ``named_params_and_buffers``).
            param: The fully-gathered parameter tensor.

        Returns:
            List of ``(hf_name, hf_tensor)`` tuples (quantised if configured).
        """
        self.init_tasks()

        global_name = _normalize_global_name(name)

        global_name, task = self._lookup_task(global_name)

        if task is None:
            from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
            from megatron.bridge.models.conversion.param_mapping import AutoMapping

            global_name, mapping = self._lookup_mapping(global_name)

            assert mapping is not None, (
                f"Bridge mapping registry has no entry for '{global_name}'. "
                f"Available task map keys: {list(self._bridge_task_map.keys())[:10]}..."
            )
            prefix = global_name.split(".")[0]
            config = self._config_map.get(prefix)
            if config is None:
                config = next(iter(self._config_map.values()), None)
            donor = SimpleNamespace(config=config) if config is not None else None
            task = WeightConversionTask(
                param_name=global_name,
                global_param_name=global_name,
                mapping=mapping,
                megatron_module=donor,
                param_weight=None,
            )
            if isinstance(mapping, AutoMapping) and mapping._mapping is None:
                mapping._detected_type = "replicated"
                mapping._mapping = mapping._get_or_create_mapping("replicated")
            inner_tp = getattr(mapping, "_tp_mapping", None)
            if isinstance(inner_tp, AutoMapping) and inner_tp._mapping is None:
                inner_tp._detected_type = "replicated"
                inner_tp._mapping = inner_tp._get_or_create_mapping("replicated")
            self._bridge_task_map[global_name] = task

        hc_alpha = _HC_ALPHA_RE.match(global_name)
        if hc_alpha is not None:
            return self._convert_hc_alpha(name, global_name, param, task, hc_alpha)

        mapping = task.mapping

        if self._mxfp4_pack_format and _is_mxfp4_routed_expert(mapping):
            return self._convert_mxfp4_expert(global_name, name, param, task, mapping)

        with self._disable_mapping_collectives(mapping):
            param = remove_padding(name, param, self._args.vocab_size)
            try:
                converted_dict = mapping.megatron_to_hf(param, task.megatron_module)
            except Exception:
                logger.error(
                    "megatron_to_hf failed: name=%s mapping=%s param.shape=%s module=%s",
                    global_name,
                    type(mapping).__name__,
                    tuple(param.shape),
                    type(task.megatron_module).__name__ if task.megatron_module else "None",
                )
                raise

        converted_named_tensors = []
        for hf_name, tensor in converted_dict.items():
            if self._hf_alias_checkpoint_keys is None:
                # Preserve the original Relax fallback for Bridge versions that
                # do not expose checkpoint keys.
                first, second = _HF_NAME_ALIAS_PAIRS[0]
                resolved_name = hf_name.replace(first, second)
            else:
                resolved_name = _resolve_hf_name_against_checkpoint(hf_name, self._hf_alias_checkpoint_keys)
            converted_named_tensors.append((resolved_name, tensor))

        # Post-process expert weights: split fused gate_up_proj, fix transposes
        expert_id_match = re.search(r"weight(\d+)", global_name)
        if expert_id_match is not None:
            expert_id = expert_id_match.group(1)
            postprocessed: list[tuple[str, torch.Tensor]] = []
            for hf_name, tensor in converted_named_tensors:
                if hf_name.endswith(".experts.gate_up_proj"):
                    base = hf_name[: -len(".gate_up_proj")]
                    if tensor.ndim == 3:
                        gate_tensor = tensor[0].transpose(-1, -2).contiguous()
                        up_tensor = tensor[1].transpose(-1, -2).contiguous()
                    else:
                        gate_tensor, up_tensor = tensor.chunk(2, dim=0)
                    postprocessed.append((f"{base}.{expert_id}.gate_proj.weight", gate_tensor))
                    postprocessed.append((f"{base}.{expert_id}.up_proj.weight", up_tensor))
                elif hf_name.endswith(".experts.down_proj"):
                    base = hf_name[: -len(".down_proj")]
                    if tensor.ndim == 2 and not self._bridge_expert_transposes_down:
                        postprocessed.append((f"{base}.{expert_id}.down_proj.weight", tensor))
                    else:
                        postprocessed.append(
                            (f"{base}.{expert_id}.down_proj.weight", tensor.transpose(-1, -2).contiguous())
                        )
                else:
                    postprocessed.append((hf_name, tensor))
            converted_named_tensors = postprocessed

        return quantize_params(self._args, name, converted_named_tensors, self._quantization_config)
