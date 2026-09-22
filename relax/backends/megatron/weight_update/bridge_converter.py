# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
import re
from argparse import Namespace
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.backends.megatron.weight_conversion.processors import quantize_params, remove_padding
from relax.backends.megatron.weight_update.common import named_params_and_buffers
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


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
        with patch_megatron_model(self._model):
            tasks = bridge.get_conversion_tasks(self._model)

        self._bridge_task_map = {}
        for task in tasks:
            if task.param_weight is not None:
                self._bridge_task_map[task.global_param_name] = task

        self._bridge_mapping_registry = bridge._model_bridge.mapping_registry()
        mapping_registry = self._bridge_mapping_registry
        for name, _param in named_params_and_buffers(self._args, self._model):
            global_name = strip_param_name_prefix(name)
            if global_name not in self._bridge_task_map:
                mapping = mapping_registry.megatron_to_hf_lookup(global_name)
                if mapping is not None:
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

    def broadcast_and_apply_configs(self) -> None:
        """Broadcast ``_config_map`` across PP ranks and patch remaining tasks.

        Must be called by all PP ranks after :meth:`init_tasks`.  After this
        call every task in ``_bridge_task_map`` has a non-None
        ``megatron_module`` with the correct ``.config`` for QKV split.

        Safe to call multiple times; the broadcast only runs once.
        """
        if self._configs_broadcast_done:
            return
        self._configs_broadcast_done = True
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size > 1:
            all_config_maps: list[dict[str, Any] | None] = [None] * pp_size
            dist.all_gather_object(
                obj=self._config_map,
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

        global_name = strip_param_name_prefix(name)
        if global_name.startswith("vp_stages."):
            parts = global_name.split(".", 2)
            if len(parts) >= 3:
                global_name = parts[2]

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

        mapping = task.mapping
        all_mappings = self.collect_all_mappings(mapping)

        saved_groups: list[tuple] = []
        for m in all_mappings:
            saved_groups.append((m.pp_group, m._tp_group, m._etp_group, m.ep_group))

        patched_classes: set[type] = set()

        try:
            for m in all_mappings:
                m.pp_group = None
                m._tp_group = None
                m._etp_group = None
                m.ep_group = None

            for m in all_mappings:
                cls = type(m)
                if cls not in patched_classes:
                    cls.gather_from_ep_ranks = _noop_gather_from_ep_ranks
                    patched_classes.add(cls)

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
        finally:
            for m, (pp, tp, etp, ep) in zip(all_mappings, saved_groups):
                m.pp_group = pp
                m._tp_group = tp
                m._etp_group = etp
                m.ep_group = ep
            for cls in patched_classes:
                if "gather_from_ep_ranks" in cls.__dict__:
                    del cls.gather_from_ep_ranks

        converted_named_tensors = list(converted_dict.items())

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
