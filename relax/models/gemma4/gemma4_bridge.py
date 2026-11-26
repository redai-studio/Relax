# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Relax's gemma-4 bridge: upstream's, with two variant fixes.

Subclasses upstream's bridge and only re-points the provider at its Relax
subclass in :data:`~relax.models.gemma4.gemma4_provider.RELAX_PROVIDERS`:
dense needs a packed-safe attention, MoE needs its ``rotary_base`` tuple kept
out of reach of Relax's ``bridge_keys`` override.
"""

from collections.abc import Callable

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4VLModelProvider
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

from relax.models.gemma4.gemma4_provider import RELAX_PROVIDERS, stash_dual_rope
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@MegatronModelBridge.register_bridge(
    source="Gemma4ForConditionalGeneration",
    target=Gemma4VLModel,
    provider=Gemma4VLModelProvider,
    model_type="gemma4_vl",
)
class Gemma4DenseBridge(Gemma4VLBridge):
    """Bridge for gemma-4 full-parameter SFT -- dense and MoE alike, despite
    the ``Dense`` in the name.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("google/gemma-4-31B-it")
        >>> provider = bridge.to_megatron_provider()

    ``GEMMA4_CONVERSION_MODE=text`` loads the VL checkpoint as text-only.
    """

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Keep absent consumer norms synthetic; source-layer checks stay
        strict."""
        registry = super().mapping_registry()
        config = self._text_config()
        if self._conversion_mode() != "text" or not self._is_dense_config() or not config.num_kv_shared_layers:
            return registry
        synthetic = []
        for layer in range(config.num_hidden_layers - config.num_kv_shared_layers, config.num_hidden_layers):
            name = f"decoder.layers.{layer}.self_attention.k_layernorm.weight"
            mapping = registry.megatron_to_hf_lookup(name)
            if mapping is None:
                raise ValueError(f"Missing Gemma shared-KV normalization mapping: {name}")
            mapping.allow_hf_name_mismatch = True
            synthetic.append(mapping)
        return MegatronMappingRegistry(*synthetic, *registry.mappings)

    def _make_weight_sync_export_filter(self) -> Callable | None:
        """Reuse native filtering for synthetic shared-KV parameters."""
        config = self._text_config()
        if self._conversion_mode() != "text" or not config.num_kv_shared_layers:
            return None
        import torch
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        with torch.device("meta"):
            model = Gemma4ForCausalLM(Gemma4TextConfig.from_dict(config.to_dict())).to(dtype=torch.bfloat16)
        state = model.state_dict(keep_vars=True)
        if config.tie_word_embeddings:
            if state.get("lm_head.weight") is not state.get("model.embed_tokens.weight"):
                raise ValueError("Gemma4 shared embeddings disagree with the native model")
            del state["lm_head.weight"]
        schema = {
            self._hf_layer_prefix() + name[len("model.") :] if name.startswith("model.") else name: tensor
            for name, tensor in state.items()
        }

        def filter_export(task, converted):
            return self.maybe_modify_converted_hf_weight(task, converted, schema)

        return filter_export

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM):
        """Return upstream's provider, re-pointed at its Relax subclass."""
        provider = super().provider_bridge(hf_pretrained)

        # Exact type, not isinstance -- each variant has its own subclass.
        replacement = RELAX_PROVIDERS.get(type(provider))
        if replacement is None:
            logger.debug("%s is not a known gemma-4 provider; leaving it alone", type(provider).__name__)
            return provider

        # Must run before Relax's bridge_keys override loop, which fires as soon
        # as this returns. No-op on dense.
        stash_dual_rope(provider)
        provider.__class__ = replacement
        return provider
