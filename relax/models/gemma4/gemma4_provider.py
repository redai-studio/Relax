# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Gemma-4 providers adapted for Relax's training path.

Two unrelated upstream assumptions break under Relax, one per variant:

* **dense** -- Megatron-Bridge wires it to TE's attention, which emits NaN
  gradients in the packed (THD) backward. See :mod:`relax.models.gemma4.attention`.
* **MoE** -- its ``rotary_base`` is a tuple, and Relax's ``bridge_keys`` override
  flattens it to a scalar. See :class:`_Gemma4MoEProvideMixin`.

Both are delivered by :class:`~relax.models.gemma4.gemma4_bridge.Gemma4DenseBridge`
re-pointing the provider instance at its counterpart in :data:`RELAX_PROVIDERS`.

Keep the subclasses declared statically -- Relax pickles the provider to reach
the Ray train actors, and a class built by ``type(name, (mixin, base), {})`` is
not importable, so pickling it raises ``PicklingError``.
"""

from contextlib import contextmanager
from functools import partial

from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4DenseVLProvider, Gemma4VLModelProvider

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@contextmanager
def _relax_core_attention():
    """Rebind ``get_gemma4_layer_spec`` for the duration of one ``build()``.

    ``Gemma4DenseProvider.build()`` ignores its ``transformer_layer_spec``
    field and calls the module-level ``get_gemma4_layer_spec(config)``
    directly, so assigning that field does nothing here. Restored in
    ``finally``.
    """
    from megatron.bridge.models.gemma import gemma4_provider

    from relax.models.gemma4.attention import Gemma4CoreAttention

    original = gemma4_provider.get_gemma4_layer_spec

    def get_gemma4_layer_spec(config=None):
        spec = original(config)
        # AttributeError here is intentional if upstream restructures the spec.
        spec.submodules.self_attention.submodules.core_attention = Gemma4CoreAttention
        return spec

    gemma4_provider.get_gemma4_layer_spec = get_gemma4_layer_spec
    try:
        yield
    finally:
        gemma4_provider.get_gemma4_layer_spec = original


class _PackedSafeBuildMixin:
    """Builds the model with Relax's core attention in place of TE's."""

    def build(self, *args, **kwargs):
        with _relax_core_attention():
            model = super().build(*args, **kwargs)
        logger.info("gemma-4 dense: core_attention -> Gemma4CoreAttention (packed-safe)")
        return model


class PackedSafeGemma4DenseProvider(_PackedSafeBuildMixin, Gemma4DenseProvider):
    """``Gemma4DenseProvider`` for the text-only path
    (GEMMA4_CONVERSION_MODE=text)."""


class PackedSafeGemma4DenseVLProvider(_PackedSafeBuildMixin, Gemma4DenseVLProvider):
    """``Gemma4DenseVLProvider`` for the VL path.

    Untested -- Relax's gemma-4
    work is text-only SFT so far.
    """


#: Where the bridge parks the ``(local, global)`` rope-theta pair. Deliberately
#: not a ``bridge_keys`` name, so Relax's override loop leaves it alone.
_ROPE_PAIR_ATTR = "_relax_gemma4_rotary_base_pair"

#: Guards against double-wrapping ``transformer_layer_spec`` if ``provide()`` runs
#: more than once on the same provider (e.g. virtual pipeline stages).
_PACKED_SAFE_ATTR = "_relax_gemma4_packed_safe_installed"


def stash_dual_rope(provider) -> None:
    """Park a MoE provider's ``rotary_base`` tuple out of ``bridge_keys``'
    reach.

    Must run before Relax's arg-override loop; read back in ``provide()``.
    """
    if isinstance(provider.rotary_base, tuple):
        setattr(provider, _ROPE_PAIR_ATTR, provider.rotary_base)


def _packed_safe_moe_block_spec(inner_spec_fn, config):
    """Build upstream's MoE block spec, then swap the attention on every layer.

    Module level and bound through ``functools.partial`` rather than a closure,
    for the pickling reason in the module docstring. Upstream's per-layer
    ``Gemma4TEDotProductAttention`` lands on the cuDNN kernel that NaNs in the
    THD backward -- see :mod:`relax.models.gemma4.attention`.

    The signature must not grow a ``vp_stage`` parameter: ``provide()`` inspects
    for one and would pass it to a builder that does not take it.
    """
    from relax.models.gemma4.attention import Gemma4CoreAttention

    block_spec = inner_spec_fn(config)
    # AttributeError here is intentional if upstream restructures the spec.
    for layer_spec in block_spec.layer_specs:
        layer_spec.submodules.self_attention.submodules.core_attention = Gemma4CoreAttention
    return block_spec


class _Gemma4MoEProvideMixin:
    """Fix-ups that must run before upstream's MoE ``provide()``.

    1. Restore the ``(local, global)`` rope thetas that ``bridge_keys`` flattened
       to a scalar -- ``provide()`` unpacks the tuple on its first line, so a
       scalar raises ``TypeError``. No launch-script flag avoids it: one int
       cannot express two thetas.
    2. Install the packed-safe attention. See :func:`_packed_safe_moe_block_spec`.
    """

    def provide(self, *args, **kwargs):
        pair = getattr(self, _ROPE_PAIR_ATTR, None)
        if pair is not None and not isinstance(self.rotary_base, tuple):
            logger.info(
                "gemma-4 MoE: restoring rotary_base %r -> %r (bridge_keys had flattened it)", self.rotary_base, pair
            )
            self.rotary_base = pair

        if not getattr(self, _PACKED_SAFE_ATTR, False):
            self.transformer_layer_spec = partial(_packed_safe_moe_block_spec, self.transformer_layer_spec)
            setattr(self, _PACKED_SAFE_ATTR, True)
            logger.info("gemma-4 MoE: core_attention -> Gemma4CoreAttention (packed-safe)")

        return super().provide(*args, **kwargs)


class RelaxGemma4MoEProvider(_Gemma4MoEProvideMixin, Gemma4ModelProvider):
    """``Gemma4ModelProvider`` for the text-only MoE path."""


class RelaxGemma4MoEVLProvider(_Gemma4MoEProvideMixin, Gemma4VLModelProvider):
    """``Gemma4VLModelProvider`` for the MoE VL path.

    The VL model reaches the language model through
    ``provide_language_model()``, which delegates to the ``provide()`` hooked
    here.
    """


# No subclass adds fields, so an instance can be re-pointed in place and keep
# the attributes the bridge set outside the dataclass.
RELAX_PROVIDERS = {
    Gemma4DenseProvider: PackedSafeGemma4DenseProvider,
    Gemma4DenseVLProvider: PackedSafeGemma4DenseVLProvider,
    Gemma4ModelProvider: RelaxGemma4MoEProvider,
    Gemma4VLModelProvider: RelaxGemma4MoEVLProvider,
}
