# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Patch ``ColumnParallelMapping.hf_to_megatron`` in
# ``megatron.bridge.models.conversion.param_mapping`` so the vocab-padding
# check reads the *TP-local* target shape.
#
# Under ``--use-megatron-fsdp`` the module parameters the bridge inspects are
# Megatron-FSDP DTensors whose ``.shape`` is the TP-**global** shape
# (``make_fsdp_dtensor`` does ``global_shape[tp_dim] *= tp_mesh.numel()``); the
# TP-local shard shape lives on ``.orig_param``. Upstream already accounts for
# this when computing ``output_shape`` for the scatter, but the padding check a
# few lines above still uses ``target_param.shape[0] * tp_size``, which
# over-counts by ``tp_size`` and makes every column-parallel weight look like it
# needs vocab padding. The first such weight is the embedding, so an HF load
# with megatron-fsdp + TP>1 dies at startup with e.g.:
#
#   AssertionError: hf_weights.shape=torch.Size([248320, 2048])
#   target_param.shape=torch.Size([248320, 2048]) self.tp_size=4
#   self.megatron_param='language_model.embedding.word_embeddings.weight'
#
# Two corrections vs. upstream, both confined to that check:
#   1. Derive the expected dim-0 size from ``orig_param`` for DTensor targets,
#      i.e. the same shape upstream already scatters into.
#   2. Match the embedding/output-layer allowlist on the parameter-name suffix.
#      VLM/Omni bridges prefix their mappings (``language_model.embedding...``,
#      ``thinker.language_model.embedding...``), so the exact-equality check
#      rejects the one case padding is legitimately meant for.
#
# Everything else is an as-is copy of upstream. The patch is skipped (with a
# warning) if the upstream body no longer contains the buggy expression, so a
# Megatron upgrade that fixes this does not get silently clobbered.
#
# History: the megatron-fsdp branch carried an earlier fix for the same call
# site (``_patch_column_parallel_fsdp_output_shape``, a context manager around
# ``bridge.load_hf_weights``). It targeted MEGATRON_BRIDGE_COMMIT=2faedbf6,
# where upstream derived the scatter shape as ``target.shape[0] // tp_size``
# for every DTensor and so double-split ``output_layer``. That fix does not
# carry forward: on the current MEGATRON_BRIDGE_COMMIT=af17edf5 upstream
# switched to ``orig_param`` (fixing output_layer) and added the vocab-padding
# check patched here, and the old fix deliberately delegated genuinely
# TP-sharded DTensors like ``word_embeddings`` straight into it. Keying off
# ``orig_param`` covers both parameters uniformly, so no separate output_layer
# special case is needed.

import inspect

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_BUGGY_EXPR = "expect_dim0_size = target_param.shape[0] * self.tp_size"
_PADDED_PARAMS = ("embedding.word_embeddings.weight", "output_layer.weight")

try:
    import torch
    from megatron.bridge.models.conversion.param_mapping import (
        ColumnParallelMapping,
        _pad_right_dim0,
        get_module_and_param_from_name,
    )
    from torch.distributed._tensor import DTensor

    if _BUGGY_EXPR not in inspect.getsource(ColumnParallelMapping.hf_to_megatron):
        logger.info(
            "Relax bridge column-parallel DTensor patch not needed — upstream "
            "ColumnParallelMapping.hf_to_megatron no longer sizes the vocab-padding "
            "check off target_param.shape."
        )
    else:

        def hf_to_megatron(self, hf_weights, megatron_module):
            """Split weight along dim 0 and distribute to TP ranks."""
            if self.tp_size == 1:
                return hf_weights

            # Some parameters are named with global expert number, e.g. experts.weight15,
            # normalize it to experts.weight0, note we are only use the shape, dtype, device info,
            # not the actual value, so it is safe to do this.
            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)

            # Megatron-FSDP wraps params in DTensors whose ``.shape`` is TP-global;
            # ``orig_param`` keeps the TP-local shard shape that the scatter below
            # writes into, so size the padding check off the same shape.
            if isinstance(target_param, DTensor):
                output_shape = target_param.orig_param.shape
            else:
                output_shape = target_param.shape

            # On rank 0, check for divisibility and split
            if self.tp_rank == 0:
                if hf_weights is None:
                    raise ValueError("hf_weights should not be None on rank 0")

                # Dtype may differ (e.g. MambaMixer A_log is FP32 in MCore but BF16
                # in HF checkpoints). Cast to match the Megatron parameter so the
                # scatter doesn't fail on dtype mismatch.
                if hf_weights.dtype != target_param.dtype:
                    if not getattr(ColumnParallelMapping, "_dtype_warned", False):
                        ColumnParallelMapping._dtype_warned = True
                        logger.warning(
                            f"Dtype mismatch: HF weights are {hf_weights.dtype} but Megatron "
                            f"module uses {target_param.dtype}. Casting all mismatched weights "
                            f"to {target_param.dtype} (further warnings suppressed)."
                        )
                    hf_weights = hf_weights.to(target_param.dtype)

                actual_dim0_size = hf_weights.shape[0]
                expect_dim0_size = output_shape[0] * self.tp_size
                if actual_dim0_size != expect_dim0_size:
                    assert self.megatron_param.endswith(_PADDED_PARAMS), (
                        f"{hf_weights.shape=} {output_shape=} {self.tp_size=} {self.megatron_param=} {self.hf_param=}"
                    )
                    hf_weights = _pad_right_dim0(hf_weights, pad_size=expect_dim0_size - actual_dim0_size)

                # For bias (1D), we still split along dim 0
                # For weight (2D), we split along dim 0 (output dimension)
                full_size = hf_weights.shape[0]
                if full_size % self.tp_size != 0:
                    raise ValueError(
                        f"Cannot evenly split dimension 0 size {full_size} across {self.tp_size} TP ranks"
                    )
                splits = torch.chunk(hf_weights, self.tp_size, dim=0)

            else:
                splits = None

            # Scatter to all ranks. Each rank gets its sharded shape from its module.
            return self.scatter_to_tp_ranks(
                splits,
                output_shape,
                target_param.dtype,
                target_param.device,
            )

        ColumnParallelMapping.hf_to_megatron = hf_to_megatron

        logger.info(
            "Relax bridge column-parallel DTensor patch applied to "
            "megatron.bridge.models.conversion.param_mapping.ColumnParallelMapping.hf_to_megatron"
        )

except ImportError as exc:
    logger.warning(
        "Relax bridge column-parallel DTensor patch not applied — Megatron import failed (%r). "
        "If this is a Megatron upgrade, the symbol layout may have changed; without this patch, "
        "loading an HF checkpoint with --use-megatron-fsdp and TP>1 will assert on the embedding.",
        exc,
    )
