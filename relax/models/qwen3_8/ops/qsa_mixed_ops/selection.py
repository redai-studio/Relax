# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Invocation-local mixed metadata preserved by Megatron checkpoint copies."""

from dataclasses import dataclass, fields

from megatron.core.transformer.experimental_attention_variant.qsa import QSASelection


@dataclass
class MixedQSASelection(QSASelection):
    # Megatron strips/restores indexer tensors with dataclasses.replace twice.
    # A declared field survives both copies; an ad-hoc attribute does not.
    _relax_qsa_mixed_bounds: tuple[int, ...] | None = None


def bind_document_bounds(selection: QSASelection, bounds: tuple[int, ...] | None) -> MixedQSASelection:
    return MixedQSASelection(
        **{field.name: getattr(selection, field.name) for field in fields(QSASelection) if field.init},
        _relax_qsa_mixed_bounds=bounds,
    )
