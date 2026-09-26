# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bundle subsetting for replay selection.

select_bundle materializes a replayable subset of a loaded bundle: it expands
the selection to a complete semantic-group closure, filters the sample records,
slices the flat per-token tensor payloads, and filters per-sample expected
lists — so the per-sample/per-token adapters run unchanged on the subset.
Cohort-level stages (loss.policy) are not subset-replayable and are handled by
the runner (see relax.utils.replay.runner).
"""

from __future__ import annotations

import torch

from relax.utils.replay.bundle import LoadedBundle
from relax.utils.replay.identity import expand_selection
from relax.utils.replay.schema import BundleIndex, SampleRecord


def _slice_flat(tensor: torch.Tensor, full_samples: list[SampleRecord], selected_ids: list[str]) -> torch.Tensor:
    """Slice a flat per-token tensor to the selected samples' token spans."""
    offsets: dict[str, int] = {}
    lengths: dict[str, int] = {}
    offset = 0
    for record in full_samples:
        offsets[record.sample_id] = offset
        lengths[record.sample_id] = record.response_length
        offset += record.response_length
    chunks = [tensor[offsets[sample_id] : offsets[sample_id] + lengths[sample_id]] for sample_id in selected_ids]
    if not chunks:
        return tensor.new_empty((0,), dtype=tensor.dtype)
    return torch.cat(chunks, dim=0)


def _filter_expected(
    expected: dict[str, object], full_samples: list[SampleRecord], selected_ids: list[str]
) -> dict[str, object]:
    """Filter per-sample expected lists (length == sample count) to the
    selection."""
    sample_count = len(full_samples)
    positions = {record.sample_id: index for index, record in enumerate(full_samples)}
    selected_positions = [positions[sample_id] for sample_id in selected_ids]

    def convert(value: object) -> object:
        if isinstance(value, list) and len(value) == sample_count:
            return [value[index] for index in selected_positions]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return {key: convert(value) for key, value in expected.items()}


def select_bundle(
    bundle: LoadedBundle,
    *,
    sample_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
    batch_ids: list[str] | None = None,
) -> LoadedBundle:
    """Return a subset bundle over the selection's semantic-group closure.

    The identity, manifest and per-step expected outputs are preserved; only
    the sample records, flat tensors and per-sample expected lists are
    narrowed. A selection that expands to the whole bundle is returned
    unchanged.
    """
    closure = expand_selection(bundle.index, sample_ids=sample_ids, group_ids=group_ids, batch_ids=batch_ids)
    full_samples = bundle.index.samples
    if len(closure.sample_ids) == len(full_samples):
        return bundle

    selected_ids = closure.sample_ids
    records_by_id = {record.sample_id: record for record in full_samples}
    samples = [records_by_id[sample_id] for sample_id in selected_ids]

    tensors = {name: _slice_flat(tensor, full_samples, selected_ids) for name, tensor in bundle.tensors.items()}
    expected = _filter_expected(bundle.expected, full_samples, selected_ids)

    index = BundleIndex(
        bundle_id=bundle.index.bundle_id,
        identity=bundle.index.identity,
        samples=samples,
        config=bundle.index.config,
    )
    return LoadedBundle(manifest=bundle.manifest, index=index, expected=expected, tensors=tensors)
