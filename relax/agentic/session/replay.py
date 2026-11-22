# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay bodies and response leases owned by an Agentic session or group."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from relax.agentic.session.state import InflightRequest, merge_routed_experts
from relax.distributed import replay_manager
from relax.distributed.replay_manager.config import rollout_kinds
from relax.distributed.replay_manager.producer import PublisherHoldings
from relax.distributed.replay_manager.ref_data import validate_sample


class ReplayHoldings(PublisherHoldings):
    """Connect selected Session exports and accepted token lineage to their
    publisher holds."""

    def transfer_exports(self, group: ReplayHoldings, exports: Sequence[dict]) -> None:
        self.transfer_to(group, (export["sample_payload"]["rollout_replay_refs"] or {} for export in exports))

    def apply_response(self, args: Any, ir: InflightRequest, result: Any) -> None:
        kinds = rollout_kinds(args)
        meta = result.meta_info
        producer = replay_manager.replay_data_manager.register_producer(kinds)
        media = producer.response_media(meta, required=bool(result.new_tokens) and result.finish_type != "abort")
        if media is not None:
            ir.pending_mm_embedding_refs = media
        if "moe" not in kinds:
            return
        size = len(result.new_tokens)
        if not size and meta.get("routed_experts") is None:
            return
        raw = getattr(result, "raw_output_ids", None)
        captured_delta = ir.pending_token_delta
        if size and raw is not None:
            captured_delta = ir.pending_token_delta[:-size] + raw
        capture_prefix = ir.history_rollout_token_prefix
        if len(capture_prefix) != len(ir.history_train_token_prefix):
            capture_prefix = ir.history_train_token_prefix
        tokens = ir.history_train_token_prefix + ir.pending_token_delta
        with producer.response_route_span(
            meta, args, tokens, capture_prefix + captured_delta, ir.referenced_route_rows, required=bool(size)
        ) as span:
            if span is not None:
                ir.pending_routed_experts = merge_routed_experts(
                    ir.pending_routed_experts, (span,), expected_rows=len(tokens) - 1
                )
                self.refs[span.ref.record] = span.ref


def validate_reference_exports(args: Any, exports: Sequence[dict]) -> None:
    kinds = rollout_kinds(args)
    if not kinds:
        return
    for export in exports:
        sample = export["sample_payload"]
        manifests = sample["rollout_replay_refs"] or {}
        if not kinds.issubset(manifests):
            raise ValueError("Agentic export is missing enabled replay kinds")
        validate_sample(manifests, sample["tokens"])
