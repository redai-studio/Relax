# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Replay bodies and response leases owned by an Agentic session or group."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from relax.agentic.session.state import InflightRequest, merge_routed_experts
from relax.distributed import replay_manager
from relax.distributed.replay_manager.backend.mm_sglang import META_FIELD
from relax.distributed.replay_manager.config import rollout_kinds
from relax.distributed.replay_manager.ref_data import (
    MoERouteSpan,
    ReplayRef,
    iter_refs,
    moe_route_spans,
    validate_mm_envelope,
    validate_sample,
)


@dataclass
class ReplayHoldings:
    refs: dict[str, ReplayRef] = field(default_factory=dict)
    media: dict[str, dict] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.refs or self.media)

    def transfer_exports(self, group: ReplayHoldings, exports: Sequence[dict]) -> None:
        for export in exports:
            manifests = export["sample_payload"]["rollout_replay_refs"] or {}
            for ref in iter_refs(manifests):
                if ref.kind != "mm_embedding":
                    group.refs[ref.record] = ref
                    self.refs.pop(ref.record, None)
            for entry in manifests.get("mm_embedding", {}).get("media", ()):
                group.media[entry["lease"]] = entry
                self.media.pop(entry["lease"], None)

    async def discard(self) -> None:
        if self:
            await self._release(list(self.refs.values()), list(self.media.values()))

    async def _release(self, refs: list[ReplayRef], media: list[dict]) -> None:
        store = replay_manager.replay_data_manager.store()
        await store.discard_async(refs, media=media)
        for ref in refs:
            self.refs.pop(ref.record, None)
        for entry in media:
            self.media.pop(entry["lease"], None)

    async def adopt_response(
        self, args: Any, session_id: str, meta: dict, is_current: Callable[[], bool], *, cleanup_timeout: float
    ) -> bool:
        kinds = rollout_kinds(args)
        manager = replay_manager.replay_data_manager
        if not manager.uses_references or not kinds:
            return True
        producer = manager.register_producer(kinds)
        with producer.response(meta):
            refs = producer.response_refs(meta)
            media = meta.get(META_FIELD, []) if "mm_embedding" in kinds else []
            if "mm_embedding" in kinds and META_FIELD in meta:
                validate_mm_envelope({"version": 1, "media": media})
            if not is_current():
                await manager.store().invoke_async("discard", refs, unadopted=True, media=media)
                return False
            # Media holds are distinct even for repeated occurrences of one record.
            self.media.update((entry["lease"], entry) for entry in media)
            if refs:
                await producer.adopt_response(refs, session_id)
                self.refs.update((ref.record, ref) for ref in refs)
            if not is_current():
                await asyncio.wait_for(self._release(refs, media), cleanup_timeout)
                return False
        return True

    def apply_response(self, args: Any, ir: InflightRequest, result: Any) -> None:
        kinds = rollout_kinds(args)
        meta = result.meta_info
        if "mm_embedding" in kinds:
            if META_FIELD in meta:
                ir.pending_mm_embedding_refs = {"version": 1, "media": meta[META_FIELD]}
            elif result.new_tokens and result.finish_type != "abort":
                raise ValueError("Backend omitted multimodal embedding references")
        if "moe" not in kinds:
            return
        if meta.get("routed_experts") is None:
            if result.new_tokens:
                raise ValueError("Backend omitted routed experts for newly accepted tokens")
            return
        producer = replay_manager.replay_data_manager.register_producer(["moe"])
        with producer.response(meta):
            size = len(result.new_tokens)
            raw = getattr(result, "raw_output_ids", None)
            captured_delta = ir.pending_token_delta
            if size and raw is not None:
                captured_delta = ir.pending_token_delta[:-size] + raw
            capture_prefix = ir.history_rollout_token_prefix
            if len(capture_prefix) != len(ir.history_train_token_prefix):
                capture_prefix = ir.history_train_token_prefix
            tokens = ir.history_train_token_prefix + ir.pending_token_delta
            start = ir.referenced_route_rows
            refs = producer.accept_route_refs(meta, args, tokens[start:], (capture_prefix + captured_delta)[start:])
            (span,) = moe_route_spans(refs["moe"])
            ref = span.ref
            if (ref.layers, ref.topk) != (args.num_layers, args.moe_router_topk):
                raise ValueError("Agentic MoE replay shape does not match the model")
            ir.pending_routed_experts = merge_routed_experts(
                ir.pending_routed_experts,
                (MoERouteSpan(ref, start, len(tokens) - 1, span.training_tokens),),
                expected_rows=len(tokens) - 1,
            )
            self.refs[ref.record] = ref


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
