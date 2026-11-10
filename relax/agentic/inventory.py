# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Creation inventories, owned by the driver before Session preparation."""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from argparse import Namespace
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Sequence

from relax.utils.agentic_obs.config import resolve_agentic_obs_config, write_manifest
from relax.utils.agentic_obs.records import (
    ExportStageRecord,
    GroupRecord,
    PromptRecord,
    SessionRecord,
    strip_media,
    utc_now,
)
from relax.utils.agentic_obs.source import prompt_identity
from relax.utils.agentic_obs.writers import CLOSE_TIMEOUT, AsyncJsonlAppender, best_effort, close_background
from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from relax.utils.types import Sample


logger = get_logger(__name__)


def _prompt_preview(prompt: Any) -> str:
    if isinstance(prompt, str):
        return strip_media([{"text": prompt}])[0]["text"][:512]
    messages = [prompt] if isinstance(prompt, dict) else prompt if isinstance(prompt, list) else []
    texts = []
    for message in strip_media(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
    return " ".join(texts)[:512]


def session_sampling_config(args: Namespace, overrides: dict[str, Any] | None, position: int) -> dict[str, Any]:
    params = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "max_new_tokens": args.rollout_max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    # --sglang-* attrs only exist when sglang args were parsed (not skip_sglang).
    # This function is rollout-path only, so they are always present here.
    if args.sglang_enable_deterministic_inference:
        params["sampling_seed"] = args.rollout_seed + position
    if overrides is not None:
        params.update(overrides)
    return {key: value for key, value in params.items() if value is not None}


class InventoryRecorder:
    def __init__(self, args: Namespace, rollout_mode: str) -> None:
        self.config = resolve_agentic_obs_config(args)
        self.args = args
        self.rollout_mode = rollout_mode
        self._writers: dict[str, AsyncJsonlAppender] = {}
        self._prompts: OrderedDict[str, None] = OrderedDict()
        self._groups: OrderedDict[str, None] = OrderedDict()
        if self.config is not None:
            write_manifest(args, self.config)
            producer_id = f"{rollout_mode}_{uuid.uuid4().hex}"
            self._writers = {
                stream: AsyncJsonlAppender(
                    self.config.inventory_path(stream, producer_id),
                    media_dir=self.config.media_dir if self.config.dump_media else None,
                )
                for stream in ("prompts", "groups", "sessions", "stages")
            }

    def record_group(
        self,
        group_id: str,
        samples: Sequence[Sample],
        session_ids: Sequence[str] | None = None,
        *,
        started_at: str | None = None,
    ) -> None:
        if self.config is None or not samples:
            return
        try:
            prompt_ids = []
            for sample in samples:
                source = sample.agentic_prompt_source
                if source is None:
                    source = {
                        "dataset_id": None,
                        "dataset_version": None,
                        "split": self.rollout_mode,
                        "source_row_id": None,
                        "input_payload": {"prompt": sample.prompt, "metadata": sample.metadata},
                        "input_payload_ref": None,
                        "identity_kind": "content_fallback",
                    }
                prompt_id = source.get("prompt_id") or prompt_identity(source)
                prompt_ids.append(prompt_id)
                if prompt_id in self._prompts:
                    self._prompts.move_to_end(prompt_id)
                    continue
                media: dict[str, str] = {}
                payload = strip_media([source["input_payload"]], media_sink=media if self.config.dump_media else None)[
                    0
                ]
                self._writers["prompts"].submit(
                    PromptRecord(
                        run_id=self.config.run_id,
                        prompt_id=prompt_id,
                        dataset_id=source.get("dataset_id"),
                        dataset_version=source.get("dataset_version"),
                        split=source.get("split", self.rollout_mode),
                        source_row_id=source.get("source_row_id"),
                        input_payload=payload,
                        input_payload_ref=source.get("input_payload_ref"),
                        prompt_preview=_prompt_preview(sample.prompt),
                        identity_kind=source.get("identity_kind", "source_row"),
                    ),
                    media=media,
                )
                self._prompts[prompt_id] = None
                if len(self._prompts) > 100_000:
                    self._prompts.popitem(last=False)
            if group_id not in self._groups:
                self._writers["groups"].submit(
                    GroupRecord(
                        self.config.run_id,
                        group_id,
                        samples[0].group_index,
                        prompt_ids[0],
                        len(samples),
                        self.rollout_mode,
                    )
                )
                self._groups[group_id] = None
                if len(self._groups) > 100_000:
                    self._groups.popitem(last=False)
            if session_ids is None:
                return
            for position, (sample, session_id, prompt_id) in enumerate(
                zip(samples, session_ids, prompt_ids, strict=True)
            ):
                created_at = started_at or utc_now()
                self._writers["sessions"].submit(
                    SessionRecord(
                        self.config.run_id,
                        session_id,
                        group_id,
                        sample.group_index,
                        sample.index,
                        prompt_id,
                        position,
                        len(samples),
                        copy.deepcopy(
                            session_sampling_config(self.args, getattr(sample, "sampling_params", None), position)
                        ),
                        created_at=created_at,
                    )
                )
                self._writers["stages"].submit(
                    ExportStageRecord(
                        None,
                        "rollout",
                        "start",
                        group_id,
                        scope="session",
                        session_id=session_id,
                        run_id=self.config.run_id,
                        rollout_mode=self.rollout_mode,
                        stage_id=f"session:{session_id}:rollout",
                        occurred_at=created_at,
                    )
                )
        except Exception as error:
            logger.warning("agentic_obs: inventory failed for %s: %s", group_id, error)

    @best_effort
    def preparation_failed(
        self,
        group_id: str,
        session_ids: Sequence[str],
        error: BaseException,
        *,
        uncertain: bool = False,
        started_at: float | None = None,
    ) -> None:
        if self.config is None:
            return
        if uncertain:
            logger.warning("agentic_obs: registration outcome unknown for %s: %s", group_id, error)
            return
        for session_id in session_ids:
            self._writers["stages"].submit(
                ExportStageRecord(
                    None,
                    "rollout",
                    "cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                    group_id,
                    scope="session",
                    session_id=session_id,
                    run_id=self.config.run_id,
                    rollout_mode=self.rollout_mode,
                    is_terminal=True,
                    event_seq=2,
                    stage_id=f"session:{session_id}:rollout",
                    reason_code="prepare_cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "session_prepare_failed",
                    category="selection" if isinstance(error, asyncio.CancelledError) else "execution",
                    reason=f"{type(error).__name__}: {error}",
                    wall_ms=(time.monotonic() - started_at) * 1000 if started_at is not None else None,
                    request_count=0,
                    attempt_count=0,
                    generated_tokens=0,
                    token_count_complete=True,
                    export_ids=[],
                    active_ms=0,
                    llm_calls=[],
                    llm_calls_complete=True,
                )
            )

    async def aclose(self) -> None:
        deadline = time.monotonic() + CLOSE_TIMEOUT
        for writer in self._writers.values():
            await writer.aclose(timeout=max(0.0, deadline - time.monotonic()))
        await close_background(timeout=max(0.0, deadline - time.monotonic()))
        self._prompts.clear()
        self._groups.clear()
