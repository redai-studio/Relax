# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch

from relax.agentic.profile import TRACE_KEY, merge_agentic_trace
from relax.utils.types import Sample


MsgKind = Literal["obs", "resp"]
# Roles accepted into the canonical Forest. Expanding this set requires the
# compiler, OpenAI ingress, and export validator to agree on the new role.
_ALLOWED_MESSAGE_ROLES = {"user", "assistant", "tool", "system"}


class RequestKind(str, Enum):
    FRESH = "fresh"
    RESUMED = "resumed"
    PROTECTED = "protected"


# Zero-valued SGLang accounting shape merged across interrupted IR attempts.
# Keys are part of exported metadata and therefore must track backend schema.
_EMPTY_SPEC_DELTA = {
    "spec_accept_token_num": 0,
    "spec_draft_token_num": 0,
    "spec_verify_ct": 0,
    "completion_token_num": 0,
}
# Zero-valued prefix-cache accounting shape accumulated across resumptions.
_EMPTY_PREFIX_CACHE_DELTA = {
    "cached_tokens": 0,
    "total_prompt_tokens": 0,
}


def _canonical_json_value(value: Any, *, field: str) -> Any:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field} JSON object keys must be strings")
        return {key: _canonical_json_value(item, field=field) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_json_value(item, field=field) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} numbers must be finite")
        # JavaScript JSON.stringify drops the decimal part of safe integral numbers.
        # Match that representation so cross-language replays keep one state identity.
        if value.is_integer() and abs(value) <= 2**53 - 1:
            return int(value)
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"{field} must contain JSON-compatible values, got {type(value)}")


def _canonical_tool_arguments(arguments: Any, *, field: str) -> str:
    if isinstance(arguments, str):

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            parsed: dict[str, Any] = {}
            for key, value in pairs:
                if key in parsed:
                    raise ValueError(f"{field} contains duplicate key: {key}")
                parsed[key] = value
            return parsed

        def reject_non_finite(value: str) -> None:
            raise ValueError(f"{field} contains non-finite number: {value}")

        try:
            arguments = json.loads(
                arguments,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_non_finite,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be valid JSON") from exc
    if not isinstance(arguments, dict):
        raise TypeError(f"{field} must be a JSON object or a JSON string encoding an object, got {type(arguments)}")
    canonical_arguments = _canonical_json_value(arguments, field=field)
    canonical = json.dumps(
        canonical_arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    try:
        canonical.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be valid UTF-8") from exc
    return canonical


def _normalize_tool_calls(message: dict[str, Any], *, message_index: int) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, list):
        raise TypeError(f"messages[{message_index}].tool_calls must be a list, got {type(tool_calls)}")
    normalized = []
    for call_index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            raise TypeError(
                f"messages[{message_index}].tool_calls[{call_index}] must be a dict, got {type(tool_call)}"
            )
        call_id = tool_call.get("id")
        if call_id is not None and (not isinstance(call_id, str) or not call_id):
            raise ValueError(f"messages[{message_index}].tool_calls[{call_index}].id must be a non-empty string")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise TypeError(
                f"messages[{message_index}].tool_calls[{call_index}].function must be a dict, got {type(function)}"
            )
        function_name = function.get("name")
        if function_name is not None and not isinstance(function_name, str):
            raise TypeError(
                f"messages[{message_index}].tool_calls[{call_index}].function.name must be a string, "
                f"got {type(function_name)}"
            )
        arguments_field = f"messages[{message_index}].tool_calls[{call_index}].function.arguments"
        arguments = function.get("arguments")
        normalized_tool_call = copy.deepcopy(tool_call)
        normalized_tool_call["function"]["arguments"] = _canonical_tool_arguments(arguments, field=arguments_field)
        normalized.append(normalized_tool_call)
    return normalized


def check_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages)}")

    system_chunks: list[str] = []
    ensured: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{index}] must be a dict, got {type(message)}")
        if "role" not in message:
            raise ValueError(f"messages[{index}] must include role")
        role = message["role"]
        if not isinstance(role, str):
            raise TypeError(f"messages[{index}].role must be a string, got {type(role)}")
        if role not in _ALLOWED_MESSAGE_ROLES:
            allowed_roles = ", ".join(sorted(_ALLOWED_MESSAGE_ROLES))
            raise ValueError(f"messages[{index}].role must be one of: {allowed_roles}")
        tool_calls = _normalize_tool_calls(message, message_index=index) if role == "assistant" else []
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise TypeError(f"messages[{index}].reasoning_content must be a string, got {type(reasoning_content)}")
        has_reasoning_content = isinstance(reasoning_content, str) and bool(reasoning_content)
        assistant_allows_empty_content = role == "assistant" and (tool_calls or has_reasoning_content)
        if role == "tool" and "tool_call_id" in message:
            tool_call_id = message["tool_call_id"]
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError(f"messages[{index}].tool_call_id must be a non-empty string")
        else:
            tool_call_id = None
        if "content" not in message and not assistant_allows_empty_content:
            raise ValueError(f"messages[{index}] must include content")
        content = message.get("content")
        if content is None:
            if not assistant_allows_empty_content:
                raise ValueError(f"messages[{index}].content must not be empty")
            content = ""
        elif isinstance(content, str):
            if not content and role != "tool" and not assistant_allows_empty_content:
                raise ValueError(f"messages[{index}].content must not be empty")
        elif isinstance(content, list):
            if not content:
                raise ValueError(f"messages[{index}].content must not be empty")
            for item_index, item in enumerate(content):
                if not isinstance(item, dict):
                    raise TypeError(f"messages[{index}].content[{item_index}] must be a dict, got {type(item)}")
                if item.get("type") == "text" and isinstance(item.get("text"), str) and not item["text"]:
                    raise ValueError(f"messages[{index}].content[{item_index}].text must not be empty")
        else:
            raise TypeError(f"messages[{index}].content must be a list, string, or None, got {type(content)}")
        if role == "system":
            if isinstance(content, str):
                system_chunks.append(content)
            else:
                system_chunks.extend(part["text"] for part in content if isinstance(part.get("text"), str))
            continue
        rendered_message = {"role": role, "content": copy.deepcopy(content)}
        if has_reasoning_content:
            rendered_message["reasoning_content"] = reasoning_content
        if tool_calls:
            rendered_message["tool_calls"] = tool_calls
        if role == "tool" and tool_call_id is not None:
            rendered_message["tool_call_id"] = tool_call_id
        ensured.append(rendered_message)
    if system_chunks:
        return [{"role": "system", "content": "\n\n".join(system_chunks)}, *ensured]
    return ensured


def normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if tools is None:
        return normalized
    if not isinstance(tools, list):
        raise TypeError(f"tools must be a list, got {type(tools)}")
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise TypeError(f"tools[{index}] must be a dict, got {type(tool)}")
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            continue
        normalized_function = {
            "name": function.get("name"),
            "parameters": function.get("parameters"),
        }
        if function.get("description") is not None:
            normalized_function["description"] = function["description"]
        normalized.append({"type": "function", "function": normalized_function})
    return normalized


def normalize_template_kwargs(template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if template_kwargs is None:
        return {}
    if not isinstance(template_kwargs, dict):
        raise TypeError(f"chat_template_kwargs must be a dict, got {type(template_kwargs)}")

    def _normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    return _normalize(template_kwargs)


def _messages_tools_template_state_hash(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "messages": messages,
            "tools": tools,
            "template_kwargs": template_kwargs,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merge_multimodal_train_inputs(deltas: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not deltas:
        return None
    values_by_key: dict[str, list[Any]] = {}
    for delta in deltas:
        for key, value in delta.items():
            if value is not None:
                values_by_key.setdefault(key, []).append(value)
    if not values_by_key:
        return None
    merged: dict[str, Any] = {}
    for key, values in values_by_key.items():
        if all(isinstance(value, torch.Tensor) for value in values):
            merged[key] = torch.cat(values, dim=0) if len(values) > 1 else copy.deepcopy(values[0])
        else:
            merged[key] = copy.deepcopy(values[-1])
    return merged


def _extend_media_data(base: list[str], delta: list[str] | None) -> list[str]:
    if delta:
        base.extend(delta)
    return base


def _merge_export_metadata(target: dict[str, Any], overlay: dict[str, Any]) -> None:
    overlay = copy.deepcopy(overlay)
    overlay_trace = overlay.pop(TRACE_KEY, None)
    if overlay_trace is not None:
        target[TRACE_KEY] = merge_agentic_trace(target.get(TRACE_KEY), overlay_trace)
    target.update(overlay)


def _sum_counter_dict(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    summed = dict(left)
    for key, value in right.items():
        summed[key] = int(summed.get(key, 0)) + int(value or 0)
    return summed


def _decode_tokens(
    *,
    tokenizer: Any,
    token_ids: list[int],
) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _multimodal_inputs_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    multimodal_inputs = {
        "images": [],
        # "videos": [],
        # "audio": [],
    }
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            item_type = item.get("type")
            if item_type == "image_url" and item.get("image_url") is not None:
                multimodal_inputs["images"].append(copy.deepcopy(item["image_url"]["url"]))
            # elif item_type == "video" and item.get("video") is not None:
            #     multimodal_inputs["videos"].append(copy.deepcopy(item.get("video")))
            # elif item_type == "audio" and item.get("audio") is not None:
            #     multimodal_inputs["audio"].append(copy.deepcopy(item.get("audio")))
    compact = {key: value for key, value in multimodal_inputs.items() if value}
    return compact or None


@dataclass
class MsgNode:
    kind: MsgKind
    state_hash: str
    parent_state_hash: str | None
    rollout_id: int
    abort_count: int
    messages_delta: list[dict[str, Any]] = field(default_factory=list)
    train_token_delta: list[int] = field(default_factory=list)
    rollout_token_delta: list[int] = field(default_factory=list)
    logprob_delta: list[float] = field(default_factory=list)
    multimodal_train_inputs_delta: dict[str, Any] | None = None
    backend_image_data_delta: list[str] = field(default_factory=list)
    backend_audio_data_delta: list[str] = field(default_factory=list)
    backend_video_data_delta: list[str] = field(default_factory=list)
    weight_version_delta: list[str] = field(default_factory=list)
    spec_delta: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_SPEC_DELTA))
    prefix_cache_delta: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_PREFIX_CACHE_DELTA))
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    wall_elapsed_s: float = 0.0
    generation_elapsed_s: float = 0.0
    status: str | None = None
    rollout_routed_experts: Any = None
    export_metadata_patch: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class InflightRequest:
    request_id: str
    parent_state_hash: str
    rollout_id: int
    kind: RequestKind
    abort_count: int
    waiter: asyncio.Future[dict[str, Any]] = field(repr=False)
    wall_started_at: float
    runner_task: asyncio.Task[None] | None = field(default=None, repr=False)
    backend_started: bool = False
    sampling_params: dict[str, Any] = field(default_factory=dict)
    logprobs: bool = False
    history_train_token_prefix: list[int] = field(default_factory=list)
    history_rollout_token_prefix: list[int] = field(default_factory=list)
    history_backend_image_data: list[str] = field(default_factory=list)
    history_backend_audio_data: list[str] = field(default_factory=list)
    history_backend_video_data: list[str] = field(default_factory=list)
    pending_token_delta: list[int] = field(default_factory=list)
    pending_logprob_delta: list[float] = field(default_factory=list)
    pending_weight_version_delta: list[str] = field(default_factory=list)
    pending_spec_delta: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_SPEC_DELTA))
    pending_prefix_cache_delta: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_PREFIX_CACHE_DELTA))
    pending_generation_elapsed_s: float = 0.0
    pending_status: str | None = None
    pending_routed_experts: Any = None
    pending_export_metadata_patch: dict[str, Any] = field(default_factory=dict)
    latest_backend_meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPrefix:
    train_token_prefix: list[int] = field(default_factory=list)
    rollout_token_prefix: list[int] = field(default_factory=list)
    backend_image_data: list[str] = field(default_factory=list)
    backend_audio_data: list[str] = field(default_factory=list)
    backend_video_data: list[str] = field(default_factory=list)


@dataclass
class SessionForest:
    session_id: str
    group_index: int | None
    index: int | None
    label: str | None
    train_metadata: dict[str, Any] | None
    static_metadata: dict[str, Any] = field(default_factory=dict)
    root_state_hash: str | None = None
    leaf_state_hashes: set[str] = field(default_factory=set)
    nodes_by_hash: dict[str, MsgNode] = field(default_factory=dict)

    @classmethod
    def create_empty(
        cls,
        *,
        session_id: str,
        group_index: int | None = None,
        index: int | None = None,
        label: str | None = None,
        train_metadata: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SessionForest":
        static_metadata = copy.deepcopy(metadata or {})
        forest = cls(
            session_id=session_id,
            group_index=group_index,
            index=index,
            label=label,
            train_metadata=copy.deepcopy(train_metadata),
            static_metadata=static_metadata,
        )
        forest.append_obs(
            parent_state_hash=None,
            rollout_id=0,
            abort_count=0,
            messages_delta=[],
            train_token_delta=[],
            rollout_token_delta=[],
            multimodal_train_inputs_delta=None,
            backend_image_data_delta=[],
            backend_audio_data_delta=[],
            backend_video_data_delta=[],
            tools=None,
            chat_template_kwargs=normalize_template_kwargs(static_metadata.get("chat_template_kwargs")),
        )
        return forest

    @staticmethod
    def resolve_request_kind(
        *,
        abort_count: int = 0,
        resumed: bool = False,
        protected_abort_count_threshold: int | None = None,
    ) -> RequestKind:
        if protected_abort_count_threshold is not None and abort_count >= protected_abort_count_threshold:
            return RequestKind.PROTECTED
        if resumed or abort_count > 0:
            return RequestKind.RESUMED
        return RequestKind.FRESH

    def lineage(self, state_hash: str) -> list[MsgNode]:
        lineage: list[MsgNode] = []
        current_hash: str | None = state_hash
        while current_hash is not None:
            node = self.nodes_by_hash[current_hash]
            lineage.append(node)
            current_hash = node.parent_state_hash
        lineage.reverse()
        return lineage

    def full_messages(self, state_hash: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for node in self.lineage(state_hash):
            if node.messages_delta:
                messages.extend(copy.deepcopy(node.messages_delta))
        return messages

    def train_token_count(self, state_hash: str) -> int:
        total = 0
        for node in self.lineage(state_hash):
            total += len(node.train_token_delta)
        return total

    def build_execution_prefix(self, state_hash: str) -> ExecutionPrefix:
        train_token_prefix: list[int] = []
        rollout_token_prefix: list[int] = []
        backend_image_data: list[str] = []
        backend_audio_data: list[str] = []
        backend_video_data: list[str] = []
        for node in self.lineage(state_hash):
            train_token_prefix.extend(node.train_token_delta)
            rollout_token_prefix.extend(node.rollout_token_delta)
            _extend_media_data(backend_image_data, node.backend_image_data_delta)
            _extend_media_data(backend_audio_data, node.backend_audio_data_delta)
            _extend_media_data(backend_video_data, node.backend_video_data_delta)
        return ExecutionPrefix(
            train_token_prefix=train_token_prefix,
            rollout_token_prefix=rollout_token_prefix,
            backend_image_data=backend_image_data,
            backend_audio_data=backend_audio_data,
            backend_video_data=backend_video_data,
        )

    def export_leaf_hashes(self) -> list[str]:
        return list(self.leaf_state_hashes)

    def subtree_root_node(self, state_hash: str | None) -> MsgNode | None:
        if state_hash is None:
            return None
        lineage = self.lineage(state_hash)
        if len(lineage) <= 1:
            return None
        return lineage[1]

    def subtree_tools(self, state_hash: str | None) -> list[dict[str, Any]]:
        subtree_root = self.subtree_root_node(state_hash)
        if subtree_root is None:
            return []
        return normalize_tools(subtree_root.tools)

    def subtree_chat_template_kwargs(
        self,
        state_hash: str | None,
        base_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_kwargs = normalize_template_kwargs(base_kwargs)
        subtree_root = self.subtree_root_node(state_hash)
        if subtree_root is None:
            return base_kwargs
        return normalize_template_kwargs(subtree_root.chat_template_kwargs)

    def _register_node(self, node: MsgNode) -> MsgNode:
        existing = self.nodes_by_hash.get(node.state_hash)
        if existing is not None:
            if existing.kind != node.kind:
                raise ValueError(f"State hash collision with mismatched kinds: {existing.kind} vs {node.kind}")
            return existing

        if node.parent_state_hash is None:
            if self.root_state_hash is not None:
                raise ValueError("SessionForest root already exists")
            self.root_state_hash = node.state_hash
        self.nodes_by_hash[node.state_hash] = node
        if node.parent_state_hash is not None:
            self.leaf_state_hashes.add(node.state_hash)
            self.leaf_state_hashes.discard(node.parent_state_hash)
        return node

    def _next_state_hash(
        self,
        *,
        parent_state_hash: str | None,
        messages_delta: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
    ) -> str:
        parent_messages = self.full_messages(parent_state_hash) if parent_state_hash is not None else []
        full_messages = parent_messages + messages_delta
        current_tools = tools if tools is not None else self.subtree_tools(parent_state_hash)
        current_chat_template_kwargs = (
            chat_template_kwargs
            if chat_template_kwargs is not None
            else self.subtree_chat_template_kwargs(parent_state_hash, self.static_metadata.get("chat_template_kwargs"))
        )
        return _messages_tools_template_state_hash(full_messages, current_tools, current_chat_template_kwargs)

    def append_obs(
        self,
        *,
        parent_state_hash: str | None,
        rollout_id: int,
        abort_count: int,
        messages_delta: list[dict[str, Any]],
        train_token_delta: list[int],
        rollout_token_delta: list[int],
        multimodal_train_inputs_delta: dict[str, Any] | None = None,
        backend_image_data_delta: list[str] | None = None,
        backend_audio_data_delta: list[str] | None = None,
        backend_video_data_delta: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        wall_elapsed_s: float = 0.0,
        generation_elapsed_s: float = 0.0,
    ) -> MsgNode:
        state_hash = self._next_state_hash(
            parent_state_hash=parent_state_hash,
            messages_delta=messages_delta,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        return self._register_node(
            MsgNode(
                kind="obs",
                state_hash=state_hash,
                parent_state_hash=parent_state_hash,
                rollout_id=rollout_id,
                abort_count=abort_count,
                messages_delta=messages_delta,
                train_token_delta=train_token_delta,
                rollout_token_delta=rollout_token_delta,
                multimodal_train_inputs_delta=multimodal_train_inputs_delta,
                backend_image_data_delta=backend_image_data_delta if backend_image_data_delta is not None else [],
                backend_audio_data_delta=backend_audio_data_delta if backend_audio_data_delta is not None else [],
                backend_video_data_delta=backend_video_data_delta if backend_video_data_delta is not None else [],
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
                wall_elapsed_s=wall_elapsed_s,
                generation_elapsed_s=generation_elapsed_s,
            )
        )

    def append_resp(
        self,
        *,
        parent_state_hash: str,
        rollout_id: int,
        abort_count: int,
        messages_delta: list[dict[str, Any]],
        token_delta: list[int],
        logprob_delta: list[float],
        weight_version_delta: list[str] | None = None,
        spec_delta: dict[str, int] | None = None,
        prefix_cache_delta: dict[str, int] | None = None,
        wall_elapsed_s: float = 0.0,
        generation_elapsed_s: float = 0.0,
        status: str = "completed",
        rollout_routed_experts: Any = None,
        export_metadata_patch: dict[str, Any] | None = None,
    ) -> MsgNode:
        state_hash = self._next_state_hash(
            parent_state_hash=parent_state_hash,
            messages_delta=messages_delta,
            tools=None,
            chat_template_kwargs=None,
        )
        return self._register_node(
            MsgNode(
                kind="resp",
                state_hash=state_hash,
                parent_state_hash=parent_state_hash,
                rollout_id=rollout_id,
                abort_count=abort_count,
                messages_delta=messages_delta,
                train_token_delta=token_delta,
                rollout_token_delta=token_delta,
                logprob_delta=logprob_delta,
                weight_version_delta=weight_version_delta if weight_version_delta is not None else [],
                spec_delta=spec_delta if spec_delta is not None else dict(_EMPTY_SPEC_DELTA),
                prefix_cache_delta=prefix_cache_delta
                if prefix_cache_delta is not None
                else dict(_EMPTY_PREFIX_CACHE_DELTA),
                wall_elapsed_s=wall_elapsed_s,
                generation_elapsed_s=generation_elapsed_s,
                status=status,
                rollout_routed_experts=rollout_routed_experts,
                export_metadata_patch=export_metadata_patch if export_metadata_patch is not None else {},
            )
        )

    @staticmethod
    def _agentic_trace_turn_from_node(node: MsgNode, turn_idx: int) -> dict[str, Any]:
        patch = copy.deepcopy(node.export_metadata_patch)
        patch_trace = patch.get(TRACE_KEY)
        events = merge_agentic_trace(None, patch_trace).get("events") if patch_trace is not None else None
        return {
            "turn_idx": turn_idx,
            "resp_state_hash": node.state_hash,
            "parent_state_hash": node.parent_state_hash,
            "rollout_id": node.rollout_id,
            "request_id": patch.get("request_id"),
            "request_kind": patch.get("request_kind"),
            "admission": patch.get("admission"),
            "base_state_hash": patch.get("base_state_hash"),
            "abort_count": node.abort_count,
            "status": str(node.status),
            "response_length": len(node.train_token_delta),
            "wall_elapsed_s": float(node.wall_elapsed_s),
            "generation_elapsed_s": float(node.generation_elapsed_s),
            "events": copy.deepcopy(events) if isinstance(events, dict) else {},
        }

    def build_sample(
        self,
        *,
        leaf_state_hash: str,
        tokenizer: Any,
    ) -> Sample:
        lineage = self.lineage(leaf_state_hash)
        leaf = lineage[-1]

        tokens: list[int] = []
        rollout_tokens: list[int] = []
        continuation_train_tokens: list[int] = []
        loss_mask: list[int] = []
        rollout_log_probs: list[float] = []
        messages: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        response_node_spans: list[list[int]] = []
        multimodal_train_inputs_buffer: list[dict[str, Any]] = []
        weight_versions: list[str] = []
        spec_info = dict(_EMPTY_SPEC_DELTA)
        prefix_cache_info = dict(_EMPTY_PREFIX_CACHE_DELTA)
        wall_elapsed_s = 0.0
        generation_elapsed_s = 0.0
        first_response_node: MsgNode | None = None

        for idx, node in enumerate(lineage):
            tokens.extend(node.train_token_delta)
            rollout_tokens.extend(node.rollout_token_delta)
            messages.extend(node.messages_delta)
            wall_elapsed_s += float(node.wall_elapsed_s)
            generation_elapsed_s += float(node.generation_elapsed_s)
            if node.multimodal_train_inputs_delta is not None:
                multimodal_train_inputs_buffer.append(node.multimodal_train_inputs_delta)
            if node.kind == "resp":
                if first_response_node is None:
                    first_response_node = node
                turns.append(self._agentic_trace_turn_from_node(node, len(turns)))
                response_start = len(continuation_train_tokens)
                continuation_train_tokens.extend(node.train_token_delta)
                response_node_spans.append([response_start, len(continuation_train_tokens)])
                loss_mask.extend([1] * len(node.train_token_delta))
                rollout_log_probs.extend(node.logprob_delta)
                weight_versions.extend(node.weight_version_delta)
                spec_info = _sum_counter_dict(spec_info, node.spec_delta)
                prefix_cache_info = _sum_counter_dict(prefix_cache_info, node.prefix_cache_delta)
                continue
            if idx == 0 or first_response_node is None:
                continue
            continuation_train_tokens.extend(node.train_token_delta)
            loss_mask.extend([0] * len(node.train_token_delta))
            rollout_log_probs.extend(node.logprob_delta or ([0.0] * len(node.train_token_delta)))

        if first_response_node is None:
            raise ValueError("A sample export leaf must have at least one resp node in its lineage.")
        prompt_train_token_count = len(tokens) - len(continuation_train_tokens)
        effective_prompt = _decode_tokens(tokenizer=tokenizer, token_ids=tokens[:prompt_train_token_count])
        merged_metadata = copy.deepcopy(self.static_metadata)
        _merge_export_metadata(merged_metadata, leaf.export_metadata_patch)
        if "start_rollout_id" not in merged_metadata:
            merged_metadata["start_rollout_id"] = first_response_node.rollout_id
        subtree_root = lineage[1] if len(lineage) > 1 else None
        merged_metadata["tools"] = normalize_tools(subtree_root.tools) if subtree_root is not None else []
        merged_metadata["chat_template_kwargs"] = (
            normalize_template_kwargs(subtree_root.chat_template_kwargs) if subtree_root is not None else {}
        )
        status = Sample.Status.TRUNCATED if leaf.kind == "obs" else Sample.Status(leaf.status)
        trace = merge_agentic_trace(merged_metadata.get(TRACE_KEY), None)
        trace.update(
            {
                "session_id": self.session_id,
                "leaf_state_hash": leaf_state_hash,
                "terminal_status": status.value,
                "turn_count": len(turns),
                "turns": turns,
            }
        )
        merged_metadata[TRACE_KEY] = trace
        # Mirror the turn count to ``rollout_turns`` so consumers that predate
        # the agentic trace (train_dump_utils, the non-agentic rollout metric
        # path) see a single source of truth without having to know about
        # ``agentic_trace``.
        merged_metadata["rollout_turns"] = len(turns)
        sample = Sample(
            group_index=self.group_index,
            index=self.index,
            prompt=effective_prompt,
            tokens=tokens,
            rollout_tokens=rollout_tokens,
            multimodal_inputs=_multimodal_inputs_from_messages(messages),
            multimodal_train_inputs=_merge_multimodal_train_inputs(multimodal_train_inputs_buffer),
            response=_decode_tokens(tokenizer=tokenizer, token_ids=continuation_train_tokens),
            response_length=len(continuation_train_tokens),
            label=self.label,
            custom_advantage=response_node_spans,
            loss_mask=loss_mask,
            weight_versions=weight_versions,
            rollout_log_probs=rollout_log_probs,
            rollout_routed_experts=copy.deepcopy(leaf.rollout_routed_experts),
            abort_count=leaf.abort_count,
            status=status,
            metadata=merged_metadata,
            train_metadata=copy.deepcopy(self.train_metadata),
            session_id=self.session_id,
            non_generation_time=wall_elapsed_s - generation_elapsed_s,
        )
        sample.spec_info = Sample.SpecInfo.from_dict(spec_info)
        sample.prefix_cache_info = Sample.PrefixCacheInfo.from_dict(prefix_cache_info)
        return sample
