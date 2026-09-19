# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest


_ray_stub_names: list[str] = []
try:
    import ray as _ray  # noqa: F401
except ModuleNotFoundError:
    ray_stub = ModuleType("ray")
    serve_stub = ModuleType("ray.serve")
    torch_stub = ModuleType("torch")

    def _identity_decorator(*args: Any, **kwargs: Any) -> Any:
        del kwargs
        if len(args) == 1 and isinstance(args[0], type):
            return args[0]
        return lambda decorated: decorated

    ray_stub.remote = _identity_decorator
    ray_stub.method = _identity_decorator
    ray_stub.ObjectRef = type("ObjectRef", (), {})
    ray_stub.exceptions = SimpleNamespace(RayTaskError=RuntimeError, TaskCancelledError=RuntimeError)
    ray_stub.serve = serve_stub
    serve_stub.deployment = _identity_decorator
    serve_stub.ingress = _identity_decorator
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.dtype = type("dtype", (), {})
    torch_stub.Size = tuple
    torch_stub.cat = lambda values, dim=0: values

    pipeline_stub = ModuleType("relax.agentic.pipeline")
    runtime_stub = ModuleType("relax.agentic.pipeline.runtime")
    runner_stub = ModuleType("relax.agentic.runner")
    coordinator_stub = ModuleType("relax.agentic.session.admission_coordinator")
    types_stub = ModuleType("relax.utils.types")
    placeholder = type("_ImportPlaceholder", (), {})
    for name in (
        "RuntimeGroupError",
        "SessionExportTransport",
        "SessionGroupProgress",
        "SessionShardProgress",
        "SessionSpec",
        "TrainingFieldArtifact",
    ):
        setattr(pipeline_stub, name, placeholder)
    runtime_stub.BackendContextLengthExceededError = type("BackendContextLengthExceededError", (Exception,), {})
    runtime_stub.SGLangBackendAdapter = placeholder
    runtime_stub.finish_before_cancellation = lambda function: function
    for name in (
        "AgentExecutionError",
        "ManagedAgentLauncher",
        "ManagedAgentProcess",
        "SessionInput",
    ):
        setattr(runner_stub, name, placeholder)
    runner_stub.load_agent_app_spec_from_args = lambda _args: None
    coordinator_stub.AdmissionCoordinator = placeholder
    coordinator_stub.RayAdmissionClient = placeholder
    types_stub.Sample = placeholder
    types_stub.get_spec_token_counts = lambda _metadata: (0, 0)

    for module_name, module in (
        ("ray", ray_stub),
        ("ray.serve", serve_stub),
        ("torch", torch_stub),
        ("relax.agentic.pipeline", pipeline_stub),
        ("relax.agentic.pipeline.runtime", runtime_stub),
        ("relax.agentic.runner", runner_stub),
        ("relax.agentic.session.admission_coordinator", coordinator_stub),
        ("relax.utils.types", types_stub),
    ):
        sys.modules[module_name] = module
        _ray_stub_names.append(module_name)

_web_stub_names: list[str] = []
try:
    import fastapi as _fastapi  # noqa: F401
except ModuleNotFoundError:
    fastapi_stub = ModuleType("fastapi")
    responses_stub = ModuleType("fastapi.responses")
    starlette_stub = ModuleType("starlette")
    requests_stub = ModuleType("starlette.requests")

    class _FastAPI:
        def get(self, *_args: Any, **_kwargs: Any) -> Any:
            return lambda decorated: decorated

        post = get

    class _HTTPException(Exception):
        def __init__(self, *, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _Response:
        pass

    class _ClientDisconnect(Exception):
        pass

    fastapi_stub.FastAPI = _FastAPI
    fastapi_stub.HTTPException = _HTTPException
    fastapi_stub.Request = type("Request", (), {})
    responses_stub.JSONResponse = _Response
    responses_stub.Response = _Response
    responses_stub.StreamingResponse = _Response
    requests_stub.ClientDisconnect = _ClientDisconnect
    for module_name, module in (
        ("fastapi", fastapi_stub),
        ("fastapi.responses", responses_stub),
        ("starlette", starlette_stub),
        ("starlette.requests", requests_stub),
    ):
        sys.modules[module_name] = module
        _web_stub_names.append(module_name)

try:
    from relax.agentic.session.service import (
        AgenticChatRequestError,
        _normalized_anthropic_request,
        _normalized_chat_request,
        _normalized_responses_request,
    )
    from relax.agentic.session.state import (
        _messages_tools_template_state_hash,
        normalize_template_kwargs,
        normalize_tools,
    )
finally:
    for module_name in (*_ray_stub_names, *_web_stub_names):
        sys.modules.pop(module_name, None)
    if _ray_stub_names:
        service_module = sys.modules.pop("relax.agentic.session.service", None)
        state_module = sys.modules.pop("relax.agentic.session.state", None)
        session_package = sys.modules.get("relax.agentic.session")
        if session_package is not None and getattr(session_package, "service", None) is service_module:
            delattr(session_package, "service")
        if session_package is not None and getattr(session_package, "state", None) is state_module:
            delattr(session_package, "state")
        agentic_package = sys.modules.get("relax.agentic")
        if agentic_package is not None:
            for attribute, stub in (("pipeline", pipeline_stub), ("runner", runner_stub)):
                if getattr(agentic_package, attribute, None) is stub:
                    delattr(agentic_package, attribute)
        utils_package = sys.modules.get("relax.utils")
        if utils_package is not None and getattr(utils_package, "types", None) is types_stub:
            delattr(utils_package, "types")


FIXTURE_DIR = Path(__file__).with_name("fixtures")
CANONICAL_FIELDS = ("messages", "tools", "chat_template_kwargs")
CASE_NAMES = ("text", "text_blocks", "tool_round_trip", "images")
EXPECTED_HASHES = {
    "text": "a00bb147c18e3f4adb70a741b72796b5badc929c0593b5066d42ff06c21bcd16",
    "text_blocks": "ec281e76b19fd0ec4ce8c0b295977c5d477bc3de77402d5d02169c99767e00eb",
    "tool_round_trip": "bb620c5313c10d474bc1c9611dd26a622534df79b931c70bf0c42b3ff2499bd2",
    "images": "391f1785a3bc84574ae4ab5b8b241e3e4d06414a888bb636ba7be896011bccd3",
}

Normalizer = Callable[[dict[str, Any]], dict[str, Any]]
PROTOCOLS: dict[str, tuple[str, Normalizer]] = {
    "chat_completions": ("chat_completions.json", _normalized_chat_request),
    "responses": ("responses.json", _normalized_responses_request),
    "anthropic_messages": ("anthropic_messages.json", _normalized_anthropic_request),
}


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _canonical_fields(normalized: dict[str, Any]) -> dict[str, Any]:
    return {field: normalized[field] for field in CANONICAL_FIELDS}


def _normalize(protocol: str, request: dict[str, Any]) -> dict[str, Any]:
    return _canonical_fields(PROTOCOLS[protocol][1](request))


def _assert_protocol_fixture(protocol: str) -> None:
    fixture_name, normalizer = PROTOCOLS[protocol]
    requests = _load_fixture(fixture_name)
    expected = _load_fixture("canonical.json")
    assert tuple(requests) == CASE_NAMES
    for case_name in CASE_NAMES:
        request = requests[case_name]
        snapshot = copy.deepcopy(request)
        normalized = _canonical_fields(normalizer(request))
        assert normalized == expected[case_name]
        assert request == snapshot


def _reverse_object_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_object_keys(item) for key, item in reversed(tuple(value.items()))}
    if isinstance(value, list):
        return [_reverse_object_keys(item) for item in value]
    return value


def _state_hash(canonical: dict[str, Any]) -> str:
    return _messages_tools_template_state_hash(
        canonical["messages"],
        canonical["tools"],
        canonical["chat_template_kwargs"],
    )


def _assert_request_error(
    normalizer: Normalizer,
    payload: dict[str, Any],
    *,
    path: str,
    param: str,
) -> str:
    with pytest.raises(AgenticChatRequestError) as caught:
        normalizer(payload)
    assert caught.value.status_code == 400
    assert caught.value.param == param
    assert path in caught.value.message
    assert "0x" not in caught.value.message
    return caught.value.message


def test_chat_completions_golden_fixture() -> None:
    _assert_protocol_fixture("chat_completions")


def test_responses_golden_fixture() -> None:
    _assert_protocol_fixture("responses")


def test_anthropic_messages_golden_fixture() -> None:
    _assert_protocol_fixture("anthropic_messages")


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_semantically_equivalent_protocols_share_canonical_state(case_name: str) -> None:
    normalized = {
        protocol: _normalize(protocol, _load_fixture(fixture_name)[case_name])
        for protocol, (fixture_name, _normalizer) in PROTOCOLS.items()
    }
    expected = _load_fixture("canonical.json")[case_name]
    assert all(value == expected for value in normalized.values())
    assert len({_state_hash(value) for value in normalized.values()}) == 1
    assert _state_hash(expected) == EXPECTED_HASHES[case_name]


@pytest.mark.parametrize("protocol", tuple(PROTOCOLS))
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_normalization_is_deterministic_and_does_not_alias_input(protocol: str, case_name: str) -> None:
    fixture_name, normalizer = PROTOCOLS[protocol]
    request = _load_fixture(fixture_name)[case_name]
    snapshot = copy.deepcopy(request)
    first = _canonical_fields(normalizer(request))
    second = _canonical_fields(normalizer(copy.deepcopy(request)))
    assert first == second
    assert _state_hash(first) == _state_hash(second)
    assert request == snapshot

    request.clear()
    assert first == _load_fixture("canonical.json")[case_name]


@pytest.mark.parametrize("protocol", tuple(PROTOCOLS))
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_object_key_insertion_order_does_not_change_canonical_state(protocol: str, case_name: str) -> None:
    fixture_name, _normalizer = PROTOCOLS[protocol]
    request = _load_fixture(fixture_name)[case_name]
    original = _normalize(protocol, request)
    reordered = _normalize(protocol, _reverse_object_keys(request))
    assert reordered == original
    assert _state_hash(reordered) == _state_hash(original)


def test_array_order_is_preserved_and_changes_state_hash() -> None:
    expected = _load_fixture("canonical.json")
    original = expected["tool_round_trip"]
    original_hash = _state_hash(original)

    messages_reordered = copy.deepcopy(original)
    messages_reordered["messages"][2:4] = reversed(messages_reordered["messages"][2:4])
    assert _state_hash(messages_reordered) != original_hash

    tools_reordered = copy.deepcopy(original)
    tools_reordered["tools"].reverse()
    assert _state_hash(tools_reordered) != original_hash

    calls_reordered = copy.deepcopy(original)
    calls_reordered["messages"][1]["tool_calls"].reverse()
    assert _state_hash(calls_reordered) != original_hash

    image_content = expected["images"]
    content_reordered = copy.deepcopy(image_content)
    content_reordered["messages"][0]["content"].reverse()
    assert _state_hash(content_reordered) != _state_hash(image_content)


def test_normalization_helpers_produce_stable_json_without_mutation() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"b": {}, "a": {}}},
            },
        }
    ]
    kwargs = {"z": [2, 1], "nested": {"b": 2.0, "a": "值"}}
    tools_snapshot = copy.deepcopy(tools)
    kwargs_snapshot = copy.deepcopy(kwargs)

    normalized_tools = normalize_tools(tools)
    normalized_kwargs = normalize_template_kwargs(kwargs)
    assert tools == tools_snapshot
    assert kwargs == kwargs_snapshot
    assert list(normalized_tools[0]["function"]["parameters"]) == ["properties", "type"]
    assert list(normalized_tools[0]["function"]["parameters"]["properties"]) == ["a", "b"]
    assert normalized_kwargs == {"nested": {"a": "值", "b": 2}, "z": [2, 1]}
    assert normalize_template_kwargs(None) == {}
    assert normalize_template_kwargs({}) == {}


@pytest.mark.parametrize("value", [{"bad": {1, 2}}, {1: "non-string key"}, {"bad": float("nan")}])
def test_template_kwargs_reject_non_json_values(value: dict[Any, Any]) -> None:
    with pytest.raises((TypeError, ValueError), match="chat_template_kwargs"):
        normalize_template_kwargs(value)


def test_chat_rejects_falsy_non_object_template_kwargs() -> None:
    for value in ([], "", 0, False):
        _assert_request_error(
            _normalized_chat_request,
            {"messages": [{"role": "user", "content": "hello"}], "chat_template_kwargs": value},
            path="chat_template_kwargs must be a dict",
            param="chat_template_kwargs",
        )


INVALID_ROLE_CASES = (
    (
        _normalized_chat_request,
        {"messages": [{"role": "unknown", "content": "x"}]},
        "messages[0].role",
        "messages",
    ),
    (
        _normalized_chat_request,
        {"messages": [{"role": "", "content": "x"}]},
        "messages[0].role",
        "messages",
    ),
    (
        _normalized_chat_request,
        {"messages": [{"role": 1, "content": "x"}]},
        "messages[0].role",
        "messages",
    ),
    (
        _normalized_responses_request,
        {"input": [{"type": "message", "role": "unknown", "content": "x"}]},
        "input[0].role",
        "input",
    ),
    (
        _normalized_responses_request,
        {"input": [{"type": "message", "role": "", "content": "x"}]},
        "input[0].role",
        "input",
    ),
    (
        _normalized_responses_request,
        {"input": [{"type": "message", "role": 1, "content": "x"}]},
        "input[0].role",
        "input",
    ),
    (
        _normalized_anthropic_request,
        {"max_tokens": 8, "messages": [{"role": "unknown", "content": "x"}]},
        "messages[0].role",
        "messages",
    ),
    (
        _normalized_anthropic_request,
        {"max_tokens": 8, "messages": [{"role": "", "content": "x"}]},
        "messages[0].role",
        "messages",
    ),
    (
        _normalized_anthropic_request,
        {"max_tokens": 8, "messages": [{"role": 1, "content": "x"}]},
        "messages[0].role",
        "messages",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path", "param"), INVALID_ROLE_CASES)
def test_protocols_reject_unknown_roles(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
    param: str,
) -> None:
    message = _assert_request_error(normalizer, payload, path=path, param=param)
    reordered_message = _assert_request_error(normalizer, _reverse_object_keys(payload), path=path, param=param)
    assert reordered_message == message


MISSING_CALL_ID_CASES = (
    (
        _normalized_chat_request,
        {
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Search."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{}"}}],
                },
            ]
        },
        "messages[2].tool_calls[0].id",
        "messages",
    ),
    (
        _normalized_responses_request,
        {
            "input": [
                {"type": "message", "role": "user", "content": "Search."},
                {"type": "message", "role": "assistant", "content": "Checking."},
                {"type": "function_call", "name": "search", "arguments": "{}"},
            ]
        },
        "input[2].call_id",
        "input",
    ),
    (
        _normalized_anthropic_request,
        {
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "Search."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "name": "search", "input": {}},
                    ],
                },
            ],
        },
        "messages[1].content[1].id",
        "messages",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path", "param"), MISSING_CALL_ID_CASES)
def test_protocols_reject_tool_calls_without_ids(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
    param: str,
) -> None:
    _assert_request_error(normalizer, payload, path=path, param=param)


INCOMPLETE_TOOL_CASES = (
    (
        _normalized_chat_request,
        {
            "messages": [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"arguments": "{}"}}],
                },
            ]
        },
        "messages[1].tool_calls[0].function.name",
        "messages",
    ),
    (
        _normalized_chat_request,
        {
            "messages": [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search"}}],
                },
            ]
        },
        "messages[1].tool_calls[0].function.arguments",
        "messages",
    ),
    (
        _normalized_responses_request,
        {"input": [{"type": "function_call", "call_id": "call_1", "arguments": "{}"}]},
        "input[0].name",
        "input",
    ),
    (
        _normalized_responses_request,
        {"input": [{"type": "function_call", "call_id": "call_1", "name": "search"}]},
        "input[0].arguments",
        "input",
    ),
    (
        _normalized_anthropic_request,
        {
            "max_tokens": 8,
            "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "input": {}}]}],
        },
        "messages[0].content[0].name",
        "messages",
    ),
    (
        _normalized_anthropic_request,
        {
            "max_tokens": 8,
            "messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "search"}]}],
        },
        "messages[0].content[0].input",
        "messages",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path", "param"), INCOMPLETE_TOOL_CASES)
def test_protocols_reject_incomplete_tool_calls(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
    param: str,
) -> None:
    _assert_request_error(normalizer, payload, path=path, param=param)


MISSING_RESULT_ID_CASES = (
    (
        _normalized_chat_request,
        {
            "messages": [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "content": "result"},
            ]
        },
        "messages[2].tool_call_id",
        "messages",
    ),
    (
        _normalized_responses_request,
        {
            "input": [
                {"type": "message", "role": "user", "content": "x"},
                {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{}"},
                {"type": "function_call_output", "output": "result"},
            ]
        },
        "input[2].call_id",
        "input",
    ),
    (
        _normalized_anthropic_request,
        {
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "call_1", "name": "search", "input": {}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "content": "result"}]},
            ],
        },
        "messages[2].content[0].tool_use_id",
        "messages",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path", "param"), MISSING_RESULT_ID_CASES)
def test_protocols_reject_tool_results_without_ids(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
    param: str,
) -> None:
    _assert_request_error(normalizer, payload, path=path, param=param)


EMPTY_CONTENT_CASES = (
    (_normalized_chat_request, {"messages": [{"role": "user", "content": ""}]}, "messages[0].content", "messages"),
    (_normalized_chat_request, {"messages": [{"role": "user", "content": []}]}, "messages[0].content", "messages"),
    (
        _normalized_responses_request,
        {"input": [{"type": "message", "role": "user", "content": ""}]},
        "messages[0].content",
        "input",
    ),
    (
        _normalized_responses_request,
        {"input": [{"type": "message", "role": "user", "content": []}]},
        "messages[0].content",
        "input",
    ),
    (
        _normalized_anthropic_request,
        {"max_tokens": 8, "messages": [{"role": "user", "content": ""}]},
        "messages[0].content",
        "messages",
    ),
    (
        _normalized_anthropic_request,
        {"max_tokens": 8, "messages": [{"role": "user", "content": []}]},
        "messages[0].content",
        "messages",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path", "param"), EMPTY_CONTENT_CASES)
def test_protocols_reject_empty_user_content(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
    param: str,
) -> None:
    _assert_request_error(normalizer, payload, path=path, param=param)


INVALID_IMAGE_CASES = (
    (
        _normalized_chat_request,
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
        "messages[0].content[0].image_url.url",
        "messages",
    ),
    (
        _normalized_chat_request,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,***"}}],
                }
            ]
        },
        "messages[0].content[0].image_url.url",
        "messages",
    ),
    (
        _normalized_chat_request,
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "https://x"}]}]},
        "messages[0].content[0].image_url",
        "messages",
    ),
    (
        _normalized_responses_request,
        {"input": [{"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": ""}]}]},
        "input[0].content[0].image_url",
        "input",
    ),
    (
        _normalized_responses_request,
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,***"}],
                }
            ]
        },
        "messages[0].content[0].image_url.url",
        "input",
    ),
    (
        _normalized_anthropic_request,
        {
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}]}],
        },
        "messages[0].content[0].source.media_type",
        "messages",
    ),
    (
        _normalized_anthropic_request,
        {
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "***"}}
                    ],
                }
            ],
        },
        "messages[0].content[0].image_url.url",
        "messages",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path", "param"), INVALID_IMAGE_CASES)
def test_protocols_reject_invalid_images(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
    param: str,
) -> None:
    _assert_request_error(normalizer, payload, path=path, param=param)


INVALID_CONTENT_BLOCK_CASES = (
    (
        _normalized_chat_request,
        {"messages": [{"role": "user", "content": [{"type": "audio", "audio": {}}]}]},
        "messages[0].content[0].type",
        "messages",
    ),
    (
        _normalized_responses_request,
        {"input": [{"type": "message", "role": "user", "content": [{"type": "input_audio"}]}]},
        "input[0].content[0].type",
        "input",
    ),
    (
        _normalized_anthropic_request,
        {
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}],
        },
        "messages[0].content[0].type",
        "messages",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path", "param"), INVALID_CONTENT_BLOCK_CASES)
def test_protocols_reject_unsupported_content_blocks(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
    param: str,
) -> None:
    _assert_request_error(normalizer, payload, path=path, param=param)


EMPTY_OR_UNSUPPORTED_REQUEST_CASES = (
    (_normalized_chat_request, {"messages": []}, "messages projected to zero supported messages", "messages"),
    (_normalized_responses_request, {"input": []}, "input projected to zero supported messages", "input"),
    (
        _normalized_anthropic_request,
        {"max_tokens": 8, "messages": []},
        "messages projected to zero supported messages",
        "messages",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path", "param"), EMPTY_OR_UNSUPPORTED_REQUEST_CASES)
def test_protocols_reject_empty_message_sequences(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
    param: str,
) -> None:
    _assert_request_error(normalizer, payload, path=path, param=param)


def test_anthropic_rejects_tool_result_without_content() -> None:
    _assert_request_error(
        _normalized_anthropic_request,
        {
            "max_tokens": 8,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "call_1", "name": "search", "input": {}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1"}]},
            ],
        },
        path="messages[1].content[0].content",
        param="messages",
    )


def _mismatched_tool_result_payloads() -> tuple[tuple[Normalizer, dict[str, Any], str], ...]:
    chat = {
        "messages": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_other", "content": "x"},
        ]
    }
    responses = {
        "input": [
            {"type": "message", "role": "user", "content": "x"},
            {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_other", "output": "x"},
        ]
    }
    anthropic = {
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_other", "content": "x"}]},
        ],
    }
    return (
        (_normalized_chat_request, chat, "messages"),
        (_normalized_responses_request, responses, "input"),
        (_normalized_anthropic_request, anthropic, "messages"),
    )


@pytest.mark.parametrize(("normalizer", "payload", "param"), _mismatched_tool_result_payloads())
def test_protocols_reject_tool_results_without_matching_calls(
    normalizer: Normalizer,
    payload: dict[str, Any],
    param: str,
) -> None:
    _assert_request_error(normalizer, payload, path="messages[2].tool_call_id", param=param)


def test_complete_tool_call_without_result_is_allowed_as_intermediate_state() -> None:
    requests = {
        protocol: _load_fixture(fixture_name)["tool_round_trip"]
        for protocol, (fixture_name, _normalizer) in PROTOCOLS.items()
    }
    requests["chat_completions"]["messages"] = requests["chat_completions"]["messages"][:2]
    requests["responses"]["input"] = requests["responses"]["input"][:4]
    requests["anthropic_messages"]["messages"] = requests["anthropic_messages"]["messages"][:2]

    normalized = {protocol: _normalize(protocol, request) for protocol, request in requests.items()}
    assert normalized["chat_completions"] == normalized["responses"] == normalized["anthropic_messages"]
    assert len(normalized["chat_completions"]["messages"][-1]["tool_calls"]) == 2


def test_empty_tool_result_is_preserved_consistently() -> None:
    chat = {
        "messages": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": ""},
        ]
    }
    responses = {
        "input": [
            {"type": "message", "role": "user", "content": "x"},
            {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": ""},
        ]
    }
    anthropic = {
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": ""}]},
        ],
    }
    normalized = (
        _normalize("chat_completions", chat),
        _normalize("responses", responses),
        _normalize("anthropic_messages", anthropic),
    )
    assert normalized[0] == normalized[1] == normalized[2]
    assert normalized[0]["messages"][-1]["content"] == ""


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ("not-json", "must be valid JSON"),
        ("[]", "must be a JSON object"),
        ('{"a":1,"a":2}', "contains duplicate key"),
    ],
)
def test_tool_arguments_reject_noncanonical_values(arguments: str, error: str) -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": arguments}}
                ],
            },
        ]
    }
    _assert_request_error(
        _normalized_chat_request,
        payload,
        path=f"messages[1].tool_calls[0].function.arguments {error}",
        param="messages",
    )


def test_tool_definitions_require_stable_function_schema() -> None:
    with pytest.raises(TypeError, match=r"tools\[0\]\.function\.parameters"):
        normalize_tools([{"type": "function", "function": {"name": "search"}}])
    with pytest.raises(ValueError, match=r"tools\[0\]\.function\.name"):
        normalize_tools([{"type": "function", "function": {"parameters": {}}}])


def test_responses_rejects_tool_without_function_type() -> None:
    _assert_request_error(
        _normalized_responses_request,
        {"input": "x", "tools": [{"name": "search", "parameters": {}}]},
        path="tools[0].type",
        param="tools",
    )


MISSING_TOOL_SCHEMA_CASES = (
    (
        _normalized_chat_request,
        {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        },
        "tools[0].function.parameters",
    ),
    (
        _normalized_responses_request,
        {"input": "x", "tools": [{"type": "function", "name": "search"}]},
        "tools[0].parameters",
    ),
    (
        _normalized_anthropic_request,
        {"max_tokens": 8, "messages": [{"role": "user", "content": "x"}], "tools": [{"name": "search"}]},
        "tools[0].input_schema",
    ),
)


@pytest.mark.parametrize(("normalizer", "payload", "path"), MISSING_TOOL_SCHEMA_CASES)
def test_protocols_reject_tools_without_parameter_schemas(
    normalizer: Normalizer,
    payload: dict[str, Any],
    path: str,
) -> None:
    _assert_request_error(normalizer, payload, path=path, param="tools")
