import os
import sys
import types

import pytest

# The normalizers are pure request adapters, while importing the service also
# defines a Ray Serve deployment.  Keep this test local and deterministic on
# platforms where Ray cannot pickle FastAPI's thread locks.
if os.name == "nt":
    sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))
    import ray.serve

    _original_serve_ingress = ray.serve.ingress
    ray.serve.ingress = lambda app: (lambda cls: cls)

try:
    from relax.agentic.session.service import (
        AgenticChatRequestError,
        _normalized_anthropic_request,
        _normalized_chat_request,
        _normalized_responses_request,
    )
finally:
    if os.name == "nt":
        ray.serve.ingress = _original_serve_ingress


def _canonical_projection(request):
    return {
        "messages": request["messages"],
        "tools": request["tools"],
        "chat_template_kwargs": request["chat_template_kwargs"],
    }


def test_chat_responses_and_anthropic_requests_share_canonical_projection():
    chat = _normalized_chat_request(
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this image."},
                        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-weather",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":"Beijing"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-weather", "content": "sunny"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Weather lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    responses = _normalized_responses_request(
        {
            "instructions": "You are helpful.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect this image."},
                        {"type": "input_image", "image_url": "https://example.test/image.png"},
                    ],
                },
                {"type": "function_call", "call_id": "call-weather", "name": "weather", "arguments": '{"city":"Beijing"}'},
                {"type": "function_call_output", "call_id": "call-weather", "output": "sunny"},
            ],
            "tools": [{"type": "function", "name": "weather", "description": "Weather lookup", "parameters": {"type": "object"}}],
            "reasoning": {"effort": "none"},
        }
    )
    anthropic = _normalized_anthropic_request(
        {
            "system": "You are helpful.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this image."},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.test/image.png"},
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call-weather", "name": "weather", "input": {"city": "Beijing"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-weather", "content": "sunny"}]},
            ],
            "tools": [{"name": "weather", "description": "Weather lookup", "input_schema": {"type": "object"}}],
            "max_tokens": 32,
            "thinking": {"type": "disabled"},
        }
    )

    assert _canonical_projection(chat) == _canonical_projection(responses) == _canonical_projection(anthropic)


def test_protocols_share_tool_shape_when_optional_description_is_omitted():
    chat = _normalized_chat_request(
        {"messages": [{"role": "user", "content": "hello"}], "tools": [{"type": "function", "function": {"name": "ping"}}]}
    )
    responses = _normalized_responses_request(
        {
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "tools": [{"type": "function", "name": "ping"}],
            "max_output_tokens": 1,
        }
    )
    anthropic = _normalized_anthropic_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "ping"}],
            "max_tokens": 1,
        }
    )

    assert _canonical_projection(chat) == _canonical_projection(responses) == _canonical_projection(anthropic)


@pytest.mark.parametrize(
    ("normalizer", "payload", "path"),
    [
        (_normalized_chat_request, {"messages": [{"role": "alien", "content": "x"}]}, "messages[0].role"),
        (_normalized_chat_request, {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "", "function": {"name": "f", "arguments": "{}"}}]}]}, "messages[0].tool_calls[0].id"),
        (_normalized_responses_request, {"input": [{"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": ""}]}], "max_output_tokens": 1}, "input[0].content[0].image_url"),
        (_normalized_anthropic_request, {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": ""}}]}], "max_tokens": 1}, "messages[0].content[0].source.url"),
    ],
)
def test_protocol_errors_preserve_complete_field_paths(normalizer, payload, path):
    with pytest.raises(AgenticChatRequestError, match=path.replace("[", r"\[").replace("]", r"\]")):
        normalizer(payload)
