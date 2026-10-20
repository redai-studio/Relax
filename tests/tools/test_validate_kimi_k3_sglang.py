# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "examples/models/kimi-k3/tools/validate_kimi_k3_sglang.py"
_SPEC = importlib.util.spec_from_file_location("k3_serve_validation", _PATH)
validation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validation)


@pytest.mark.parametrize("failure", [None, "empty_final_answer", "nan_logprob"])
def test_verify_requires_final_answers_and_finite_logprobs(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    def request(base_url: str, path: str, body: dict, timeout: int) -> dict:
        if path == "/v1/chat/completions":
            return {
                "choices": [
                    {
                        "message": {
                            "content": "" if failure == "empty_final_answer" else "5",
                            "reasoning_content": "Thinking",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 1},
            }
        return {
            "text": " Paris",
            "meta_info": {
                "completion_tokens": 1,
                "output_token_logprobs": [[float("nan") if failure == "nan_logprob" else -0.1, 123, None]],
            },
        }

    monkeypatch.setattr(validation, "_request", request)
    if failure:
        with pytest.raises(ValueError):
            validation._verify("unused", 1)
    else:
        assert len(validation._verify("unused", 1)) == 3
