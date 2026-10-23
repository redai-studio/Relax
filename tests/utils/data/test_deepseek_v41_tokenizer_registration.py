# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tokenizer registration order without importing ML dependencies or the model
package."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


_SOURCE = Path(__file__).resolve().parents[3] / "relax/utils/data/processing_utils.py"


@pytest.fixture
def tokenizer_loader(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, list[Any]]:
    events: list[Any] = []
    tokenizer = SimpleNamespace(chat_template=None)
    configuration = ModuleType("relax.models.deepseekv41.configuration")
    configuration.register_hf_config = lambda: events.append(("register",))
    monkeypatch.setitem(sys.modules, configuration.__name__, configuration)

    def from_pretrained(name_or_path: str, **kwargs: Any) -> Any:
        events.append(("tokenizer", name_or_path, kwargs))
        return tokenizer

    nodes = [
        node
        for node in ast.parse(_SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "load_tokenizer"
    ]
    assert len(nodes) == 1
    namespace = {
        "os": os,
        "json": json,
        "AutoTokenizer": SimpleNamespace(from_pretrained=from_pretrained),
        "logger": SimpleNamespace(info=lambda *args: None),
    }
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[])
    exec(compile(module, str(_SOURCE), "exec"), namespace)
    return namespace["load_tokenizer"], tokenizer, events


@pytest.mark.parametrize("model_type", ["deepseek_v41", "deepseek_v41_text", "qwen2", None])
def test_load_tokenizer_registers_only_v41_before_auto_tokenizer_and_preserves_kwargs(
    tokenizer_loader: tuple[Any, Any, list[Any]], tmp_path: Path, model_type: str | None
) -> None:
    load, tokenizer, events = tokenizer_loader
    config = {"model_type": model_type} if model_type is not None else {}
    (tmp_path / "config.json").write_text(json.dumps(config))
    options = {"keep": True}
    kwargs = {"trust_remote_code": True, "use_fast": False, "padding_side": "left", "custom_options": options}

    loaded = load(str(tmp_path), **kwargs)

    assert loaded is tokenizer
    expected_registration = [("register",)] if model_type in ("deepseek_v41", "deepseek_v41_text") else []
    assert events == [*expected_registration, ("tokenizer", str(tmp_path), kwargs)]
    assert events[-1][2]["custom_options"] is options
    assert options == {"keep": True}


def test_load_tokenizer_without_local_config_does_not_register(
    tokenizer_loader: tuple[Any, Any, list[Any]], tmp_path: Path
) -> None:
    load, tokenizer, events = tokenizer_loader

    assert load(str(tmp_path), trust_remote_code=True) is tokenizer

    assert events == [("tokenizer", str(tmp_path), {"trust_remote_code": True})]
