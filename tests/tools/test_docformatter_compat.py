# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import ast
import importlib.util
import sys
import tokenize
from pathlib import Path

import pytest


@pytest.fixture
def hook():
    pytest.importorskip("docformatter")
    path = Path(__file__).resolve().parents[2] / ".pre-commit-hooks/docformatter_compat.py"
    spec = importlib.util.spec_from_file_location("docformatter_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("symbols", ["✓ / ✗ / ○", "中文 — 符号"])
def test_docformatter_preserves_unicode_with_incorrect_token_columns(hook, monkeypatch, tmp_path, newline, symbols):
    source = (
        "def example():\n"
        f'    """Describe {symbols} in a multiline docstring.\n'
        "\n"
        "    Preserve the original text and closing quotes.\n"
        '    """\n'
        "    return 1\n"
    ).replace("\n", newline)
    path = tmp_path / "example.py"
    path.write_bytes(source.encode())
    original = tokenize.generate_tokens

    def broken_tokens(readline):
        for token in original(readline):
            if token.type == tokenize.STRING and token.start[0] != token.end[0]:
                # Reproduce the bad end column even on newer Python versions.
                correct_end = len(token.string.rsplit("\n", 1)[-1])
                token = token._replace(end=(token.end[0], correct_end - 2))
            yield token

    monkeypatch.setattr(hook, "_original_generate_tokens", broken_tokens)
    monkeypatch.setattr(sys, "argv", ["docformatter", "--in-place", "--wrap-descriptions", "79", str(path)])
    for _ in range(2):
        assert hook.main() == 0
        assert path.read_bytes() == source.encode()
        ast.parse(path.read_bytes())
        assert tokenize.generate_tokens is original


def test_docformatter_still_formats_and_converges(hook, monkeypatch, tmp_path):
    path = tmp_path / "example.py"
    source = 'def example():\n    """' + "A long description " * 8 + 'ends here."""\n    return 1\n'
    path.write_text(source)
    monkeypatch.setattr(sys, "argv", ["docformatter", "--in-place", "--wrap-descriptions", "79", str(path)])
    assert hook.main() == 0
    formatted = path.read_text()
    assert formatted != source
    ast.parse(formatted)
    assert hook.main() == 0
    assert path.read_text() == formatted
