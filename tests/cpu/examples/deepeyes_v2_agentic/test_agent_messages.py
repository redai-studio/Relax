# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unit tests for agent observation-message construction."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

import app.agent as agent_module  # noqa: E402
import app.sandboxes as sandboxes_module  # noqa: E402
from app.agent import build_tool_message  # noqa: E402


def test_text_only_observation_uses_string_content():
    msg = build_tool_message(body_text="search result: foo", images=[])
    assert msg["role"] == "tool"
    assert msg["content"] == "search result: foo"


def test_images_observation_emits_content_parts():
    img = Image.new("RGB", (32, 32), color="red")
    msg = build_tool_message(body_text="Code result", images=[img, img])
    assert msg["role"] == "tool"
    assert isinstance(msg["content"], list)
    text_parts = [p for p in msg["content"] if p["type"] == "text"]
    image_parts = [p for p in msg["content"] if p["type"] == "image_url"]
    assert len(text_parts) == 1 and text_parts[0]["text"] == "Code result"
    assert len(image_parts) == 2
    for part in image_parts:
        assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_executor_apptainer_image_path_overrides_yaml(tmp_path, monkeypatch):
    """APPTAINER_IMAGE_PATH wins over YAML; SANDBOX_BACKEND overrides backend
    name."""
    config_file = tmp_path / "apptainer_config.yaml"
    config_file.write_text("image: ./relative.sif\nnv: false\n", encoding="utf-8")

    monkeypatch.setenv("SANDBOX_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("APPTAINER_IMAGE_PATH", "/tmp/override.sif")
    monkeypatch.setenv("SANDBOX_BACKEND", "spy_backend")

    captured: dict = {}

    def spy(*, name, config=None, custom_path=None):
        captured["name"] = name
        captured["config"] = config
        raise RuntimeError("stop")

    monkeypatch.setattr(sandboxes_module, "get_sandbox_backend", spy)
    monkeypatch.setattr(agent_module, "get_sandbox_backend", spy)

    try:
        agent_module._build_executor("default_backend", 60)
    except RuntimeError:
        pass

    assert captured["name"] == "spy_backend"
    assert captured["config"]["image"] == "/tmp/override.sif"


def test_build_executor_resolves_relative_image_against_yaml_dir(tmp_path, monkeypatch):
    """Without APPTAINER_IMAGE_PATH override, relative image resolves vs YAML
    dir."""
    config_file = tmp_path / "apptainer_config.yaml"
    config_file.write_text("image: ./local.sif\n", encoding="utf-8")

    monkeypatch.setenv("SANDBOX_CONFIG_PATH", str(config_file))
    monkeypatch.delenv("APPTAINER_IMAGE_PATH", raising=False)
    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)

    captured: dict = {}

    def spy(*, name, config=None, custom_path=None):
        captured["name"] = name
        captured["config"] = config
        raise RuntimeError("stop")

    monkeypatch.setattr(agent_module, "get_sandbox_backend", spy)

    try:
        agent_module._build_executor("apptainer_jupyter", 60)
    except RuntimeError:
        pass

    assert captured["name"] == "apptainer_jupyter"
    assert captured["config"]["image"] == str(tmp_path / "local.sif")
