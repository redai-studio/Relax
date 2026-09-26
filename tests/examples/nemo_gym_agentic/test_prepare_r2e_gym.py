# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "nemo_gym_agentic"
    / "recipes"
    / "r2e-gym"
    / "prepare_r2e_gym.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("prepare_r2e_gym", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load R2E-Gym preparation module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
convert_r2e_row = MODULE.convert_r2e_row
instance_id_from_docker_image = MODULE.instance_id_from_docker_image
sif_name_from_docker_image = MODULE.sif_name_from_docker_image


DOCKER_IMAGE = "namanjain12/aiohttp_final:f0d74880deec8fcd982bce639c93c5e130d41198"


def test_r2e_image_names_match_pinned_gym_lookup() -> None:
    assert (
        instance_id_from_docker_image(DOCKER_IMAGE) == "namanjain12__aiohttp-f0d74880deec8fcd982bce639c93c5e130d41198"
    )
    assert sif_name_from_docker_image(DOCKER_IMAGE) == "aiohttp_final_f0d74880deec8fcd982bce639c93c5e130d41198.sif"


def test_convert_r2e_row_builds_swe_agent_metadata() -> None:
    row = {
        "repo_name": "aiohttp",
        "docker_image": DOCKER_IMAGE,
        "commit_hash": "f0d74880deec8fcd982bce639c93c5e130d41198",
        "problem_statement": "Fix route parsing.",
        "parsed_commit_content": ('{"old_commit_hash":"f0d74880deec8fcd982bce639c93c5e130d41198^","file_diffs":[]}'),
    }

    converted = convert_r2e_row(row)
    metadata = converted["responses_create_params"]["metadata"]

    assert metadata["dataset_name"] == "R2E-Gym/R2E-Gym-Lite"
    assert metadata["split"] == "train"
    assert metadata["instance_id"] == converted["instance_id"]
    assert metadata["base_commit"] == "f0d74880deec8fcd982bce639c93c5e130d41198^"
    assert json.loads(metadata["instance_dict"]) == {
        **row,
        "instance_id": converted["instance_id"],
        "repo": "aiohttp",
        "version": "f0d74880deec8fcd982bce639c93c5e130d41198",
        "base_commit": "f0d74880deec8fcd982bce639c93c5e130d41198^",
    }
    assert converted["agent_ref"]["name"] == "swe_agents"


def test_convert_r2e_row_rejects_inconsistent_commit() -> None:
    row = {
        "repo_name": "aiohttp",
        "docker_image": DOCKER_IMAGE,
        "commit_hash": "deadbeef",
        "problem_statement": "Fix route parsing.",
        "parsed_commit_content": '{"file_diffs":[]}',
    }

    with pytest.raises(ValueError, match="does not match"):
        convert_r2e_row(row)
