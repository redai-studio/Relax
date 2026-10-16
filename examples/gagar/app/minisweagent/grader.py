#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A mini-swe-agent process using Relax GenRM and sandbox-only tools."""

from urllib.parse import quote

import httpx

from examples.gagar.app.config import load_task_config
from examples.gagar.app.minisweagent.scoring import grade_to_weights
from examples.gagar.core.grs import GraderEnvironment


class BridgeBash:
    """Route model commands to the selected candidate through the bridge."""

    def __init__(self, manifest: dict, config: dict) -> None:
        self.manifest = manifest
        group_id = quote(manifest["group_id"], safe="")
        self.url = f"{manifest['bridge_url'].rstrip('/')}/gagar/groups/{group_id}/bash"
        self.client = httpx.Client(timeout=config["request_timeout"], trust_env=False)

    def execute(self, action: dict) -> dict:
        # Validate the exact model argument set at the bridge, including
        # a missing rollout_n. Never accept an endpoint from the model.
        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.manifest['token']}"},
            json={key: value for key, value in action.items() if key != "name"},
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.client.close()


def create_environment(manifest: dict, config: dict | None = None) -> GraderEnvironment:
    """Bind this task's tools and scoring to the shared grader environment."""
    if config is None:
        config = load_task_config("minisweagent")["environment"]
    return GraderEnvironment(
        manifest,
        backend=BridgeBash(manifest, config),
        grade_to_weights=grade_to_weights,
        sections=set(config["evidence_sections"]),
        evidence_access=config["evidence_access"],
        environment_type=config["environment_type"],
    )
