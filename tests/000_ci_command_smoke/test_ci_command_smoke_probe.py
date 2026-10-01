# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Temporary synthetic job outcomes for the disposable CI command test PR."""

import os
import sys

import pytest


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") != "true"
    or os.environ.get("GITHUB_EVENT_NAME") != "pull_request"
    or os.environ.get("GITHUB_REPOSITORY") != "redai-studio/Relax"
    or os.environ.get("GITHUB_WORKFLOW") != "CI"
    or os.environ.get("GITHUB_JOB") != "test"
    or os.environ.get("GITHUB_HEAD_REF") != "codex/ci-command-smoke"
    or sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)},
    reason="Synthetic outcomes are restricted to the disposable draft PR's CPU CI jobs.",
)
def test_ci_command_smoke_synthetic_job_outcome() -> None:
    if sys.version_info[:2] == (3, 10):
        pytest.fail("Expected draft-only Python 3.10 failure for /rerun failed verification.")
    pytest.exit(
        "Draft-only successful control: the full unit test suite was intentionally not exercised.",
        returncode=0,
    )
