# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Disposable draft-PR probe for the failed-job rerun path."""

import os
import sys

import pytest


INJECT_FAILURE = True


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") != "true"
    or os.environ.get("GITHUB_WORKFLOW") != "CI"
    or os.environ.get("GITHUB_HEAD_REF") != "codex/ci-command-smoke"
    or os.environ.get("GITHUB_JOB") != "test"
    or sys.version_info[:2] != (3, 10),
    reason="This probe only exercises the Python 3.10 CPU job in the disposable draft PR.",
)
def test_ci_command_smoke_failed_job() -> None:
    if INJECT_FAILURE:
        pytest.fail("Expected draft-only failure for live /rerun failed verification.")
