#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Ray submit can exit 0 for STOPPED; synchronous launchers require SUCCEEDED.
set -euo pipefail
[[ -z "${RAY_NO_WAIT:-}" ]] || exit 0
python3 - "$@" <<'PYSTATUS'
import sys

from ray.job_submission import JobStatus, JobSubmissionClient

address, submission_id = sys.argv[1:]
status = JobSubmissionClient(address).get_job_status(submission_id)
if status != JobStatus.SUCCEEDED:
    sys.exit(f"Ray job {submission_id} ended with status {status}; expected SUCCEEDED")
PYSTATUS
