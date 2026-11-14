# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Read a Ray job's status/logs and save them beside its frozen run record."""

import argparse
import json
from pathlib import Path


def main() -> None:
    from ray.job_submission import JobSubmissionClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.run_dir / "run.json").is_file():
        parser.error("run-dir must contain the launch-time run.json")
    client = JobSubmissionClient(args.address)
    status = client.get_job_status(args.job_id)
    (args.run_dir / "training.log").write_text(client.get_job_logs(args.job_id))
    (args.run_dir / "status.json").write_text(json.dumps({"job_id": args.job_id, "status": str(status)}) + "\n")
    print(json.dumps({"job_id": args.job_id, "status": str(status), "run_dir": str(args.run_dir)}))


if __name__ == "__main__":
    main()
