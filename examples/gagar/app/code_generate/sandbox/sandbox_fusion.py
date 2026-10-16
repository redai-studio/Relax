# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run the original SandboxFusion /submit harness inside Apptainer, without
HTTP."""

import asyncio
import json
import sys
from pathlib import Path


async def main() -> None:
    import structlog

    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    from sandbox.configs.run_config import RunConfig
    from sandbox.datasets.types import SubmitRequest
    from sandbox.runners import major
    from sandbox.server.online_judge_api import get_dataset_cls

    config = RunConfig.get_instance_sync()
    if config.dataset.cpu_runner_url or config.dataset.gpu_runner_url:
        raise RuntimeError("sandbox grading requires local SandboxFusion runners")
    # This Python-only image uses a venv in place of SF's conda runtime.
    major.get_python_rt_env = lambda _name: {"PATH": "/opt/sf/bin:/usr/local/bin:/usr/bin:/bin"}
    request = SubmitRequest.model_validate_json(Path(sys.argv[1]).read_text())
    dataset = get_dataset_cls(request.dataset, request.config)
    result = await dataset.evaluate_single(request)
    # Candidate stdout belongs to RunCodeResponse, not this result channel.
    sys.stdout.write(json.dumps(result.model_dump(mode="json")) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
