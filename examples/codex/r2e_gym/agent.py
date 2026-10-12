# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run official Codex with optional transparent compaction recording."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import yaml


def _score_swebench(row: dict, log: Path) -> float:
    from swebench.harness.constants import ResolvedStatus
    from swebench.harness.grading import get_eval_tests_report, get_logs_eval, get_resolution_status
    from swebench.harness.log_parsers.python import MAP_REPO_TO_PARSER_PY

    spec = SimpleNamespace(**row, log_parser=MAP_REPO_TO_PARSER_PY[row["repo"]].__name__)
    statuses, valid = get_logs_eval(spec, str(log))
    if not valid:
        return 0.0
    report = get_eval_tests_report(statuses, row)
    return float(get_resolution_status(report) == ResolvedStatus.FULL.value)


def _configure_codex(codex_home: Path, *, compact: bool, context_window: int, compact_limit: int) -> None:
    # Codex clamps auto-compaction to 90% of its context window. Raise both
    # values in direct mode so Relax rejects long histories before compaction.
    window = context_window if compact else 1_000_000_000
    threshold = compact_limit if compact else 900_000_000
    source = Path(__file__).with_name("codex.toml").read_text(encoding="utf-8")
    (codex_home / "config.toml").write_text(
        f"model_context_window = {window}\nmodel_auto_compact_token_limit = {threshold}\n" + source,
        encoding="utf-8",
    )


def _parse_pytest_status(log: str) -> dict[str, str]:
    if "short test summary info" not in log:
        return {}
    statuses = {}
    for line in log.split("short test summary info", 1)[1].strip().splitlines():
        clean = re.sub(r"\u001b\[\d+m", "", line)
        if "PASSED" in clean:
            statuses[".".join(clean.split("::")[1:])] = "PASSED"
        elif "FAILED" in clean:
            statuses[".".join(clean.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in clean:
            statuses[".".join(clean.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return {key.split(" - ")[0]: statuses[key] for key in sorted(statuses)}


def _reward(expected_output_json: str, test_output: str) -> float:
    expected = json.loads(expected_output_json)
    expected = {re.sub(r"\u001b\[\d+m", "", key).split(" - ")[0]: value for key, value in sorted(expected.items())}
    actual = _parse_pytest_status(test_output)
    return float(actual == expected)


def main() -> None:
    input_path = Path(os.environ["RELAX_INPUT_JSON"])
    output_path = Path(os.environ["RELAX_OUTPUT_JSON"])
    session_dir = Path(os.environ["RELAX_SESSION_IO_DIR"]).resolve()
    sif_dir = Path(os.environ["R2E_SIF_DIR"])
    wheel_dir = Path(os.environ["R2E_WHEEL_DIR"])
    runtime_dir = Path(os.environ["CODEX_RUNTIME_DIR"])
    config = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8"))
    apptainer_config = config["apptainer"]
    apptainer = apptainer_config["executable"]
    timeout_s = int(config["codex"]["timeout_s"])
    reward_timeout_s = int(apptainer_config["reward_timeout_s"])
    mode = os.environ.get("RELAX_ROLLOUT_MODE", "train")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    prompt = "\n".join(message["content"] for message in payload["messages"] if message["role"] == "user")
    r2e = payload["metadata"]["r2e"]
    sif = sif_dir / f"{r2e['docker_image'].replace('/', '__').replace(':', '__')}.sif"

    codex_home = session_dir / "codex-home"
    codex_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    compact = os.environ.get("CODEX_COMPACT", "0") == "1"
    _configure_codex(
        codex_home,
        compact=compact,
        context_window=int(os.environ.get("CODEX_CONTEXT_WINDOW", "32768")),
        compact_limit=int(os.environ.get("CODEX_COMPACT_LIMIT", "16384")),
    )
    verifier_log = session_dir / "verifier.log"
    connection = nullcontext((os.environ["RELAX_BASE_URL"].rstrip("/") + "/v1", None))
    if compact:
        from recording import recording_proxy

        connection = recording_proxy(os.environ["RELAX_BASE_URL"])
    with connection as (base_url, recorder):
        command = [
            apptainer,
            "exec",
            "--memory",
            apptainer_config["memory"],
            "--memory-swap",
            apptainer_config["memory_swap"],
            "--pids-limit",
            str(apptainer_config["pids_limit"]),
            "--contain",
            "--pid",
            "--cleanenv",
            "--no-home",
            "--no-mount",
            "/etc/localtime",
            f"--bind={wheel_dir}:/r2e_wheels:ro",
            f"--bind={runtime_dir}:{apptainer_config['runtime_mount']}:ro",
            f"--bind={session_dir}:/relax-session",
            "--writable-tmpfs",
            "--pwd",
            "/testbed",
            f"--env=RELAX_SESSION_ID={os.environ['RELAX_SESSION_ID']}",
            f"--env=RELAX_CODEX_BASE_URL={base_url}",
            "--env=CODEX_HOME=/relax-session/codex-home",
            f"--env=CODEX_R2E_RUNTIME_MOUNT={apptainer_config['runtime_mount']}",
            "--env=CODEX_R2E_TEST_LOG=/relax-session/verifier.log",
            f"--env=CODEX_R2E_TEST_SCRIPT={apptainer_config['test_script'][mode]}",
            f"--env=CODEX_R2E_TIMEOUT={timeout_s}",
            f"--env=CODEX_R2E_VERIFIER_TIMEOUT={reward_timeout_s}",
            "--env=PATH=/root/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--env=OMP_NUM_THREADS=2",
            "--env=MKL_NUM_THREADS=2",
            "--env=OPENBLAS_NUM_THREADS=2",
            str(sif),
            "bash",
            "-lc",
            "set -e\n" + apptainer_config["setup"][mode] + "\n" + apptainer_config["command"],
        ]
        with tempfile.TemporaryFile() as log_file:
            result = subprocess.run(
                command,
                input=prompt,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s + reward_timeout_s + 300,
            )
            if result.returncode:
                size = os.fstat(log_file.fileno()).st_size
                log_file.seek(max(size - 4000, 0))
                tail = log_file.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Codex container exited with code {result.returncode}: {tail}")
    if mode == "eval":
        reward = _score_swebench(r2e, verifier_log)
    else:
        reward = _reward(r2e["expected_output_json"], verifier_log.read_text(encoding="utf-8"))
    records = [{"reward": reward}]
    if recorder is not None:
        records = recorder.exports(reward)
        if mode == "eval":
            records = records[-1:]  # Keep one sample per Eval Session so task-level metric denominators stay correct.
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    temporary.replace(output_path)


if __name__ == "__main__":
    main()
