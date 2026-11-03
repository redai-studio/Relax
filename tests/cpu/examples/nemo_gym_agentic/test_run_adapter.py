# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
from relax_nemo_gym_example.app.protocol import InterruptPolicy
from relax_nemo_gym_example.service.run_adapter import (
    HttpNemoGymRunAdapter,
    TrialRunContext,
    _build_run_payload,
)

from .test_gateway_service import _request, _settings


def test_run_payload_injects_rollout_correlation_without_mutating_task():
    request = _request("request-one")
    original = request.to_payload()["environment"]["task"]
    request.task["metadata"]["responses_create_params"] = {
        "metadata": {"instance_dict": '{"instance_id":"task-one"}'},
        "temperature": 0.7,
    }
    original = request.to_payload()["environment"]["task"]

    payload = _build_run_payload(request, rollout_id="capability-0")

    assert payload["responses_create_params"]["input"] == original["messages"]
    assert payload["responses_create_params"]["metadata"] == {"instance_dict": '{"instance_id":"task-one"}'}
    assert payload["responses_create_params"]["temperature"] == 0.7
    assert payload["_ng_task_index"] == "capability"
    assert payload["_ng_rollout_index"] == "0"
    assert request.task == original


def test_run_payload_restores_empty_gym_input_for_r2e_task():
    request = _request("request-r2e")
    request.task["messages"] = [{"role": "user", "content": "Fix the route parser."}]
    request.task["metadata"]["responses_create_params"] = {
        "input": [],
        "metadata": {"instance_dict": '{"instance_id":"task-r2e"}'},
    }

    payload = _build_run_payload(request, rollout_id="r2e-0")

    assert payload["responses_create_params"]["input"] == []
    assert payload["responses_create_params"]["metadata"] == {"instance_dict": '{"instance_id":"task-r2e"}'}


async def test_http_adapter_normalizes_gym_result_and_checks_readiness():
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "reward": 0.5,
                "response": {
                    "output": [
                        {"type": "function_call"},
                        {"type": "function_call_output"},
                        {"type": "message"},
                    ]
                },
            },
        )

    spec = next(iter(_settings().environments.values()))
    adapter = HttpNemoGymRunAdapter([spec], transport=httpx.MockTransport(handler))
    try:
        assert await adapter.ready() is True
        handle = await adapter.start(
            TrialRunContext(
                request=_request("request-one"),
                spec=spec,
                rollout_id="capability-0",
            )
        )
        result = await handle.wait()

        assert result.reward == 0.5
        assert result.metrics == {
            "model_output_items": 3,
            "tool_calls": 1,
            "tool_outputs": 1,
            "assistant_messages": 1,
        }
        run_request = next(request for request in seen if request.method == "POST")
        assert json.loads(run_request.content)["_ng_task_index"] == "capability"
    finally:
        await adapter.close()


async def test_http_adapter_persists_opt_in_r2e_artifacts(tmp_path):
    source_dir = tmp_path / "source"
    trajectory_dir = source_dir / "trajectories" / "instance-one"
    logs_dir = source_dir / "apptainer_logs"
    private_dir = source_dir / "eval_private"
    trajectory_dir.mkdir(parents=True)
    logs_dir.mkdir()
    private_dir.mkdir()
    (source_dir / "nemo_gym_metrics.json").write_text(
        json.dumps({"patch_exists": False, "resolved": False}),
        encoding="utf-8",
    )
    (trajectory_dir / "output.jsonl").write_text(
        json.dumps(
            {
                "test_result": {
                    "git_patch": "",
                    "skipped": True,
                    "skip_reason": "completions_exist_no_result",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "agent.log").write_text("Failed to get git diff (None)\n", encoding="utf-8")
    (private_dir / "ground_truth.json").write_text('{"patch":"private"}\n', encoding="utf-8")

    def handler(_):
        return httpx.Response(
            200,
            json={
                "reward": 0.0,
                "resolved": False,
                "patch_exists": False,
                "model_patch": None,
                "instance_config": {"persistent_dir": str(source_dir)},
                "response": {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "str_replace_editor",
                            "arguments": '{"command":"str_replace"}',
                        },
                        {
                            "type": "function_call_output",
                            "output": "No replacement was performed",
                        },
                    ]
                },
            },
        )

    request = _request("request-artifact")
    request.metadata["capture_artifacts"] = True
    spec = next(iter(_settings().environments.values()))
    artifact_root = tmp_path / "artifacts"
    adapter = HttpNemoGymRunAdapter(
        [spec],
        transport=httpx.MockTransport(handler),
        artifact_root=artifact_root,
    )
    try:
        handle = await adapter.start(
            TrialRunContext(
                request=request,
                spec=spec,
                rollout_id="capability-0",
            )
        )
        result = await handle.wait()

        assert result.artifact_ref is not None
        artifact_path = Path(result.artifact_ref)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact["evaluation"] == {
            "resolved": False,
            "patch_exists": False,
            "model_patch": None,
        }
        assert artifact["openhands_result"]["skip_reason"] == "completions_exist_no_result"
        assert artifact["response_output"][0]["name"] == "str_replace_editor"
        assert (artifact_path.parent / "apptainer_logs" / "agent.log").is_file()
        assert not (artifact_path.parent / "eval_private").exists()
        assert result.metrics["artifact_captured"] is True
        assert result.metrics["patch_exists"] is False
        assert result.metrics["resolved"] is False
        assert os.stat(artifact_path).st_mode & 0o777 == 0o600
    finally:
        await adapter.close()


async def test_http_adapter_does_not_claim_cleanup_without_abort_contract():
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        await asyncio.Event().wait()

    spec = next(iter(_settings().environments.values()))
    assert spec.interrupt_policy is InterruptPolicy.PROTECTED
    adapter = HttpNemoGymRunAdapter([spec], transport=httpx.MockTransport(handler))
    try:
        handle = await adapter.start(
            TrialRunContext(
                request=_request("request-one"),
                spec=spec,
                rollout_id="capability-0",
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        cleanup = await handle.abort()

        assert cleanup.confirmed is False
        assert cleanup.error_code == "cleanup_unverified"
    finally:
        await adapter.close()
