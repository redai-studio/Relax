# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run-handle abstraction and HTTP adapter for a long-lived NeMo Gym agent."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from http.cookiejar import Cookie, CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..app.protocol import RewardValue, TrialRequest, TrialStatus
from .config import EnvironmentSpec
from .verbose_logging import log_verbose_payload


@dataclass(frozen=True)
class AdapterResult:
    status: TrialStatus
    reward: RewardValue = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_ref: str | None = None
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class CleanupResult:
    confirmed: bool
    error_code: str | None = None


@dataclass(frozen=True)
class TrialRunContext:
    request: TrialRequest
    spec: EnvironmentSpec
    rollout_id: str


class RunHandle(Protocol):
    async def wait(self) -> AdapterResult: ...

    async def abort(self) -> CleanupResult: ...

    async def force_cleanup(self) -> CleanupResult: ...

    async def probe_cleanup(self) -> CleanupResult: ...


class RunAdapter(Protocol):
    async def start(self, context: TrialRunContext) -> RunHandle: ...

    async def ready(self) -> bool: ...

    async def close(self) -> None: ...


class _RejectAllCookiesPolicy(DefaultCookiePolicy):
    def set_ok(self, cookie: Cookie, request: Any) -> bool:
        return False


class HttpNemoGymRunAdapter:
    """Calls a pre-started NeMo Gym agent's ``/run`` endpoint.

    Upstream NeMo Gym does not yet expose a generic run cancellation contract.
    Unless a deployment supplies an explicit abort endpoint that guarantees
    cleanup, aborting the local HTTP request is reported as unconfirmed.
    """

    def __init__(
        self,
        specs: Iterable[EnvironmentSpec],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        artifact_root: Path | None = None,
    ) -> None:
        self._readiness_urls = tuple(sorted({url for spec in specs for url in spec.readiness_urls}))
        self._artifact_root = artifact_root
        self._client = httpx.AsyncClient(
            timeout=None,
            transport=transport,
            trust_env=False,
            cookies=CookieJar(policy=_RejectAllCookiesPolicy()),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
        )

    async def start(self, context: TrialRunContext) -> RunHandle:
        payload = _build_run_payload(context.request, rollout_id=context.rollout_id)
        log_verbose_payload(
            "request",
            payload,
            route="swe_agents/run",
            request_id=context.request.request_id,
            rollout_id=context.rollout_id,
        )
        task = asyncio.create_task(
            self._client.post(
                f"{context.spec.agent_url}/run",
                json=payload,
            )
        )
        return _HttpRunHandle(
            task=task,
            client=self._client,
            spec=context.spec,
            rollout_id=context.rollout_id,
            request_id=context.request.request_id,
            capture_artifacts=context.request.metadata.get("capture_artifacts") is True,
            artifact_root=self._artifact_root,
        )

    async def ready(self) -> bool:
        try:
            responses = await asyncio.gather(
                *(self._client.get(url, timeout=2.0) for url in self._readiness_urls),
            )
        except httpx.RequestError:
            return False
        return bool(responses) and all(response.is_success for response in responses)

    async def close(self) -> None:
        await self._client.aclose()


@dataclass
class _HttpRunHandle:
    task: asyncio.Task[httpx.Response]
    client: httpx.AsyncClient
    spec: EnvironmentSpec
    rollout_id: str
    request_id: str
    capture_artifacts: bool
    artifact_root: Path | None

    async def wait(self) -> AdapterResult:
        response = await self.task
        try:
            payload = response.json()
        except ValueError as exc:
            log_verbose_payload(
                "response",
                {"raw_text": response.text},
                route="swe_agents/run",
                rollout_id=self.rollout_id,
                status=response.status_code,
            )
            response.raise_for_status()
            raise RuntimeError("NeMo Gym agent returned a non-JSON response") from exc
        log_verbose_payload(
            "response",
            payload,
            route="swe_agents/run",
            rollout_id=self.rollout_id,
            status=response.status_code,
        )
        response.raise_for_status()
        if not isinstance(payload, dict):
            raise RuntimeError("NeMo Gym agent response must be a JSON object")
        result = _normalize_run_result(payload, environment=self.spec.environment)
        if not self.capture_artifacts:
            return result
        if self.artifact_root is None:
            return _with_artifact_capture_error(result, "artifact_root_not_configured")
        try:
            artifact_ref = await asyncio.to_thread(
                _persist_run_artifacts,
                payload,
                artifact_root=self.artifact_root,
                request_id=self.request_id,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return _with_artifact_capture_error(result, type(exc).__name__)
        return AdapterResult(
            status=result.status,
            reward=copy.deepcopy(result.reward),
            metrics={**copy.deepcopy(result.metrics), "artifact_captured": True},
            artifact_ref=artifact_ref,
            error=copy.deepcopy(result.error),
        )

    async def abort(self) -> CleanupResult:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        return await self._post_cleanup(self.spec.abort_url)

    async def force_cleanup(self) -> CleanupResult:
        return await self._post_cleanup(self.spec.force_cleanup_url)

    async def probe_cleanup(self) -> CleanupResult:
        if self.spec.cleanup_probe_url is None:
            return CleanupResult(confirmed=False, error_code="cleanup_unverified")
        url = self.spec.cleanup_probe_url.format(rollout_id=self.rollout_id)
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return CleanupResult(confirmed=False, error_code="cleanup_probe_failed")
        if isinstance(payload, dict) and payload.get("clean") is True:
            return CleanupResult(confirmed=True)
        return CleanupResult(confirmed=False, error_code="cleanup_unverified")

    async def _post_cleanup(self, url_template: str | None) -> CleanupResult:
        if url_template is None:
            return CleanupResult(confirmed=False, error_code="cleanup_unverified")
        url = url_template.format(rollout_id=self.rollout_id)
        try:
            response = await self.client.post(url)
            response.raise_for_status()
        except httpx.HTTPError:
            return CleanupResult(confirmed=False, error_code="cleanup_failed")
        # An acknowledgement only proves that cleanup was accepted. The
        # separate probe owns the proof that remote resources disappeared.
        return CleanupResult(confirmed=False, error_code="cleanup_unverified")


def _build_run_payload(request: TrialRequest, *, rollout_id: str) -> dict[str, Any]:
    task = copy.deepcopy(request.task)
    if isinstance(task.get("responses_create_params"), dict):
        payload = task
    else:
        messages = task.pop("messages", [])
        metadata = task.pop("metadata", {})
        if not isinstance(messages, list) or not isinstance(metadata, dict):
            raise ValueError("Relax task must contain messages and metadata objects")
        responses_create_params: dict[str, Any] = {"input": copy.deepcopy(messages)}
        preserved_responses_params = metadata.pop("responses_create_params", {})
        if not isinstance(preserved_responses_params, dict):
            raise ValueError("Relax task metadata.responses_create_params must be an object")
        responses_create_params.update(copy.deepcopy(preserved_responses_params))
        for key in ("tools", "parallel_tool_calls"):
            if key in metadata:
                responses_create_params[key] = copy.deepcopy(metadata[key])
        if metadata.get("developer_message"):
            responses_create_params["instructions"] = metadata["developer_message"]
        configured_params = request.generation.get("responses_create_params", {})
        if isinstance(configured_params, dict):
            responses_create_params.update(copy.deepcopy(configured_params))
        payload = {
            "responses_create_params": responses_create_params,
            **copy.deepcopy(task),
            **{
                key: copy.deepcopy(value)
                for key, value in metadata.items()
                if key not in {"tools", "parallel_tool_calls", "developer_message"}
            },
        }

    task_index, rollout_index = rollout_id.rsplit("-", maxsplit=1)
    payload["_ng_task_index"] = task_index
    payload["_ng_rollout_index"] = rollout_index
    return payload


def _normalize_run_result(payload: dict[str, Any], environment: str | None = None) -> AdapterResult:
    reward = payload.get("reward")
    status = TrialStatus.TRUNCATED if payload.get("truncated") else TrialStatus.COMPLETED
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    response = payload.get("response")
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            metrics = {
                **metrics,
                "model_output_items": len(output),
                "tool_calls": sum(
                    1 for item in output if isinstance(item, dict) and item.get("type") == "function_call"
                ),
                "tool_outputs": sum(
                    1 for item in output if isinstance(item, dict) and item.get("type") == "function_call_output"
                ),
                "assistant_messages": sum(
                    1 for item in output if isinstance(item, dict) and item.get("type") == "message"
                ),
            }
    for field_name in (
        "resolved",
        "patch_exists",
        "agent_error_kind",
        "agent_timed_out",
        "eval_timed_out",
        "oom_killed",
        "eval_oom_killed",
    ):
        if field_name in payload:
            metrics[field_name] = copy.deepcopy(payload[field_name])
    # r2e_gym partial credit: Gym returns reward 0.0 both when the agent never
    # produced a runnable patch and when it did but pytest failed. patch_exists
    # is the strongest signal we have that the eval harness moved past patch
    # apply into test execution — reward that intermediate progress with 0.1
    # so GRPO can distinguish it from a total no-op.
    if (
        environment == "r2e_gym"
        and isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and float(reward) == 0.0
        and metrics.get("patch_exists") is True
    ):
        reward = 0.1
    artifact_ref = payload.get("artifact_ref")
    if artifact_ref is not None and not isinstance(artifact_ref, str):
        artifact_ref = None
    return AdapterResult(
        status=status,
        reward=copy.deepcopy(reward),
        metrics=copy.deepcopy(metrics),
        artifact_ref=artifact_ref,
    )


def _with_artifact_capture_error(result: AdapterResult, error_code: str) -> AdapterResult:
    return AdapterResult(
        status=result.status,
        reward=copy.deepcopy(result.reward),
        metrics={
            **copy.deepcopy(result.metrics),
            "artifact_captured": False,
            "artifact_capture_error": error_code,
        },
        artifact_ref=result.artifact_ref,
        error=copy.deepcopy(result.error),
    )


def _persist_run_artifacts(
    payload: dict[str, Any],
    *,
    artifact_root: Path,
    request_id: str,
) -> str:
    artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(artifact_root, 0o700)
    artifact_id = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    destination = artifact_root / artifact_id
    manifest_path = destination / "artifact.json"
    if manifest_path.is_file():
        return str(manifest_path)

    temporary = Path(tempfile.mkdtemp(prefix=".capture-", dir=artifact_root))
    try:
        source_dir = _artifact_source_dir(payload)
        if source_dir is not None:
            _copy_selected_artifact_files(source_dir, temporary)

        response = payload.get("response")
        response_output = response.get("output", []) if isinstance(response, dict) else []
        if not isinstance(response_output, list):
            response_output = []
        evaluation_fields = (
            "resolved",
            "patch_exists",
            "model_patch",
            "agent_error_kind",
            "agent_timed_out",
            "eval_timed_out",
            "oom_killed",
            "eval_oom_killed",
            "agent_peak_rss_mb",
            "eval_peak_rss_mb",
        )
        manifest = {
            "protocol_version": "relax-nemo-gym/artifact-v1",
            "request_id": request_id,
            "evaluation": {
                field_name: copy.deepcopy(payload[field_name])
                for field_name in evaluation_fields
                if field_name in payload
            },
            "openhands_result": _read_openhands_result(source_dir),
            "response_output": copy.deepcopy(response_output),
            "files": sorted(str(path.relative_to(temporary)) for path in temporary.rglob("*") if path.is_file()),
        }
        manifest_path_in_temporary = temporary / "artifact.json"
        manifest_path_in_temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        _restrict_artifact_permissions(temporary)
        if destination.exists():
            shutil.rmtree(temporary)
            return str(manifest_path)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return str(manifest_path)


def _artifact_source_dir(payload: dict[str, Any]) -> Path | None:
    instance_config = payload.get("instance_config")
    if not isinstance(instance_config, dict):
        return None
    persistent_dir = instance_config.get("persistent_dir")
    if not isinstance(persistent_dir, str):
        return None
    source_dir = Path(persistent_dir)
    if not source_dir.is_absolute() or not source_dir.is_dir():
        return None
    return source_dir


def _copy_selected_artifact_files(source_dir: Path, destination: Path) -> None:
    for filename in ("nemo_gym_metrics.json", "patch.diff"):
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)
    for directory_name in ("trajectories", "apptainer_logs", "eval-outputs"):
        source = source_dir / directory_name
        if source.is_dir():
            shutil.copytree(source, destination / directory_name)


def _read_openhands_result(source_dir: Path | None) -> dict[str, Any] | None:
    if source_dir is None:
        return None
    for output_path in sorted((source_dir / "trajectories").glob("*/output.jsonl")):
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    return None
                test_result = payload.get("test_result")
                return copy.deepcopy(test_result) if isinstance(test_result, dict) else None
    return None


def _restrict_artifact_permissions(root: Path) -> None:
    os.chmod(root, 0o700)
    for path in root.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o700)
        elif path.is_file():
            os.chmod(path, 0o600)
