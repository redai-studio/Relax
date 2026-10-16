# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Recipe-local ownership of retained rollout sandboxes and grader jobs."""

import json
import secrets
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Candidate:
    group_id: str
    task: str
    env: Any
    tmp_dir: Path
    evidence: dict
    expires_at: float
    lock: Any = field(default_factory=threading.Lock)


@dataclass
class GradingJob:
    session_ids: list[str]
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    cancel: threading.Event = field(default_factory=threading.Event)
    status: str = "running"
    result: dict | None = None
    error: str = ""
    expires_at: float = 0


class SandboxBridge:
    def __init__(
        self,
        work_dir: Path,
        command: list[str],
        run_process: Callable,
        *,
        concurrency: int = 4,
        retention_seconds: float = 14400,
        grading_seconds: float = 1800,
        command_seconds: int = 120,
        max_retained_groups: int = 32,
    ) -> None:
        self.work_dir = work_dir
        self.command = command
        self.run_process = run_process
        self.retention_seconds = retention_seconds
        self.grading_seconds = grading_seconds
        self.command_seconds = command_seconds
        self.max_retained_groups = max_retained_groups
        self.candidates: dict[str, Candidate] = {}
        self.jobs: dict[str, GradingJob] = {}
        self.released_groups: dict[str, float] = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="gagar")
        self.closed = False

    def retain(self, session_id: str, **kwargs: Any) -> bool:
        with self.lock:
            if self.closed:
                raise RuntimeError("grader bridge is shutting down")
            if kwargs["group_id"] in self.released_groups:
                return False
            groups = {candidate.group_id for candidate in self.candidates.values()}
            if kwargs["group_id"] not in groups and len(groups) >= self.max_retained_groups:
                # Never hold a rollout worker waiting for retention capacity:
                # the remaining siblings must finish to reach group RM.
                return False
            self.candidates[session_id] = Candidate(expires_at=time.monotonic() + self.retention_seconds, **kwargs)
            return True

    def start(self, payload: dict) -> dict:
        group_id = payload["group_id"]
        session_ids = payload["session_ids"]
        if not session_ids or len(set(session_ids)) != len(session_ids):
            raise ValueError("a grading group needs distinct candidate sessions")
        with self.lock:
            if self.closed:
                raise RuntimeError("grader bridge is shutting down")
            if group_id in self.jobs:
                job = self.jobs[group_id]
                if job.session_ids != session_ids:
                    raise ValueError("grading group roster changed")
                return self.status(group_id)
            if group_id in self.released_groups:
                raise ValueError("grading group was already released")
            candidates = [self.candidates[sid] for sid in session_ids]
            if any(c.group_id != group_id or c.task != candidates[0].task for c in candidates):
                raise ValueError("candidate is outside this task group")
            if any(c.expires_at <= time.monotonic() for c in candidates):
                raise ValueError("candidate sandbox lease expired")
            job = GradingJob(session_ids=session_ids, expires_at=time.monotonic() + self.grading_seconds)
            self.jobs[group_id] = job
            manifest = {
                "group_id": group_id,
                "session_ids": session_ids,
                "bridge_url": payload["bridge_url"],
                "token": job.token,
                "genrm_url": payload["genrm_url"],
                "task": candidates[0].task,
                "candidates": [{"rollout_n": i, **candidate.evidence} for i, candidate in enumerate(candidates)],
            }
            self.pool.submit(self._run, group_id, job, manifest)
            return self.status(group_id)

    def status(self, group_id: str) -> dict:
        with self.lock:
            job = self.jobs[group_id]
            return {
                "status": job.status,
                "result": job.result,
                "error": job.error,
                "timeout_seconds": self.grading_seconds,
            }

    def execute(self, group_id: str, token: str, action: dict) -> dict:
        # All command execution resolves through this immutable roster. There
        # is deliberately no endpoint, instance name or host-shell fallback.
        rollout_n = action.get("rollout_n")
        command = action.get("command")
        if (
            set(action) != {"rollout_n", "command"}
            or type(rollout_n) is not int
            or not isinstance(command, str)
            or not command.strip()
        ):
            raise ValueError("bash requires an integer rollout_n and a nonempty command")
        with self.lock:
            job = self.jobs[group_id]
            if not secrets.compare_digest(token, job.token) or job.cancel.is_set() or job.status != "running":
                raise ValueError("invalid or inactive grading lease")
            if job.expires_at <= time.monotonic() or not 0 <= rollout_n < len(job.session_ids):
                raise ValueError("expired lease or unknown rollout_n")
            candidate = self.candidates[job.session_ids[rollout_n]]
        with candidate.lock:
            if job.cancel.is_set() or candidate.env is None:
                raise ValueError("candidate sandbox has been released")
            result = candidate.env.execute({"command": command}, timeout=self.command_seconds, check_finished=False)
            # Test output can be enormous; frozen evidence is kept separately.
            result["output"] = str(result.get("output", ""))[-40000:]
            return result

    def _run(self, group_id: str, job: GradingJob, manifest: dict) -> None:
        directory = self.work_dir / uuid.uuid4().hex
        input_path = directory / "input.json"
        output_path = directory / "result.json"
        try:
            directory.mkdir(parents=True)
            input_path.write_text(json.dumps(manifest, ensure_ascii=False))
            input_path.chmod(0o600)
            if job.cancel.is_set() or job.expires_at <= time.monotonic():
                raise RuntimeError("grading cancelled")
            incomplete = [c["rollout_n"] for c in manifest["candidates"] if c["reward"] and c.get("capture_error")]
            if incomplete:
                raise RuntimeError(f"passing candidates have incomplete evidence: {incomplete}")
            with (directory / "process.log").open("w") as log:
                self.run_process(
                    [*self.command, "--input", str(input_path), "--output", str(output_path)],
                    timeout=self.grading_seconds,
                    cancel_event=job.cancel,
                    check=True,
                    stdout=log,
                    stderr=-2,
                )
            result = json.loads(output_path.read_text())
            weights = result["weights"]
            if len(weights) != len(job.session_ids) or any(
                type(w) not in (int, float) or not 0 <= w <= 1 for w in weights
            ):
                raise ValueError("grader must return one weight in [0, 1] per candidate")
            corrected = result.get("rewards", [c["reward"] for c in manifest["candidates"]])
            original = [c["reward"] for c in manifest["candidates"]]
            if len(corrected) != len(original) or any(r not in (0, 1) or r > o for r, o in zip(corrected, original)):
                raise ValueError("grader may only correct an outcome from 1 to 0")
            with self.lock:
                job.result = {**result, "rewards": corrected}
                job.status = "completed" if not job.cancel.is_set() else "cancelled"
        except Exception as exc:
            summary_path = directory / "summary.json"
            detail = ""
            try:
                summary = json.loads(summary_path.read_text())
                detail = (
                    f"; exit_status={summary['exit_status']}; api_calls={summary['api_calls']}; {summary['error']}"
                )
            except (OSError, ValueError, KeyError, TypeError):
                # A killed grader may leave a partial diagnostic file.
                detail = ""
            with self.lock:
                job.error = f"{type(exc).__name__}: {exc}{detail}; audit_dir={directory}"
                job.status = "cancelled" if job.cancel.is_set() else "failed"
        finally:
            # Keep the evidence/trajectory for review, but never persist the
            # short-lived bridge capability in the grading manifest.
            try:
                manifest.pop("token", None)
                if input_path.exists():
                    input_path.write_text(json.dumps(manifest, ensure_ascii=False))
            finally:
                self.release(group_id)

    def release(self, group_id: str) -> None:
        with self.lock:
            self.released_groups[group_id] = time.monotonic() + self.retention_seconds
            job = self.jobs.get(group_id)
            if job is not None:
                job.cancel.set()
                if job.status == "running":
                    job.status = "cancelled"
                job.expires_at = time.monotonic() + 300
            candidates = [(sid, c) for sid, c in self.candidates.items() if c.group_id == group_id]
        errors = []
        for sid, candidate in candidates:
            try:
                with candidate.lock:
                    if candidate.env is not None:
                        # On failure, retain ownership and scratch for retry.
                        candidate.env.cleanup()
                        candidate.env = None
                        shutil.rmtree(candidate.tmp_dir, ignore_errors=True)
                with self.lock:
                    self.candidates.pop(sid, None)
            except Exception as exc:
                errors.append(f"{sid}: {exc}")
        if errors:
            raise RuntimeError("Sandbox release failed: " + "; ".join(errors))

    def reap(self) -> None:
        now = time.monotonic()
        with self.lock:
            expired = {
                c.group_id
                for c in self.candidates.values()
                if ((job := self.jobs.get(c.group_id)) is None and c.expires_at <= now)
                or (job is not None and (job.cancel.is_set() or job.expires_at <= now))
                or c.group_id in self.released_groups
            }
        errors = []
        for group_id in expired:
            try:
                self.release(group_id)
            except RuntimeError as exc:
                errors.append(str(exc))
        with self.lock:
            for group_id, job in list(self.jobs.items()):
                if job.status != "running" and job.expires_at <= now:
                    self.jobs.pop(group_id)
            for group_id, deadline in list(self.released_groups.items()):
                if deadline <= now:
                    self.released_groups.pop(group_id)
        if errors:
            raise RuntimeError("; ".join(errors))

    def shutdown(self) -> None:
        with self.lock:
            self.closed = True
            groups = {c.group_id for c in self.candidates.values()}
        errors = []
        for group_id in groups:
            try:
                self.release(group_id)
            except RuntimeError as exc:
                errors.append(str(exc))
        self.pool.shutdown(wait=True, cancel_futures=True)
        if errors:
            raise RuntimeError("; ".join(errors))


def register_routes(app: Any, bridge: SandboxBridge) -> None:
    from flask import jsonify, request

    @app.post("/gagar/groups")
    def start():
        try:
            return jsonify(bridge.start(request.get_json(force=True)))
        except (KeyError, ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/gagar/groups/<group_id>")
    def status(group_id: str):
        try:
            return jsonify(bridge.status(group_id))
        except KeyError:
            return jsonify({"error": "unknown grading group"}), 404

    @app.post("/gagar/groups/<group_id>/bash")
    def execute(group_id: str):
        try:
            token = request.headers.get("Authorization", "").removeprefix("Bearer ")
            return jsonify(bridge.execute(group_id, token, request.get_json(force=True)))
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/gagar/groups/<group_id>/release")
    def release(group_id: str):
        bridge.release(group_id)
        return "", 204
