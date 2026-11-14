# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise Session, publication and consumption auditing with a CPU backend.

Run in its own process: this example owns one local Ray actor and one Gloo
rank. The backend and Megatron topology are controlled; audit records are
produced by the real Session, publication, export and training hooks, never
filled by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist

from relax.agentic.observability import SessionObservation, ShardTrajectoryRecorder
from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.pipeline.runtime import BackendGenerateResult, EncodedMessages
from relax.agentic.session import service
from relax.agentic.session.admission_coordinator import AdmissionCoordinator
from relax.agentic.session.state import SessionForest
from relax.utils import distributed_utils
from relax.utils.agentic_obs import train_hooks
from relax.utils.agentic_obs.weight_loss import observe_policy_tokens
from relax.utils.agentic_obs.weight_publication import (
    begin_publication,
    commit_publication,
    consumption_snapshot,
    observe_install_result,
    publication_label,
)
from relax.utils.agentic_obs.weight_report import build_weight_report, read_records
from relax.utils.agentic_obs.writers import close_observation_writers
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class _CharacterCodec:
    """A deterministic text-only compiler, with one token per ASCII
    character."""

    apply_chat_template_kwargs: dict[str, Any] = {}
    eos_token_id = None

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return "".join(chr(token) for token in tokens)

    async def encode_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> EncodedMessages:
        tokens = [ord(char) for message in messages for char in message["content"]]
        return EncodedMessages(train_prompt_ids=tokens, backend_prompt_ids=tokens)

    encode_observation_delta = encode_messages


class _CpuBackend:
    """Generate from installed logits and return the corresponding install
    label."""

    def __init__(self) -> None:
        self.tokenizer = self.compiler = _CharacterCodec()
        self.logits = torch.zeros(128)
        self.label: str | None = None

    def install(self, logits: torch.Tensor, label: str) -> dict[str, bool]:
        self.logits = logits.detach().clone()
        self.label = label
        return {"success": True}

    async def generate(self, *, input_ids: list[int], sampling_params: dict, **kwargs: Any) -> BackendGenerateResult:
        count = sampling_params["max_new_tokens"]
        token = int(self.logits.argmax())
        logprob = float(self.logits.log_softmax(0)[token])
        return BackendGenerateResult(
            new_tokens=[token] * count,
            new_log_probs=[logprob] * count,
            finish_type="stop",
            meta_info={"weight_version": self.label, "prompt_tokens": len(input_ids)},
            elapsed=0.0,
        )

    async def abort_request(self, request_id: str) -> None:
        pass


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dump_agentic_obs=str(root.parent),
        agentic_obs_run_id=root.name,
        agentic_obs_train_generation_id="cpu-demo-actor",
        dump_agentic_obs_media=False,
        use_agentic_rollout=True,
        agentic_program_admission=False,
        agentic_admission_scope="all",
        agentic_admission_max_wait_s=5,
        agentic_admission_headroom=0.1,
        agentic_admission_pressure_threshold=0.9,
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_server_concurrency=1,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        agentic_session_lifecycle=False,
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        use_rollout_routing_replay=False,
        partial_rollout=False,
        fully_async=True,
        use_slime_router=False,
        rollout_max_context_len=128,
        dynamic_context_parallel=False,
        context_parallel_size=1,
        n_samples_per_prompt=1,
        rollout_batch_size=1,
        advantage_estimator="grpo",
        agentic_custom_advantage_path=None,
        reward_key=None,
    )


def _local_shard(args: argparse.Namespace, coordinator: Any, backend: _CpuBackend) -> tuple[Any, Any]:
    # Avoid the production constructor's model/agent-process launch, while using
    # the production scheduler, permits, Forest, response and export methods.
    shard = object.__new__(service.AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = args
    shard._generation_backend = backend
    shard._sglang_request_limiter = coordinator
    shard._admission_client = None
    shard._permit_cleanup_tasks = set()
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=0)
    shard._train_generation_open = True
    shard._state_revision = 0
    shard._state_changed = asyncio.Event()
    cell = service._SessionResultCell()
    group = service.ResidentGroup(
        "train", "demo-group", {"demo-session": cell}, [], ownership=service.GroupOwnership.RUNTIME
    )
    session = service._SessionRecord(
        group, "demo-session", {"max_new_tokens": 4}, cell, observation=SessionObservation()
    )
    session.forest = SessionForest.create_empty(session_id=session.session_id, group_index=0, index=0)
    group.sessions.append(session)
    shard._session_records = {session.session_id: session}
    shard._groups = {group.group_id: group}
    return shard, session


def _train(args: argparse.Namespace, sample: Any, policy: torch.Tensor) -> dict[str, float]:
    # The only Megatron replacement is rank/topology discovery. All records and
    # objective masks below go through the same hooks as a policy train step.
    mpu = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda **kwargs: True,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_data_parallel_rank=lambda **kwargs: 0,
        get_data_parallel_world_size=lambda **kwargs: 1,
    )
    mask = torch.tensor(sample.loss_mask, dtype=torch.bool)
    batch = {
        "sample_export_ids": [sample.sample_export_id],
        "agentic_group_ids": [sample.agentic_group_id],
        "response_lengths": [sample.response_length],
        "total_lengths": [len(sample.tokens)],
        "loss_masks": [mask],
    }
    targets = torch.tensor(sample.tokens[-sample.response_length :], dtype=torch.long)
    optimizer = torch.optim.SGD([policy], lr=0.01)
    before = policy.detach().clone()
    with patch.dict("sys.modules", {"megatron.core": SimpleNamespace(mpu=mpu)}):
        with train_hooks.observe_train_step(args, 0, 0):
            train_hooks.capture_training_input(batch)
            per_token = torch.nn.functional.cross_entropy(
                policy.expand(sample.response_length, -1), targets, reduction="none"
            )
            loss = per_token[mask].mean()
            observe_policy_tokens(args, batch, reducer_masks=batch["loss_masks"])
            loss.backward()
            optimizer.step()
            train_hooks.record_optimizer_result(True)
    assert torch.isfinite(loss) and not torch.equal(before, policy)
    return {"loss": float(loss.detach()), "parameter_update_norm": float((policy.detach() - before).norm())}


async def _scenario(args: argparse.Namespace, coordinator: Any) -> dict[str, Any]:
    backend = _CpuBackend()
    policy = torch.nn.Parameter(torch.zeros(128))
    publisher = SimpleNamespace(args=args, weight_version=0, _weight_audit_coordinator=coordinator)
    shard, session = _local_shard(args, coordinator, backend)
    events: list[dict[str, Any]] = []

    def publish(counter: int, character: str) -> None:
        publisher.weight_version = counter
        with torch.no_grad():
            policy.zero_()
            policy[ord(character)] = 4
        begin_publication(publisher)
        observe_install_result(publisher, backend.install(policy, publication_label(publisher)))
        commit_publication(publisher)
        version = consumption_snapshot(args)
        assert version is not None
        events.append({"event": "published", "version": version})

    tool_started, tool_done = asyncio.Event(), asyncio.Event()

    async def tool() -> str:
        events.append({"event": "tool_wait_started"})
        tool_started.set()
        await tool_done.wait()
        events.append({"event": "tool_wait_finished"})
        return "ok!"

    async def chat(messages: list[dict[str, Any]], count: int) -> dict[str, Any]:
        result = await shard.chat(
            session_id=session.session_id,
            messages=messages,
            tools=[],
            chat_template_kwargs={},
            max_completion_tokens=count,
            stop=None,
            seed=None,
        )
        assert "error" not in result, result
        return result

    async def agent() -> Any:
        messages = [{"role": "user", "content": "Use a tool then answer."}]
        first = await chat(messages, 4)
        assert first["message"]["content"] == "aaaa"
        events.append({"event": "first_turn_completed", "request_id": first["request_id"]})
        result = await tool()
        messages += [first["message"], {"role": "user", "content": result}]
        second = await chat(messages, 6)
        assert second["message"]["content"] == "cccccc"
        events.append({"event": "second_turn_completed", "request_id": second["request_id"]})
        transport = shard._build_session_export_transport(
            session=session, reward=1.0, metadata={}, output_records=(), finalize_events={}
        )
        sample = TrainingFieldArtifact(sample_payload=transport.exports[0]["sample_payload"]).to_sample()
        assert await shard._finish_session(session, transport) is None
        return sample

    async def update_during_tool() -> None:
        await tool_started.wait()
        publish(20, "b")
        publish(90, "c")
        tool_done.set()

    try:
        publish(10, "a")
        sample, _ = await asyncio.gather(agent(), update_during_tool())
        publish(100, "d")
        training = _train(args, sample, policy)
        assert not shard._session_records and not session.live_irs and not session.queued_irs
        assert await coordinator.weight_publication_snapshot.remote() == consumption_snapshot(args)
        return {"events": events, "training": training}
    finally:
        for active in tuple(shard._session_records.values()):
            await shard._finish_session(active, None)
        await shard._trajectory_recorder.aclose()


def run_demo(output: Path) -> dict[str, Any]:
    """Run the bounded CPU scenario and retain its raw records beside the
    report."""
    if ray.is_initialized() or dist.is_initialized():
        raise RuntimeError("Run this demo in a fresh process; it owns its Ray and Gloo runtimes")
    output = output.resolve()
    root = output.parent / f"{output.stem}-records"
    root.mkdir(parents=True, exist_ok=False)
    args = _args(root)
    try:
        ray.init(address="local", num_cpus=1, num_gpus=0, include_dashboard=False, log_to_driver=False)
        with TemporaryDirectory(prefix="weight-audit-gloo-") as rendezvous:
            dist.init_process_group(
                "gloo",
                init_method=(Path(rendezvous) / "store").as_uri(),
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=20),
            )
            distributed_utils.init_gloo_group()
            coordinator = AdmissionCoordinator.options(num_cpus=0).remote(args)
            try:
                evidence = asyncio.run(asyncio.wait_for(_scenario(args, coordinator), timeout=45))
            finally:
                ray.get(coordinator.shutdown.remote(), timeout=5)
    finally:
        train_hooks.close_train_appenders()
        close_observation_writers()
        if dist.is_initialized():
            dist.destroy_process_group()
            distributed_utils.GLOO_GROUP = None
        ray.shutdown()
    report = build_weight_report(
        read_records(root, "trajectories"), read_records(root, "stages"), read_records(root, "train")
    )
    assert report["complete"], report
    (sample,) = report["samples"]
    assert sample["session_id"] == "demo-session"
    assert sample["effective_tokens"] == 10
    assert sample["generation_span"] == 2 and sample["post_generation_lag"] == 1
    assert [segment["effective_tokens"] for segment in sample["segments"]] == [4, 6]
    assert [segment["version"]["ordinal"] for segment in sample["segments"]] == [1, 3]
    assert [segment["consumption_age"] for segment in sample["segments"]] == [3, 1]
    assert math.isclose(report["token_weighted_age"], 1.8)
    report["demo"] = {"records": str(root), **evidence}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("CPU weight audit passed; report=%s records=%s", output, root)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run_demo(parser.parse_args().output)


if __name__ == "__main__":
    main()
