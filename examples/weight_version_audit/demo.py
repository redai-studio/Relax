# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU reproduction using real Session commit/export and training
serialization.

The model backend and successful weight-load notifications are controlled
fixtures. No Ray cluster, downloaded model, or GPU training is started.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.pipeline.runtime import BackendGenerateResult
from relax.agentic.session.service import AgenticSessionShard, ResidentGroup, _SessionRecord, _SessionResultCell
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest
from relax.utils import weight_version_audit as audit
from relax.utils.utils import convert_samples_to_train_data


class Tokenizer:
    def decode(self, ids, **_kwargs):
        return "".join(chr(value) for value in ids)


async def run_demo(scenario: str = "complete") -> dict:
    """Order tool wait and two updates with events, then consume a real CPU
    loss."""
    previous = os.environ.get("RELAX_WEIGHT_AUDIT_DIR")
    with tempfile.TemporaryDirectory(prefix="relax-weight-audit-") as directory:
        os.environ["RELAX_WEIGHT_AUDIT_DIR"] = directory
        try:
            return await _run(directory, scenario)
        finally:
            if previous is None:
                os.environ.pop("RELAX_WEIGHT_AUDIT_DIR", None)
            else:
                os.environ["RELAX_WEIGHT_AUDIT_DIR"] = previous


async def _run(directory: str, scenario: str) -> dict:
    publisher = audit.PublicationLedger(directory, "demo-policy")
    for _ in range(10):
        first_version = publisher.publish()
    current_version = first_version
    events = []
    tool_started, updates_done = asyncio.Event(), asyncio.Event()
    forest = SessionForest.create_empty(session_id="demo-session", index=0, group_index=0)
    prompt = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "pppp"}],
        train_token_delta=[112] * 4,
        rollout_token_delta=[112] * 4,
    )
    group = ResidentGroup(rollout_mode="train", group_id="demo-group", result_cells={}, sessions=[])
    session = _SessionRecord(
        group=group, session_id="demo-session", session_sampling_params={}, result_cell=_SessionResultCell()
    )
    session.forest = forest
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = SimpleNamespace(
        agentic_reasoning_parser=None, agentic_tool_call_parser=None, use_rollout_routing_replay=False
    )
    shard._generation_backend = SimpleNamespace(tokenizer=Tokenizer())

    def commit(parent, turn, text, version):
        request = InflightRequest(
            request_id=f"request-{turn}",
            parent_state_hash=parent,
            rollout_id=0,
            kind=RequestKind.FRESH,
            abort_count=0,
            waiter=asyncio.get_running_loop().create_future(),
            wall_started_at=time.monotonic(),
            sampling_params={"max_new_tokens": 8},
        )
        meta = {"finish_reason": {"type": "stop"}, "prompt_tokens": 4}
        if version is not None:
            meta["weight_versions"] = [{"start": 0, "end": len(text), "version": version}]
            meta["weight_version"] = version
        shard._apply_generate_result(
            request,
            BackendGenerateResult(
                new_tokens=[ord(c) for c in text],
                new_log_probs=[-0.5] * len(text),
                finish_type="stop",
                meta_info=meta,
                elapsed=0.1,
            ),
        )
        request.pending_status = "completed"
        shard._terminal_response_locked(session=session, ir=request, finish_type="stop")
        events.append({"event": "turn_committed", "turn": turn})
        return session.resp_state_hash_by_request_id[request.request_id]

    async def agent():
        leaf = commit(prompt.state_hash, 0, "abc", first_version)
        events.append({"event": "tool_wait_begin"})
        tool_started.set()
        await updates_done.wait()
        events.append({"event": "tool_wait_end"})
        observation = forest.append_obs(
            parent_state_hash=leaf,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "tool", "tool_call_id": "tool-0", "content": "ok"}],
            train_token_delta=[111, 107],
            rollout_token_delta=[111, 107],
        )
        if scenario == "missing":
            version = None
        elif scenario == "lagging":
            version = first_version
        else:
            version = current_version
        leaf = commit(observation.state_hash, 1, "de", version)
        return forest.build_sample(leaf_state_hash=leaf, tokenizer=Tokenizer())

    async def update_during_tool_wait():
        nonlocal current_version
        await tool_started.wait()
        for _ in range(2):
            current_version = publisher.publish()
            events.append({"event": "publish_during_tool_wait", "published_seq": publisher.published_seq})
        updates_done.set()

    sample, _ = await asyncio.gather(agent(), update_during_tool_wait())
    sample.reward = 1.0
    restored = TrainingFieldArtifact.from_sample(sample).to_sample()
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        reward_key=None,
        rewards_normalization=False,
        advantage_estimator="grpo",
        use_opd=False,
        multimodal_keys=None,
        debug_train_only=False,
    )
    data = convert_samples_to_train_data(args, [restored])
    data = dict(pickle.loads(pickle.dumps(data)).items())
    audit.prepare_batch(data)
    publisher.publish()
    events.append({"event": "actor_consumes", "published_seq": publisher.published_seq})
    step = audit.AuditStep("demo-consumption")
    losses = torch.ones(sample.response_length, requires_grad=True)
    loss = (losses * torch.tensor(sample.loss_mask)).sum()
    step.observe(0, data[audit.PREPARED_FIELD])
    loss.backward()
    summary = step.commit(rank=0)
    report = audit.audit_row(data[audit.PREPARED_FIELD][0], audit.read_frontier())
    assert summary["eligible_tokens"] == 5
    assert report["post_generation_published_advance"] == 1
    assert report["token_weighted_age"] == (2.2 if scenario == "complete" else 3)
    return {
        "scope": "CPU Session/serialization/loss reproduction with controlled backend and publisher",
        "scenario": scenario,
        "events": events,
        "summary": summary,
        "sample": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("complete", "missing", "lagging"), default="complete")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_demo(args.scenario))
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
