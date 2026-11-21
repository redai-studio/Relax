# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU audit demo: a tool wait spans two simulated successful publications."""

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from relax.agentic.session.state import SessionForest
from relax.agentic.weight_audit import AUDIT_FIELD, PublicationLedger, diagnose
from relax.utils.logging_utils import get_logger
from relax.utils.utils import convert_samples_to_train_data


class _Tokenizer:
    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


async def run_demo() -> dict:
    ledger = PublicationLedger()
    forest = SessionForest.create_empty(
        session_id="tool-wait-demo", group_index=0, index=0, label=None, train_metadata=None, metadata={}
    )
    root = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "Use a tool, then answer."}],
        train_token_delta=[1],
        rollout_token_delta=[1],
    )

    def respond(parent, version, tokens, request_id):
        return forest.append_resp(
            parent_state_hash=parent.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[{"role": "assistant", "content": request_id}],
            token_delta=tokens,
            logprob_delta=[-0.5] * len(tokens),
            weight_version_spans=[
                {
                    "start": 0,
                    "end": len(tokens),
                    "weight_version": version,
                    "publication": ledger.snapshot(version),
                }
            ],
            generation_end_publication=ledger.snapshot(version),
            export_metadata_patch={"request_id": request_id},
        )

    ledger.record("demo-policy", 1, "10")
    first = respond(root, "10", [10] * 4, "turn-0-tool-call")
    tool_started, tool_finished = asyncio.Event(), asyncio.Event()

    async def tool():
        tool_started.set()
        await tool_finished.wait()
        return [20] * 3

    async def publish_during_tool():
        await tool_started.wait()
        ledger.record("demo-policy", 2, "11")
        ledger.record("demo-policy", 3, "12")
        tool_finished.set()

    tool_tokens, _ = await asyncio.gather(tool(), publish_during_tool())
    observation = forest.append_obs(
        parent_state_hash=first.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "tool result"}],
        train_token_delta=tool_tokens,
        rollout_token_delta=tool_tokens,
    )
    second = respond(observation, "12", [30] * 6, "turn-1-answer")
    sample = forest.build_sample(leaf_state_hash=second.state_hash, tokenizer=_Tokenizer())
    sample.reward = 1.0
    args = SimpleNamespace(
        use_agentic_rollout=True,
        custom_reward_post_process_path=None,
        reward_key=None,
        rewards_normalization=False,
        multimodal_keys=None,
        use_opd=False,
        debug_train_only=False,
    )
    batch = convert_samples_to_train_data(args, [sample])
    ledger.record("demo-policy", 4, "13")
    report = diagnose(list(batch[AUDIT_FIELD])[0], ledger.snapshot())
    assert [s["consumption_age"] for s in report["spans"]] == [3, 1]
    assert report["generation_version_span"] == 2
    assert report["post_generation_wait_lag"] == 1
    assert report["consumption_age_token_mean"] == 1.8
    assert report["version_missing_token_rate"] == 0
    return report


def summarize(reports: list[dict], unaudited_rows: int = 0) -> dict:
    tokens = sum(r["effective_tokens"] for r in reports)
    known = sum(r["known_age_tokens"] for r in reports)
    missing = sum(r["version_missing_tokens"] for r in reports)
    return {
        "audited_rows": len(reports),
        "unaudited_rows": unaudited_rows,
        "audited_effective_tokens": tokens,
        "known_age_token_mean": sum(r["weighted_age_sum"] for r in reports) / known if known else None,
        "version_missing_token_rate": missing / tokens if tokens and not unaudited_rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-dump", type=Path, nargs="+", help="Trusted local debug train dumps from reporting ranks only (pickle)"
    )
    args = parser.parse_args()
    if args.train_dump:
        import torch

        rows = [
            sample.get(AUDIT_FIELD)
            for path in args.train_dump
            for sample in torch.load(path, map_location="cpu", weights_only=False)["samples"]
        ]
        reports = [row for row in rows if row is not None and "consumption" in row]
        result = {"reports": reports, "summary": summarize(reports, sum(row is None for row in rows))}
    else:
        report = asyncio.run(run_demo())
        result = {
            "mode": "CPU simulated publication; no GPU inference/training",
            "reports": [report],
            "summary": summarize([report]),
        }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    get_logger(__name__).info("Audit written to %s", args.output)


if __name__ == "__main__":
    main()
