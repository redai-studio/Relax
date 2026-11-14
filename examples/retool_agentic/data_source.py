# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Seeded, checkpointable groups for resident ReTool Agentic rollout."""

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from examples.retool_agentic.protocol import PREFIX, VERSION, render_prompt
from relax.engine.rollout.data_source import DataSource
from relax.utils.agentic_obs.source import SourceRow, snapshot_source
from relax.utils.types import Sample


class ReToolAgenticDataSource(DataSource):
    def __init__(self, args: Any) -> None:
        self.hybrid = bool(getattr(args, "hybrid", False))
        if getattr(args, "partial_rollout", False) or (
            not self.hybrid and any(getattr(args, key, False) for key in ("agentic_prelaunch", "fully_async"))
        ):
            raise ValueError("This recipe requires synchronous complete batches without prelaunch or partial replay")
        if self.hybrid and (getattr(args, "start_rollout_id", 0) or 0) != 0:
            raise ValueError("The hybrid recipe starts fresh; resident agent state is not checkpointed")
        if getattr(args, "eval_datasets", None):
            raise ValueError("This initial Agentic recipe supports training only; do not configure Eval datasets")
        if getattr(args, "dynamic_sampling_filter_path", None) or (
            not self.hybrid and getattr(args, "over_sampling_batch_size", None) not in (None, args.rollout_batch_size)
        ):
            raise ValueError(
                "ReTool checkpoint accounting requires complete batches without filtering or oversampling"
            )
        from transformers import AutoTokenizer

        self.args = args
        raw = Path(args.prompt_data).read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.cases = [
            SourceRow(json.loads(line), args.prompt_data, row_id)
            for row_id, line in enumerate(raw.decode().splitlines())
            if line.strip()
        ]
        if not self.cases:
            raise ValueError("Empty ReTool training data")
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, local_files_only=True)
        self.prompts = []
        for case in self.cases:
            prompt = self.tokenizer.apply_chat_template(case["prompt"], tokenize=False, add_generation_prompt=True)
            if not case["prompt"][0]["content"].startswith(PREFIX):
                raise ValueError("Prepared training data must use the author-code prompt")
            if len(self.tokenizer.encode(prompt, add_special_tokens=False)) > args.rollout_max_prompt_len:
                raise ValueError("ReTool prompt exceeds budget; no silent filtering or truncation")
            self.prompts.append(prompt)
        self.cursor = 0
        self.rng = random.Random(args.rollout_seed)
        if len(self.tokenizer) != (getattr(args, "policy_vocab_size", None) or 151665):
            raise ValueError("This recipe requires the original Qwen2.5 ReTool tokenizer")
        for case, prompt in zip(self.cases, self.prompts, strict=True):
            if render_prompt(case["prompt"]) != prompt:
                raise ValueError("Agentic template differs from the model's original initial prompt")
        self.group_cursor = 0
        start_rollout_id = getattr(args, "start_rollout_id", 0) or 0
        if start_rollout_id > 0:
            if args.load is None:
                raise ValueError("Resume requires a ReTool checkpoint directory")
            # The framework catches load() errors; construction must fail before
            # this source can serve samples with an unvalidated resume state.
            self.load(start_rollout_id - 1)

    def __len__(self) -> int:
        return len(self.cases)

    def lengths(self) -> int:
        return len(self.cases)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        batch, k = self.args.rollout_batch_size, self.args.n_samples_per_prompt
        if num_samples < 1:
            raise ValueError("Positive group count required")
        count = num_samples if self.hybrid else min(num_samples, batch - self.group_cursor % batch)
        groups = []
        for _ in range(count):
            question = self.rng.randrange(len(self.cases))
            case = self.cases[question]
            # Capture the original row before adding per-replica sampling metadata.
            prompt_source = snapshot_source(case, metadata_key="metadata", split="train")
            group = []
            for member in range(k):
                index = self.group_cursor * k + member
                metadata = copy.deepcopy(case.get("metadata", {}))
                metadata.update(
                    protocol_id=VERSION,
                    scheduled_question=question,
                    group_sample=member,
                    retool_agentic=dict(
                        max_turns=self.args.max_turns,
                        response_budget=self.args.rollout_max_response_len,
                        context_budget=self.args.rollout_max_context_len,
                        seed=self.args.rollout_seed + index * 1009,
                        max_consecutive_timeouts=getattr(self.args, "retool_max_consecutive_timeouts", 0),
                        tool_wall_time_budget_seconds=getattr(self.args, "retool_tool_wall_time_budget_seconds", 0),
                    ),
                )
                sample = Sample(
                    prompt=copy.deepcopy(case["prompt"]),
                    label=case["label"],
                    metadata=metadata,
                    group_index=self.group_cursor,
                    index=index,
                    agentic_prompt_source=prompt_source,
                )
                group.append(sample)
            groups.append(group)
            self.group_cursor += 1
        return groups

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise RuntimeError("Partial trajectory replay is disabled")

    def _state(self, cursor: int) -> dict:
        state = dict(
            cursor=cursor,
            seed=self.args.rollout_seed,
            sha256=self.digest,
            batch=self.args.rollout_batch_size,
            K=self.args.n_samples_per_prompt,
            rng_state_version=1,
            rng_state=self.rng.getstate(),
        )
        if self.hybrid:
            state["execution_mode"] = "hybrid"
        return state

    def save(self, rollout_id: int) -> None:
        # Fully async has a terminal backfill step without a new train batch.
        completed_steps = min(rollout_id + 1, self.args.num_rollout) if self.hybrid else rollout_id + 1
        if rollout_id < 0 or self.group_cursor < completed_steps * self.args.rollout_batch_size:
            raise RuntimeError("Checkpoint requires a complete Agentic prompt batch")
        path = Path(self.args.save) / "dataset" / f"retool_{rollout_id}.json"
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with temporary.open("x") as stream:
            json.dump(dict(self._state(completed_steps), next_group_index=self.group_cursor), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.cursor = completed_steps

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None or rollout_id is None or rollout_id < 0:
            return
        if self.hybrid:
            raise ValueError("The hybrid recipe cannot resume resident agents; choose a fresh experiment")
        state = json.loads((Path(self.args.load) / "dataset" / f"retool_{rollout_id}.json").read_text())
        if state.get("execution_mode", "sync") != "sync":
            raise ValueError("A hybrid data snapshot cannot resume synchronous training")
        expected = self._state(rollout_id + 1)
        if any(state.get(key) != expected[key] for key in ("cursor", "seed", "sha256", "batch", "K")):
            raise ValueError("Resume data, seed or trajectory grouping changed")
        if state.get("rng_state_version") != 1 or not state.get("rng_state"):
            raise ValueError("Resume requires a native sampling RNG snapshot")

        def tuples(value: Any) -> Any:
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value

        cursor = rollout_id + 1
        group_cursor = state.get("next_group_index", cursor * self.args.rollout_batch_size)
        if type(group_cursor) is not int or group_cursor < cursor * self.args.rollout_batch_size:
            raise ValueError("Invalid resumed group index")
        rng = random.Random()
        rng.setstate(tuples(state["rng_state"]))
        self.rng, self.cursor, self.group_cursor = rng, cursor, group_cursor
