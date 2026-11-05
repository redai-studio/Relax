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
from examples.retool_agentic.vocab import MODEL_VOCAB, TOKENIZER_LIMIT
from relax.engine.rollout.data_source import DataSource
from relax.utils.types import Sample


class ReToolAgenticDataSource(DataSource):
    def __init__(self, args: Any) -> None:
        if any(getattr(args, key, False) for key in ("agentic_prelaunch", "partial_rollout", "fully_async")):
            raise ValueError("This recipe requires synchronous complete batches without prelaunch or partial replay")
        if getattr(args, "eval_datasets", None):
            raise ValueError("This initial Agentic recipe supports training only; do not configure Eval datasets")
        if getattr(args, "dynamic_sampling_filter_path", None) or (
            getattr(args, "over_sampling_batch_size", None) not in (None, args.rollout_batch_size)
        ):
            raise ValueError(
                "ReTool checkpoint accounting requires complete batches without filtering or oversampling"
            )
        from transformers import AutoTokenizer

        self.args = args
        raw = Path(args.prompt_data).read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
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
        if len(self.tokenizer) != TOKENIZER_LIMIT:
            raise ValueError("This recipe requires the original Qwen2.5 ReTool tokenizer")
        for case, prompt in zip(self.cases, self.prompts, strict=True):
            if render_prompt(case["prompt"]) != prompt:
                raise ValueError("Agentic template differs from the model's original initial prompt")
        self.group_cursor = 0

    def __len__(self) -> int:
        return len(self.cases)

    def lengths(self) -> int:
        return len(self.cases)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        batch, k = self.args.rollout_batch_size, self.args.n_samples_per_prompt
        if num_samples < 1:
            raise ValueError("Positive group count required")
        count = min(num_samples, batch - self.group_cursor % batch)
        groups = []
        for _ in range(count):
            question = self.rng.randrange(len(self.cases))
            case = self.cases[question]
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
                    ),
                )
                sample = Sample(
                    prompt=copy.deepcopy(case["prompt"]),
                    label=case["label"],
                    metadata=metadata,
                    group_index=self.group_cursor,
                    index=index,
                )
                # Agentic transports sampling_params to the native /generate
                # backend. Keep padded model vocabulary out of both rollout
                # and training, without a custom HTTP request extension.
                sample.sampling_params = {"logit_bias": {str(i): -1e30 for i in range(TOKENIZER_LIMIT, MODEL_VOCAB)}}
                group.append(sample)
            groups.append(group)
            self.group_cursor += 1
        return groups

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise RuntimeError("Partial trajectory replay is disabled")

    def _state(self, cursor: int) -> dict:
        return dict(
            cursor=cursor,
            seed=self.args.rollout_seed,
            sha256=self.digest,
            batch=self.args.rollout_batch_size,
            K=self.args.n_samples_per_prompt,
            rng_state_version=1,
            rng_state=self.rng.getstate(),
        )

    def save(self, rollout_id: int) -> None:
        if rollout_id < 0 or self.group_cursor < (rollout_id + 1) * self.args.rollout_batch_size:
            raise RuntimeError("Checkpoint requires a complete Agentic prompt batch")
        path = Path(self.args.save) / "dataset" / f"retool_{rollout_id}.json"
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with temporary.open("x") as stream:
            json.dump(dict(self._state(rollout_id + 1), next_group_index=self.group_cursor), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.cursor = rollout_id + 1

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None or rollout_id is None or rollout_id < 0:
            return
        state = json.loads((Path(self.args.load) / "dataset" / f"retool_{rollout_id}.json").read_text())
        expected = self._state(rollout_id + 1)
        if any(state.get(key) != expected[key] for key in ("cursor", "seed", "sha256", "batch", "K")):
            raise ValueError("Resume data, seed or trajectory grouping changed")
        if state.get("rng_state_version") != 1 or not state.get("rng_state"):
            raise ValueError("Resume requires a native sampling RNG snapshot")

        def tuples(value: Any) -> Any:
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value

        self.rng.setstate(tuples(state["rng_state"]))
        self.cursor = rollout_id + 1
        self.group_cursor = state.get("next_group_index", self.cursor * self.args.rollout_batch_size)
        if type(self.group_cursor) is not int or self.group_cursor < self.cursor * self.args.rollout_batch_size:
            raise ValueError("Invalid resumed group index")
