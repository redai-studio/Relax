# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Seeded draws with replacement, K independent trajectories per draw."""

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from examples.retool.rollout import PREFIX, VERSION
from relax.engine.rollout.data_source import DataSource
from relax.utils.types import Sample


class ReToolDataSource(DataSource):
    def __init__(self, args: Any) -> None:
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

    def __len__(self) -> int:
        return len(self.cases)

    def lengths(self) -> int:
        return len(self.cases)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = []
        k = self.args.n_samples_per_prompt
        if num_samples != self.args.rollout_batch_size or self.cursor >= self.args.num_rollout:
            raise ValueError("Require one complete prompt batch within the planned rollout budget")
        for slot in range(num_samples):
            index = self.rng.randrange(len(self.cases))
            case = self.cases[index]
            group = []
            for member in range(k):
                metadata = copy.deepcopy(case.get("metadata", {}))
                metadata.update(
                    protocol_id=VERSION, policy_outer=self.cursor, scheduled_question=index, group_sample=member
                )
                group.append(
                    Sample(
                        prompt=self.prompts[index],
                        label=case["label"],
                        metadata=metadata,
                        group_index=self.cursor * num_samples + slot,
                        index=(self.cursor * num_samples + slot) * k + member,
                    )
                )
            groups.append(group)
        self.cursor += 1
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
        if self.cursor != rollout_id + 1:
            raise RuntimeError("Data cursor and checkpoint iteration differ")
        path = Path(self.args.save) / "dataset" / f"retool_{rollout_id}.json"
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_name(path.name + f".pending.{os.getpid()}")
        with temporary.open("x") as stream:
            json.dump(self._state(self.cursor), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None or rollout_id is None or rollout_id < 0:
            return
        state = json.loads((Path(self.args.load) / "dataset" / f"retool_{rollout_id}.json").read_text())
        expected = dict(
            cursor=rollout_id + 1,
            seed=self.args.rollout_seed,
            sha256=self.digest,
            batch=self.args.rollout_batch_size,
            K=self.args.n_samples_per_prompt,
        )
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Resume data, seed or trajectory grouping changed")
        if state.get("rng_state_version") != 1 or not state.get("rng_state"):
            raise ValueError("Native RNG snapshot required; historical reconstruction is forbidden")

        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value

        self.cursor = rollout_id + 1
        self.rng = random.Random()
        self.rng.setstate(tuples(state["rng_state"]))
