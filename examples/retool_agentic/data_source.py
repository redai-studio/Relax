# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpointable ReTool draws in the smaller chunks requested by Agentic."""

import copy
from typing import Any

from examples.retool.data_source import ReToolDataSource
from examples.retool.rollout import VERSION
from examples.retool.vocab import MODEL_VOCAB, TOKENIZER_LIMIT
from examples.retool_agentic.protocol import render_prompt
from relax.utils.types import Sample


class ReToolAgenticDataSource(ReToolDataSource):
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
        super().__init__(args)
        if len(self.tokenizer) != TOKENIZER_LIMIT:
            raise ValueError("This recipe requires the original Qwen2.5 ReTool tokenizer")
        for case, prompt in zip(self.cases, self.prompts, strict=True):
            if render_prompt(case["prompt"]) != prompt:
                raise ValueError("Agentic template differs from the model's original initial prompt")
        self.group_cursor = 0

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        batch, k = self.args.rollout_batch_size, self.args.n_samples_per_prompt
        if num_samples < 1 or self.group_cursor >= batch * self.args.num_rollout:
            raise ValueError("Positive group count within the planned rollout budget required")
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
                    policy_outer=self.group_cursor // batch,
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
        self.cursor = self.group_cursor // batch
        return groups

    def save(self, rollout_id: int) -> None:
        if self.group_cursor != (rollout_id + 1) * self.args.rollout_batch_size:
            raise RuntimeError("Checkpoint requires exactly one complete Agentic prompt batch")
        super().save(rollout_id)

    def load(self, rollout_id: int | None = None) -> None:
        super().load(rollout_id)
        self.group_cursor = self.cursor * self.args.rollout_batch_size
