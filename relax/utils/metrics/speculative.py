# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any

from relax.utils.types import Sample


def compute_spec_metrics(args, all_samples: list[Sample]) -> dict[str, Any]:
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}

    # Deduplicate shared generations by request ID within each session.
    generations: dict[tuple[str, str], dict[str, Any]] = {}
    fallback_spec_infos: list[Sample.SpecInfo] = []

    for sample in all_samples:
        trace = sample.metadata.get("agentic_trace")
        if isinstance(trace, dict) and "spec_generations" in trace:
            session_id = trace["session_id"]
            for generation in trace["spec_generations"]:
                generations[(session_id, generation["request_id"])] = generation
        else:
            fallback_spec_infos.append(sample.spec_info)

    record_count = len(generations) + len(fallback_spec_infos)
    accepted = 0
    proposed = 0
    accept_covered = 0
    verify = 0
    completion = 0
    length_covered = 0

    for generation in generations.values():
        if "spec_accept_token_num" in generation and "spec_draft_token_num" in generation:
            accepted += generation["spec_accept_token_num"]
            proposed += generation["spec_draft_token_num"]
            accept_covered += 1
        if "spec_verify_ct" in generation and "completion_token_num" in generation:
            verify += generation["spec_verify_ct"]
            completion += generation["completion_token_num"]
            length_covered += 1

    # Samples without generation-level accounting cannot distinguish explicit zero from historical defaults.
    for spec_info in fallback_spec_infos:
        if spec_info.spec_draft_token_num > 0:
            accepted += spec_info.spec_accept_token_num
            proposed += spec_info.spec_draft_token_num
            accept_covered += 1
        if spec_info.spec_verify_ct > 0:
            verify += spec_info.spec_verify_ct
            completion += spec_info.completion_token_num
            length_covered += 1

    metrics = {
        "spec_accept_rate_coverage": accept_covered / record_count if record_count else 0.0,
        "spec_accept_length_coverage": length_covered / record_count if record_count else 0.0,
    }

    # Compute ratios only after the underlying counters have been aggregated.
    if proposed > 0:
        metrics["spec_accept_rate"] = accepted / proposed
    if verify > 0:
        metrics["spec_accept_length"] = completion / verify

    return metrics
