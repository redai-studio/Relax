import torch

from relax.engine.sft.dataset.chat_template import (
    _keep_last_n_mask_runs,
    _last_round_assistant_run_count,
    _last_round_learn_indices,
)
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


def _tool_use_sample() -> CanonicalSample:
    """A prior round plus a current round of [tool-call, tool result,
    answer]."""
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="old query", learn=False),
            CanonicalMessage(role="assistant", content="old answer", learn=True),
            CanonicalMessage(role="user", content="current query", learn=False),
            CanonicalMessage(role="assistant", content="tool call", learn=True),
            CanonicalMessage(role="tool", content="tool result", learn=False),
            CanonicalMessage(role="assistant", content="final answer", learn=True),
        ],
        metadata={"source_dataset": "unit-test", "row_index": 0},
    )


def test_last_round_includes_tool_call_and_final_answer() -> None:
    # Path 2 (per-message fallback) semantics: last round = every learnable
    # message after the final user query (tool-call idx 3 AND final answer idx 5).
    assert _last_round_learn_indices(_tool_use_sample()) == {3, 5}


def test_last_round_assistant_run_count_counts_masked_roles() -> None:
    # Two learnable assistant messages in the last round (tool-call + answer);
    # the tool result is not assistant-masked, so it is not counted.
    assert _last_round_assistant_run_count(_tool_use_sample()) == 2


def test_keep_last_n_mask_runs_keeps_tool_call_and_answer() -> None:
    # HF assistant_masks for the last round: the tool-call run and the final
    # answer run are separated by 0s over the tool response. Path 1 must keep
    # BOTH (n=2), matching Path 2 — not just the final contiguous run.
    #        [ prior ][   0s   ][ tool-call ][0 tool 0][ answer ]
    masks = torch.tensor([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
    kept = _keep_last_n_mask_runs(masks, 2)
    assert kept.tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]


def test_keep_last_n_mask_runs_single_run_matches_old_behavior() -> None:
    # n=1 reduces to "keep only the final contiguous run" (non-tool-use case).
    masks = torch.tensor([1, 1, 0, 1, 0, 1, 1])
    assert _keep_last_n_mask_runs(masks, 1).tolist() == [0, 0, 0, 0, 0, 1, 1]


def test_keep_last_n_mask_runs_n_ge_runs_is_noop() -> None:
    masks = torch.tensor([1, 0, 1, 1, 0, 1])
    assert torch.equal(_keep_last_n_mask_runs(masks, 5), masks)


def test_keep_last_n_mask_runs_zero_clears() -> None:
    masks = torch.tensor([1, 1, 0, 1])
    assert int(_keep_last_n_mask_runs(masks, 0).sum()) == 0
