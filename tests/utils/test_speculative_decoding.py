# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.utils.types import get_spec_token_counts


@pytest.mark.parametrize(
    ("accept_key", "draft_key"),
    [
        ("spec_num_correct_drafts", "spec_num_proposed_drafts"),
        ("spec_accepted_drafts", "spec_proposed_drafts"),
        ("spec_accept_token_num", "spec_draft_token_num"),
    ],
)
def test_get_spec_token_counts_preserves_missing_vs_zero(accept_key: str, draft_key: str) -> None:
    assert get_spec_token_counts({}) is None
    assert get_spec_token_counts({accept_key: 0, draft_key: 10}) == (0, 10)
    assert get_spec_token_counts({accept_key: 0, draft_key: 0}) == (0, 0)
