# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regression for the DeepSeek-V4 indexer QAT / replay guard.

``install_indexer_qat`` checks ``config.relax_use_indexer_replay``, but that
flag is only set later by ``fill_indexer_replay``, so the check could be
bypassed on the normal initialization path. The provider now rejects the
combination from the launch argument before building the model.
"""

import pytest

from relax.models.deepseek_v4.provider import _validate_indexer_qat_replay


pytestmark = pytest.mark.cpu


def test_native_indexer_qat_rejects_rollout_indexer_replay():
    with pytest.raises(ValueError, match="cannot be combined with --use-rollout-indexer-replay"):
        _validate_indexer_qat_replay(enabled=True, use_rollout_indexer_replay=True)


@pytest.mark.parametrize(
    ("enabled", "replay"),
    [(True, False), (False, True), (False, False)],
)
def test_guard_allows_supported_combinations(enabled, replay):
    _validate_indexer_qat_replay(enabled=enabled, use_rollout_indexer_replay=replay)
