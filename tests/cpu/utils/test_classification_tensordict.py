# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Classification label conversion must stay dense through TensorDict."""

import torch

from relax.utils.utils import dict_to_tensordict


def test_scalar_classification_labels_are_int64():
    data = dict_to_tensordict({"classification_labels": [0, 2]}, batch_size=2)

    assert data["classification_labels"].dtype == torch.long
    assert data["classification_labels"].tolist() == [0, 2]


def test_multi_label_targets_are_dense_float32():
    data = dict_to_tensordict(
        {"classification_labels": [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]},
        batch_size=2,
    )

    labels = data["classification_labels"]
    assert labels.dtype == torch.float32
    assert labels.shape == (2, 3)
    assert not labels.is_nested
