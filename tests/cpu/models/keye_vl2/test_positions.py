# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.models.keye_vl2.indexer import text_rope
from relax.models.keye_vl2.positions import image_positions


def test_image_mrope_resets_documents_and_compresses_image_extent():
    image = 99
    ids = torch.tensor(
        [[1, image, image, image, image, 2, image, image, 3], [1, image, image, image, image, 2, image, image, 3]]
    )
    positions = image_positions(ids, torch.tensor([[1, 4, 4], [1, 2, 4]] * 2), image)
    expected = torch.tensor([[0, 1, 1, 1, 1, 3, 4, 4, 6], [0, 1, 1, 2, 2, 3, 4, 4, 6], [0, 1, 2, 1, 2, 3, 4, 5, 6]])
    torch.testing.assert_close(positions[:, 0], expected)
    torch.testing.assert_close(positions[:, 1], expected)


def test_main_and_indexer_rope_use_distinct_contiguous_axes():
    for dim, sections in ((128, (16, 24, 24)), (64, (8, 12, 12))):
        query = torch.randn(7, 2, dim)
        positions = torch.stack((torch.arange(7), torch.arange(7) // 2, torch.arange(7) // 3), -1)
        inv = 1 / (1e7 ** (torch.arange(0, dim, 2).float() / dim))
        angles = torch.cat([positions[:, axis, None] * part for axis, part in enumerate(inv.split(sections))], -1)
        angles = torch.cat((angles, angles), -1)[:, None]
        first, second = query.chunk(2, -1)
        expected = query * angles.cos() + torch.cat((-second, first), -1) * angles.sin()
        torch.testing.assert_close(text_rope(query, positions, 1e7), expected)
