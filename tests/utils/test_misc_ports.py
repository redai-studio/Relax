# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import socket

import pytest

from relax.utils import http_utils, misc


def test_misc_exports_production_port_probe():
    assert misc.is_port_available is http_utils.is_port_available


def test_misc_get_free_port_rejects_occupied_single_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert not misc.is_port_available(port)
        with pytest.raises(RuntimeError, match="exhausted"):
            misc.get_free_port(start_port=port, max_port=port)


def test_misc_get_free_port_rejects_block_outside_window():
    with pytest.raises(RuntimeError, match="exceeds max_port"):
        misc.get_free_port(start_port=20000, consecutive=2, max_port=20000)
