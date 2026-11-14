"""Same-target retry retains commitment; earlier-target rollback is
distinct."""

from unittest.mock import patch

from relax.distributed.weight_sync.session import InstallSession


class Engines:
    def close(self):
        pass


def test_committed_same_target_recover_returns_active_install_result(tmp_path):
    session = InstallSession(tmp_path, Engines())
    manifest = dict(stream="s", version=51, root="r")
    session.status = lambda: dict(
        committed=True, target=manifest, installation_id="lost-load", committed_checkpoint={"checkpoint": "kept"}
    )
    with patch.object(session, "install", return_value={"phase": "ACTIVE"}) as install:
        assert session.recover(tmp_path, manifest, "new-id")["phase"] == "ACTIVE"
    assert install.call_args.args == (tmp_path, manifest, "lost-load")
    assert "_recovery_for" not in install.call_args.kwargs


def test_uncommitted_target_rollback_retains_requested_target(tmp_path):
    session = InstallSession(tmp_path, Engines())
    target = dict(stream="s", version=52, root="new")
    old = dict(stream="s", version=51, root="old")
    session.status = lambda: dict(committed=False, target=target, active_checkpoint={"checkpoint": "old"})
    with patch.object(session, "install", return_value={"phase": "WAITING_ACTION"}) as install:
        assert session.recover(tmp_path, old, "rollback")["phase"] == "WAITING_ACTION"
    assert install.call_args.kwargs["_recovery_for"] == target
