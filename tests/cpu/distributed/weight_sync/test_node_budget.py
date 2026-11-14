import pytest

from relax.distributed.weight_sync.budget import NodeBudget


def test_shared_budget_counts_resident_and_incomplete_installations(tmp_path):
    budget = NodeBudget(tmp_path, {"host": 100, "gpu:0": 50}, {"host": 20, "gpu:0": 30})
    with budget.operation("first", {"host": 60, "gpu:0": 10}, quiescent=lambda: False):
        pass
    reopened = NodeBudget(tmp_path, {"host": 100, "gpu:0": 50}, {"host": 20, "gpu:0": 30})
    with pytest.raises(OSError, match="Node capacity unavailable"):
        with reopened.operation("second", {"host": 30}, quiescent=lambda: True):
            pytest.fail("Preparation must not start without headroom")
    with reopened.operation("first", {"host": 60, "gpu:0": 10}, quiescent=lambda: True):
        pass
    with reopened.operation("second", {"host": 30}, quiescent=lambda: True):
        pass
    assert not list(tmp_path.glob("reservation-*.json"))


def test_failed_preparation_releases_but_uncertain_device_work_remains(tmp_path):
    budget = NodeBudget(tmp_path, {"host": 100}, {})
    with pytest.raises(ValueError):
        with budget.operation("prepare", {"host": 100}, quiescent=lambda: True):
            raise ValueError("Corrupt input before installation")
    assert not list(tmp_path.glob("reservation-*.json"))
    with pytest.raises(TimeoutError):
        with budget.operation("install", {"host": 100}, quiescent=lambda: False):
            raise TimeoutError("Worker completion unknown")
    assert len(list(tmp_path.glob("reservation-*.json"))) == 1


def test_shared_configuration_and_dimensions_are_explicit(tmp_path):
    budget = NodeBudget(tmp_path, {"host": 100}, {"host": 30})
    with pytest.raises(ValueError, match="same limits"):
        NodeBudget(tmp_path, {"host": 200}, {"host": 40})
    with pytest.raises(ValueError, match="Invalid operation"):
        with budget.operation("consumer", {"unconfigured_gpu": 1}, quiescent=lambda: True):
            pass


def test_expanding_capacity_preserves_reservations_and_updates_live_peers(tmp_path):
    original = NodeBudget(tmp_path, {"host": 100}, {"host": 30})
    with original.operation("waiting", {"host": 60}, quiescent=lambda: False):
        pass
    NodeBudget(tmp_path, {"host": 200}, {"host": 30})
    with original.operation("new", {"host": 100}, quiescent=lambda: True):
        assert (tmp_path / "reservation-waiting.json").is_file()
    with original.operation("waiting", {"host": 60}, quiescent=lambda: True):
        pass
    assert not list(tmp_path.glob("reservation-*.json"))
