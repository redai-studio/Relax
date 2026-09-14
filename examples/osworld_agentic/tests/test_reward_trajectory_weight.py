from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


def _processor():
    path = Path(__file__).resolve().parents[1] / "app" / "reward_post_process.py"
    spec = spec_from_file_location("osworld_reward_weight_check", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.post_process_rewards


def _samples():
    samples, lengths = [], []
    for group, outcomes in enumerate(([1] * 4 + [0] * 4, [0] * 8, [1] * 8)):
        for attempt, (outcome, records) in enumerate(zip(outcomes, [2, 31, 4, 10, 2, 31, 4, 10])):
            for _ in range(records):
                samples.append(
                    SimpleNamespace(
                        group_index=group,
                        session_id=f"{group}-{attempt}",
                        metadata={"agent_turns": min(records, 30)},
                        get_reward_value=lambda args, value=outcome: value,
                    )
                )
                lengths.append(records)
    return samples, lengths


def test_disable_record_weight_preserves_outcome_and_group_filter(monkeypatch):
    settings = {
        "OSWORLD_MAX_STEPS_REF": "30",
        "OSWORLD_LENGTH_PENALTY_COEF": "0.05",
        "OSWORLD_CAP_PENALTY": "0.2",
        "OSWORLD_ACC_LO": "0.05",
        "OSWORLD_ACC_HI": "0.95",
        "OSWORLD_ADV_CLIP": "0",
        "OSWORLD_REWARD_DUMP": "",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    processor = _processor()
    samples, lengths = _samples()
    args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True)
    monkeypatch.setenv("OSWORLD_PER_TRAJ_NORM", "1")
    shaped_before, advantage_before = processor(args, samples)
    monkeypatch.setenv("OSWORLD_PER_TRAJ_NORM", "0")
    shaped_after, advantage_after = processor(args, samples)
    assert shaped_after == shaped_before
    for sample, records, old, new, reward in zip(samples, lengths, advantage_before, advantage_after, shaped_after):
        turns = min(records, 30)
        outcome = 1 if sample.get_reward_value(args) else -1
        assert reward == pytest.approx(outcome - 0.05 * turns / 30 - (0.2 if turns >= 30 else 0))
        if sample.group_index:
            assert old == new == 0
        else:
            assert new == pytest.approx(old * records)
            assert (new > 0) == (outcome > 0)
