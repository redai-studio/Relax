# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the real Serve class with controlled training collaborators."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def actor_class():
    pytest.importorskip("ray", reason="requires Ray/Serve to import the actual Actor deployment")
    pytest.importorskip("transfer_queue", reason="requires the training service runtime")
    from relax.components.actor import Actor

    return Actor.func_or_class


@pytest.mark.parametrize("mode", ["sync", "hybrid", "fully_async"])
def test_training_receives_export_intent_before_rpc_and_collects_once(actor_class, mode, monkeypatch):
    import relax.components.actor as module

    events = []
    actor = actor_class.__new__(actor_class)
    actor.config = SimpleNamespace(num_critic_only_steps=0, hybrid=mode == "hybrid", fully_async=mode != "sync")
    actor.step = 7
    actor._lora_profile = object()
    descriptor = {"version_id": "B"}
    actor.actor_model = SimpleNamespace(
        **{
            name: lambda step, **options: (mode, step, options)
            for name in ("train_hybrid", "train_fully_async", "async_train")
        }
    )
    monkeypatch.setattr(module.ray, "get", lambda refs: events.append(refs))
    actor._prepare_lora_export = lambda step: events.append(("intent", step)) or descriptor
    actor._finish_lora_export = lambda: events.append("sealed")
    actor._maybe_save_model = lambda: pytest.fail("publication checkpoint already ran before worker sleep")
    assert actor._execute_training()
    assert events == [("intent", 8), (mode, 7, {"lora_export": descriptor}), "sealed"]


@pytest.mark.parametrize("mode", ["sync", "hybrid", "fully_async"])
def test_publication_bootstrap_never_pushes_mutable_rollout_weights(actor_class, mode):
    actor = actor_class.__new__(actor_class)
    actor.config = SimpleNamespace(hybrid=mode == "hybrid", fully_async=mode != "sync", loss_type="policy_loss")
    actor._lora_profile = object()
    events = []
    actor.actor_model = SimpleNamespace(
        set_rollout_manager=lambda manager: events.append("manager"),
        update_weights=lambda: pytest.fail("bootstrap called legacy weight replacement"),
    )
    actor._bootstrap_lora_publication = lambda: events.append("bootstrap")
    actor.set_rollout_manager(object())
    assert events == ["manager", "bootstrap"]
