# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import asyncio
from argparse import Namespace
from unittest.mock import Mock

import pytest

from relax.utils.arguments import get_slime_extra_args_provider


def test_hybrid_parser_needs_no_backend_flag():
    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)
    args = parser.parse_args(["--hybrid", "--autoscaler-config", "elastic.yaml"])
    assert args.hybrid
    assert args.autoscaler_config == "elastic.yaml"
    assert not hasattr(args, "hybrid_weight_sync_backend")


@pytest.mark.parametrize("offload_flag", ["--offload-rollout", "--offload"])
@pytest.mark.parametrize("elastic", [False, True])
def test_hybrid_accepts_rollout_offload(offload_flag, elastic):
    from relax.utils.arguments import slime_validate_args
    from tests.cpu.utils.test_arguments_opd_teacher_colocate import _opd_args

    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)
    args = parser.parse_args(["--hybrid", offload_flag], namespace=Namespace(**vars(_opd_args())))
    args.autoscaler_config = "elastic.yaml" if elastic else None
    slime_validate_args(args)
    assert args.offload_rollout is True
    assert args.offload_train is (offload_flag == "--offload")


@pytest.mark.parametrize("extra_flags", [[], ["--offload-train"], ["--no-offload-rollout"]])
def test_hybrid_accepts_training_only_offload(extra_flags):
    from relax.utils.arguments import slime_validate_args
    from tests.cpu.utils.test_arguments_opd_teacher_colocate import _opd_args

    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)
    args = parser.parse_args(["--hybrid", *extra_flags], namespace=Namespace(**vars(_opd_args())))
    slime_validate_args(args)
    assert args.offload_rollout is False
    assert args.offload_train is ("--offload-train" in extra_flags)


@pytest.mark.parametrize("offload_flag", ["--offload-rollout", "--offload"])
def test_colocate_still_accepts_rollout_offload(offload_flag):
    from relax.utils.arguments import slime_validate_args
    from tests.cpu.utils.test_arguments_opd_teacher_colocate import _opd_args

    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)
    args = parser.parse_args(["--colocate", offload_flag], namespace=Namespace(**vars(_opd_args())))
    slime_validate_args(args)
    assert args.offload_rollout is True


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("loss_type", ["policy_loss", "sft", "dpo"])
def test_hybrid_init_weight_sync_routes_elastic_and_skips_offline(enabled, loss_type, tmp_path):
    from relax.components.actor import Actor

    cls = Actor.func_or_class
    actor = cls.__new__(cls)
    config_path = None
    if enabled is not None:
        config_path = tmp_path / "elastic.yaml"
        config_path.write_text(f"enabled: {str(enabled).lower()}\n")
    actor.config = Namespace(hybrid=True, fully_async=True, autoscaler_config=config_path, loss_type=loss_type)
    actor.actor_model = Mock()
    manager = Mock()
    asyncio.run(actor.set_rollout_manager(manager))
    actor.actor_model.set_rollout_manager.assert_called_once_with(manager)
    if loss_type != "policy_loss":
        actor.actor_model.update_weights.assert_not_called()
        actor.actor_model.update_weights_fully_async.assert_not_called()
    else:
        actor.actor_model.update_weights.assert_called_once_with()
        actor.actor_model.update_weights_fully_async.assert_not_called()


@pytest.mark.parametrize("elastic", [False, True])
def test_hybrid_requires_real_weights_backuper(elastic):
    from relax.utils.arguments import slime_validate_args
    from tests.cpu.utils.test_arguments_opd_teacher_colocate import _opd_args

    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)
    args = parser.parse_args([])
    vars(args).update(vars(_opd_args()))
    args.hybrid = True
    args.enable_weights_backuper = False
    args.autoscaler_config = "elastic.yaml" if elastic else None
    with pytest.raises(ValueError, match="Hybrid training requires the weights backuper"):
        slime_validate_args(args)


@pytest.mark.parametrize("config_path", [None, "elastic.yaml"])
def test_full_async_does_not_add_hybrid_init_sync(config_path):
    from relax.components.actor import Actor

    actor = Actor.func_or_class.__new__(Actor.func_or_class)
    actor.config = Namespace(hybrid=False, fully_async=True, autoscaler_config=config_path, loss_type="policy_loss")
    actor.actor_model = Mock()
    asyncio.run(actor.set_rollout_manager(Mock()))
    actor.actor_model.update_weights.assert_not_called()
    actor.actor_model.update_weights_fully_async.assert_not_called()
