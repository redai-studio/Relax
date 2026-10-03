# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from tests.utils.test_arguments_opd_teacher_colocate import _opd_args
from tests.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


def _versioned_args(tmp_path, total_gpus, gpus_per_engine):
    args = _opd_args()
    args.enable_versioned_lora_publication = True
    args.lora_adapter_mode = True
    args.lora_rank = 8
    args.lora_scope = "language"
    args.fully_async = True
    args.use_dynamic_global_batch_size = False
    args.use_agentic_rollout = True
    args.rollout_num_gpus = total_gpus
    args.rollout_num_gpus_per_engine = gpus_per_engine
    args.agent_command = "python agent.py"
    args.agent_cwd = str(tmp_path)
    args.agent_timeout = 60
    args.agent_env = []
    args.agentic_concurrency = None
    args.agentic_eval_concurrency = None
    args.agentic_program_admission = False
    args.agentic_session_lifecycle = False
    return args


@pytest.mark.parametrize(("total_gpus", "gpus_per_engine"), [(2, 1), (4, 2)])
def test_versioned_lora_accepts_two_complete_engines(arguments_module, tmp_path, total_gpus, gpus_per_engine):
    args = _versioned_args(tmp_path, total_gpus, gpus_per_engine)
    arguments_module.slime_validate_args(args)


@pytest.mark.parametrize(
    ("total_gpus", "gpus_per_engine"),
    [(4, 1), (5, 2), (1, 1), (0, 1), (2, 0), (2, -1), (None, 1), (2, None)],
)
def test_versioned_lora_rejects_invalid_engine_allocation(arguments_module, tmp_path, total_gpus, gpus_per_engine):
    args = _versioned_args(tmp_path, total_gpus, gpus_per_engine)
    with pytest.raises(ValueError, match="exactly two rollout engines"):
        arguments_module.slime_validate_args(args)


def test_non_versioned_lora_keeps_existing_engine_allocation(arguments_module, tmp_path):
    args = _versioned_args(tmp_path, 4, 1)
    args.enable_versioned_lora_publication = False
    arguments_module.slime_validate_args(args)
