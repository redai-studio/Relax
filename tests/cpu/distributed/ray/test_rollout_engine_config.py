# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine configuration tests that still require rollout runtime
dependencies."""

import pytest


try:
    from relax.distributed.ray.rollout import EngineGroupConfig, ModelConfig

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


# ========================== EngineGroupConfig ===============================


class TestEngineGroupConfig:
    @pytest.mark.parametrize("wt", ["regular", "prefill", "decode", "placeholder"])
    def test_valid_worker_types(self, wt):
        cfg = EngineGroupConfig(worker_type=wt, num_gpus=4)
        assert cfg.worker_type == wt

    def test_invalid_worker_type(self):
        with pytest.raises(AssertionError, match="Invalid worker_type"):
            EngineGroupConfig(worker_type="bad", num_gpus=4)

    def test_zero_gpus(self):
        with pytest.raises(AssertionError, match="num_gpus must be > 0"):
            EngineGroupConfig(worker_type="regular", num_gpus=0)

    def test_negative_gpus(self):
        with pytest.raises(AssertionError, match="num_gpus must be > 0"):
            EngineGroupConfig(worker_type="regular", num_gpus=-1)

    def test_optional_fields_defaults(self):
        cfg = EngineGroupConfig(worker_type="regular", num_gpus=4)
        assert cfg.num_gpus_per_engine is None
        assert cfg.overrides == {}

    def test_optional_fields_set(self):
        cfg = EngineGroupConfig(
            worker_type="regular",
            num_gpus=4,
            num_gpus_per_engine=2,
            overrides={"model_path": "/m"},
        )
        assert cfg.num_gpus_per_engine == 2
        assert cfg.overrides == {"model_path": "/m"}


# ============================= ModelConfig ==================================


class TestModelConfig:
    def test_resolve_defaults(self):
        args = type(
            "Args",
            (),
            {
                "rollout_num_gpus_per_engine": 4,
                "hf_checkpoint": "/default/model",
                "sglang_hf_checkpoint": None,
            },
        )()
        cfg = ModelConfig(
            name="actor",
            engine_groups=[
                EngineGroupConfig(worker_type="regular", num_gpus=8),
            ],
        )
        cfg.resolve(args)
        assert cfg.engine_groups[0].num_gpus_per_engine == 4
        assert cfg.engine_groups[0].overrides["model_path"] == "/default/model"

    def test_resolve_per_group_override(self):
        args = type(
            "Args",
            (),
            {
                "rollout_num_gpus_per_engine": 4,
                "hf_checkpoint": "/default/model",
                "sglang_hf_checkpoint": None,
            },
        )()
        cfg = ModelConfig(
            name="actor",
            num_gpus_per_engine=2,
            engine_groups=[
                EngineGroupConfig(worker_type="regular", num_gpus=4, num_gpus_per_engine=8),
            ],
        )
        cfg.resolve(args)
        # Per-group override takes precedence
        assert cfg.engine_groups[0].num_gpus_per_engine == 8

    def test_has_pd_disaggregation(self):
        cfg_no = ModelConfig(
            name="m",
            engine_groups=[EngineGroupConfig(worker_type="regular", num_gpus=4)],
        )
        assert cfg_no.has_pd_disaggregation is False

        cfg_yes = ModelConfig(
            name="m",
            engine_groups=[
                EngineGroupConfig(worker_type="prefill", num_gpus=4),
                EngineGroupConfig(worker_type="decode", num_gpus=4),
            ],
        )
        assert cfg_yes.has_pd_disaggregation is True

    def test_total_num_gpus(self):
        cfg = ModelConfig(
            name="m",
            engine_groups=[
                EngineGroupConfig(worker_type="regular", num_gpus=4),
                EngineGroupConfig(worker_type="regular", num_gpus=8),
            ],
        )
        assert cfg.total_num_gpus == 12
