# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DPO_DIR = ROOT / "scripts" / "training" / "dpo"


def test_qwen36_dpo_scripts_have_required_runtime_flags():
    for name in (
        "run-qwen36-35B-A3B-dpo-8xgpu.sh",
        "run-qwen36-35B-A3B-dpo-32xgpu.sh",
    ):
        script = (DPO_DIR / name).read_text(encoding="utf-8")
        assert "--enable-weights-backuper" in script
        assert "--use-gloo-process-groups" in script
        if "expandable_segments:True" in script and "--colocate" in script:
            assert "--selective-offload" in script
        assert '--load "${CHECKPOINT_DIR}"' in script
        assert '--save "${CHECKPOINT_DIR}"' in script


def test_qwen36_8gpu_dpo_topologies_support_cp_pp_comparison():
    script = (DPO_DIR / "run-qwen36-35B-A3B-dpo-8xgpu.sh").read_text(encoding="utf-8")
    assert "--tensor-model-parallel-size 1" in script
    assert '--pipeline-model-parallel-size "${PP_SIZE}"' in script
    assert '--context-parallel-size "${CP_SIZE}"' in script
    assert '--expert-model-parallel-size "${EP_SIZE}"' in script
    assert 'PARALLEL_CASE="${PARALLEL_CASE:?Set PARALLEL_CASE to baseline, cp, or pp}"' in script
    assert "baseline)\n        PP_SIZE=1\n        CP_SIZE=1" in script
    assert "cp)\n        PP_SIZE=1\n        CP_SIZE=2" in script
    assert "pp)\n        PP_SIZE=2\n        CP_SIZE=1" in script
    assert "ACTOR_WORLD_SIZE=8" in script
    assert "EP_SIZE=4" in script
    assert "DP_SIZE=$((ACTOR_WORLD_SIZE / (PP_SIZE * CP_SIZE)))" in script
    assert ': "${EXP_NAME:?Set a fresh EXP_NAME for every comparison run}"' in script
    assert 'if [ -e "${CHECKPOINT_DIR}" ]; then' in script
    assert 'NUM_ROLLOUT="${NUM_ROLLOUT:-50}"' in script
    assert 'PACKED_TOKEN_CAPACITY="${PACKED_TOKEN_CAPACITY:-4096}"' in script
    assert '--max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"' in script
    assert '--save-interval "${SAVE_INTERVAL:-100}"' in script
    assert '--resource "{\\"sft\\":[1,0],\\"actor\\":[1,${ACTOR_WORLD_SIZE}]}"' in script
    assert "--eval-prompt-data" not in script
    assert "--eval-interval" not in script


def test_qwen36_32gpu_dpo_topology_supports_ep8():
    script = (DPO_DIR / "run-qwen36-35B-A3B-dpo-32xgpu.sh").read_text(encoding="utf-8")
    assert "--tensor-model-parallel-size 1" in script
    assert "--pipeline-model-parallel-size 2" in script
    assert "--context-parallel-size 16" in script
    assert "--expert-model-parallel-size 8" in script
    assert "--max-tokens-per-gpu 4096" in script
    assert "--sequence-parallel" in script
    assert "TP1 / PP2 / CP16 / EP8" in script


def test_qwen36_32gpu_dpo_propagates_worker_environment():
    script = (DPO_DIR / "run-qwen36-35B-A3B-dpo-32xgpu.sh").read_text(encoding="utf-8")
    assert 'names.update(("PYTORCH_CUDA_ALLOC_CONF", "CLEARML_CONFIG_FILE"))' in script
    assert 'env["RELAX_PROPAGATE_ENV_VARS"]' in script
