# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Load the shared LamRA model and adapter arguments from the stage's HF input.
read -r lamra_model_type lamra_vocab_size lamra_hidden_size < <(python3 - "${HF_CHECKPOINT}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    config = json.load(stream)
text = config.get("text_config", config)
print(config["model_type"], text["vocab_size"], text.get("hidden_size", 0))
PY
)
case "${lamra_model_type}" in
    qwen2_vl)
        LAMRA_MODEL_NAME=qwen2-vl-2b
        QWEN2_VL_VOCAB_SIZE=${lamra_vocab_size}
        source "${RELAX_ROOT}/scripts/models/qwen2-vl-2B.sh"
        LAMRA_LORA_TARGETS=(linear_q linear_k linear_v linear_proj linear_fc1_gate linear_fc1_up linear_fc2)
        ;;
    qwen3_5)
        if [ "${lamra_hidden_size}" != 1024 ]; then
            echo "The LamRA Qwen3.5 profile requires Qwen3.5-0.8B" >&2
            exit 1
        fi
        LAMRA_MODEL_NAME=qwen35-0.8b
        QWEN35_VOCAB_SIZE=${lamra_vocab_size}
        source "${RELAX_ROOT}/scripts/models/qwen35-0.8B.sh"
        LAMRA_LORA_TARGETS=(linear_proj linear_fc2)
        ;;
    *)
        echo "Unsupported LamRA checkpoint type: ${lamra_model_type}" >&2
        exit 1
        ;;
esac
