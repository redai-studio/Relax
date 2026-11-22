# MiniMax-M3.1-Flash text and image SFT parity

This recipe compares Relax and Miles MSA on the same four-layer BF16 model with language-only rank-16 LoRA. Both runs use Relax's SFT loop, the same frozen vision model, and the official MiniMax processor. The Miles run selects the upstream Miles attention and indexer. The input preflight uses Miles' `MultiTurnLossMaskGenerator` as the reference for supervision.

Use the same two-node, 16-GPU Ray cluster for both runs. The tested Megatron Core and Bridge revisions were `4a4de86579` and `6862ee170a`. Add the Miles checkout at `fa13e2ec42622f6e77aabd8e0c295ab8b4347eac` to `PYTHONPATH` using a directory visible to the Ray workers.

## Prepare the same inputs

Set `MODEL_DIR`, `DATA_DIR`, and `SAVE_DIR` to shared directories and `RAY_ADDRESS` to the Ray Jobs dashboard. Start with a fresh `DATA_DIR` for the synthetic inputs. The `--make-fixture` option generates PNG files in `$DATA_DIR/media` with Pillow.

```bash
export PYTHONPATH="$MILES_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
python -m scripts.tools.convert_minimax_m3_to_bf16 \
    --model-dir "$MODEL_DIR/MiniMax-M3.1-Flash" \
    --save-dir "$MODEL_DIR/MiniMax-M3.1-Flash-4L-BF16" \
    --layers 0,1,3,4 \
    --device cpu
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.tools.minimax_m31_sft_parity prepare \
    --checkpoint "$MODEL_DIR/MiniMax-M3.1-Flash-4L-BF16" \
    --data-dir "$DATA_DIR" \
    --make-fixture
```

Use a complete official checkpoint so the converter copies the processor modules and chat template. Share one fixture between both runs. It contains text-only, single-image, two-image, and long image samples that exercise sparse block selection. Preparation checks the full token sequence, mask, image tensors, and grids for each distinct row. It writes an experiment-specific template that includes `</mm:think>` in the supervised suffix, as Miles does on these single-turn answers.

## Run and compare

Submit the jobs in sequence on the same cluster. Use fresh experiment names to start both runs from the BF16 checkpoint. The recipe sets seed 1234, strict Megatron/PyTorch determinism, cuBLAS and Transformer Engine settings, and NCCL Ring/Simple. It retains every step's adapter and optimizer checkpoint.

```bash
export MINIMAX_PARITY=1 SEED=1234 TRAIN_STEPS=10
export MULTIMODAL_KEYS='{"image":"images"}'
export SFT_CHAT_TEMPLATE="$DATA_DIR/miles-mask.jinja"
export RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_NO_WAIT=
recipe=scripts/training/sft/run-minimax-m3.1-flash-4l-lora-sft-16xgpu.sh

MINIMAX_MSA_IMPL=relax EXP_NAME=parity-relax SUBMISSION_ID=parity-relax \
    bash scripts/entrypoint/ray-job.sh "$recipe"
MINIMAX_MSA_IMPL=miles EXP_NAME=parity-miles SUBMISSION_ID=parity-miles \
    bash scripts/entrypoint/ray-job.sh "$recipe"

ray job logs --address="$RAY_ADDRESS" parity-relax > "$DATA_DIR/relax.log"
ray job logs --address="$RAY_ADDRESS" parity-miles > "$DATA_DIR/miles.log"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.tools.minimax_m31_sft_parity compare \
    --logs "$DATA_DIR/relax.log" "$DATA_DIR/miles.log" \
    --checkpoints "$SAVE_DIR/parity-relax" "$SAVE_DIR/parity-miles"
```

The comparison checks loss, gradient norm, global batch size, mean packed token count, and every stored adapter and optimizer tensor at each of the ten steps.

## Independent Relax and Miles SFT runs

These runs use Relax's SFT entry point and Miles' `train_async.py` in train-only mode. They share the BF16 checkpoint,
input rows, assistant-mask template, and seed 1234. The reference data was selected from OpenSearch-VL. Before
launch, match token IDs and supervised positions for each ordered batch of four rows. For image inputs, also match
`pixel_values` and `image_grid_thw`. Prepare the shared JSONL as `$DATA_DIR/minimax-m3.1-flash-sft.jsonl`, with
`messages`, `tools`, and `images` fields as applicable, and set `SFT_CHAT_TEMPLATE` to the verified template.
Use the identical physical row order in Miles and Relax by supplying Miles with the seed-1234 Relax index order
in `MINIMAX_E2E_INDEX_ORDER`.

| Mode | Layers | TP  | EP  | Learning rate |
| ---- | ------ | --- | --- | ------------- |
| LoRA | 4      | 4   | 16  | 1e-4          |
| LoRA | 60     | 4   | 16  | 1e-4          |
| Full | 4      | 4   | 8   | 1e-5          |

Use two eight-GPU nodes and PP=CP=1 for each pair. The 60-layer case uses
`$MODEL_DIR/MiniMax-M3.1-Flash-BF16`, produced by omitting `--layers` from the converter command above.
Set `SFT_MODE` and `MINIMAX_PARITY_LAYERS` for the pair. Leave `MULTIMODAL_KEYS` unset for text-only runs,
or export `MULTIMODAL_KEYS='{"image":"images"}'` for image runs.

The Miles checkout in the recorded experiment is pinned to `fa13e2ec42622f6e77aabd8e0c295ab8b4347eac`
with the experiment overlay from `reproduce/sources-v2.tar.gz`
(SHA-256 `84761b687a8fae4f787242bb4b871da30446a35ed3b83eda4870bbe0c5db5bcf`). The source archive is
an external prerequisite for this recipe. Set `MILES_PARITY_CHECKOUT` to its restored `Miles/` directory and
`MILES_PARITY_PYTHONPATH` to the shared runtime module paths for the pinned Core, Bridge, and offline input
processor. The overlay's `miles_plugins/megatron_bridge/__init__.py` imports `minimax_vlm`, whose registered
`MilesMiniMaxVLMBridge` installs Miles' `miles_plugins.models.minimax_m3.minimax_m3.get_minimax_m3_spec`.
Use that registration on Bridge `6862ee170a`. The default Bridge in the pinned Miles image uses dense
attention. The overlay also forwards the deterministic provider options and includes the offline SFT callbacks.
For the image callback, populate the experiment metadata's `e2e_images` with the same paths as `images`,
`e2e_videos` with an empty array, and `tools` with the same tool definitions as the Relax row.
Keep the archived input processor importable for that callback.

Launch Relax first. Use a fresh `EXP_NAME` for each paired run.

```bash
export MINIMAX_NATIVE_PARITY=1 MINIMAX_MSA_IMPL=relax SEED=1234 TRAIN_STEPS=10
export SFT_MODE="${SFT_MODE:-lora}" MINIMAX_PARITY_LAYERS="${MINIMAX_PARITY_LAYERS:-4}"
export RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_NO_WAIT=
EXP_NAME=native-parity-relax SUBMISSION_ID=native-parity-relax \
    bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-minimax-m3.1-flash-4l-lora-sft-16xgpu.sh
```

Run the following from Relax in Bash with the same environment. Copy the loss callback into the restored
experiment checkout to include the framework marker used by the comparison command.

```bash
cp scripts/tools/minimax_m31_text_parity.py "$MILES_PARITY_CHECKOUT/"
model="$MODEL_DIR/MiniMax-M3.1-Flash-4L-BF16"
model_spec=minimax-m3.1-flash-4layer.py
if [[ "$MINIMAX_PARITY_LAYERS" == 60 ]]; then
    model="$MODEL_DIR/MiniMax-M3.1-Flash-BF16"
    model_spec=minimax-m3.1-flash.py
fi
args_path="$MILES_PARITY_CHECKOUT/miles/utils/external_utils:$MILES_PARITY_CHECKOUT/scripts/models"
mapfile -t spec < <(PYTHONPATH="$args_path${PYTHONPATH:+:$PYTHONPATH}" \
    python - "$MILES_PARITY_CHECKOUT/scripts/models/$model_spec" <<'PYTHON'
import runpy
import shlex
import sys

args = shlex.split(runpy.run_path(sys.argv[1])["model_args"]())
args.remove("--moe-permute-fusion")
print("\n".join(args))
PYTHON
)
ep=8
lr=1e-5
lora=()
if [[ "$SFT_MODE" == lora ]]; then
    ep=16
    lr=1e-4
    prefix='model.language_model.layers.*.self_attn'
    lora=(--lora-train-only --lora-rank 16 --lora-alpha 32 --lora-type lora --lora-dropout 0
          --target-modules "$prefix.q_proj,$prefix.k_proj,$prefix.v_proj,$prefix.o_proj")
fi
text_only=1
input_hook=minimax_miles_sft.generate_rollout
if [[ -n "${MULTIMODAL_KEYS:-}" ]]; then
    text_only=0
    input_hook=minimax_miles_media.generate_rollout
fi

runtime_env=$(jq -cn \
    --arg path "./miles/utils/external_utils:./scripts/models:$MILES_PARITY_PYTHONPATH" \
    --arg order "$MINIMAX_E2E_INDEX_ORDER" --arg text "$text_only" \
    '{env_vars: {PYTHONPATH: $path, MINIMAX_E2E_INDEX_ORDER: $order, MINIMAX_E2E_TEXT_ONLY: $text,
      NCCL_ALGO: "Ring", NCCL_PROTO: "Simple", NCCL_NVLS_ENABLE: "0", CUDA_DEVICE_MAX_CONNECTIONS: "1",
      CUBLAS_WORKSPACE_CONFIG: ":4096:8", NVTE_ALLOW_NONDETERMINISTIC_ALGO: "0", NVTE_FRAMEWORK: "pytorch",
      PYTHONHASHSEED: "0", MILES_FILL_UNINIT: "1", RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES: "1"}}')
ray job submit --address="$RAY_ADDRESS" --submission-id=native-parity-miles \
    --entrypoint-num-cpus=1 --working-dir="$MILES_PARITY_CHECKOUT" --runtime-env-json="$runtime_env" \
    -- python -u train_async.py "${spec[@]}" "${lora[@]}" \
    --hf-checkpoint "$model" --ref-load "$model" --load "$model" --megatron-to-hf-mode bridge \
    --train-backend megatron --debug-train-only --debug-skip-weight-update --disable-compute-advantages-and-returns \
    --rollout-function-path "$input_hook" --data-source-path minimax_miles_sft.PairedOpenSearchDataSource \
    --prompt-data "$DATA_DIR/minimax-m3.1-flash-sft.jsonl" --chat-template-path "$SFT_CHAT_TEMPLATE" \
    --input-key messages --tool-key tools --metadata-key metadata --no-rollout-shuffle --balance-data \
    --num-rollout 10 --rollout-batch-size 4 --global-batch-size 4 --n-samples-per-prompt 1 \
    --loss-type sft_loss --calculate-per-token-loss --use-dynamic-batch-size \
    --custom-megatron-init-path minimax_m31_text_parity.configure_miles_worker \
    --max-tokens-per-gpu 9216 --seq-length 9216 --max-position-embeddings 1048576 \
    --qkv-format thd --data-pad-size-multiplier 128 \
    --actor-num-nodes 2 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 \
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --expert-model-parallel-size "$ep" --expert-tensor-parallel-size 1 --sequence-parallel \
    --moe-token-dispatcher-type alltoall --moe-router-load-balancing-type none --moe-aux-loss-coeff 0 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --optimizer adam --lr "$lr" --lr-decay-style constant --weight-decay 0 \
    --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1 \
    --bf16 --no-bias-swiglu-fusion --attention-backend flash --attention-dropout 0 --hidden-dropout 0 \
    --deterministic-mode --seed 1234 --log-probs-chunk-size 256
```

Compare the ten reported loss and gradient-norm values using the framework markers in the two logs.
In native parity mode, Relax saves a checkpoint at the final step and omits the per-step rank digests.

```bash
ray job logs --address="$RAY_ADDRESS" native-parity-relax > "$DATA_DIR/native-relax.log"
ray job logs --address="$RAY_ADDRESS" native-parity-miles > "$DATA_DIR/native-miles.log"
python -m scripts.tools.minimax_m31_text_parity \
    --logs "$DATA_DIR/native-relax.log" "$DATA_DIR/native-miles.log" --steps 10
```

## Full-parameter SFT with both MSA implementations

In [Run and compare](#run-and-compare), set `SFT_MODE=full` for both launches and use separate experiment and submission names. The four-layer checkpoint and inputs are the same. The recipe uses TP=4, EP=8, and a learning rate of `1e-5`. It trains the language model, vision tower, and projectors. The MSA block indexer stays frozen because it produces discrete block selections outside autograd.

Add `--full` to the comparison command. Full-parameter parity records SHA-256 digests of the rank-local model parameters, Adam master parameters, and optimizer tensor state after each successful update. Both runs save a regular full-training checkpoint after the last step. For ordinary full-parameter training, set `SFT_MODE=full` and use the same `SAVE_DIR` and `EXP_NAME` to resume.

### Resume after router precision changes

For full-parameter MiniMax-M3.1-Flash and mixed-attention MiniMax-M3, Relax now puts trainable router weights in
FP32 buckets. A checkpoint saved with BF16 router weights and the `dp_reshardable` optimizer layout has different
bucket sizes. Start a new run from the old model weights by setting `--load` to the old checkpoint root,
`--save` to a fresh root, and adding `--finetune` on the training command. This starts new optimizer, scheduler,
and RNG state. Keep optimizer-state resume for checkpoints already saved with the FP32 router layout.
