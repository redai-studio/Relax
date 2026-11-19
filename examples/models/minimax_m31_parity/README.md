# MiniMax-M3.1-Flash multimodal SFT parity

This recipe compares Relax and Miles MSA on the same four-layer BF16 model with language-only rank-16 LoRA. Both runs use Relax's SFT loop, the same frozen vision model, and the official MiniMax processor. The Miles run selects the upstream Miles attention and indexer. The input preflight uses Miles' `MultiTurnLossMaskGenerator` as the reference for supervision.

Use the same two-node, 16-GPU Ray cluster for both runs. The tested Megatron Core and Bridge revisions were `4a4de86579` and `6862ee170a`. Add the Miles checkout at `fa13e2ec42622f6e77aabd8e0c295ab8b4347eac` to `PYTHONPATH` using a directory visible to the Ray workers.

## Prepare the same inputs

Set `MODEL_DIR`, `DATA_DIR`, and `SAVE_DIR` to shared directories and `RAY_ADDRESS` to the Ray Jobs dashboard. Start with a fresh `DATA_DIR` for the synthetic inputs. The `--make-fixture` option generates the PNG and MP4 files in `$DATA_DIR/media` with Pillow and ImageIO's FFmpeg backend.

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

Use a complete official checkpoint so the converter copies the processor modules and chat template. Generate the fixture once and share it between the two runs because video encoding can vary by FFmpeg build. The fixture includes image-only, video-only, mixed, text-only, and long image samples that exercise sparse block selection. Preparation confirms that the full token sequence and mask match for each distinct row, along with the media pixel tensors and grids. It writes an experiment-specific template that includes `</mm:think>` in the supervised suffix, as Miles does on these single-turn answers.

## Run and compare

Submit the jobs in sequence on the same cluster. Use fresh experiment names to start both runs from the BF16 checkpoint. The recipe sets seed 1234, strict Megatron/PyTorch determinism, cuBLAS and Transformer Engine settings, and NCCL Ring/Simple. It retains every step's adapter and optimizer checkpoint.

```bash
export MINIMAX_PARITY=1 SEED=1234 TRAIN_STEPS=10
export MULTIMODAL_KEYS='{"image":"images","video":"videos"}'
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

## Full-parameter SFT

Set `SFT_MODE=full` for both launches and use separate experiment and submission names. The four-layer checkpoint and inputs are the same. The recipe uses TP=4, EP=8, and a learning rate of `1e-5`. It trains the language model, vision tower, and projectors. The MSA block indexer stays frozen because it produces discrete block selections outside autograd.

Add `--full` to the comparison command. Full-parameter parity records SHA-256 digests of the rank-local model parameters, Adam master parameters, and optimizer tensor state after each successful update. Both runs save a regular full-training checkpoint after the last step. For ordinary full-parameter training, set `SFT_MODE=full` and use the same `SAVE_DIR` and `EXP_NAME` to resume.
