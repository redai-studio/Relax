# ReTool with native SGLang and GRPO

[中文](README.md)

This example uses Relax's native Megatron trainer, SGLang engines, checkpointing and periodic evaluation. Generation pauses at `</code>`, sends the new snippet to an independent SandboxFusion service, appends its actual output inside `<interpreter>`, and continues. Generated tokens contribute to the loss; feedback tokens remain context only.

Defaults follow the author-data workflow in `liyuanfan/retool@691aa8062`: Qwen2.5-7B-Instruct, two SFT epochs/50 updates, then 200 GRPO updates. Each update samples 64 questions with replacement and generates 16 trajectories per draw. Temperature/top-p are 1, top-k is -1, both clip bounds are 0.2, learning rate is 1e-6, and KL/entropy coefficients are zero. The pinned DAPO verifier preserves `Answer:` priority with boxed fallback. Reward is answer correctness ±1 minus 0.1 per failed tool call, with no floor or success/call bonus. This is the branch's GRPO variant; the original paper uses PPO and answer-only rewards.

Each call is a stateless Python program limited to 30 seconds/1 GiB. Code errors/timeouts are returned as observations; malformed calls also count as failures. HTTP and service faults raise without execution retries or negative model rewards. Deploy an isolated execution service separately with these limits; the HTTP client does not provide isolation.

## Prepare data

Run from the repository root in the standard Relax training environment. MODEL_DIR identifies the initial HF model, DATA_DIR holds raw/prepared data, and SAVE_DIR holds persistent checkpoints. All participating nodes must have access.

```bash
export MODEL_DIR=/path/to/Qwen2.5-7B-Instruct
export DATA_DIR=/path/to/retool-data
export SAVE_DIR=/path/to/checkpoints

hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 --local-dir "$MODEL_DIR"
hf download JoeYing/ReTool-SFT train_2000.parquet --repo-type dataset \
  --revision 74943ce52f389f16926702302fcff6255875cbb2 --local-dir "$DATA_DIR/author_raw"
hf download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset \
  --revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674 --local-dir "$DATA_DIR/dapo_raw"

python -m examples.retool.prepare_sft \
  --parquet "$DATA_DIR/author_raw/train_2000.parquet" --model "$MODEL_DIR" --out "$DATA_DIR/sft"
python -m examples.retool.prepare_data train \
  --input /path/to/downloaded-dapo-train.parquet --model "$MODEL_DIR" --out "$DATA_DIR/train.jsonl"
```

Point the final command at the complete downloaded training parquet. Conversion checks 1791700 rows and repeated content for 17917 native IDs, quarantines 14 IDs with contradictory gold labels, and retains 17903 questions with original questions/labels. SFT feedback is embedded in assistant text: preprocessing preserves text and token order and masks feedback spans without executing historical code or changing chat protocols.

## Train and resume

Submit to an existing Ray cluster, using its Jobs HTTP address. The scripts do not create or clean a shared cluster. Defaults use eight GPUs, training TP2 and one GPU per SGLang engine. SFT uses batch80 and 16K sequences; GRPO uses prompt2048, response8192 including feedback, context10240, and at most 16 assistant turns.

```bash
export RAY_ADDRESS=http://ray-head:8265
bash scripts/entrypoint/ray-job.sh examples/retool/scripts/run-qwen25-7B-sft-8xgpu.sh

# After SFT finishes, initialize a new GRPO experiment from its HF export.
export MODEL_DIR="$SAVE_DIR/qwen25-7b-retool-sft-8xgpu/hf_49"
export SANDBOX_FUSION_URL=http://sandbox-host:8080/run_code
bash scripts/entrypoint/ray-job.sh examples/retool/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

Rollout workers must reach the sandbox URL. A built-in SGLang processor and existing Megatron logprob/train hooks mask the same vocabulary tail, preserving the source branch's token\<151665 distribution. Different kernels do not guarantee bitwise identical logprobs.

EXP_NAME and PROJECT_NAME override the script defaults. Resume uses stable `$SAVE_DIR/$EXP_NAME` paths with native model/optimizer/scheduler/RNG and data cursors. Save interval is 25 updates; retention is two SFT or eight GRPO checkpoints. Rerunning the same script restores the original target budget. Do not append `--no-load-optim`, `--no-load-rng` or `--finetune` when resuming. Preflight rejects incomplete native state/cursors; loading additionally checks the sampling seed, grouping and data digest. Use a different experiment name or save root for new experiments.

ClearML and metrics service are enabled using existing credentials; W&B is not enabled. Submissions block by default; explicitly set RAY_NO_WAIT=1 for asynchronous submission. Trailing CLI arguments can override defaults while respecting model, grouping and resource constraints.

## Periodic evaluation

Convert the original benchmark records:

```bash
python -m examples.retool.prepare_data eval --benchmark math500 \
  --input /path/to/math500.json --model "$MODEL_DIR" --out "$DATA_DIR/eval/math500.jsonl"
export EVAL_CONFIG="$PWD/examples/retool/eval.yaml"
bash scripts/entrypoint/ray-job.sh examples/retool/scripts/run-qwen25-7B-grpo-8xgpu.sh
```

Inputs may be JSON arrays or JSONL. GSM8K uses question/answer, GSM-Hard uses input/target, and the other benchmarks use problem/answer. The template lists five datasets with tools on/off, for ten panels. Prepare all corresponding files or copy the template and select the required datasets. Evaluation uses K1/greedy and the same token budgets. `tool_enabled=false` selects a direct prompt without constructing an execution client. Metrics read `acc` independently of tool penalties. Experiment owners must handle dataset overlap and contamination analysis.

When enabled, evaluation runs every 25 updates on current rollout engines without restarting training or restoring Adam between panels. Some native evaluation failures are logged as warnings; this example does not add a scheduler that blocks training until every panel passes. Without EVAL_CONFIG, periodic evaluation is disabled.

Run the CPU protocol/reward/sampling-resume regressions with:

```bash
python -m pytest tests/examples/test_retool.py -q
```
