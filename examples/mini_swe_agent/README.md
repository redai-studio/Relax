# Mini SWE Agent Example

Relax agentic rollout integration for R2E-Gym / SWE-Bench tasks using
`mini-swe-agent` installed from PyPI and a central agent server.

This example has two purposes:

- It shows how to train a coding agent on R2E-Gym / SWE-Bench style tasks.
- It shows how to train with a remote agent runtime. The process launched by
  Relax can be a lightweight entrypoint, while the actual agent, sandbox, data
  access, and scoring logic run behind an agent server.

This example uses `mini-swe-agent` from a Python virtual environment. A local
mini-swe-agent source checkout is unnecessary.

This example builds on:

- [R2E-Gym/R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) provides the R2E-Gym
  task format, containerized SWE environments, and execution-based scoring
  semantics used by the training split.
- [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
  provides the lightweight software-engineering agent used by the central
  agent server.

It demonstrates three agentic rollout patterns:

- Running a cheap local managed-agent entrypoint for each Relax session.
- Serving task rows and executing mini-swe-agent from a central agent server.
- Controlling execution concurrency outside the Relax group scheduler.

The remote-agent contract is intentionally small: Relax starts a local process,
the local process asks the agent server to run one session, the server talks back
to Relax through the OpenAI-compatible agentic chat API, and the local process
exits after writing the final Relax output JSON. The agent server may live on
another machine as long as the intermediate chat messages follow the expected
format and the client process exits when the remote run finishes.

## Files

- `run_mini_swe_agent.sh`: original non-PD training entry point.
- `run_mini_swe_agent_pd.sh`: 8-GPU, 2P2D TP2 training entry point.
- `pd-2p2d-tp2.yaml`: pure PD engine allocation without hierarchical caching.
- `run_mini_swe_agent_hicache.sh`: 8-GPU Mini SWE with CPU caching in regular (default) or PD deployment.
- `hicache-cpu-4x-tp2.yaml` / `pd-hicache-cpu-2p2d-tp2.yaml`: regular and PD CPU cache configurations.
- `scripts/run-qwen35-35B-A3B-8xgpu-hicache-fully-async.sh`: four training GPUs plus four rollout GPUs,
  with oversampling and resumable eager Session scheduling.
- `hicache-cpu-2x-tp2-blackwell.yaml`: two TP2 rollout engines for the fully async recipe.
- `pd-hicache-cpu-1p1d-dp2-tp2-ep4.yaml`: one prefill and one decode engine with attention DP2 × TP2, EP4, and CPU cache.
- `agent_server.py`: HTTP service that leases task rows, runs mini-swe-agent,
  controls execution concurrency, scores task output, and returns final results.
- `agent_client.sh`: per-session cheap entrypoint launched by Relax.
- `setup_r2e_data_and_sifs.sh`: one-time setup for local parquet data and SIF
  images.

## Prepare the mini-swe-agent venv

Create the venv on node-local storage for high-concurrency runs. The agent
processes import `mini-swe-agent`, LiteLLM, OpenAI, and related packages in
parallel, so putting the venv on a shared FUSE mount can serialize startup.

```bash
uv venv /local/path/mini-swe-agent-venv --python 3.12
source /local/path/mini-swe-agent-venv/bin/activate

uv pip install \
  mini-swe-agent==2.4.1 \
  flask \
  pyarrow \
  datasets \
  swebench==4.1.0 \
  huggingface_hub
```

Export the venv path when running setup or training:

```bash
export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
```

`setup_r2e_data_and_sifs.sh` also requires command-line tools outside Python:

```bash
# Required commands:
# - hf
# - duckdb
# - jq
# - apptainer https://apptainer.org/docs/admin/main/installation.html#install-debian-packages
```

`setup_r2e_data_and_sifs.sh` requires the DuckDB CLI (`duckdb` on `PATH`).

Install and expose the DuckDB CLI with:

```bash
curl https://install.duckdb.org | sh
echo 'export PATH="/root/.duckdb/cli/latest":$PATH' >> ~/.bashrc
source ~/.bashrc
```

## Prepare data and SIF images

The setup step downloads parquet data and Python wheels once, then builds local
SIF files for the Docker images referenced by the train/eval datasets.

```bash
cd /path/to/Relax

export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export R2E_DATA_PATH=/path/to/r2e_data
export R2E_SIF_DIR=/path/to/r2e_sifs

bash examples/mini_swe_agent/setup_r2e_data_and_sifs.sh
```

If the setup process needs an HTTP proxy, scope it to this command:

```bash
http_proxy=http://host:port https_proxy=http://host:port \
  bash examples/mini_swe_agent/setup_r2e_data_and_sifs.sh
```

Training and agent execution read local parquet/SIF files and do not need proxy
variables.

## Launch training

```bash
cd /path/to/Relax

export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export MODEL_DIR=/path/to/models
export SAVE_DIR=/path/to/checkpoints
export R2E_DATA_PATH=/path/to/r2e_data
export R2E_SIF_DIR=/path/to/r2e_sifs

bash examples/mini_swe_agent/run_mini_swe_agent.sh
```

By default the script uses `AGENT_SERVER_TRAIN_CONCURRENCY=64` (reduced from a
full rollout step of `ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT = 16 * 8 = 128`
to cut concurrent apptainer launch contention). Set
`AGENT_SERVER_TRAIN_CONCURRENCY` and `AGENT_SERVER_EVAL_CONCURRENCY` when
changing the server-side execution limits.

The server uses FIFO scheduling plus one semaphore. `group_id` is used for task
row reuse across samples from the same prompt, not for launch ordering.

The script launches `agent_server.py` and passes its URL to Relax managed agent
processes:

```bash
--agent-env "AGENT_SERVER_URL=${AGENT_SERVER_URL}" "AGENT_CLIENT_TRACE_DIR=${AGENT_SERVER_WORK_DIR}/client_events"
```

Keep these values in a single `--agent-env` occurrence. The Relax argument uses a
list value, so repeating `--agent-env` can drop earlier entries.

## Train with separate training and rollout GPUs

The fully async Blackwell recipe uses TP4/CP1/EP4 for training and two regular
TP2 HiCache engines on separate GPUs. It defaults to GBS 256, 32 accepted Groups
of eight members, an oversampling batch of 48 Groups, and 128 physical Session
slots. Each completed Group stays intact. Eight TransferQueue batches stream
the partition into training, with one optimizer update per partition.

After preparing the model, data and venv paths above, set `EXP_ROOT` to a new
directory for this run and launch:

```bash
bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/scripts/run-qwen35-35B-A3B-8xgpu-hicache-fully-async.sh
```

The recipe uses a 131072-token context, 100 agent turns and 20 training steps.
It saves rollout evidence but **does not save checkpoints**. ClearML and metrics
are enabled using the existing credentials; `PROJECT_NAME`, `EXP_NAME` and
`RUN_ID` customize experiment naming. Submission blocks by default; set
`RAY_NO_WAIT=1` for an externally monitored run.

`AGENTIC_SCHEDULING=incremental` enables physical per-Session release,
incremental member startup and `--agentic-eager-session-generation`. A member
can generate as soon as its first request is ready, before its peers. Unfinished
Sessions keep their identities, accumulated token/logprob/version history and
physical credits across rollout boundaries. Credits return only after confirmed
external resource release. `AGENTIC_SCHEDULING=group` is a whole-Group admission
control with the same 128-slot physical upper bound; `atomic` uses the capacity
ledger with whole-Group startup.

`OVER_SAMPLING_BATCH_SIZE` controls logical candidate Groups, independently of
the physical slot limit. Failed or filtered Groups are replaced using the
existing rules. Setup/reward test timeout remains 300 seconds and is still a
failure; a timeout is not converted to reward zero. Synchronous oversampling
also supports the new eager flag and explicitly cancels and joins surplus work
before releasing its resources.

This recipe defaults to `AGENT_SERVER_SESSION_MEMORY_MIB=8192`. A private
cgroup-v1 memory leaf caps the aggregate worker, container, tools and descendants
of each Session, including detached subprocesses. The supervisor remains outside
the leaf to verify cleanup. A kernel OOM kill fails the Session and replaces its
whole Group, even if a tool parent subsequently reports success. Terminal events
retain the cap, peak, OOM counter and release evidence. Startup fails if the host
does not provide a writable, hierarchical cgroup-v1 memory controller with swap
accounting. Other recipes leave this optional limit disabled by default (`0`).
Budget the cap times physical slots together with training CPU offload and
HiCache host pools; the cap is not a reservation. Use the same limit in both
arms of a scheduling comparison.

The default is `MAX_STALENESS=1`. `MAX_STALENESS=0` requires the current partition to be complete before waiting
for training; it still overlaps streamed training with the rest of that
partition's rollout. Positive staleness permits interrupted Groups to carry
partition debt into the next step; older debt is filled first and final backfill
opens no extra training partition. The resumable eager flag currently rejects
partial rollout, prelaunch and dynamic global batch size. The older
`--agentic-fixed-step-eager` retains its strict synchronous, no-oversampling
contract.

Each TP2 engine has a 4194304-token KV pool and 1332 Mamba states, with CPU
HiCache ratio 2. These caches share host memory with training offload, the
TransferQueue and agent containers. Budget their combined peak when adapting
the recipe to a different machine. The staleness limit applies to physical
training partitions, not to the weight-version age of every retained token.

The model client's read deadline is disabled only for this resumable recipe,
so parked requests survive training pauses. Connection/write/pool deadlines and
the Runtime's active agent budget remain enforced. TIS uses the actual behavior
logprobs, including retained trajectories; inline old logprobs are valid here
because dropout is disabled and each partition receives one optimizer update.

## Train with prefill–decode disaggregation

`run_mini_swe_agent_pd.sh` uses the same Mini SWE task pipeline with two prefill
and two decode engines, each TP2, sharing eight GPUs with the trainer. The
original `run_mini_swe_agent.sh` is unchanged. Use the PD script through
`scripts/entrypoint/ray-job.sh` on a dedicated single-node Ray cluster; its Ray
driver owns the agent server so asynchronous submission keeps the server alive.

See the [English PD guide](../../docs/en/guide/pd-disaggregation.md) or
[中文 PD 指南](../../docs/zh/guide/pd-disaggregation.md) for general PD
configuration, resource allocation, routing policies, and limitations.

## Train with CPU caching

`run_mini_swe_agent_hicache.sh` enables GPU+CPU caching independently of the
deployment topology. `HICACHE_DEPLOYMENT=regular` is the default: four regular
TP2 engines each handle prefill and decode. Set `HICACHE_DEPLOYMENT=pd` for two
prefill and two decode TP2 engines, with CPU cache enabled on the prefill side.
Both layouts share eight GPUs with the trainer and use the existing Ray driver
and Mini SWE training settings from `run_mini_swe_agent_pd.sh`; the selected
YAML determines the actual topology. No storage directory is required.

The supplied YAML targets Qwen3.5-35B-A3B with Unified Full KV and Mamba state
and the repository's patched SGLang v0.5.17. The default recipes use single-node
DP1; an additional 1P1D recipe uses attention DP2 × TP2 and EP4. Complete the venv,
data, and SIF setup above, then set the paths for this run:

```bash
cd /path/to/Relax

export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export MODEL_DIR=/path/to/models
export R2E_DATA_PATH=/path/to/r2e_data
export R2E_SIF_DIR=/path/to/r2e_sifs
export RUN_ID=mini-swe-cpu-cache-$(date +%Y%m%d-%H%M%S)
export SAVE_DIR=/path/to/checkpoints/${RUN_ID}
export EXP_ROOT=/path/to/experiments/${RUN_ID}
export RAY_ADDRESS=<ray-head-ip>:6379
export RAY_JOB_ADDRESS=http://<ray-head-ip>:8265
export NCCL_SOCKET_IFNAME=<network-interface>
```

Submit regular CPU caching on a dedicated single-node Ray cluster through the
canonical Ray entry point:

```bash
HICACHE_DEPLOYMENT=regular NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_hicache.sh
```

For PD with CPU caching, use fresh run and output paths and change the topology:

```bash
HICACHE_DEPLOYMENT=pd NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_hicache.sh
```

For one prefill and one decode engine, each using attention DP2 × TP2 and EP4,
first build and expose the [paired router patch](../../docker/patch/sglang-router/README.md).
The pinned R3 router wheel alone does not implement this PD DP-aware dispatch
correctly. Keep the router overlay on `PYTHONPATH` for the driver and Ray actors,
then submit with fresh run and output paths:

```bash
HICACHE_DEPLOYMENT=pd NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_hicache.sh \
  --sglang-config examples/mini_swe_agent/pd-hicache-cpu-1p1d-dp2-tp2-ep4.yaml \
  --rollout-num-gpus-per-engine 4 \
  --sglang-dp-size 2 --sglang-ep-size 4 --sglang-enable-dp-attention \
  --sglang-router-decode-policy consistent_hashing
```

Each physical engine occupies four GPUs. SGLang reports `tp_size=4`, which
becomes attention TP2 with DP2; expert parallelism spans all four ranks. Global
DP options also enable the DP-aware router. The recipe uses consistent hashing
on each side; Relax sends the same routing key for all turns in a session.
Prefill and decode select their ranks independently, and the decode request must
carry the selected prefill rank for KV transfer. Each attention-DP rank owns a
separate CPU cache. Check the actual P/D rank on completed requests, rather than
inferring cache ownership from the router's virtual worker index.

The Ray driver owns the agent server until training ends, including asynchronous
submission. Each command runs three training iterations, each with one task
group and eight samples. L3 storage is not supported by these recipes.

The [regular CPU YAML](./hicache-cpu-4x-tp2.yaml) and
[PD CPU YAML](./pd-hicache-cpu-2p2d-tp2.yaml) use 64-token pages, a 256-token
Mamba tracking interval, 8192-token prefill chunks, and 666 GPU Mamba slots per
cache-enabled rank. These are recipe choices. Neither YAML caps
`max_total_tokens`; SGLang derives its capacity from the memory budget. The
[1P1D DP2 YAML](./pd-hicache-cpu-1p1d-dp2-tp2-ep4.yaml) supplies 16384-token
chunks and 1332 Mamba slots before SGLang divides these across two attention-DP
ranks, yielding the same 8192/666 values per rank. Set
`SGLANG_MAX_TOTAL_TOKENS` when an explicit cap is needed for a controlled run.

Budget Full KV and Mamba host pools across all eight regular TP ranks or all
four prefill TP ranks in PD, plus training offload and other processes.
`hicache_ratio=2.0` scales with device capacity; use actual startup allocations
to budget memory. All CPU configurations set `hicache_storage_backend: null`.

See the [English CPU cache guide](../../docs/en/guide/hierarchical-cache.md) or
[中文 CPU 缓存指南](../../docs/zh/guide/hierarchical-cache.md) for configuration,
capacity planning, cache-hit checks, and weight invalidation.

## Runtime data flow

1. Relax creates placeholder prompts from a local dummy JSONL file.
2. Relax launches one cheap `agent_client.sh` per session.
3. The client sends `session_id`, `group_id`, `mode`, `base_url`, and `api_key`
   to `agent_server.py`.
4. The server leases the task row, resolves the row's `docker_image` to a local
   SIF, and runs mini-swe-agent in FIFO order under its concurrency limits.
5. The server-side mini-swe-agent model talks to Relax through the
   OpenAI-compatible agentic chat API.
6. `agent_server.MyAgent` runs the task tests and returns the final
   `exit_status`, `submission`, and scalar `reward`.
7. The client writes Relax's `RELAX_OUTPUT_JSON` locally and exits with the
   server-provided exit code.
8. Relax finalizes the session into rollout samples and rewards.

This split is the reusable part of the example. The local entrypoint is only a
bridge between Relax's managed-agent lifecycle and the remote runtime. A
different agent server can replace `mini-swe-agent` if it accepts the same
session request, emits OpenAI-compatible chat turns during execution, returns
the final fields required by the client, and lets the local process terminate
cleanly.

Server logs stay with the `agent_server.py` process log.

`agent_client.sh` sends `/cancel` when it receives `SIGTERM` or `SIGINT`, so
discarded Relax sessions also stop the remote mini-swe-agent process.
