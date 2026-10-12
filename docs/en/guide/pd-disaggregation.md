# Prefill–Decode Disaggregation

Prefill–decode (PD) disaggregation assigns prompt processing and token generation to separate SGLang engines. In a multi-turn coding task, each tool result extends the prompt for the next model request. Separating these two phases lets the prefill engines process new context while decode engines generate responses. The benefit depends on prompt lengths, concurrency, and the cost of transferring KV state; PD is not a throughput guarantee.

The [Mini SWE Agent PD script](../../../examples/mini_swe_agent/run_mini_swe_agent_pd.sh) runs Qwen3.5-35B-A3B on R2E-Gym tasks with **eight GPUs: two prefill engines and two decode engines, each using TP2**. It reuses the existing Mini SWE agent, sandbox, task data, reward computation, and GRPO training recipe. The original [non-PD script](../../../examples/mini_swe_agent/run_mini_swe_agent.sh) remains a separate entry point.

## GPU layout and request flow

| Role during rollout | Engines | GPUs per engine | Total GPUs |
| --- | ---: | ---: | ---: |
| Prefill | 2 | 2 (TP2) | 4 |
| Decode | 2 | 2 (TP2) | 4 |

The trainer and rollout engines use `--colocate` to share the same eight GPUs at different times. Megatron uses TP2, CP4, PP1, EP8, and expert TP1, as in the original Mini SWE recipe. PD separates phases **within rollout inference**; this example does not put training and rollout on separate GPU pools or enable fully async training.

1. Relax creates a Group of eight resident Sessions for one task.
2. Each Session starts `agent_client.sh`, which asks `agent_server.py` to run Mini SWE in an Apptainer sandbox.
3. Mini SWE sends its complete conversation to Relax's agentic Chat Completions API. The PD router selects a prefill engine and a decode engine; prefill state is transferred to decode for generation.
4. The agent executes tools and repeats model requests. After submission, the server runs the task tests and returns a scalar reward. Relax exports the trajectory for GRPO.
5. At the training boundary, Relax updates the weights of all prefill and decode engines before the next rollout.

The recipe uses `consistent_hashing` for the general and prefill policies, keeping a Session routing key for prefill affinity. Decode uses `power_of_two`: the router compares two eligible decode workers when choosing where to generate. These policies do not reserve an engine for the lifetime of a Session.

## Prepare the environment

Use a **dedicated, single-node, eight-GPU Ray cluster** with enough GPU memory for Qwen3.5-35B-A3B at TP2, its KV state, and colocated training. Run the submission command on that node. The canonical Ray entry point cleans residual Relax workers and Serve applications, so run it only when the cluster is available for this job.

Use the repository's training environment and its matching SGLang patch. The current `docker/patch/latest/sglang.patch` targets SGLang v0.5.17. The PD transfer backend is Mooncake; its dependencies must be available in the training environment. A running router alone does not establish that model initialization and state transfer work.

Follow the [Mini SWE setup instructions](../../../examples/mini_swe_agent/README.md#prepare-the-mini-swe-agent-venv) to prepare the venv, parquet task data, Python wheels, and matching SIF images. Training uses `R2E-Gym-Lite`; the agent server also loads the configured `SWE-Bench-Lite` dataset at startup even when Eval is disabled. Use Bash 5.1 or newer, install Apptainer, and make `jq` and `curl` available. Place the Qwen3.5 checkpoint at `$MODEL_DIR/Qwen3.5-35B-A3B`.

## Run the PD example

Use new output directories for each run. Replace the paths, Ray head address, and network interface below with your environment:

```bash
cd /path/to/Relax

export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export MODEL_DIR=/path/to/models
export R2E_DATA_PATH=/path/to/r2e_data
export R2E_SIF_DIR=/path/to/r2e_sifs
export RUN_ID=mini-swe-pd-$(date +%Y%m%d-%H%M%S)
export SAVE_DIR=/path/to/checkpoints/${RUN_ID}
export EXP_ROOT=/path/to/experiments/${RUN_ID}
export RAY_ADDRESS=<ray-head-ip>:6379
export RAY_JOB_ADDRESS=http://<ray-head-ip>:8265
export NCCL_SOCKET_IFNAME=<network-interface>

NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_pd.sh
```

Submission is asynchronous. The Ray driver starts the agent server, waits for HTTP readiness, then starts training from this checkout. The driver owns the server until training exits or the job is stopped. The server binds to the node address detected by the Ray entry point, using port `8765` by default; set `AGENT_SERVER_PORT` if that port is occupied.

The command above runs three rollout/training iterations, with one task and eight samples per iteration. It is a small end-to-end run of the real task: the agent still has up to 40 model calls, a response limit of 8192 tokens per call, and a context limit of 32768 tokens. Task execution and scoring can take substantially longer than model generation alone.

| Setting | Recipe default | Meaning |
| --- | --- | --- |
| `NUM_ROLLOUT` | `1000` | Number of rollout/training iterations; the command above sets `3` |
| `ROLLOUT_BATCH_SIZE` | `16` | Task Groups per iteration; the command above sets `1` |
| Samples per Group | `8` | Matches `r2e.datasets.train.samples` in `agent_config.yaml` |
| `AGENT_SERVER_TRAIN_CONCURRENCY` | `ROLLOUT_BATCH_SIZE * 8` | Server slots; must cover all resident train Sessions |
| `AGENT_SERVER_REWARD_CONCURRENCY` | `24` | Maximum simultaneous task-scoring executions |
| `ENABLE_EVAL` | `0` | Optional SWE-Bench Eval every 100 iterations |
| `AGENT_SERVER_EVAL_CONCURRENCY` | `64` | Separate Eval slots; at least `50` when Eval is enabled |
| `MAX_RESPONSE_LEN` / `MAX_CONTEXT_LEN` | `8192` / `32768` | Per-call generation and context limits |

The global training batch is `ROLLOUT_BATCH_SIZE * 8`. Keep the eight-sample Group size aligned with the agent server configuration. Reducing server slots below the resident Session count can prevent complete Groups from reaching their first model request; the script rejects this configuration. Eval also needs its dataset, SIF images, and the wheels required by the Eval setup commands.

The agent's per-model-request timeout and Relax's agent-process active-time limit are both 7200 seconds; they protect different parts of execution. This recipe leaves prelaunch, partial rollout, session KV lifecycle, and program admission disabled.

## PD configuration

The script passes [pd-2p2d-tp2.yaml](../../../examples/mini_swe_agent/pd-2p2d-tp2.yaml) through `--sglang-config`:

```yaml
# 8 shared GPUs: two prefill engines and two decode engines, each with TP2.
sglang:
  - name: default
    num_gpus_per_engine: 2
    engine_groups:
      - worker_type: prefill
        num_gpus: 4
        overrides:
          enable_hierarchical_cache: false
          hicache_storage_backend: null
      - worker_type: decode
        num_gpus: 4
        overrides:
          enable_hierarchical_cache: false
          hicache_storage_backend: null

```

`num_gpus` is the total allocation for one role; `num_gpus_per_engine` splits it into engines. Four GPUs divided by two GPUs per engine creates two engines for each role. Both roles keep hierarchical caching disabled and leave its storage backend unset. Ordinary backend GPU caching follows SGLang's defaults; this example introduces no multi-level cache configuration.

## Router startup and recovery

With `power_of_two` decode routing, Relax reserves the router endpoint before creating Agentic Sessions, starts the real prefill and decode engines, then starts the router with those workers. It confirms that every expected worker has the correct role, bootstrap port, and healthy registration before publishing the engine groups. An engine adopts its initial registration without adding itself a second time.

This managed startup path requires native, framework-launched SGLang engines with `dp_size=1`, at least one prefill engine, and at least two decode engines. The example provides two of each. External engines, custom engine classes, DP-aware worker identities, and mixed regular/PD groups are outside this path. Use the fixed topology here; dynamic PD scale-out is not supported.

The example enables Relax's fault tolerance. Before replacing a dead managed PD engine, the manager confirms that its old router registration is removed. Replacement registration must become healthy before recovery returns, and the training loop updates recovered engines' weights before the next rollout. If startup or registration validation fails, Relax cleans up the engines and router it owns. This does not promise that every in-flight Agent request survives an engine failure; the run may need a restart when the request cannot be recovered.

To use consistent hashing for decode while keeping the same eight-GPU task and topology, append this argument to the launch command:

```bash
--sglang-router-decode-policy consistent_hashing
```

This uses the existing dynamic registration path. The managed static startup described above applies to `power_of_two`.

## Follow progress and stop

```bash
ray job status --address "$RAY_JOB_ADDRESS" "$RUN_ID"
ray job logs --address "$RAY_JOB_ADDRESS" "$RUN_ID"
tail -f "$EXP_ROOT/agent_server/server.log"

ray job stop --address "$RAY_JOB_ADDRESS" "$RUN_ID"
```

Look for successful task setup, repeated model/tool turns, completed Sessions with rewards, rollout export, and optimizer steps. For a PD run, confirm two prefill and two decode engines are registered and that their weight versions agree after each update. A submitted job or a ready agent server alone does not demonstrate a completed training cycle.

The agent server writes logs and Session events under `$EXP_ROOT/agent_server`; Relax writes rollout details under `$EXP_ROOT/dump`. When training ends, the driver stops its server, whose shutdown handler cleans up its Sessions and Apptainer instances.

If startup fails, inspect the job log and server log before resubmitting. A bind failure means the configured agent port is already in use. Missing parquet, SIF, or wheel errors belong to the Mini SWE setup. PD initialization or transfer errors require checking the model/backend version, Mooncake availability, and communication interfaces. For GPU OOM, first reduce `ROLLOUT_BATCH_SIZE`; preserve the 2P2D TP2 allocation while checking memory limits.

For the resident Session and export contracts, see [Agentic Rollout](./agentic-rollout.md). For separate training and rollout scheduling, see [Fully Async](./fully-async-training.md).
