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

- `run_mini_swe_agent.sh`: training entry point.
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
  swebench \
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
