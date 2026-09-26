# Acquire devices

Reserve accelerator devices on a shared Linux runner host for the remaining
steps of a job. The post action releases the reservation at job completion,
including normal failure and cancellation cleanup.

The action name and allocation core are independent of the accelerator vendor.
The `nvidia` backend discovers whole GPUs using
`nvidia-smi`, resolves physical indices to UUIDs, and exports
`CUDA_VISIBLE_DEVICES`. The `ascend` backend discovers compute chips using
`npu-smi info -m` and exports their physical IDs through `ASCEND_VISIBLE_DEVICES`
for Ascend Docker Runtime. Both backends share the same lock lifecycle.
Unsupported backends fail explicitly; virtual NPU, XPU, and MIG allocation is
not supported.

## Usage

```yaml
jobs:
  test:
    runs-on: [self-hosted, linux, gpu]
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v7
      - uses: ./.github/actions/acquire-devices
        id: devices
        with:
          backend: nvidia
          count: 4
          timeout: 1800
      - run: python -m pytest tests/
```

`./` loads the action from the checked-out workspace. Run `actions/checkout`
before this step; the action uses that checkout's version without a separate
action repository download.

## Inputs and outputs

| Input      | Default                | Meaning                                                                                                                            |
| ---------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `backend`  | `nvidia`               | Device discovery/visibility backend: `nvidia` or `ascend`.                                                                         |
| `count`    | `1`                    | Positive integer; reserve this many devices from the candidate pool.                                                               |
| `devices`  | All discovered devices | Optional comma-separated candidate pool of physical `nvidia-smi` indices or full GPU UUIDs. It is a pool, not an additional count. |
| `timeout`  | `1800`                 | Nonnegative integer seconds waiting for locks, after discovery. `0` tries once. This does not expire an acquired reservation.      |
| `lock-dir` | `/tmp/acquire-devices` | Absolute local directory shared by all competing runners on the host.                                                              |

Outputs are `devices` (comma-separated NVIDIA UUIDs or Ascend chip physical IDs) and `count`. NVIDIA
allocations also set `CUDA_VISIBLE_DEVICES` for subsequent steps. No Python or
Node package installation is needed; the runner needs Python 3.9+ and
`nvidia-smi` or `npu-smi` for the selected backend. Node 24 is supplied by the Actions runner.

Use `devices` to restrict allocation to the host devices assigned to CI, for
example `devices: '0,1,2,3'` with `count: 2`. Indices and UUIDs for the same GPU
share one lock; duplicate physical devices are rejected. If the runner already
sets `CUDA_VISIBLE_DEVICES`, specify `devices` explicitly to avoid confusing
CUDA logical indices with physical `nvidia-smi` indices. The explicit pool is
authoritative and the action replaces the inherited visibility value.

When devices are unavailable, the action logs the candidate pool's lock status
immediately and every 30 seconds while waiting: elapsed time, available count,
and available/locked devices as `physical-index (UUID)`. The successful reservation
log uses the same format. Indices are the host's `nvidia-smi` indices; Action
outputs and visibility variables use UUIDs. The waiting log reports cooperative
device reservations, not GPU utilization or memory usage.

## Docker workloads

Acquire on the host **before** creating the workload container, then pass the
UUID output to Docker. A step action cannot configure the device allocation of
an already-started job-level `container:`.

```yaml
- uses: actions/checkout@v7
- uses: ./.github/actions/acquire-devices
  id: devices
  with:
    backend: nvidia
    count: 4
    timeout: 1800
- name: Create workload container
  env:
    CI_DEVICES: ${{ steps.devices.outputs.devices }}
  run: |
    docker create --rm --init --name "$CI_CONTAINER_NAME" \
      --gpus "\"device=$CI_DEVICES\"" \
      "$CI_IMAGE" sleep infinity
    docker start "$CI_CONTAINER_NAME"
- name: Run tests
  run: docker exec "$CI_CONTAINER_NAME" python -m pytest tests/
- name: Stop workload before releasing devices
  if: always()
  run: |
    if docker inspect "$CI_CONTAINER_NAME" >/dev/null 2>&1; then
      docker rm --force "$CI_CONTAINER_NAME"
    fi
```

Set `CI_IMAGE` and a unique `CI_CONTAINER_NAME` in the job environment. Prepare
the test workspace/dependencies in the container as appropriate. Container
cleanup runs before the action's post step. Do not pass host numeric indices
into the container's `CUDA_VISIBLE_DEVICES`; use UUIDs or let Docker expose
only the selected devices.

For Ascend, use `backend: ascend` and pass the allocation as
`--runtime ascend --env "ASCEND_VISIBLE_DEVICES=$CI_DEVICES"`. The optional
`devices` pool contains chip physical IDs from `npu-smi info -m`, not board IDs
or container logical IDs. A dual-chip 910C card contributes two devices, so
`count: 4` reserves four chips (two cards when selecting paired chips).
Inside a four-chip container, use `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
Do not copy host physical IDs into that variable. All jobs competing for these
chips must use this backend and the same host lock directory.

## Lock and lifecycle contract

- All consumers must cooperate through this action and the **same host lock
  directory**, including jobs in other repositories. Ordinary CUDA processes
  do not honor file locks. Visibility variables are not a security boundary.
- Each physical device has one kernel `flock`. Acquisition is all-or-none:
  partial allocations are released before retrying. There is no FIFO guarantee.
- Locks cover only this host. Use a local filesystem, not NFS. Containerized
  runners must bind-mount the same host lock directory and run as users with
  compatible filesystem permissions. The action never deletes shared lock
  files, since doing so can break mutual exclusion.
- A small Python process holds the locks after the main step exits. The post
  step releases it through a private control directory. The action preserves
  `RUNNER_TRACKING_ID` for the runner's final orphan-process cleanup.
- Stop all workload containers, Ray services, and background processes before
  the post step. Register any cleanup actions after acquisition, because post
  steps execute in reverse order. The action does not own or stop workloads.
  If workload cleanup fails, the post step still releases the reservation;
  the host supervisor must handle any remaining workloads.
- A hard runner crash or force-killed holder cannot guarantee workload cleanup.
  An abandoned live holder intentionally keeps its locks until stopped; there
  is no TTL that silently frees devices still used by a workload. Recover by
  stopping the abandoned workloads and their holder on the host, never by
  deleting shared lock files. Runner orphan cleanup is only a fallback.
