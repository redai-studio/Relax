# Native router support for R3 and PD DP-aware routing

`v0.3.2-9daabcd-pd-dp-aware.patch` applies to the R3-capable
[`zhuzilin/sgl-router` fork at `9daabcd801c861088d93b8ce8a2a831490ce4bee`](https://github.com/zhuzilin/sgl-router/tree/9daabcd801c861088d93b8ce8a2a831490ce4bee).
Both CUDA Dockerfiles (`docker/Dockerfile` and `docker/Dockerfile.cu13`) build
and install this patched router automatically. Use it together with Relax's
SGLang v0.5.17 patch for the
[1P1D attention-DP2 recipe](../../../examples/mini_swe_agent/README.md#train-with-cpu-caching).
CPU cache itself also supports centralized deployment; it does not require PD
separation or DP-aware routing.

The patch keeps virtual worker identities and per-rank metrics, while sending
HTTP requests and KV bootstrap to physical endpoints. It passes independent
`routed_dp_rank` values to prefill and decode, plus `disagg_prefill_dp_rank` to
decode. Raw native fields, including R3 `routed_experts`, remain intact. The
paired SGLang patch also accepts the legacy `data_parallel_rank` field used by
this router's regular path; an explicit `routed_dp_rank` takes precedence.

## R3 installation requirement

`--use-rollout-routing-replay` requires raw `/generate` JSON forwarding so
`return_routed_experts` and `routed_experts_start_len` reach SGLang. The stock
PyPI package accepted by the non-R3 dependency range can discard these fields.
Increasing the version floor alone does not establish this capability.

Relax checks for `0.3.2+slime.dpaware` and the pinned source commit above in
`_relax_build.json` during argument validation and again on the Ray worker that
starts a managed router. This is a supported-build check, not a generic remote
capability probe. Keep the packaged record with the wheel. External router
operators must deploy a router with the same R3 forwarding capability; Relax
cannot infer that capability from the client's installed package. Non-R3 jobs
continue to use the normal dependency range.

## Image build

The `router_builder` stage uses the same `TRAIN_IMAGE` as the final image,
including when an external training image is supplied. It installs Rust 1.97.1
and maturin 1.9.6, verifies the pinned source archive, applies the router patch,
and builds a wheel with the checked-in Cargo lockfile and `--locked`. Rust and
build-only dependencies remain in the builder stage. `ROUTER_BUILD_JOBS`
(default: 16) controls build parallelism.

The final image installs the wheel after Python requirements, replacing any
inherited stock router. Its version is `0.3.2+slime.dpaware`. The packaged
`sglang_router/_relax_build.json` records the source commit, archive checksum,
patch checksum, lockfile checksum and Rust version. The image build verifies
that the native extension imports and the expected patched version is loaded.

## Build an isolated wheel

The same script can build the wheel outside Docker. Use Linux and the same
Python/libc/OpenSSL environment as the workers, with Rust 1.97.1, maturin 1.9.6,
C/C++ tools, OpenSSL development libraries, pkg-config and protobuf-compiler.
Set these absolute paths to a Relax checkout and a new build directory:

```bash
export RELAX_ROOT=/path/to/Relax
export ROUTER_BUILD=/path/to/new-router-build
export CARGO_TARGET_DIR="$ROUTER_BUILD/target"
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
export CARGO_BUILD_JOBS=16
bash "$RELAX_ROOT/docker/scripts/build_sglang_router.sh" \
  "$RELAX_ROOT/docker/patch/sglang-router" "$ROUTER_BUILD/wheels"
```

After building the wheel, choose one of the following installation methods.

### Install into the active training environment

Install the wheel after `requirements.txt`, using the Python environment that
will launch the router. If switching from an isolated installation, first
remove its directory from `PYTHONPATH` so it cannot shadow this environment.

```bash
python -m pip install --no-deps --force-reinstall "$ROUTER_BUILD"/wheels/*.whl
python -c 'import sglang_router, sglang_router.sglang_router_rs as native; print(sglang_router.__version__, native.__file__)'
```

Use that environment on the driver and every host that starts a managed router.

### Use an isolated installation

Install into a separate directory and verify it with a command-scoped import
path. This assignment does not change `PYTHONPATH` for later shell commands.

```bash
python -m pip install --no-deps --target "$ROUTER_BUILD/python" "$ROUTER_BUILD"/wheels/*.whl
PYTHONPATH="$ROUTER_BUILD/python${PYTHONPATH:+:$PYTHONPATH}" \
  python -c 'import sglang_router, sglang_router.sglang_router_rs as native; print(sglang_router.__version__, native.__file__)'
```

Prefix the training launch with the same
`PYTHONPATH="$ROUTER_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"` assignment.
The Mini SWE Ray entry point preserves an existing `PYTHONPATH`. For a cluster,
make this installation available at the same path on the driver and every worker,
and inspect the actual imported module inside Ray processes. Also retain the
patched SGLang path when an isolated SGLang installation is used.

## Verify the routing contract

The router patch includes three Rust unit tests covering independent P/D
owners, DP1 passthrough, and the physical bootstrap host: `test_pd_dp` and
`test_dp_aware_bootstrap_host_uses_physical_url` select them in a patched source
checkout.

For runtime verification, register each physical P/D engine with `dp_size=2`
and confirm four virtual workers. Send multiple turns with a stable
`X-SMG-Routing-Key`, including changed prompts. Inspect actual P/D completion
ranks, the decode request's prefill rank, and matching bootstrap room/host/port.
Exercise both ranks, streaming, R3 data, and physical-worker removal/re-addition.
Remove a physical worker using its registration UUID so all its virtual ranks
are removed together. Stable decode routing alone does not prove prefill
affinity or CPU cache restoration. `X-SMG-Target-Worker` overrides the hashing
policy; use separate requests without this header to validate session affinity.
