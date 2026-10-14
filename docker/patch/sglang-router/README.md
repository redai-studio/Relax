# PD DP-aware router compatibility

`v0.3.2-9daabcd-pd-dp-aware.patch` applies to the R3-capable
[`zhuzilin/sgl-router` fork at `9daabcd801c861088d93b8ce8a2a831490ce4bee`](https://github.com/zhuzilin/sgl-router/tree/9daabcd801c861088d93b8ce8a2a831490ce4bee),
the source of the wheel pinned by Relax's Dockerfiles. Use it together with
Relax's SGLang v0.5.17 patch for the
[1P1D attention-DP2 recipe](../../../examples/mini_swe_agent/README.md#train-with-cpu-caching).
The Dockerfiles still install the original wheel; this overlay is required for
PD DP-aware deployment with that version.

The patch keeps virtual worker identities and per-rank metrics, while sending
HTTP requests and KV bootstrap to physical endpoints. It passes independent
`routed_dp_rank` values to prefill and decode, plus `disagg_prefill_dp_rank` to
decode. Raw native fields, including R3 `routed_experts`, remain intact. The
paired SGLang patch also accepts the legacy `data_parallel_rank` field used by
this router's regular path; an explicit `routed_dp_rank` takes precedence.

## Build an isolated overlay

Use an environment with the existing router's Python dependencies, C/C++ and
Rust toolchains, and OpenSSL development libraries. Validation used Linux x86_64, Python
3.12, and Rust 1.97.1. Build on the same platform as the workers. Set these
absolute paths to a Relax checkout and a new build directory:

```bash
export RELAX_ROOT=/path/to/Relax
export ROUTER_BUILD=/path/to/new-router-build
mkdir -p "$ROUTER_BUILD"
git clone https://github.com/zhuzilin/sgl-router.git "$ROUTER_BUILD/source"
cd "$ROUTER_BUILD/source"
git checkout --detach 9daabcd801c861088d93b8ce8a2a831490ce4bee
git apply --check "$RELAX_ROOT/docker/patch/sglang-router/v0.3.2-9daabcd-pd-dp-aware.patch"
git apply "$RELAX_ROOT/docker/patch/sglang-router/v0.3.2-9daabcd-pd-dp-aware.patch"

export CARGO_TARGET_DIR="$ROUTER_BUILD/target"
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
export CARGO_BUILD_JOBS=16
cargo build --release --manifest-path bindings/python/Cargo.toml

mkdir -p "$ROUTER_BUILD/python"
cp -a bindings/python/src/sglang_router "$ROUTER_BUILD/python/"
cp "$CARGO_TARGET_DIR/release/libsglang_router_rs.so" \
  "$ROUTER_BUILD/python/sglang_router/sglang_router_rs.abi3.so"
export PYTHONPATH="$ROUTER_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import sglang_router, sglang_router.sglang_router_rs as native; print(sglang_router.__version__, native.__file__)'
cd "$RELAX_ROOT"
```

The fork has no committed Cargo lockfile. Preserve the generated
`bindings/python/Cargo.lock` with build records, and use `--locked` for subsequent
builds with that file. The package version remains `0.3.2+slime`; record the
loaded extension path and SHA-256 to distinguish it from the original wheel.
The overlay changes neither the globally installed package nor CUDA libraries.

The Mini SWE Ray entry point preserves an existing `PYTHONPATH`. For a cluster,
make the overlay available at the same path on every worker and inspect the
actual imported module inside Ray processes. Also retain the patched SGLang
path when an isolated SGLang installation is used.

## Verify the routing contract

The patch includes three Rust unit tests covering independent P/D owners,
DP1 passthrough, and the physical bootstrap host:

```bash
cd "$ROUTER_BUILD/source"
cargo test --release --lib test_pd_dp
cargo test --release --lib test_dp_aware_bootstrap_host_uses_physical_url
cd "$RELAX_ROOT"
```

For runtime verification, register each physical P/D engine with `dp_size=2`
and confirm four virtual workers. Send multiple turns with a stable
`X-SMG-Routing-Key`, including changed prompts. Inspect actual P/D completion
ranks, the decode request's prefill rank, and matching bootstrap room/host/port.
Exercise both ranks, streaming, R3 data, and physical-worker removal/re-addition.
Remove a physical worker using its registration UUID so all its virtual ranks
are removed together. Stable decode routing alone does not prove prefill
affinity or CPU cache restoration.
