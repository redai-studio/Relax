# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pre-flight probe: can the installed sglang serve Qwen3-0.6B on this box?

Isolates sglang-engine issues from relax integration issues. Launches a vanilla
sglang server exactly the way GenRMEngine does (same ServerArgs fields
_compute_genrm_server_args produces), waits for health, runs one greedy
generation, checks /metrics exists, then shuts down.
"""

import argparse
import subprocess
import sys
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=16099)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()

    from sglang.srt.server_args import ServerArgs

    # The same fields relax's _compute_genrm_server_args produces for a
    # single-GPU genrm engine (drop-unknown-kwargs is NOT available here --
    # these must all exist in the installed sglang).
    server_kwargs = dict(
        model_path=args.model_path,
        trust_remote_code=True,
        random_seed=42,
        host="127.0.0.1",
        port=args.port,
        nccl_port=args.port + 1,
        nnodes=1,
        node_rank=0,
        dist_init_addr=f"127.0.0.1:{args.port + 2}",
        gpu_id_step=1,
        base_gpu_id=args.gpu,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        skip_server_warmup=False,
        enable_metrics=True,
        enable_weights_cpu_backup=True,
        load_format="auto",
    )
    unknown = []
    import dataclasses

    valid = {f.name for f in dataclasses.fields(ServerArgs)}
    for k in list(server_kwargs):
        if k not in valid:
            unknown.append((k, server_kwargs.pop(k)))
    if unknown:
        print(f"[probe] dropped unknown ServerArgs fields (installed sglang): {unknown}")

    ServerArgs(**server_kwargs)  # validation: raises on bad values
    print(f"[probe] sglang {__import__('sglang').__version__}, torch {__import__('torch').__version__}")
    print(f"[probe] launching server on 127.0.0.1:{args.port} gpu {args.gpu}")

    import os

    env = dict(os.environ)
    env.update(
        {
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0",
        }
    )

    proc = subprocess.Popen(
        [sys.executable, "-m", "sglang.launch_server"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        # Pass server args via a file to avoid CLI drift between versions.
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(dict(server_kwargs), f)
            argfile = f.name
        # launch_server reads CLI args; simplest robust path: use the python API.
        proc.terminate()
        proc.wait(timeout=10)

        proc = subprocess.Popen(
            [sys.executable, __file__, "--_serve", argfile],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        deadline = time.time() + 600
        ok = False
        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"[probe] server process exited: {proc.returncode}")
                return 3
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=3) as r:
                    if r.status == 200:
                        ok = True
                        break
            except Exception:
                pass
            time.sleep(3)
        if not ok:
            print("[probe] FAILED: server did not become healthy in 600s")
            return 2
        print(f"[probe] healthy after {600 - (deadline - time.time()):.0f}s")

        import requests

        r = requests.post(
            f"http://127.0.0.1:{args.port}/generate",
            json={
                "input_ids": [151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198],
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        r.raise_for_status()
        out = r.json()
        print(f"[probe] generation ok: text={str(out.get('text'))[:60]!r}")

        m = requests.get(f"http://127.0.0.1:{args.port}/metrics", timeout=10)
        has_token = "token_usage" in m.text
        has_queue = "num_queue_reqs" in m.text
        print(f"[probe] /metrics: HTTP {m.status_code}, token_usage={has_token}, num_queue_reqs={has_queue}")
        return 0 if (has_token or has_queue) else 4
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _serve_mode() -> int:
    """Child mode: read server args from the json file and launch."""
    import json
    import sys

    with open(sys.argv[sys.argv.index("--_serve") + 1]) as f:
        server_kwargs = json.load(f)
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    launch_server(ServerArgs(**server_kwargs))
    return 0


if __name__ == "__main__":
    if "--_serve" in sys.argv:
        sys.exit(_serve_mode())
    sys.exit(main())
