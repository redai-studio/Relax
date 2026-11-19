# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Replay a FlashQLA local-cumsum compiler crash from a trusted Ray worker log.

Pass the failed 32-GPU recipe's DRY_RUN JSON with --recipe-json. This
reconstructs the recipe environment on the current host; it does not recover
the old worker's complete process environment. If the worker log records an
extra preloaded library, pass its absolute path with --extra-preload.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


def failed_compile(log: Path) -> tuple[str, str, list[str]]:
    """Keep the generated source, error and NVCC command from the same
    actor."""
    actor = re.compile(r"(\(MegatronTrainRayActor pid=\d+(?:, ip=[\w.:-]+)?\)) ?(.*)")
    colors = re.compile(r"\x1b\[[0-9;]*m")
    history: dict[str, list[str]] = {}
    with log.open(encoding="utf-8", errors="surrogateescape") as stream:
        for line in stream:
            match = actor.search(colors.sub("", line.rstrip("\n")))
            if not match:
                continue
            worker, payload = match.groups()
            if payload.startswith("RuntimeError: #"):
                history[worker] = [payload.removeprefix("RuntimeError: ")]
            elif worker in history and payload.startswith("Command: "):
                block = "\n".join(history.pop(worker))
                before, marker, stderr = block.rpartition("\nCompilation error:\n")
                if (
                    marker
                    and "Segmentation fault" in stderr
                    and 'extern "C" __global__' in before
                    and "tilelang_chunk_local_cumsum_kernel" in before
                    and before.rstrip().endswith("}")
                    and "[repeated " not in before
                    and "tvm_kernels.cu" in payload
                ):
                    return worker, before, shlex.split(payload.removeprefix("Command: "))
            elif worker in history:
                history[worker].append(payload)
    raise ValueError("No complete actor-prefixed local-cumsum CUDA source, segfault and command in this log")


def replay(log: Path, recipe_json: Path, out: Path, extra_preload: Path | None = None) -> None:
    worker, source, argv = failed_compile(log)
    if (
        Path(argv[0]).name != "nvcc"
        or "--cubin" not in argv
        or "-arch=sm_103a" not in argv
        or Path(argv[-1]).name != "tvm_kernels.cu"
        or argv[-3] != "-o"
        or not all(
            a in {"--cubin", "-O3", "-lineinfo", "-arch=sm_103a", "-std=c++20"} or a.startswith(("-I/", "-ccbin=/"))
            for a in argv[1:-3]
        )
    ):
        raise ValueError(f"Unexpected compiler invocation: {shlex.join(argv)}")
    evidence = Path(tempfile.mkdtemp(prefix="tilelang-cumsum-", dir=out))
    print(f"Evidence: {evidence}", flush=True)
    (evidence / "tvm_kernels.cu").write_text(source)
    (evidence / "original-command.txt").write_text(shlex.join(argv) + "\n")
    recipe = json.loads(recipe_json.read_text())
    args = recipe["argv"]
    env = {**os.environ, **recipe["runtime_env"]["env_vars"], **json.loads(args[args.index("--train-env-vars") + 1])}
    if not env.get("LD_PRELOAD"):
        raise ValueError("No LD_PRELOAD in recipe environment; the two conditions would be identical")
    clean = env.copy()
    del clean["LD_PRELOAD"]
    conditions = {"A": env, "B": clean}
    labels = ("A1", "B1", "B2", "A2")
    if extra_preload is not None:
        if not extra_preload.is_absolute() or not extra_preload.is_file():
            raise ValueError("--extra-preload must name an existing absolute library path from the worker log")
        if any(Path(lib).name == extra_preload.name for lib in re.split(r"[:\s]+", env["LD_PRELOAD"])):
            raise ValueError(f"{extra_preload.name} is already in the recipe preload")
        full = {**env, "LD_PRELOAD": env["LD_PRELOAD"] + ":" + str(extra_preload)}
        conditions = {"A": full, "B": env, "C": clean}
        labels = ("A1", "B1", "C1", "C2", "B2", "A2")
    compiler = Path(argv[0]).resolve(strict=True)
    for lib in re.split(r"[:\s]+", env["LD_PRELOAD"]):
        if lib.startswith("/") and not Path(lib).is_file():
            raise FileNotFoundError(f"Missing preloaded library on this host: {lib}")

    manifest = {
        "log": str(log.resolve()),
        "worker": worker,
        "replay_host": platform.node(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "original_argv": argv.copy(),
        "compiler": str(compiler),
        "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH"),
        "LD_PRELOAD": env["LD_PRELOAD"],
        "preload_by_condition": {key: values.get("LD_PRELOAD") for key, values in conditions.items()},
    }
    print(f"Original worker: {worker}\nSource SHA-256: {manifest['source_sha256']}", flush=True)
    for key, value in manifest["preload_by_condition"].items():
        print(f"{key} LD_PRELOAD={value or '<unset>'}", flush=True)
    version = subprocess.check_output([argv[0], "--version"], env=clean, text=True)
    manifest["nvcc_version"] = version
    results = {}
    with tempfile.TemporaryDirectory(prefix="tilelang-cumsum-") as tmp:
        src = Path(tmp) / "tvm_kernels.cu"
        src.write_bytes((evidence / src.name).read_bytes())
        cubin = src.with_suffix(".cubin")
        argv[-1] = str(src)
        argv[argv.index("-o") + 1] = str(cubin)
        manifest["replay_argv"] = argv
        (evidence / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for label in labels:
            cubin.unlink(missing_ok=True)
            with (evidence / f"{label}.log").open("wb") as stream:
                rc = subprocess.run(
                    argv, cwd=tmp, env=conditions[label[0]], stdout=stream, stderr=subprocess.STDOUT
                ).returncode
            diagnostic = (evidence / f"{label}.log").read_text(errors="replace")
            data = cubin.read_bytes() if cubin.is_file() else b""
            elf = len(data) >= 64 and data.startswith(b"\x7fELF") and int.from_bytes(data[18:20], "little") == 190
            if elf:
                (evidence / f"{label}.cubin").write_bytes(data)
            state = "other-error"
            if "cannot be preloaded" in diagnostic:
                state = "invalid-preload"
            elif rc == 0 and elf:
                state = "cubin"
            elif rc in (-11, 139) or (rc != 0 and re.search(r"Segmentation fault|SIGSEGV", diagnostic, re.I)):
                state = "segfault"
            results[label] = state
            print(f"{label}: rc={rc} {state}", flush=True)
            (evidence / "results.json").write_text(json.dumps(results, indent=2))
    if results["A1"] == results["A2"] == "segfault" and results["B1"] == results["B2"] == "cubin":
        difference = f"adding {extra_preload.name} to the recipe preload" if extra_preload else "the recipe preload"
        print(f"Reproduced: {difference} triggers the segfault in this compiler environment.")
    else:
        print("Inconclusive for the preload hypothesis; see each compiler log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Original Ray training log")
    parser.add_argument("--recipe-json", type=Path, required=True, help="DRY_RUN output of the recipe used by the job")
    parser.add_argument(
        "--extra-preload", type=Path, help="Compare recipe plus this library, recipe alone, and no preload"
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Existing directory for evidence")
    options = parser.parse_args()
    replay(options.log, options.recipe_json, options.output_dir.resolve(), options.extra_preload)
