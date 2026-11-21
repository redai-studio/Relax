# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent NCCL probe used before elastic weight sync."""

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys


def transport_fingerprint() -> dict:
    return {
        f"nccl_{name}": os.getenv(f"NCCL_{name.upper()}")
        for name in ("ib_disable", "socket_ifname", "ib_hca", "ib_gid_index")
    }


def launch_probe(
    master_address: str, port: str, rank: int, world_size: int, device: int, timeout: float, token: str, **popen_kwargs
) -> subprocess.Popen:
    """Start one isolated rank on the caller's assigned GPU."""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            __name__,
            "--master-address",
            master_address,
            "--master-port",
            port,
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--device-id",
            str(device),
            "--timeout-secs",
            str(timeout),
            "--run-token",
            token,
        ],
        start_new_session=True,
        **popen_kwargs,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--timeout-secs", required=True, type=float)
    parser.add_argument("--run-token", required=True)
    args = parser.parse_args()

    result = {
        "success": False,
        "rank": args.rank,
        "device_id": args.device_id,
        "hostname": socket.gethostname(),
        "run_token": args.run_token,
        "fingerprint": transport_fingerprint(),
    }
    process_group_initialized = False
    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(args.device_id)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{args.master_address}:{args.master_port}",
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(seconds=args.timeout_secs),
            device_id=torch.device("cuda", args.device_id),
        )
        process_group_initialized = True
        value = torch.tensor([float(args.rank)], device=f"cuda:{args.device_id}")
        dist.all_reduce(value)
        torch.cuda.synchronize(args.device_id)
        if float(value.cpu()[0]) != args.world_size * (args.world_size - 1) / 2:
            raise RuntimeError(f"unexpected all_reduce result: {value}")
        result["success"] = True
    except Exception as exc:
        # Report the exception structurally; the coordinator surfaces/propagates
        # it verbatim rather than scanning this probe's NCCL log output.
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        if process_group_initialized:
            try:
                import torch.distributed as dist

                dist.destroy_process_group()
            except Exception as exc:
                result["success"] = False
                result["destroy_error"] = f"{type(exc).__name__}: {exc}"
        # Native NCCL logs can end mid-line on the same descriptor. Write the
        # verdict as one framed record so the parent can decode a whole line.
        os.write(sys.stdout.fileno(), f"\n{json.dumps(result, sort_keys=True)}\n".encode())

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
