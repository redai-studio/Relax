# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent two-node NCCL probe used before elastic weight sync."""

import argparse
import datetime
import json
import os
import socket
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
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
        "fingerprint": {
            "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
            "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
            "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
            "nccl_ib_gid_index": os.environ.get("NCCL_IB_GID_INDEX"),
        },
    }
    process_group_initialized = False
    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(args.device_id)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{args.master_address}:{args.master_port}",
            world_size=2,
            rank=args.rank,
            timeout=datetime.timedelta(seconds=args.timeout_secs),
            device_id=torch.device("cuda", args.device_id),
        )
        process_group_initialized = True
        value = torch.tensor([float(args.rank)], device=f"cuda:{args.device_id}")
        dist.all_reduce(value)
        torch.cuda.synchronize(args.device_id)
        if float(value.cpu()[0]) != 1.0:
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
        print(json.dumps(result, sort_keys=True), flush=True)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
