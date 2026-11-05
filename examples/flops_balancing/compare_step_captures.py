# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare all rank-local reduced gradients and actual FP32 optimizer
deltas."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from relax.utils.training.numerical_comparison import compare_tensors


def input_signature(directory: Path, ranks: list[str]) -> tuple[list[str], str, int]:
    replicas = {}
    normalization = None
    batch_size = None
    for rank in ranks:
        inputs = json.loads((directory / f"inputs-{rank}.json").read_text())
        dp = inputs["topology"]["dp"]
        if dp in replicas and replicas[dp] != inputs["samples"]:
            raise ValueError("TP/PP/CP replicas received different effective inputs")
        replicas[dp] = inputs["samples"]
        if normalization is not None and normalization != inputs["normalization"]:
            raise ValueError("Ranks disagree on normalization")
        if batch_size is not None and batch_size != inputs["global_batch_size"]:
            raise ValueError("Ranks disagree on effective batch size")
        normalization, batch_size = inputs["normalization"], inputs["global_batch_size"]
    samples = sorted(sample for group in replicas.values() for sample in group)
    if len(samples) != batch_size:
        raise ValueError("Effective input coverage differs from global batch size")
    return samples, normalization, batch_size


def compare_captures(
    reference: Path,
    candidate: Path,
    *,
    loss_atol: float,
    loss_rtol: float,
    grad_atol: float,
    grad_rtol: float,
    update_atol: float,
    update_rtol: float,
) -> dict[str, Any]:
    ranks = sorted(path.name for path in reference.glob("rank-*"))
    if not ranks or ranks != sorted(path.name for path in candidate.glob("rank-*")):
        raise ValueError("Captures must contain the same nonempty rank set")
    if input_signature(reference, ranks) != input_signature(candidate, ranks):
        raise ValueError("Effective samples, tokens, masks or normalization changed")
    report: dict[str, Any] = {
        "ranks": {},
        "tolerances": dict(
            loss_atol=loss_atol,
            loss_rtol=loss_rtol,
            grad_atol=grad_atol,
            grad_rtol=grad_rtol,
            update_atol=update_atol,
            update_rtol=update_rtol,
        ),
    }
    losses_seen = False
    total_elements = 0
    for rank in ranks:
        left = json.loads((reference / rank / "manifest.json").read_text())
        right = json.loads((candidate / rank / "manifest.json").read_text())
        for manifest in (left, right):
            if manifest.get("format") != 1 or not manifest.get("successful"):
                raise ValueError("Incomplete or skipped captured update")
            if manifest.get("gradient_stage") != "unscaled_reduced_before_clipping":
                raise ValueError("Unknown gradient capture stage")
        if left["parameters"] != right["parameters"] or left["loss"].keys() != right["loss"].keys():
            raise ValueError("Optimizer shard layout or reported loss keys changed")
        results = {}
        for entry in left["parameters"]:
            filename = entry["file"]
            if Path(filename).name != filename:
                raise ValueError("Parameter filename must be local to its rank")
            x = torch.load(reference / rank / filename, map_location="cpu", weights_only=True)
            y = torch.load(candidate / rank / filename, map_location="cpu", weights_only=True)
            if set(x) != {"initial", "gradient", "update"} or x.keys() != y.keys():
                raise ValueError("Capture must contain initial weights, complete gradients and FP32 updates")
            if any(tensor.dtype != torch.float32 for payload in (x, y) for tensor in payload.values()):
                raise ValueError("Captured tensors must be FP32")
            if any(
                list(tensor.shape) != entry["shape"] or tensor.numel() != entry["elements"]
                for payload in (x, y)
                for tensor in payload.values()
            ):
                raise ValueError("Captured tensor shape differs from manifest")
            results[filename] = {
                "initial": compare_tensors(x["initial"], y["initial"], atol=0, rtol=0),
                "gradient": compare_tensors(x["gradient"], y["gradient"], atol=grad_atol, rtol=grad_rtol),
                "update": compare_tensors(x["update"], y["update"], atol=update_atol, rtol=update_rtol),
            }
            total_elements += entry["elements"]
        losses_seen |= bool(left["loss"])
        losses = {
            key: compare_tensors(
                torch.tensor(value, dtype=torch.float64),
                torch.tensor(right["loss"][key], dtype=torch.float64),
                atol=loss_atol,
                rtol=loss_rtol,
            )
            for key, value in left["loss"].items()
        }
        passed = all(result["passed"] for checks in results.values() for result in checks.values())
        report["ranks"][rank] = {
            "parameters": results,
            "loss": losses,
            "passed": passed and all(check["passed"] for check in losses.values()),
        }
    if not losses_seen or total_elements == 0:
        raise ValueError("No reported loss or optimizer parameters were compared")
    report.update(
        elements_including_replicas=total_elements, passed=all(rank["passed"] for rank in report["ranks"].values())
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for metric in ("loss", "grad", "update"):
        for tolerance in ("atol", "rtol"):
            parser.add_argument(f"--{metric}-{tolerance}", type=float, required=True)
    args = parser.parse_args()
    report = compare_captures(
        args.reference,
        args.candidate,
        **{
            f"{metric}_{tol}": getattr(args, f"{metric}_{tol}")
            for metric in ("loss", "grad", "update")
            for tol in ("atol", "rtol")
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "ranks": len(report["ranks"]),
                "elements_including_replicas": report["elements_including_replicas"],
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
