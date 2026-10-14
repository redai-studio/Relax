#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Export OSWorld training curves from one run's TensorBoard directory."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(args.tensorboard_dir), size_guidance={"scalars": 0}).Reload()
    available = accumulator.Tags()["scalars"]
    panels = (
        ("rollout/raw_reward", "Training raw reward", "Rollout"),
        ("train/loss", "Training loss", "Optimizer step"),
        ("train/grad_norm", "Gradient norm", "Optimizer step"),
        ("train/kl_loss", "KL loss", "Optimizer step"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "step", "value", "wall_time"))
        for tag in available:
            for event in accumulator.Scalars(tag):
                writer.writerow((tag, event.step, event.value, event.wall_time))
    for ax, (tag, title, xlabel) in zip(axes.flat, panels, strict=True):
        if tag in available:
            events = accumulator.Scalars(tag)
            ax.plot([e.step for e in events], [e.value for e in events], marker="o", markersize=4)
        ax.set(title=title, xlabel=xlabel)
        ax.grid(alpha=0.25)
    fig.suptitle("OSWorld + NexSandbox / Qwen3-VL-8B-Thinking")
    fig.savefig(args.output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
