# FLOPs scheduling regression tests

The CPU tests exercise the production estimator, planner, sampler adapters and extracted training iterator/fetch functions. Heavy runtime dependencies have explicit test seams; these tests are not GPU training or real TransferQueue service evidence.

Run from the repository root in a Python 3.10+ environment with NumPy and pytest:

```bash
python -m pytest -q tests/cpu/no16_p2 tests/cpu/no16_p3p4 tests/cpu/no16_p45
```

The pinned reference sources and licenses support sampler compatibility and the off-mode golden regression. Model configs support arithmetic checks for Dense and MoE; they contain no model weights.

With a CPU PyTorch installation supporting Gloo, the distributed checks can be run separately, each with a new output directory:

```bash
python -m tests.cpu.no16_p3p4.distributed_probe /path/to/new-legacy-results
python -m tests.cpu.no16_p45.distributed_probe /path/to/new-pack-dp2-results 2
python -m tests.cpu.no16_p45.post_review_probe /path/to/new-fetch-dp2-results 2
```

These Gloo checks cover collective agreement and error propagation. They do not validate NCCL, Megatron training, numerical equivalence or throughput.
