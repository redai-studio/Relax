# ML/PyTorch Checklist (Relax Project)

Contents: [Tensors](#tensor-operations), [gradients](#gradient-issues), [memory](#memory-management), [distributed training](#distributed-training), [numerics](#numerical-stability), [Relax patterns](#relax-specific-patterns), [questions](#review-questions).

## Tensor Operations

### Shape / Dtype / Device Mismatches

Trace shape, dtype, and device contracts from producers through transforms to consumers. Report a mismatch reachable in a supported mode; established internal contracts do not need repeated assertions at every function. Validate genuinely external data at its boundary.

- `squeeze()` without specifying dim — removes ALL size-1 dims
- `view()` vs `reshape()` — view requires contiguous memory
- Creating tensors without matching `dtype`/`device` of existing tensors

______________________________________________________________________

## Gradient Issues

### Key Anti-patterns

- **Retained graphs** in metrics or buffers that do not need gradients; verify whether later differentiation is intended before recommending `.detach()`
- **In-place ops** that violate autograd or shared ownership; owned buffers can be mutated when their contracts permit it
- **Using `.data`** (deprecated, breaks autograd)
- **Unnecessary graph construction** in inference-only work; check whether an enclosing context already disables gradients
- **Metric accumulation** that retains computation graphs; avoid fixing it by introducing GPU-CPU synchronization in a hot path

______________________________________________________________________

## Memory Management

```python
# Retains the computation graph for each collected loss
losses.append(loss)

# Instead, drop the graph when metrics do not need gradients
losses.append(loss.detach())
```

Detached tensors still occupy memory and share storage: bound or aggregate metric retention, consume it at an explicit logging boundary, and account for later mutation. Do not add `.item()`, `.tolist()`, or tensor printing to training hot paths.

- References that retain large tensors beyond their useful lifetime; lack of an explicit `del` alone is not a leak
- Investigate actual live allocations and memory pressure before proposing cache clearing or gradient checkpointing

______________________________________________________________________

## Distributed Training

### Collective Operation Ordering

All required members of the selected process group **must** participate in compatible collective calls in the same order.

```python
# Bad: conditional collective → hang
if rank == 0:
    dist.broadcast(tensor, src=0)

# Good: all ranks participate
dist.broadcast(tensor, src=0)
```

### Process Group Usage

```python
# Bad: implicit default group
dist.all_reduce(tensor)

# Good: explicit group
dist.all_reduce(tensor, group=self.data_parallel_group)
```

- Verify shapes, dtypes, and devices meet the particular collective's contract across participating ranks
- Missing gradients must not make only some ranks skip a collective; verify group-wide participation rather than adding rank-local guards

______________________________________________________________________

## Numerical Stability

- Establish whether zero denominators or empty masks are supported cases, invalid inputs, or broken invariants. Add a defined numerical treatment only when it preserves the algorithm; arbitrary epsilon/clamping can hide corrupt data or change the estimator
- Check stability of probability/log-probability calculations and NaN/Inf propagation; prefer a mathematically equivalent stable formulation where applicable
- Verify clipping and mixed-precision loss scaling against the configured training backend and algorithm; do not add clipping or a scaler simply because precision is reduced

______________________________________________________________________

## Relax-Specific Patterns

### RolloutBatch

- Trace required keys to the producer's contract; report supported paths that fail to populate them
- Handle optional keys (`batch.get("values")` for non-PPO)
- Be deliberate about in-place mutation of batch dicts

### Loss Scaling (Megatron)

Loss must account for: `num_microbatches`, `global_batch_size`, `data_parallel_world_size`.

### Context Parallelism

- Use `all_gather_with_cp` for proper gathering across CP ranks
- Verify offset computation via `get_logits_and_tokens_offset_with_cp`

______________________________________________________________________

## Review Questions

| Area | Question |
|------|----------|
| Shapes | "What are the expected shapes here?" |
| Gradients | "Should this be detached?" |
| Distributed | "Do the required members of this process group participate compatibly?" |
| Memory | "Could this accumulate tensors?" |
| Numerical | "Could this overflow / divide by zero?" |
