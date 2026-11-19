# Prefill–Decode Disaggregation

Prefill–decode (PD) disaggregation assigns prompt processing and token generation to separate SGLang engine pools. Each pool can then be sized for its workload.

| Phase | Work | Typical bottleneck |
| --- | --- | --- |
| Prefill | Process the prompt and build KV state | Compute |
| Decode | Generate tokens using the accumulated KV state | Memory bandwidth |

PD separates phases within rollout inference. Training and rollout scheduling is configured separately; see [Fully Async Training](./fully-async-training.md) for deployment on separate GPU pools.

## Enable PD

Use the environment described in [Installation](./installation.md), including the matching SGLang patch. The selected model and transfer backend must support PD with your parallelism settings. When using Mooncake, prepare its dependencies and communication interfaces as described in the [SGLang PD guide](https://docs.sglang.io/docs/advanced_features/pd_disaggregation).

For engines with the same GPU allocation, add the following arguments to your existing training configuration:

```bash
--rollout-num-gpus 8 \
--rollout-num-gpus-per-engine 2 \
--prefill-num-servers 2 \
--sglang-router-prefill-policy consistent_hashing \
--sglang-router-decode-policy power_of_two
```

`--prefill-num-servers` counts **engines**, not GPUs. Relax assigns the requested engines to prefill and uses the remaining rollout GPUs for decode. With PP1 and DP1, this example creates:

| Role | Engines | GPUs per engine | Total GPUs |
| --- | ---: | ---: | ---: |
| Prefill | 2 | 2 (TP2) | 4 |
| Decode | 2 | 2 (TP2) | 4 |

Eight GPUs is an example allocation. Choose engine sizes that fit the model and its KV state, and leave GPUs for both phases. With `--colocate`, also align the rollout allocation with the GPUs shared with training: in the usual configuration without a separate reward model or teacher, Relax sets the rollout GPU count to the Actor GPU count.

`--prefill-num-servers` and `--sglang-config` are mutually exclusive. Neither can be combined with `--rollout-external`. The `power_of_two` example also requires the topology described under [Routing and limits](#routing-and-limits).

## Configure engine groups with YAML

Use `--sglang-config` when you need explicit P/D allocations or per-group SGLang settings. Save the following as a YAML file accessible to the training process:

```yaml
sglang:
  - name: default
    num_gpus_per_engine: 2
    engine_groups:
      - worker_type: prefill
        num_gpus: 4
      - worker_type: decode
        num_gpus: 4
```

Replace `--prefill-num-servers 2` in the previous example with the path to that file:

```bash
--sglang-config /path/to/pd-config.yaml
```

The YAML describes the same two-prefill, two-decode allocation. The path above is a placeholder for your own file.

- `num_gpus` is the total GPU allocation for a group. It must be divisible by that group's `num_gpus_per_engine`.
- A group's `num_gpus_per_engine` overrides the model-level value, which otherwise falls back to `--rollout-num-gpus-per-engine`. Different P/D engine sizes still require support from the model and transfer backend.
- The sum of all `num_gpus` values must match the effective `--rollout-num-gpus` allocation.
- An optional group-level `overrides` mapping accepts SGLang `ServerArgs` field names, such as `mem_fraction_static`, without the `sglang_` prefix. These values override the corresponding `--sglang-*` arguments.

## Request flow

1. The SGLang router selects a prefill engine and a decode engine for the request.
2. The prefill engine processes the prompt. SGLang transfers the resulting KV state to the decode engine through its transfer backend.
3. The decode engine generates tokens, and the router returns the response to the caller.

Relax configures the P/D worker roles, bootstrap ports, and router registration from the engine groups. The training weight-update path covers both prefill and decode engines.

## Size the pools

Long prompts and concurrent generation are useful workloads for evaluating PD: separating the phases can reduce interference between prompt processing and ongoing decode. Short prompts or low concurrency may not provide enough work to offset routing and KV-transfer overhead.

Compare against a non-PD run with the same total rollout GPUs, model, request lengths, and concurrency. Measure end-to-end rollout time as well as queueing, time to first token, and decode throughput. Adjust the allocation according to the bottleneck:

| Observation | Adjustment to evaluate |
| --- | --- |
| Prefill queues grow while decode has spare capacity | Allocate more engines to prefill |
| Decode queues grow while prefill has spare capacity | Allocate more engines to decode |
| Both pools remain saturated | Increase total resources or reduce request concurrency |
| KV transfer dominates latency | Check the transfer backend, network interfaces, and bandwidth before changing the P/D ratio |

With a fixed GPU budget, adding capacity to one pool reduces capacity in the other. Change the allocation between runs; runtime PD scale-out is not currently supported.

## Routing and limits

The example explicitly selects `consistent_hashing` for prefill and `power_of_two` for decode. To use consistent hashing for decode as well, replace the decode policy:

```bash
--sglang-router-decode-policy consistent_hashing
```

For a Relax-managed router using `power_of_two` on decode, Relax starts the engines before starting the router with their addresses. It checks worker roles, prefill bootstrap ports, and healthy registration before making the deployment available. This startup path requires:

- Native SGLang engines launched by Relax, with at least one prefill engine and two decode engines.
- `dp_size=1` on both sides and no DP-aware routing.
- P/D worker groups without mixed `regular` workers; custom engine classes are not supported by this path.

The managed startup sequence applies to the explicit `--sglang-router-decode-policy power_of_two` setting. Using `consistent_hashing` for decode follows the dynamic worker-registration path. Dynamic registration does not imply support for resizing P/D pools at runtime.

PD uses the SGLang router; `--use-slime-router` does not support it.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| GPU allocation error | Match the YAML total to the effective rollout GPU count and check each group's engine size |
| Router startup or registration failure | Check P/D engine counts, worker roles, DP settings, and prefill bootstrap connectivity |
| KV transfer timeout | Check model/backend compatibility, transfer dependencies, and communication interfaces |
| GPU OOM | Reduce request concurrency or context length, then check per-engine memory settings |
