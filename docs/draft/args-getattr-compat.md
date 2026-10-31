# Remaining getattr(args, …) inventory (Task 13)

Total: **183 sites / 105 attrs** after the Relax-owned sweep and
compat centralization for dynamic CP / CP size.

## Classification legend

| Class | Action |
| --- | --- |
| `sglang-optional` | Keep `getattr` (SGLang parse may be skipped; see slime #1604). |
| `megatron-schema-compat` | Prefer helpers in `relax/backends/megatron/compat.py` with version notes. |
| `derived-post-init` | Ensure parse/validate assigns the field, then direct-read. |
| `private/runtime marker` | Internal `_relax_*` markers; keep defensive access. |
| `megatron-or-upstream-optional` | Upstream optional knobs; migrate to compat when version-gated. |
| `slime-owned` | Must not remain; treat as a bug if listed. |

## By attribute

- `is_vl_model` ×0 — **derived-post-init (False in parse; actor may overwrite; direct-read)**
- `merge_lora` ×6 — **megatron-or-upstream-optional**
- `uses_unsplit_forward` ×0 — **derived-post-init (False in parse; bridge may set True; direct-read)**
- `empty_unused_memory_level` ×4 — **megatron-or-upstream-optional**
- `optimizer_cpu_offload` ×4 — **megatron-or-upstream-optional**
- `overlap_moe_expert_parallel_comm` ×4 — **megatron-or-upstream-optional**
- `sglang_enable_deterministic_inference` ×4 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_speculative_algorithm` ×4 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `virtual_pipeline_model_parallel_size` ×4 — **megatron-or-upstream-optional**
- `cuda_graph_impl` ×3 — **megatron-or-upstream-optional**
- `fp8` ×3 — **megatron-or-upstream-optional**
- `prefill_num_servers` ×3 — **megatron-or-upstream-optional**
- `recompute_granularity` ×3 — **megatron-or-upstream-optional**
- `sglang_config` ×3 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `tensor_model_parallel_size` ×3 — **megatron-or-upstream-optional**
- `use_megatron_fsdp` ×3 — **megatron-or-upstream-optional**
- `_relax_managed_pd_router_endpoint` ×2 — **private/runtime marker**
- `bf16` ×2 — **megatron-or-upstream-optional**
- `ckpt_step` ×2 — **megatron-or-upstream-optional**
- `decoder_first_pipeline_num_layers` ×2 — **megatron-or-upstream-optional**
- `decoder_last_pipeline_num_layers` ×2 — **megatron-or-upstream-optional**
- `eval_datasets` ×0 — **derived-post-init (assigned in validate; direct-read)**
- `experiment_name` ×2 — **megatron-or-upstream-optional**
- `expert_gather_backend` ×2 — **megatron-or-upstream-optional**
- `expert_tensor_parallel_size` ×2 — **megatron-or-upstream-optional**
- `fp16` ×2 — **megatron-or-upstream-optional**
- `fp4` ×2 — **megatron-or-upstream-optional**
- `loss_scale` ×2 — **megatron-or-upstream-optional**
- `lr_decay_iters` ×2 — **megatron-or-upstream-optional**
- `lr_warmup_iters` ×2 — **megatron-or-upstream-optional**
- `moe_flex_dispatcher_backend` ×2 — **megatron-schema-compat (prefer relax/backends/megatron/compat.py)**
- `moe_shared_expert_overlap` ×2 — **megatron-or-upstream-optional**
- `moe_token_dispatcher_type` ×2 — **megatron-or-upstream-optional**
- `no_load_optim` ×2 — **megatron-or-upstream-optional**
- `num_rollout_per_epoch` ×2 — **megatron-or-upstream-optional**
- `optimizer` ×2 — **megatron-or-upstream-optional**
- `overlap_grad_reduce` ×2 — **megatron-or-upstream-optional**
- `pipeline_model_parallel_size` ×2 — **megatron-or-upstream-optional**
- `recompute_num_layers` ×2 — **megatron-or-upstream-optional**
- `reward_delay` ×2 — **megatron-or-upstream-optional**
- `router_decode_policy` ×2 — **megatron-or-upstream-optional**
- `score_centering_vocab_size` ×2 — **megatron-or-upstream-optional**
- `seq_length` ×2 — **megatron-or-upstream-optional**
- `sglang_dp_size` ×2 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_num_steps` ×2 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_sampling_backend` ×2 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `_i2i_sft_attempt_id` ×1 — **private/runtime marker**
- `_model_source_original_hf_checkpoint` ×1 — **private/runtime marker**
- `_qwen_fused_head` ×1 — **private/runtime marker**
- `_relax_resident_sft_actor_verified` ×1 — **private/runtime marker**
- `adam_beta1` ×1 — **megatron-or-upstream-optional**
- `adam_beta2` ×1 — **megatron-or-upstream-optional**
- `adam_eps` ×1 — **megatron-or-upstream-optional**
- `advantage_estimator` ×1 — **pre-parse helper only** (`algorithm_needs_critic` may run before parse; keep getattr)
- `apply_dsa_kernel_fusion` ×1 — **megatron-or-upstream-optional**
- `apply_rope_fusion` ×1 — **megatron-or-upstream-optional**
- `attention_dropout` ×1 — **megatron-or-upstream-optional**
- `check_for_nan_in_loss_and_grad` ×1 — **megatron-or-upstream-optional**
- `ckpt_format` ×1 — **megatron-or-upstream-optional**
- `cp_partition_mode` ×1 — **megatron-or-upstream-optional**
- `cpu_offloading` ×1 — **megatron-or-upstream-optional**
- `data_parallel_size` ×1 — **megatron-or-upstream-optional**
- `defer_embedding_wgrad_compute` ×1 — **megatron-or-upstream-optional**
- `disable_jit_fuser` ×1 — **megatron-or-upstream-optional**
- `distribute_saved_activations` ×1 — **megatron-or-upstream-optional**
- `enable_cuda_graph` ×1 — **megatron-or-upstream-optional**
- `enable_gloo_process_groups` ×1 — **megatron-or-upstream-optional**
- `eval_at` ×1 — **megatron-or-upstream-optional**
- `eviction_check_interval` ×1 — **megatron-or-upstream-optional**
- `expert_model_parallel_size` ×1 — **megatron-or-upstream-optional**
- `finetune` ×1 — **megatron-or-upstream-optional**
- `fp8_recipe` ×1 — **megatron-or-upstream-optional**
- `hidden_dropout` ×1 — **megatron-or-upstream-optional**
- `lr_decay_style` ×1 — **megatron-or-upstream-optional**
- `lr_warmup_fraction` ×1 — **megatron-or-upstream-optional**
- `lr_warmup_init` ×1 — **megatron-or-upstream-optional**
- `min_lr` ×1 — **megatron-or-upstream-optional**
- `model_source` ×1 — **derived-post-init (set during parse/validate, then direct-read)**
- `moe_layer_freq` ×1 — **megatron-or-upstream-optional**
- `mtp_loss_normalization` ×1 — **megatron-or-upstream-optional**
- `normalize_moe_lora` ×1 — **megatron-or-upstream-optional**
- `num_distributed_optimizer_instances` ×1 — **megatron-or-upstream-optional**
- `num_layers_per_virtual_pipeline_stage` ×1 — **megatron-or-upstream-optional**
- `num_rollout_per_epoch_for_metrics` ×1 — **megatron-or-upstream-optional**
- `pipeline_model_parallel_layout` ×1 — **megatron-or-upstream-optional**
- `profile_ranks` ×1 — **megatron-or-upstream-optional**
- `rollout_service_url` ×1 — **megatron-or-upstream-optional**
- `router_cb_timeout_duration_secs` ×1 — **megatron-or-upstream-optional**
- `router_dp_aware` ×1 — **megatron-or-upstream-optional**
- `sglang_enable_hierarchical_cache` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_enable_metrics` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_activities` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_by_stage` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_output_dir` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_record_shapes` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_step_end` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_step_start` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_steps` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_profile_with_stack` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_rl_on_policy_target` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `sglang_router_ip` ×1 — **sglang-optional (keep getattr; parse may skip SGLang)**
- `untie_embeddings_and_output_weights` ×1 — **megatron-or-upstream-optional**
- `use_gloo_process_groups` ×1 — **megatron-or-upstream-optional**
- `weight_decay` ×1 — **megatron-or-upstream-optional**

## By file

### `examples/models/kimi-k3/tools/convert_kimi_k3_torch_dist_to_hf.py`
- L346: `merge_lora` default="merge_lora", False
- L435: `merge_lora` default="merge_lora", False
- L437: `merge_lora` default=?
- L438: `merge_lora` default=?

### `examples/models/kimi-k3/tools/convert_kimi_k3_torch_dist_to_hf_parallel.py`
- L266: `merge_lora` default=?
- L267: `expert_gather_backend` default=?
- L278: `merge_lora` default=?
- L375: `expert_gather_backend` default=?

### `examples/models/kimi-k3/tools/kimi_k3_streaming_lora.py`
- L120: `normalize_moe_lora` default=?
- L168: `tensor_model_parallel_size` default=?
- L169: `pipeline_model_parallel_size` default=?
- L170: `expert_model_parallel_size` default=?
- L171: `expert_tensor_parallel_size` default=?

### `relax/backends/fsdp/actor.py`
- L49: `weight_decay` default="weight_decay", None
- L52: `adam_beta1` default="adam_beta1", None
- L53: `adam_beta2` default="adam_beta2", None
- L56: `adam_eps` default="adam_eps", None
- L71: `lr_warmup_iters` default="lr_warmup_iters", 0) or 0
- L78: `lr_decay_iters` default="lr_decay_iters", None
- L81: `min_lr` default="min_lr", 0.0) or 0.0

### `relax/backends/fsdp/arguments.py`
- L616: `lr_decay_style` default=?

### `relax/backends/megatron/actor.py`
- L479: `moe_flex_dispatcher_backend` default=?

### `relax/backends/megatron/arguments.py`
- L51: `loss_scale` default=?
- L83: `loss_scale` default=?
- L199: `moe_token_dispatcher_type` default=?
- L224: `moe_layer_freq` default="moe_layer_freq", None
- L263: `apply_rope_fusion` default=?
- L309: `optimizer` default=?

### `relax/backends/megatron/checkpoint.py`
- L325: `expert_tensor_parallel_size` default=?
- L326: `virtual_pipeline_model_parallel_size` default=?
- L327: `pipeline_model_parallel_layout` default=?
- L328: `decoder_first_pipeline_num_layers` default=?
- L329: `decoder_last_pipeline_num_layers` default=?
- L330: `num_layers_per_virtual_pipeline_stage` default=?
- L332: `data_parallel_size` default=?
- L635: `ckpt_step` default="ckpt_step", None))
- L637: `ckpt_step` default=?
- L660: `no_load_optim` default="no_load_optim", False) and not getattr(args, "finetune", False
- L660: `finetune` default="finetune", False
- L690: `optimizer_cpu_offload` default="optimizer_cpu_offload", False
- L705: `optimizer_cpu_offload` default="optimizer_cpu_offload", False
- L706: `no_load_optim` default="no_load_optim", False
- L779: `lr_decay_iters` default=?
- L780: `lr_warmup_iters` default=?
- L781: `lr_warmup_fraction` default=?

### `relax/backends/megatron/checkpoint_metadata.py`
- L50: `ckpt_format` default=?

### `relax/backends/megatron/data.py`
- L935: `is_vl_model` default=?
- L1358: `is_vl_model` default="is_vl_model", False
- L1359: `uses_unsplit_forward` default="uses_unsplit_forward", False
- L1495: `is_vl_model` default="is_vl_model", False
- L1497: `uses_unsplit_forward` default=?

### `relax/backends/megatron/initialize.py`
- L81: `disable_jit_fuser` default=?

### `relax/backends/megatron/loss.py`
- L453: `overlap_grad_reduce` default=?
- L457: `defer_embedding_wgrad_compute` default=?
- L732: `is_vl_model` default="is_vl_model", False
- L734: `uses_unsplit_forward` default=?
- L1677: `_qwen_fused_head` default=?

### `relax/backends/megatron/model.py`
- L451: `lr_warmup_init` default="lr_warmup_init", 0.0
- L502: `overlap_grad_reduce` default="overlap_grad_reduce", False
- L506: `untie_embeddings_and_output_weights` default="untie_embeddings_and_output_weights", False
- L507: `cuda_graph_impl` default=?
- L509: `num_distributed_optimizer_instances` default=?
- L662: `recompute_granularity` default=?
- L666: `recompute_granularity` default=?
- L978: `is_vl_model` default="is_vl_model", False
- L1008: `uses_unsplit_forward` default="uses_unsplit_forward", False
- L1127: `empty_unused_memory_level` default=?
- L1294: `is_vl_model` default="is_vl_model", False
- L1370: `uses_unsplit_forward` default="uses_unsplit_forward", False
- L1540: `empty_unused_memory_level` default=?
- L1586: `check_for_nan_in_loss_and_grad` default=?
- L1628: `empty_unused_memory_level` default=?
- L1735: `empty_unused_memory_level` default=?
- L1744: `_relax_resident_sft_actor_verified` default=?
- L1972: `num_rollout_per_epoch_for_metrics` default="num_rollout_per_epoch_for_metrics", None
- L1974: `num_rollout_per_epoch` default="num_rollout_per_epoch", None

### `relax/backends/megatron/model_provider.py`
- L492: `decoder_first_pipeline_num_layers` default=?
- L494: `decoder_last_pipeline_num_layers` default=?
- L562: `apply_dsa_kernel_fusion` default="apply_dsa_kernel_fusion", False)

### `relax/backends/megatron/resident_shared_expert.py`
- L62: `moe_shared_expert_overlap` default="moe_shared_expert_overlap", False
- L70: `moe_token_dispatcher_type` default=?
- L71: `moe_flex_dispatcher_backend` default=?
- L73: `cuda_graph_impl` default=?
- L74: `enable_cuda_graph` default="enable_cuda_graph", False

### `relax/backends/megatron/score_centering.py`
- L20: `score_centering_vocab_size` default="score_centering_vocab_size", None

### `relax/backends/megatron/weight_conversion/processors/quantizer_fp8.py`
- L29: `fp8` default=?
- L35: `fp8_recipe` default="fp8_recipe", None

### `relax/backends/megatron/weight_update/train_offload.py`
- L151: `optimizer_cpu_offload` default="optimizer_cpu_offload", False)
- L384: `optimizer` default=?

### `relax/backends/sglang/arguments.py`
- L336: `sglang_router_ip` default=?
- L340: `prefill_num_servers` default=?
- L344: `sglang_config` default=?
- L349: `prefill_num_servers` default=?
- L349: `sglang_config` default=?

### `relax/backends/sglang/sglang_engine.py`
- L1916: `sglang_speculative_algorithm` default="sglang_speculative_algorithm", None

### `relax/distributed/ray/placement_group.py`
- L207: `num_rollout_per_epoch` default="num_rollout_per_epoch", None

### `relax/distributed/ray/rollout.py`
- L1005: `eviction_check_interval` default="eviction_check_interval", 10.0
- L4616: `_relax_managed_pd_router_endpoint` default=?
- L4630: `router_decode_policy` default=?
- L4972: `_relax_managed_pd_router_endpoint` default=?
- L4978: `router_decode_policy` default=?
- L4984: `router_dp_aware` default=?
- L4994: `sglang_dp_size` default=?
- L5065: `sglang_config` default=?
- L5073: `prefill_num_servers` default=?

### `relax/distributed/ray/teacher_manager.py`
- L50: `fp16` default=?

### `relax/engine/rollout/native_generation.py`
- L528: `eval_datasets` default=?

### `relax/engine/rollout/sglang_rollout.py`
- L148: `sglang_enable_deterministic_inference` default=?
- L368: `router_cb_timeout_duration_secs` default=?
- L825: `sglang_enable_deterministic_inference` default=?
- L1355: `eval_datasets` default=?
- L1429: `sglang_enable_deterministic_inference` default=?
- L1464: `sglang_enable_deterministic_inference` default=?

### `relax/engine/sft/data_pipeline_config.py`
- L30: `virtual_pipeline_model_parallel_size` default=?

### `relax/engine/sft/runtime.py`
- L116: `tensor_model_parallel_size` default=?
- L117: `pipeline_model_parallel_size` default=?
- L129: `use_gloo_process_groups` default=?
- L138: `hidden_dropout` default=?
- L139: `attention_dropout` default=?
- L150: `seq_length` default="seq_length", max_length) or max_length
- L208: `_i2i_sft_attempt_id` default="_i2i_sft_attempt_id", None
- L330: `moe_shared_expert_overlap` default=?

### `relax/models/common/chunked_mtp_loss.py`
- L274: `mtp_loss_normalization` default=?

### `relax/models/kimi_k3/configuration.py`
- L45: `cp_partition_mode` default=?

### `relax/models/qwen3_5/grouped_mlp.py`
- L45: `use_megatron_fsdp` default="use_megatron_fsdp", False
- L47: `fp8` default="fp8", None) and not getattr(args, "fp4", None
- L47: `fp4` default="fp4", None
- L64: `use_megatron_fsdp` default=?

### `relax/models/qwen3_5/recipe.py`
- L24: `use_megatron_fsdp` default="use_megatron_fsdp", False
- L26: `optimizer_cpu_offload` default="optimizer_cpu_offload", False
- L82: `overlap_moe_expert_parallel_comm` default="overlap_moe_expert_parallel_comm", False

### `relax/models/qwen3_8/qsa_attention_recompute.py`
- L31: `fp8` default=?
- L31: `fp4` default=?
- L31: `bf16` default=?
- L33: `cpu_offloading` default=?
- L33: `distribute_saved_activations` default=?
- L35: `virtual_pipeline_model_parallel_size` default=?
- L37: `cuda_graph_impl` default=?

### `relax/utils/arguments.py`
- L3316: `overlap_moe_expert_parallel_comm` default=?
- L4029: `enable_gloo_process_groups` default="enable_gloo_process_groups", False
- L4082: `sglang_dp_size` default=?
- L4111: `sglang_speculative_algorithm` default=?
- L4180: `sglang_enable_hierarchical_cache` default=?
- L4186: `overlap_moe_expert_parallel_comm` default=?
- L4370: `tensor_model_parallel_size` default="tensor_model_parallel_size", 1
- L4720: `overlap_moe_expert_parallel_comm` default=?

### `relax/utils/autoscaler/config.py`
- L268: `rollout_service_url` default="rollout_service_url", None

### `relax/utils/data/stream_dataloader.py`
- L1219: `is_vl_model` default="is_vl_model", False
- L1221: `uses_unsplit_forward` default=?

### `relax/utils/data/wholepack_policy.py`
- L37: `virtual_pipeline_model_parallel_size` default=?

### `relax/utils/metrics/adapters/wandb.py`
- L125: `sglang_enable_metrics` default=?

### `relax/utils/metrics/rollout_metrics.py`
- L130: `sglang_speculative_algorithm` default=?

### `relax/utils/model_source.py`
- L67: `_model_source_original_hf_checkpoint` default="_model_source_original_hf_checkpoint", None
- L73: `model_source` default="model_source", None

### `relax/utils/profile_utils.py`
- L54: `profile_ranks` default="profile_ranks", None
- L289: `sglang_profile_output_dir` default="sglang_profile_output_dir", None
- L306: `sglang_profile` default=?
- L309: `sglang_profile_steps` default="sglang_profile_steps", None
- L313: `sglang_profile_step_start` default="sglang_profile_step_start", None
- L314: `sglang_profile_step_end` default="sglang_profile_step_end", None
- L360: `sglang_profile_num_steps` default="sglang_profile_num_steps", None
- L368: `sglang_profile_activities` default=?
- L369: `sglang_profile_by_stage` default=?
- L370: `sglang_profile_with_stack` default=?
- L371: `sglang_profile_record_shapes` default=?
- L397: `sglang_profile_num_steps` default=?

### `relax/utils/record_function_tags.py`
- L360: `recompute_granularity` default=?
- L361: `recompute_num_layers` default="recompute_num_layers", None), int
- L362: `recompute_num_layers` default=?

### `relax/utils/score_centering.py`
- L26: `score_centering_vocab_size` default="score_centering_vocab_size", None
- L67: `sglang_rl_on_policy_target` default=?
- L72: `sglang_speculative_algorithm` default=?
- L82: `sglang_sampling_backend` default=?
- L84: `sglang_sampling_backend` default=?

### `relax/utils/tq/config.py`
- L296: `seq_length` default="seq_length", 0) or 0

### `relax/utils/training/nonfinite_grad.py`
- L21: `fp16` default=?
- L21: `bf16` default=?

### `scripts/tools/model_upload_hook.py`
- L168: `experiment_name` default="experiment_name", None
- L503: `experiment_name` default="experiment_name", None

### `tests/algorithms/test_policy_loss_dispatch.py`
- L276: `advantage_estimator` default=?

### `tests/backends/megatron/test_sft_train_prefetch.py`
- L60: `eval_at` default=?

### `tests/engine/rewards/custom_reward_fixtures.py`
- L18: `reward_delay` default=?
- L113: `reward_delay` default=?

## Compat helpers already centralized

- `megatron_norm_epsilon_arg_name`
- `ensure_megatron_rope_type`
- `megatron_dynamic_context_parallel`
- `megatron_context_parallel_size`
