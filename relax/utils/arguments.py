# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import json
import os
import sys
import warnings
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs

from relax.backends.sglang.arguments import sglang_parse_args
from relax.backends.sglang.arguments import validate_args as sglang_validate_args
from relax.utils import device as device_utils
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.model_source import is_model_source_alias, is_model_uri
from relax.utils.opd.opd_utils import (
    add_opd_arguments,
    is_managed_opd_teacher_enabled,
    maybe_enable_sglang_weights_cpu_backup,
    teacher_sglang_parse_args,
    validate_managed_opd_teacher_colocate_args,
    validate_opd_args,
)
from relax.utils.training.eval_config import (
    EvalDatasetConfig,
    build_eval_dataset_configs,
    build_named_prompt_data_configs,
    ensure_dataset_list,
)


logger = get_logger(__name__)


# Minimum required TransferQueue version and the command to upgrade to it.
_MIN_TQ_VERSION = "0.1.10.dev0"
_TQ_UPGRADE_CMD = (
    'pip install "transferqueue @ git+https://github.com/redai-studio/'
    'TransferQueue.git@58054a33834aadbcf76aacd6b1e32e25c030f2c9" --no-deps'
)

_MTP_DETACH_PATHS = ("embedding", "backbone", "lm-head")
_REMOVED_MTP_DETACH_FLAGS = ("--mtp-detach-main-model", "--no-mtp-detach-main-model")


def check_transfer_queue_version() -> None:
    """Fail fast if the installed TransferQueue is older than the required
    version.

    Only fully-async mode needs the newer TransferQueue, so this is called from
    ``parse_args`` only when ``--fully-async`` is set.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    from packaging.version import parse as parse_version

    try:
        installed = pkg_version("transferqueue")
    except PackageNotFoundError as e:
        raise ImportError(
            f"transferqueue is not installed. Install it with:\n    {_TQ_UPGRADE_CMD}\nor use the latest image."
        ) from e

    if parse_version(installed) < parse_version(_MIN_TQ_VERSION):
        raise RuntimeError(
            f"transferqueue {installed} is out of date (requires >= {_MIN_TQ_VERSION}). "
            f"Upgrade with:\n    {_TQ_UPGRADE_CMD}\nor use the latest image."
        )


def _positive_int(value: str) -> int:
    """argparse type that rejects non-positive integers at parse time."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def reset_arg(parser, name, **kwargs):
    """Reset the default value of a Megatron argument.

    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def _add_fp16_optimizer_arguments(parser):
    """Expose FP16 optimizer settings while preserving an unset sentinel."""
    reset_arg(
        parser,
        "--initial-loss-scale",
        type=float,
        default=None,
        help="Initial loss scale for dynamic FP16 loss scaling.",
    )
    reset_arg(
        parser,
        "--min-loss-scale",
        type=float,
        default=None,
        help="Minimum loss scale for dynamic FP16 loss scaling.",
    )
    reset_arg(
        parser,
        "--use-precision-aware-optimizer",
        action="store_true",
        default=None,
        help="Use TransformerEngine's precision-aware optimizer.",
    )
    if "--no-use-precision-aware-optimizer" not in parser._option_string_actions:
        parser.add_argument(
            "--no-use-precision-aware-optimizer",
            action="store_false",
            dest="use_precision_aware_optimizer",
            default=None,
            help="Disable TransformerEngine's precision-aware optimizer.",
        )
    reset_arg(
        parser,
        "--store-param-remainders",
        action="store_true",
        default=None,
        help="Store parameter remainders in the distributed optimizer.",
    )
    if "--no-store-param-remainders" not in parser._option_string_actions:
        parser.add_argument(
            "--no-store-param-remainders",
            action="store_false",
            dest="store_param_remainders",
            default=None,
            help="Do not store parameter remainders in the distributed optimizer.",
        )
    return parser


def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # Ray
        def add_serve_arguments(parser):
            parser.add_argument(
                "--resource",
                type=json.loads,
                help="JSON config dict",
            )
            parser.add_argument(
                "--ref-actor-config",
                type=json.loads,
                help="JSON config dict",
            )
            parser.add_argument(
                "--only-load-weight",
                action="store_true",
                default=False,
                help=("Only load weights for reference and actor fwd."),
            )
            parser.add_argument(
                "--fully-async",
                action="store_true",
                default=False,
                help=("Whether to use fully asynchronous training pipeline."),
            )
            parser.add_argument(
                "--hybrid",
                action="store_true",
                default=False,
                help=(
                    "Enable hybrid training mode. Combines the fully-async streaming data pipeline "
                    "(transfer queue + max-staleness) with colocate-style weight sharing "
                    "(TensorBackuper + _switch_model), so the actor handles ref / actor_fwd / advantages "
                    "internally on its own GPUs while rollout runs on a separate GPU placement group. "
                    "Mutually exclusive with passing --fully-async and --colocate together."
                ),
            )
            parser.add_argument(
                "--checkpoint-engine-backend",
                type=str,
                default=device_utils.get_dist_backend(),
                help=("Backend for checkpoint engine."),
            )
            parser.add_argument(
                "--rlsp-server-port",
                type=int,
                default=8234,
                help="Port number for the RLSP Server HTTP server.",
            )
            parser.add_argument(
                "--rotate-ckpt",
                action="store_true",
                default=False,
                help=("Whether to rotate checkpoints."),
            )
            parser.add_argument(
                "--max-actor-ckpt-to-keep",
                type=int,
                default=None,
                help=("Max number of actor checkpoints to keep."),
            )

            return parser

        def add_transfer_queue_arguments(parser):
            parser.add_argument(
                "--num-data-storage-units",
                type=int,
                default=1,
                help="Number of TransferQueue SimpleStorageUnit actors.",
            )
            parser.add_argument(
                "--per-rank-fetch",
                action="store_true",
                default=False,
                help=(
                    "Let every TP/PP rank pull its own copy from TransferQueue in parallel "
                    "instead of paying one rank-0 pickle + one TP/PP broadcast. Cross-rank "
                    "consistency relies on the TQ sampler's (partition_id, task_name, dp_rank, "
                    "batch_index) cache, which is PP/TP-invariant. Auto-disabled when "
                    "'rollout_routed_experts' is in data_fields (jagged NestedTensor bcast "
                    "path is incompatible). Recommended for multi-GPU training together with "
                    "--num-data-storage-units >= TP world size."
                ),
            )
            parser.add_argument(
                "--max-staleness",
                type=int,
                default=0,
                help="Max staleness for TransferQueue data system (0=on-policy).",
            )
            parser.add_argument(
                "--polling-mode",
                type=bool,
                default=True,
                help="Use polling for get metadata",
            )
            parser.add_argument(
                "--num-iters-per-train-update",
                type=int,
                default=1,
                help="Fully async pipeline num of iters every global batch.",
            )
            return parser

        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )
            parser.add_argument(
                "--critic-num-nodes", type=int, default=None, help="Number of nodes for training actor"
            )
            parser.add_argument(
                "--critic-num-gpus-per-node", type=int, default=None, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, "
                    "this param will be set to actor_num_gpus_per_node * actor_num_nodes "
                    "unless genRM is enabled (--genrm-model-path is set). "
                    "When genRM is enabled, rollout and genRM can share actor GPUs."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default=device_utils.get_dist_backend())
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=30)

            return parser

        def add_train_arguments(parser):
            # --train-backend is parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--qkv-format",
                type=str,
                choices=["thd", "bshd"],
                default="thd",
                help="The qkv layout for Megatron backend.",
            )
            parser.add_argument(
                "--true-on-policy-mode",
                action="store_true",
                default=False,
                help=(
                    "Skip the actor_fwd role and reuse the train forward's log_probs as "
                    "old_log_probs (ppo_kl ≡ 0, ratio ≡ 1), saving the dedicated actor_fwd "
                    "GPU group and one weight-sync per step. "
                    "Auto-enabled when --fully-async and "
                    "rollout_batch_size * n_samples_per_prompt == global_batch_size; no need "
                    "to pass this flag explicitly. The caller is responsible for ensuring the "
                    "regime is actually on-policy (e.g. --max-staleness=0, "
                    "--num-iters-per-train-update=1); off-policy use yields incorrect gradients. "
                    "TIS (--use-tis) and --get-mismatch-metrics remain valid in this mode."
                ),
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--selective-offload",
                action="store_true",
                default=False,
                help=(
                    "Offload the training actor to CPU during colocate sleep/wake by copying only the "
                    "live train state (weights + optimizer state) instead of torch_memory_saver's "
                    "whole-pool pause. Default is torch_memory_saver; use this on backends where "
                    "its VMM pause() is unavailable/unsafe (e.g. Kunlunxin P800)."
                ),
            )
            parser.add_argument(
                "--disable-weights-backuper",
                action="store_false",
                dest="enable_weights_backuper",
                help="Whether to disable weights backuper to save host memory.",
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            parser.add_argument(
                "--warm-hf-checkpoint-page-cache",
                action="store_true",
                default=False,
                help=(
                    "Pre-read HF checkpoint files into OS page cache before mmap-based loading, "
                    "to speed up NFS-backed reads. Applies uniformly to the main model (bridge), "
                    "and to genrm / teacher SGLang engines. Coordinated per node per checkpoint path, "
                    "so each unique checkpoint is streamed at most once per node."
                ),
            )
            parser.add_argument(
                "--disable-s3-model-download",
                action="store_true",
                default=False,
                help="Disable Relax-managed S3 model loading, including registered model-source providers",
            )
            parser.add_argument(
                "--disable-s3-model-cleanup",
                action="store_true",
                default=False,
                help="Keep downloaded S3 model weight shards in SHM after service initialization",
            )
            parser.add_argument(
                "--s3-model-download-workers",
                type=int,
                default=20,
                help="Number of concurrent workers used to download an S3 model to SHM",
            )
            parser.add_argument(
                "--s3-model-shm-root",
                type=str,
                default="/dev/shm",
                help=(
                    "S3 model SHM root; it must already exist on every model consumer node, "
                    "otherwise loading fails without falling back to disk"
                ),
            )
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom model provider function. "
                    "If set, we will use this function instead of the default model provider. "
                    "The function should have the signature "
                    "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                    "Example: 'my_module.my_model_provider'."
                ),
            )
            parser.add_argument(
                "--freeze-language-model",
                action="store_true",
                default=False,
                help="Whether to freeze the language model parameters (used in bridge mode for multimodal models).",
            )
            parser.add_argument(
                "--freeze-vision-model",
                action="store_true",
                default=False,
                help="Whether to freeze the vision model parameters (used in bridge mode for multimodal models).",
            )
            parser.add_argument(
                "--freeze-vision-projection",
                action="store_true",
                default=False,
                help="Whether to freeze the vision projection parameters (used in bridge mode for multimodal models).",
            )
            parser.add_argument(
                "--freeze-audio-model",
                action="store_true",
                default=False,
                help="Whether to freeze the audio encoder backbone parameters "
                "(used in bridge mode for multimodal models with an audio "
                "encoder).  Does NOT freeze the audio projection — pass "
                "--freeze-audio-projection for that.",
            )
            parser.add_argument(
                "--freeze-audio-projection",
                action="store_true",
                default=False,
                help="Whether to freeze the audio→LM projection parameters "
                "(used in bridge mode for multimodal models with an audio "
                "encoder).  Independent of --freeze-audio-model.",
            )
            parser.add_argument(
                "--vision-dp-when-tp",
                action="store_true",
                default=False,
                help="Split vision encoder workload across TP ranks (data-parallel over TP). "
                "Each TP rank processes a chunk of images, then all-reduce gathers the full embedding.",
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to recompute the loss function to save memory during training.",
            )
            parser.add_argument(
                "--recompute-loss-function-use-reentrant",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Whether loss-function recomputation uses PyTorch's reentrant checkpoint implementation. "
                "Defaults to True to preserve existing behavior; pass "
                "--no-recompute-loss-function-use-reentrant to use the recommended non-reentrant implementation.",
            )
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )
            parser.add_argument(
                "--sft-logits-chunk-size",
                type=int,
                default=1024,
                help="SFT only: chunk size for the lm_head + CE matmul under "
                "sft_loss_function_chunked (avoids materializing full [B,S,V/TP] logits). "
                "Independent from --log-probs-chunk-size, which only chunks the post-logits "
                "log_prob reduce used by RL paths and SFT eval (PPL).",
            )
            parser.add_argument(
                "--sft-chunked-logits",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="SFT only: defer lm_head into the loss and chunk the lm_head + CE "
                "matmul (sft_loss_function_chunked) to avoid materializing full [B,S,V/TP] "
                "logits. Default off — legacy external-loss SFT path materializes full logits "
                "and runs CE externally. Set --sft-chunked-logits to opt in. MTP head calls "
                "continue to use the real output_layer before the main head is deferred. "
                "Incompatible when embeddings are tied "
                "(--untie-embeddings-and-output-weights not set: output_layer is built with "
                "skip_weight_param_allocation=True so output_layer.weight is None and the "
                "chunked path's lm_head matmul has nothing to multiply against; tied models "
                "are small enough that the chunked path's memory win is marginal anyway).",
            )
            parser.add_argument(
                "--only-train-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help=r"""List of regex patterns of parameter names to TRAIN. All other parameters will be FROZEN.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Train ONLY MoE experts:
                            --only-train-params-name-list experts

                        2. Train ONLY Indexer parameters:
                            --only-train-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Train ONLY Layer 20 to 23:
                            --only-train-params-name-list layers\.2[0-3]\.
                        """,
            )

            parser.add_argument(
                "--freeze-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to FREEZE. Other parameters will remain trainable.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Freeze Embeddings and Output Layer (common for fine-tuning):
                            --freeze-params-name-list embedding output_layer

                        2. Freeze Indexer parameters:
                            --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Freeze specific projection layers (e.g., all Gate/Up projections):
                            --freeze-params-name-list linear_fc1
                        """,
            )
            parser.add_argument(
                "--allgather-cp",
                action="store_true",
                default=False,
            )

            # ---- SFT / Predict ----
            parser.add_argument(
                "--custom-dataset-class",
                "--custom-dataset-class-path",
                dest="custom_dataset_class_path",
                type=str,
                default=None,
                help=(
                    "Import path to a custom SFT streaming dataset class. "
                    "The class must define from_args(args, *, tokenizer, processor_pool, pad_token_ids) "
                    "and expose the SFTStreamingDataset public methods."
                ),
            )
            parser.add_argument(
                "--eval-size",
                type=float,
                default=None,
                help=(
                    "Carve a held-out eval split from --prompt-data instead of providing a separate "
                    "--eval-prompt-data. A value <1 is treated as a fraction of the train dataset "
                    "(e.g. 0.05 → 5%); a value ≥1 is treated as an absolute sample count. "
                    "Rows are randomly split once using --seed; held-out rows are excluded from every "
                    "training epoch. Mutually exclusive with --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--sft-max-in-flight-steps",
                type=int,
                default=None,
                help=(
                    "SFT-only name for the train-step TransferQueue buffer depth. "
                    "This is the maximum number of sft_<step> partitions allowed in flight, "
                    "including the current train step. When set, it maps to "
                    "--max-staleness = sft_max_in_flight_steps - 1."
                ),
            )
            parser.add_argument(
                "--sft-prefetch-buffer-size",
                type=int,
                default=256,
                help=(
                    "Max pre-loaded samples held by the SFT streaming dataset's PrefetchBuffer. "
                    "Set to 0 to disable prefetching; the producer will then fall back to an "
                    "asyncio.gather path over the ProcessorPool for batch-level parallelism."
                ),
            )
            parser.add_argument(
                "--sft-prefetch-chunk-size",
                type=int,
                default=32,
                help="Chunk size dispatched to the SFT prefetch thread-pool per round.",
            )
            parser.add_argument(
                "--sft-prefetch-num-workers",
                type=int,
                default=4,
                help="Worker threads inside the SFT PrefetchBuffer for I/O-bound media decoding.",
            )
            parser.add_argument(
                "--sft-oversize-strategy",
                type=str,
                default="keep",
                choices=["skip", "keep", "truncate_left", "truncate_right", "custom"],
                help=(
                    "How to handle SFT samples whose (expanded) length exceeds per-GPU capacity. "
                    "All branches emit a WARNING log per oversized sample. "
                    "`skip` drops the sample; `keep` (default) returns it unchanged (may OOM downstream); "
                    "`truncate_left` keeps the last `capacity` tokens; `truncate_right` keeps the first "
                    "`capacity` tokens; `custom` delegates to --sft-oversize-custom-function-path. "
                    "Note: truncating multimodal samples in-place may misalign multimodal_train_inputs — "
                    "use `custom` if you need to also trim media inputs."
                ),
            )
            parser.add_argument(
                "--sft-oversize-custom-function-path",
                type=str,
                default=None,
                help=(
                    "Required when --sft-oversize-strategy custom. Importable path to a function with "
                    "signature `def truncate(tokens, loss_mask, capacity, idx) -> (tokens, loss_mask) | None`. "
                    "Returning None is treated as skip."
                ),
            )
            parser.add_argument(
                "--sft-invalid-multimodal-strategy",
                type=str,
                default="error",
                choices=["error", "skip"],
                help=(
                    "How to handle SFT samples where a rendered image/video/audio marker cannot resolve to "
                    "exactly one inline or top-level media source, or duplicate sources are supplied. "
                    "`error` (default) fails before model forward; `skip` emits a WARNING and refills the batch."
                ),
            )
            parser.add_argument(
                "--sft-tq-timeout-minutes",
                type=int,
                default=None,
                help=(
                    "SFT-only timeout (in minutes) for the producer's TransferQueue waits. "
                    "If the consumer dies, the SFT producer would otherwise spin forever on "
                    "_wait_for_buffer_capacity. On timeout the producer raises TimeoutError "
                    "and the job crashes. Defaults to --distributed-timeout-minutes."
                ),
            )
            parser.add_argument(
                "--sft-predict-interval",
                type=int,
                default=None,
                help=(
                    "When set under --loss-type sft, every N rollout steps run a predict pass "
                    "over the eval dataset (--eval-prompt-data/--eval-config or --eval-size) and write "
                    "completions to <save>/predict/predictions_step_<rollout_id>.jsonl. "
                    "Setting this flag implicitly spins up the Rollout role (SGLang must be online to serve generation)."
                ),
            )
            parser = _add_fp16_optimizer_arguments(parser)
            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--sglang-hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "Optional override for the HF checkpoint that SGLang loads. "
                    "When set, SGLang's model_path uses this directory instead of "
                    "args.hf_checkpoint, while training-side consumers (Megatron "
                    "loader, AutoConfig, tokenizer) keep using args.hf_checkpoint. "
                    "Used by INT4 QAT runs so SGLang loads the source compressed-"
                    "tensors directory directly (registering weight_packed/scale/shape "
                    "params) while training reads from a separately-prepared BF16 "
                    "checkpoint."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="relax.engine.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `status`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )

            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            parser.add_argument(
                "--partial-rollout-max-aborted-count",
                type=int,
                default=None,
                help=(
                    "Maximum number of times a sample can be aborted before it is protected from further interruption. "
                    "When a sample's abort count reaches this threshold, it will not be aborted in the next rollout "
                    "and will be allowed to complete its generation. Requires --partial-rollout to be effective. "
                    "If not set (None), no staleness protection is applied."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to mask previous generation in partial rollout. "
                    "If set, only on-policy generated tokens will be used in training"
                ),
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            parser.add_argument(
                "--rollout-external",
                action="store_true",
                default=False,
                help="Use external SGLang instances instead of launching them inside the framework.",
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--use-health-check",
                action="store_true",
                default=False,
                help="Whether to enable the global health check system. "
                "When enabled, the Controller's HealthManager monitors all services "
                "and triggers automatic restarts (in-place or global) on failure.",
            )
            parser.add_argument(
                "--max-global-restart",
                type=int,
                default=3,
                help="Maximum number of global restarts allowed. "
                "If the global restart count exceeds this limit, the training process "
                "will raise an error and terminate instead of attempting another restart. "
                "Only effective when --use-health-check is enabled.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            parser.add_argument(
                "--rollout-http-timeout",
                type=float,
                default=120.0,
                help="Timeout in seconds for actor HTTP probes to rollout and actor_fwd services.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            parser.add_argument(
                "--rollout-health-check-max-consecutive-failures",
                type=int,
                default=2,
                help="Number of consecutive health check failures before killing a rollout engine. "
                "A single timeout (e.g. engine busy with a large batch) will not kill the engine. "
                "Only after this many consecutive failures will the engine be killed.",
            )
            parser.add_argument(
                "--rollout-engine-init-timeout",
                type=float,
                default=3600.0,
                help="Total timeout in seconds to wait for ALL rollout engines to finish init() "
                "(server launch + weight loading) at training startup. Acts as a soft barrier so "
                "stragglers caused by storage/IO jitter on large clusters do not leak into "
                "downstream NCCL collectives. Progress is logged every 60s while waiting.",
            )
            # Elastic rollout scale-out arguments
            parser.add_argument(
                "--scale-out-timeout",
                type=float,
                default=300.0,
                help="Timeout in seconds for all scale-out operations (engine startup, connect, health check, weight sync, and default).",
            )
            parser.add_argument(
                "--scale-out-partial-success-policy",
                type=str,
                default="rollback_all",
                choices=["rollback_all", "keep_partial"],
                help="Policy for handling partial success during scale-out. 'rollback_all' reverts all engines on any failure. 'keep_partial' keeps successfully scaled engines.",
            )
            parser.add_argument(
                "--scale-weight-sync-precheck",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Run an independent NCCL precheck before scale-out weight sync; fail-closed on "
                "incompatible transport. Disable with --no-scale-weight-sync-precheck.",
            )
            # Elastic rollout scale-in arguments
            parser.add_argument(
                "--scale-in-drain-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for in-flight requests to drain before force-aborting. Default: 30s.",
            )
            parser.add_argument(
                "--scale-in-shutdown-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds for graceful SGLang engine shutdown. If exceeded, ray.kill is used. Default: 30s.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            parser.add_argument(
                "--use-streaming-dataset",
                action="store_true",
                default=False,
                help="Use streaming dataset for memory-efficient data loading.",
            )
            parser.add_argument(
                "--streaming-buffer-size",
                type=int,
                default=10000,
                help="Buffer size for streaming dataset.",
            )
            parser.add_argument(
                "--prefetch-chunk-size",
                type=int,
                default=32,
                help="Number of samples to dispatch to the thread-pool in each prefetch round. "
                "Larger values increase throughput but also memory pressure. Only effective when "
                "--use-streaming-dataset is set and the dataset contains multimodal data.",
            )
            parser.add_argument(
                "--prefetch-max-cached",
                type=int,
                default=256,
                help="Maximum number of pre-loaded samples kept in the prefetch cache. "
                "When the cache is full the background prefetch thread pauses until consumers "
                "free space. Set to 0 to disable prefetching. Only effective when "
                "--use-streaming-dataset is set and the dataset contains multimodal data.",
            )
            parser.add_argument(
                "--prefetch-num-workers",
                type=int,
                default=1,
                help="Number of parallel worker threads inside the prefetch buffer for "
                "I/O-bound media decoding (video/image). Set to 1 to serialise all "
                "decoding (safest for FFmpeg which is not fully thread-safe). "
                "Higher values increase parallelism but may trigger EAGAIN errors "
                "on some platforms. Only effective when prefetching is enabled.",
            )
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs over the dataset. When set, "
                    "`actual_num_rollout = num_epoch * dataset_size // rollout_batch_size`. "
                    "If both --num-rollout and --num-epoch are set, the smaller of the two wins "
                    "(epoch acts as a cap). At least one of --num-rollout / --num-epoch must be set. "
                    "Applies to both RL and SFT."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="relax.engine.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Supports jsonl/parquet paths, directories, file lists, and row slices. "
                    "Each row should contain --input-key; --label-key is used when the prompt row stores "
                    "prompt and response separately. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--task-type",
                type=str,
                choices=["causal_lm", "seq_cls"],
                default="causal_lm",
                help="SFT task type. `seq_cls` replaces the vocabulary head with a classification head.",
            )
            parser.add_argument(
                "--num-labels",
                type=int,
                default=None,
                help="Number of classes for --task-type seq_cls.",
            )
            parser.add_argument(
                "--problem-type",
                type=str,
                choices=["single_label_classification", "multi_label_classification"],
                default="single_label_classification",
                help="Classification objective used by --task-type seq_cls.",
            )
            parser.add_argument(
                "--classification-threshold",
                type=float,
                default=0.5,
                help="Sigmoid threshold used by multi-label classification evaluation.",
            )
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
                ),
            )
            parser.add_argument(
                "--conversation-key-map",
                type=json.loads,
                default=None,
                help=(
                    "JSON string that rewrites non-OpenAI conversation messages into the OpenAI "
                    "{role, content} shape that SFT expects. A single flat map covers both "
                    "field-name renames (applied to every message-dict key) and role-value "
                    "renames (applied only to the resulting `role` field, so message bodies "
                    "containing the same words are left untouched). "
                    'Example for sharegpt: \'{"from":"role","value":"content","human":"user","gpt":"assistant"}\'. '
                    "Only consulted when --label-key is unset (i.e., --input-key holds a full messages list)."
                ),
            )
            parser.add_argument(
                "--use-audio-in-video",
                action="store_true",
                default=False,
                help="Whether to process the audio in the video or not.",
            )
            # Multimodal data processing parameters
            parser.add_argument(
                "--image-max-token-num",
                type=int,
                default=None,
                help="Maximum number of tokens for image processing. If not set, uses default value (16384).",
            )
            parser.add_argument(
                "--image-min-token-num",
                type=int,
                default=None,
                help="Minimum number of tokens for image processing. If not set, uses default value (4).",
            )
            parser.add_argument(
                "--video-min-token-num",
                type=int,
                default=None,
                help="Minimum number of tokens for video frame processing. If not set, uses default value (128).",
            )
            parser.add_argument(
                "--video-max-token-num",
                type=int,
                default=None,
                help="Maximum number of tokens for video frame processing. If not set, uses default value (768).",
            )
            parser.add_argument(
                "--video-fps",
                type=float,
                default=None,
                help="Target FPS for video processing. If not set, uses default value (2.0).",
            )
            parser.add_argument(
                "--video-fps-min-frames",
                type=int,
                default=None,
                help="Minimum number of frames for video processing. If not set, uses default value (4).",
            )
            parser.add_argument(
                "--video-fps-max-frames",
                type=int,
                default=None,
                help="Maximum number of frames for video processing. If not set, uses default value (768).",
            )
            parser.add_argument(
                "--image-resize-scale-factor",
                type=int,
                default=None,
                help=(
                    "Scale factor for image resize dimension alignment. "
                    "Default uses patch_size * spatial_merge_size. Set to 0 to disable alignment."
                ),
            )
            parser.add_argument(
                "--audio-sample-rate",
                type=int,
                default=None,
                help="Sample rate for audio processing. If not set, uses default value (16000).",
            )
            parser.add_argument(
                "--frame-factor",
                type=int,
                default=None,
                help="Frame count alignment factor. If not set, uses default value (2).",
            )
            parser.add_argument(
                "--mm-processor-pool-size",
                type=int,
                default=0,
                help=(
                    "Size of the multimodal processor pool. "
                    "0 (default) disables the processor pool and uses ThreadPoolExecutor instead. "
                    "When set to a positive integer, creates a ProcessPoolExecutor with the specified number of workers "
                    "for true parallelism without GIL contention."
                ),
            )
            parser.add_argument(
                "--encode-max-workers",
                type=_positive_int,
                default=None,
                help=(
                    "Worker threads for the shared media-encoding thread pool (image/video/audio "
                    "encoding offloaded from the asyncio event loop). Positive integer. If unset, "
                    "falls back to $RELAX_ENCODE_MAX_WORKERS, then to min(32, usable CPU count)."
                ),
            )
            parser.add_argument(
                "--custom-prompt-path",
                type=str,
                default=None,
                help=(
                    "Dotted import path to a custom function that transforms the prompt before "
                    "conversation/multimodal processing. The function signature must be "
                    "`def custom_fn(prompt, data: dict) -> prompt`, where `prompt` is the raw "
                    "value from the dataset and `data` is the full sample dict. "
                    "Example: my_package.prompt_utils.add_prefix"
                ),
            )
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                default=None,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                    "If omitted but --global-batch-size is set, it is derived as "
                    "`global_batch_size // n_samples_per_prompt`."
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. "
                    "Note that this may allocate the different response of the same prompt into different training steps. "
                    "In fully-async + --use-dynamic-batch-size mode this is effectively always on: the "
                    "StreamingTokenBudgetSampler already balances tokens across DP ranks per sample, so the "
                    "flag is accepted but has no additional effect there."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, relax will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            parser.add_argument(
                "--system-prompt",
                type=str,
                default=None,
                help=(
                    "Optional system prompt added before user input."
                    "The final message will be <system_prompt> + <dataset_prompt>."
                ),
            )
            parser.add_argument(
                "--use-agentic-rollout",
                action="store_true",
                default=False,
                help="Enable agentic rollout mode. Automatically sets --rollout-function-path to the agentic rollout entry point.",
            )
            parser.add_argument(
                "--agent-command",
                type=str,
                default=None,
                help=(
                    "Managed-command agent entry command for agentic rollout. "
                    "Required when --use-agentic-rollout is set."
                ),
            )
            parser.add_argument(
                "--agent-cwd",
                type=str,
                default=None,
                help=(
                    "Working directory for the managed-command agent process. "
                    "Required when --use-agentic-rollout is set."
                ),
            )
            parser.add_argument(
                "--agent-timeout",
                type=float,
                default=1800.0,
                help=(
                    "Active runtime budget in seconds for each managed-command agent session. "
                    "The clock runs while the session is admitted and pauses while it is gated; "
                    "on timeout, SIGTERM is sent to the managed agent process group."
                ),
            )
            parser.add_argument(
                "--agent-env",
                nargs="+",
                default=[],
                help=(
                    "Extra environment variables for the managed-command agent process, formatted as KEY=VALUE. "
                    "Examples: --agent-env FOO=bar; --agent-env FOO=bar BAZ=qux; "
                    "--agent-env 'FOO=value with spaces' BAZ=qux."
                ),
            )
            parser.add_argument(
                "--agentic-tool-call-parser",
                type=str,
                default=None,
                help=("SGLang tool-call parser for agentic rollout. Runs only when tools are present."),
            )
            parser.add_argument(
                "--agentic-reasoning-parser",
                type=str,
                default=None,
                help="SGLang reasoning parser for agentic rollout.",
            )
            parser.add_argument(
                "--agentic-custom-advantage-path",
                type=str,
                default=None,
                help="Custom group-level advantage function for explicit agentic exports.",
            )
            parser.add_argument(
                "--agentic-concurrency",
                type=int,
                default=None,
                help=(
                    "Maximum number of resident training groups across Prepare and Runtime. "
                    "If unset, defaults to over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--agentic-eval-concurrency",
                type=int,
                default=None,
                help=("Maximum number of resident eval groups. If unset, derives from the training session capacity."),
            )
            parser.add_argument(
                "--agentic-prelaunch",
                action="store_true",
                default=False,
                help="Prelaunch agent processes for the next rollout step.",
            )
            parser.add_argument(
                "--agentic-session-lifecycle",
                action="store_true",
                default=False,
                help=(
                    "Tag every backend attempt with its Agentic session ID and release the session's radix-cache "
                    "entries when the session terminates. Requires --sglang-enable-session-radix-cache."
                ),
            )
            parser.add_argument(
                "--agentic-program-admission",
                action="store_true",
                default=False,
                help=(
                    "Gate each backend attempt with a cluster-wide execution-token budget. Capacity-bound attempts "
                    "wait in a global FIFO queue; stale or unavailable metrics fail open to the request limiter."
                ),
            )
            parser.add_argument(
                "--agentic-admission-headroom",
                type=float,
                default=0.90,
                help="Fraction of aggregate KV token capacity usable as the admission ceiling (0, 1].",
            )
            parser.add_argument(
                "--agentic-admission-expected-decode-cap",
                type=int,
                default=None,
                help="Upper bound on expected decode tokens per reservation. Defaults to --rollout-max-response-len.",
            )
            parser.add_argument(
                "--agentic-admission-pressure-threshold",
                type=float,
                default=0.92,
                help="Per-worker KV token usage at or above which new attempts wait, in (0, 1].",
            )
            parser.add_argument(
                "--agentic-admission-max-wait-s",
                type=float,
                default=30.0,
                help="Maximum FIFO wait before the attempt bypasses admission, in seconds. Must be >= 0.",
            )
            parser.add_argument(
                "--agentic-admission-scope",
                type=str,
                default="train",
                choices=["train", "all"],
                help="Which scopes admission applies to: 'train' only (default) or 'all' (train + eval).",
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)
            parser.add_argument("--eval-top-p", type=float, default=None)
            parser.add_argument("--eval-top-k", type=int, default=None)
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            parser.add_argument("--eval-max-prompt-len", type=int, default=None)
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            parser.add_argument("--eval-max-context-len", type=int, default=None)

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            parser.add_argument(
                "--lora-rank",
                type=int,
                default=0,
                help="LoRA rank for parameter-efficient fine-tuning (0=disabled).",
            )
            parser.add_argument(
                "--lora-alpha",
                type=int,
                default=32,
                help="LoRA alpha scaling factor for learning rate adjustment.",
            )
            parser.add_argument(
                "--lora-target-modules",
                type=str,
                nargs="+",
                default=["linear_qkv", "linear_proj"],
                help=(
                    "Target modules for LoRA (Megatron-style names, e.g. linear_qkv, "
                    "linear_proj, linear_fc1, linear_fc2). Expanded to HF-style names "
                    "automatically when exporting the adapter."
                ),
            )
            parser.add_argument(
                "--lora-scope",
                type=str,
                choices=["all", "language", "vision"],
                default="all",
                help=(
                    "Which model region receives LoRA adapters (VL models only; no effect "
                    "on text models, which have no vision tower). 'all' (default) wraps every "
                    "matched module including the vision tower; 'language' excludes the vision "
                    "tower / projector / audio encoder (typical for language-only RL); 'vision' "
                    "wraps only those. Controls adapter INJECTION, not base-weight freezing."
                ),
            )
            parser.add_argument(
                "--lora-dropout",
                type=float,
                default=0.0,
                help="Dropout probability for LoRA layers.",
            )
            parser.add_argument(
                "--lora-merge-mode",
                action="store_true",
                default=False,
                help=(
                    "Merge LoRA adapters into base weights before weight synchronization. "
                    "Simplifies rollout but slower rollout inference."
                ),
            )
            parser.add_argument(
                "--lora-adapter-mode",
                action="store_true",
                default=False,
                help=(
                    "Enable LoRA adapter mode: sync the base model once, then each step push only "
                    "the trained LoRA adapter to the rollout engine via SGLang's runtime LoRA API "
                    "(rollouts select it via lora_path). Saves per-step full-weight-sync bandwidth. "
                    "Requires --enable-lora (auto-set by the engine) and sglang_dp_size == 1. "
                    "Supported in colocate and fully-async modes. Mutually exclusive with --lora-merge-mode."
                ),
            )
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. "
                    "The model will be saved to `save_hf.format(rollout_id)`. "
                ),
            )
            parser.add_argument(
                "--save-hf-dtype",
                type=str,
                choices=["bf16", "fp8"],
                default="bf16",
                help=(
                    "Precision for online --save-hf export. 'bf16' (default) preserves prior "
                    "behavior; 'fp8' streams e4m3 quantized shards. FP8 requires --save-hf and "
                    "an --hf-checkpoint backed by safetensors."
                ),
            )
            parser.add_argument(
                "--save-hf-fp8-quant-mode",
                type=str,
                choices=["block", "channel", "tensor"],
                default="block",
                help="FP8 quantization strategy for --save-hf-dtype fp8 (default: block).",
            )
            parser.add_argument(
                "--save-hf-fp8-block-size",
                type=int,
                nargs=2,
                default=None,
                metavar=("ROWS", "COLS"),
                help="Block shape for block FP8; defaults to (128, 128) when quant-mode=block.",
            )
            parser.add_argument(
                "--save-hf-post-hook-path",
                type=str,
                default=None,
                help=(
                    "Dotted path (module.func) to a callable invoked on WORLD rank 0 after each "
                    "HF checkpoint is written. Signature: "
                    "`hook(args, hf_path: str, rollout_id: int, *, dtype: str, is_lora: bool) -> None`. "
                    "Called synchronously; heavy work (uploads, RPCs) must be enqueued to a background "
                    "thread by the hook itself. Exceptions are logged and swallowed."
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument("--num-critic-only-steps", type=int, default=0, help="Number of critic only steps")
            parser.add_argument("--critic-load", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-save", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-lr", type=float, default=None, help="The lr for critic model")
            parser.add_argument("--critic-train-only", action="store_true", default=False, help="Only train critic")
            parser.add_argument(
                "--critic-lr-warmup-iters",
                type=int,
                default=0,
                help="number of iterations to linearly warmup for critic model.",
            )

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft", "sft_loss", "sft-loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft (or deprecated sft_loss/sft-loss), "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Choose KL loss type: kl, k2, k3, low_var_kl",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                    "sapo",
                    "cispo",
                    "m2po",
                    "rloo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator. "
                    "'rloo' uses a leave-one-out baseline with an unclipped REINFORCE loss (sync only)."
                ),
            )
            parser.add_argument(
                "--sapo-tau-pos",
                type=float,
                default=1.0,
                help="Temperature for positive advantages in SAPO (default: 1.0)",
            )
            parser.add_argument(
                "--sapo-tau-neg",
                type=float,
                default=1.05,
                help="Temperature for negative advantages in SAPO (default: 1.05)",
            )
            parser.add_argument(
                "--m2po-kl2-budget",
                type=float,
                default=0.01,
                help="M2PO second-moment budget per harmful token (KL2_budget in paper; paper uses 0.04, default: 0.01)",
            )
            parser.add_argument(
                "--m2po-miniclip-low",
                type=float,
                default=0.3,
                help="M2PO minimum lower clip epsilon floor (paper default: 0.3)",
            )
            parser.add_argument(
                "--m2po-miniclip-high",
                type=float,
                default=0.5,
                help="M2PO minimum upper clip epsilon floor (paper default: 0.5)",
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Whether to calculate the mismatch metrics.",
            )
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Whether to reset optimizer states after each rollout. "
                    "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
                ),
            )
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                    "If not set, we will use the logprobs from the actor model."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/Dr.GRPO/custom_reducer.py:get_pg_loss_reducer).",
            )

            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="The routing replay technique from https://arxiv.org/abs/2507.18071",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370",
            )
            parser.add_argument(
                "--optimize-routing-replay",
                action="store_true",
                default=False,
                help="Enable async D-to-H optimization for rollout routing replay. "
                "Reduces rollout latency (~20%%) via staging buffer and keeps broadcast "
                "results on GPU to avoid redundant CPU round-trips.",
            )
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Whether to enable Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="The threshold for Off-Policy Sequence Masking (OPSM).",
            )
            return parser

        def add_on_policy_distillation_arguments(parser):
            return add_opd_arguments(parser)

        def add_router_arguments(parser):
            parser.add_argument(
                "--use-slime-router",
                action="store_true",
                default=False,
                help="Whether to use SlimeRouter for text-based routing instead of SGLang token-based routing",
            )
            parser.add_argument(
                "--slime-router-middleware-paths",
                type=str,
                nargs="+",
                default="",
            )
            parser.add_argument(
                "--slime-router-timeout",
                type=float,
                default=None,
                help="Timeout for SlimeRouter HTTP requests in seconds.",
            )
            parser.add_argument(
                "--slime-router-max-connections",
                type=int,
                default=None,
                help="Max connections for SlimeRouter HTTP client.",
            )
            parser.add_argument(
                "--slime-router-health-check-failure-threshold",
                type=int,
                default=3,
                help="Number of consecutive failures before marking a worker as unhealthy.",
            )
            parser.add_argument(
                "--slime-router-sticky",
                action="store_true",
                default=False,
                help="Enable sticky-session routing in SlimeRouter: pin a routing key (read from the "
                "X-SMG-Routing-Key header) to a worker so repeated requests for the same key reuse that "
                "worker's prefix/KV cache. Keyless requests and the initial pin fall back to least-load "
                "selection; a live pin is never redistributed (only remapped when its worker leaves the "
                "healthy set). Has no effect unless --use-slime-router is set.",
            )
            parser.add_argument(
                "--slime-router-sticky-idle-secs",
                type=float,
                default=600.0,
                help="Evict a sticky routing-key -> worker assignment after it has been idle (not routed to) "
                "for this many seconds, bounding the map against unbounded routing-key cardinality. Scanned on "
                "the SlimeRouter health-check cadence. Requires --slime-router-sticky.",
            )
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                    "Specify the key in the reward dict using this argument.",
                ),
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser

        # tensorboard
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            parser.add_argument("--use-tensorboard", action=argparse.BooleanOptionalAction, default=True)
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
            )
            parser.add_argument("--tb-experiment-name", type=str, default=None)

            return parser

        # clearml
        def add_clearml_arguments(parser):
            # TODO(yuetian): Reuse tb-experiment-name --tb-project-name as the ClearML experiment name currently.
            # Need to refactor the experiment management part later.
            parser.add_argument("--use-clearml", action="store_true", default=False)

            return parser

        # apprise
        def add_apprise_arguments(parser):
            parser.add_argument(
                "--notify-urls",
                type=str,
                default=None,
                help=(
                    "Comma-separated list of Apprise notification URLs. "
                    "Example: https://github.com/caronc/apprise?tab=readme-ov-file#productivity-based-notifications"
                ),
            )

            return parser

        # metrics service
        def add_metrics_service_arguments(parser):
            """Add metrics service arguments for centralized metrics
            collection."""
            parser.add_argument(
                "--use-metrics-service",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Enable metrics service for centralized metrics collection and reporting. "
                    "Default: True. Use --no-use-metrics-service to disable."
                ),
            )
            parser.add_argument(
                "--timeline-dump-dir",
                type=str,
                default=None,
                help=(
                    "Directory to dump timeline trace events (Chrome Trace format). "
                    "If not set, timeline tracing is disabled. "
                    "The timeline will be dumped to {timeline_dump_dir}/timeline_step_{{step}}.json"
                ),
            )
            return parser

        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            # --load-debug-rollout-data, --debug-rollout-only, --debug-train-only
            # are parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--load-forge-rollout-data",
                type=str,
                default=None,
                help=(
                    "Path (or {rollout_id} template) to a dumped rollout .pt replayed by "
                    "relax.engine.rollout.forge_load.generate_rollout. A path without the placeholder is "
                    "a literal file reused for every rollout; a path with {rollout_id} loads a per-rollout "
                    "file. Unlike --load-debug-rollout-data, this does NOT set skip_sglang, so sglang, "
                    "router, weight sync and the colocate offload/onload dance stay live (the point: "
                    "measuring real colocate memory at long context)."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            parser.add_argument(
                "--rollout-result-dir",
                type=str,
                default=None,
                help=(
                    "Directory to save per-step rollout result JSONL files. "
                    "Each step produces one JSONL file with prompt, response, reward, "
                    "and sequence length for every sample. "
                    "Defaults to {save}/rollout_result when --save is set. "
                    "Set to empty string to disable."
                ),
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            parser.add_argument(
                "--memory-snapshot-dir",
                type=str,
                default=None,
                help=("Directory for memory snapshot dumps. Defaults to traces/<tb_experiment_name>/memory_snapshot."),
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
                help="Number of rollout steps after which to dump the memory snapshot. "
                "For example, --memory-snapshot-num-steps 3 dumps after step 2 (0-indexed).",
            )
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
            )
            parser.add_argument(
                "--profile-with-stack",
                action="store_true",
                default=False,
                help="Record stack information in profiler traces.",
            )
            parser.add_argument(
                "--profile-with-memory",
                action="store_true",
                default=False,
                help="Record memory information in profiler traces.",
            )
            parser.add_argument(
                "--profile-with-flops",
                action="store_true",
                default=False,
                help="Estimate FLOPs in profiler traces.",
            )
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
            )
            parser.add_argument("--check-weight-update-equal", action="store_true")
            parser.add_argument(
                "--enable-cuda-memory-check",
                action="store_true",
                default=False,
                help=(
                    "Enable memory check around low-level NCCL communication calls. "
                    "WARNING: this introduces ~20%% training performance degradation."
                ),
            )
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--http-proxy", type=str, default=None)
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--rm-type-fallback",
                type=str,
                default=None,
                help=(
                    "Fallback for samples whose reward type is unknown or missing. "
                    "None (default) keeps the current behavior of raising. 'zero' scores "
                    "the sample 0.0 (reward-key aware) with a warning. Any registered "
                    "reward type name routes degraded samples to that reward instead."
                ),
            )
            parser.add_argument(
                "--rm-type-infer",
                action="store_true",
                default=False,
                help=(
                    "Infer the reward type from the sample label via registered label "
                    "matchers when neither sample metadata nor --rm-type provides one. "
                    "Also enables conflict detection (warn; the explicit type wins)."
                ),
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            parser.add_argument(
                "--defer-reward-to-post-process",
                action="store_true",
                default=False,
                help=(
                    "When set, actor.update_weights will NOT re-onload GenRM at the end of "
                    "weight sync. Use this with --rm-type dummy + --custom-reward-post-process-path "
                    "when the post-process function manages GenRM sleep/wake itself (shared-bundles "
                    "colocate: rollout owns all GPUs during generate, GenRM owns them during scoring)."
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "If set, this function will replace the default _convert_samples_to_train_data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            parser.add_argument(
                "--reward-max-concurrency",
                type=int,
                default=64,
                help=(
                    "Maximum number of concurrent reward computations (per sample; "
                    "group_rm=False scores by sample). This controls the global "
                    "asyncio.Semaphore that limits how many reward tasks run at once. "
                    "Reward counts by sample, so over_sampling=N groups x n_samples = N*ns "
                    "samples per step flow through this gate. Each sample fans out to several "
                    "LLM judges, so when raising this keep it within the upstream judge "
                    "endpoint concurrency (see reward_config *.yaml max_concurrency). "
                    "Default 64; set explicitly on the launch command when over-sampling."
                ),
            )
            parser.add_argument(
                "--reward-num-workers",
                type=int,
                default=16,
                help=(
                    "Number of Ray RewardWorker actors for CPU-bound / thread-unsafe "
                    "reward functions (e.g. deepscaler, math_verify). Each worker runs "
                    "in a separate process to avoid blocking the async event loop and "
                    "to isolate thread-unsafe libraries. Default: 16."
                ),
            )
            return parser

        def add_genrm_arguments(parser):
            """Add Generative Reward Model (genRM) arguments. Engine init and
            sampling parameters are consolidated into two JSON dict arguments,
            following the same pattern as --ref-actor-config.

            --genrm-engine-config keys (with defaults):
              model_path (str, required) — genRM model path; presence enables genRM
              num_gpus (int, 1) — total number of GPUs for genRM
              num_gpus_per_engine (int, 1) — GPUs per genRM engine instance
              max_context_len (int, 8192) — maximum context length
            --genrm-sampling-config keys (with defaults):
              temperature (float, 0.1) — sampling temperature
              top_p (float, 1.0) — nucleus sampling probability
              top_k (int, -1) — top-k sampling (-1 disables)
              max_response_len (int, 4096) — maximum response length
            """
            parser.add_argument(
                "--genrm-model-path",
                type=str,
                default=None,
                help="genRM model path. If set to None, genRM will not be enabled.",
            )
            parser.add_argument(
                "--genrm-num-gpus",
                type=int,
                default=1,
                help="Total number of GPUs for genRM.",
            )
            parser.add_argument(
                "--genrm-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per genRM engine instance.",
            )
            parser.add_argument(
                "--genrm-engine-config",
                type=json.loads,
                default=None,
                help=(
                    "JSON dict for genRM engine initialisation. "
                    "Setting this enables genRM. Example: "
                    '{ "dp_size": 1, "pp_size": 1, "max_total_tokens": 8192}. '
                    'When sharing GPUs with rollout, set "mem_fraction_static" here '
                    "to control genRM's per-GPU memory share independently from rollout."
                ),
            )
            parser.add_argument(
                "--genrm-sampling-config",
                type=json.loads,
                default=None,
                help=(
                    "JSON dict for genRM sampling parameters. "
                    "Keys: temperature (float, default 0.1), top_p (float, default 1.0), "
                    "top_k (int, default -1), max_response_len (int, default 4096). "
                    'Example: \'{ "temperature": 0.2, "max_response_len": 2048 }\''
                ),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "distill_qwen"],
                help="Loss mask type",
            )
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier for data padding size in data processing.",
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function should take args and samples (list[Sample]) as input, and return None. "
                    "Please directly modify the remove_sample attribute of Sample. "
                    "Note: This attribute does not determine whether the sample participates in advantage normalization."
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            parser.add_argument(
                "--disable-rollout-trim-samples",
                action="store_true",
                default=False,
                help="disable trim samples in rollout buffer when converting samples to train data",
            )
            parser.add_argument(
                "--use-dynamic-global-batch-size",
                action="store_true",
                default=False,
                help="enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data",
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """Add custom Megatron plugins arguments.

            This is a placeholder for any additional arguments that might be
            needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser

        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            reset_arg(parser, "--mtp-num-layers", type=int, default=None)
            reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training",
            )
            parser.add_argument(
                "--mtp-detach-paths",
                nargs="+",
                choices=(*_MTP_DETACH_PATHS, "none"),
                default=_MTP_DETACH_PATHS,
                help=(
                    "MTP auxiliary-loss gradient paths to detach. Choose any combination of embedding, "
                    "backbone, and lm-head, or use none by itself for fully joint gradients. "
                    "Defaults to detaching all three paths."
                ),
            )
            parser.add_argument(
                "--mtp-only-training",
                action="store_true",
                default=False,
                help=(
                    "Train only MTP parameters on SFT data. This enables MTP training, freezes every "
                    "non-MTP parameter before DDP/optimizer construction, and skips the main language-model loss. "
                    "Defaults --mtp-num-layers to 1 when it is not specified."
                ),
            )

            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            parser.add_argument(
                "--ci-disable-kl-checker",
                action="store_true",
            )
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            return parser

        def add_autoscaler_arguments(parser):
            """Add autoscaler specific arguments for dynamic engine scaling."""
            parser.add_argument(
                "--autoscaler-config",
                type=str,
                default=None,
                help=(
                    "Path to autoscaler YAML configuration file. "
                    "If provided, autoscaler will be enabled with settings from the config file. "
                    "If not provided (None), autoscaler is disabled. "
                    "Example: --autoscaler-config relax/utils/autoscaler/autoscaler.yaml"
                ),
            )
            parser.add_argument(
                "--enable-affinity",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "Whether baseline roles (actor / rollout-seed / colocate shared PG) require the "
                    "'stable' worker-group node-group affinity markers, keeping elastic "
                    "(autoscaler-enabled) jobs' baseline off elastic workers (which get reclaimed -> "
                    "whole-job restart). Defaults to True. This flag is forwarded to "
                    "create_placement_group(node_group_affinity=...); passing --no-enable-affinity "
                    "opts every baseline placement group out of the marker requirement (escape valve) "
                    "-- use it on plain/local clusters that don't declare the "
                    "'{group}_gpu'/'{group}_cpu' custom resources so placement stays unconstrained "
                    "and never hangs waiting for the markers. (Whether the job is elastic at all is "
                    "gated separately by --autoscaler-config + its YAML 'enabled'.)"
                ),
            )
            return parser

        # Add custom arguments in front to prevent overwritten some slime arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)
        parser = add_serve_arguments(parser)
        parser = add_transfer_queue_arguments(parser)
        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_on_policy_distillation_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_clearml_arguments(parser)
        parser = add_apprise_arguments(parser)
        parser = add_metrics_service_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_genrm_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_autoscaler_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        parser.add_argument(
            "--normalize-bbox",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Convert model-output bbox coordinates from normalized [0, 1000] to absolute pixels. "
                "Required for Qwen-VL/Qwen2-VL/Qwen3-VL (default True). "
                "Set --no-normalize-bbox for Qwen2.5-VL which outputs absolute pixel coordinates."
            ),
        )
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        return parser

    return add_slime_arguments


def _pre_parse_mode():
    """Pre-parse CLI to extract arguments that control parsing flow.

    These arguments are removed from add_slime_arguments to avoid registering
    them twice.  The returned namespace is merged into the final ``args`` after
    Phase 2 parsing.
    """
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    temp_parser.add_argument("--train-backend", type=str, choices=["megatron"], default="megatron")
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    temp_parser.add_argument("--skip-hf-validate", action="store_true", default=False)
    temp_args, _ = temp_parser.parse_known_args()
    return temp_args


def _pre_parse_cli_model_source():
    """Build a serializable model source from the generic CLI."""
    from relax.utils.model_source import ModelSource
    from relax.utils.s3_model_loader import is_s3_uri

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--hf-checkpoint", help="Hugging Face checkpoint path or model URI")
    parser.add_argument(
        "--disable-s3-model-download",
        action="store_true",
        help="Disable Relax-managed S3 model loading, including registered model-source providers",
    )
    pre, _ = parser.parse_known_args()
    if pre.disable_s3_model_download or not is_s3_uri(pre.hf_checkpoint):
        return None
    return ModelSource(uri=pre.hf_checkpoint)


def parse_args(add_custom_arguments=None):
    """Parse Relax arguments with an optional registered model source."""
    from relax.utils.model_source import resolve_model_source

    cli_source = _pre_parse_cli_model_source()
    provider_source = None if _s3_model_download_disabled() else resolve_model_source(sys.argv)
    return _parse_args_impl(add_custom_arguments, model_source=provider_source or cli_source)


def _reject_removed_mtp_detach_flags(argv: list[str]) -> None:
    for token in argv:
        option = token.split("=", 1)[0]
        if option in _REMOVED_MTP_DETACH_FLAGS:
            raise ValueError(f"{option} has been removed; use --mtp-detach-paths instead.")


def _s3_model_download_disabled() -> bool:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--disable-s3-model-download", action="store_true")
    pre, _ = parser.parse_known_args()
    return pre.disable_s3_model_download


def _parse_args_impl(add_custom_arguments=None, *, model_source=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    from relax.utils.s3_model_loader import is_s3_uri

    _reject_removed_mtp_detach_flags(sys.argv[1:])

    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    # Phase 1: Parse sglang args independently (separate parser, parse_known_args).
    # Skipped when sglang servers are not needed.
    sglang_ns = None
    teacher_sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()
        teacher_sglang_ns = teacher_sglang_parse_args()

    # Phase 2: Parse megatron + slime args.
    # Uses ignore_unknown_args=True so that --sglang-* and pre-parsed CLI flags
    # are silently ignored by the megatron parser.
    from relax.backends.megatron.arguments import megatron_parse_args
    from relax.backends.megatron.arguments import validate_args as megatron_validate_args

    args = megatron_parse_args(
        extra_args_provider=add_slime_arguments,
        model_source=model_source,
        skip_hf_validate=(
            pre.debug_rollout_only
            or pre.skip_hf_validate
            or (model_source is not None and is_s3_uri(model_source.uri))
        ),
    )

    # Merge pre-parsed args into the main namespace
    for key, value in vars(pre).items():
        setattr(args, key, value)

    # Merge sglang args into the main namespace
    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)

    if teacher_sglang_ns is not None:
        for key, value in vars(teacher_sglang_ns).items():
            setattr(args, key, value)

    # Preserve whether the user explicitly requested a rollout start before
    # validation derives 0 for an apparent HF/cold-start load. Actor init uses
    # this provenance to recover from a checkpoint-readiness race without
    # overriding an intentional --start-rollout-id.
    args._start_rollout_id_explicit = args.start_rollout_id is not None

    # Serialize the driver-derived descriptor with args to every Ray actor.
    args.model_source = model_source

    slime_validate_args(args)

    if not args.debug_rollout_only:
        args = megatron_validate_args(args)

    if not args.debug_train_only:
        sglang_validate_args(args)

    # Only fully-async mode relies on the newer TransferQueue (e.g.
    # StreamingTokenBudgetSampler), so gate the version requirement on it.
    if getattr(args, "fully_async", False):
        check_transfer_queue_version()

    return args


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """Build evaluation dataset configurations from either --eval-config or.

    --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


_MTP_ONLY_PARAM_PATTERN = r"(^|\.)mtp(\.|$)"


def _normalize_mtp_detach_paths(args) -> None:
    """Validate and canonicalize the MTP auxiliary-loss detach paths."""
    requested_paths = tuple(getattr(args, "mtp_detach_paths", _MTP_DETACH_PATHS))
    if "none" in requested_paths:
        if requested_paths != ("none",):
            raise ValueError("--mtp-detach-paths none cannot be combined with other paths.")
        args.mtp_detach_paths = ()
        return

    unknown_paths = sorted(set(requested_paths) - set(_MTP_DETACH_PATHS))
    if unknown_paths:
        raise ValueError(f"Unknown --mtp-detach-paths values: {', '.join(unknown_paths)}.")
    args.mtp_detach_paths = tuple(path for path in _MTP_DETACH_PATHS if path in requested_paths)


def _normalize_mtp_only_training_args(args) -> None:
    """Resolve the public MTP-only mode into existing training primitives."""
    if not getattr(args, "mtp_only_training", False):
        return

    if getattr(args, "loss_type", None) != "sft":
        raise ValueError("--mtp-only-training requires --loss-type sft.")

    conflicts = []
    if getattr(args, "only_train_params_name_list", None):
        conflicts.append("--only-train-params-name-list")
    if getattr(args, "freeze_params_name_list", None):
        conflicts.append("--freeze-params-name-list")
    if getattr(args, "lora_rank", 0) > 0:
        conflicts.append("--lora-rank")
    if getattr(args, "sft_chunked_logits", False):
        conflicts.append("--sft-chunked-logits")
    if getattr(args, "overlap_moe_expert_parallel_comm", False):
        conflicts.append("--overlap-moe-expert-parallel-comm")
    if getattr(args, "fully_async", False):
        conflicts.append("--fully-async")
    if getattr(args, "hybrid", False):
        conflicts.append("--hybrid")
    if tuple(getattr(args, "mtp_detach_paths", _MTP_DETACH_PATHS)) != _MTP_DETACH_PATHS:
        conflicts.append("--mtp-detach-paths")
    if conflicts:
        raise ValueError(f"--mtp-only-training is incompatible with: {', '.join(conflicts)}.")

    if args.mtp_num_layers is None:
        args.mtp_num_layers = 1
    if args.mtp_num_layers != 1:
        raise ValueError("--mtp-only-training currently supports exactly one Qwen3.5 MTP layer.")
    if args.mtp_loss_scaling_factor <= 0:
        raise ValueError("--mtp-loss-scaling-factor must be greater than 0 with --mtp-only-training.")

    args.enable_mtp_training = True
    args.mtp_detach_paths = _MTP_DETACH_PATHS
    args.only_train_params_name_list = [_MTP_ONLY_PARAM_PATTERN]


def _normalize_sft_max_in_flight_steps(args, is_sft: bool) -> None:
    sft_max_in_flight_steps = getattr(args, "sft_max_in_flight_steps", None)
    if sft_max_in_flight_steps is None:
        return

    if not is_sft:
        raise ValueError("--sft-max-in-flight-steps is only meaningful under --loss-type sft.")
    if sft_max_in_flight_steps < 1:
        raise ValueError("--sft-max-in-flight-steps must be >= 1.")
    args.max_staleness = sft_max_in_flight_steps - 1


def _normalize_sft_tq_timeout(args, is_sft: bool) -> None:
    if not is_sft:
        return
    timeout = getattr(args, "sft_tq_timeout_minutes", None)
    if timeout is None:
        args.sft_tq_timeout_minutes = args.distributed_timeout_minutes
    elif timeout <= 0:
        raise ValueError("--sft-tq-timeout-minutes must be > 0.")


def validate_save_hf_fp8_args(args) -> None:
    """Validate --save-hf-dtype and its FP8 sub-options; fill block-size
    default."""
    dtype = getattr(args, "save_hf_dtype", "bf16")
    if dtype != "fp8":
        return
    if not getattr(args, "save_hf", None):
        raise ValueError("--save-hf-dtype fp8 requires --save-hf to be set.")

    from relax.utils.quant_cast.fp8 import validate_fp8_options

    strategy = getattr(args, "save_hf_fp8_quant_mode", "block")
    block_size = getattr(args, "save_hf_fp8_block_size", None)
    if strategy == "block" and block_size is None:
        block_size = [128, 128]
        args.save_hf_fp8_block_size = block_size
    validate_fp8_options(strategy, block_size)


def validate_save_hf_post_hook_args(args) -> None:
    """Validate --save-hf-post-hook-path resolves and is paired with --save-
    hf."""
    hook_path = getattr(args, "save_hf_post_hook_path", None)
    if not hook_path:
        return
    if not getattr(args, "save_hf", None):
        raise ValueError("--save-hf-post-hook-path requires --save-hf to be set.")

    from relax.utils.misc import load_function

    hook = load_function(hook_path)
    if not callable(hook):
        raise TypeError(f"--save-hf-post-hook-path {hook_path!r} is not callable: got {type(hook).__name__}")


def _validate_agentic_rollout_args(args) -> None:
    if not args.use_agentic_rollout:
        return
    args.rollout_function_path = "relax.agentic.rollout.generate_rollout"
    args.eval_function_path = "relax.agentic.rollout.generate_rollout"
    args.apply_chat_template = False
    if not isinstance(args.agent_command, str) or not args.agent_command.strip():
        raise ValueError("--agent-command is required when --use-agentic-rollout is set.")
    if not isinstance(args.agent_cwd, str) or not args.agent_cwd.strip():
        raise ValueError("--agent-cwd is required when --use-agentic-rollout is set.")
    if not os.path.isdir(os.path.expanduser(args.agent_cwd)):
        raise ValueError(f"--agent-cwd must point to an existing directory, got {args.agent_cwd!r}.")
    if args.agent_timeout <= 0:
        raise ValueError("--agent-timeout must be > 0.")
    if not isinstance(args.agent_env, list) or not all(isinstance(item, str) for item in args.agent_env):
        raise TypeError("--agent-env must be provided as a list of KEY=VALUE strings.")
    for item in args.agent_env:
        if "=" not in item:
            raise ValueError(f"--agent-env entry must be KEY=VALUE, got {item!r}.")
        key = item.split("=", 1)[0].strip()
        if not key:
            raise ValueError(f"--agent-env entry must include a non-empty key, got {item!r}.")
        if key.startswith("RELAX_"):
            raise ValueError(f"--agent-env does not allow reserved key {key!r}.")
    if args.agentic_concurrency is not None and args.agentic_concurrency <= 0:
        raise ValueError("--agentic-concurrency must be > 0.")
    if args.agentic_eval_concurrency is not None and args.agentic_eval_concurrency <= 0:
        raise ValueError("--agentic-eval-concurrency must be > 0.")
    if args.agentic_program_admission:
        if args.agentic_admission_expected_decode_cap is None:
            args.agentic_admission_expected_decode_cap = args.rollout_max_response_len
        elif args.agentic_admission_expected_decode_cap <= 0:
            raise ValueError("--agentic-admission-expected-decode-cap must be > 0.")
        if not 0.0 < args.agentic_admission_headroom <= 1.0:
            raise ValueError("--agentic-admission-headroom must be in (0, 1].")
        if not 0.0 < args.agentic_admission_pressure_threshold <= 1.0:
            raise ValueError("--agentic-admission-pressure-threshold must be in (0, 1].")
        if args.agentic_admission_max_wait_s < 0:
            raise ValueError("--agentic-admission-max-wait-s must be >= 0.")
    if args.agentic_session_lifecycle:
        if not args.sglang_enable_session_radix_cache:
            raise ValueError("--agentic-session-lifecycle requires --sglang-enable-session-radix-cache.")
        if args.sglang_radix_eviction_policy != "priority":
            raise ValueError("--agentic-session-lifecycle requires --sglang-radix-eviction-policy priority.")


def _validate_reinforce_plus_plus_args(args, is_sft: bool) -> None:
    """Validate the frozen Task 29 REINFORCE++ algorithm contracts."""
    if is_sft:
        return

    estimator = getattr(args, "advantage_estimator", None)
    variants = {"reinforce_plus_plus", "reinforce_plus_plus_baseline"}
    if estimator not in variants:
        return

    if not getattr(args, "normalize_advantages", False):
        raise ValueError(f"--advantage-estimator {estimator} requires --normalize-advantages.")
    if getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
        raise ValueError(
            f"--advantage-estimator {estimator} supports synchronous colocate training only; "
            "--fully-async and --hybrid are not supported because normalization requires a closed global batch."
        )
    if not getattr(args, "colocate", False):
        raise ValueError(f"--advantage-estimator {estimator} currently requires --colocate.")
    if getattr(args, "context_parallel_size", 1) != 1:
        raise ValueError(f"--advantage-estimator {estimator} currently requires --context-parallel-size 1.")
    if getattr(args, "calculate_per_token_loss", False):
        raise ValueError(
            f"--advantage-estimator {estimator} uses response-mean loss reduction; "
            "--calculate-per-token-loss is not supported."
        )

    kl_coef = getattr(args, "kl_coef", 0.0)
    kl_loss_coef = getattr(args, "kl_loss_coef", 0.0)
    kl_loss_type = getattr(args, "kl_loss_type", "k1")
    use_kl_loss = getattr(args, "use_kl_loss", False)

    if estimator == "reinforce_plus_plus":
        if kl_coef <= 0:
            raise ValueError("reinforce_plus_plus requires --kl-coef > 0 for token-level KL reward shaping.")
        if kl_loss_type != "k1":
            raise ValueError("reinforce_plus_plus requires --kl-loss-type k1 for token-level KL reward shaping.")
        if use_kl_loss or kl_loss_coef != 0:
            raise ValueError(
                "reinforce_plus_plus uses KL reward shaping and does not support a separate --use-kl-loss."
            )
        return

    if getattr(args, "n_samples_per_prompt", 1) <= 1:
        raise ValueError("reinforce_plus_plus_baseline requires --n-samples-per-prompt > 1.")
    if getattr(args, "custom_reward_post_process_path", None) is not None:
        raise ValueError(
            "reinforce_plus_plus_baseline freezes inclusive group-mean centering; "
            "--custom-reward-post-process-path is not supported."
        )
    if getattr(args, "agentic_custom_advantage_path", None) is not None:
        raise ValueError(
            "reinforce_plus_plus_baseline freezes inclusive group-mean centering; "
            "--agentic-custom-advantage-path is not supported."
        )
    if not getattr(args, "rewards_normalization", True):
        raise ValueError(
            "reinforce_plus_plus_baseline requires group-mean centering; "
            "--disable-rewards-normalization is not supported."
        )
    if kl_coef != 0:
        raise ValueError("reinforce_plus_plus_baseline does not put token KL in the advantage; set --kl-coef 0.")
    if not use_kl_loss or kl_loss_coef <= 0:
        raise ValueError(
            "reinforce_plus_plus_baseline requires an independent k2 penalty via --use-kl-loss and --kl-loss-coef > 0."
        )
    if kl_loss_type != "k2":
        raise ValueError("reinforce_plus_plus_baseline requires --kl-loss-type k2.")
    if getattr(args, "use_unbiased_kl", False):
        raise ValueError(
            "reinforce_plus_plus_baseline uses the plain k2 estimator; --use-unbiased-kl is not supported."
        )


def _declares_non_language_encoder(hf_checkpoint: str) -> bool | None:
    """Whether the HF config declares a vision / audio encoder next to the LM.

    Tri-state on purpose: ``None`` means the config could not be read (path not
    materialized yet, or custom code that needs ``trust_remote_code``). Callers
    degrade to a warning in that case — a best-effort guard must not become a
    new way for a valid run to fail.
    """
    from relax.utils.misc import get_hf_config

    try:
        config = get_hf_config(hf_checkpoint)
    except Exception as e:  # noqa: BLE001 - best-effort probe, never fatal
        logger.warning("Could not read HF config at %s to check for a vision tower: %s", hf_checkpoint, e)
        return None
    return any(getattr(config, key, None) is not None for key in ("vision_config", "audio_config"))


def _validate_lora_vision_scope(args) -> None:
    """Reject LoRA adapter mode that would train a vision-tower adapter.

    SGLang hosts LoRA on language-model layers only: ``should_apply_lora`` accepts
    ``model.layers.*`` and nothing else, and its adapter loader bins tensors by a
    ``layers.<N>.`` regex that a vision name (``model.visual.blocks.<N>....``) never
    matches. A vision adapter therefore trains but is dropped on the way to the
    rollout engine WITHOUT any log line — the actor optimizes a policy the rollout
    never runs, and the gap widens every step.

    Only adapter mode is affected. Merge mode folds the adapter into the base weights
    and syncs those, so the engine never sees LoRA at all and any scope is safe.
    """
    if not getattr(args, "lora_adapter_mode", False) or getattr(args, "lora_scope", "all") != "all":
        return

    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        return

    has_encoder = _declares_non_language_encoder(hf_checkpoint)
    if has_encoder is None:
        logger.warning(
            "Skipped the --lora-scope check for --lora-adapter-mode (HF config unreadable). "
            "If %s is a VL/omni model, pass --lora-scope language: SGLang cannot host a "
            "vision-tower adapter and would silently drop it.",
            hf_checkpoint,
        )
        return
    if not has_encoder:
        return

    raise ValueError(
        "--lora-adapter-mode with --lora-scope all is not supported on a model that has a "
        "vision/audio encoder: SGLang only hosts LoRA on language-model layers, so the "
        "vision-tower adapter would be trained but silently dropped before rollout, breaking "
        "the on-policy assumption. Either pass --lora-scope language (keeps adapter mode and "
        "its per-step bandwidth saving; the vision tower is left un-adapted), or switch to "
        "--lora-merge-mode (folds the vision adapter into the synced base weights, at the cost "
        "of a full weight sync every step). Pass --lora-scope language explicitly if your "
        "--lora-target-modules provably match nothing in the vision tower."
    )


def _normalize_sync_ppo_kl_args(args) -> bool:
    """Disable KL options that have no ref-logprob producer in sync PPO."""
    is_sync_ppo = (
        getattr(args, "use_critic", False)
        and not getattr(args, "fully_async", False)
        and not getattr(args, "hybrid", False)
    )
    if not is_sync_ppo or (not getattr(args, "use_kl_loss", False) and getattr(args, "kl_coef", 0.0) == 0):
        return False

    args.use_kl_loss = False
    args.kl_coef = 0.0
    return True


def _validate_ref_load(args: argparse.Namespace) -> None:
    """Validate a local reference checkpoint or defer a source alias."""
    if is_model_source_alias(args, args.ref_load):
        return
    if is_model_uri(args.ref_load):
        raise ValueError(
            f"ref_load URI {args.ref_load!r} does not match the configured model source; "
            "only a ref_load alias of hf_checkpoint or a local path is supported."
        )
    if not os.path.exists(args.ref_load):
        raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

    if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
        logger.info(
            f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
            "please make sure it is a valid megatron checkpoint directory."
        )


def slime_validate_args(args):
    # Backward compatibility: old scripts may pass --enable-gloo-process-groups
    if not hasattr(args, "use_gloo_process_groups"):
        args.use_gloo_process_groups = getattr(args, "enable_gloo_process_groups", False)

    is_sft = args.loss_type in ("sft", "sft_loss", "sft-loss")
    if is_sft:
        # Force-disable RL-only state so SFT users don't have to pass
        # `--disable-compute-advantages-and-returns` and friends.
        args.compute_advantages_and_returns = False
        args.use_kl_loss = False
        args.kl_coef = 0.0
        args.kl_loss_coef = 0.0
        args.use_opd = False
        # SFT owns --prompt-data through components/sft.py. If predict is
        # enabled, the injected Rollout role should not also build an RL global
        # dataset from it.
        args.rollout_global_dataset = False

    if is_sft:
        if args.eval_config is not None:
            raise ValueError("--loss-type sft uses --eval-prompt-data for eval; --eval-config is not supported.")
        if args.eval_prompt_data and len(args.eval_prompt_data) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
        eval_prompt_data = build_named_prompt_data_configs(args.eval_prompt_data)
        args.eval_prompt_data = (
            [item for dataset in eval_prompt_data for item in (dataset.name, dataset.path)]
            if eval_prompt_data
            else None
        )
        if hasattr(args, "eval_datasets"):
            delattr(args, "eval_datasets")
    else:
        args.eval_datasets = _resolve_eval_datasets(args)

    if args.max_staleness < 0:
        raise ValueError("--max-staleness must be >= 0.")

    if getattr(args, "lora_rank", 0) > 0:
        if getattr(args, "lora_merge_mode", False) and getattr(args, "lora_adapter_mode", False):
            raise ValueError(
                "--lora-merge-mode and --lora-adapter-mode are mutually exclusive; pick one LoRA rollout path."
            )
        if getattr(args, "lora_adapter_mode", False) and getattr(args, "sglang_dp_size", 1) != 1:
            raise ValueError(
                "--lora-adapter-mode requires --sglang-dp-size 1 (SGLang dynamic LoRA loading does not "
                "support dp_size > 1)."
            )
        _validate_lora_vision_scope(args)
        if "router" in getattr(args, "lora_target_modules", []) and not getattr(args, "lora_adapter_mode", False):
            # Merge folds each adapter into its base via LoRAMerge, which only transforms
            # LoRALinear modules; the router uses LoRATopKRouter, so its adapter would be
            # silently dropped at sync time. Fail loud instead of training a router LoRA that
            # never reaches the rollout engine. (Adapter mode exports via the bridge, which
            # does handle the router, so it is not blocked here.)
            raise ValueError(
                "LoRA on 'router' is only supported in --lora-adapter-mode; merge mode cannot fold "
                "the router adapter (LoRAMerge skips non-LoRALinear modules). Remove 'router' from "
                "--lora-target-modules or switch to --lora-adapter-mode."
            )
        if not getattr(args, "lora_merge_mode", False) and not getattr(args, "lora_adapter_mode", False):
            logger.info(
                "LoRA enabled (lora_rank=%d): forcing --lora-merge-mode (default supported LoRA rollout path).",
                args.lora_rank,
            )
            args.lora_merge_mode = True

    # Refuse SGLANG_ENABLE_SPEC_V2=1 with speculative decoding on SGLang <= 0.5.9.
    # There, spec_v2 routes requests through EAGLEWorkerV2.verify(), which does
    # not populate output_token_logprobs — rollout sees response_length=1 for
    # every sample and training silently degenerates. Fixed after 0.5.9, so
    # newer builds may combine the two freely.
    if getattr(args, "sglang_speculative_algorithm", None) and Envs.SGLANG_ENABLE_SPEC_V2.lower() in (
        "1",
        "true",
        "yes",
        "y",
    ):
        import sglang
        from packaging.version import parse

        if parse(sglang.__version__) <= parse("0.5.9"):
            raise ValueError(
                f"SGLANG_ENABLE_SPEC_V2=1 is not supported together with "
                f"--sglang-speculative-algorithm on sglang {sglang.__version__}: the spec_v2 "
                f"EAGLE worker does not populate output_token_logprobs, which collapses "
                f"rollout response_length to 1 and silently breaks training. Unset "
                f"SGLANG_ENABLE_SPEC_V2 (or set it to 0) to fall back to the spec_v1 EAGLE "
                f"worker, or upgrade past 0.5.9. For Qwen3.5-MoE-style hybrid models, keep "
                f"--sglang-mamba-scheduler-strategy extra_buffer — that flag alone satisfies "
                f"SGLang's mamba radix-cache check and does NOT auto-enable spec_v2."
            )

    _normalize_sft_max_in_flight_steps(args, is_sft)
    _normalize_sft_tq_timeout(args, is_sft)
    _validate_agentic_rollout_args(args)
    validate_save_hf_fp8_args(args)
    validate_save_hf_post_hook_args(args)

    if not is_sft and args.partial_rollout and args.use_rollout_routing_replay:
        raise ValueError(
            "The options 'partial_rollout' and 'use_rollout_routing_replay' cannot be enabled simultaneously. "
            "'use_rollout_routing_replay' addresses mismatch problem between training and inference, "
            "whereas 'partial_rollout' introduces partial off-policy behavior. These two features are mutually exclusive."
        )

    if not is_sft and args.advantage_estimator == "rloo" and args.kl_coef != 0:
        raise ValueError(
            "--advantage-estimator rloo does not support nonzero --kl-coef: reward-side KL shaping "
            "is not implemented for the completion-level leave-one-out signal. Set --kl-coef 0; "
            "for a supported direct KL penalty, provide --ref-load and use --use-kl-loss with "
            "--kl-loss-coef."
        )

    if not is_sft and (args.kl_coef != 0 or args.use_kl_loss):
        _validate_ref_load(args)

    validate_opd_args(args, is_sft=is_sft, log=logger)

    if args.megatron_to_hf_mode == "bridge":
        if (
            args.load is not None
            and os.path.exists(args.load)
            and os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            # If is a Megatron checkpoint, won't use bridge to load hf weight.
            pass
        else:
            if args.load is None:
                args.load = args.ref_load or args.hf_checkpoint
            # If is a HF checkpoint, set start_rollout_id to 0 here.
            args.start_rollout_id = 0
    else:
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            args.start_rollout_id = 0

    if args.eval_interval is not None:
        if args.loss_type == "sft":
            has_eval_source = bool(args.eval_prompt_data) or (args.eval_size is not None)
            if has_eval_source:
                assert bool(args.eval_prompt_data) ^ (args.eval_size is not None), (
                    "Under --loss-type sft with --eval-interval set, at most one of "
                    "--eval-prompt-data or --eval-size may be configured."
                )
            elif not getattr(args, "custom_dataset_class_path", None):
                raise ValueError(
                    "Under --loss-type sft with --eval-interval set, exactly one of "
                    "--eval-prompt-data or --eval-size must be configured."
                )
        else:
            assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.eval_size is not None:
        assert args.loss_type == "sft", "--eval-size is only meaningful under --loss-type sft."
        assert args.eval_size > 0, "--eval-size must be positive."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    if getattr(args, "sft_predict_interval", None) is not None:
        assert args.loss_type == "sft", "--sft-predict-interval is only meaningful under --loss-type sft."
        assert args.sft_predict_interval > 0, "--sft-predict-interval must be positive."
        assert args.save is not None, "--sft-predict-interval requires --save (predictions land in <save>/predict/)."
        has_eval_source = bool(getattr(args, "eval_prompt_data", None)) or (
            getattr(args, "eval_size", None) is not None
        )
        assert has_eval_source, (
            "--sft-predict-interval requires either --eval-prompt-data or --eval-size "
            "(the predict pass reuses the SFT eval data source)."
        )

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    _validate_reinforce_plus_plus_args(args, is_sft)

    if args.rollout_batch_size is None:
        if args.global_batch_size is None:
            raise ValueError("Either --rollout-batch-size or --global-batch-size must be set.")
        if args.n_samples_per_prompt <= 0:
            raise ValueError("--n-samples-per-prompt must be positive when deriving --rollout-batch-size.")
        if args.advantage_estimator == "rloo" and args.global_batch_size % args.n_samples_per_prompt != 0:
            raise ValueError(
                "--global-batch-size must be divisible by --n-samples-per-prompt for RLOO when "
                "--rollout-batch-size is omitted, got "
                f"{args.global_batch_size} % {args.n_samples_per_prompt} != 0."
            )
        args.rollout_batch_size = args.global_batch_size // args.n_samples_per_prompt
        logger.info(
            f"--rollout-batch-size not set; derived as global_batch_size ({args.global_batch_size}) "
            f"// n_samples_per_prompt ({args.n_samples_per_prompt}) = {args.rollout_batch_size}"
        )

    if not is_sft:
        if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
            assert args.normalize_advantages, (
                "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
                "require advantage normalization. Please add `--normalize-advantages` to your command."
            )
        if args.fully_async:
            assert not args.normalize_advantages, (
                "Advantage normalization is not supported in fully-async mode (--fully-async). "
                "Please remove --normalize-advantages from your command."
            )
            assert not args.opd_type == "megatron", (
                "On-policy distillation with megatron teacher is not supported in fully-async mode (--fully-async)."
                " Please set --opd-type to sglang or remove --use-opd."
            )
            assert not args.use_dynamic_global_batch_size, (
                "--use-dynamic-global-batch-size is only supported in colocate mode. "
                "fully-async training does not support dynamic global batch size yet."
            )

        # Auto-enable true_on_policy_mode when the per-step rollout output exactly fills
        # one global batch in fully-async mode. In this regime the train forward's
        # log_probs equal what actor_fwd would have produced, so the actor_fwd role
        # can be skipped (see relax/backends/megatron/loss.py:policy_loss_function).
        if args.fully_async and args.rollout_batch_size * args.n_samples_per_prompt == args.global_batch_size:
            if not args.true_on_policy_mode:
                logger.info(
                    "Auto-enabling --true-on-policy-mode: rollout_batch_size * n_samples_per_prompt "
                    f"== global_batch_size ({args.global_batch_size}). actor_fwd will be skipped."
                )
            args.true_on_policy_mode = True

        # Validate --resource has the producer roles the trainer will fetch from
        # TransferQueue in fully-async mode. Without these, train_async would poll
        # forever for a field nobody writes (see backends/megatron/actor.py:train_async).
        if args.fully_async and args.resource is not None:
            if (args.use_kl_loss or args.kl_coef != 0) and "reference" not in args.resource:
                raise ValueError(
                    "--use-kl-loss / --kl-coef != 0 requires a 'reference' entry in --resource "
                    "(produces ref_log_probs via TransferQueue in fully-async mode). "
                    f"Current --resource keys: {sorted(args.resource.keys())}."
                )
            if not args.true_on_policy_mode and "actor_fwd" not in args.resource:
                raise ValueError(
                    "actor_fwd is required in --resource when true_on_policy_mode is False "
                    "(produces log_probs via TransferQueue in fully-async mode). "
                    "true_on_policy_mode is auto-enabled only when "
                    "rollout_batch_size * n_samples_per_prompt == global_batch_size. "
                    f"Current --resource keys: {sorted(args.resource.keys())}."
                )

        if args.use_rollout_logprobs:
            assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

        if args.get_mismatch_metrics:
            assert args.custom_tis_function_path is not None, (
                "custom_tis_function_path must be set when get_mismatch_metrics is set"
            )

            if args.use_rollout_logprobs:
                logger.info(
                    "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
                )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

        # The token-budget sampler always emits at least one sample per micro-batch,
        # even if that single sample exceeds the budget (otherwise the stream stalls).
        # So the per-GPU token budget (max_tokens_per_gpu * context_parallel_size,
        # since a sequence is split across CP ranks) must be able to hold the longest
        # possible single sample (rollout_max_context_len). Otherwise an over-long
        # sample produces an oversized micro-batch that OOMs mid-training.
        max_ctx_len = getattr(args, "rollout_max_context_len", None)
        if max_ctx_len is not None:
            cp_size = getattr(args, "context_parallel_size", 1)
            token_budget = args.max_tokens_per_gpu * cp_size
            if token_budget < max_ctx_len and not getattr(args, "dynamic_context_parallel", False):
                raise ValueError(
                    f"max_tokens_per_gpu * context_parallel_size ({args.max_tokens_per_gpu} * {cp_size} = "
                    f"{token_budget}) must be >= rollout_max_context_len ({max_ctx_len}); otherwise a single "
                    f"over-long sample forms an oversized micro-batch and OOMs. Increase max_tokens_per_gpu "
                    f"(or context_parallel_size), or reduce rollout_max_context_len."
                )

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    rm_type_fallback = getattr(args, "rm_type_fallback", None)
    if rm_type_fallback is not None and rm_type_fallback != "zero":
        # Lazy import: only pay for the rewards package when the flag is set.
        from relax.engine.rewards.registry import list_reward_types

        if rm_type_fallback not in list_reward_types():
            raise ValueError(
                f"--rm-type-fallback {rm_type_fallback!r} is not a registered reward type. "
                f"Use 'zero' or one of: {list_reward_types()}"
            )

    if hasattr(args, "rollout_result_dir"):
        if args.rollout_result_dir is None and getattr(args, "save", None):
            args.rollout_result_dir = f"{args.save}/rollout_result"
        elif args.rollout_result_dir == "":
            args.rollout_result_dir = None

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    if args.loss_type in ("sft_loss", "sft-loss"):
        warnings.warn(
            f"--loss-type {args.loss_type} is deprecated; use --loss-type sft instead. "
            "This alias will be removed in the next minor release.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.loss_type = "sft"

    if args.loss_type == "sft":
        if not args.custom_dataset_class_path and not args.prompt_data:
            raise ValueError("--loss-type sft requires --prompt-data.")
        if args.sft_oversize_strategy == "custom" and not args.sft_oversize_custom_function_path:
            raise ValueError("--sft-oversize-strategy custom requires --sft-oversize-custom-function-path.")
        # SFT does not use advantages / reference; force-disable to avoid wasted compute.
        args.compute_advantages_and_returns = False
        # SFT samples have highly variable length; only the dynamic-batch-size
        # path knows how to (a) cap per-GPU tokens (CP-aware) and (b) build
        # balanced micro-batches. Static --micro-batch-size cannot do either,
        # so we require --use-dynamic-batch-size.
        if not args.use_dynamic_batch_size:
            raise ValueError(
                "--loss-type sft requires --use-dynamic-batch-size (with --max-tokens-per-gpu). "
                "SFT relies on dynamic batching to bound per-GPU tokens (CP-aware) and to filter "
                "samples that cannot fit on a single GPU."
            )
        # The controller always installs SeqlenBalancedSampler for SFT (see
        # `core/controller.py:_initialize_data_system`). That sampler can hand
        # different sample counts to each DP rank, which the Megatron data
        # path only handles correctly when args.balance_data is True. Force it
        # on so the two layers stay consistent.
        if not args.balance_data:
            logger.info("--loss-type sft: auto-enabling --balance-data for DP-balanced batching.")
            args.balance_data = True

    task_type = getattr(args, "task_type", "causal_lm")
    if task_type == "seq_cls":
        if args.loss_type != "sft":
            raise ValueError("--task-type seq_cls requires --loss-type sft.")
        if not args.label_key:
            raise ValueError("--task-type seq_cls requires --label-key.")
        if args.num_labels is None or args.num_labels < 2:
            raise ValueError("--task-type seq_cls requires --num-labels >= 2.")
        if not 0.0 <= args.classification_threshold <= 1.0:
            raise ValueError("--classification-threshold must be in [0, 1].")
        incompatible = {
            "--fully-async": bool(args.fully_async),
            "--hybrid": bool(args.hybrid),
            "--sft-predict-interval": args.sft_predict_interval is not None,
            "--sft-chunked-logits": bool(args.sft_chunked_logits),
            "--enable-mtp-training": bool(args.enable_mtp_training),
            "--mtp-num-layers": bool(args.mtp_num_layers),
            "--save-hf": args.save_hf is not None,
            "--allgather-cp": bool(args.allgather_cp),
            "--custom-dataset-class": args.custom_dataset_class_path is not None,
            "--sft-oversize-strategy custom": args.sft_oversize_strategy == "custom",
            "--debug-train-only": bool(args.debug_train_only),
        }
        enabled = [name for name, is_enabled in incompatible.items() if is_enabled]
        if enabled:
            raise ValueError(f"--task-type seq_cls does not support: {', '.join(enabled)}.")
    elif getattr(args, "num_labels", None) is not None:
        raise ValueError("--num-labels is only meaningful under --task-type seq_cls.")

    args.use_critic = args.advantage_estimator == "ppo"
    # Synchronous PPO has no producer for
    # `ref_log_probs`: actor's ref forward in backends/megatron/actor.py:800 is
    # gated on `advantage_estimator != "ppo"`, and the sync role set does not
    # deploy a separate reference service. Either KL option would make
    # advantages / actor request a field that never arrives and hang in
    # TQ.get_meta, so disable both for the currently supported sync topology.
    if _normalize_sync_ppo_kl_args(args):
        logger.warning(
            "Synchronous PPO (--advantage-estimator ppo) does not support --use-kl-loss or "
            "--kl-coef != 0 because its service graph has no producer for ref_log_probs. "
            "Auto-disabling --use-kl-loss and resetting --kl-coef to 0.0. "
            "Drop these KL options from the launch script to silence this warning."
        )
    elif args.use_critic and args.use_kl_loss:
        # Preserve the existing behavior outside the synchronous topology.
        logger.warning("PPO does not support --use-kl-loss. Auto-disabling --use-kl-loss.")
        args.use_kl_loss = False
    if args.critic_num_gpus_per_node is None:
        args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    if args.critic_num_nodes is None:
        args.critic_num_nodes = args.actor_num_nodes
    if args.critic_load is None:
        args.critic_load = args.load
    if args.critic_lr is None:
        args.critic_lr = args.lr

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    # Resolve --hybrid into the underlying execution flags so downstream
    # machinery (StreamDataLoader broadcast_pp, UpdateWeightFromTensor
    # selection, sglang_engine DCS gating) keeps a single semantic axis.
    # `args.hybrid` remains the canonical switch for hybrid-specific
    # branches (registry, controller dispatch, train_hybrid call site).
    if args.hybrid:
        args.fully_async = True
        args.colocate = True
        logger.info(
            "hybrid mode: actor/reference/actor_fwd/advantages will share GPUs "
            "via offload/onload role switching, while rollout uses separate GPUs."
        )
    elif args.fully_async and args.colocate:
        raise ValueError(
            "--fully-async and --colocate cannot be combined directly. "
            "Use --hybrid instead, which is the supported public flag for hybrid training mode."
        )

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, please set only one of them."
    )

    # Check if genRM is enabled
    genrm_enabled = args.genrm_model_path is not None
    managed_opd_teacher_enabled = is_managed_opd_teacher_enabled(args)
    args._genrm_colocate_with_rollout = False

    # always true on offload for colocate at the moment.
    if args.hybrid:
        # hybrid mode: actor and rollout use SEPARATE GPUs,
        # so no offload needed between them. Actor internally handles
        # ref/actor_fwd via _switch_model (same model, weight swap only).
        if args.offload_train is None:
            args.offload_train = False
        if args.offload_rollout is None:
            args.offload_rollout = False
        # Mark that actor should compute advantages and ref/actor_fwd internally
        args.compute_advantages_and_returns = True
    elif args.colocate and managed_opd_teacher_enabled:
        validate_managed_opd_teacher_colocate_args(args)
    elif args.colocate and not genrm_enabled:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
    elif args.colocate and genrm_enabled:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        # When genRM is enabled, allow split GPU allocation
        # Check that rollout + genRM GPUs don't exceed actor GPUs
        if args.rollout_num_gpus is None:
            raise ValueError(
                "When genRM is enabled in colocated mode, --rollout-num-gpus must be explicitly set. "
                "For example: --rollout-num-gpus 4 --genrm-num-gpus 4 on an 8-GPU machine."
            )

        actor_total_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        if args.use_critic:
            actor_total_gpus += args.critic_num_gpus_per_node * args.critic_num_nodes

        rollout_g = args.rollout_num_gpus
        genrm_g = args.genrm_num_gpus
        if rollout_g + genrm_g == actor_total_gpus:
            args._genrm_colocate_with_rollout = False
            logger.info(
                f"GenRM colocate (split bundles): rollout={rollout_g}, genrm={genrm_g}, "
                f"actor total={actor_total_gpus}."
            )
        elif rollout_g == actor_total_gpus and genrm_g == actor_total_gpus:
            args._genrm_colocate_with_rollout = True
            logger.info(
                f"GenRM colocate (shared bundles with rollout): rollout=genrm={actor_total_gpus} GPUs. "
                f"Set per-engine SGLang mem_fraction_static via --sglang-config (rollout) and "
                f"--genrm-engine-config '{{\"mem_fraction_static\": <float>}}' (genrm)."
            )
        else:
            raise ValueError(
                "In colocated mode with genRM enabled, GPU allocation must satisfy one of:\n"
                f"  (1) split: --rollout-num-gpus + --genrm-num-gpus == actor total ({actor_total_gpus}), or\n"
                f"  (2) shared: --rollout-num-gpus == --genrm-num-gpus == actor total ({actor_total_gpus}).\n"
                f"Got rollout={rollout_g}, genrm={genrm_g}, actor total={actor_total_gpus}."
            )

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False
    maybe_enable_sglang_weights_cpu_backup(args)

    if args.use_critic:
        args.offload_train = True

    # expandable_segments cannot coexist with torch_memory_saver, the default mechanism
    # behind --offload-train. TMS's hook is armed from TMS_INIT_ENABLE inside the
    # LD_PRELOAD'ed .so before any Python runs, so neither its own sanity check nor any
    # in-process guard can intervene — the actor just dies with a bare
    # "CUresult error: 1 (invalid argument)" from cu_mem_create. Fail here instead, while
    # the message can still be read. Only the CUDA variable names are checked on purpose:
    # NPU uses a different torch_memory_saver build and its scripts already combine
    # PYTORCH_NPU_ALLOC_CONF=expandable_segments:True with offload successfully.
    if args.offload_train and not getattr(args, "selective_offload", False):
        alloc_conf_sources = {
            **{name: os.environ.get(name, "") for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
            **{key: str(value) for key, value in getattr(args, "train_env_vars", {}).items()},
        }
        for name, value in alloc_conf_sources.items():
            if "expandable_segments:True" in value:
                raise ValueError(
                    f"{name} enables expandable_segments, which torch_memory_saver cannot track. "
                    "Add --selective-offload to use the application-level offload instead, "
                    "or drop expandable_segments."
                )

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.advantage_estimator == "rloo" and args.num_steps_per_rollout not in (None, 1):
        raise ValueError(
            "--advantage-estimator rloo requires --num-steps-per-rollout 1 "
            "(the unclipped objective has no ratio correction, so repeated updates on the same "
            "rollout would be off-policy)."
        )

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.advantage_estimator == "rloo":
        if args.n_samples_per_prompt < 2:
            raise ValueError(
                "--advantage-estimator rloo requires --n-samples-per-prompt >= 2 "
                "(the leave-one-out baseline divides by G-1; G=1 is undefined)."
            )
        if args.fully_async or getattr(args, "hybrid", False):
            raise ValueError(
                "--advantage-estimator rloo only supports synchronous (colocate) training. "
                "Please remove --fully-async / --hybrid."
            )
        if not args.calculate_per_token_loss:
            raise ValueError(
                "--advantage-estimator rloo requires --calculate-per-token-loss so policy loss is "
                "normalized by the global number of valid response tokens. The per-sample token-mean "
                "reducer would reweight unequal-length responses by 1 / response_length."
            )
        if args.max_staleness != 0:
            raise ValueError(
                "--advantage-estimator rloo requires --max-staleness 0: the unclipped objective has no "
                "importance-ratio correction for stale rollout data."
            )
        if not args.rewards_normalization:
            raise ValueError(
                "--advantage-estimator rloo requires rewards normalization to be enabled "
                "(the leave-one-out baseline is computed inside the group-reward path that "
                "--disable-rewards-normalization skips). Please remove --disable-rewards-normalization."
            )
        if args.normalize_advantages:
            raise ValueError(
                "--advantage-estimator rloo is incompatible with --normalize-advantages: "
                "the latter re-whitens advantages after DP sharding "
                "(distributed_masked_whiten in loss.py), which re-introduces the std "
                "normalization RLOO removes and makes the result depend on the DP partition. "
                "Please remove --normalize-advantages."
            )
        if args.rollout_batch_size * args.n_samples_per_prompt != args.global_batch_size:
            raise ValueError(
                "--advantage-estimator rloo requires exactly one optimizer update per rollout "
                "(the unclipped objective has no ratio correction, so a second update on the "
                "same rollout is off-policy without correction). This means "
                "rollout_batch_size * n_samples_per_prompt must equal global_batch_size, "
                f"got {args.rollout_batch_size} * {args.n_samples_per_prompt} = "
                f"{args.rollout_batch_size * args.n_samples_per_prompt} != "
                f"{args.global_batch_size}."
            )
        if args.partial_rollout or args.use_dynamic_global_batch_size:
            raise ValueError(
                "--advantage-estimator rloo is incompatible with --partial-rollout / "
                "--use-dynamic-global-batch-size: they cause the effective batch size to drift "
                "at runtime, breaking the one-update-per-rollout guarantee."
            )

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    if args.use_agentic_rollout and args.agentic_concurrency is None:
        args.agentic_concurrency = args.over_sampling_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if (
        args.over_sampling_batch_size > args.rollout_batch_size
        and not getattr(args, "fully_async", False)
        and not getattr(args, "partial_rollout", False)
    ):
        # Without fully_async/partial_rollout there is no next-step carry path for the
        # over-sampled surplus, so the extra completed groups are discarded each step
        # rather than reused. Surface this so it is not mistaken for free throughput.
        logger.warning(
            "over_sampling_batch_size (%s) > rollout_batch_size (%s) without fully_async/partial_rollout; "
            "surplus over-sampled groups are discarded per-step instead of carried to the next step.",
            args.over_sampling_batch_size,
            args.rollout_batch_size,
        )

    if args.num_epoch is None:
        assert args.num_rollout is not None, "Neither --num-rollout nor --num-epoch is set; please set at least one."
    elif getattr(args, "loss_type", None) != "sft":
        assert args.rollout_global_dataset, (
            "num_epoch is set, but rollout_global_dataset is not set, "
            "please remove --disable-rollout-global-dataset to use num_epoch"
        )

    _normalize_mtp_detach_paths(args)
    _normalize_mtp_only_training_args(args)

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    # --sft-chunked-logits incompatibilities. Both are flagged here so
    # downstream (engine/sft/runtime.py should_use_sft_chunked + the three loss.py direct
    # reads of args.sft_chunked_logits) sees a single, consistent truth.
    # Both are hard asserts — the user must remove --sft-chunked-logits
    # from their script rather than have it silently flipped off.
    if getattr(args, "sft_chunked_logits", False):
        # 1) Tied-embedding (set automatically from HF config.tie_word_embeddings).
        #    Output_layer is built with skip_weight_param_allocation=True →
        #    output_layer.weight is None → chunked path's lm_head matmul
        #    crashes on NoneType. The chunked memory win is marginal on the
        #    small models that ship with tied embeddings anyway, so the user
        #    should just drop the flag.
        assert getattr(args, "untie_embeddings_and_output_weights", False), (
            "--sft-chunked-logits is incompatible with tied embeddings "
            "(HF config.tie_word_embeddings=true → "
            "--untie-embeddings-and-output-weights not set; output_layer.weight "
            "is None and the chunked path's lm_head matmul has nothing to "
            "multiply against). Remove --sft-chunked-logits; the chunked "
            "memory win is marginal on tied-weight models."
        )
        # 2) Combined 1F1B. overlap_moe_expert_parallel_comm routes training
        #    forward through model.build_schedule_plan(), which does NOT call
        #    model(**kwargs) and so never hits _bypass_output_layer — chunked
        #    silently degrades to the full-logits path.
        assert not getattr(args, "overlap_moe_expert_parallel_comm", False), (
            "--sft-chunked-logits is incompatible with "
            "--overlap-moe-expert-parallel-comm (combined-1f1b path bypasses "
            "_bypass_output_layer; chunked would silently degrade). "
            "Remove one of the two flags."
        )

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert args.rollout_max_prompt_len <= args.rollout_max_context_len - 1, (
            f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."
        )

    if args.qkv_format == "bshd":
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        assert args.use_dynamic_batch_size is False, (
            "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."
        )
    if args.only_train_params_name_list and args.freeze_params_name_list:
        raise ValueError("You can only specify ONE of: --only-train-params-name-list, or --freeze-params-name-list.")

    if args.rotate_ckpt:
        assert args.save is not None, "--save must be set when --rotate-ckpt is set."
        assert args.save_interval is not None, "--save-interval must be set when --rotate-ckpt is set."
        assert args.async_save is True, "--async-save must be set when --rotate-ckpt is set."

    if args.genrm_model_path:
        args.genrm_engine_config = args.genrm_engine_config or {}
        args.genrm_sampling_config = args.genrm_sampling_config or {}
