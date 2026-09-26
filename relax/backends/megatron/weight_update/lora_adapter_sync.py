# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Transport-independent LoRA adapter-mode sync logic.

In LoRA *adapter mode* the base model is synced to the rollout engine once and, every
subsequent step, only the trained LoRA adapter is pushed. The colocate
(``UpdateWeightFromTensor``) and fully-async (``DeviceDirectBackend``) backends differ only
in *how* the adapter reaches the engine (CUDA-IPC handles vs an NCCL broadcast from rank 0).
Everything else — building the export bridge, exporting the local adapter, the collective
delta-skip decision and the PP gather — is identical. This helper holds that shared logic
(and the per-instance cross-call state) so both backends compose it instead of duplicating it.

Collective contract: ``export_local_adapter``, ``gather_full_adapter`` and ``should_skip``
issue collective ops (splice, PP gather, gloo all-reduce) and MUST be called on every rank in
the same order. They never gate a collective behind a rank check.
"""

import torch
import torch.distributed as dist

from relax.backends.megatron.misc_utils import strip_param_name_prefix
from relax.utils import megatron_bridge_utils
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger
from relax.utils.megatron_peft_utils import (
    build_hf_peft_config_dict,
    convert_megatron_to_sglang_target_modules,
    extract_lora_delta,
    is_lora_adapter_param,
    repack_gdn_adapter_for_sglang,
)


logger = get_logger(__name__)


class LoraAdapterSync:
    """Shared, transport-independent state and logic for LoRA adapter-mode
    weight sync.

    One instance per backend (composition — no inheritance). The backend keeps only its
    transport (IPC / NCCL) and orchestration (pause/continue); this object owns:

    * ``bridge`` — a lazily-built ``AutoBridge`` used to export the adapter in HF format.
    * ``prev_state`` — the previous adapter snapshot backing the delta-skip comparison.
    * ``base_sync_done`` — gates the one-time base-weight sync.
    * ``adapter_loaded`` — gates the unload-before-reload on the engine.
    """

    def __init__(self, args, model) -> None:
        self.args = args
        self.model = model
        self.bridge = None
        self.prev_state: dict[str, torch.Tensor] | None = None
        self.base_sync_done = False
        self.adapter_loaded = False

    # ------------------------------------------------------------------
    # Bridge / IO
    # ------------------------------------------------------------------

    def _bridge(self):
        """Lazily build and cache an ``AutoBridge`` for adapter export.

        The fast weight iterator wraps a ``BridgeConverter`` and does not
        expose a raw bridge, so adapter export builds its own bridge from the
        HF checkpoint — the same construction the checkpoint save uses
        (``_save_lora_to_checkpoint``).
        """
        if self.bridge is None:
            from megatron.bridge import AutoBridge

            self.bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        return self.bridge

    def config_dict(self) -> dict:
        """Adapter config dict (the in-memory counterpart of
        ``adapter_config.json``).

        ``args.lora_target_modules`` holds canonical Megatron names; SGLang
        matches modules by leaf name and validates that the adapter's
        ``target_modules`` are a subset of the engine's, so both sides use the
        SGLang flavor (see ``convert_megatron_to_sglang_target_modules``). The
        *checkpoint* adapter written by ``_save_lora_to_checkpoint`` keeps HF-
        PEFT names instead — that one is a portable artifact, this one only
        ever feeds the engine.
        """
        return build_hf_peft_config_dict(
            lora_rank=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=convert_megatron_to_sglang_target_modules(self.args.lora_target_modules),
            lora_dropout=self.args.lora_dropout,
        )

    # ------------------------------------------------------------------
    # Export / collectives
    # ------------------------------------------------------------------

    def export_local_adapter(self, all_params) -> dict[str, torch.Tensor]:
        """Export this rank's LoRA adapter tensors in SGLang's naming (CPU; TP
        already gathered by the bridge).

        Must run inside ``splice_adapter_weights`` so the bridge reads the tms-
        safe CPU backup rather than the paused live params. Collective (runs
        the bridge export) — call on every rank.

        The bridge emits HF-PEFT names; ``repack_gdn_adapter_for_sglang`` then folds the
        GDN ``in_proj`` group into the fused ``in_proj_qkvz`` the engine wraps. Both
        consumers of this method push to an engine (colocate IPC / device-direct HTTP);
        the checkpoint export keeps HF names and runs its own bridge call.
        """
        renamed = {strip_param_name_prefix(k): v for k, v in all_params.items()}
        bridge = self._bridge()
        with (
            megatron_bridge_utils.patch_megatron_model(self.model),
            megatron_bridge_utils.splice_adapter_weights(renamed),
        ):
            # cpu=True: adapter tensors are gathered over gloo, so they must live on CPU.
            exported = {
                item.param_name: item.weight.detach().cpu()
                for item in bridge.export_adapter_weights(self.model, cpu=True, show_progress=False)
            }
        return repack_gdn_adapter_for_sglang(exported)

    def should_skip(self, all_params) -> tuple[bool, dict[str, torch.Tensor]]:
        """Collective delta-skip decision: has any rank's adapter changed
        beyond threshold?

        Returns ``(unchanged, new_state)``. ``unchanged`` is True only when NO
        rank saw a change (the MAX all-reduce over gloo makes this a world-
        collective decision — a per-rank early return would desync the gather
        inside the push and hang). ``new_state`` is the fresh snapshot for the
        caller to store in ``prev_state`` (this method never mutates
        ``prev_state`` itself, so callers keep their exact skip-vs-success
        assignment ordering).
        """
        lora_params = {k: v for k, v in all_params.items() if is_lora_adapter_param(k)}
        lora_delta, new_state = extract_lora_delta(lora_params, self.prev_state)
        any_changed = torch.tensor([1 if lora_delta else 0], dtype=torch.int32)
        dist.all_reduce(any_changed, op=dist.ReduceOp.MAX, group=get_gloo_group())
        return any_changed.item() == 0, new_state

    def gather_full_adapter(self, local_adapter: dict, *, all_gather: bool) -> dict | None:
        """Gather every PP rank's adapter shard into one full adapter dict.

        The adapter differs only across PIPELINE ranks (the bridge already TP-gathers each
        tensor and DP ranks are identical replicas), so gather over the PP group only — a
        no-op for PP=1.

        * ``all_gather=True`` — every PP rank receives the merged dict (``all_gather_object``).
          Needed when the load is issued from a rank that may not be PP-rank 0 (colocate
          tensor transport: each engine's gather-src rank pushes in-memory).
        * ``all_gather=False`` — only PP-rank 0 (== world rank 0) receives the merged dict;
          all other ranks get ``None`` (``gather_object``). Used by the device-direct
          transport, where only rank 0 is the broadcast source.
        """
        from megatron.core import mpu

        pp_group = mpu.get_pipeline_model_parallel_group()
        pp_ranks = dist.get_process_group_ranks(pp_group)
        if len(pp_ranks) == 1:
            # PP=1: this rank's local_adapter IS the full adapter. Skip gather_object,
            # which otherwise pickle-serializes the entire (multi-GB, per-expert) adapter
            # dict to a byte tensor even for a 1-rank group — the dominant cost of the
            # adapter push at PP=1. Numerically identical (every rank already held the
            # full adapter; callers still gate use on dist.get_rank()==0).
            return dict(local_adapter)
        if all_gather:
            gathered: list = [None] * len(pp_ranks)
            dist.all_gather_object(gathered, local_adapter, group=pp_group)
        else:
            pp_dst = pp_ranks[0]
            gathered = [None] * len(pp_ranks) if dist.get_rank() == pp_dst else None
            dist.gather_object(local_adapter, object_gather_list=gathered, dst=pp_dst, group=pp_group)
            if gathered is None:
                return None
        merged: dict[str, torch.Tensor] = {}
        for shard in gathered:
            if shard:
                merged.update(shard)
        return merged
