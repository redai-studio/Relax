# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _cp_token_indices(
    input_ids: Tensor, packed_seq_params: PackedSeqParams | None, cp_size: int, cp_rank: int
) -> tuple[Tensor, tuple[int, int]]:
    """Select each sample's front/back CP chunks in the full token layout."""
    batch, width = input_ids.shape
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        boundaries = getattr(packed_seq_params, "cu_seqlens_q_cpu", None)
        if boundaries is None:
            raise ValueError("Kimi K3 packed CP requires host physical cu_seqlens_q_cpu from get_batch.")
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)]
        if batch == len(lengths):
            starts = [row * width for row in range(batch)]
            if max(lengths) > width:
                raise ValueError("Kimi K3 padded sample length exceeds its input row width.")
        elif batch == 1 and boundaries[-1] == width:
            starts = boundaries[:-1]
        else:
            raise ValueError("Kimi K3 packed CP input rows must match the physical sequence boundaries.")
        output_shape = (1, sum(lengths) // cp_size)
    else:
        lengths = [width] * batch
        starts = [row * width for row in range(batch)]
        output_shape = (batch, width // cp_size)
    if not lengths or any(length <= 0 or length % (2 * cp_size) for length in lengths):
        raise ValueError("Kimi K3 zigzag CP requires each physical sequence length divisible by 2 * CP.")
    indices = []
    for start, length in zip(starts, lengths, strict=True):
        chunk = length // (2 * cp_size)
        offset = torch.arange(chunk, device=input_ids.device)
        indices.extend((offset + start + cp_rank * chunk, offset + start + (2 * cp_size - cp_rank - 1) * chunk))
    return torch.cat(indices), output_shape


def _enable_vision_layer_checkpointing(vision_tower: nn.Module) -> None:
    """Recompute individual MoonViT blocks without changing parameter names."""

    def wrap_forward(block: nn.Module) -> Callable[..., Tensor]:
        forward = block.forward

        @wraps(forward)
        def checkpointed_forward(*args: Any, **kwargs: Any) -> Tensor:
            if block.training and torch.is_grad_enabled():
                return checkpoint(forward, *args, use_reentrant=False, **kwargs)
            return forward(*args, **kwargs)

        return checkpointed_forward

    for block in vision_tower.encoder.blocks:
        block.forward = wrap_forward(block)


def _build_vision_modules(config: Any) -> tuple[nn.Module, nn.Module]:
    """Load only MoonViT and PatchMergerMLPV2 from the checkpoint's custom
    code."""
    from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.utils import generic, is_flash_attn_2_available

    if not config.hf_model_path or not config.trust_remote_code:
        raise ValueError("Kimi K3 vision requires a HF checkpoint path with trust_remote_code=True.")
    if config.vision_config is None or config.vision_config.mm_projector_type != "patchmergerv2":
        raise ValueError("Kimi K3 vision requires its patchmergerv2 projector configuration.")

    # K3's HF module imports its language definitions alongside MoonViT. Recent
    # Transformers moved this class; preserve the old import used by that code.
    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder

    def load_class(name: str) -> type:
        return get_class_from_dynamic_module(
            f"modeling_kimi_k3.{name}", config.hf_model_path, trust_remote_code=config.trust_remote_code
        )

    vision_cls = load_class("MoonViT3dPretrainedModel")
    vision_config = load_class("VisionTowerConfig")(config.vision_config)
    projector_config = load_class("ProjectorConfig")(config.vision_config)
    if is_flash_attn_2_available():
        vision_config._attn_implementation = "flash_attention_2"
        # Transformers 5 checks the renamed capability flag; HF's MoonViT
        # advertises the Transformers 4 spelling.
        if getattr(vision_cls, "_supports_flash_attn_2", False):
            vision_cls._supports_flash_attn = True
    else:
        vision_config._attn_implementation = "eager"
        logger.warning("flash-attn is unavailable; Kimi K3 vision is using eager attention.")

    vision_tower = vision_cls(vision_config)
    _enable_vision_layer_checkpointing(vision_tower)
    projector = load_class("PatchMergerMLPV2")(projector_config)
    for module in (vision_tower, projector):
        module.to(dtype=config.params_dtype)
        hook_hf_module_setattr_for_tp_grad_sync(module)
    return vision_tower, projector


class KimiK3VLModel(MegatronModule):
    """K3's Megatron language model with HF vision features inserted before
    SP."""

    def __init__(
        self,
        config: Any,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)
        self.language_model = config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
        for layer in self.language_model.modules():
            # Relax uses MCore's Float16Module rather than Bridge's mixed
            # precision wrapper, so translate KDA's FP32 parameter metadata.
            for name in getattr(layer, "_keep_in_float32_parameter_names", ()):
                getattr(layer, name).keep_in_fp32 = True
            if getattr(layer, "layer_number", None) == 1:
                # The first layer enters with an empty AttnRes bank, so these
                # two modules never run. Keep their HF weights, but exclude
                # them from DDP's expected gradient-ready parameter set.
                for name in ("self_attention_res_norm", "self_attention_res_proj"):
                    unused_module = getattr(layer, name, None)
                    if unused_module is not None:
                        unused_module.requires_grad_(False)
        for parameter in self.language_model.parameters():
            if getattr(parameter, "sum_gradients_across_tp_domain", False):
                # Pinned MCore and Relax's coalesced reducer consume the SP
                # marker, while K3 Bridge emits a separate SUM marker.
                # K3's provider requires SP whenever TP > 1.
                parameter.sequence_parallel = True
        self.pre_process = self.language_model.pre_process
        self.post_process = self.language_model.post_process
        self.vp_stage = vp_stage
        self.pg_collection = self.language_model.pg_collection
        self.tp_group = self.pg_collection.tp
        self.cp_group = getattr(self.pg_collection, "cp", None)
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self.image_token_id = config.image_token_id
        self.vision_dp_when_tp = getattr(config, "vision_dp_when_tp", False)
        self.vision_tower = None
        self.mm_projector = None
        if self.pre_process:
            self.vision_tower, self.mm_projector = _build_vision_modules(config)

    def shared_embedding_or_output_weight(self) -> Tensor | None:
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor: Tensor | list[Tensor]) -> None:
        # K3 carries AttnRes state between PP stages in its language input. Keep
        # the entire payload intact rather than interpreting it as vision data.
        self.language_model.set_input_tensor(input_tensor)

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool) -> None:
        for module, frozen in (
            (self.language_model, freeze_language_model),
            (self.vision_tower, freeze_vision_model),
            (self.mm_projector, freeze_vision_projection),
        ):
            if frozen and module is not None:
                module.requires_grad_(False)

    def _image_features(self, pixel_values: Tensor, grid_thws: Tensor) -> Tensor:
        vision_parameter = next(self.vision_tower.parameters())
        pixel_values = pixel_values.to(device=vision_parameter.device, dtype=vision_parameter.dtype)
        frozen = not any(parameter.requires_grad for parameter in self.vision_tower.parameters())
        if self.vision_dp_when_tp and dist.get_world_size(self.tp_group) > 1:
            if not frozen:
                raise ValueError("Kimi K3 --vision-dp-when-tp requires --freeze-vision-model.")
            # SFT keeps this metadata on CPU; accept device grids from other
            # callers too, since HF MoonViT reads the geometry in Python.
            grid_thws = grid_thws.cpu()
            grids = grid_thws.tolist()
            patch_counts = [t * h * w for t, h, w in grids]
            merge_h, merge_w = self.config.vision_config.merge_kernel_size
            feature_counts = [h // merge_h * (w // merge_w) for _, h, w in grids]
            rank = dist.get_rank(self.tp_group)
            size = dist.get_world_size(self.tp_group)
            start = sum(patch_counts[: rank * len(grids) // size])
            stop = sum(patch_counts[: (rank + 1) * len(grids) // size])
            local_start = rank * len(grids) // size
            local_stop = (rank + 1) * len(grids) // size
            with torch.no_grad():
                local_features = (
                    self.vision_tower(pixel_values[start:stop], grid_thws[local_start:local_stop])
                    if local_start < local_stop
                    else []
                )
            # The frozen encoder needs no backward collective. Replicate its
            # output before the trainable projector so TP gradient sync remains valid.
            feature_width = self.config.vision_config.vt_hidden_size
            gathered = pixel_values.new_zeros((sum(feature_counts), merge_h * merge_w, feature_width))
            offset = sum(feature_counts[:local_start])
            for feature in local_features:
                gathered[offset : offset + feature.shape[0]].copy_(feature)
                offset += feature.shape[0]
            dist.all_reduce(gathered, group=self.tp_group)
            vision_features = list(gathered.split(feature_counts))
        elif frozen:
            with torch.no_grad():
                vision_features = self.vision_tower(pixel_values, grid_thws)
        else:
            vision_features = self.vision_tower(pixel_values, grid_thws)
        if self.training and torch.is_grad_enabled():
            features = checkpoint(lambda *items: self.mm_projector(list(items)), *vision_features, use_reentrant=False)
        else:
            features = self.mm_projector(vision_features)
        return torch.cat(features, dim=0)

    def _dummy_vision_dependency(self) -> Tensor | None:
        """Run the same module hooks for text batches so DDP can finish
        reduction."""
        if not self.training or not torch.is_grad_enabled():
            return None
        if not any(p.requires_grad for module in (self.vision_tower, self.mm_projector) for p in module.parameters()):
            return None
        vision_config = self.config.vision_config
        height, width = vision_config.merge_kernel_size
        patch_size = vision_config.patch_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        parameter = next(self.vision_tower.parameters())
        pixels = parameter.new_zeros((height * width, 3, patch_height, patch_width))
        # HF MoonViT reads grid geometry in Python; dummy metadata starts on CPU.
        grid = torch.tensor([[1, height, width]], dtype=torch.long)
        return self._image_features(pixels, grid).sum() * 0

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        padding_mask: Tensor | None = None,
        pixel_values: Tensor | None = None,
        grid_thws: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        extra_block_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        cp_size = self.cp_group.size() if self.cp_group is not None else 1
        cp_indices = None
        cp_shape = None
        original_shape = input_ids.shape
        if cp_size > 1:
            cp_indices, cp_shape = _cp_token_indices(input_ids, packed_seq_params, cp_size, self.cp_group.rank())

        def partition_tokens(value: Tensor | None) -> Tensor | None:
            if value is None or cp_indices is None:
                return value
            if value.shape == original_shape:
                return value.reshape(-1).index_select(0, cp_indices).view(cp_shape)
            if value.shape == cp_shape:
                return value
            raise ValueError("Kimi K3 CP token metadata must match either full or local input shape.")

        decoder_input = None
        if self.pre_process:
            if input_ids.ndim != 2:
                raise ValueError("Kimi K3 expects [batch, sequence] token IDs, including [1, tokens] for packed THD.")
            embedding = self.language_model.embedding
            dropout = getattr(embedding, "embedding_dropout", None)
            # Preserve full-sequence RNG consumption for training with dropout.
            # K3 disables embedding-internal SP; the wrapper scatters after merge.
            local_embedding = cp_indices is not None and not (
                dropout is not None and dropout.training and dropout.p > 0
            )
            decoder_input = embedding(
                input_ids=partition_tokens(input_ids) if local_embedding else input_ids,
                position_ids=partition_tokens(position_ids) if local_embedding else position_ids,
            )
            image_mask = input_ids == self.image_token_id
            grid = grid_thws if grid_thws is not None else image_grid_thw
            has_pixels = pixel_values is not None and pixel_values.shape[0] > 0
            has_grid = grid is not None and grid.shape[0] > 0
            if has_pixels != has_grid:
                raise ValueError("Kimi K3 requires both pixel_values and grid_thws for visual inputs.")
            if has_pixels:
                image_features = self._image_features(pixel_values, grid)
                # Keep this check on device: nonzero()/item() would synchronize
                # the training stream, and masked_scatter alone accepts surplus features.
                torch._assert_async(
                    image_mask.sum() == image_features.shape[0],
                    "Kimi K3 visual feature count must match the pre-expanded media placeholder count.",
                )
                embeddings_bsh = decoder_input.transpose(0, 1)
                if local_embedding:
                    local_image_mask = partition_tokens(image_mask)
                    feature_count = image_features.shape[0]
                    if feature_count:
                        feature_rows = image_mask.reshape(-1).long().cumsum(0) - 1
                        local_rows = feature_rows.index_select(0, cp_indices).view(cp_shape)
                        # Fixed-size gather avoids CUDA nonzero/boolean indexing
                        # synchronization. Text rows select a safe, unused feature.
                        local_rows = torch.where(local_image_mask, local_rows, 0)
                        local_features = image_features.index_select(0, local_rows.reshape(-1))
                        local_features = local_features.view(*cp_shape, image_features.size(-1))
                        decoder_input = torch.where(
                            local_image_mask.unsqueeze(-1), local_features.to(embeddings_bsh), embeddings_bsh
                        ).transpose(0, 1)
                        del local_features
                    else:
                        # An empty visual result must retain its backward hooks.
                        decoder_input = decoder_input + image_features.sum().to(decoder_input) * 0
                else:
                    decoder_input = embeddings_bsh.masked_scatter(
                        image_mask.unsqueeze(-1), image_features.to(embeddings_bsh)
                    ).transpose(0, 1)
            else:
                torch._assert_async(~image_mask.any(), "Kimi K3 media placeholders require visual inputs.")
                dependency = self._dummy_vision_dependency()
                if dependency is not None:
                    decoder_input = decoder_input + dependency.to(decoder_input)
            if cp_indices is not None:
                embeddings_bsh = decoder_input.transpose(0, 1)
                if attention_mask is not None:
                    if attention_mask.shape != original_shape or attention_mask.dtype != torch.bool:
                        raise ValueError("Kimi K3 packed CP expects a boolean [batch, sequence] valid-token mask.")
                    valid_mask = partition_tokens(attention_mask) if local_embedding else attention_mask
                    embeddings_bsh = embeddings_bsh.masked_fill(~valid_mask.unsqueeze(-1), 0)
                if not local_embedding:
                    embeddings_bsh = (
                        embeddings_bsh.reshape(-1, embeddings_bsh.size(-1))
                        .index_select(0, cp_indices)
                        .view(*cp_shape, embeddings_bsh.size(-1))
                    )
                decoder_input = embeddings_bsh.transpose(0, 1)
            decoder_input = decoder_input.contiguous()
            if self.config.sequence_parallel:
                decoder_input = scatter_to_sequence_parallel_region(decoder_input, group=self.tp_group).contiguous()

        if cp_indices is not None:
            input_ids = partition_tokens(input_ids)
            position_ids = partition_tokens(position_ids)
            labels = partition_tokens(labels)
            loss_mask = partition_tokens(loss_mask)
            padding_mask = partition_tokens(padding_mask)
            if attention_mask is not None:
                invalid_tokens = ~partition_tokens(attention_mask)
                padding_mask = invalid_tokens if padding_mask is None else padding_mask | invalid_tokens
                # The input mask describes BSHD padding, not an attention matrix.
                # Packed boundaries and causal attention now describe the local stream.
                attention_mask = None

        # GPT skips its embedding preprocessing when decoder_input is supplied,
        # so this wrapper also owns SP partitioning of the padding mask.
        if padding_mask is not None and self.config.sequence_parallel:
            padding_mask = (
                scatter_to_sequence_parallel_region(padding_mask.transpose(0, 1).contiguous(), group=self.tp_group)
                .transpose(0, 1)
                .contiguous()
            )
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            extra_block_kwargs=extra_block_kwargs,
            **kwargs,
        )
