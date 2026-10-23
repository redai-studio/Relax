# Modified for Relax integration; see NOTICE.

"""Frozen CPU-mapped Engram tables and the trainable V4.1 retrieval gate."""

import json
import os
import struct
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer: Any) -> tuple[list[int], int]:
    """Match the checkpoint's tokenizer-normalized vocabulary numbering."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


@lru_cache(maxsize=8)
def get_compressed_token_map(hf_checkpoint: str) -> tuple[list[int], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    return build_compressed_token_map(tokenizer)


def compute_hash_multipliers(layer_ids: tuple[int, ...], max_ngram_size: int, vocab_size: int) -> torch.Tensor:
    if not layer_ids or max_ngram_size < 2 or vocab_size <= 0:
        raise ValueError("Engram hashing requires layers, ngram size >= 2 and a positive compressed vocabulary.")
    multiplier_bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def _mmap_rows(
    path: str,
    name: str,
    row_start: int,
    row_end: int,
    row_bytes: int,
    *,
    expected_rows: int,
    expected_dtypes: tuple[str, ...],
) -> torch.Tensor:
    """Read-only byte view of a safetensors row shard, without materializing
    the table.

    ``torch.from_numpy`` retains the NumPy memmap owner. The mapping remains
    live while a tensor/view references it and is released when that reference
    chain dies; no independently closed mmap or permanently open file is
    retained.
    """
    with open(path, "rb") as source:
        file_size = os.fstat(source.fileno()).st_size
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid safetensors header in {path!r}.")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > file_size - 8:
            raise ValueError(f"Safetensors header exceeds the file size in {path!r}.")
        header = json.loads(source.read(header_len))
    if name not in header:
        raise ValueError(f"Missing Engram tensor {name!r} in {path!r}.")
    metadata = header[name]
    shape = metadata.get("shape")
    if shape != [expected_rows, row_bytes] or metadata.get("dtype") not in expected_dtypes:
        raise ValueError(
            f"Engram tensor {name!r}: expected shape {[expected_rows, row_bytes]} and dtype {expected_dtypes}, "
            f"got shape {shape} and dtype {metadata.get('dtype')!r}."
        )
    if not 0 <= row_start <= row_end <= expected_rows or row_bytes <= 0:
        raise ValueError(f"Invalid Engram row range [{row_start}, {row_end}) for {expected_rows} rows.")
    offsets = metadata.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"Invalid safetensors offsets for {name!r}.")
    begin, end = offsets
    if not 0 <= begin <= end <= file_size - 8 - header_len or end - begin != expected_rows * row_bytes:
        raise ValueError(f"Invalid safetensors byte range for {name!r}.")
    n_rows = row_end - row_start
    if n_rows == 0:
        return torch.empty(0, row_bytes, dtype=torch.uint8)
    offset = 8 + header_len + begin + row_start * row_bytes
    mapped = np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(n_rows, row_bytes))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(mapped)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int


@lru_cache(maxsize=8)
def _build_layout(
    layer_ids: tuple[int, ...],
    num_embeddings: tuple[int, ...],
    max_ngram_size: int,
    n_heads: int,
    head_dim: int,
    vocab_size: int,
) -> EngramLayout:
    if len(layer_ids) != len(num_embeddings) or len(set(layer_ids)) != len(layer_ids):
        raise ValueError("Engram layer IDs must be unique and match the number of embedding tables.")
    if max_ngram_size < 2 or n_heads <= 0 or vocab_size <= 0 or head_dim <= 0 or head_dim % 32:
        raise ValueError("Engram requires ngram size >= 2, positive heads/vocabulary and head_dim divisible by 32.")
    primes: list[tuple[tuple[int, ...], ...]] = []
    seen: set[int] = set()
    for layer_id, rows in zip(layer_ids, num_embeddings):
        if layer_id < 0 or rows <= 0:
            raise ValueError("Engram layer IDs must be nonnegative and table row counts positive.")
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        if sum(sum(sizes) for sizes in per_ngram) != rows:
            raise ValueError(f"Engram layer {layer_id}: hash layout does not match the configured {rows} table rows.")
        primes.append(tuple(per_ngram))
    return EngramLayout(max_ngram_size, layer_ids, num_embeddings, tuple(primes), n_heads, head_dim)


def build_engram_layout(config: TransformerConfig) -> EngramLayout | None:
    layer_ids = tuple(config.v41_engram_layer_ids)
    if not layer_ids:
        return None
    return _build_layout(
        layer_ids,
        tuple(config.v41_engram_num_embeddings),
        config.v41_engram_max_ngram_size,
        config.v41_engram_n_heads,
        config.v41_engram_head_dim,
        config.v41_engram_vocab_size,
    )


def _table_shard(rows: int, tp_size: int, tp_rank: int) -> tuple[int, int]:
    if rows <= 0 or tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid Engram table size or tensor-parallel rank.")
    rows_local = -(-rows // tp_size)
    start = min(tp_rank * rows_local, rows)
    return start, min(start + rows_local, rows)


def _engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    hidden = x.float()
    rstd = torch.rsqrt(hidden.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (hidden * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    chunk: int = 4096,
) -> torch.Tensor:
    if chunk <= 0 or x.shape[0] <= chunk:
        return _engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)
    if not torch.is_grad_enabled():
        output = torch.empty_like(x)
        for start in range(0, x.shape[0], chunk):
            output[start : start + chunk] = _engram_gate(
                x[start : start + chunk], kv[start : start + chunk], q_weight, k_weight, eps, clamp_value
            )
        return output
    q32, k32 = q_weight.float(), k_weight.float()
    outputs = [
        torch.utils.checkpoint.checkpoint(
            _engram_gate,
            x[start : start + chunk],
            kv[start : start + chunk],
            q32,
            k32,
            eps,
            clamp_value,
            use_reentrant=False,
        )
        for start in range(0, x.shape[0], chunk)
    ]
    return torch.cat(outputs, dim=0)


class DeepSeekV41Engram(MegatronModule):
    """TP row-sharded frozen tables; only the retrieval projection/gate
    train."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_id: int,
        tp_group: torch.distributed.ProcessGroup | None,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        super().__init__(config=config)
        layout = build_engram_layout(config)
        if layout is None or layer_id not in layout.layer_ids:
            raise ValueError(f"Layer {layer_id} is not configured as an Engram layer.")
        if not getattr(config, "v41_hf_checkpoint", None):
            raise ValueError("Engram requires config.v41_hf_checkpoint to locate frozen tables and the tokenizer.")
        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1
        self.tp_group = tp_group
        self.tp_size = tp_group.size() if tp_group is not None else 1
        tp_rank = tp_group.rank() if tp_group is not None else 0
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.hc_mult = config.num_residual_streams
        self.dim = config.hidden_size
        self.head_dim = layout.head_dim
        self.max_ngram_size = layout.max_ngram_size
        self.eps = config.layernorm_epsilon
        self.clamp_value = 1e-6
        self.rows = layout.num_embeddings[self.layer_hash_index]
        self.row_start, row_end = _table_shard(self.rows, self.tp_size, tp_rank)
        self.rows_avail = row_end - self.row_start
        # Deliberately ordinary attributes: never move/checkpoint/synchronize the giant tables.
        self.table: torch.Tensor | None = None
        self.table_scale: torch.Tensor | None = None
        self._stage: tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None] | None = None
        self._stage_device: torch.device | None = None

        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.linear_wkv = TELinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=torch.float32))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=torch.float32))
        for weight in (self.q_weight, self.k_weight):
            mark_keep_in_fp32(weight)
            set_tensor_model_parallel_attributes(weight, False, 0, 1)
        self.sequence_parallel = config.sequence_parallel
        for parameter in self.parameters():
            parameter.sequence_parallel = False
        token_map, vocab_size = get_compressed_token_map(config.v41_hf_checkpoint)
        if vocab_size != config.v41_engram_compressed_vocab_size:
            raise ValueError(
                f"Engram tokenizer has {vocab_size} compressed tokens, "
                f"but HF config requires {config.v41_engram_compressed_vocab_size}."
            )
        if not 0 <= config.v41_engram_pad_token_id < len(token_map):
            raise ValueError("Engram pad token ID is outside the checkpoint tokenizer vocabulary.")
        self.pad_id = token_map[config.v41_engram_pad_token_id]
        flat_primes = [prime for sizes in layout.primes[self.layer_hash_index] for prime in sizes]
        offsets = np.cumsum([0, *flat_primes[:-1]])
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.register_buffer("token_map", torch.tensor(token_map, dtype=torch.int64), persistent=False)
        self.register_buffer("multipliers", multipliers[self.layer_hash_index].clone(), persistent=False)
        self.register_buffer(
            "primes", torch.tensor(layout.primes[self.layer_hash_index], dtype=torch.int64), persistent=False
        )
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False)
        self._load_table()

    def _load_table(self) -> None:
        checkpoint = self.config.v41_hf_checkpoint
        with open(os.path.join(checkpoint, "model.safetensors.index.json")) as source:
            weight_map = json.load(source)["weight_map"]
        weight_name = f"layers.{self.layer_id}.engram.embed.weight"
        scale_name = f"layers.{self.layer_id}.engram.embed.scale"
        for name in (weight_name, scale_name):
            if name not in weight_map:
                raise ValueError(f"Missing frozen Engram tensor {name!r} in checkpoint index.")
        end = self.row_start + self.rows_avail
        table = _mmap_rows(
            os.path.join(checkpoint, weight_map[weight_name]),
            weight_name,
            self.row_start,
            end,
            self.head_dim,
            expected_rows=self.rows,
            expected_dtypes=("F8_E4M3", "U8"),
        )
        scale = _mmap_rows(
            os.path.join(checkpoint, weight_map[scale_name]),
            scale_name,
            self.row_start,
            end,
            self.head_dim // 32,
            expected_rows=self.rows,
            expected_dtypes=("F8_E8M0", "U8"),
        )
        self.table, self.table_scale = table, scale

    def hash_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram input IDs must be an integer [batch, sequence] tensor.")
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        positions = torch.arange(seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens: list[torch.Tensor] = []
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = compressed.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift)
            tokens.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(tokens, dim=-1) * self.multipliers
        rolling, hashes = products[..., 0], []
        for index in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., index])
            hashes.append(rolling.unsqueeze(-1) % self.primes[index - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    def _staging(
        self, n_rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        if self._stage is not None and self._stage[2] is not None:
            # Do not overwrite host rows while the prior asynchronous H2D copy reads them.
            self._stage[2].synchronize()
        if self._stage is None or self._stage[0].shape[0] < n_rows or self._stage_device != device:
            is_cuda = device.type == "cuda"
            self._stage = (
                torch.empty(n_rows, self.head_dim, dtype=torch.uint8, pin_memory=is_cuda),
                torch.empty(n_rows, self.head_dim // 32, dtype=torch.uint8, pin_memory=is_cuda),
                torch.cuda.Event() if is_cuda else None,
            )
            self._stage_device = device
        rows, scales, done = self._stage
        return rows[:n_rows], scales[:n_rows], done

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        if self.table is None or self.table_scale is None:
            self._load_table()
        local = ids - self.row_start
        owned = (local >= 0) & (local < self.rows_avail)
        local_cpu = local.masked_fill(~owned, 0).reshape(-1).to("cpu")
        row_buffer, scale_buffer, done = self._staging(local_cpu.numel(), ids.device)
        if self.rows_avail:
            torch.index_select(self.table, 0, local_cpu, out=row_buffer)
            torch.index_select(self.table_scale, 0, local_cpu, out=scale_buffer)
        else:
            row_buffer.zero_()
            scale_buffer.zero_()
        rows = row_buffer.to(ids.device, non_blocking=done is not None).view(*ids.shape, -1)
        scales = scale_buffer.to(ids.device, non_blocking=done is not None).view(*ids.shape, -1)
        if done is not None:
            done.record(torch.cuda.current_stream(ids.device))
        rows = rows.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
        scales = scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        values = (rows * scales).flatten(-2).to(torch.bfloat16)
        values = values.masked_fill(~owned.unsqueeze(-1), 0)
        if self.tp_size > 1:
            torch.distributed.all_reduce(values, group=self.tp_group)
        return values

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )
        seqlen, batch, _ = hidden_states.shape
        if input_ids is None or input_ids.shape != (batch, seqlen):
            raise ValueError("Engram needs [batch, CP-local sequence] IDs matching the hidden-state sequence.")
        with torch.no_grad():
            if self.cp_size > 1:
                # V4.1 uses contiguous CP chunks, not Megatron's zigzag token ordering.
                chunks = [torch.empty_like(input_ids) for _ in range(self.cp_size)]
                torch.distributed.all_gather(chunks, input_ids.contiguous(), group=self.cp_group)
                start = self.cp_group.rank() * seqlen
                ids = self.hash_ids(torch.cat(chunks, dim=1))[:, start : start + seqlen].transpose(0, 1)
            else:
                ids = self.hash_ids(input_ids).transpose(0, 1)
            rows = self.lookup(ids)
        kv, _ = self.linear_wkv(rows.flatten(-2))
        hidden = hidden_states.view(seqlen, batch, self.hc_mult, self.dim)
        output = engram_gate(hidden, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value)
        output = output.view(seqlen, batch, self.hc_mult * self.dim)
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output
