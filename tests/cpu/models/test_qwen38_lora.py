# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real HC/PLE adapters on CPU, plus two-rank HC TP gradient reduction.

Requires the Qwen3.8 Bridge/Core runtime on PYTHONPATH. PLE's TP>1 lookup
allocates CUDA tensors, so that path belongs to GPU integration tests.
"""

import copy
import io
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
dist = torch.distributed


def _runtime() -> tuple[type, type, type]:
    lora = pytest.importorskip("megatron.bridge.peft.lora")
    hc = pytest.importorskip("megatron.core.transformer.hyper_connection")
    ple = pytest.importorskip("megatron.core.transformer.per_layer_embedding")
    if not hasattr(hc, "GatedResidualHyperConnection") or not hasattr(ple, "PerLayerEmbedding"):
        pytest.skip("Requires the Qwen3.8 HC/PLE model implementation")
    return lora.LoRA, hc.GatedResidualHyperConnection, ple.PerLayerEmbedding


def _model(tp_size: int, with_ple: bool) -> torch.nn.Module:
    from megatron.core.transformer.transformer_config import TransformerConfig

    _, hc, ple = _runtime()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=tp_size > 1,
        enable_hyper_connections=True,
        mhc_variant="gated_residual",
        num_residual_streams=2,
        mhc_gated_residual_rank=4,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_ngram_size=2,
        ple_heads_per_ngram=1,
        ple_ngram_vocab_size_base=17,
        ple_ngram_vocab_divisible_by=2,
        ple_unigram_vocab_size=32,
        ple_eos_token_id=0,
        ple_seed=1234,
        ple_conv_kernel_size=2,
    )
    torch.manual_seed(912)
    model = torch.nn.Module()
    model.config = config
    model.hc = hc(config)
    if with_ple:
        model.ple = ple(config, 1)
        with torch.no_grad():
            model.ple.conv1d.weight.uniform_(-0.1, 0.1)
    return model


def _inject(model: torch.nn.Module) -> Namespace:
    from relax.models.qwen3_8.lora import HC_TARGETS, PLE_TARGETS, qwen38_lora_targets
    from relax.utils.training.lora_grad_sync import inherit_lora_tp_grad_sync

    lora, _, _ = _runtime()
    targets = list(HC_TARGETS) + (list(PLE_TARGETS) if hasattr(model, "ple") else [])
    args = Namespace(lora_rank=2, lora_dropout=0.0, lora_scope="all", lora_target_modules=targets)
    resolved = qwen38_lora_targets(model, targets, args)
    lora(target_modules=resolved, dim=2, alpha=4, dropout=0.0)(model)
    inherit_lora_tp_grad_sync(model)
    return args


def _forward(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "ple"):
        tokens = torch.tensor([[1, 4, 2, 7, 9, 3, 2, 8, 5, 1, 3, 9, 2, 4, 8, 6]])
        model.ple.prepare(tokens, torch.tensor([0, 7, 16], dtype=torch.int32))
        hidden = hidden + model.ple(hidden)
    block_input, h_res, h_post, residual = model.hc(hidden)
    return model.hc.fused_h_res_h_post_bda(h_res, residual, h_post, (block_input, None), 0.0, True, False)


def _adapters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def _loss(output: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((output.square() + 0.1 * output.sin()) * weights).sum()


def _single_rank_contract() -> None:
    from megatron.bridge.peft.lora_layers import LoRALinear
    from megatron.bridge.peft.lora_merge import LoRAMerge

    model = _model(1, with_ple=True)
    hidden = torch.randn(16, 1, 16)
    weights = torch.arange(1, 17, dtype=torch.float32).view(16, 1, 1) / 16
    with torch.no_grad():
        base_output = _forward(model, hidden)
    _inject(model)
    assert len(_adapters(model)) == 10
    torch.testing.assert_close(_forward(model, hidden), base_output, rtol=0, atol=0)
    frozen = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    optimizer = torch.optim.Adam(_adapters(model).values(), lr=0.01)

    def step(index: int) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, hidden)
        loss = _loss(output, weights)
        loss.backward()
        for name, parameter in _adapters(model).items():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if index > 0 or "linear_out" in name:
                assert torch.count_nonzero(parameter.grad) > 0, name
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                assert parameter.grad is None, name
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        return loss.detach()

    step(0)
    step(1)
    checkpoint = io.BytesIO()
    torch.save(
        {
            "adapter": {name: value for name, value in model.state_dict().items() if ".adapter." in name},
            "optimizer": optimizer.state_dict(),
        },
        checkpoint,
    )
    expected_loss = step(2)
    expected_params = {name: value.detach().clone() for name, value in _adapters(model).items()}
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=True)
    incompatible = model.load_state_dict(restored["adapter"], strict=False)
    assert not incompatible.unexpected_keys
    assert not any(".adapter." in name for name in incompatible.missing_keys)
    optimizer.load_state_dict(restored["optimizer"])
    torch.testing.assert_close(step(2), expected_loss, rtol=0, atol=0)
    for name, parameter in _adapters(model).items():
        torch.testing.assert_close(parameter, expected_params[name], rtol=0, atol=0)

    model.eval()
    with torch.no_grad():
        expected = _forward(model, hidden)
        for name, module in list(model.named_modules()):
            if not isinstance(module, LoRALinear):
                continue
            adapter = module.adapter
            merged = LoRAMerge().merge(
                module.to_wrap.weight,
                adapter.linear_out.weight,
                adapter.linear_in.weight,
                adapter.alpha,
                adapter.dim,
                tp_group=None,
            )
            module.to_wrap.weight.copy_(merged)
            owner, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(owner), leaf, module.to_wrap)
        torch.testing.assert_close(_forward(model, hidden), expected, rtol=1e-5, atol=1e-6)


def _tp_contract(rank: int) -> None:
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

    from relax.models.common.chunked_grad_coalesce import _allreduce_non_tensor_model_parallel_grads
    from relax.models.qwen3_8.lora import validate_qwen38_lora
    from relax.utils.distributed_utils import init_gloo_group

    model = _model(2, with_ple=False)
    args = _inject(model)
    model.ddp_config = DistributedDataParallelConfig()
    # The validator must synchronize fresh replicas, not only mark their gradients.
    with torch.no_grad():
        for name, parameter in _adapters(model).items():
            if "linear_in" in name:
                parameter.add_(rank)
    init_gloo_group(distributed_timeout_minutes=2)
    validate_qwen38_lora([model], args)
    for parameter in _adapters(model).values():
        copies = [torch.empty_like(parameter) for _ in range(2)]
        dist.all_gather(copies, parameter, group=dist.group.WORLD)
        torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)

    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(_adapters(model).values(), lr=0.1)
    reference_optimizer = torch.optim.SGD(_adapters(reference).values(), lr=0.1)
    hidden = torch.randn(16, 1, 16)
    weights = torch.arange(1, 17, dtype=torch.float32).view(16, 1, 1) / 16
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        full_input = hidden.clone().requires_grad_()
        local_input = hidden.chunk(2)[rank].clone().requires_grad_()
        expected = _forward(reference, full_input)
        actual = _forward(model, local_input)
        torch.testing.assert_close(actual, expected.chunk(2)[rank], rtol=1e-5, atol=1e-6)
        _loss(expected, weights).backward()
        _loss(actual, weights.chunk(2)[rank]).backward()
        _allreduce_non_tensor_model_parallel_grads([model], model.config, dist.group.WORLD)
        torch.testing.assert_close(local_input.grad, full_input.grad.chunk(2)[rank], rtol=1e-5, atol=1e-6)
        for name, parameter in _adapters(model).items():
            torch.testing.assert_close(parameter.grad, _adapters(reference)[name].grad, rtol=1e-4, atol=1e-6)
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in _adapters(model).items():
            torch.testing.assert_close(parameter, _adapters(reference)[name], rtol=1e-5, atol=1e-6)


def _worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
        _single_rank_contract()
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        _tp_contract(rank)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_qwen38_lora_updates_restore_merge_and_tp_sum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime()
    # Child processes use the model's real CPU initialization path, even on GPU hosts.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("QWEN38_PLE_CPU_OFFLOAD", "0")
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
