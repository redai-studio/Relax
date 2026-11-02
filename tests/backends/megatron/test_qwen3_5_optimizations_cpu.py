# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU correctness contracts for the optional Qwen training optimizations."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
PATCHES = ROOT / "relax/models/qwen3_5"


def load(name):
    spec = importlib.util.spec_from_file_location(f"qwen_test_{name}", PATCHES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tampered", [False, True])
def test_cublas_verification_uses_image_manifest(tmp_path, monkeypatch, tampered):
    import hashlib
    import sys
    from types import SimpleNamespace

    library = tmp_path / "libcublasLt.so.13"
    library.write_bytes(b"image-built-library")
    manifest = {
        "cublas_library_root": str(tmp_path),
        "cublas_library_sha256": {library.name: hashlib.sha256(library.read_bytes()).hexdigest()},
    }
    monkeypatch.setitem(
        sys.modules,
        "relax.models.qwen3_5.dependency_checks",
        SimpleNamespace(runtime_manifest=lambda: manifest),
    )
    monkeypatch.setitem(sys.modules, "transformer_engine_torch", SimpleNamespace(get_cublasLt_version=lambda: 130300))
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/self/maps":
            return f"0000-1000 r-xp 00000000 00:00 0 {library}\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    module = load("grouped_mlp")
    if tampered:
        library.write_bytes(b"unexpected-library")
        with pytest.raises(AssertionError):
            module.verify_runtime()
    else:
        assert module.verify_runtime()["mapped"] == [str(library)]


@pytest.mark.parametrize("mask", [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
def test_selected_mtp_preserves_loss_and_gradients(mask):
    module = load("mtp_loss")
    torch.manual_seed(7)
    hidden = torch.randn(4, 1, 3, requires_grad=True)
    weights = torch.randn(5, 3, requires_grad=True)
    labels = torch.tensor([[0, 3, 1, 4]])
    loss_mask = torch.tensor([mask], dtype=torch.float32)
    reference = F.cross_entropy(hidden[:, 0] @ weights.T, labels.flatten(), reduction="none")
    expected = (reference * loss_mask.flatten()).sum()
    gradients = torch.autograd.grad(expected, (hidden, weights))

    class Head:
        sequence_parallel = False

        def __call__(self, x, *, labels, **kwargs):
            return F.cross_entropy(x[:, 0] @ weights.T, labels.flatten(), reduction="none").reshape_as(labels)

    head = Head()
    actual = module.selected_mtp_head(head, loss_mask, hidden, labels=labels, output_cross_entropy_loss=True)
    torch.testing.assert_close((actual * loss_mask).sum(), expected)
    actual_gradients = torch.autograd.grad((actual * loss_mask).sum(), (hidden, weights))
    for actual_gradient, expected_gradient in zip(actual_gradients, gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert head.sequence_parallel is False


def test_selected_mtp_restores_head_on_failure():
    module = load("mtp_loss")

    class Head:
        sequence_parallel = False

        def __call__(self, *args, **kwargs):
            raise RuntimeError("head failed")

    head = Head()
    with pytest.raises(RuntimeError, match="head failed"):
        module.selected_mtp_head(
            head,
            torch.ones(1, 2),
            torch.ones(2, 1, 3),
            labels=torch.zeros(1, 2, dtype=torch.long),
            output_cross_entropy_loss=True,
        )
    assert head.sequence_parallel is False


def test_checkpoint_masks_preserve_gdn_and_long_wave_policy():
    module = load("recompute")
    short, long = module.checkpoint_mask(262144), module.checkpoint_mask(262145)
    assert len(short) == 16 and len(long) == 32
    assert len(set(long)) == 32 and set(short) <= set(long)
    assert all((index + 1) % 4 != 0 for index in short)
    assert [index for index in long if (index + 1) % 4 == 0] == [3, 7]


def test_pair_plan_preserves_actual_tp_order():
    module = load("kv_parallel")
    assert module.pair_plan([[3, 1, 0, 2], [3, 1, 0, 2]], 4) == [[3, 1], [0, 2]]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_norm_guard_rejects_nonfinite_before_update(value):
    with pytest.raises(FloatingPointError):
        load("grad_norm").finite_norm(torch.tensor(value))


@pytest.mark.parametrize("rows", [[0, 2], [], [0, 1, 2]])
def test_selected_main_head_matches_dense_masked_loss_and_gradients(rows):
    module = load("selective_loss")
    torch.manual_seed(11)
    hidden = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    target = torch.tensor([1, 5, 3])
    mask = torch.tensor([int(i in rows) for i in range(3)], dtype=torch.float32)
    expected = (F.cross_entropy(hidden @ weight.T, target, reduction="none") * mask).sum()
    gradients = torch.autograd.grad(expected, (hidden, weight))
    calls = []

    def head(h, *, labels):
        calls.append(h.shape[0])
        return F.cross_entropy(h @ weight.T, labels, reduction="none")

    actual = module.selected_linear_loss([(hidden, target)], [(3, torch.tensor(rows, dtype=torch.long))], head)
    torch.testing.assert_close((actual * mask).sum(), expected)
    for got, want in zip(torch.autograd.grad((actual * mask).sum(), (hidden, weight)), gradients):
        torch.testing.assert_close(got, want)
    assert calls == [max(len(rows), 1)]


@pytest.mark.parametrize("full_layer", [False, True])
def test_expert_static_forward_preserves_checkpoint_gradients(full_layer):
    from torch.utils.checkpoint import checkpoint

    module = load("expert_recompute")
    hidden = torch.randn(3, 4, requires_grad=True)
    probs = torch.randn(3, 1, requires_grad=True)
    counts = torch.tensor([3])
    expected = (hidden.sin() * probs).sum()
    expected_grads = torch.autograd.grad(expected, (hidden, probs))

    def native(x, counts, probs):
        return x.sin() * probs, None

    def checkpoint_function(body, distribute_saved_activations, *args):
        assert distribute_saved_activations is False
        return checkpoint(body, *args, use_reentrant=True)

    state = dict(
        active=full_layer,
        checkpoint=checkpoint_function,
        checkpoint_calls=0,
        replay_calls=0,
        full_bypass_calls=0,
    )
    output, bias = module.forward(None, hidden, counts, probs, native, state)
    # Layer exit clears current-wave state before backward. Replay must capture
    # its original native call and never consult the next wave's packed mask.
    state["active"] = None
    assert bias is None
    output.sum().backward()
    for actual, expected in zip((hidden.grad, probs.grad), expected_grads):
        torch.testing.assert_close(actual, expected)
    assert state["checkpoint_calls"] == int(not full_layer)
    assert state["replay_calls"] == int(not full_layer)
    assert state["full_bypass_calls"] == int(full_layer)


def test_expert_static_forward_rejects_missing_layer_context():
    with pytest.raises(AssertionError, match="outside its transformer layer"):
        load("expert_recompute").forward(None, None, None, None, None, {"active": None})


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_static_layer_checkpoint_context_restores_state(masked, fails):
    from types import SimpleNamespace

    module = load("layer_recompute")
    expert = dict(layer=3, active=None, mask_key="mask")
    gdn = SimpleNamespace(recompute_gdn=False)
    state = dict(layer=3, module=gdn, mask_key="mask", entered=False)
    layer = SimpleNamespace(_qwen_expert_checkpoint_state=expert, _qwen_gdn_checkpoint_state=state)
    packed = SimpleNamespace(mask=(3,) if masked else ())
    try:
        with module.layer_context(layer, packed):
            assert expert["active"] is masked
            assert gdn.recompute_gdn is not masked
            with pytest.raises(AssertionError, match="recursive"):
                with module.layer_context(layer, packed):
                    pytest.fail("recursive context must be rejected")
            assert expert["active"] is masked
            if fails:
                raise RuntimeError("layer failed")
    except RuntimeError as error:
        assert fails and str(error) == "layer failed"
    assert expert["active"] is None
    assert gdn.recompute_gdn is False
    assert state["entered"] is False


def test_static_layer_checkpoint_context_leaves_mtp_unchanged():
    from types import SimpleNamespace

    with load("layer_recompute").layer_context(SimpleNamespace(), None):
        pass


@pytest.mark.parametrize("rows", [253952, 258048, 262144, 262148])
@pytest.mark.parametrize("mode", ["column", "row"])
@pytest.mark.parametrize("fails", [False, True])
def test_static_projection_selects_exact_buffers_and_restores(monkeypatch, rows, mode, fails):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "relax.models.qwen3_5.userbuffers", load("userbuffers"))
    module = load("projection_hooks")
    flags = {field: field == "ub_overlap_ag_fprop" for field in module.FIELDS}
    state = dict(entered=False, ub_forwards=0, native_forwards=0, name="qkv", active=flags)
    projection = SimpleNamespace(parallel_mode=mode, ub_name="qkv_s262144", _qwen_exact_ub_state=state, **flags)
    inputs = SimpleNamespace(shape=(rows // 4 if mode == "column" else rows, 1, 2048))
    supported = rows in module.CAPACITIES
    calls = []

    def native(value):
        calls.append(value)
        assert state["entered"] is True
        assert projection.ub_name == (f"qkv_s{rows}" if supported else "qkv_s262144")
        assert {field: getattr(projection, field) for field in module.FIELDS} == {
            field: enabled if supported else False for field, enabled in flags.items()
        }
        if fails:
            raise RuntimeError("projection failed")
        return value, None

    if fails:
        with pytest.raises(RuntimeError, match="projection failed"):
            module.run_projection(projection, inputs, native)
    else:
        assert module.run_projection(projection, inputs, native) == (inputs, None)
    assert calls == [inputs]
    assert state["entered"] is False
    assert state["ub_forwards"] == int(supported)
    assert state["native_forwards"] == int(not supported)
    assert projection.ub_name == "qkv_s262144"
    assert {field: getattr(projection, field) for field in module.FIELDS} == flags
