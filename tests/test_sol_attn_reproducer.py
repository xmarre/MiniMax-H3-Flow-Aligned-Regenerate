"""CPU diagnostic contracts and opt-in actual CUDA regression coverage."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sol_reproducer", ROOT / "tools/reproduce_sol_attn_determinism.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.torch = torch


def test_histogram_whole_bin_admission_and_clamped_top_overflow():
    h = [0] * 128
    h[20], h[21] = 100, 200
    threshold = probe.histogram_threshold(h, 0.0, 256)
    assert threshold == -8 + 21 * 0.25
    # The 200 members of bin 21 fit; the 100 members of bin 20 do not.
    assert -8 + 20 * 0.25 < threshold <= -8 + 21 * 0.25
    h = [0] * 128
    h[-1] = 1952
    threshold = probe.histogram_threshold(h, 0.0, 256)
    assert threshold == 80.0
    # Scores beyond the clamped bin upper edge still meet the raw threshold.
    assert threshold <= 112.0
    # This is a CPU source model, not measured reservation or GPU evidence.


def test_empty_histogram_and_exact_budget():
    h = [0] * 128
    assert probe.histogram_threshold(h, 3.0, 256) == -5.0
    h[-1] = 256
    assert probe.histogram_threshold(h, 3.0, 256) == -5.0
    with pytest.raises(ValueError):
        probe.histogram_threshold([-1] * 128, 0, 256)


def test_canonical_ids_ignore_only_unwritten_tail_and_preserve_rows():
    ids = torch.tensor([[2, 7, 123, 456], [1, 3, 8, 9]])
    counts = torch.tensor([2, 5])
    assert probe.canonical_indices(ids, counts, 4).tolist() == [[2, 7, 0, 0], [1, 3, 8, 9]]
    assert torch.equal(ids, torch.tensor([[2, 7, 123, 456], [1, 3, 8, 9]]))
    with pytest.raises(ValueError):
        probe.canonical_indices(ids, torch.tensor([-1, 0]), 4)


def test_digest_bfloat16_full_bytes_and_noncontiguous():
    a = torch.arange(24).reshape(4, 6).to(torch.bfloat16)
    assert probe.digest(a.T) == probe.digest(a.T.contiguous())
    assert probe.digest(a[1:2, 2:3]) == probe.digest(a[1:2, 2:3].clone())
    b = a.clone()
    b[-1, -1] += 1
    assert probe.digest(a) != probe.digest(b)


def test_comparison_finds_exact_ids_and_numerical_difference():
    c = probe.Comparisons()
    a = torch.tensor([[[1, 2, 0, 0], [3, 4, 0, 0]]])
    c.observe("token", {"tokIdx": a}, 0)
    b = a.clone()
    b[0, 1, 0] = 8
    r = c.observe("token", {"tokIdx": b}, 1)["tokIdx"]
    assert r["bitwise_equal"] is False
    assert r["first_changed_list"] == {"location": [0, 1], "before": [3, 4, 0, 0], "after": [8, 4, 0, 0]}
    assert r["max_abs"] == 5
    assert c.first_difference == {"iteration": 1, "stage": "token", "tensor": "tokIdx"}


def test_workspace_slot_bounds_and_initialization_mask():
    ws = torch.arange(64, dtype=torch.uint8)
    x = probe.view_slot(ws, {"x": 16}, "x", torch.int32, (2, 2))
    assert x.data_ptr() == ws.data_ptr() + 16
    with pytest.raises(ValueError, match="out-of-range"):
        probe.view_slot(ws, {"x": 60}, "x", torch.int32, (2,))


def test_unsigned_block_ids_preserve_65535_without_loading_unused_tail():
    ids = torch.tensor([[-1, 1, -1234]], dtype=torch.int16)
    assert probe.canonical_indices(ids, torch.tensor([2]), 3).tolist() == [[65535, 1, 0]]


def test_admission_order_changes_do_not_imply_changed_retained_set():
    counts = torch.tensor([3])
    a = torch.tensor([[9, 2, 5, 123]])
    b = torch.tensor([[5, 9, 2, 456]])
    assert torch.equal(probe.sorted_token_indices(a, counts, 4), probe.sorted_token_indices(b, counts, 4))
    b[0, 2] = 7
    assert not torch.equal(probe.sorted_token_indices(a, counts, 4), probe.sorted_token_indices(b, counts, 4))


def test_wrapper_clone_uses_private_globals_and_restores_no_state():
    def original():
        return _C, _wrap_for_dlpack  # noqa: F821

    old_c, old_wrap = object(), object()
    namespace = dict(original.__globals__, _C=old_c, _wrap_for_dlpack=old_wrap)
    import types

    original = types.FunctionType(original.__code__, namespace)
    backend = types.SimpleNamespace(sol_attn_chunked=original, _C=old_c)
    observer = object.__new__(probe.ExtensionObserver)
    observer.backend = backend
    cloned = observer.wrapper()
    assert cloned()[0] is observer
    assert cloned()[1](42) == 42
    assert original() == (old_c, old_wrap)


def test_token_summary_distinguishes_candidate_window_reservations_and_cap():
    # One group, one candidate block, two live tokens in that last block.
    tensors = {
        "tokCnt": torch.tensor([[5]]),
        "tokIdx": torch.tensor([[[0, 1, 2, 3]]]),
        "tokHist": torch.zeros(1, 1, 128, dtype=torch.int64),
        "gBits": torch.tensor([[[2]]]),
    }
    tensors["tokHist"][..., -1] = 5
    summary = probe.token_summary(tensors, 66, 4)
    assert summary["overflow_groups"] == 1
    assert summary["max_window_candidates"] == 5
    assert summary["total_reservations"] == 5
    assert summary["total_retained"] == 4
    assert summary["candidate_tokens_before_score_window"] == probe.receipt(torch.tensor([[2]]))


def test_verified_source_rejects_changed_bytes(tmp_path, monkeypatch):
    import hashlib
    import json

    manifest = tmp_path / "manifest.json"
    data = {"commit": probe.SOURCE_COMMIT, "files": {"a": hashlib.sha256(b"original").hexdigest()}}
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(probe, "MANIFEST", manifest)
    (tmp_path / "a").write_bytes(b"original")
    assert probe.verified_source(tmp_path) == data
    (tmp_path / "a").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source mismatch"):
        probe.verified_source(tmp_path)



def test_comparisons_reuses_sha_after_exact_byte_check(monkeypatch):
    original_digest = probe.digest
    calls = []

    def counted(tensor):
        calls.append(1)
        return original_digest(tensor)

    monkeypatch.setattr(probe, "digest", counted)
    comparison = probe.Comparisons()
    first = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
    initial = comparison.observe("out", {"out": first}, 0)["out"]
    assert len(calls) == 1
    assert initial["bitwise_equal"] is True
    # The stored reference must not alias the caller's mutable input.
    first[0] = -0.0
    repeated = comparison.observe("out", {"out": first.clone()}, 1)["out"]
    assert repeated["bitwise_equal"] is False
    assert repeated["sha256"] != initial["sha256"]
    assert repeated["rms"] == 0.0
    assert len(calls) == 2
    unchanged = comparison.observe("out", {"out": torch.tensor([0.0, 1.0], dtype=torch.bfloat16)}, 2)["out"]
    assert unchanged["bitwise_equal"] is True
    assert unchanged["sha256"] == initial["sha256"]
    assert len(calls) == 2


def test_comparisons_preserves_signed_zero_and_nan_payload_bytes():
    comparison = probe.Comparisons()
    original = torch.tensor([0, 0x7FC1], dtype=torch.int16).view(torch.bfloat16)
    same_values_new_nan_payload = torch.tensor([0, 0x7FC2], dtype=torch.int16).view(torch.bfloat16)
    baseline = comparison.observe("raw", {"out": original}, 0)["out"]
    # Both NaNs compare unequal numerically; the equal bit payload is still
    # byte-identical on replay and must not cause a false divergence.
    assert comparison.observe("raw", {"out": original.clone()}, 1)["out"]["bitwise_equal"] is True
    different = comparison.observe("raw", {"out": same_values_new_nan_payload}, 2)["out"]
    assert different["bitwise_equal"] is False
    assert different["sha256"] != baseline["sha256"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU byte comparison requires CUDA")
def test_comparisons_gpu_exact_and_reference_unchanged():
    c = probe.Comparisons()
    a = torch.arange(512, dtype=torch.int32, device="cuda")
    sha = c.observe("gpu", {"idx": a}, 0)["idx"]["sha256"]
    a[0] = -1
    different = c.observe("gpu", {"idx": a}, 1)["idx"]
    assert different["bitwise_equal"] is False
    assert different["max_abs"] == 1.0
    assert c.observe("gpu", {"idx": torch.arange(512, dtype=torch.int32, device="cuda")}, 2)["idx"]["sha256"] == sha


def test_extension_observer_freezes_effective_stats_without_replacing_backend(monkeypatch):
    import types

    calls = []
    extension = types.SimpleNamespace(
        sol_attn_plan=lambda *_args, **_kw: {},
        sol_producer_begin=lambda *args, **kw: calls.append(("begin", args, kw)),
        sol_producer_chunk=lambda *args, **kw: calls.append(("chunk", args, kw)),
        sol_attn_core=lambda *args, **kw: calls.append(("core", args, kw)),
    )
    backend = types.SimpleNamespace(_C=extension, _wrap_for_dlpack=lambda x: x)
    monkeypatch.setattr(probe, "workspace_observations", lambda *_args: {})
    observer = probe.ExtensionObserver(backend, 64, 2, 0, probe.Comparisons(), 0)
    ws = torch.zeros(64, dtype=torch.uint8)
    x = torch.ones(2, 128)
    observer.sol_producer_begin(ws, 1, 64, 2, 0)
    observer.sol_producer_chunk(ws, x, x, x, x, x, x * 2, 1e-5, 96, 0)
    observer.sol_attn_core(ws, x, x * 2)
    assert backend._C is extension
    assert [c[0] for c in calls] == ["begin", "chunk", "core"]
    assert torch.equal(observer.effective_stats[0], x)
    assert torch.equal(observer.effective_stats[1], x * 2)
    ws.fill_(7)
    x.fill_(8)
    assert torch.equal(observer.frozen, torch.zeros(64, dtype=torch.uint8))
    assert float(observer.effective_stats[0].mean()) == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA required; CPU is not determinism evidence")
@pytest.mark.parametrize("case", ["overflow", "flat", "random"])
def test_cuda_frozen_chunked_primitive(case):
    kitchen = pytest.importorskip("comfy_kitchen")
    import importlib.metadata

    if importlib.metadata.version("comfy-kitchen") != "0.2.37":
        pytest.skip("this regression targets exact comfy-kitchen 0.2.37")
    from comfy_kitchen.backends import cuda as backend

    del kitchen
    result = probe.run_case(backend, case, 4196 if case == "random" else 4096, 2, 256, 16, None)
    counts = [r["token_summary"]["overflow_groups"] for r in result["frozen_core"]["runs"]]
    if case == "overflow":
        assert min(counts) > 0, "fixture failed to force saturation; no overflow conclusion is valid"
    if case == "flat":
        assert max(counts) == 0
    assert result["wrapper_first_difference"] is None, result["wrapper_first_difference"]
    assert result["frozen_core"]["first_difference"] is None, result["frozen_core"]["first_difference"]
