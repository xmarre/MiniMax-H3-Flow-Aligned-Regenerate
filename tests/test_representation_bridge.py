from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.representation_bridge import (
    apply_suffix_representation_bridge,
    disabled_suffix_representation_bridge_metrics,
)
from h3_flow_regenerate.tone_bridge import apply_suffix_dc_bridge


def _pair(dtype=torch.float32):
    torch.manual_seed(91)
    learned = torch.randn(1, 24, 5, 8, 10, dtype=torch.float32)
    exact = learned[:, :, :2].clone()
    yy = torch.linspace(-1, 1, 8).view(1, 1, 1, 8, 1)
    xx = torch.linspace(-1, 1, 10).view(1, 1, 1, 1, 10)
    exact = exact + 0.12 + 0.08 * yy - 0.05 * xx
    return learned.to(dtype), exact.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_structure_plus_dc_preserves_upscaler_native_boundary_transition(dtype):
    learned, exact = _pair(dtype)
    before = learned.clone()
    structured, report = apply_suffix_representation_bridge(learned, exact, requested=True)
    assert report["suffix_representation_bridge_accepted"] is True
    assert report["suffix_representation_bridge_corrected_tokens"] == 1
    assert torch.equal(structured[:, :, :2], before[:, :, :2])
    assert torch.equal(structured[:, :, 3:], before[:, :, 3:])

    combined, dc = apply_suffix_dc_bridge(structured, exact, weights=(1.0,))
    assert dc["suffix_dc_bridge_corrected_tokens"] == 1
    native = before[:, :, 2].float() - before[:, :, 1].float()
    restored = combined[:, :, 2].float() - exact[:, :, 1].float()

    # The algebra is exact in float32. For half/bfloat16 the structural and DC
    # corrections are each rounded back to the required output dtype. Bound that
    # unavoidable error by two output-ULPs at the largest participating value;
    # this checks the implementation contract rather than using a hand-tuned
    # tolerance that merely happens to pass this fixture.
    if dtype == torch.float32:
        assert torch.allclose(restored, native, atol=2e-6, rtol=0)
    else:
        scale = torch.maximum(restored.abs(), native.abs()).clamp_min(1.0)
        quantization_bound = 2.0 * torch.finfo(dtype).eps * scale
        assert bool(((restored - native).abs() <= quantization_bound).all())

    assert torch.equal(combined[:, :, :2], before[:, :, :2])
    assert torch.equal(combined[:, :, 3:], before[:, :, 3:])


def test_structure_bridge_transfers_zero_mean_residual_only():
    learned, exact = _pair()
    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)
    change = corrected[:, :, 2].float() - learned[:, :, 2].float()
    assert torch.allclose(change.mean(dim=(-2, -1)), torch.zeros_like(change.mean(dim=(-2, -1))), atol=2e-7)
    assert (
        report["suffix_representation_bridge_centered_error_after"]
        < report["suffix_representation_bridge_centered_error_before"]
    )
    assert report["suffix_representation_bridge_centered_error_ratio"] < 1e-4


def test_bridge_has_no_unmeasured_temporal_fade():
    learned, exact = _pair()
    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)
    assert report["suffix_representation_bridge_corrected_tokens"] == 1
    assert not torch.equal(corrected[:, :, 2], learned[:, :, 2])
    assert torch.equal(corrected[:, :, 3:], learned[:, :, 3:])


def test_disabled_and_already_matched_paths_are_exact_object_noops():
    learned, exact = _pair()
    disabled, report = apply_suffix_representation_bridge(learned, exact, requested=False)
    assert disabled is learned
    assert report == disabled_suffix_representation_bridge_metrics(prefix_t=2)

    exact_match = learned[:, :, :2].clone()
    matched, report = apply_suffix_representation_bridge(learned, exact_match, requested=True)
    assert matched is learned
    assert report["suffix_representation_bridge_accepted"] is False
    assert report["suffix_representation_bridge_reason"] == "structural_overlap_already_matched"


def test_nonfinite_input_is_rejected():
    learned, exact = _pair()
    learned = learned.clone()
    learned[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        apply_suffix_representation_bridge(learned, exact)
