from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import h3_flow_regenerate.representation_bridge as bridge
from h3_flow_regenerate.representation_bridge import (
    apply_suffix_representation_bridge,
    disabled_suffix_representation_bridge_metrics,
)


def _pair(dtype=torch.float32):
    torch.manual_seed(91)
    learned = torch.randn(1, 24, 7, 24, 32, dtype=torch.float32)
    exact = learned[:, :, :4].clone()
    yy = torch.linspace(-1, 1, 24).view(1, 1, 1, 24, 1)
    xx = torch.linspace(-1, 1, 32).view(1, 1, 1, 1, 32)
    # Large local/chroma-like residual: v2 must never copy this difference field
    # onto a generated token merely to make the latent boundary algebra close.
    exact = exact + 0.12 + 0.08 * yy - 0.05 * xx
    return learned.to(dtype), exact.to(dtype)


def _register_translation(_reference, _moving, *, comparison=None):
    transform = (1.0, 1.0, 0.5, -0.25)
    if comparison is None:
        return {
            "identity_error": 1.0,
            "aligned_error": 0.4,
            "transform": transform,
            "reason": "single_transition_diagnostic",
        }
    return {
        "identity_error": 1.0,
        "aligned_error": 0.35,
        "comparison_error": 0.4,
        "transform": transform,
        "reason": "single_transition_diagnostic",
    }


def _warp_suffix(value: torch.Tensor, transform):
    b, c, t, h, w = value.shape
    flat = value.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    theta = torch.zeros((b * t, 2, 3), dtype=torch.float32)
    theta[:, 0, 0] = transform[0]
    theta[:, 0, 2] = 2.0 * transform[2] / w
    theta[:, 1, 1] = transform[1]
    theta[:, 1, 2] = 2.0 * transform[3] / h
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    warped = F.grid_sample(flat.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(value.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_authorized_overlap_geometry_warps_entire_suffix_without_residual_transplant(monkeypatch, dtype):
    monkeypatch.setattr(bridge, "register_pair", _register_translation)
    learned, exact = _pair(dtype)
    before = learned.clone()

    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)

    assert report["suffix_representation_bridge_version"] == 2
    assert report["suffix_representation_bridge_mode"] == "constant_overlap_geometry_v1"
    assert report["suffix_representation_bridge_accepted"] is True
    assert report["suffix_representation_bridge_raw_structural_residual_transplanted"] is False
    assert report["suffix_representation_bridge_corrected_tokens"] == 3
    assert report["suffix_representation_bridge_transform"] == pytest.approx((1.0, 1.0, 0.5, -0.25))
    assert torch.equal(corrected[:, :, :4], before[:, :, :4])
    expected = _warp_suffix(before[:, :, 4:], (1.0, 1.0, 0.5, -0.25))
    torch.testing.assert_close(corrected[:, :, 4:].float(), expected.float(), rtol=0, atol=2e-3)

    raw_structural = exact[:, :, -1:].float() - before[:, :, 3:4].float()
    raw_structural -= raw_structural.mean(dim=(-2, -1), keepdim=True)
    first_change = corrected[:, :, 4:5].float() - before[:, :, 4:5].float()
    assert not torch.allclose(first_change, raw_structural, rtol=0, atol=1e-3)


def test_large_raw_overlap_residual_is_noop_without_safe_geometry(monkeypatch):
    def identity_registration(_reference, _moving, *, comparison=None):
        base = {
            "identity_error": 1.0,
            "aligned_error": 1.0,
            "transform": (1.0, 1.0, 0.0, 0.0),
            "reason": "single_transition_diagnostic",
        }
        if comparison is not None:
            base["comparison_error"] = 1.0
        return base

    monkeypatch.setattr(bridge, "register_pair", identity_registration)
    learned, exact = _pair()
    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)
    assert corrected is learned
    assert report["suffix_representation_bridge_accepted"] is False
    assert report["suffix_representation_bridge_raw_structural_residual_transplanted"] is False
    assert report["suffix_representation_bridge_structural_rms"] > 0.01


def test_inconsistent_overlap_geometry_is_rejected(monkeypatch):
    calls = {"index": 0}

    def alternating(_reference, _moving, *, comparison=None):
        if comparison is not None:
            return {
                "identity_error": 1.0,
                "aligned_error": 0.4,
                "comparison_error": 0.4,
                "transform": (1.0, 1.0, 0.5, 0.0),
                "reason": "single_transition_diagnostic",
            }
        calls["index"] += 1
        tx = 0.5 if calls["index"] % 2 else -0.5
        return {
            "identity_error": 1.0,
            "aligned_error": 0.4,
            "transform": (1.0, 1.0, tx, 0.0),
            "reason": "single_transition_diagnostic",
        }

    monkeypatch.setattr(bridge, "register_pair", alternating)
    learned, exact = _pair()
    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)
    assert corrected is learned
    assert report["suffix_representation_bridge_accepted"] is False


def test_disabled_path_is_exact_object_noop():
    learned, exact = _pair()
    disabled, report = apply_suffix_representation_bridge(learned, exact, requested=False)
    assert disabled is learned
    assert report == disabled_suffix_representation_bridge_metrics(prefix_t=4)


def test_nonfinite_input_is_rejected():
    learned, exact = _pair()
    learned = learned.clone()
    learned[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        apply_suffix_representation_bridge(learned, exact)
