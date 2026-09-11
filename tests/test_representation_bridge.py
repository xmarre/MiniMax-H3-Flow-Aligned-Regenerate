from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import h3_flow_regenerate.representation_bridge as bridge
from h3_flow_regenerate.representation_bridge import (
    apply_suffix_representation_bridge,
    disabled_suffix_representation_bridge_metrics,
)


def _learned(dtype=torch.float32):
    torch.manual_seed(91)
    return torch.randn(1, 24, 7, 24, 32, dtype=torch.float32).to(dtype)


def _identity_register(_reference, _moving, *, comparison=None):
    result = {
        "identity_error": 1.0,
        "aligned_error": 1.0,
        "transform": (1.0, 1.0, 0.0, 0.0),
        "reason": "single_transition_diagnostic",
    }
    if comparison is not None:
        result["comparison_error"] = 1.0
    return result


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
def test_stable_overlap_tone_bias_rebases_entire_suffix_without_structural_transplant(monkeypatch, dtype):
    monkeypatch.setattr(bridge, "register_pair", _identity_register)
    learned = _learned(dtype)
    bias = torch.linspace(-0.12, 0.16, 24, dtype=torch.float32).view(1, 24, 1, 1, 1)
    exact = learned[:, :, :4].float() + bias
    exact = exact.to(dtype)
    before = learned.clone()

    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)

    assert report["suffix_representation_bridge_version"] == 4
    assert report["suffix_representation_bridge_mode"] == "persistent_suffix_rebase_v1"
    assert report["suffix_representation_bridge_accepted"] is True
    assert report["suffix_representation_bridge_tone_bias_accepted"] is True
    assert report["suffix_representation_bridge_geometry_accepted"] is False
    assert report["suffix_representation_bridge_raw_structural_residual_transplanted"] is False
    assert report["suffix_representation_bridge_corrected_tokens"] == 3
    assert report["suffix_representation_bridge_tone_validation_dc_improvement"] > 0.99
    assert torch.equal(corrected[:, :, :4], before[:, :, :4])
    expected = before[:, :, 4:].float() + bias
    torch.testing.assert_close(corrected[:, :, 4:].float(), expected, rtol=0, atol=3e-3)


def test_unstable_overlap_tone_bias_is_rejected(monkeypatch):
    monkeypatch.setattr(bridge, "register_pair", _identity_register)
    learned = _learned()
    exact = learned[:, :, :4].clone()
    exact[:, :, :3] += 0.10
    exact[:, :, 3:4] -= 0.10

    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)

    assert corrected is learned
    assert report["suffix_representation_bridge_accepted"] is False
    assert report["suffix_representation_bridge_tone_bias_accepted"] is False
    assert report["suffix_representation_bridge_tone_reason"] in {
        "heldout_tone_gain_too_small",
        "heldout_tone_total_regression",
    }


def test_geometry_rebase_uses_recent_learned_motion_and_validates_internal_suffix(monkeypatch):
    plain_calls = 0
    comparison_calls = 0

    def measured_register(_reference, _moving, *, comparison=None):
        nonlocal plain_calls, comparison_calls
        if comparison is None:
            plain_calls += 1
            return {
                "identity_error": 0.4,
                "aligned_error": 0.2,
                "transform": (1.0, 1.0, 0.0, 0.0),
                "reason": "single_transition_diagnostic",
            }
        comparison_calls += 1
        if comparison_calls == 1:
            return {
                "identity_error": 1.0,
                "aligned_error": 0.25,
                "comparison_error": 0.8,
                "transform": (1.0, 1.0, 0.5, 0.0),
                "reason": "single_transition_diagnostic",
            }
        return {
            "identity_error": 1.0,
            "aligned_error": 0.2,
            "comparison_error": 0.4,
            "transform": (1.0, 1.0, 0.0, 0.0),
            "reason": "single_transition_diagnostic",
        }

    monkeypatch.setattr(bridge, "register_pair", measured_register)
    learned = _learned()
    correction, report = bridge._persistent_geometry_rebase(learned, 4)

    assert correction == pytest.approx((1.0, 1.0, 0.5, 0.0))
    assert report["accepted"] is True
    assert report["reason"] == "persistent_motion_frame_rebase_authorized"
    assert report["improvement"] == pytest.approx(0.5)
    assert report["validation_transitions"] == 2
    assert report["worst_internal_regression"] == pytest.approx(0.0)
    assert plain_calls == 7
    assert comparison_calls == 2


def test_authorized_geometry_rebase_warps_the_complete_suffix(monkeypatch):
    learned = _learned()
    exact = learned[:, :, :4].clone()
    transform = (1.0, 1.0, 0.5, -0.25)

    monkeypatch.setattr(
        bridge,
        "_persistent_tone_bias",
        lambda *_args, **_kwargs: (
            None,
            {
                "accepted": False,
                "reason": "overlap_tone_already_matched",
                "overlap_frames": 4,
                "bias_rms": 0.0,
                "validation_rms_before": 0.0,
                "validation_rms_after": 0.0,
                "validation_dc_rms_before": 0.0,
                "validation_dc_rms_after": 0.0,
                "validation_dc_improvement": 0.0,
            },
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_persistent_geometry_rebase",
        lambda *_args, **_kwargs: (
            transform,
            {
                "accepted": True,
                "reason": "persistent_motion_frame_rebase_authorized",
                "expected_transform": (1.0, 1.0, 0.0, 0.0),
                "observed_transform": transform,
                "residual_transform": transform,
                "boundary_error_before": 0.8,
                "boundary_error_after": 0.4,
                "improvement": 0.5,
                "validation_transitions": 2,
                "improving_transitions": 2,
                "median_transition_improvement": 0.0,
                "worst_internal_regression": 0.0,
            },
        ),
    )

    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)

    assert report["suffix_representation_bridge_geometry_accepted"] is True
    assert report["suffix_representation_bridge_corrected_tokens"] == 3
    assert torch.equal(corrected[:, :, :4], learned[:, :, :4])
    expected = _warp_suffix(learned[:, :, 4:], transform)
    torch.testing.assert_close(corrected[:, :, 4:], expected, rtol=0, atol=2e-5)


def test_large_raw_structural_overlap_is_not_transplanted_when_rebases_reject(monkeypatch):
    monkeypatch.setattr(bridge, "register_pair", _identity_register)
    learned = _learned()
    exact = learned[:, :, :4].clone()
    yy = torch.linspace(-1, 1, 24).view(1, 1, 1, 24, 1)
    xx = torch.linspace(-1, 1, 32).view(1, 1, 1, 1, 32)
    exact += 0.08 * yy - 0.05 * xx

    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)

    assert corrected is learned
    assert report["suffix_representation_bridge_accepted"] is False
    assert report["suffix_representation_bridge_raw_structural_residual_transplanted"] is False
    assert report["suffix_representation_bridge_structural_rms"] > 0.01


def test_disabled_path_is_exact_object_noop():
    learned = _learned()
    exact = learned[:, :, :4].clone()
    disabled, report = apply_suffix_representation_bridge(learned, exact, requested=False)
    assert disabled is learned
    assert report == disabled_suffix_representation_bridge_metrics(prefix_t=4)


def test_nonfinite_input_is_rejected():
    learned = _learned()
    exact = learned[:, :, :4].clone()
    learned = learned.clone()
    learned[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        apply_suffix_representation_bridge(learned, exact)
