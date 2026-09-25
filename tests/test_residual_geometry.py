from __future__ import annotations

import math

import pytest
import torch

from h3_flow_regenerate.partitioned_scheduler import _frame_gauge_clean_postprocess
from h3_flow_regenerate.residual_geometry import (
    RESIDUAL_GEOMETRY_POLICY_VERSION,
    measure_residual_geometry,
)


def _analytic_residual_pair(
    *,
    a: float = 0.0,
    b: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
    frames: int = 4,
    height: int = 56,
    width: int = 74,
    per_frame_a: tuple[float, ...] | None = None,
    vertical_scale: float = 0.0,
    shear: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    exact_frames = []
    learned_frames = []
    for frame in range(frames):
        frame_a = float(per_frame_a[frame]) if per_frame_a is not None else float(a)
        exact_channels = []
        learned_channels = []
        temporal_pan = 0.17 * frame
        for channel in range(24):
            phase = 0.071 * channel + 0.037 * frame

            def field(y, x, phase=phase):
                return (
                    torch.sin(0.29 * x + phase)
                    + 0.71 * torch.cos(0.23 * y - 0.7 * phase)
                    + 0.33 * torch.sin(0.11 * (x + 1.7 * y) + 1.3 * phase)
                    + 0.17 * torch.cos(0.19 * (x - 1.4 * y) - phase)
                    + 0.0027 * (x - cx) * (y - cy)
                )

            exact_channels.append(field(yy, xx + temporal_pan))

            # The estimator samples learned at q=r-d-u(r).  Generate L(q)
            # analytically by evaluating the exact field at the inverse of that
            # pullback map, rather than by warping a discrete tensor.
            denom_x = 1.0 - frame_a
            source_x = (xx - frame_a * cx + float(b) + float(dx) - float(shear) * (yy - cy)) / denom_x
            denom_y = 1.0 - float(vertical_scale)
            source_y = (yy - float(vertical_scale) * cy + float(dy)) / denom_y
            learned_channels.append(field(source_y, source_x + temporal_pan))
        exact_frames.append(torch.stack(exact_channels))
        learned_frames.append(torch.stack(learned_channels))
    exact = torch.stack(exact_frames, dim=1).unsqueeze(0).float()
    learned = torch.stack(learned_frames, dim=1).unsqueeze(0).float()
    return learned, exact


def test_measurement_recovers_bounded_horizontal_residual_without_rng_use():
    learned, exact = _analytic_residual_pair(a=0.006, b=0.03125, dx=0.5, dy=-0.25)
    rng_before = torch.random.get_rng_state().clone()

    receipt = measure_residual_geometry(
        learned,
        exact,
        rigid_dx=0.5,
        rigid_dy=-0.25,
    )

    assert receipt["policy"] == RESIDUAL_GEOMETRY_POLICY_VERSION
    assert receipt["status"] == "measured"
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert receipt["counts"]["observations"] == 36
    assert receipt["counts"]["search_scores"] <= 36 * 162
    horizontal = receipt["models"]["horizontal"]
    assert horizontal["status"] == "accepted", receipt
    assert horizontal["a"] == pytest.approx(0.006, abs=0.003)
    assert horizontal["b"] == pytest.approx(0.03125, abs=0.08)
    assert receipt["models"]["diagnostic_fits"]["local_projective"]["status"] == "not_implemented"


def test_identity_residual_is_not_misclassified_as_horizontal_evidence():
    learned, exact = _analytic_residual_pair()

    receipt = measure_residual_geometry(
        learned,
        exact,
        rigid_dx=0.0,
        rigid_dy=0.0,
    )

    assert receipt["status"] == "measured"
    assert receipt["counts"]["identity"] > 0
    assert receipt["models"]["eligible"] is False
    assert "zero_slope_in_envelope" in receipt["models"].get("eligibility_failures", []) or (
        receipt["models"]["horizontal"]["status"] != "accepted"
    )


def test_vertical_scale_is_diagnostic_but_cannot_authorize_horizontal_model():
    learned, exact = _analytic_residual_pair(
        dx=0.375,
        dy=-0.25,
        vertical_scale=0.006,
    )

    receipt = measure_residual_geometry(
        learned,
        exact,
        rigid_dx=0.375,
        rigid_dy=-0.25,
    )

    axis = receipt["models"]["diagnostic_fits"]["axis_scales"]
    assert axis["status"] == "accepted", receipt
    assert abs(axis["e"]) > 0.002
    assert receipt["models"]["eligible"] is False
    assert "cross_axis_residual" in receipt["models"]["eligibility_failures"]


def test_shear_is_reported_by_affine_diagnostic_and_horizontal_rejects():
    learned, exact = _analytic_residual_pair(
        dx=0.375,
        dy=0.0,
        shear=0.006,
    )

    receipt = measure_residual_geometry(
        learned,
        exact,
        rigid_dx=0.375,
        rigid_dy=0.0,
    )

    affine = receipt["models"]["diagnostic_fits"]["affine"]
    assert affine["status"] == "accepted", receipt
    assert abs(affine["h"]) > 0.002
    assert receipt["models"]["eligible"] is False


def test_temporal_slope_sign_reversal_rejects_horizontal_eligibility():
    per_frame_a = (0.006, 0.006, -0.006, -0.006)
    learned, exact = _analytic_residual_pair(
        dx=0.5,
        dy=0.0,
        per_frame_a=per_frame_a,
    )

    receipt = measure_residual_geometry(
        learned,
        exact,
        rigid_dx=0.5,
        rigid_dy=0.0,
    )

    assert receipt["models"]["eligible"] is False
    failures = receipt["models"]["eligibility_failures"]
    assert "holdout_slope_sign" in failures or "holdout_improvement" in failures


def test_low_texture_and_repeated_structure_fail_closed():
    constant = torch.ones(1, 24, 4, 56, 74)
    receipt = measure_residual_geometry(
        constant,
        constant,
        rigid_dx=0.0,
        rigid_dy=0.0,
    )
    assert receipt["status"] == "rejected"
    assert receipt["eligible"] is False

    x = torch.arange(74, dtype=torch.float32)
    stripe = torch.cos(math.pi * x).view(1, 1, 1, 1, 74)
    exact = stripe.expand(1, 24, 4, 56, 74).clone()
    learned = -exact
    receipt = measure_residual_geometry(
        learned,
        exact,
        rigid_dx=0.0,
        rigid_dy=0.0,
    )
    assert receipt["status"] in {"measured", "rejected"}
    if receipt["status"] == "measured":
        assert receipt["models"]["eligible"] is False


def test_measurement_only_frame_gauge_is_byte_identical_to_rigid_v2():
    _prefix_learned, exact_prefix = _analytic_residual_pair(
        dx=0.5,
        dy=-0.25,
        frames=6,
        height=56,
        width=74,
    )
    # Extend the learned provider state with four analytically consistent suffix
    # frames.  Exact ownership remains only the six-frame prefix.
    learned_full, _ = _analytic_residual_pair(
        dx=0.5,
        dy=-0.25,
        frames=10,
        height=56,
        width=74,
    )
    learned_before = learned_full.clone()
    common = dict(
        exact_prefix=exact_prefix,
        guidance_run=None,
        guidance=None,
        target_h=56,
        target_w=74,
        prefix_t=6,
        split_coordinate=0.35,
        high_sigmas=torch.tensor([0.35, 0.0]),
        video_shift=5.0,
    )

    rigid, rigid_guidance, _, rigid_txn = _frame_gauge_clean_postprocess(
        learned_full,
        residual_mode="off",
        **common,
    )
    measured, measured_guidance, _, measured_txn = _frame_gauge_clean_postprocess(
        learned_full,
        residual_mode="measure",
        **common,
    )

    assert torch.equal(learned_full, learned_before)
    assert rigid_txn["result"] == measured_txn["result"] == "accepted"
    assert rigid_guidance is measured_guidance is None
    assert torch.equal(rigid.clean_video, measured.clean_video)
    assert rigid_txn["dc_metrics"] == measured_txn["dc_metrics"]
    assert measured_txn["residual_geometry"]["decision"] == "not_evaluated"
    assert measured_txn["residual_geometry"]["applied"] is False
    assert measured_txn["residual_geometry"]["final_path"] == "rigid_v2"
