from __future__ import annotations

import pytest
import torch

import h3_flow_regenerate.post_high_video as post_high_video
from h3_flow_regenerate.post_high_video import (
    POST_HIGH_VIDEO_RELEASE_WEIGHTS,
    apply_weighted_vertical_translation,
    plan_post_high_vertical_residual,
    repair_post_high_vertical_residual,
)


def _trajectory(*, first_dx, first_dy, pre_dx, pre_dy, response=12.0):
    return {
        "pairwise_dx": [float(first_dx), 0.0, 0.0, 0.0, 0.0],
        "pairwise_dy": [float(first_dy), 0.0, 0.0, 0.0, 0.0],
        "pairwise_response": [float(response), 12.0, 12.0, 12.0, 12.0],
        "pairwise_clipped": [False, False, False, False, False],
        "pre_pairwise_median_dx": float(pre_dx),
        "pre_pairwise_median_dy": float(pre_dy),
    }


def _00712_receipts():
    pre_high = {
        "upper45": _trajectory(
            first_dx=-0.0075787891,
            first_dy=0.0555063963,
            pre_dx=-0.0076364889,
            pre_dy=0.0153863830,
            response=25.48,
        ),
        "full": _trajectory(
            first_dx=-0.0227513474,
            first_dy=-0.0023144214,
            pre_dx=-0.0482336879,
            pre_dy=-0.0135173397,
            response=29.59,
        ),
    }
    post_high = {
        "upper45": _trajectory(
            first_dx=0.0745397818,
            first_dy=0.0981616589,
            pre_dx=-0.0076364889,
            pre_dy=0.0153863830,
            response=11.26,
        ),
        "full": _trajectory(
            first_dx=-0.0388245862,
            first_dy=0.1836336335,
            pre_dx=-0.0482336879,
            pre_dy=-0.0135173397,
            response=10.19,
        ),
    }
    return pre_high, post_high


def test_00712_receipts_select_bounded_vertical_only_plan():
    pre_high, post_high = _00712_receipts()
    plan = plan_post_high_vertical_residual(pre_high, post_high)

    assert plan["eligible"] is True
    assert plan["reason"] == "coherent_target_high_vertical_residual"
    assert plan["horizontal_application_enabled"] is False
    assert plan["correction_magnitude_cells"] == pytest.approx(0.1143016587, abs=1e-6)
    assert plan["max_induced_step_cells"] < 0.0625


def test_vertical_release_preserves_prefix_and_suffix_outside_support():
    torch.manual_seed(123)
    video = torch.randn(1, 4, 10, 12, 12)
    prefix_t = 3
    corrected = apply_weighted_vertical_translation(video, prefix_t=prefix_t, dy=-0.14)

    assert torch.equal(corrected[:, :, :prefix_t], video[:, :, :prefix_t])
    support_stop = prefix_t + len(POST_HIGH_VIDEO_RELEASE_WEIGHTS)
    assert not torch.equal(corrected[:, :, prefix_t:support_stop], video[:, :, prefix_t:support_stop])
    assert torch.equal(corrected[:, :, support_stop:], video[:, :, support_stop:])


def test_plan_fails_closed_when_target_high_vertical_delta_disagrees_between_rois():
    pre_high, post_high = _00712_receipts()
    post_high["upper45"]["pairwise_dy"][0] = 0.02
    plan = plan_post_high_vertical_residual(pre_high, post_high)

    assert plan["eligible"] is False
    assert plan["reason"] in {
        "vertical_boundary_error_not_coherent",
        "target_high_vertical_delta_not_coherent",
        "boundary_error_not_explained_by_target_high_delta",
    }


def test_repair_trials_both_signs_and_accepts_only_measured_improvement(monkeypatch):
    pre_high, post_high = _00712_receipts()
    improved = {
        "upper45": _trajectory(
            first_dx=0.074,
            first_dy=0.018,
            pre_dx=-0.0076364889,
            pre_dy=0.0153863830,
            response=11.0,
        ),
        "full": _trajectory(
            first_dx=-0.039,
            first_dy=-0.006,
            pre_dx=-0.0482336879,
            pre_dy=-0.0135173397,
            response=10.0,
        ),
    }
    worsened = {
        "upper45": _trajectory(
            first_dx=0.074,
            first_dy=0.24,
            pre_dx=-0.0076364889,
            pre_dy=0.0153863830,
            response=11.0,
        ),
        "full": _trajectory(
            first_dx=-0.039,
            first_dy=0.32,
            pre_dx=-0.0482336879,
            pre_dy=-0.0135173397,
            response=10.0,
        ),
    }
    receipts = iter((post_high, improved, worsened))
    monkeypatch.setattr(
        post_high_video,
        "measure_post_high_video_trajectory",
        lambda _video, _prefix_t: next(receipts),
    )

    torch.manual_seed(42)
    video = torch.randn(1, 4, 10, 12, 12)
    corrected, receipt = repair_post_high_vertical_residual(pre_high, video, prefix_t=3)

    assert receipt["applied"] is True
    assert receipt["accepted"] is True
    assert receipt["selected_dy_cells"] < 0.0
    assert receipt["selected_score"]["mean_improvement_ratio"] > 0.25
    assert receipt["selected_score"]["temporal_release_stable"] is True
    assert torch.equal(corrected[:, :, :3], video[:, :, :3])
    assert not torch.equal(corrected[:, :, 3:7], video[:, :, 3:7])
