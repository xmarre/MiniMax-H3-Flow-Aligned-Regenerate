from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from h3_flow_regenerate.source_trajectory_bridge import (
    _effective_transform,
    _temporal_axis_state,
    apply_source_trajectory_bridge,
)


def _texture() -> torch.Tensor:
    torch.manual_seed(77)
    frame = torch.randn(1, 24, 32, 40)
    return F.avg_pool2d(F.pad(frame, (2, 2, 2, 2), mode="reflect"), 5, stride=1)


def _warp(frame: torch.Tensor, *, sy: float) -> torch.Tensor:
    b, _c, _h, _w = frame.shape
    theta = frame.new_tensor([[1.0, 0.0, 0.0], [0.0, sy, 0.0]], dtype=torch.float32)
    grid = F.affine_grid(theta[None].expand(b, -1, -1), frame.shape, align_corners=False)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _recovering_video() -> torch.Tensor:
    frame = _texture()
    prefix = [frame.clone() for _ in range(6)]
    # Registration of the first suffix against an identity-motion prefix is ~0.9775 sy.
    first_suffix = _warp(frame, sy=1.0 / 0.975)
    return torch.stack([*prefix, first_suffix, frame.clone(), frame.clone()], dim=2)


def test_recovering_source_sy_is_closed_before_upscale_and_tail_is_exact():
    video = _recovering_video()
    before = video.clone()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=True)

    assert report["source_trajectory_bridge_accepted"] is True
    assert report["source_trajectory_bridge_reason"] == "source_trajectory_residual_closed_before_upscale"
    assert report["source_trajectory_axis_authorized"] == [False, True, False, False]
    assert report["source_trajectory_temporal_profile"]["axis_mode"][1] == "recovering"
    assert report["source_trajectory_temporal_profile"]["axis"][1]["source_evidence_gate"] == "diagnostic_only"
    assert report["source_trajectory_bridge_tokens_corrected"] == 1
    assert report["source_trajectory_residual_reduction_ratio"][1] < 1.0
    assert torch.equal(corrected[:, :, :6], before[:, :, :6])
    assert not torch.equal(corrected[:, :, 6:7], before[:, :, 6:7])
    assert torch.equal(corrected[:, :, 7:], before[:, :, 7:])
    assert torch.equal(video, before)


def test_disabled_bridge_is_exact_object_noop():
    video = _recovering_video()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=False)
    assert corrected is video
    assert report["source_trajectory_bridge_accepted"] is False
    assert report["source_trajectory_bridge_reason"] == "disabled"


@pytest.mark.parametrize("prefix_t", [1, 2, 8])
def test_insufficient_motion_window_fails_safe(prefix_t):
    video = torch.randn(1, 24, 9, 32, 40)
    corrected, report = apply_source_trajectory_bridge(video, prefix_t, requested=True)
    assert corrected is video
    assert report["source_trajectory_bridge_accepted"] is False
    assert report["source_trajectory_bridge_reason"] == "insufficient_source_motion_window"


def test_estimator_floor_residual_can_be_temporally_authorized_without_cross_domain_gate():
    # Mirrors the 00288 ty evidence. The boundary residual is exactly one
    # estimator floor and the first observed follow-up does not move farther
    # away, so the state is already inside the unresolved/recovered band.
    # Correct only token 0; do not project a persistent correction.
    report = _temporal_axis_state(
        0.25,
        [0.0, 0.0, -0.125, 0.125],
        estimator_floor=0.25,
        suffix_length=8,
    )
    assert report["accepted"] is True
    assert report["mode"] == "recovering"
    assert report["recovery_suffix_token"] == 1
    assert report["active_tokens"] == 1
    assert report["applied_envelope"]["kind"] == "measured_monotonic"
    assert report["applied_envelope"]["weights"] == [1.0, 0.0]
    assert report["applied_envelope"]["terminal_zero_observed"] is True


def test_same_direction_source_drift_uses_measured_cumulative_state_only():
    report = _temporal_axis_state(
        0.010161524669189069,
        [0.007677082358897666, 0.012664654887307421, 0.020149107614304457, 0.01762725812058288],
        estimator_floor=0.005,
        suffix_length=8,
    )
    assert report["accepted"] is True
    assert report["mode"] == "measured_drift"
    assert report["active_tokens"] == 5
    envelope = report["applied_envelope"]
    assert envelope["kind"] == "measured_cumulative"
    assert envelope["measured_followup_transitions"] == 4
    assert envelope["extrapolated_beyond_observation"] is False
    assert envelope["signed_states"] == pytest.approx(report["cumulative_signed_state"])


def test_measured_cumulative_transform_uses_observed_state_not_scaled_boundary():
    axis_reports = [
        {
            "accepted": True,
            "mode": "measured_drift",
            "applied_envelope": {
                "kind": "measured_cumulative",
                "signed_states": [0.01, 0.03, 0.06],
                "active_tokens": 3,
            },
        },
        {"accepted": False},
        {"accepted": False},
        {"accepted": False},
    ]
    transform = _effective_transform((1.01, 1.0, 0.0, 0.0), axis_reports, 2)
    assert transform[0] == pytest.approx(torch.exp(torch.tensor(0.06)).item(), rel=1e-6)
    assert transform[1:] == pytest.approx((1.0, 0.0, 0.0))


def test_persistent_source_state_is_limited_to_measured_window():
    frame = _texture()
    prefix = [frame.clone() for _ in range(6)]
    shifted = _warp(frame, sy=1.0 / 0.975)
    suffix = [shifted.clone() for _ in range(7)]
    video = torch.stack([*prefix, *suffix], dim=2)
    before = video.clone()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=True)
    assert report["source_trajectory_bridge_accepted"] is True
    assert report["source_trajectory_temporal_profile"]["axis_mode"][1] == "persistent"
    envelope = report["source_trajectory_temporal_profile"]["axis"][1]["applied_envelope"]
    assert envelope["measured_followup_transitions"] == 4
    assert envelope["extrapolated_beyond_observation"] is False
    assert report["source_trajectory_bridge_tokens_corrected"] == 5
    assert not torch.equal(corrected[:, :, 6:11], before[:, :, 6:11])
    assert torch.equal(corrected[:, :, 11:], before[:, :, 11:])
