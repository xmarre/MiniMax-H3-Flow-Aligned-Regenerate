from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

import h3_flow_regenerate.source_trajectory_bridge as source_bridge
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


def test_measured_drift_truncates_at_existing_axis_safety_bound():
    report = _temporal_axis_state(
        0.026772269063550715,
        [0.022834929080955513, 0.038891567782737106, 0.04992944577483918, 0.0459608545226586],
        estimator_floor=0.005,
        suffix_length=8,
        safety_limit=float(torch.log(torch.tensor(1.03))),
    )
    assert report["accepted"] is True
    assert report["mode"] == "measured_drift"
    assert report["reason"] == "observed_same_direction_cumulative_drift_safety_limited"
    assert report["active_tokens"] == 1
    envelope = report["applied_envelope"]
    assert envelope["signed_states"] == pytest.approx([0.026772269063550715])
    assert envelope["measured_tokens_available"] == 5
    assert envelope["safety_limited"] is True
    assert envelope["extrapolated_beyond_observation"] is False


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


def test_persistent_source_state_without_observed_recovery_fails_safe():
    frame = _texture()
    prefix = [frame.clone() for _ in range(6)]
    shifted = _warp(frame, sy=1.0 / 0.975)
    suffix = [shifted.clone() for _ in range(7)]
    video = torch.stack([*prefix, *suffix], dim=2)
    before = video.clone()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=True)
    assert report["source_trajectory_bridge_accepted"] is False
    assert report["source_trajectory_temporal_profile"]["axis_mode"][1] == "persistent"
    axis = report["source_trajectory_temporal_profile"]["axis"][1]
    assert axis["reason"] == "persistent_state_without_observed_recovery"
    assert axis["applied_envelope"]["measured_followup_transitions"] == 4
    assert axis["applied_envelope"]["extrapolated_beyond_observation"] is False
    assert corrected is video
    assert torch.equal(corrected, before)


def test_persistent_source_state_observed_through_suffix_end_can_apply():
    frame = _texture()
    prefix = [frame.clone() for _ in range(6)]
    shifted = _warp(frame, sy=1.0 / 0.975)
    suffix = [shifted.clone() for _ in range(5)]
    video = torch.stack([*prefix, *suffix], dim=2)
    before = video.clone()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=True)
    assert report["source_trajectory_bridge_accepted"] is True
    assert report["source_trajectory_temporal_profile"]["axis_mode"][1] == "persistent"
    axis = report["source_trajectory_temporal_profile"]["axis"][1]
    assert axis["reason"] == "persistent_state_observed_through_suffix_end"
    assert report["source_trajectory_bridge_tokens_corrected"] == 5
    assert not torch.equal(corrected[:, :, 6:], before[:, :, 6:])


def test_one_token_axis_is_verified_at_its_trailing_transition(monkeypatch):
    video = torch.zeros(1, 24, 6, 8, 8)
    corrected = video.clone()
    motion = {"accepted": True, "expected_transform": (1.0, 1.0, 0.0, 0.0)}
    candidate = {
        "raw_signed_residual": (0.02, 0.0, 0.0, 0.0),
        "transform": (float(torch.exp(torch.tensor(0.02))), 1.0, 0.0, 0.0),
        "axis_estimator_floor": (0.005, 0.005, 0.25, 0.25),
    }
    profile = {
        "axis": [
            {
                "accepted": True,
                "mode": "measured_drift",
                "active_tokens": 1,
                "applied_envelope": {
                    "kind": "measured_cumulative",
                    "signed_states": [0.02],
                    "active_tokens": 1,
                },
            },
            {"accepted": False},
            {"accepted": False},
            {"accepted": False},
        ],
        "transitions": [{"motion_residual_signed": (0.01, 0.0, 0.0, 0.0)}],
    }
    results = iter(
        [
            {"aligned_error": 0.1, "transform": (float(torch.exp(torch.tensor(0.01))), 1.0, 0.0, 0.0)},
            {"aligned_error": 0.1, "transform": (float(torch.exp(torch.tensor(0.02))), 1.0, 0.0, 0.0)},
        ]
    )
    monkeypatch.setattr(source_bridge, "register_pair", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(source_bridge, "_predict_motion_transform", lambda *args, **kwargs: (1.0, 1.0, 0.0, 0.0))

    report = source_bridge._verify_source_warp(
        video,
        corrected,
        3,
        motion,
        candidate,
        profile,
        [True, False, False, False],
        {"tokens_corrected": 1},
    )
    assert report["ok"] is False
    assert report["axis_verified"] == [False, False, False, False]
    assert len(report["transition_verification"]) == 1
    transition = report["transition_verification"][0]
    assert transition["suffix_transition_offset"] == 1
    assert transition["axis_reduction_ratio"][0] == pytest.approx(2.0, rel=1e-4)
    assert transition["axis_verified"][0] is False


def test_failed_axis_is_pruned_and_surviving_axis_is_retried(monkeypatch):
    video = torch.zeros(1, 24, 7, 8, 8)
    prefix_t = 3
    motion = {
        "accepted": True,
        "reason": "robust_recent_motion",
        "expected_transform": (1.0, 1.0, 0.0, 0.0),
    }
    boundary = {"aligned_error": 0.1, "transform": (1.0, 1.0, 0.0, 0.0)}
    candidate = {
        "accepted": True,
        "reason": "provisional_motion_residual",
        "axis_applied": [True, True, False, False],
        "raw_signed_residual": (0.02, -0.02, 0.0, 0.0),
        "transform": (float(torch.exp(torch.tensor(0.02))), float(torch.exp(torch.tensor(-0.02))), 0.0, 0.0),
        "axis_estimator_floor": (0.005, 0.005, 0.25, 0.25),
    }
    profile = {
        "accepted": True,
        "reason": "source_evidence_and_temporal_state_authorized",
        "axis_accepted": [True, True, False, False],
        "axis_mode": ["measured_drift", "measured_drift", "inactive", "inactive"],
        "axis": [
            {
                "accepted": True,
                "mode": "measured_drift",
                "active_tokens": 1,
                "applied_envelope": {"kind": "measured_cumulative", "signed_states": [0.02], "active_tokens": 1},
            },
            {
                "accepted": True,
                "mode": "measured_drift",
                "active_tokens": 1,
                "applied_envelope": {"kind": "measured_cumulative", "signed_states": [-0.02], "active_tokens": 1},
            },
            {"accepted": False, "mode": "inactive"},
            {"accepted": False, "mode": "inactive"},
        ],
        "transitions": [],
    }
    monkeypatch.setattr(source_bridge, "_motion_model", lambda *args, **kwargs: motion)
    monkeypatch.setattr(source_bridge, "register_pair", lambda *args, **kwargs: boundary)
    monkeypatch.setattr(source_bridge, "_motion_residual_candidate", lambda *args, **kwargs: candidate)
    monkeypatch.setattr(source_bridge, "_temporal_profile", lambda *args, **kwargs: profile)

    def fake_warp(source, p, base_transform, attempt_profile):
        corrected = source.clone()
        corrected[:, :, p : p + 1] += 0.001
        return corrected, {
            "tokens_corrected": 1,
            "effective_active_temporal_length": 1,
            "applied_transforms": [base_transform],
            "applied_transform_sequence_length": 1,
            "applied_transforms_compact": False,
            "out_of_bounds_fraction": 0.0,
        }

    verification_masks = []

    def fake_verify(source, corrected, p, motion_arg, candidate_arg, attempt_profile, axis_mask, warp):
        verification_masks.append(list(axis_mask))
        if axis_mask[0]:
            verified = [False, True, False, False]
            ok = False
        else:
            verified = [False, True, False, False]
            ok = True
        return {
            "ok": ok,
            "axis_verified": verified,
            "boundary_axis_verified": [False, True, None, None],
            "boundary_after": boundary,
            "post_signed_residual": (0.02, 0.001, 0.0, 0.0),
            "residual_reduction_ratio": [None, 0.05, None, None],
            "transition_verification": [],
        }

    monkeypatch.setattr(source_bridge, "_warp_suffix", fake_warp)
    monkeypatch.setattr(source_bridge, "_verify_source_warp", fake_verify)

    corrected, report = source_bridge.apply_source_trajectory_bridge(video, prefix_t, requested=True)
    assert verification_masks == [[True, True, False, False], [False, True, False, False]]
    assert report["source_trajectory_bridge_accepted"] is True
    assert report["source_trajectory_axis_authorized"] == [True, True, False, False]
    assert report["source_trajectory_axis_post_verified"] == [False, True, False, False]
    assert report["source_trajectory_axis_pruned_post_verification"] == [True, False, False, False]
    assert len(report["source_trajectory_post_verification_rounds"]) == 2
    assert not torch.equal(corrected[:, :, prefix_t : prefix_t + 1], video[:, :, prefix_t : prefix_t + 1])
    assert torch.equal(corrected[:, :, prefix_t + 1 :], video[:, :, prefix_t + 1 :])
