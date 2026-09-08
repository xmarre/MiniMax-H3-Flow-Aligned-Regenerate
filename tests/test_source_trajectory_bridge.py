from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from h3_flow_regenerate.source_trajectory_bridge import (
    _MIN_SOURCE_EVIDENCE,
    apply_source_trajectory_bridge,
)


def _texture() -> torch.Tensor:
    torch.manual_seed(77)
    frame = torch.randn(1, 24, 32, 40)
    return F.avg_pool2d(F.pad(frame, (2, 2, 2, 2), mode="reflect"), 5, stride=1)


def _warp(frame: torch.Tensor, *, sy: float) -> torch.Tensor:
    b, _c, h, w = frame.shape
    theta = frame.new_tensor([[1.0, 0.0, 0.0], [0.0, sy, 0.0]], dtype=torch.float32)
    grid = F.affine_grid(theta[None].expand(b, -1, -1), frame.shape, align_corners=False)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _recovering_video() -> torch.Tensor:
    frame = _texture()
    prefix = [frame.clone() for _ in range(6)]
    # Registration of the first suffix against an identity-motion prefix is ~0.9775 sy.
    first_suffix = _warp(frame, sy=1.0 / 0.975)
    return torch.stack(prefix + [first_suffix, frame.clone(), frame.clone()], dim=2)


def test_recovering_source_sy_is_closed_before_upscale_and_tail_is_exact():
    video = _recovering_video()
    before = video.clone()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=True)

    assert report["source_trajectory_bridge_accepted"] is True
    assert report["source_trajectory_bridge_reason"] == "source_trajectory_residual_closed_before_upscale"
    assert report["source_trajectory_axis_authorized"] == [False, True, False, False]
    assert report["source_trajectory_temporal_profile"]["axis_mode"][1] == "recovering"
    assert report["source_trajectory_motion_residual"]["axis_evidence_score"][1] > _MIN_SOURCE_EVIDENCE
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


def test_source_evidence_threshold_is_derived_from_existing_combined_gate():
    assert _MIN_SOURCE_EVIDENCE == pytest.approx(2.5 / math.sqrt(2.0))
