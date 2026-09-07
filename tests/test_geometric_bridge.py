from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from h3_flow_regenerate.geometric_bridge import (
    IDENTITY,
    compose_transform,
    geometric_seam_bridge,
    invert_transform,
    register_prefix,
    warp_frame,
)


def pattern(t=3, h=56, w=76):
    generator = torch.Generator().manual_seed(774)
    x = torch.randn(1, 8, t, h // 4, w // 4, generator=generator)
    return F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)


def transform_video(x, theta):
    return torch.stack([warp_frame(x[:, :, i], theta) for i in range(x.shape[2])], 2)


def motion_sequence(transforms, *, h=56, w=76):
    generator = torch.Generator().manual_seed(991)
    frame = torch.randn(1, 8, h // 4, w // 4, generator=generator)
    frame = F.interpolate(frame, size=(h, w), mode="bilinear", align_corners=False)
    frames = [frame]
    for theta in transforms:
        frames.append(warp_frame(frames[-1], invert_transform(theta)))
    return torch.stack(frames, 2)


def append_biased_suffix(exact, natural, correction, count=4):
    expected = warp_frame(exact[:, :, -1], invert_transform(natural))
    raw = warp_frame(expected, invert_transform(correction))
    suffix = [raw]
    for _ in range(1, count):
        raw = warp_frame(raw, invert_transform(natural))
        suffix.append(raw)
    return torch.cat((exact, torch.stack(suffix, 2)), 2)


def test_transform_algebra():
    a = (1.02, 0.99, 0.75, -0.5)
    b = (0.98, 1.01, -0.25, 0.375)
    composed = compose_transform(a, b)
    x, y = 3.25, -2.5
    abx = a[0] * (b[0] * x + b[2]) + a[2]
    aby = a[1] * (b[1] * y + b[3]) + a[3]
    assert composed[0] * x + composed[2] == pytest.approx(abx)
    assert composed[1] * y + composed[3] == pytest.approx(aby)
    assert compose_transform(a, invert_transform(a)) == pytest.approx(IDENTITY, abs=1e-12)


@pytest.mark.parametrize(
    "theta",
    [
        (1, 1, 0.75, 0),
        (1, 1, 0, -0.75),
        (1.02, 1.02, 0, 0),
        (1.02, 0.98, 0, 0),
        (1.015, 0.985, 0.75, -0.5),
    ],
)
def test_known_same_time_transforms(theta):
    moving = pattern()
    reference = transform_video(moving, theta)
    report = register_prefix(reference, moving)
    assert report["accepted"], report
    for got, expected, atol in zip(report["transform"], theta, (0.005, 0.005, 0.15, 0.15), strict=True):
        assert got == pytest.approx(expected, abs=atol)


def test_same_time_prefix_bias_no_longer_drives_bridge():
    learned = pattern(t=12)
    exact = transform_video(learned[:, :, :8], (1.02, 0.99, 0.75, -0.5))
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["transfer_bias_candidate"]
    assert not report["accepted"]
    assert result is learned


def test_motion_residual_corrects_only_supported_axis_and_full_suffix():
    natural = (1.0, 1.0, -0.5, 0.0)
    exact = motion_sequence([natural] * 7)
    seam = (1.0, 0.98, 0.0, 0.0)
    learned = append_biased_suffix(exact, natural, seam, count=5)
    saved = learned.clone()
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["accepted"], report
    candidate = report["motion_residual"]
    assert candidate["axis_applied"] == [False, True, False, False]
    assert candidate["transform"][1] == pytest.approx(0.98, abs=0.005)
    assert report["tokens_corrected"] == 5
    assert torch.equal(result[:, :, : exact.shape[2]], learned[:, :, : exact.shape[2]])
    assert not torch.equal(result[:, :, exact.shape[2] :], learned[:, :, exact.shape[2] :])
    assert torch.equal(learned, saved)
    after = report["boundary_after_geometry"]
    expected = report["natural_motion_model"]["expected_transform"]
    assert after["transform"] == pytest.approx(expected, abs=0.01)


def test_natural_motion_without_seam_is_noop():
    natural = (1.0, 1.0, -0.5, 0.125)
    exact = motion_sequence([natural] * 7)
    learned = append_biased_suffix(exact, natural, IDENTITY, count=4)
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert not report["accepted"]
    assert report["reason"] == "boundary_not_better_than_motion_model"
    assert result is learned


def test_gradual_motion_trend_is_extrapolated_not_corrected():
    transitions = [(1.0, 1.0 + i * 0.002, -0.5, 0.0) for i in range(7)]
    exact = motion_sequence(transitions)
    next_motion = (1.0, 1.0 + 7 * 0.002, -0.5, 0.0)
    learned = append_biased_suffix(exact, next_motion, IDENTITY, count=4)
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert not report["accepted"]
    assert result is learned
    assert report["natural_motion_model"]["expected_transform"][1] == pytest.approx(next_motion[1], abs=0.005)


def test_disabled_is_exact_noop_but_still_measures_motion_residual():
    natural = (1.0, 1.0, 0.0, 0.0)
    exact = motion_sequence([natural] * 7)
    learned = append_biased_suffix(exact, natural, (1.0, 0.98, 0.0, 0.0), count=3)
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=False)
    assert result is learned
    assert not report["accepted"] and report["reason"] == "disabled"
    assert report["motion_residual"]["accepted"]


def test_40x54_to_56x76_geometry_is_reported():
    natural = (1.0, 1.0, 0.0, 0.0)
    exact = motion_sequence([natural] * 7)
    learned = append_biased_suffix(exact, natural, (1.0, 0.98, 0.0, 0.0), count=3)
    _, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=False)
    assert report["grid_scale_y"] == 1.4
    assert report["grid_scale_x"] == 76 / 54
    assert report["grid_anisotropy"] == pytest.approx((76 / 54) / 1.4 - 1)


def test_constant_and_nonfinite_rejected():
    exact = torch.zeros(1, 8, 8, 56, 76)
    learned = torch.zeros(1, 8, 12, 56, 76)
    _, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert not report["accepted"]
    bad = exact.clone()
    bad[..., 0, 0] = math.nan
    fit = register_prefix(bad[:, :, :3], exact[:, :, :3])
    assert not fit["accepted"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_dtype_and_prefix_exact(dtype):
    natural = (1.0, 1.0, -0.5, 0.0)
    exact = motion_sequence([natural] * 7).to(dtype)
    learned = append_biased_suffix(exact, natural, (1.0, 0.98, 0.0, 0.0), count=3)
    prefix = learned[:, :, : exact.shape[2]].clone()
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["accepted"]
    assert result.dtype == dtype and result.device == learned.device
    assert torch.equal(result[:, :, : exact.shape[2]], prefix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_path():
    natural = (1.0, 1.0, -0.5, 0.0)
    exact = motion_sequence([natural] * 7).cuda().half()
    learned = append_biased_suffix(exact, natural, (1.0, 0.98, 0.0, 0.0), count=3)
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["accepted"]
    assert result.is_cuda and result.dtype == torch.float16
