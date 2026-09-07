from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from h3_flow_regenerate.geometric_bridge import (
    IDENTITY,
    SOURCE_CLEAN_CONTEXT_ATTR,
    compose_transform,
    geometric_seam_bridge,
    invert_transform,
    register_prefix,
    warp_frame,
)


def pattern(t=3, h=56, w=76, *, seed=774):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 8, t, h // 4, w // 4, generator=generator)
    return F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)


def transform_video(x, theta):
    return torch.stack([warp_frame(x[:, :, i], theta) for i in range(x.shape[2])], 2)


def motion_sequence(transforms, *, h=56, w=76, seed=991):
    generator = torch.Generator().manual_seed(seed)
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


def append_recovering_suffix(exact, natural, correction, count=4):
    expected = warp_frame(exact[:, :, -1], invert_transform(natural))
    first = warp_frame(expected, invert_transform(correction))
    second = warp_frame(expected, invert_transform(natural))
    suffix = [first, second]
    raw = second
    for _ in range(2, count):
        raw = warp_frame(raw, invert_transform(natural))
        suffix.append(raw)
    return torch.cat((exact, torch.stack(suffix, 2)), 2)


def attach_source(learned, source):
    setattr(learned, SOURCE_CLEAN_CONTEXT_ATTR, source)
    return learned


def matched_pair(*, source_correction=(1.0, 0.98, 0.0, 0.0), target_correction=None, count=5):
    if target_correction is None:
        target_correction = source_correction
    natural = IDENTITY
    target_prefix = motion_sequence([natural] * 7, h=56, w=76, seed=101)
    source_prefix = motion_sequence([natural] * 7, h=40, w=54, seed=202)
    learned = append_biased_suffix(target_prefix, natural, target_correction, count=count)
    source = append_biased_suffix(source_prefix, natural, source_correction, count=count)
    attach_source(learned, source)
    return target_prefix, learned, source


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


def test_transfer_prefix_bias_is_diagnostic_only():
    target_prefix, learned, source = matched_pair(source_correction=IDENTITY, target_correction=IDENTITY)
    learned = learned.clone()
    learned[:, :, : target_prefix.shape[2]] = transform_video(
        learned[:, :, : target_prefix.shape[2]],
        (1.02, 0.99, 0.75, -0.5),
    )
    attach_source(learned, source)
    result, _, report = geometric_seam_bridge(learned, target_prefix, source_hw=(40, 54), requested=True)
    assert report["transfer_bias_candidate"]
    assert not report["accepted"]
    assert result is learned


def test_persistent_vertical_scale_is_corrected_only_after_cross_grid_corroboration():
    exact, learned, _ = matched_pair()
    saved = learned.clone()
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["accepted"], report
    assert report["motion_residual"]["axis_applied"] == [False, True, False, False]
    assert report["suffix_persistence"]["accepted"]
    corroboration = report["cross_grid_corroboration"]
    assert corroboration["accepted"]
    assert corroboration["axis_applied"] == [False, True, False, False]
    assert corroboration["transform"][1] == pytest.approx(0.98, abs=0.005)
    assert report["tokens_corrected"] == learned.shape[2] - exact.shape[2]
    assert torch.equal(result[:, :, : exact.shape[2]], learned[:, :, : exact.shape[2]])
    assert not torch.equal(result[:, :, exact.shape[2] :], learned[:, :, exact.shape[2] :])
    assert torch.equal(learned, saved)
    expected = report["target_natural_motion_model"]["expected_transform"]
    assert report["boundary_after_geometry"]["transform"] == pytest.approx(expected, abs=0.01)


def test_source_residual_without_target_residual_is_noop():
    exact, learned, _ = matched_pair(source_correction=(1.0, 0.98, 0.0, 0.0), target_correction=IDENTITY)
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["motion_residual"]["accepted"]
    assert report["suffix_persistence"]["accepted"]
    assert not report["target_motion_residual"]["accepted"]
    assert not report["accepted"]
    assert report["reason"] == "target_boundary_not_significant"
    assert result is learned


def test_target_residual_without_source_residual_is_noop():
    exact, learned, _ = matched_pair(source_correction=IDENTITY, target_correction=(1.0, 0.98, 0.0, 0.0))
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["target_motion_residual"]["accepted"]
    assert not report["motion_residual"]["accepted"]
    assert not report["accepted"]
    assert result is learned


def test_opposite_cross_grid_residual_sign_is_noop():
    exact, learned, _ = matched_pair(
        source_correction=(1.0, 0.98, 0.0, 0.0),
        target_correction=(1.0, 1.02, 0.0, 0.0),
    )
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["motion_residual"]["accepted"]
    assert report["target_motion_residual"]["accepted"]
    assert not report["cross_grid_corroboration"]["accepted"]
    assert report["reason"] == "cross_grid_residual_not_corroborated"
    assert result is learned


def test_large_cross_grid_magnitude_disagreement_is_noop():
    exact, learned, _ = matched_pair(
        source_correction=(1.0, 0.9925, 0.0, 0.0),
        target_correction=(1.0, 0.975, 0.0, 0.0),
    )
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert not report["accepted"]
    assert result is learned


def test_recovering_source_boundary_rejects_persistent_full_suffix_warp():
    natural = IDENTITY
    target_prefix = motion_sequence([natural] * 7, h=56, w=76, seed=101)
    source_prefix = motion_sequence([natural] * 7, h=40, w=54, seed=202)
    learned = append_biased_suffix(target_prefix, natural, (1.0, 0.98, 0.0, 0.0), count=5)
    source = append_recovering_suffix(source_prefix, natural, (1.0, 0.98, 0.0, 0.0), count=5)
    attach_source(learned, source)
    result, _, report = geometric_seam_bridge(learned, target_prefix, source_hw=(40, 54), requested=True)
    assert report["motion_residual"]["accepted"]
    assert not report["suffix_persistence"]["accepted"]
    assert report["reason"] == "suffix_recovery_or_drift_detected"
    assert result is learned


def test_natural_motion_is_preserved():
    target_natural = (1.0, 1.0, -0.75, 0.125)
    source_natural = (1.0, 1.0, target_natural[2] * 54 / 76, target_natural[3] * 40 / 56)
    target_prefix = motion_sequence([target_natural] * 7, h=56, w=76, seed=101)
    source_prefix = motion_sequence([source_natural] * 7, h=40, w=54, seed=202)
    learned = append_biased_suffix(target_prefix, target_natural, IDENTITY, count=5)
    source = append_biased_suffix(source_prefix, source_natural, IDENTITY, count=5)
    attach_source(learned, source)
    result, _, report = geometric_seam_bridge(learned, target_prefix, source_hw=(40, 54), requested=True)
    assert not report["accepted"]
    assert result is learned


def test_gradual_zoom_trend_is_extrapolated_not_corrected():
    target_transitions = [(1.0, 1.0 + i * 0.002, 0.0, 0.0) for i in range(7)]
    source_transitions = list(target_transitions)
    target_prefix = motion_sequence(target_transitions, h=56, w=76, seed=101)
    source_prefix = motion_sequence(source_transitions, h=40, w=54, seed=202)
    next_motion = (1.0, 1.014, 0.0, 0.0)
    learned = append_biased_suffix(target_prefix, next_motion, IDENTITY, count=5)
    source = append_biased_suffix(source_prefix, next_motion, IDENTITY, count=5)
    attach_source(learned, source)
    result, _, report = geometric_seam_bridge(learned, target_prefix, source_hw=(40, 54), requested=True)
    assert not report["accepted"]
    assert result is learned
    expected = report["low_grid_trajectory"]["natural_motion_model"]["expected_transform"]
    assert expected[1] == pytest.approx(next_motion[1], abs=0.005)


def test_translation_is_measured_on_source_but_applied_in_target_units():
    source_tx = 0.75
    target_tx = source_tx * 76 / 54
    exact, learned, _ = matched_pair(
        source_correction=(1.0, 1.0, source_tx, 0.0),
        target_correction=(1.0, 1.0, target_tx, 0.0),
    )
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["accepted"], report
    corroboration = report["cross_grid_corroboration"]
    assert corroboration["axis_applied"] == [False, False, True, False]
    assert corroboration["projected_source_transform"][2] == pytest.approx(target_tx)
    assert corroboration["transform"][2] == pytest.approx(target_tx, abs=0.2)
    assert result is not learned


def test_missing_source_context_is_safe_noop():
    natural = IDENTITY
    exact = motion_sequence([natural] * 7)
    learned = append_biased_suffix(exact, natural, (1.0, 0.98, 0.0, 0.0), count=5)
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert not report["accepted"]
    assert report["reason"] == "source_trajectory_context_unavailable"
    assert result is learned


def test_disabled_is_exact_noop_but_diagnostics_still_identify_candidate():
    exact, learned, _ = matched_pair()
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=False)
    assert result is learned
    assert not report["accepted"] and report["reason"] == "disabled"
    assert report["persistent_bias_candidate"]
    assert report["corroborated_bias_candidate"]


def test_40x54_to_56x76_geometry_is_reported():
    exact, learned, _ = matched_pair()
    _, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=False)
    assert report["grid_scale_y"] == 1.4
    assert report["grid_scale_x"] == 76 / 54
    assert report["grid_anisotropy"] == pytest.approx((76 / 54) / 1.4 - 1)


def test_constant_and_nonfinite_inputs_reject():
    exact = torch.zeros(1, 8, 8, 56, 76)
    learned = torch.zeros(1, 8, 12, 56, 76)
    source = torch.zeros(1, 8, 12, 40, 54)
    attach_source(learned, source)
    _, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert not report["accepted"]
    bad = exact.clone()
    bad[..., 0, 0] = math.nan
    fit = register_prefix(bad[:, :, :3], exact[:, :, :3])
    assert not fit["accepted"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_dtype_and_prefix_exact(dtype):
    exact, learned, source = matched_pair(count=3)
    exact = exact.to(dtype)
    learned = learned.to(dtype)
    source = source.to(dtype)
    attach_source(learned, source)
    prefix = learned[:, :, : exact.shape[2]].clone()
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["accepted"], report
    assert result.dtype == dtype and result.device == learned.device
    assert torch.equal(result[:, :, : exact.shape[2]], prefix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_path():
    exact, learned, source = matched_pair(count=3)
    exact = exact.cuda().half()
    learned = learned.cuda().half()
    source = source.cuda().half()
    attach_source(learned, source)
    result, _, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=True)
    assert report["accepted"], report
    assert result.is_cuda and result.dtype == torch.float16
