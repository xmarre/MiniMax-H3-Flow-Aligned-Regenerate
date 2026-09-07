from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from h3_flow_regenerate.geometric_bridge import (
    BRIDGE_WEIGHTS,
    geometric_seam_bridge,
    register_prefix,
    warp_frame,
)


def pattern(t=3, h=56, w=76):
    g = torch.Generator().manual_seed(774)
    x = torch.randn(1, 8, t, h // 4, w // 4, generator=g)
    return F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)


def transform_video(x, theta):
    return torch.stack([warp_frame(x[:, :, i], theta) for i in range(x.shape[2])], 2)


@pytest.mark.parametrize(
    "theta", [(1, 1, 0.75, 0), (1, 1, 0, -0.75), (1.02, 1.02, 0, 0), (1.02, 0.98, 0, 0), (1.015, 0.985, 0.75, -0.5)]
)
def test_known_transforms(theta):
    # Analytic moving->reference sampling, including independent sign assertion.
    moving = pattern()
    reference = transform_video(moving, theta)
    with torch.inference_mode():
        report = register_prefix(reference, moving)
    assert report["accepted"], report
    for got, expected, atol in zip(report["transform"], theta, (0.005, 0.005, 0.15, 0.15), strict=True):
        assert got == pytest.approx(expected, abs=atol)
    assert report["aligned_error"] < report["identity_error"] * 0.3


def test_sampling_translation_convention():
    x = torch.arange(32.0).reshape(1, 1, 1, 32).expand(1, 1, 32, 32)
    assert torch.allclose(warp_frame(x, (1, 1, 1, 0))[..., 2:-2], x[..., 3:-1])


@pytest.mark.parametrize("offset", [0.0, 3.0])
def test_identity_dc_noop(offset):
    x = pattern()
    report = register_prefix(x + offset, x)
    assert not report["accepted"]
    assert report["transform"] == (1.0, 1.0, 0.0, 0.0)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_uninformative_nonfinite_rejected(value):
    x = torch.full((1, 8, 3, 56, 76), value)
    assert not register_prefix(x, x)["accepted"]


def test_extreme_and_temporally_inconsistent_rejected():
    x = pattern()
    assert not register_prefix(transform_video(x, (1.06, 1, 2.5, 0)), x)["accepted"]
    y = torch.stack([warp_frame(x[:, :, i], (1, 1, (-1) ** i, 0)) for i in range(3)], 2)
    assert not register_prefix(y, x)["accepted"]


def test_deterministic():
    x = pattern()
    y = transform_video(x, (1.02, 1, 0.5, 0))
    assert register_prefix(y, x) == register_prefix(y, x)


@pytest.mark.parametrize("requested", [False, True])
def test_rejected_or_disabled_returns_original(requested):
    x = pattern(7)
    exact = x[:, :, :3].clone() if requested else transform_video(x[:, :, :3], (1, 1, 0.75, 0))
    result, _, report = geometric_seam_bridge(x, exact, source_hw=(40, 54), requested=requested)
    assert result is x
    assert not report["accepted"]
    assert report["grid_scale_y"] == 1.4
    assert report["grid_scale_x"] == 76 / 54
    assert report["grid_anisotropy"] == pytest.approx((76 / 54) / 1.4 - 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_bridge_prefix_tail_decay_dtype(dtype):
    x = pattern(7).to(dtype)
    saved = x.clone()
    exact = transform_video(x[:, :, :3], (1.02, 0.99, 0.75, -0.5))
    original_exact = exact.clone()
    result, _, report = geometric_seam_bridge(x, exact, source_hw=(40, 54), requested=True)
    assert report["accepted"], report
    assert result.dtype == x.dtype and result.device == x.device
    assert torch.equal(exact, original_exact) and torch.equal(x, saved)
    assert torch.equal(result[:, :, :3], x[:, :, :3]) and torch.equal(result[:, :, 6:], x[:, :, 6:])
    theta = report["prefix_registration"]["transform"]
    for i, weight in enumerate(BRIDGE_WEIGHTS):
        tr = (theta[0] ** weight, theta[1] ** weight, theta[2] * weight, theta[3] * weight)
        assert torch.allclose(result[:, :, 3 + i], warp_frame(x[:, :, 3 + i], tr), atol=1e-6, rtol=1e-6)
    target = warp_frame(x[:, :, 3], (1.02, 0.99, 0.75, -0.5))
    assert (result[:, :, 3].float() - target.float()).square().mean() < (
        x[:, :, 3].float() - target.float()
    ).square().mean() * 0.1


def test_single_token_and_extreme_aspect_do_not_apply():
    x = pattern(t=1)
    y = transform_video(x, (1.02, 1, 0.75, 0))
    report = register_prefix(y, x)
    assert report["reason"] == "insufficient_paired_tokens"
    assert report["improvement"] > 0.15
    thin = torch.randn(1, 8, 2, 16, 512)
    assert register_prefix(thin, thin)["reason"] == "unsupported_geometry"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_inference_and_cpu_estimate_agree(dtype):
    x = pattern(7).to(device="cuda", dtype=dtype)
    y = transform_video(x[:, :, :3], (1.02, 0.99, 0.75, -0.5))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out, _, report = geometric_seam_bridge(x, y, source_hw=(40, 54), requested=True)
    assert report["accepted"]
    assert out.dtype == dtype and out.device == x.device
    cpu = register_prefix(y.cpu(), x[:, :, :3].cpu())
    assert cpu["transform"] == pytest.approx(report["prefix_registration"]["transform"], abs=0.005)
    assert torch.equal(out[:, :, :3], x[:, :, :3])


def test_per_channel_dc_and_local_detail_do_not_drive_geometry():
    moving = pattern()
    theta = (1.02, 0.99, 0.75, -0.5)
    reference = transform_video(moving, theta)
    offsets = torch.arange(8.0).reshape(1, 8, 1, 1, 1)
    noisy = reference + offsets
    g = torch.Generator().manual_seed(889)
    noisy = noisy + torch.randn(noisy.shape, generator=g) * 0.03
    noisy[:, :, :, 20:24, 30:34] += 0.2
    fit = register_prefix(noisy, moving)
    assert fit["accepted"], fit
    assert fit["transform"] == pytest.approx(theta, abs=0.005)
