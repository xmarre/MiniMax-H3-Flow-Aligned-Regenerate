"""Image-grid noise coupling: independent image operator and distribution checks."""

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import (
    H3_HANDOFF_NOISE_IMAGE_DRIFT,
    H3_HANDOFF_NOISE_IMAGE_RESIDUAL,
    build_handoff_state,
)
from h3_flow_regenerate.image_residual import refine_image_residual, transport_image_flow_residual


def _image_analysis(source_hw, target_hw):
    # Independently build the full 2D operator from actual image resize. The
    # production implementation builds separable 1D matrices instead.
    basis = torch.eye(target_hw[0] * target_hw[1], dtype=torch.float64).reshape(-1, 1, *target_hw)
    resized = F.interpolate(basis, size=source_hw, mode="bicubic", align_corners=False, antialias=True)
    return resized.flatten(1).T


def _white_analysis(a):
    eig, vec = torch.linalg.eigh(a @ a.T)
    return (vec / eig.sqrt()) @ vec.T @ a


@pytest.mark.parametrize("source_hw,target_hw", [((4, 6), (6, 8)), ((6, 4), (8, 6)), ((4, 8), (4, 12))])
def test_whole_image_coarse_noise_is_retained_and_seed_changes_only_nullspace(source_hw, target_hw):
    gen = torch.Generator().manual_seed(53)
    source = torch.randn(1, 3, 2, *source_hw, generator=gen)
    innovation = torch.randn(1, 3, 2, *target_hw, generator=gen)
    another = torch.randn(1, 3, 2, *target_hw, generator=gen)
    target, report = refine_image_residual(source, innovation)
    other, _ = refine_image_residual(source, another)
    q = _white_analysis(_image_analysis(source_hw, target_hw))
    coarse = target.double().flatten(-2) @ q.T
    coarse_other = other.double().flatten(-2) @ q.T
    torch.testing.assert_close(coarse, source.double().flatten(-2), atol=2e-6, rtol=0)
    torch.testing.assert_close(coarse, coarse_other, atol=3e-6, rtol=0)
    assert not torch.equal(target, other)
    assert report["projection_max_abs_error"] < 2e-6
    assert torch.equal(refine_image_residual(source, innovation)[0], target)


def test_gaussian_noise_has_unit_variance_and_no_cross_cell_or_temporal_correlation():
    gen = torch.Generator().manual_seed(770)
    source = torch.randn(8192, 1, 2, 2, 4, generator=gen)
    innovation = torch.randn(8192, 1, 2, 4, 6, generator=gen)
    target, _ = refine_image_residual(source, innovation)
    samples = target.flatten(1)
    covariance = torch.cov(samples.T)
    assert samples.mean(0).abs().max() < 0.04
    assert (covariance.diag() - 1).abs().max() < 0.06
    offdiag = covariance - torch.diag(covariance.diag())
    assert offdiag.abs().max() < 0.06
    # Ordinary interpolation alone fails this law; do not "simplify" the
    # coupling into a resized Gaussian plus unconstrained innovation.
    naive = F.interpolate(source.reshape(-1, 1, 2, 4), size=(4, 6), mode="bicubic", align_corners=False)
    assert (naive.var(0) - 1).abs().max() > 0.3


@pytest.mark.parametrize("source_hw,target_hw", [((50, 38), (72, 54)), ((38, 50), (54, 72)), ((36, 54), (50, 76))])
@pytest.mark.parametrize("drift_kind", ["constant", "moving", "random"])
def test_deterministic_model_drift_preserves_actual_antialiased_image_projection(source_hw, target_hw, drift_kind):
    gen = torch.Generator().manual_seed(72)
    initial = torch.randn(1, 3, 3, *source_hw, generator=gen)
    innovation = torch.randn(1, 3, 3, *target_hw, generator=gen)
    if drift_kind == "constant":
        drift = torch.full_like(initial, 0.25)
    elif drift_kind == "random":
        drift = torch.randn(initial.shape, generator=gen) * 0.05
    else:
        drift = torch.zeros_like(initial)
        for frame in range(3):
            drift[:, :, frame, source_hw[0] // 2, source_hw[1] // 2 + frame] = 0.2
    scale = 1.7
    baseline, _ = transport_image_flow_residual(initial * scale, initial, innovation, noise_scale=scale)
    target, report = transport_image_flow_residual(initial * scale + drift, initial, innovation, noise_scale=scale)
    delta = target - baseline
    projected = F.interpolate(
        delta.permute(0, 2, 1, 3, 4).reshape(-1, 3, *target_hw),
        size=source_hw,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    expected = drift.permute(0, 2, 1, 3, 4).reshape_as(projected)
    torch.testing.assert_close(projected, expected, atol=1.5e-6, rtol=0)
    if drift_kind == "constant":
        torch.testing.assert_close(delta, torch.full_like(delta, 0.25), atol=1.5e-6, rtol=0)
    if drift_kind == "moving":
        assert not torch.equal(delta[:, :, 0], delta[:, :, 1])
    assert report["gaussian_component_variance_if_initial_standard"] == pytest.approx(scale**2)
    assert report["drift_projection_max_abs_error"] < 2e-6
    assert "gaussian_marginal_if_source_standard" not in report


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_same_grid_identity_retains_bytes_and_ownership(dtype):
    source = torch.randn(1, 3, 2, 4, 6).to(dtype)
    initial, innovation = torch.randn_like(source), torch.randn_like(source)
    target, report = transport_image_flow_residual(source, initial, innovation, noise_scale=1.7)
    assert torch.equal(target, source) and target.data_ptr() != source.data_ptr()
    assert report["identity"] is True


@pytest.mark.parametrize("policy", [H3_HANDOFF_NOISE_IMAGE_DRIFT, H3_HANDOFF_NOISE_IMAGE_RESIDUAL])
def test_handoff_uses_measured_state_residual_keeps_audio_and_calls_provider_once(policy):
    class Provider:
        api_version = 1
        kind = "minimax_h3_learned_latent_upscaler"
        model_name = "image-fixture"
        device = inference_device = "cpu"
        precision = "fp32"
        offload_after_upscale = False
        calls = 0

        def upscale_clean_video(self, x, *, target_h, target_w):
            self.calls += 1
            return (
                F.interpolate(
                    x.permute(0, 2, 1, 3, 4).flatten(0, 1),
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )
                .reshape(1, 2, 24, target_h, target_w)
                .transpose(1, 2)
            )

    initial = torch.randn(1, 24, 2, 4, 6)
    clean = torch.randn_like(initial)
    residual = initial * 1.7 + 0.03
    sigma = 0.87804878
    state = (1 - sigma) * clean + sigma * residual
    audio = torch.randn(1, 32, 2, 7)
    packed, shapes = pack_streams((state, audio))
    x0, _ = pack_streams((clean, torch.zeros_like(audio)))
    provider = Provider()
    receipts = {}
    target, out_shapes = build_handoff_state(
        source_packed_state=packed,
        source_x0_packed=x0,
        source_shapes=shapes,
        sigma=sigma,
        target_h=6,
        target_w=8,
        seed=17,
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        transfer_metrics=receipts,
        noise_mode=policy,
        initial_source_noise=initial,
        model_noise_scale=1.7,
    )
    target_video, target_audio = unpack_streams(target, out_shapes)
    learned = (
        F.interpolate(clean.transpose(1, 2).flatten(0, 1), size=(6, 8), mode="bicubic", align_corners=False)
        .reshape(1, 2, 24, 6, 8)
        .transpose(1, 2)
    )
    effective = (target_video - (1 - sigma) * learned) / sigma
    assert torch.equal(target_audio, audio)
    assert provider.calls == 1
    assert receipts["handoff_noise"]["policy"] == policy
    assert receipts["handoff_noise"]["source_state_reconstruction_max_abs_error"] < 5e-7
    if policy == H3_HANDOFF_NOISE_IMAGE_RESIDUAL:
        q = _white_analysis(_image_analysis((4, 6), (6, 8)))
        torch.testing.assert_close(
            effective.double().flatten(-2) @ q.T, residual.double().flatten(-2), atol=2e-6, rtol=0
        )
