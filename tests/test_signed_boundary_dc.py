import pytest
import torch

from h3_flow_regenerate.signed_boundary_dc import (
    SIGNED_BOUNDARY_DC_STAGES,
    latent_preview_factors,
    measure_signed_boundary_dc,
    signed_boundary_dc_ledger,
)

PREFIX_T, TEMPORAL, C = 12, 18, 24
SOURCE_HW, TARGET_HW = (4, 6), (8, 12)


def _detail(hw, seed):
    generator = torch.Generator().manual_seed(seed)
    detail = torch.randn(1, C, TEMPORAL, *hw, generator=generator)
    return detail - detail.mean(dim=(-2, -1), keepdim=True)


def _video(hw, dc, seed):
    return _detail(hw, seed) + dc.view(1, C, TEMPORAL, 1, 1)


def _continuous_dc(seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(C, 1, generator=generator)
    drift = 0.01 * torch.randn(C, 1, generator=generator)
    return base + drift * torch.arange(TEMPORAL, dtype=torch.float32)


def _step(seed=5):
    generator = torch.Generator().manual_seed(seed)
    step = torch.zeros(C, TEMPORAL)
    step[:, PREFIX_T:] = 0.4 * torch.randn(C, 1, generator=generator)
    return step


def _upscale(video):
    return video.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def _receipts(source, provider, exact_prefix, final):
    restored = provider.clone()
    restored[:, :, :PREFIX_T] = exact_prefix
    videos = dict(zip(SIGNED_BOUNDARY_DC_STAGES, (source, provider, restored, final), strict=True))
    return {stage: measure_signed_boundary_dc(video, PREFIX_T) for stage, video in videos.items()}


def test_signed_step_recovers_a_per_channel_dc_shift_under_spatial_detail():
    dc = _continuous_dc()
    step = _step()
    factors = torch.randn(C, 3)
    receipt = measure_signed_boundary_dc(_video(TARGET_HW, dc + step, 1), PREFIX_T, preview_factors=factors)
    expected = (dc + step)[:, PREFIX_T] - (dc + step)[:, PREFIX_T - 1]
    assert receipt["tokens"] == list(range(PREFIX_T - 3, PREFIX_T + 4))
    torch.testing.assert_close(torch.tensor(receipt["boundary_step"]), expected, atol=1e-5, rtol=0)
    assert receipt["boundary_step_rms"] == pytest.approx(float(expected.square().mean().sqrt()), abs=1e-5)
    pairs = receipt["pair_step_rms"]
    assert pairs[2] == pytest.approx(receipt["boundary_step_rms"])
    assert max(pairs[:2] + pairs[3:]) < 0.05
    luma = torch.tensor([0.299, 0.587, 0.114])
    assert receipt["preview_luma_boundary_step"] == pytest.approx(float(expected @ factors @ luma), abs=1e-4)


def test_ledger_attributes_a_source_created_seam_to_the_low_stage():
    dc = _continuous_dc() + _step()
    source = _video(SOURCE_HW, dc, 1)
    provider = _upscale(source) + _detail(TARGET_HW, 2)
    final = provider + 0.1 * _detail(TARGET_HW, 3)
    ledger = signed_boundary_dc_ledger(_receipts(source, provider, provider[:, :, :PREFIX_T], final))
    rows = ledger["components"]
    assert ledger["status"] == "complete"
    assert rows["source_low"]["signed_contribution_to_final_norm"] == pytest.approx(ledger["final_step_norm"], rel=1e-4)
    for name in ("provider_transfer", "exact_prefix_restore", "target_high"):
        assert abs(rows[name]["signed_contribution_to_final_norm"]) < 1e-4


def test_ledger_attributes_a_provider_created_seam_to_the_transfer():
    source = _video(SOURCE_HW, _continuous_dc(), 1)
    provider = _upscale(source) + _video(TARGET_HW, _step(), 2)
    final = provider + 0.1 * _detail(TARGET_HW, 3)
    ledger = signed_boundary_dc_ledger(_receipts(source, provider, provider[:, :, :PREFIX_T], final))
    rows = ledger["components"]
    assert rows["provider_transfer"]["signed_contribution_to_final_norm"] == pytest.approx(
        ledger["final_step_norm"], rel=0.02
    )
    assert abs(rows["source_low"]["signed_contribution_to_final_norm"]) < 0.05 * ledger["final_step_norm"]


def test_ledger_contributions_sum_to_the_final_step_and_separate_restore_and_high():
    source = _video(SOURCE_HW, _continuous_dc(), 1)
    provider = _upscale(source) + _video(TARGET_HW, 0.5 * _step(7), 2)
    offset = torch.zeros(1, C, PREFIX_T, 1, 1)
    offset[:, :, -1] = 0.3 * torch.randn(C, 1, 1, generator=torch.Generator().manual_seed(8))
    exact_prefix = provider[:, :, :PREFIX_T] + offset
    restored = provider.clone()
    restored[:, :, :PREFIX_T] = exact_prefix
    final = restored + _video(TARGET_HW, 0.2 * _step(9), 3)
    ledger = signed_boundary_dc_ledger(_receipts(source, provider, exact_prefix, final))
    rows = ledger["components"]
    total = sum(row["signed_contribution_to_final_norm"] for row in rows.values())
    assert total == pytest.approx(ledger["final_step_norm"], rel=1e-6)
    assert all(rows[name]["rms"] > 0.01 for name in ("provider_transfer", "exact_prefix_restore", "target_high"))


def test_incomplete_ledger_and_preview_factor_validation():
    receipt = measure_signed_boundary_dc(_video(TARGET_HW, _continuous_dc(), 1), PREFIX_T)
    ledger = signed_boundary_dc_ledger({"provider_native": receipt})
    assert ledger["status"] == "incomplete" and "final" in ledger["missing_stages"]
    assert receipt["preview_luma"] is None and receipt["preview"] == "unavailable"
    assert latent_preview_factors(type("F", (), {"latent_rgb_factors": [[0.0] * 3] * C})(), C).shape == (C, 3)
    assert latent_preview_factors(type("F", (), {"latent_rgb_factors": [[0.0] * 3] * 4})(), C) is None
    assert latent_preview_factors(None, C) is None
    with pytest.raises(ValueError):
        measure_signed_boundary_dc(torch.zeros(1, C, 4, 2, 2), 4)
