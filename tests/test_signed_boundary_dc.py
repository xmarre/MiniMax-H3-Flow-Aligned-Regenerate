import pytest
import torch

from h3_flow_regenerate.signed_boundary_dc import (
    latent_preview_factors,
    measure_signed_boundary_dc,
    measure_signed_prefix_dc,
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


def _step(seed=5, start=PREFIX_T, stop=TEMPORAL):
    generator = torch.Generator().manual_seed(seed)
    step = torch.zeros(C, TEMPORAL)
    step[:, start:stop] = 0.4 * torch.randn(C, 1, generator=generator)
    return step


def _upscale(video):
    return video.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def _with_prefix(video, prefix):
    out = video.clone()
    out[:, :, :PREFIX_T] = prefix
    return out


def _pipeline(native, context, provider, exact_prefix, final):
    videos = {
        "source_low_native": native,
        "source_low_exact_context": context,
        "provider_native": provider,
        "pre_high_exact_restored": _with_prefix(provider, exact_prefix),
        "final": final,
    }
    receipts = {stage: measure_signed_boundary_dc(video, PREFIX_T) for stage, video in videos.items()}
    return signed_boundary_dc_ledger(receipts)


def _contributions(ledger):
    return {name: row["signed_contribution_to_final_norm"] for name, row in ledger["components"].items()}


def test_signed_step_recovers_a_per_channel_dc_shift_under_spatial_detail():
    dc = _continuous_dc() + _step()
    factors = torch.randn(C, 3)
    receipt = measure_signed_boundary_dc(_video(TARGET_HW, dc, 1), PREFIX_T, preview_factors=factors)
    expected = dc[:, PREFIX_T] - dc[:, PREFIX_T - 1]
    assert receipt["status"] == "measured"
    assert receipt["tokens"] == list(range(PREFIX_T - 3, PREFIX_T + 4))
    torch.testing.assert_close(torch.tensor(receipt["boundary_step"]), expected, atol=1e-5, rtol=0)
    assert receipt["boundary_step_rms"] == pytest.approx(float(expected.square().mean().sqrt()), abs=1e-5)
    pairs = receipt["pair_step_rms"]
    assert pairs[2] == pytest.approx(receipt["boundary_step_rms"])
    assert max(pairs[:2] + pairs[3:]) < 0.05
    luma = torch.tensor([0.299, 0.587, 0.114])
    assert receipt["preview_luma_boundary_step"] == pytest.approx(float(expected @ factors @ luma), abs=1e-4)


def test_ledger_attributes_a_low_generated_seam_to_the_low_stage():
    source = _video(SOURCE_HW, _continuous_dc() + _step(), 1)
    provider = _upscale(source) + _detail(TARGET_HW, 2)
    final = provider + 0.1 * _detail(TARGET_HW, 3)
    ledger = _pipeline(source, source, provider, provider[:, :, :PREFIX_T], final)
    parts = _contributions(ledger)
    assert ledger["status"] == "complete"
    assert parts["source_low"] == pytest.approx(ledger["final_step_norm"], rel=1e-4)
    assert all(abs(parts[name]) < 1e-4 for name in parts if name != "source_low")


def test_ledger_separates_a_prefix_write_on_another_gauge_from_low_generation():
    dc = _continuous_dc()
    native = _video(SOURCE_HW, dc, 1)
    offset = _step(6, 0, PREFIX_T)
    context = _with_prefix(native, (native + offset.view(1, C, TEMPORAL, 1, 1))[:, :, :PREFIX_T])
    provider = _upscale(context)
    final = provider + 0.1 * _detail(TARGET_HW, 3)
    ledger = _pipeline(native, context, provider, provider[:, :, :PREFIX_T], final)
    parts = _contributions(ledger)
    assert parts["source_prefix_write"] == pytest.approx(ledger["final_step_norm"], rel=0.02)
    assert abs(parts["source_low"]) < 0.05 * ledger["final_step_norm"]
    assert abs(parts["provider_transfer"]) < 1e-4


def test_ledger_attributes_a_transfer_created_seam_to_the_provider():
    source = _video(SOURCE_HW, _continuous_dc(), 1)
    provider = _upscale(source) + _video(TARGET_HW, _step(), 2)
    final = provider + 0.1 * _detail(TARGET_HW, 3)
    ledger = _pipeline(source, source, provider, provider[:, :, :PREFIX_T], final)
    parts = _contributions(ledger)
    assert parts["provider_transfer"] == pytest.approx(ledger["final_step_norm"], rel=0.02)
    assert abs(parts["source_low"]) < 0.05 * ledger["final_step_norm"]


def test_ledger_sums_exactly_and_reports_cancelling_components():
    source = _video(SOURCE_HW, _continuous_dc() + _step(7), 1)
    provider = _upscale(source) - _video(TARGET_HW, 0.9 * _step(7), 2)
    exact_offset = torch.zeros(1, C, PREFIX_T, 1, 1)
    exact_offset[:, :, -1] = 0.3 * torch.randn(C, 1, 1, generator=torch.Generator().manual_seed(8))
    exact_prefix = provider[:, :, :PREFIX_T] + exact_offset
    final = _with_prefix(provider, exact_prefix) + _video(TARGET_HW, 0.2 * _step(9), 3)
    ledger = _pipeline(source, source, provider, exact_prefix, final)
    rows = ledger["components"]
    assert sum(_contributions(ledger).values()) == pytest.approx(ledger["final_step_norm"], rel=1e-6)
    assert rows["source_low"]["cosine_to_final"] * rows["provider_transfer"]["cosine_to_final"] < 0
    assert ledger["component_norm_sum_over_final_norm"] > 2.0


def test_optional_boundary_pair_stage_and_failed_or_missing_stages():
    video = _video(TARGET_HW, _continuous_dc() + _step(), 1)
    receipt = measure_signed_boundary_dc(video, PREFIX_T)
    stages = ("source_low_native", "source_low_exact_context", "provider_native", "pre_high_exact_restored", "final")
    receipts = {stage: receipt for stage in stages}
    ledger = signed_boundary_dc_ledger(receipts)
    assert ledger["status"] == "complete" and "provider_boundary_pair_replacement" not in ledger["components"]
    replaced = signed_boundary_dc_ledger({**receipts, "provider_boundary_pair_replaced": receipt})
    assert replaced["stages"][3] == "provider_boundary_pair_replaced"
    failed = signed_boundary_dc_ledger({**receipts, "provider_native": {"status": "error", "error": "x"}})
    assert failed["status"] == "incomplete" and failed["failed_stages"] == ["provider_native"]
    missing = signed_boundary_dc_ledger({"final": receipt})
    assert missing["status"] == "incomplete" and "source_low_native" in missing["missing_stages"]


def test_signed_prefix_dc_and_preview_factor_validation():
    carried = _video(SOURCE_HW, _continuous_dc(), 1)[:, :, :PREFIX_T]
    shift = _step(4, 0, PREFIX_T)[:, :PREFIX_T].view(1, C, PREFIX_T, 1, 1)
    receipt = measure_signed_prefix_dc(carried, carried + shift)
    assert receipt["tokens"] == [PREFIX_T - 3, PREFIX_T - 2, PREFIX_T - 1]
    torch.testing.assert_close(torch.tensor(receipt["reconciled_minus_carried"][-1]), shift[0, :, -1, 0, 0])
    assert latent_preview_factors(type("F", (), {"latent_rgb_factors": [[0.0] * 3] * C})(), C).shape == (C, 3)
    assert latent_preview_factors(type("F", (), {"latent_rgb_factors": [[0.0] * 3] * 4})(), C) is None
    assert latent_preview_factors(None, C) is None
    assert measure_signed_boundary_dc(carried, 4)["preview"] == "unavailable"
    with pytest.raises(ValueError):
        measure_signed_boundary_dc(torch.zeros(1, C, 4, 2, 2), 4)
