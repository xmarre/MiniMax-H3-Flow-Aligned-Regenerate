from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_handoff import FakeLearnedProvider

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.runtime import FlowBinding, _run_progressive
from h3_flow_regenerate.seam_diagnostics import (
    measure_exact_prefix_splice,
    recover_conditional_clean_for_diagnostics,
)


def _packed_inputs(t=7, prefix=2, h=8, w=12):
    video = torch.randn(1, 24, t, h, w)
    audio = torch.randn(1, 32, 2, 11)
    packed, shapes = pack_streams((video, audio))
    mask_video = torch.ones_like(video)
    mask_video[:, :, :prefix] = 0
    mask = pack_streams((mask_video, torch.ones_like(audio)))[0]
    return packed, shapes, mask


def test_conditional_clean_recovery_is_exact_for_float32_contract():
    torch.manual_seed(3)
    clean = torch.randn(1, 24, 4, 8, 8)
    noise = torch.randn_like(clean)
    sigma = 0.8575096726417542
    state = (1.0 - sigma) * clean + sigma * noise
    recovered = recover_conditional_clean_for_diagnostics(state, noise, sigma=sigma)
    assert torch.allclose(recovered, clean, rtol=1e-5, atol=1e-5)


def test_splice_diagnostics_detect_exact_prefix_restore_amplification():
    learned = torch.zeros(1, 24, 4, 8, 8)
    learned[:, :, 1] = 1.0
    learned[:, :, 2] = 2.0
    learned[:, :, 3] = 3.0
    exact = learned[:, :, :2].clone()
    exact[:, :, 1] = 10.0

    fields = measure_exact_prefix_splice(learned, exact)

    assert fields["splice_diagnostic_version"] == 1
    assert fields["splice_prefix_temporal_length"] == 2
    assert fields["splice_lowpass_kernel"] == 5
    assert fields["upscaler_native_seam_rms"] == pytest.approx(1.0)
    assert fields["exact_restored_seam_rms"] == pytest.approx(8.0)
    assert fields["seam_rms_amplification"] == pytest.approx(8.0)
    assert fields["upscaler_native_seam_lowpass_rms"] == pytest.approx(1.0)
    assert fields["exact_restored_seam_lowpass_rms"] == pytest.approx(8.0)
    assert fields["seam_lowpass_amplification"] == pytest.approx(8.0)
    assert fields["upscaler_native_seam_spatial_mean_rms"] == pytest.approx(1.0)
    assert fields["exact_restored_seam_spatial_mean_rms"] == pytest.approx(8.0)
    assert fields["seam_spatial_mean_amplification"] == pytest.approx(8.0)
    assert fields["corrected_exact_seam_rms"] == pytest.approx(8.0)
    assert fields["corrected_exact_seam_lowpass_rms"] == pytest.approx(8.0)
    assert fields["corrected_exact_seam_spatial_mean_rms"] == pytest.approx(8.0)
    assert fields["corrected_over_uncorrected_exact_seam_rms_ratio"] == pytest.approx(1.0)
    assert fields["prefix_restore_rms_tail_1"] == pytest.approx(9.0)
    assert fields["prefix_restore_rms_tail_2"] == pytest.approx(9.0 / math.sqrt(2.0))


def test_mixed_runtime_emits_transfer_and_final_seam_diagnostics(monkeypatch):
    fake = ModuleType("comfy")
    fake.samplers = ModuleType("comfy.samplers")
    fake.samplers.KSAMPLER = lambda function, **kw: SimpleNamespace(sampler_function=function, extra_options={})
    monkeypatch.setitem(sys.modules, "comfy", fake)
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake.samplers)

    packed, shapes, mask = _packed_inputs()
    base = SimpleNamespace(process_latent_in=lambda value: value, diffusion_model=SimpleNamespace(blocks=[]))
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=base),
        conds={"positive": []},
    )
    binding = FlowBinding()
    provider = FakeLearnedProvider()
    config = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=6,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        suffix_dc_bridge=True,
    )

    def execute(noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):
        stage = guider.model_options["transformer_options"]["h3_flow_stage"]
        if stage == "high":
            assert latent_shapes == shapes
            assert torch.equal(call_mask, mask)
            binding.metrics.event("model_call", actual=True)
            return packed.clone()
        if stage == "probe":
            return latent.clone()
        return latent / (1.0 - sigmas[-1])

    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})
    result = _run_progressive(
        execute,
        guider,
        binding,
        config,
        torch.randn_like(packed),
        packed,
        sampler,
        torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
        mask,
        None,
        True,
        7,
        list(shapes),
    )
    assert torch.equal(result, packed)

    transfer = next(event for event in binding.metrics.events if event.kind == "mixed_grid_transfer")
    required_transfer = {
        "splice_diagnostic_version",
        "splice_diagnostic_elapsed_ms",
        "upscaler_native_seam_rms",
        "exact_restored_seam_rms",
        "seam_rms_amplification",
        "upscaler_native_seam_lowpass_rms",
        "exact_restored_seam_lowpass_rms",
        "seam_lowpass_amplification",
        "upscaler_native_seam_spatial_mean_rms",
        "exact_restored_seam_spatial_mean_rms",
        "seam_spatial_mean_amplification",
        "corrected_exact_seam_rms",
        "corrected_exact_seam_lowpass_rms",
        "corrected_exact_seam_spatial_mean_rms",
        "corrected_over_uncorrected_exact_seam_rms_ratio",
        "suffix_dc_bridge_enabled",
        "suffix_dc_bridge_corrected_tokens",
        "prefix_restore_rms_tail_1",
        "prefix_restore_rms_tail_2",
    }
    assert required_transfer <= transfer.fields.keys()
    assert transfer.fields["suffix_dc_bridge_enabled"] is True
    assert transfer.fields["suffix_dc_bridge_corrected_tokens"] == 1
    numeric_transfer = required_transfer - {"splice_diagnostic_version", "suffix_dc_bridge_enabled"}
    assert all(math.isfinite(float(transfer.fields[key])) for key in numeric_transfer)

    complete = next(event for event in binding.metrics.events if event.kind == "mixed_grid_complete")
    required_final = {
        "final_seam_rms",
        "final_seam_lowpass_rms",
        "final_seam_spatial_mean_rms",
        "final_over_transfer_exact_seam_rms_ratio",
        "final_over_transfer_exact_seam_lowpass_ratio",
        "final_over_transfer_exact_seam_spatial_mean_ratio",
        "final_over_transfer_corrected_seam_rms_ratio",
        "final_over_transfer_corrected_seam_lowpass_ratio",
        "final_over_transfer_corrected_seam_spatial_mean_ratio",
    }
    assert required_final <= complete.fields.keys()
    assert all(math.isfinite(float(complete.fields[key])) for key in required_final)


def test_mixed_runtime_suffix_dc_bridge_reports_native_dc_boundary(monkeypatch):
    fake = ModuleType("comfy")
    fake.samplers = ModuleType("comfy.samplers")
    fake.samplers.KSAMPLER = lambda function, **kw: SimpleNamespace(sampler_function=function, extra_options={})
    monkeypatch.setitem(sys.modules, "comfy", fake)
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake.samplers)

    packed, shapes, mask = _packed_inputs()
    base = SimpleNamespace(process_latent_in=lambda value: value, diffusion_model=SimpleNamespace(blocks=[]))
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=base),
        conds={"positive": []},
    )
    binding = FlowBinding()
    config = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=6,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=FakeLearnedProvider(),
        suffix_dc_bridge=True,
    )

    def execute(noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):
        stage = guider.model_options["transformer_options"]["h3_flow_stage"]
        if stage == "high":
            binding.metrics.event("model_call", actual=True)
            return packed.clone()
        if stage == "probe":
            return latent.clone()
        return latent / (1.0 - sigmas[-1])

    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})
    _run_progressive(
        execute,
        guider,
        binding,
        config,
        torch.randn_like(packed),
        packed,
        sampler,
        torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
        mask,
        None,
        True,
        7,
        list(shapes),
    )

    transfer = next(event for event in binding.metrics.events if event.kind == "mixed_grid_transfer")
    assert transfer.fields["suffix_dc_bridge_enabled"] is True
    assert transfer.fields["suffix_dc_bridge_corrected_tokens"] == 1
    assert transfer.fields["suffix_dc_bridge_first_weight"] == pytest.approx(1.0)
    assert transfer.fields["corrected_exact_seam_spatial_mean_rms"] == pytest.approx(
        transfer.fields["upscaler_native_seam_spatial_mean_rms"], abs=2e-5
    )
    assert transfer.fields["suffix_dc_bridge_state_mapping"] == "affine_equivalent_pre_renoise"
