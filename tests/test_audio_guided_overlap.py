from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.audio_guided_overlap import (
    AUDIO_GUIDED_OVERLAP_ENV,
    DEFAULT_AUDIO_GUIDED_OVERLAP_TICKS,
    apply_audio_guided_overlap_mask,
    configured_audio_guided_overlap_ticks,
    measure_audio_latent_boundary,
    validate_audio_guided_overlap_ticks,
)
from h3_flow_regenerate.comfy_compat import flow_outer_wrapper_with_exact_mask
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, PROGRESSIVE_KEY, FlowBinding


def _packed_case(*, audio_t=11, audio_prefix=5, video_t=7, video_prefix=2, h=8, w=12):
    video = torch.randn(1, 24, video_t, h, w)
    audio = torch.randn(1, 32, 2, audio_t)
    packed, shapes = pack_streams((video, audio))

    video_mask = torch.ones_like(video)
    video_mask[:, :, :video_prefix] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :audio_prefix] = 0
    mask = pack_streams((video_mask, audio_mask))[0]
    return packed, list(shapes), mask


def _core_grid_ramp(dtype):
    return torch.tensor([52 / 256, 103 / 256, 154 / 256, 205 / 256], dtype=dtype)


def test_audio_guided_overlap_defaults_to_validated_four_ticks(monkeypatch):
    monkeypatch.delenv(AUDIO_GUIDED_OVERLAP_ENV, raising=False)
    assert DEFAULT_AUDIO_GUIDED_OVERLAP_TICKS == 4
    assert configured_audio_guided_overlap_ticks() == 4


def test_audio_guided_overlap_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv(AUDIO_GUIDED_OVERLAP_ENV, "0")
    assert configured_audio_guided_overlap_ticks() == 0


def test_audio_guided_overlap_changes_only_tail_of_exact_audio_prefix():
    _packed, shapes, mask = _packed_case(audio_prefix=5)
    original = mask.clone()

    runtime_mask, report = apply_audio_guided_overlap_mask(mask, shapes, ticks=4)

    assert runtime_mask is not mask
    assert torch.equal(mask, original)
    old_video, old_audio = unpack_streams(original, shapes)
    new_video, new_audio = unpack_streams(runtime_mask, shapes)
    assert torch.equal(new_video, old_video)
    assert torch.equal(new_audio[..., :1], old_audio[..., :1])
    expected = _core_grid_ramp(new_audio.dtype)
    torch.testing.assert_close(new_audio[0, 0, 0, 1:5], expected)
    torch.testing.assert_close(new_audio[0, -1, 1, 1:5], expected)
    assert torch.equal(new_audio[..., 5:], old_audio[..., 5:])
    assert report["applied"] is True
    assert report["audio_prefix_ticks"] == 5
    assert report["ramp_start_tick"] == 1
    assert report["ramp_stop_tick"] == 5
    assert report["mask_grid"] == "ceil_1_over_256"
    assert report["original_exact_output_restore"] is True


def test_audio_guided_overlap_supports_explicit_32_tick_runtime_ramp():
    _packed, shapes, mask = _packed_case(audio_t=80, audio_prefix=65)
    original = mask.clone()

    runtime_mask, report = apply_audio_guided_overlap_mask(mask, shapes, ticks=32)

    assert runtime_mask is not mask
    assert torch.equal(mask, original)
    old_video, old_audio = unpack_streams(original, shapes)
    new_video, new_audio = unpack_streams(runtime_mask, shapes)
    assert torch.equal(new_video, old_video)
    assert torch.equal(new_audio[..., :33], old_audio[..., :33])
    assert torch.equal(new_audio[..., 65:], old_audio[..., 65:])
    ramp = new_audio[0, 0, 0, 33:65].to(dtype=torch.float32)
    expected_raw = torch.arange(1, 33, dtype=torch.float32) / 33.0
    expected = torch.ceil(expected_raw * 256.0) / 256.0
    torch.testing.assert_close(ramp, expected)
    assert report["applied"] is True
    assert report["audio_prefix_ticks"] == 65
    assert report["ramp_start_tick"] == 33
    assert report["ramp_stop_tick"] == 65
    assert len(report["ramp_values"]) == 32


def test_audio_guided_overlap_all_generated_audio_is_expected_noop():
    packed, shapes, mask = _packed_case(audio_prefix=0)

    runtime_mask, report = apply_audio_guided_overlap_mask(mask, shapes, ticks=4)

    assert runtime_mask is mask
    assert report["applied"] is False
    assert report["reason"] == "no_exact_audio_prefix"
    assert report["audio_prefix_ticks"] == 0
    assert packed.shape == mask.shape


def test_audio_guided_overlap_all_protected_audio_is_expected_noop():
    _packed, shapes, mask = _packed_case(audio_t=11, audio_prefix=11)
    original = mask.clone()

    runtime_mask, report = apply_audio_guided_overlap_mask(mask, shapes, ticks=4)

    assert runtime_mask is mask
    assert torch.equal(mask, original)
    assert report["applied"] is False
    assert report["reason"] == "no_generated_audio_suffix"
    assert report["audio_prefix_ticks"] == 11


def test_audio_guided_overlap_short_exact_prefix_preserves_native_path():
    _packed, shapes, mask = _packed_case(audio_prefix=4)
    original = mask.clone()

    runtime_mask, report = apply_audio_guided_overlap_mask(mask, shapes, ticks=4)

    assert runtime_mask is mask
    assert torch.equal(mask, original)
    assert report["applied"] is False
    assert report["reason"] == "exact_audio_prefix_too_short"
    assert report["audio_prefix_ticks"] == 4


def test_audio_guided_overlap_rejects_noncanonical_partial_audio_mask():
    _packed, shapes, mask = _packed_case(audio_prefix=5)
    _video_mask, audio_mask = unpack_streams(mask, shapes)
    audio_mask[..., 7] = 0

    with pytest.raises(ValueError, match="contiguous exact audio prefix"):
        apply_audio_guided_overlap_mask(mask, shapes, ticks=4)


def test_fallback_sampler_sees_guided_audio_but_return_restores_original_exact_prefix(monkeypatch):
    monkeypatch.delenv(AUDIO_GUIDED_OVERLAP_ENV, raising=False)
    packed, shapes, original_mask = _packed_case(audio_prefix=5)
    binding = FlowBinding()
    config = ProgressiveTargetInputConfig(source_latent_h=4, source_latent_w=6)
    base = SimpleNamespace(process_latent_in=lambda value: value, diffusion_model=SimpleNamespace(blocks=[]))
    guider = SimpleNamespace(
        model_options={
            "transformer_options": {},
            FLOW_BINDING_KEY: binding,
            PROGRESSIVE_KEY: config,
        },
        model_patcher=SimpleNamespace(model=base),
        conds={"positive": []},
    )

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):
            assert latent_shapes == shapes
            original_video_mask, original_audio_mask = unpack_streams(original_mask, shapes)
            runtime_video_mask, runtime_audio_mask = unpack_streams(call_mask, shapes)
            assert torch.equal(runtime_video_mask, original_video_mask)
            expected = _core_grid_ramp(runtime_audio_mask.dtype)
            torch.testing.assert_close(runtime_audio_mask[0, 0, 0, 1:5], expected)
            assert torch.equal(runtime_audio_mask[..., 5:], original_audio_mask[..., 5:])

            result = packed.clone()
            result_video, result_audio = unpack_streams(result, shapes)
            result_video[:, :, :2] += 1.0
            result_audio[..., :5] += 1.0
            return result

    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})
    result = flow_outer_wrapper_with_exact_mask(
        Executor(),
        torch.randn_like(packed),
        packed,
        sampler,
        torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
        original_mask,
        None,
        True,
        7,
        latent_shapes=shapes,
    )

    assert torch.equal(result, packed)
    guided_events = [event for event in binding.metrics.events if event.kind == "audio_guided_overlap"]
    assert len(guided_events) == 1
    assert guided_events[0].fields["applied"] is True
    assert guided_events[0].fields["requested_ticks"] == 4
    assert guided_events[0].fields["audio_prefix_ticks"] == 5
    assert guided_events[0].fields["mask_grid"] == "ceil_1_over_256"
    exact_events = [event for event in binding.metrics.events if event.kind == "exact_mask_output"]
    assert len(exact_events) == 1
    assert exact_events[0].fields["source"] == "target_input_fallback_return"
    assert exact_events[0].fields["final_exact"] is True


def test_explicit_audio_guided_overlap_validation_rejects_bool_and_out_of_range_values():
    assert validate_audio_guided_overlap_ticks(0) == 0
    assert validate_audio_guided_overlap_ticks(16) == 16
    with pytest.raises(ValueError):
        validate_audio_guided_overlap_ticks(True)
    with pytest.raises(ValueError):
        validate_audio_guided_overlap_ticks(-1)
    with pytest.raises(ValueError):
        validate_audio_guided_overlap_ticks(17)


def test_audio_latent_boundary_measurement_reports_fixed_physical_windows():
    video = torch.zeros(1, 24, 3, 4, 4)
    audio = torch.ones(1, 32, 2, 50)
    audio[..., 25:] = 2.0
    packed, shapes = pack_streams((video, audio))

    video_mask = torch.ones_like(video)
    video_mask[:, :, :1] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :25] = 0
    mask = pack_streams((video_mask, audio_mask))[0]

    report = measure_audio_latent_boundary(
        packed,
        list(shapes),
        mask,
        windows=(4, 20),
    )

    assert report["available"] is True
    assert report["audio_prefix_ticks"] == 25
    assert report["audio_total_ticks"] == 50
    assert set(report["windows"]) == {"4", "20"}
    for key, expected_ms in (("4", 100.0), ("20", 500.0)):
        window = report["windows"][key]
        assert window["duration_ms_at_40hz"] == expected_ms
        assert window["pre_rms"] == pytest.approx(1.0)
        assert window["post_rms"] == pytest.approx(2.0)
        assert window["post_over_pre_rms_ratio"] == pytest.approx(2.0)
        assert window["post_over_pre_db"] == pytest.approx(6.020599913, rel=1e-6)
        assert window["pre_mean_abs"] == pytest.approx(1.0)
        assert window["post_mean_abs"] == pytest.approx(2.0)


def test_audio_latent_boundary_measurement_uses_original_exact_mask_not_guided_mask():
    packed, shapes, exact_mask = _packed_case(audio_t=12, audio_prefix=6)
    guided_mask, guided = apply_audio_guided_overlap_mask(exact_mask, shapes, ticks=4)
    assert guided["applied"] is True

    exact_report = measure_audio_latent_boundary(packed, shapes, exact_mask, windows=(4,))

    assert exact_report["audio_prefix_ticks"] == 6
    assert exact_report["available"] is True
    with pytest.raises(ValueError, match="contiguous exact audio prefix"):
        measure_audio_latent_boundary(packed, shapes, guided_mask, windows=(4,))
