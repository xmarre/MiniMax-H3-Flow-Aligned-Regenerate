from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.comfy_compat import flow_outer_wrapper_with_exact_mask
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.pr32_audio_guided_overlap import (
    AUDIO_GUIDED_OVERLAP_ENV,
    apply_audio_guided_overlap_mask,
    configured_audio_guided_overlap_ticks,
)
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


def test_audio_guided_overlap_is_disabled_without_explicit_env(monkeypatch):
    monkeypatch.delenv(AUDIO_GUIDED_OVERLAP_ENV, raising=False)
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


def test_audio_guided_overlap_all_generated_audio_is_expected_noop():
    packed, shapes, mask = _packed_case(audio_prefix=0)

    runtime_mask, report = apply_audio_guided_overlap_mask(mask, shapes, ticks=4)

    assert runtime_mask is mask
    assert report["applied"] is False
    assert report["reason"] == "no_exact_audio_prefix"
    assert report["audio_prefix_ticks"] == 0
    assert packed.shape == mask.shape


def test_audio_guided_overlap_rejects_noncanonical_partial_audio_mask():
    _packed, shapes, mask = _packed_case(audio_prefix=5)
    _video_mask, audio_mask = unpack_streams(mask, shapes)
    audio_mask[..., 7] = 0

    with pytest.raises(ValueError, match="contiguous exact audio prefix"):
        apply_audio_guided_overlap_mask(mask, shapes, ticks=4)


def test_fallback_sampler_sees_guided_audio_but_return_restores_original_exact_prefix(monkeypatch):
    monkeypatch.setenv(AUDIO_GUIDED_OVERLAP_ENV, "4")
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

    # The sampler was allowed to see a guided audio context, but the existing
    # exact-output canonicalizer still owns the caller-visible Native Masked
    # contract and restores every original mask==0 element exactly.
    assert torch.equal(result, packed)
    guided_events = [event for event in binding.metrics.events if event.kind == "pr32_audio_guided_overlap"]
    assert len(guided_events) == 1
    assert guided_events[0].fields["applied"] is True
    assert guided_events[0].fields["requested_ticks"] == 4
    assert guided_events[0].fields["audio_prefix_ticks"] == 5
    assert guided_events[0].fields["mask_grid"] == "ceil_1_over_256"
    exact_events = [event for event in binding.metrics.events if event.kind == "exact_mask_output"]
    assert len(exact_events) == 1
    assert exact_events[0].fields["source"] == "target_input_fallback_return"
    assert exact_events[0].fields["final_exact"] is True
