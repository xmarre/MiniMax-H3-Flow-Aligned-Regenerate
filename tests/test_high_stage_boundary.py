import itertools
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.high_stage_boundary import (
    HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS,
    HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS,
    HIGH_BOUNDARY_AUDIO_REFERENCE_WEIGHTS,
    HIGH_BOUNDARY_AUDIO_RELEASE_TICKS,
    HIGH_BOUNDARY_AUDIO_SEAM_TICKS,
    HIGH_BOUNDARY_REFERENCE_POLICY,
    HIGH_BOUNDARY_VIDEO_FULL_REFERENCE_TOKENS,
    HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS,
    HIGH_BOUNDARY_VIDEO_RELEASE_TOKENS,
    align_audio_reference_to_authoritative_prefix,
    apply_final_boundary_reference,
    high_boundary_contract,
)
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FLOW_STAGE_KEY, FlowBinding, flow_predict_wrapper


def test_boundary_trace_is_bounded_nonmutating_and_cleans_up_on_exception():
    video = torch.randn(1, 24, 12, 16, 16)
    audio = torch.randn(1, 32, 2, 40)
    packed, shapes = pack_streams((video, audio))
    original = packed.clone()
    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        metrics=H3FlowMetrics(),
    )
    with (
        pytest.raises(RuntimeError, match="sampler failure"),
        high_boundary_contract(binding, video[:, :, :3], shapes, measure=True),
    ):
        assert binding.guidance_protected_prefix_t == 3
        trace = binding.high_boundary_trace
        for point in ("before_flow", "after_flow"):
            trace.observe(packed, point=point, call_index=0, sigma=0.5, actual=False)
        assert trace.previous_prediction is None
        trace.observe(packed, point="before_flow", call_index=16, sigma=0.1, actual=True)
        assert len(binding.metrics.events) == 2
        assert binding.metrics.events[-1].fields["flow_suffix_delta_rms"] == 0.0
        assert binding.metrics.events[-1].fields["suffix_tokens"] == 4
        assert binding.metrics.events[-1].fields["actual"] is False
        assert torch.equal(packed, original)
        raise RuntimeError("sampler failure")
    assert binding.guidance_protected_prefix_t == 0
    assert binding.high_boundary_trace is None


def test_ownership_without_measurement_and_nested_rejection():
    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        metrics=H3FlowMetrics(),
    )
    prefix = torch.zeros(1, 24, 3, 4, 4)
    with high_boundary_contract(binding, prefix, [], measure=False):
        assert binding.high_boundary_trace is None
        with (
            pytest.raises(RuntimeError, match="nested"),
            high_boundary_contract(binding, prefix, [], measure=False),
        ):
            pass
        assert binding.guidance_protected_prefix_t == 3
    assert binding.guidance_protected_prefix_t == 0


def test_boundary_reference_anchor_changes_only_bounded_generated_clean_support():
    torch.manual_seed(7)
    video = torch.randn(1, 24, 12, 6, 6)
    audio = torch.randn(1, 32, 2, 100)
    packed, shapes = pack_streams((video, audio))
    original = packed.clone()
    prefix_t = 3
    audio_prefix = 6

    video_reference = video[:, :, prefix_t : prefix_t + len(HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS)].clone()
    video_reference += torch.randn_like(video_reference) * 0.25
    audio_reference = audio.clone()
    audio_reference[..., audio_prefix : audio_prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS] += (
        torch.randn_like(audio_reference[..., audio_prefix : audio_prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS]) * 0.4
    )

    video_mask = torch.ones_like(video)
    video_mask[:, :, :prefix_t] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :audio_prefix] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]

    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        metrics=H3FlowMetrics(),
    )
    with high_boundary_contract(
        binding,
        video[:, :, :prefix_t],
        shapes,
        measure=False,
        video_reference_suffix=video_reference,
        audio_reference=audio_reference,
        exact_denoise_mask=exact_mask,
    ):
        anchor = binding.high_boundary_anchor
        assert anchor is not None
        result = anchor.apply(packed, call_index=0, sigma=0.8, actual=True)

    assert torch.equal(packed, original)
    result_video, result_audio = unpack_streams(result, shapes)
    original_video, original_audio = unpack_streams(original, shapes)
    assert torch.equal(result_video[:, :, :prefix_t], original_video[:, :, :prefix_t])
    video_stop = prefix_t + len(HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS)
    assert torch.equal(result_video[:, :, video_stop:], original_video[:, :, video_stop:])
    assert torch.equal(result_audio[..., :audio_prefix], original_audio[..., :audio_prefix])
    assert torch.equal(
        result_audio[..., audio_prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS :],
        original_audio[..., audio_prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS :],
    )
    assert torch.equal(result_video[:, :, prefix_t], video_reference[:, :, 0])
    for offset, weight in enumerate(HIGH_BOUNDARY_AUDIO_REFERENCE_WEIGHTS):
        original_tick = original_audio[..., audio_prefix + offset].float()
        reference_tick = audio_reference[..., audio_prefix + offset].float()
        expected_audio = (original_tick + weight * (reference_tick - original_tick)).to(result_audio.dtype)
        torch.testing.assert_close(result_audio[..., audio_prefix + offset], expected_audio, rtol=0, atol=0)

    for offset, weight in enumerate(HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS[1:], start=1):
        expected_video = (
            original_video[:, :, prefix_t + offset].float()
            + weight * (video_reference[:, :, offset].float() - original_video[:, :, prefix_t + offset].float())
        ).to(result_video.dtype)
        torch.testing.assert_close(result_video[:, :, prefix_t + offset], expected_video, rtol=0, atol=0)

    anchor_events = [
        event for event in binding.metrics.events if event.kind == "partitioned_high_boundary_reference_anchor"
    ]
    assert len(anchor_events) == 1
    fields = anchor_events[0].fields
    assert fields["policy"] == HIGH_BOUNDARY_REFERENCE_POLICY
    assert fields["video_support_tokens"] == len(HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS)
    assert fields["audio_support_ticks"] == HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS
    assert fields["video_temporal_weights"] == list(HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS)
    assert fields["audio_temporal_weights"] == list(HIGH_BOUNDARY_AUDIO_REFERENCE_WEIGHTS)
    assert fields["video_first_reference_error_rms"] == 0.0
    assert fields["audio_first_reference_error_rms"] == 0.0
    assert fields["authoritative_prefix_modified"] is False
    assert fields["suffix_outside_support_modified"] is False
    assert binding.high_boundary_anchor is None


def test_video_reference_covers_boundary_decode_chunk_and_overlap_release():
    weights = HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS
    assert HIGH_BOUNDARY_VIDEO_FULL_REFERENCE_TOKENS == 5
    assert HIGH_BOUNDARY_VIDEO_RELEASE_TOKENS == 2
    assert weights[:HIGH_BOUNDARY_VIDEO_FULL_REFERENCE_TOKENS] == (1.0,) * 5
    assert weights[HIGH_BOUNDARY_VIDEO_FULL_REFERENCE_TOKENS:] == pytest.approx(
        (2.0 / 3.0, 1.0 / 3.0)
    )


def test_audio_reference_weights_hold_decoder_window_then_release_monotonically():
    weights = HIGH_BOUNDARY_AUDIO_REFERENCE_WEIGHTS
    assert len(weights) == HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS
    assert weights[:HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS] == (1.0,) * HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS

    release = weights[HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS:]
    assert len(release) == HIGH_BOUNDARY_AUDIO_RELEASE_TICKS
    assert release[0] == pytest.approx(HIGH_BOUNDARY_AUDIO_RELEASE_TICKS / (HIGH_BOUNDARY_AUDIO_RELEASE_TICKS + 1))
    assert release[-1] == pytest.approx(1.0 / (HIGH_BOUNDARY_AUDIO_RELEASE_TICKS + 1))
    assert all(left > right for left, right in itertools.pairwise(release))
    assert HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS == HIGH_BOUNDARY_AUDIO_SEAM_TICKS + 32


def test_audio_reference_alignment_preserves_clean_edge_on_authoritative_prefix_gauge():
    torch.manual_seed(17)
    video = torch.zeros(1, 24, 4, 2, 2)
    low_probe = torch.randn(1, 32, 2, 100)
    authoritative = low_probe.clone()
    prefix = 6
    authoritative[..., :prefix] += 0.75

    video_mask = torch.ones_like(video)
    audio_mask = torch.ones_like(low_probe)
    audio_mask[..., :prefix] = 0
    exact_mask, shapes = pack_streams((video_mask, audio_mask))

    reference, report = align_audio_reference_to_authoritative_prefix(
        low_probe,
        authoritative,
        exact_mask,
        shapes,
    )
    translation = authoritative[..., prefix - 1].float() - low_probe[..., prefix - 1].float()

    assert torch.equal(reference[..., :prefix], authoritative[..., :prefix])
    expected = (
        low_probe[..., prefix : prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS].float() + translation.unsqueeze(-1)
    ).to(reference.dtype)
    torch.testing.assert_close(
        reference[..., prefix : prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS],
        expected,
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        reference[..., prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS :],
        low_probe[..., prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS :],
    )
    source_edge = low_probe[..., prefix].float() - low_probe[..., prefix - 1].float()
    aligned_edge = reference[..., prefix].float() - authoritative[..., prefix - 1].float()
    torch.testing.assert_close(aligned_edge, source_edge, rtol=0, atol=1e-6)
    assert report["audio_prefix_ticks"] == prefix
    assert report["audio_support_ticks"] == HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS
    assert report["audio_prefix_translation_rms"] == pytest.approx(0.75)
    assert report["audio_first_edge_relation_error_rms"] < 1e-6
    assert report["reference_domain"] == "authoritative_prefix_aligned_low_probe_clean"


def test_boundary_reference_anchor_rejects_noncanonical_audio_mask():
    video = torch.zeros(1, 24, 7, 4, 4)
    audio = torch.zeros(1, 32, 2, 10)
    packed, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    video_mask[:, :, :3] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :5] = 0
    audio_mask[..., 7] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]
    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        metrics=H3FlowMetrics(),
    )

    with (
        pytest.raises(ValueError, match="contiguous exact prefix"),
        high_boundary_contract(
            binding,
            video[:, :, :3],
            shapes,
            measure=False,
            audio_reference=audio,
            exact_denoise_mask=exact_mask,
        ),
    ):
        pass
    assert torch.equal(packed, pack_streams((video, audio))[0])
    assert binding.guidance_protected_prefix_t == 0
    assert binding.high_boundary_anchor is None


def test_runtime_applies_boundary_reference_before_flow_observation():
    torch.manual_seed(11)
    video = torch.randn(1, 24, 9, 16, 16)
    audio = torch.randn(1, 32, 2, 14)
    packed, shapes = pack_streams((video, audio))
    prefix_t = 3
    video_reference = video[:, :, prefix_t : prefix_t + 4].clone()
    video_reference[:, :, 0] += 0.5

    binding = FlowBinding()
    guider = SimpleNamespace(model_options={FLOW_BINDING_KEY: binding})

    class Executor:
        class_obj = guider

        def __call__(self, x, timestep, model_options, seed):
            del timestep, model_options, seed
            return x.clone()

    model_options = {"transformer_options": {FLOW_STAGE_KEY: "high"}}
    with high_boundary_contract(
        binding,
        video[:, :, :prefix_t],
        shapes,
        measure=True,
        video_reference_suffix=video_reference,
    ):
        result = flow_predict_wrapper(
            Executor(),
            packed,
            torch.tensor([0.8]),
            model_options=model_options,
            seed=17,
        )

    result_video, _ = unpack_streams(result, shapes)
    assert torch.equal(result_video[:, :, prefix_t], video_reference[:, :, 0])
    points = [
        event.fields["point"]
        for event in binding.metrics.events
        if event.kind == "partitioned_high_boundary_prediction"
    ]
    assert points == ["before_anchor", "before_flow", "after_flow"]
    ordered_kinds = [event.kind for event in binding.metrics.events]
    before_anchor_index = next(
        index
        for index, event in enumerate(binding.metrics.events)
        if event.kind == "partitioned_high_boundary_prediction" and event.fields["point"] == "before_anchor"
    )
    anchor_index = ordered_kinds.index("partitioned_high_boundary_reference_anchor")
    before_flow_index = next(
        index
        for index, event in enumerate(binding.metrics.events)
        if event.kind == "partitioned_high_boundary_prediction" and event.fields["point"] == "before_flow"
    )
    assert before_anchor_index < anchor_index < before_flow_index


def test_final_boundary_reference_is_bounded_and_closes_first_video_and_audio_edges():
    torch.manual_seed(23)
    video = torch.randn(1, 24, 10, 6, 6)
    audio = torch.randn(1, 32, 2, 100)
    packed, shapes = pack_streams((video, audio))
    prefix_t = 3
    audio_prefix = 7

    video_reference = video[:, :, prefix_t : prefix_t + len(HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS)].clone() + 0.4
    audio_reference = audio.clone()
    audio_reference[..., audio_prefix : audio_prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS] += 0.25

    video_mask = torch.ones_like(video)
    video_mask[:, :, :prefix_t] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :audio_prefix] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]

    result, report = apply_final_boundary_reference(
        packed,
        shapes,
        video_prefix_t=prefix_t,
        video_reference_suffix=video_reference,
        audio_reference=audio_reference,
        exact_denoise_mask=exact_mask,
    )
    result_video, result_audio = unpack_streams(result, shapes)

    assert torch.equal(result_video[:, :, :prefix_t], video[:, :, :prefix_t])
    assert torch.equal(result_video[:, :, prefix_t], video_reference[:, :, 0])
    video_stop = prefix_t + len(HIGH_BOUNDARY_VIDEO_REFERENCE_WEIGHTS)
    assert torch.equal(result_video[:, :, video_stop:], video[:, :, video_stop:])
    assert torch.equal(result_audio[..., :audio_prefix], audio[..., :audio_prefix])
    full_stop = audio_prefix + HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS
    support_stop = audio_prefix + HIGH_BOUNDARY_AUDIO_REFERENCE_TICKS
    assert torch.equal(
        result_audio[..., audio_prefix:full_stop],
        audio_reference[..., audio_prefix:full_stop],
    )
    release_weights = HIGH_BOUNDARY_AUDIO_REFERENCE_WEIGHTS[HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS:]
    for offset, weight in enumerate(release_weights, start=HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS):
        original_tick = audio[..., audio_prefix + offset].float()
        reference_tick = audio_reference[..., audio_prefix + offset].float()
        expected = (original_tick + weight * (reference_tick - original_tick)).to(result_audio.dtype)
        torch.testing.assert_close(result_audio[..., audio_prefix + offset], expected, rtol=0, atol=0)
    assert torch.equal(result_audio[..., support_stop:], audio[..., support_stop:])
    assert report["video_first_reference_error_rms"] == 0.0
    assert report["audio_full_reference_error_rms"] == 0.0
    assert report["audio_full_reference_ticks"] == HIGH_BOUNDARY_AUDIO_FULL_REFERENCE_TICKS
