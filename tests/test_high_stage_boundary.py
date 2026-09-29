from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.high_stage_boundary import (
    HIGH_BOUNDARY_REFERENCE_POLICY,
    HIGH_BOUNDARY_REFERENCE_WEIGHTS,
    HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
    HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS,
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
        high_prediction_bridge=None,
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


def test_guarded_context_extends_guidance_ownership_and_trace_witness():
    torch.manual_seed(29)
    video = torch.randn(1, 24, 9, 8, 8)
    audio = torch.randn(1, 32, 2, 12)
    packed, shapes = pack_streams((video, audio))
    guarded_context = video[:, :, :4].clone()
    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        high_prediction_bridge=None,
        metrics=H3FlowMetrics(),
    )

    with high_boundary_contract(
        binding,
        guarded_context,
        shapes,
        measure=True,
        prefix_witness="protected_first_generated_guard_tail_after_inpaint_restore",
    ):
        assert binding.guidance_protected_prefix_t == 4
        trace = binding.high_boundary_trace
        assert trace is not None
        assert trace.prefix_t == 4
        trace.observe(
            packed,
            point="before_flow",
            call_index=0,
            sigma=0.8,
            actual=True,
        )

    event = next(event for event in binding.metrics.events if event.kind == "partitioned_high_boundary_prediction")
    assert event.fields["prefix_t"] == 4
    assert event.fields["prefix_witness"] == "protected_first_generated_guard_tail_after_inpaint_restore"
    assert binding.guidance_protected_prefix_t == 0


def test_ownership_without_measurement_and_nested_rejection():
    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        high_prediction_bridge=None,
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
    video = torch.randn(1, 24, 9, 6, 6)
    audio = torch.randn(1, 32, 2, 14)
    packed, shapes = pack_streams((video, audio))
    original = packed.clone()
    prefix_t = 3
    audio_prefix = 6

    video_reference = video[:, :, prefix_t : prefix_t + 4].clone()
    video_reference += torch.randn_like(video_reference) * 0.25
    audio_reference = audio.clone()
    audio_reference[..., audio_prefix : audio_prefix + 4] += (
        torch.randn_like(audio_reference[..., audio_prefix : audio_prefix + 4]) * 0.4
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
        high_prediction_bridge=None,
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
    assert torch.equal(result_video[:, :, prefix_t + 4 :], original_video[:, :, prefix_t + 4 :])
    assert torch.equal(result_audio[..., :audio_prefix], original_audio[..., :audio_prefix])
    assert torch.equal(result_audio[..., audio_prefix + 4 :], original_audio[..., audio_prefix + 4 :])
    assert torch.equal(result_video[:, :, prefix_t], video_reference[:, :, 0])
    assert torch.equal(result_audio[..., audio_prefix], audio_reference[..., audio_prefix])

    for offset, weight in enumerate(HIGH_BOUNDARY_REFERENCE_WEIGHTS[1:], start=1):
        expected_video = (
            original_video[:, :, prefix_t + offset].float()
            + weight * (video_reference[:, :, offset].float() - original_video[:, :, prefix_t + offset].float())
        ).to(result_video.dtype)
        expected_audio = (
            original_audio[..., audio_prefix + offset].float()
            + weight
            * (audio_reference[..., audio_prefix + offset].float() - original_audio[..., audio_prefix + offset].float())
        ).to(result_audio.dtype)
        torch.testing.assert_close(result_video[:, :, prefix_t + offset], expected_video, rtol=0, atol=0)
        torch.testing.assert_close(result_audio[..., audio_prefix + offset], expected_audio, rtol=0, atol=0)

    anchor_events = [
        event for event in binding.metrics.events if event.kind == "partitioned_high_boundary_reference_anchor"
    ]
    assert len(anchor_events) == 1
    fields = anchor_events[0].fields
    assert fields["policy"] == HIGH_BOUNDARY_REFERENCE_POLICY
    assert fields["video_support_tokens"] == 4
    assert fields["audio_support_ticks"] == 4
    assert fields["video_temporal_weights"] == list(HIGH_BOUNDARY_REFERENCE_WEIGHTS)
    assert fields["audio_temporal_weights"] == list(HIGH_BOUNDARY_REFERENCE_WEIGHTS)
    assert fields["video_first_reference_error_rms"] == 0.0
    assert fields["audio_first_reference_error_rms"] == 0.0
    assert fields["authoritative_prefix_modified"] is False
    assert fields["suffix_outside_support_modified"] is False
    assert binding.high_boundary_anchor is None


def test_boundary_reference_anchor_supports_single_video_successor():
    torch.manual_seed(19)
    video = torch.randn(1, 24, 8, 6, 6)
    audio = torch.randn(1, 32, 2, 12)
    packed, shapes = pack_streams((video, audio))
    original = packed.clone()
    prefix_t = 3
    video_reference = video[:, :, prefix_t : prefix_t + 1].clone()
    video_reference += 0.375

    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        high_prediction_bridge=None,
        metrics=H3FlowMetrics(),
    )
    with high_boundary_contract(
        binding,
        video[:, :, :prefix_t],
        shapes,
        measure=False,
        video_reference_suffix=video_reference,
    ):
        anchor = binding.high_boundary_anchor
        assert anchor is not None
        result = anchor.apply(packed, call_index=0, sigma=0.8, actual=True)

    result_video, result_audio = unpack_streams(result, shapes)
    original_video, original_audio = unpack_streams(original, shapes)
    assert torch.equal(result_video[:, :, :prefix_t], original_video[:, :, :prefix_t])
    assert torch.equal(result_video[:, :, prefix_t], video_reference[:, :, 0])
    assert torch.equal(result_video[:, :, prefix_t + 1 :], original_video[:, :, prefix_t + 1 :])
    assert torch.equal(result_audio, original_audio)

    event = next(
        event for event in binding.metrics.events if event.kind == "partitioned_high_boundary_reference_anchor"
    )
    assert event.fields["video_support_tokens"] == 1
    assert event.fields["video_temporal_weights"] == [1.0]
    assert event.fields["video_first_reference_error_rms"] == 0.0
    assert event.fields["suffix_outside_support_modified"] is False


def test_audio_only_boundary_reference_leaves_video_bit_exact():
    torch.manual_seed(23)
    video = torch.randn(1, 24, 8, 6, 6)
    audio = torch.randn(1, 32, 2, 14)
    packed, shapes = pack_streams((video, audio))
    prefix_t = 3
    audio_prefix = 6
    audio_reference = audio.clone()
    audio_reference[..., audio_prefix : audio_prefix + 4] += 0.25

    video_mask = torch.ones_like(video)
    video_mask[:, :, :prefix_t] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :audio_prefix] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]

    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        high_prediction_bridge=None,
        metrics=H3FlowMetrics(),
    )
    with high_boundary_contract(
        binding,
        video[:, :, :prefix_t],
        shapes,
        measure=False,
        video_reference_suffix=None,
        audio_reference=audio_reference,
        exact_denoise_mask=exact_mask,
    ):
        anchor = binding.high_boundary_anchor
        assert anchor is not None
        result = anchor.apply(packed, call_index=0, sigma=0.8, actual=True)

    result_video, result_audio = unpack_streams(result, shapes)
    assert torch.equal(result_video, video)
    assert torch.equal(result_audio[..., :audio_prefix], audio[..., :audio_prefix])
    assert torch.equal(result_audio[..., audio_prefix], audio_reference[..., audio_prefix])
    event = next(
        event for event in binding.metrics.events if event.kind == "partitioned_high_boundary_reference_anchor"
    )
    assert event.fields["video_applied"] is False
    assert event.fields["video_support_tokens"] == 0
    assert event.fields["audio_support_ticks"] == 4


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
        high_prediction_bridge=None,
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


def test_prediction_gauge_bridge_preserves_model_native_first_transition_and_releases():
    torch.manual_seed(37)
    video = torch.randn(1, 24, 10, 6, 6)
    audio = torch.randn(1, 32, 2, 12)
    packed, shapes = pack_streams((video, audio))
    prefix_t = 3
    exact_prefix = video[:, :, :prefix_t].clone()
    exact_prefix[:, :, -1] += torch.linspace(-0.4, 0.4, 6).view(1, 1, 1, 6)

    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        high_prediction_bridge=None,
        metrics=H3FlowMetrics(),
    )
    with high_boundary_contract(
        binding,
        exact_prefix,
        shapes,
        measure=True,
        prediction_gauge_bridge_weights=HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS,
    ):
        bridge = binding.high_prediction_bridge
        assert bridge is not None
        result = bridge.apply(packed, call_index=0, sigma=0.8, actual=True)

    result_video, result_audio = unpack_streams(result, shapes)
    delta = exact_prefix[:, :, -1].float() - video[:, :, prefix_t - 1].float()
    native_first = video[:, :, prefix_t].float() - video[:, :, prefix_t - 1].float()
    rebased_first = result_video[:, :, prefix_t].float() - exact_prefix[:, :, -1].float()
    torch.testing.assert_close(rebased_first, native_first, rtol=0, atol=2e-6)
    assert torch.equal(result_video[:, :, :prefix_t], video[:, :, :prefix_t])
    assert torch.equal(
        result_video[:, :, prefix_t + len(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS) :],
        video[:, :, prefix_t + len(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS) :],
    )
    assert torch.equal(result_audio, audio)

    for offset, weight in enumerate(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS):
        expected = (video[:, :, prefix_t + offset].float() + delta * float(weight)).to(result_video.dtype)
        torch.testing.assert_close(result_video[:, :, prefix_t + offset], expected, rtol=0, atol=2e-6)

    receipt = next(
        event for event in binding.metrics.events if event.kind == "partitioned_high_prediction_gauge_bridge"
    )
    assert receipt.fields["policy"] == HIGH_PREDICTION_GAUGE_BRIDGE_POLICY
    assert receipt.fields["support_tokens"] == len(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS)
    assert receipt.fields["predicted_prefix_exact"] is True
    assert receipt.fields["audio_exact"] is True
    assert receipt.fields["suffix_outside_support_exact"] is True
    assert receipt.fields["first_transition_error_max_abs"] <= 2e-5
    complete = next(
        event for event in binding.metrics.events if event.kind == "partitioned_high_prediction_gauge_bridge_complete"
    )
    assert complete.fields["calls"] == 1


def test_prediction_gauge_bridge_recomputes_dynamic_model_gauge_each_call():
    torch.manual_seed(41)
    video = torch.randn(1, 24, 9, 5, 5)
    audio = torch.randn(1, 32, 2, 10)
    packed, shapes = pack_streams((video, audio))
    prefix_t = 3
    exact_prefix = video[:, :, :prefix_t].clone()
    exact_prefix[:, :, -1] += 0.5

    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        high_prediction_bridge=None,
        metrics=H3FlowMetrics(),
    )
    with high_boundary_contract(
        binding,
        exact_prefix,
        shapes,
        measure=False,
        prediction_gauge_bridge_weights=HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS,
    ):
        bridge = binding.high_prediction_bridge
        first = bridge.apply(packed, call_index=0, sigma=0.8, actual=True)
        evolved_video = video.clone()
        evolved_video[:, :, prefix_t - 1] -= 0.25
        evolved_video[:, :, prefix_t:] += 0.125
        evolved_packed, _ = pack_streams((evolved_video, audio))
        second = bridge.apply(evolved_packed, call_index=1, sigma=0.7, actual=False)

    first_video, _ = unpack_streams(first, shapes)
    second_video, _ = unpack_streams(second, shapes)
    assert not torch.equal(first_video[:, :, prefix_t], second_video[:, :, prefix_t])
    second_native = evolved_video[:, :, prefix_t].float() - evolved_video[:, :, prefix_t - 1].float()
    second_rebased = second_video[:, :, prefix_t].float() - exact_prefix[:, :, -1].float()
    torch.testing.assert_close(second_rebased, second_native, rtol=0, atol=2e-6)
    receipts = [event for event in binding.metrics.events if event.kind == "partitioned_high_prediction_gauge_bridge"]
    assert [event.fields["call_index"] for event in receipts] == [0, 1]
    assert [event.fields["actual"] for event in receipts] == [True, False]
