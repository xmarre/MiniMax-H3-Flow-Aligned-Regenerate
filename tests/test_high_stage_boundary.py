from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.high_stage_boundary import (
    HIGH_BOUNDARY_REFERENCE_POLICY,
    HIGH_BOUNDARY_REFERENCE_WEIGHTS,
    HIGH_BOUNDARY_VIDEO_POLICY,
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
        assert trace.previous_prediction is not None
        assert trace.after_flow_prediction is not None
        trace.observe(packed, point="before_flow", call_index=16, sigma=0.1, actual=True)
        assert len(binding.metrics.events) == 2
        assert binding.metrics.events[-1].fields["flow_suffix_delta_rms"] == 0.0
        assert binding.metrics.events[-1].fields["suffix_tokens"] == 4
        assert binding.metrics.events[-1].fields["actual"] is False
        assert binding.metrics.events[-1].fields["predicted_prefix_trajectories"] is not None
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


def test_boundary_reconciliation_preserves_same_call_video_edge_and_bounded_audio_reference():
    torch.manual_seed(7)
    video = torch.randn(1, 24, 9, 6, 6)
    audio = torch.randn(1, 32, 2, 14)
    packed, shapes = pack_streams((video, audio))
    original = packed.clone()
    prefix_t = 3
    audio_prefix = 6

    exact_prefix = video[:, :, :prefix_t].clone()
    exact_prefix[:, :, -1] += torch.randn_like(exact_prefix[:, :, -1]) * 0.25
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
        metrics=H3FlowMetrics(),
    )
    with high_boundary_contract(
        binding,
        exact_prefix,
        shapes,
        measure=False,
        video_exact_prefix_bridge=True,
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

    replacement_delta = exact_prefix[:, :, -1].float() - original_video[:, :, prefix_t - 1].float()
    for offset, weight in enumerate(HIGH_BOUNDARY_REFERENCE_WEIGHTS):
        expected_video = (
            original_video[:, :, prefix_t + offset].float() + weight * replacement_delta
        ).to(result_video.dtype)
        torch.testing.assert_close(result_video[:, :, prefix_t + offset], expected_video, rtol=0, atol=0)

        expected_audio = (
            original_audio[..., audio_prefix + offset].float()
            + weight
            * (audio_reference[..., audio_prefix + offset].float() - original_audio[..., audio_prefix + offset].float())
        ).to(result_audio.dtype)
        torch.testing.assert_close(result_audio[..., audio_prefix + offset], expected_audio, rtol=0, atol=0)

    native_edge = original_video[:, :, prefix_t].float() - original_video[:, :, prefix_t - 1].float()
    restored_edge = result_video[:, :, prefix_t].float() - exact_prefix[:, :, -1].float()
    torch.testing.assert_close(restored_edge, native_edge, rtol=0, atol=2e-7)

    anchor_events = [
        event for event in binding.metrics.events if event.kind == "partitioned_high_boundary_reference_anchor"
    ]
    assert len(anchor_events) == 1
    fields = anchor_events[0].fields
    assert fields["policy"] == HIGH_BOUNDARY_REFERENCE_POLICY
    assert fields["video_policy"] == HIGH_BOUNDARY_VIDEO_POLICY
    assert fields["application_point"] == "post_flow_pre_inpaint_restore"
    assert fields["video_support_tokens"] == 4
    assert fields["audio_support_ticks"] == 4
    assert fields["video_temporal_weights"] == list(HIGH_BOUNDARY_REFERENCE_WEIGHTS)
    assert fields["audio_temporal_weights"] == list(HIGH_BOUNDARY_REFERENCE_WEIGHTS)
    assert fields["video_post_restore_edge_error_rms"] == pytest.approx(0.0, abs=2e-7)
    assert fields["audio_first_reference_error_rms"] == 0.0
    assert fields["authoritative_prefix_modified"] is False
    assert fields["suffix_outside_support_modified"] is False
    assert binding.high_boundary_anchor is None


def test_video_reconciliation_does_not_freeze_high_stage_suffix_evolution():
    torch.manual_seed(13)
    base_video = torch.randn(1, 24, 9, 4, 4)
    audio = torch.randn(1, 32, 2, 12)
    first_video = base_video.clone()
    second_video = base_video.clone()
    second_video[:, :, 3:] += torch.randn_like(second_video[:, :, 3:]) * 0.3
    exact_prefix = base_video[:, :, :3].clone()
    exact_prefix[:, :, -1] += 0.4
    first, shapes = pack_streams((first_video, audio))
    second, _ = pack_streams((second_video, audio))

    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        metrics=H3FlowMetrics(),
    )
    with high_boundary_contract(
        binding,
        exact_prefix,
        shapes,
        measure=False,
        video_exact_prefix_bridge=True,
    ):
        anchor = binding.high_boundary_anchor
        first_result = anchor.apply(first, call_index=0, sigma=0.8, actual=True)
        second_result = anchor.apply(second, call_index=1, sigma=0.6, actual=False)

    first_video_out, _ = unpack_streams(first_result, shapes)
    second_video_out, _ = unpack_streams(second_result, shapes)
    # The bridge carries the same prefix-replacement residual when the predicted
    # prefix is unchanged; it must not replace the generated suffix with a fixed
    # stored reference trajectory.
    torch.testing.assert_close(
        second_video_out[:, :, 3:7].float() - first_video_out[:, :, 3:7].float(),
        second_video[:, :, 3:7].float() - first_video[:, :, 3:7].float(),
        rtol=0,
        atol=2e-7,
    )
    assert not torch.equal(second_video_out[:, :, 3:7], first_video_out[:, :, 3:7])


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


def test_runtime_reconciles_exact_prefix_after_flow_observation():
    torch.manual_seed(11)
    video = torch.randn(1, 24, 9, 16, 16)
    audio = torch.randn(1, 32, 2, 14)
    packed, shapes = pack_streams((video, audio))
    prefix_t = 3
    exact_prefix = video[:, :, :prefix_t].clone()
    exact_prefix[:, :, -1] += 0.5

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
        exact_prefix,
        shapes,
        measure=True,
        video_exact_prefix_bridge=True,
    ):
        result = flow_predict_wrapper(
            Executor(),
            packed,
            torch.tensor([0.8]),
            model_options=model_options,
            seed=17,
        )

    result_video, _ = unpack_streams(result, shapes)
    native_edge = video[:, :, prefix_t].float() - video[:, :, prefix_t - 1].float()
    restored_edge = result_video[:, :, prefix_t].float() - exact_prefix[:, :, -1].float()
    torch.testing.assert_close(restored_edge, native_edge, rtol=0, atol=2e-7)

    points = [
        event.fields["point"]
        for event in binding.metrics.events
        if event.kind == "partitioned_high_boundary_prediction"
    ]
    assert points == ["before_flow", "after_flow", "after_bridge"]
    ordered = [
        (event.kind, event.fields.get("point"))
        for event in binding.metrics.events
        if event.kind in {"partitioned_high_boundary_prediction", "partitioned_high_boundary_reference_anchor"}
    ]
    assert ordered == [
        ("partitioned_high_boundary_prediction", "before_flow"),
        ("partitioned_high_boundary_prediction", "after_flow"),
        ("partitioned_high_boundary_reference_anchor", None),
        ("partitioned_high_boundary_prediction", "after_bridge"),
    ]
    bridge_event = next(
        event for event in binding.metrics.events if event.kind == "partitioned_high_boundary_reference_anchor"
    )
    assert bridge_event.fields["video_policy"] == HIGH_BOUNDARY_VIDEO_POLICY
    assert bridge_event.fields["video_post_restore_edge_error_rms"] == pytest.approx(0.0, abs=2e-7)
