from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch

from test_handoff import FakeLearnedProvider
from test_mixed_grid import inputs

from h3_flow_regenerate.geometry import pack_streams, resize_spatial_5d, unpack_streams
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.runtime import FlowBinding, _run_progressive
from h3_flow_regenerate.state_transport import HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1


def test_mixed_grid_velocity_transport_preserves_state_noise_audio_and_sampler_boundaries(monkeypatch):
    fake = ModuleType("comfy")
    fake.samplers = ModuleType("comfy.samplers")
    fake.samplers.KSAMPLER = lambda function, **kw: SimpleNamespace(sampler_function=function, extra_options={})
    monkeypatch.setitem(sys.modules, "comfy", fake)
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake.samplers)

    caller_packed, target_shapes, mask = inputs(t=7, prefix=2, h=8, w=12)
    caller_noise = torch.randn_like(caller_packed)
    source_shapes = list(target_shapes)
    source_shapes[0] = (*source_shapes[0][:-2], 4, 6)

    source_clean_video = torch.linspace(
        -0.6,
        0.9,
        steps=24 * 7 * 4 * 6,
        dtype=torch.float32,
    ).reshape(1, 24, 7, 4, 6)
    source_displacement = torch.linspace(
        -0.2,
        0.3,
        steps=24 * 7 * 4 * 6,
        dtype=torch.float32,
    ).reshape_as(source_clean_video)
    source_state_video = source_clean_video + source_displacement
    source_audio_state = torch.linspace(
        -0.4,
        0.5,
        steps=32 * 2 * 11,
        dtype=torch.float32,
    ).reshape(1, 32, 2, 11)
    source_audio_clean = torch.zeros_like(source_audio_state)
    source_state_packed, _ = pack_streams((source_state_video, source_audio_state))
    source_clean_packed, _ = pack_streams((source_clean_video, source_audio_clean))

    target_clean_video = torch.linspace(
        -1.0,
        1.2,
        steps=24 * 7 * 8 * 12,
        dtype=torch.float32,
    ).reshape(1, 24, 7, 8, 12)
    provider = FakeLearnedProvider(output=target_clean_video)
    config = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=6,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        suffix_dc_bridge=False,
        suffix_geometric_bridge=False,
        handoff_state_policy=HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
    )

    base = SimpleNamespace(process_latent_in=lambda value: value, diffusion_model=SimpleNamespace(blocks=[]))
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=base),
        conds={"positive": []},
    )
    binding = FlowBinding()
    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})
    calls: list[str] = []
    observed = {}

    import h3_flow_regenerate.runtime as runtime

    real_deterministic_video_noise = runtime.deterministic_video_noise
    deterministic_noise_calls = []

    def counted_deterministic_video_noise(*args, **kwargs):
        deterministic_noise_calls.append((args, kwargs))
        return real_deterministic_video_noise(*args, **kwargs)

    monkeypatch.setattr(runtime, "deterministic_video_noise", counted_deterministic_video_noise)

    def execute(noise, latent, call_sampler, sigmas, call_mask, *args, latent_shapes):
        stage = guider.model_options["transformer_options"]["h3_flow_stage"]
        calls.append(stage)
        if stage == "low":
            sigma = float(sigmas[-1])
            return source_state_packed.to(latent) / (1.0 - sigma)
        if stage == "probe":
            return source_clean_packed.to(latent)
        if stage == "high":
            sigma = float(sigmas[0])
            assert torch.equal(latent, caller_packed)
            assert torch.equal(call_mask, mask)
            assert latent_shapes == target_shapes

            input_video_noise, _ = unpack_streams(caller_noise, target_shapes)
            high_video_noise, _ = unpack_streams(noise, target_shapes)
            video_mask, _ = unpack_streams(mask, target_shapes)
            protected = video_mask == 0
            assert torch.equal(high_video_noise[protected], input_video_noise[protected])

            initial_state = sigma * noise + (1.0 - sigma) * latent
            initial_video, initial_audio = unpack_streams(initial_state, target_shapes)
            lifted_displacement = resize_spatial_5d(source_displacement, 8, 12, mode="bicubic")
            expected_suffix = target_clean_video[:, :, 2:] + lifted_displacement[:, :, 2:]
            torch.testing.assert_close(initial_video[:, :, 2:], expected_suffix, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(initial_audio, source_audio_state, rtol=2e-5, atol=2e-5)
            observed["initial_video"] = initial_video.clone()
            binding.metrics.event("model_call", actual=True)
            return caller_packed.clone()
        raise AssertionError(stage)

    result = _run_progressive(
        execute,
        guider,
        binding,
        config,
        caller_noise,
        caller_packed,
        sampler,
        torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
        mask,
        None,
        True,
        7,
        list(target_shapes),
    )

    assert torch.equal(result, caller_packed)
    assert calls == ["low", "probe", "high"]
    assert len(provider.calls) == 1
    # The only deterministic transfer-independent draw is the existing private
    # low-grid source noise. velocity_bicubic_v1 adds no target-grid random field.
    assert len(deterministic_noise_calls) == 1

    original_video, _ = unpack_streams(caller_packed, target_shapes)
    expected_provider_context = resize_spatial_5d(original_video[:, :, :2], 4, 6, mode="bicubic")
    assert torch.equal(provider.calls[0][0][:, :, :2], expected_provider_context)

    transport_events = [event for event in binding.metrics.events if event.kind == "mixed_grid_state_transport"]
    assert len(transport_events) == 1
    transport = transport_events[0].fields
    assert transport["handoff_state_policy"] == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1
    assert transport["state_transport_applied"] is True
    assert transport["state_transport_added_rng"] is False
    assert transport["state_transport_temporal_mixing"] is False
    assert transport["state_transport_amplitude_normalization"] is False
    assert transport["state_transport_source_roundtrip_max_abs"] <= 1e-6
    assert transport["state_transport_target_closure_max_abs"] <= 2e-6

    complete = [event for event in binding.metrics.events if event.kind == "handoff_complete"]
    assert len(complete) == 1
    assert complete[0].fields["sampler_invocation_count"] == 3
    assert complete[0].fields["history_boundary_count"] == 2
    assert complete[0].fields["exact_probe_performed"] is True
    assert complete[0].fields["handoff_state_policy"] == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1
    assert binding.metrics.counters["handoff_exact_probe_nfe"] == 1
