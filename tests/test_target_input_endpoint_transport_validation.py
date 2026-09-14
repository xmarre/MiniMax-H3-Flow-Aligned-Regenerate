from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import target_input_endpoint_transport_validation as transport
from h3_flow_regenerate.geometry import pack_streams, resize_spatial_5d, unpack_streams
from h3_flow_regenerate.guidance import GuidanceConfig
from h3_flow_regenerate.handoff import (
    ProgressiveTargetInputConfig,
    deterministic_video_noise,
)


class _Metrics:
    def __init__(self):
        self.events = []
        self.counters = {}

    def event(self, kind, **fields):
        self.events.append(SimpleNamespace(kind=kind, fields=fields))

    def increment(self, key, amount=1):
        self.counters[key] = self.counters.get(key, 0) + amount


class _Provider:
    api_version = 1
    kind = "minimax_h3_learned_latent_upscaler"
    model_name = "dummy.safetensors"
    device = "cpu"
    inference_device = "cpu"
    precision = "fp32"
    offload_after_upscale = False

    def upscale_clean_video(self, *_args, **_kwargs):
        raise AssertionError("provider is owned by the wrapped production handoff")


def _binding(mode: str = "direction"):
    return SimpleNamespace(guidance=GuidanceConfig(mode=mode), metrics=_Metrics())


def _config():
    return ProgressiveTargetInputConfig(
        source_scale=0.7,
        handoff_coordinate=0.35,
        handoff_selection="fixed",
        transfer_mode="learned_3d",
        exact_prefix_mode="fallback",
        learned_upscaler=_Provider(),
    )


def test_eligibility_is_narrow_and_unprotected_only():
    binding = _binding()
    config = _config()
    assert transport._eligible(config, binding, None)
    assert not transport._eligible(config, binding, torch.zeros(1))
    assert not transport._eligible(
        ProgressiveTargetInputConfig(
            source_scale=0.7,
            transfer_mode="bicubic",
            exact_prefix_mode="fallback",
        ),
        binding,
        None,
    )
    assert not transport._eligible(config, _binding("off"), None)


def test_endpoint_transport_preserves_flow_endpoint_at_same_sigma():
    sigma = 0.8780487775802612
    generator = torch.Generator(device="cpu").manual_seed(7)
    source_clean = torch.randn((1, 24, 3, 4, 6), generator=generator)
    source_endpoint = torch.randn(source_clean.shape, generator=generator)
    source_state = sigma * source_endpoint + (1.0 - sigma) * source_clean
    target_clean = torch.randn((1, 24, 3, 8, 10), generator=generator)

    transported, facts = transport._transport_video_state(
        source_state,
        source_clean,
        target_clean,
        sigma=sigma,
    )
    expected = sigma * resize_spatial_5d(source_endpoint, 8, 10, mode="bicubic") + (1.0 - sigma) * target_clean

    torch.testing.assert_close(transported, expected, rtol=2e-5, atol=2e-5)
    assert facts["policy"] == "same_sigma_endpoint_bicubic_v1"
    assert facts["clean_weight"] == pytest.approx(1.0 - sigma)
    assert facts["target_closure_max_abs"] <= 1e-6
    assert facts["direct_form_max_abs"] <= 1e-6
    assert facts["temporal_mixing"] is False
    assert facts["amplitude_normalization"] is False


def test_failed_velocity_reanchor_overshoot_identity_is_exact():
    sigma = 0.75
    source_clean = torch.zeros((1, 24, 3, 4, 4), dtype=torch.float32)
    source_state = torch.full_like(source_clean, 0.5)
    target_clean = torch.full((1, 24, 3, 6, 8), 2.0, dtype=torch.float32)

    transported, facts = transport._transport_video_state(
        source_state,
        source_clean,
        target_clean,
        sigma=sigma,
    )
    expected = resize_spatial_5d(source_state, 6, 8, mode="bicubic") + (1.0 - sigma) * target_clean

    torch.testing.assert_close(transported, expected, rtol=0, atol=1e-6)
    assert facts["clean_anchor_delta_rms"] == pytest.approx(2.0, abs=1e-6)
    assert facts["failed_velocity_overshoot_rms"] == pytest.approx(1.5, abs=1e-6)
    assert facts["failed_velocity_overshoot_identity_max_abs"] <= 1e-6


def test_endpoint_transport_matches_direct_anchor_delta_form():
    sigma = 0.61
    generator = torch.Generator(device="cpu").manual_seed(11)
    source_clean = torch.randn((1, 24, 4, 6, 6), generator=generator)
    source_state = torch.randn(source_clean.shape, generator=generator)
    target_clean = torch.randn((1, 24, 4, 8, 10), generator=generator)

    transported, _ = transport._transport_video_state(
        source_state,
        source_clean,
        target_clean,
        sigma=sigma,
    )
    lifted_state = resize_spatial_5d(source_state, 8, 10, mode="bicubic")
    lifted_clean = resize_spatial_5d(source_clean, 8, 10, mode="bicubic")
    expected = lifted_state + (1.0 - sigma) * (target_clean - lifted_clean)

    torch.testing.assert_close(transported, expected, rtol=2e-5, atol=2e-5)


def test_build_wrapper_discards_legacy_target_noise_and_preserves_audio(monkeypatch):
    sigma = 0.7
    source_clean = torch.zeros((1, 24, 3, 4, 4), dtype=torch.float32)
    source_state = torch.full_like(source_clean, 0.5)
    source_audio = torch.randn((1, 32, 2, 5), generator=torch.Generator().manual_seed(8))
    source_packed, source_shapes = pack_streams((source_state, source_audio))
    source_x0_packed, _ = pack_streams((source_clean, torch.zeros_like(source_audio)))
    target_clean = torch.full((1, 24, 3, 6, 8), 1.25, dtype=torch.float32)
    seed = 123

    def original(**kwargs):
        noise = deterministic_video_noise(
            tuple(target_clean.shape),
            seed=kwargs["seed"],
            device=target_clean.device,
            dtype=target_clean.dtype,
        )
        legacy_video = target_clean * (1.0 - kwargs["sigma"]) + noise * kwargs["sigma"]
        return pack_streams((legacy_video, source_audio.clone()))

    monkeypatch.setattr(transport, "_ORIGINAL_BUILD_HANDOFF_STATE", original)
    binding = _binding()
    state = transport._state()
    state.tls.record = transport._TransportRecord(enabled=True, binding=binding)
    try:
        packed, shapes = transport._build_handoff_state_transport_wrapper(
            source_packed_state=source_packed,
            source_x0_packed=source_x0_packed,
            source_shapes=source_shapes,
            sigma=sigma,
            target_h=6,
            target_w=8,
            seed=seed,
            transfer_mode="learned_3d",
            learned_upscaler=_Provider(),
            transfer_metrics={},
        )
    finally:
        state.tls.record = None

    video, audio = unpack_streams(packed, shapes)
    expected = resize_spatial_5d(source_state, 6, 8, mode="bicubic") + (1.0 - sigma) * target_clean
    torch.testing.assert_close(video, expected, rtol=2e-5, atol=2e-5)
    assert torch.equal(audio, source_audio)
    assert binding.metrics.counters["target_input_endpoint_transport_validation_runs"] == 1
    event = binding.metrics.events[-1]
    assert event.kind == "target_input_endpoint_transport_validation"
    assert event.fields["final_target_rng_contribution"] is False
    assert event.fields["extra_h3_evaluations"] == 0
    assert event.fields["extra_upscaler_calls"] == 0


def test_endpoint_transport_does_not_mutate_inputs():
    source_clean = torch.zeros((1, 24, 2, 4, 4), dtype=torch.float32)
    source_state = source_clean + 0.25
    target_clean = torch.ones((1, 24, 2, 6, 6), dtype=torch.float32)
    snapshots = tuple(tensor.clone() for tensor in (source_state, source_clean, target_clean))

    transport._transport_video_state(
        source_state,
        source_clean,
        target_clean,
        sigma=0.6,
    )

    for tensor, snapshot in zip((source_state, source_clean, target_clean), snapshots, strict=True):
        assert torch.equal(tensor, snapshot)


def test_endpoint_transport_rejects_invalid_sigma_and_geometry():
    source_clean = torch.zeros((1, 24, 3, 4, 4), dtype=torch.float32)
    source_state = source_clean.clone()
    target_clean = torch.zeros((1, 24, 3, 6, 6), dtype=torch.float32)

    for sigma in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="sigma"):
            transport._transport_video_state(
                source_state,
                source_clean,
                target_clean,
                sigma=sigma,
            )

    shrinking = torch.zeros((1, 24, 3, 2, 6), dtype=torch.float32)
    with pytest.raises(ValueError, match="must not shrink"):
        transport._transport_video_state(
            source_state,
            source_clean,
            shrinking,
            sigma=0.5,
        )


def test_run_wrapper_fails_closed_if_expected_transport_did_not_execute(monkeypatch):
    monkeypatch.setattr(transport, "_ORIGINAL_RUN_PROGRESSIVE", lambda *_args, **_kwargs: "result")
    with pytest.raises(RuntimeError, match="expected exactly one learned handoff"):
        transport._run_progressive_transport_wrapper(
            None,
            SimpleNamespace(),
            _binding(),
            _config(),
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            0,
            [],
        )
