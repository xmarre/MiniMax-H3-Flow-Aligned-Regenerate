import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import (
    ProgressiveHandoffConfig,
    ProgressiveTargetInputConfig,
    build_handoff_state,
    deterministic_video_noise,
    select_handoff_index,
)
from h3_flow_regenerate.sigma import flow_shift


class FakeLearnedProvider:
    api_version = 1
    kind = "minimax_h3_learned_latent_upscaler"
    model_name = "fake.safetensors"
    device = "cpu"
    precision = "fp32"
    offload_after_upscale = False

    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def upscale_clean_video(self, video, *, target_h, target_w):
        self.calls.append((video.clone(), target_h, target_w))
        if self.output is not None:
            return self.output
        return torch.full(
            (*video.shape[:-2], target_h, target_w),
            5.0,
            dtype=video.dtype,
            device=video.device,
        )


def packed(h=4, w=4):
    video = torch.randn(1, 24, 2, h, w)
    audio = torch.randn(1, 32, 2, 7)
    state, shapes = pack_streams((video, audio))
    x0, _ = pack_streams((torch.randn_like(video), torch.randn_like(audio)))
    return state, x0, shapes, video, audio


def test_same_resolution_handoff_is_identity():
    state, x0, shapes, _, _ = packed()
    target, target_shapes = build_handoff_state(
        source_packed_state=state,
        source_x0_packed=x0,
        source_shapes=shapes,
        sigma=0.4,
        target_h=4,
        target_w=4,
        seed=12,
    )
    assert torch.equal(target, state)
    assert target_shapes == shapes


def test_handoff_geometry_audio_and_determinism():
    state, x0, shapes, _, audio = packed()
    args = dict(
        source_packed_state=state,
        source_x0_packed=x0,
        source_shapes=shapes,
        sigma=0.4,
        target_h=8,
        target_w=6,
        seed=123,
    )
    first, target_shapes = build_handoff_state(**args)
    second, _ = build_handoff_state(**args)
    target_video, target_audio = unpack_streams(first, target_shapes)
    assert target_video.shape == (1, 24, 2, 8, 6)
    assert torch.equal(target_audio, audio)
    assert torch.equal(first, second)


def test_bicubic_default_never_invokes_connected_learned_provider():
    state, x0, shapes, _, _ = packed()
    provider = FakeLearnedProvider()
    build_handoff_state(
        source_packed_state=state,
        source_x0_packed=x0,
        source_shapes=shapes,
        sigma=0.4,
        target_h=8,
        target_w=6,
        seed=123,
        transfer_mode="bicubic",
        learned_upscaler=provider,
    )
    assert provider.calls == []


def test_learned_handoff_uses_exact_probe_video_once_and_preserves_audio():
    source_video = torch.full((1, 24, 2, 4, 4), -3.0)
    exact_x0_video = torch.full_like(source_video, 2.0)
    audio = torch.randn(1, 32, 2, 7)
    state, shapes = pack_streams((source_video, audio))
    x0, _ = pack_streams((exact_x0_video, torch.full_like(audio, 99.0)))
    provider = FakeLearnedProvider()
    report = {}

    target, target_shapes = build_handoff_state(
        source_packed_state=state,
        source_x0_packed=x0,
        source_shapes=shapes,
        sigma=0.4,
        target_h=8,
        target_w=6,
        seed=123,
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        transfer_metrics=report,
    )

    target_video, target_audio = unpack_streams(target, target_shapes)
    expected_noise = deterministic_video_noise(
        target_video.shape,
        seed=123,
        device=target_video.device,
        dtype=target_video.dtype,
    )
    assert len(provider.calls) == 1
    assert torch.equal(provider.calls[0][0], exact_x0_video)
    assert provider.calls[0][1:] == (8, 6)
    assert torch.allclose(target_video, 0.6 * torch.full_like(target_video, 5.0) + 0.4 * expected_noise)
    assert torch.equal(target_audio, audio)
    assert report["transfer_mode"] == "learned_3d"
    assert report["provider_api_version"] == 1
    assert report["source_hw"] == (4, 4)
    assert report["target_hw"] == (8, 6)
    assert report["temporal_length"] == 2


@pytest.mark.parametrize(
    "output",
    [
        torch.zeros(1, 23, 2, 8, 6),
        torch.zeros(1, 24, 3, 8, 6),
        torch.zeros(1, 24, 2, 7, 6),
    ],
)
def test_learned_handoff_rejects_wrong_provider_geometry(output):
    state, x0, shapes, _, _ = packed()
    with pytest.raises(RuntimeError, match="returned shape"):
        build_handoff_state(
            source_packed_state=state,
            source_x0_packed=x0,
            source_shapes=shapes,
            sigma=0.4,
            target_h=8,
            target_w=6,
            seed=123,
            transfer_mode="learned_3d",
            learned_upscaler=FakeLearnedProvider(output),
        )


def test_suffix_dc_bridge_config_is_shared_across_continuum_modes_and_boolean():
    provider = FakeLearnedProvider()
    for exact_prefix_mode in ("fallback", "target_sparse_lifter"):
        config = ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            exact_prefix_mode=exact_prefix_mode,
        )
        assert config.suffix_dc_bridge is True
    mixed = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=4,
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        exact_prefix_mode="mixed_grid_low_suffix",
    )
    assert mixed.suffix_dc_bridge is True
    with pytest.raises(TypeError, match="must be boolean"):
        ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            suffix_dc_bridge=1,
        )


def test_learned_target_input_config_requires_supported_provider_contract():
    with pytest.raises(ValueError, match="requires a connected"):
        ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            transfer_mode="learned_3d",
        )

    incompatible = FakeLearnedProvider()
    incompatible.api_version = 2
    with pytest.raises(ValueError, match="provider API"):
        ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            transfer_mode="learned_3d",
            learned_upscaler=incompatible,
        )


def test_cpu_rng_contract_is_retry_stable_and_seed_sensitive():
    shape = (1, 24, 1, 4, 4)
    a = deterministic_video_noise(shape, seed=1, device=torch.device("cpu"), dtype=torch.float32)
    b = deterministic_video_noise(shape, seed=1, device=torch.device("cpu"), dtype=torch.float32)
    c = deterministic_video_noise(shape, seed=2, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_handoff_selection_uses_unshifted_coordinate_and_keeps_high_steps():
    sigmas = flow_shift(torch.linspace(1, 0, 9), 12.0)
    index = select_handoff_index(sigmas, 0.35, min_high_steps=2)
    assert 1 <= index < len(sigmas) - 2


def test_invalid_schedule_fails_closed():
    with pytest.raises(ValueError, match="descending"):
        select_handoff_index(torch.tensor([1.0, 0.8, 0.8, 0.0]), 0.3)


def test_nonfinite_schedule_fails_closed():
    with pytest.raises(ValueError, match="finite"):
        select_handoff_index(torch.tensor([1.0, float("nan"), 0.3, 0.0]), 0.3)


def test_auto_handoff_is_deterministic_bounded_and_uses_live_area():
    config = ProgressiveHandoffConfig(target_scale=2.0, handoff_selection="auto_compute")
    target = config.resolve_target(40, 54)
    first = config.resolve_coordinate(40, 54, *target)
    second = config.resolve_coordinate(40, 54, *target)
    assert first == second == pytest.approx(0.2)


def test_progressive_explicit_target_rejects_mixed_shrink_expand_geometry():
    config = ProgressiveHandoffConfig(target_latent_h=32, target_latent_w=80)
    with pytest.raises(ValueError, match="must not shrink either"):
        config.resolve_target(40, 54)


def test_target_input_source_rejects_mixed_expand_shrink_geometry():
    config = ProgressiveTargetInputConfig(source_latent_h=56, source_latent_w=20)
    with pytest.raises(ValueError, match="must not exceed either"):
        config.resolve_source(48, 64)


def test_default_scale_maps_motivating_grid_without_odd_padding():
    config = ProgressiveHandoffConfig(target_scale=1.2)
    assert config.resolve_target(40, 54) == (48, 64)
