import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import (
    ProgressiveHandoffConfig,
    build_handoff_state,
    deterministic_video_noise,
    select_handoff_index,
)
from h3_flow_regenerate.sigma import flow_shift


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


def test_auto_handoff_is_deterministic_bounded_and_uses_live_area():
    config = ProgressiveHandoffConfig(target_scale=2.0, handoff_selection="auto_compute")
    target = config.resolve_target(40, 54)
    first = config.resolve_coordinate(40, 54, *target)
    second = config.resolve_coordinate(40, 54, *target)
    assert first == second == pytest.approx(0.2)


def test_default_scale_maps_motivating_grid_without_odd_padding():
    config = ProgressiveHandoffConfig(target_scale=1.2)
    assert config.resolve_target(40, 54) == (48, 64)
