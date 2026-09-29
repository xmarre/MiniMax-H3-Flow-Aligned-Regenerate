from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.high_stage_guard import (
    HIGH_STAGE_VIDEO_GUARD_POLICY,
    build_high_stage_video_guard,
    disabled_high_stage_video_guard,
)


def _fixture(prefix_t=3):
    torch.manual_seed(17)
    video = torch.randn(1, 24, 8, 10, 12)
    audio = torch.randn(1, 32, 2, 31)
    packed, shapes = pack_streams((video, audio))

    video_mask = torch.ones_like(video)
    video_mask[:, :, :prefix_t] = 0
    audio_mask = torch.linspace(0.0, 1.0, audio.numel(), dtype=audio.dtype).reshape_as(audio)
    mask = pack_streams((video_mask, audio_mask))[0]

    guard_source = video.clone()
    guard_source[:, :, prefix_t] += 7.0
    return packed, shapes, mask, video, audio, video_mask, audio_mask, guard_source


def test_high_stage_guard_changes_exactly_one_generated_video_token_and_mask():
    prefix_t = 3
    packed, shapes, mask, video, audio, video_mask, audio_mask, guard_source = _fixture(prefix_t)

    guarded, guarded_mask, receipt = build_high_stage_video_guard(
        packed,
        mask,
        list(shapes),
        guard_video_caller=guard_source,
        prefix_t=prefix_t,
    )

    guarded_video, guarded_audio = unpack_streams(guarded, list(shapes))
    guarded_video_mask, guarded_audio_mask = unpack_streams(guarded_mask, list(shapes))

    assert receipt["policy"] == HIGH_STAGE_VIDEO_GUARD_POLICY
    assert receipt["applied"] is True
    assert receipt["guard_tokens"] == 1
    assert receipt["caller_exact_prefix_tokens"] == prefix_t
    assert receipt["high_protected_video_tokens"] == prefix_t + 1
    assert receipt["guard_differs_from_caller_latent"] is True

    assert torch.equal(guarded_video[:, :, :prefix_t], video[:, :, :prefix_t])
    assert torch.equal(guarded_video[:, :, prefix_t : prefix_t + 1], guard_source[:, :, prefix_t : prefix_t + 1])
    assert torch.equal(guarded_video[:, :, prefix_t + 1 :], video[:, :, prefix_t + 1 :])
    assert torch.equal(guarded_audio, audio)

    assert torch.equal(guarded_video_mask[:, :, :prefix_t], video_mask[:, :, :prefix_t])
    assert torch.count_nonzero(guarded_video_mask[:, :, prefix_t : prefix_t + 1]).item() == 0
    assert torch.equal(guarded_video_mask[:, :, prefix_t + 1 :], video_mask[:, :, prefix_t + 1 :])
    assert torch.equal(guarded_audio_mask, audio_mask)


def test_high_stage_guard_rejects_already_protected_guard_token():
    prefix_t = 3
    packed, shapes, mask, _video, _audio, _video_mask, _audio_mask, guard_source = _fixture(prefix_t)
    video_mask, audio_mask = unpack_streams(mask, list(shapes))
    video_mask = video_mask.clone()
    video_mask[:, :, prefix_t] = 0
    bad_mask = pack_streams((video_mask, audio_mask))[0]

    with pytest.raises(RuntimeError, match="was not generated"):
        build_high_stage_video_guard(
            packed,
            bad_mask,
            list(shapes),
            guard_video_caller=guard_source,
            prefix_t=prefix_t,
        )


def test_disabled_high_stage_guard_is_output_neutral_receipt():
    receipt = disabled_high_stage_video_guard(prefix_t=12, reason="safe_video_topology_not_active")

    assert receipt == {
        "policy": HIGH_STAGE_VIDEO_GUARD_POLICY,
        "enabled": False,
        "applied": False,
        "guard_tokens": 0,
        "guard_token_index": 12,
        "caller_exact_prefix_tokens": 12,
        "high_protected_video_tokens": 12,
        "reason": "safe_video_topology_not_active",
        "caller_prefix_modified": False,
        "audio_modified": False,
        "audio_mask_modified": False,
        "later_suffix_modified": False,
        "mask_outside_guard_modified": False,
        "extra_h3_nfe": 0,
        "extra_sampler_lifetimes": 0,
        "extra_history_boundaries": 0,
        "extra_provider_calls": 0,
        "extra_vae_calls": 0,
    }
