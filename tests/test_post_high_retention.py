from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from h3_flow_regenerate.post_high_retention import (
    POST_HIGH_AUDIO_RETENTION_TICKS,
    POST_HIGH_VIDEO_RETENTION_WEIGHTS,
    apply_post_high_boundary_retention,
    exact_audio_prefix_ticks,
    linear_return_weights,
)


def test_post_high_retention_is_bounded_and_returns_to_native_high():
    torch.manual_seed(692)
    video = torch.randn(1, 24, 12, 8, 10, dtype=torch.float32)
    audio = torch.randn(1, 32, 2, 48, dtype=torch.float32)
    video_prefix = 3
    audio_prefix = 8
    video_support = len(POST_HIGH_VIDEO_RETENTION_WEIGHTS)
    audio_support = POST_HIGH_AUDIO_RETENTION_TICKS
    video_reference = torch.randn(1, 24, video_support, 8, 10, dtype=torch.float32)
    audio_reference = torch.randn(1, 32, 2, audio_support, dtype=torch.float32)
    before_video = video.clone()
    before_audio = audio.clone()

    corrected_video, corrected_audio, report = apply_post_high_boundary_retention(
        video,
        audio,
        video_prefix_t=video_prefix,
        video_reference_suffix=video_reference,
        audio_prefix_t=audio_prefix,
        audio_reference_suffix=audio_reference,
    )

    assert torch.equal(video, before_video)
    assert torch.equal(audio, before_audio)
    assert torch.equal(corrected_video[:, :, :video_prefix], before_video[:, :, :video_prefix])
    assert torch.equal(
        corrected_video[:, :, video_prefix + video_support :],
        before_video[:, :, video_prefix + video_support :],
    )
    assert torch.equal(corrected_audio[..., :audio_prefix], before_audio[..., :audio_prefix])
    assert torch.equal(
        corrected_audio[..., audio_prefix + audio_support :],
        before_audio[..., audio_prefix + audio_support :],
    )

    torch.testing.assert_close(
        corrected_video[:, :, video_prefix],
        video_reference[:, :, 0],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        corrected_audio[..., audio_prefix],
        audio_reference[..., 0],
        rtol=0.0,
        atol=0.0,
    )

    expected_video_last = torch.lerp(
        before_video[:, :, video_prefix + video_support - 1],
        video_reference[:, :, -1],
        POST_HIGH_VIDEO_RETENTION_WEIGHTS[-1],
    )
    torch.testing.assert_close(
        corrected_video[:, :, video_prefix + video_support - 1],
        expected_video_last,
        rtol=0.0,
        atol=1.0e-6,
    )
    expected_audio_weights = linear_return_weights(audio_support)
    expected_audio_last = torch.lerp(
        before_audio[..., audio_prefix + audio_support - 1],
        audio_reference[..., -1],
        expected_audio_weights[-1],
    )
    torch.testing.assert_close(
        corrected_audio[..., audio_prefix + audio_support - 1],
        expected_audio_last,
        rtol=0.0,
        atol=1.0e-6,
    )

    assert report["policy"] == "partitioned_post_high_boundary_retention_v1"
    assert report["applied"] is True
    assert report["video"]["weights"] == list(POST_HIGH_VIDEO_RETENTION_WEIGHTS)
    assert report["audio"]["weights"] == pytest.approx(list(expected_audio_weights))
    assert report["video"]["support"] == 4
    assert report["audio"]["support"] == 30
    assert report["authoritative_video_prefix_modified"] is False
    assert report["authoritative_audio_prefix_modified"] is False
    assert report["video_outside_support_modified"] is False
    assert report["audio_outside_support_modified"] is False
    assert report["extra_h3_nfe"] == 0
    assert report["extra_sampler_lifetimes"] == 0
    assert report["extra_history_boundaries"] == 0
    assert report["extra_provider_calls"] == 0
    assert report["extra_vae_calls"] == 0


def test_exact_audio_prefix_ticks_requires_contiguous_binary_prefix():
    mask = torch.ones(1, 32, 2, 48)
    mask[..., :8] = 0
    assert exact_audio_prefix_ticks(mask) == 8

    broken = mask.clone()
    broken[..., 12] = 0
    with pytest.raises(ValueError, match="contiguous exact prefix"):
        exact_audio_prefix_ticks(broken)


def test_linear_return_weights_has_matching_exit_step():
    weights = linear_return_weights(30)
    assert weights[0] == 1.0
    assert weights[-1] == pytest.approx(1.0 / 30.0)
    steps = [left - right for left, right in pairwise(weights)]
    steps.append(weights[-1])
    assert max(steps) == pytest.approx(1.0 / 30.0)
    assert min(steps) == pytest.approx(1.0 / 30.0)
