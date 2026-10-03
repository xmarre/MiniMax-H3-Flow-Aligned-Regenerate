from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.audio_guided_overlap import apply_audio_guided_overlap_mask
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.video_guided_overlap import (
    apply_video_guided_overlap_mask,
    validate_video_guided_overlap_tokens,
)


def _packed_case(*, video_t=17, video_prefix=12, audio_t=20, audio_prefix=8, h=8, w=12):
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


def test_video_guided_overlap_changes_only_tail_of_exact_video_prefix():
    _packed, shapes, mask = _packed_case(video_prefix=12)
    original = mask.clone()

    runtime_mask, report = apply_video_guided_overlap_mask(mask, shapes, tokens=4)

    assert runtime_mask is not mask
    assert torch.equal(mask, original)
    old_video, old_audio = unpack_streams(original, shapes)
    new_video, new_audio = unpack_streams(runtime_mask, shapes)
    assert torch.equal(new_audio, old_audio)
    assert torch.equal(new_video[:, :, :8], old_video[:, :, :8])
    expected = _core_grid_ramp(new_video.dtype)
    torch.testing.assert_close(new_video[0, 0, 8:12, 0, 0], expected)
    torch.testing.assert_close(new_video[0, -1, 8:12, -1, -1], expected)
    assert torch.equal(new_video[:, :, 12:], old_video[:, :, 12:])
    assert report["applied"] is True
    assert report["video_prefix_tokens"] == 12
    assert report["hard_prefix_tokens"] == 8
    assert report["ramp_start_token"] == 8
    assert report["ramp_stop_token"] == 12
    assert report["mask_grid"] == "ceil_1_over_256"
    assert report["original_exact_output_restore"] is True
    assert report["audio_mask_modified"] is False


def test_video_guided_overlap_composes_with_audio_overlap_without_cross_stream_mutation():
    _packed, shapes, mask = _packed_case(video_prefix=12, audio_prefix=8)

    video_masked, video_report = apply_video_guided_overlap_mask(mask, shapes, tokens=4)
    combined, audio_report = apply_audio_guided_overlap_mask(video_masked, shapes, ticks=4)

    assert video_report["applied"] is True
    assert audio_report["applied"] is True
    exact_video, exact_audio = unpack_streams(mask, shapes)
    combined_video, combined_audio = unpack_streams(combined, shapes)
    expected = _core_grid_ramp(combined_video.dtype)
    torch.testing.assert_close(combined_video[0, 0, 8:12, 0, 0], expected)
    torch.testing.assert_close(combined_audio[0, 0, 0, 4:8], expected)
    assert torch.equal(combined_video[:, :, :8], exact_video[:, :, :8])
    assert torch.equal(combined_video[:, :, 12:], exact_video[:, :, 12:])
    assert torch.equal(combined_audio[..., :4], exact_audio[..., :4])
    assert torch.equal(combined_audio[..., 8:], exact_audio[..., 8:])


def test_video_guided_overlap_all_generated_video_is_expected_noop():
    _packed, shapes, mask = _packed_case(video_prefix=0)

    runtime_mask, report = apply_video_guided_overlap_mask(mask, shapes, tokens=4)

    assert runtime_mask is mask
    assert report["applied"] is False
    assert report["reason"] == "no_exact_video_prefix"


def test_video_guided_overlap_can_release_the_whole_exact_prefix():
    _packed, shapes, mask = _packed_case(video_prefix=4)
    original = mask.clone()

    runtime_mask, report = apply_video_guided_overlap_mask(mask, shapes, tokens=4)

    assert runtime_mask is not mask
    assert torch.equal(mask, original)
    video_mask, _ = unpack_streams(runtime_mask, shapes)
    torch.testing.assert_close(video_mask[0, 0, :4, 0, 0], _core_grid_ramp(video_mask.dtype))
    assert report["applied"] is True
    assert report["applied_tokens"] == 4
    assert report["hard_prefix_tokens"] == 0
    assert report["video_prefix_tokens"] == 4


def test_video_guided_overlap_rejects_noncanonical_partial_video_mask():
    _packed, shapes, mask = _packed_case(video_prefix=12)
    video_mask, _audio_mask = unpack_streams(mask, shapes)
    video_mask[:, :, 14] = 0

    with pytest.raises(ValueError, match="contiguous exact video prefix"):
        apply_video_guided_overlap_mask(mask, shapes, tokens=4)


def test_video_guided_overlap_validation_accepts_nonnegative_integers_and_rejects_bool():
    assert validate_video_guided_overlap_tokens(0) == 0
    assert validate_video_guided_overlap_tokens(4) == 4
    assert validate_video_guided_overlap_tokens(8) == 8
    assert validate_video_guided_overlap_tokens(10**30) == 10**30
    with pytest.raises(ValueError):
        validate_video_guided_overlap_tokens(True)
    with pytest.raises(ValueError):
        validate_video_guided_overlap_tokens(-1)
