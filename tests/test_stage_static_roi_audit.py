"""Read-only stage-local ROI diagnostics: geometry, sharpness, timing, safety."""
import json

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate.stage_static_roi_audit import (
    ROOM_01784, parse_static_rois, measure_stage_static_rois,
    _phase_displacement,
)


def _video(*, blur_after=False, stage_change=False, height=160, width=200):
    generator = torch.Generator().manual_seed(81784)
    pixels = torch.rand((height, width, 3), generator=generator) * .50 + .22
    # Rich, fixed bookshelf/picture/curtain analogues.
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    pixels[:, :, 0] += ((xx % 9) < 3).float() * .12
    pixels[:, :, 1] += ((yy % 13) < 3).float() * .10
    frames = pixels[None].repeat(14, 1, 1, 1)
    if blur_after:
        v = frames[8:].permute(0, 3, 1, 2)
        frames[8:] = F.avg_pool2d(v, 3, stride=1, padding=1).permute(0, 2, 3, 1)
    if stage_change:
        # Photometric shift alone must not be mistaken for a large translation.
        frames[8:] = (frames[8:] * .95 + .018).clamp(0, 1)
    return frames


def test_profile_and_custom_rois_are_bounded_and_fractional():
    assert parse_static_rois("off") == {}
    assert "bookshelf" in parse_static_rois("01784_room")
    custom = parse_static_rois("custom", json.dumps({
        "books": [0, .40, .14, .65], "curtain": [.8, .03, .99, .36]
    }))
    assert len(custom) == 2
    for bad in ('{"bad":[0,0,2,1],"ok":[0,0,.3,.3]}', '{"only":[0,0,.3,.3]}', "not json"):
        with pytest.raises(ValueError):
            parse_static_rois("custom", bad)
    with pytest.raises(ValueError, match="unknown"):
        parse_static_rois("bad")


def test_integer_motion_correspondence_sign_and_no_mutation():
    original = torch.zeros((40, 44))
    original[11:25, 12:31] = torch.rand((14, 19), generator=torch.Generator().manual_seed(2))
    candidate = torch.roll(original, shifts=(2, -3), dims=(0, 1))
    before = original.clone()
    result = _phase_displacement(original, candidate)
    assert result["status"] == "measured"
    assert result["dx_px"] == pytest.approx(-3, abs=.3)
    assert result["dy_px"] == pytest.approx(2, abs=.3)
    assert torch.equal(original, before)


def test_stage_texture_loss_localized_in_time_without_changing_pixels():
    movie = _video(blur_after=True)
    frozen = movie.clone()
    labels = list(range(167, 181))
    rects = {"left": (.02, .12, .30, .66), "right": (.70, .05, .98, .67)}
    got = measure_stage_static_rois(movie, labels, join_frame=175, rois=rects)
    assert got["status"] == "measured"
    assert got["pre_frame"] == 174 and got["post_frame"] == 180
    assert got["intra_stage_comparisons_only"] is True
    assert got["absolute_sharpness_across_grids_comparable"] is False
    assert got["production_modified"] is False and got["extra_vae_calls"] == 0
    assert all(r["sharpness_post_over_pre"] < .7 for r in got["regions"].values())
    assert torch.equal(movie, frozen)


def test_color_shift_without_spatial_motion_does_not_fake_background_zoom():
    movie = _video(stage_change=True)
    rects = {"left": (.02, .10, .31, .71), "right": (.67, .08, .97, .66)}
    got = measure_stage_static_rois(movie, list(range(167,181)), join_frame=175, rois=rects)
    assert got["status"] == "measured"
    assert abs(got["background_scale"]["scale_change_percent"]) < .25


def test_insufficient_timing_window_returns_explanatory_receipt():
    movie = _video()
    result = measure_stage_static_rois(
        movie, list(range(100,114)), join_frame=175,
        rois={"left": (.03,.03,.22,.22), "right": (.7,.7,.97,.97)}
    )
    assert result["status"] == "insufficient_timing_window"
    assert result["requested_pair"] == [174,180]


def test_roi_bounds_follow_native_canvas_grid():
    rects = {"left": (.02,.1,.30,.4), "right": (.7,.1,.98,.4)}
    small = measure_stage_static_rois(
        _video(height=160,width=200), list(range(167,181)), join_frame=175, rois=rects
    )
    large = measure_stage_static_rois(
        _video(height=320,width=400), list(range(167,181)), join_frame=175, rois=rects
    )
    assert small["regions"]["left"]["native_roi_hw"] == [48,56]
    assert large["regions"]["left"]["native_roi_hw"] == [96,112]
    assert small["absolute_sharpness_across_grids_comparable"] is False
    assert large["absolute_sharpness_across_grids_comparable"] is False
