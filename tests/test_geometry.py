import math

import pytest
import torch

from h3_flow_regenerate.geometry import (
    geometry_from_video,
    h3_refine_scale_target_canvas,
    normalize_source_geometry_preserving_aspect,
    normalize_target_geometry,
    pack_streams,
    pixel_to_safe_latent,
    resize_video,
    unpack_streams,
    validate_av,
)


def video(h=40, w=54, t=3):
    return torch.randn(1, 24, t, h, w)


def audio(t=20):
    return torch.randn(1, 32, 2, t)


def test_h3_geometry_and_pixel_scale():
    geometry = geometry_from_video(video(40, 54))
    assert (geometry.pixel_h, geometry.pixel_w) == (640, 864)
    assert geometry.patch_safe
    assert geometry.video_rows == 3 * 20 * 27


def test_known_76_by_57_regression_normalizes_even():
    assert normalize_target_geometry(source_h=64, source_w=48, scale=1.2) == (78, 58)
    assert normalize_target_geometry(source_h=76, source_w=57, target_h=76, target_w=57) == (76, 58)


def test_partitioned_source_geometry_preserves_accepted_56x76_case():
    assert normalize_source_geometry_preserving_aspect(
        target_h=56,
        target_w=76,
        scale=0.70,
    ) == (40, 54)


def test_partitioned_source_geometry_reduces_56x74_anisotropic_rounding_without_more_area():
    selected = normalize_source_geometry_preserving_aspect(
        target_h=56,
        target_w=74,
        scale=0.70,
    )
    assert selected == (38, 50)

    nearest = normalize_target_geometry(
        source_h=56,
        source_w=74,
        scale=0.70,
        policy="nearest",
    )
    assert nearest == (40, 52)
    assert selected[0] * selected[1] < nearest[0] * nearest[1]

    target_aspect = 74 / 56
    nearest_error = abs(math.log((nearest[1] / nearest[0]) / target_aspect))
    selected_error = abs(math.log((selected[1] / selected[0]) / target_aspect))
    assert selected_error < nearest_error / 3


def test_partitioned_source_geometry_is_orientation_symmetric():
    assert normalize_source_geometry_preserving_aspect(
        target_h=74,
        target_w=56,
        scale=0.70,
    ) == (50, 38)


@pytest.mark.parametrize(
    ("target_h", "target_w", "scale"),
    [
        (56, 74, 0.70),
        (56, 76, 0.70),
        (48, 64, 0.70),
        (64, 48, 0.70),
        (54, 78, 0.83),
        (78, 54, 0.83),
        (40, 54, 0.60),
        (54, 40, 0.60),
    ],
)
def test_partitioned_source_geometry_is_patch_safe_and_never_exceeds_nearest_budget(
    target_h,
    target_w,
    scale,
):
    selected_h, selected_w = normalize_source_geometry_preserving_aspect(
        target_h=target_h,
        target_w=target_w,
        scale=scale,
    )
    nearest_h, nearest_w = normalize_target_geometry(
        source_h=target_h,
        source_w=target_w,
        scale=scale,
        policy="nearest",
    )
    assert selected_h % 2 == 0
    assert selected_w % 2 == 0
    assert selected_h <= target_h
    assert selected_w <= target_w
    assert (selected_h, selected_w) != (target_h, target_w)
    assert selected_h * selected_w <= nearest_h * nearest_w


def test_partitioned_source_geometry_rejects_scale_that_rounds_to_target():
    with pytest.raises(ValueError, match="unchanged target geometry"):
        normalize_source_geometry_preserving_aspect(
            target_h=4,
            target_w=4,
            scale=0.99,
        )


def test_pixel_target_maps_to_safe_latent():
    assert pixel_to_safe_latent(768, 1024) == (48, 64)


def test_refine_scale_target_matches_current_continuum_geometry():
    assert h3_refine_scale_target_canvas(736, 768, 1.20, align=32, keep_proportion=True) == (
        896,
        928,
    )


@pytest.mark.parametrize(
    ("base_width", "base_height", "expected"),
    [
        (768, 768, (928, 928)),
        (864, 640, (1024, 768)),
        (640, 864, (768, 1024)),
    ],
)
def test_refine_scale_target_matches_lbh_alignment_semantics(base_width, base_height, expected):
    assert (
        h3_refine_scale_target_canvas(
            base_width,
            base_height,
            1.20,
            align=32,
            keep_proportion=True,
        )
        == expected
    )


def test_refine_scale_target_rejects_invalid_inputs():
    with pytest.raises(ValueError, match=r">= 1\.0"):
        h3_refine_scale_target_canvas(736, 768, 0.99)
    with pytest.raises(ValueError, match="positive"):
        h3_refine_scale_target_canvas(0, 768, 1.20)


@pytest.mark.parametrize("shape", [(40, 54), (18, 102), (128, 26)])
def test_arbitrary_patch_safe_aspect_ratios(shape):
    validate_av(video(*shape), audio())


def test_odd_geometry_is_rejected_before_h3_padding():
    with pytest.raises(ValueError, match="circular padding"):
        validate_av(video(57, 76), audio())


def test_pack_unpack_preserves_av_exactly():
    streams = (video(), audio())
    packed, shapes = pack_streams(streams)
    restored = unpack_streams(packed, shapes)
    assert all(torch.equal(a, b) for a, b in zip(streams, restored, strict=True))


def test_resize_is_spatial_only():
    source = video(40, 54, t=5)
    resized = resize_video(source, 48, 64)
    assert resized.shape == (1, 24, 5, 48, 64)
