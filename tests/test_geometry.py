import pytest
import torch

from h3_flow_regenerate.geometry import (
    geometry_from_video,
    h3_refine_scale_target_canvas,
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
    assert h3_refine_scale_target_canvas(
        base_width,
        base_height,
        1.20,
        align=32,
        keep_proportion=True,
    ) == expected


def test_refine_scale_target_rejects_invalid_inputs():
    with pytest.raises(ValueError, match=">= 1.0"):
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
