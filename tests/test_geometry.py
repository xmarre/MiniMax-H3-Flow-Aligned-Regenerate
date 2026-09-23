import math

import pytest
import torch

from h3_flow_regenerate.geometry import (
    geometry_from_video,
    h3_refine_scale_target_canvas,
    normalize_target_geometry,
    pack_streams,
    pixel_to_safe_latent,
    resize_spatial_5d,
    resize_spatial_5d_h3_patch_lattice,
    resize_video,
    translate_spatial_suffix_5d,
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


def _h3_patch_axis(grid_h, grid_w, axis):
    sqrt_area = math.sqrt(float(grid_h * grid_w))
    dim = grid_h if axis == 0 else grid_w
    ratio = dim / sqrt_area
    return torch.arange(dim, dtype=torch.float32) * (32.0 / sqrt_area) + (1.0 - ratio) * 16.0


def test_h3_patch_lattice_resize_is_exact_identity_when_geometry_matches():
    source = torch.randn(1, 3, 2, 56, 74)
    mapped = resize_spatial_5d_h3_patch_lattice(source, 56, 74)
    assert mapped is source


def test_h3_patch_lattice_resize_preserves_physical_linear_field():
    source_h, source_w = 56, 74
    target_h, target_w = 40, 52
    source_grid = (source_h // 2, source_w // 2)
    target_grid = (target_h // 2, target_w // 2)

    source_y = _h3_patch_axis(*source_grid, 0)
    source_x = _h3_patch_axis(*source_grid, 1)
    source_field = source_y[:, None] + 0.25 * source_x[None, :]
    latent = source_field.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    latent = latent.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    mapped = resize_spatial_5d_h3_patch_lattice(latent, target_h, target_w)
    mapped_patch = mapped[0, 0, 0, ::2, ::2]

    target_y = _h3_patch_axis(*target_grid, 0)
    target_x = _h3_patch_axis(*target_grid, 1)
    expected = target_y[:, None] + 0.25 * target_x[None, :]

    assert torch.allclose(mapped_patch[1:-1, 1:-1], expected[1:-1, 1:-1], rtol=1e-5, atol=3e-4)

    generic = resize_spatial_5d(latent, target_h, target_w, mode="bicubic")
    generic_patch = generic[0, 0, 0, ::2, ::2]
    assert torch.max(torch.abs(generic_patch[1:-1, 1:-1] - expected[1:-1, 1:-1])) > 1e-3


def test_h3_patch_lattice_resize_preserves_patch_subcell_layout():
    source = torch.zeros(1, 1, 1, 12, 16)
    for row in range(2):
        for col in range(2):
            source[..., row::2, col::2] = row * 10.0 + col

    mapped = resize_spatial_5d_h3_patch_lattice(source, 8, 12)
    assert torch.allclose(mapped[..., 0::2, 0::2], torch.zeros_like(mapped[..., 0::2, 0::2]), atol=1e-6)
    assert torch.allclose(mapped[..., 0::2, 1::2], torch.ones_like(mapped[..., 0::2, 1::2]), atol=1e-6)
    assert torch.allclose(mapped[..., 1::2, 0::2], torch.full_like(mapped[..., 1::2, 0::2], 10.0), atol=1e-6)
    assert torch.allclose(mapped[..., 1::2, 1::2], torch.full_like(mapped[..., 1::2, 1::2], 11.0), atol=1e-6)


def test_translate_spatial_suffix_preserves_prefix_and_moves_content_in_requested_direction():
    video = torch.zeros(1, 1, 3, 9, 9, dtype=torch.float32)
    video[:, :, 0] = torch.arange(81, dtype=torch.float32).reshape(1, 1, 9, 9)
    video[:, :, 1:, 4, 4] = 1.0

    shifted = translate_spatial_suffix_5d(video, start_t=1, dx=1.0, dy=-1.0)

    assert torch.equal(shifted[:, :, :1], video[:, :, :1])
    assert shifted[0, 0, 1, 3, 5] == pytest.approx(1.0, abs=1e-6)
    assert shifted[0, 0, 2, 3, 5] == pytest.approx(1.0, abs=1e-6)
    assert shifted[0, 0, 1, 4, 4] == pytest.approx(0.0, abs=1e-6)


def test_translate_spatial_suffix_zero_offset_is_value_preserving_clone():
    video = torch.randn(1, 3, 4, 8, 10)
    shifted = translate_spatial_suffix_5d(video, start_t=2, dx=0.0, dy=0.0)
    assert torch.equal(shifted, video)
    assert shifted.data_ptr() != video.data_ptr()


def test_resize_is_spatial_only():
    source = video(40, 54, t=5)
    resized = resize_video(source, 48, 64)
    assert resized.shape == (1, 24, 5, 48, 64)
