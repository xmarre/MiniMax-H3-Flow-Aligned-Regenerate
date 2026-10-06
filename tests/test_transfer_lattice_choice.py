"""Cross-grid lattice choice: geometry, provider selection and band views."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate.geometry import (
    H3_DENSE_PATCH_CENTER_LATTICE,
    H3_ROPE_BOX_HALF_PIXEL_LATTICE,
    resize_spatial_5d_h3_patch_lattice,
)
from h3_flow_regenerate.partitioned_band import target_band_source_view
from h3_flow_regenerate.partitioned_stage import PartitionedTargetBandGeometry
from h3_flow_regenerate.transfer_lattice import (
    H3_ROPE_BOX_TRANSFER_LATTICE,
    H3_TRANSFER_LATTICE,
    H3PatchLatticeTransferProvider,
)


@pytest.mark.parametrize("source,target", [((32, 48), (48, 72)), ((48, 32), (72, 48)), ((16, 16), (30, 30))])
def test_rope_box_lattice_is_the_half_pixel_map_for_equal_aspect(source, target):
    value = torch.randn(1, 3, 2, *source, generator=torch.Generator().manual_seed(4))
    box = resize_spatial_5d_h3_patch_lattice(value, *target, lattice=H3_ROPE_BOX_HALF_PIXEL_LATTICE)
    half_pixel = F.interpolate(value, size=(2, *target), mode="trilinear", align_corners=False)
    # Only float32 coordinate rounding separates the two constructions.
    torch.testing.assert_close(box, half_pixel, rtol=0, atol=1e-4)


@pytest.mark.parametrize("source,target", [((32, 44), (54, 72)), ((44, 32), (72, 54))])
def test_the_two_lattices_differ_by_one_constant_translation(source, target):
    rows = torch.arange(source[0], dtype=torch.float32)[:, None].expand(*source)
    cols = torch.arange(source[1], dtype=torch.float32)[None, :].expand(*source)
    index = torch.stack((rows, cols))[None, :, None]
    center = resize_spatial_5d_h3_patch_lattice(index, *target)
    box = resize_spatial_5d_h3_patch_lattice(index, *target, lattice=H3_ROPE_BOX_HALF_PIXEL_LATTICE)
    delta = (box - center)[..., 3:-3, 3:-3]
    expected = 1.0 - math.sqrt(source[0] * source[1]) / math.sqrt(target[0] * target[1])
    torch.testing.assert_close(delta, torch.full_like(delta, -expected), rtol=0, atol=1e-4)
    assert H3_DENSE_PATCH_CENTER_LATTICE == H3_TRANSFER_LATTICE


class _Provider:
    h3_patch_lattice_api = 2

    def __init__(self, lattices=None):
        if lattices is not None:
            self.h3_transport_lattices = lattices
        self.calls = []

    def upscale_clean_video_h3_patch_lattice(self, video, **kwargs):
        self.calls.append(kwargs)
        return video


def test_provider_keeps_the_historical_call_for_the_default_lattice():
    provider = _Provider()
    wrapped = H3PatchLatticeTransferProvider(provider)
    wrapped.upscale_clean_video(torch.zeros(1), target_h=4, target_w=6)
    assert provider.calls == [{"target_h": 4, "target_w": 6}]
    assert wrapped.lattice == H3_TRANSFER_LATTICE and wrapped.calls == 1


def test_provider_passes_the_rope_box_lattice_only_when_advertised():
    with pytest.raises(RuntimeError, match="h3_transport_lattices"):
        H3PatchLatticeTransferProvider(_Provider(), lattice=H3_ROPE_BOX_TRANSFER_LATTICE)
    provider = _Provider((H3_TRANSFER_LATTICE, H3_ROPE_BOX_TRANSFER_LATTICE))
    wrapped = H3PatchLatticeTransferProvider(provider, lattice=H3_ROPE_BOX_TRANSFER_LATTICE)
    wrapped.upscale_clean_video(torch.zeros(1), target_h=4, target_w=6)
    assert provider.calls == [{"target_h": 4, "target_w": 6, "spatial_lattice": H3_ROPE_BOX_TRANSFER_LATTICE}]
    with pytest.raises(ValueError, match="transport lattice"):
        H3PatchLatticeTransferProvider(provider, lattice="half_pixel_latent_v1")


def test_band_source_view_projects_head_frames_with_the_selected_lattice():
    band = PartitionedTargetBandGeometry(
        protected_t=1, band_t=1, temporal=3, source_h=4, source_w=6, target_h=8, target_w=12
    )
    video = torch.randn(1, 2, 3, 8, 12, generator=torch.Generator().manual_seed(9))
    video[:, :, 2:, 4:] = 0
    video[:, :, 2:, :, 6:] = 0
    for lattice in (H3_TRANSFER_LATTICE, H3_ROPE_BOX_TRANSFER_LATTICE):
        view = target_band_source_view(video, band, lattice=lattice)
        expected = resize_spatial_5d_h3_patch_lattice(video[:, :, :2], 4, 6, lattice=lattice)
        torch.testing.assert_close(view[:, :, :2], expected, rtol=0, atol=0)
        torch.testing.assert_close(view[:, :, 2:], video[:, :, 2:, :4, :6], rtol=0, atol=0)
    assert not torch.equal(
        target_band_source_view(video, band),
        target_band_source_view(video, band, lattice=H3_ROPE_BOX_TRANSFER_LATTICE),
    )
