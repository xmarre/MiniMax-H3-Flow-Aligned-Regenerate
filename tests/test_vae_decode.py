from __future__ import annotations

import torch

from h3_flow_regenerate.vae_decode import (
    _tile_boundaries,
    _validate_profile,
    decode_minimax_h3_large_tile,
)


class MiniMaxH3VideoVAE:
    def __init__(self):
        self.tiling = True
        self.tile_size = 256
        self.tile_overlap_min = 64
        self.vae_ratio = 16

    def split_tiles(self, input_len):
        import math

        tile_size = self.tile_size
        if tile_size >= input_len:
            return [0], [input_len], []
        count = math.ceil(input_len / tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (count - 1)
            remaining = tile_size * count - sum(overlaps) - input_len
            if remaining < 0:
                count += 1
            else:
                break
        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (count - 1)] += self.vae_ratio
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + tile_size - overlap)
        return starts, [tile_size] * count, overlaps


class FakeVAE:
    def __init__(self):
        self.first_stage_model = MiniMaxH3VideoVAE()
        self.seen_profile = None

    def decode(self, latent):
        model = self.first_stage_model
        self.seen_profile = (model.tile_size, model.tile_overlap_min, model.tiling)
        frames = 2
        height = latent.shape[-2] * model.vae_ratio
        width = latent.shape[-1] * model.vae_ratio
        x = torch.linspace(0.0, 1.0, width).view(1, 1, width, 1)
        return x.expand(frames, height, width, 3).contiguous()


def test_large_tile_decode_is_call_scoped_and_restores_native_profile():
    vae = FakeVAE()
    latent = {"samples": torch.zeros(1, 24, 7, 56, 76)}
    images, report = decode_minimax_h3_large_tile(vae, latent, tile_size=320, tile_overlap=128)
    assert images.shape == (2, 896, 1216, 3)
    assert vae.seen_profile == (320, 128, True)
    assert (
        vae.first_stage_model.tile_size,
        vae.first_stage_model.tile_overlap_min,
        vae.first_stage_model.tiling,
    ) == (256, 64, True)
    assert "output=1216x896" in report
    assert "tiles=6x4" in report


def test_320_128_profile_moves_and_reduces_native_tile_grid():
    model = MiniMaxH3VideoVAE()
    native_x, native_y = _tile_boundaries(model, 896, 1216)
    model.tile_size = 320
    model.tile_overlap_min = 128
    large_x, large_y = _tile_boundaries(model, 896, 1216)
    assert native_x == [192, 384, 576, 768, 960]
    assert native_y == [160, 320, 480, 640]
    assert large_x == [176, 352, 528, 704, 896]
    assert large_y == [192, 384, 576]
    assert (len(large_x) + 1) * (len(large_y) + 1) < (len(native_x) + 1) * (len(native_y) + 1)


def test_tile_profile_rejects_unaligned_or_invalid_values():
    assert _validate_profile(320, 128, 16) == (320, 128)
    for size, overlap in ((300, 128), (320, 70), (256, 256), (528, 128)):
        try:
            _validate_profile(size, overlap, 16)
        except ValueError:
            pass
        else:
            raise AssertionError((size, overlap))
