from __future__ import annotations

import torch

from h3_flow_regenerate.vae_decode import (
    H3MiniMaxVAEDecodeLargeTile,
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


class FakeBatchedVAE(FakeVAE):
    def decode(self, latent):
        return super().decode(latent).unsqueeze(0)


def test_large_tile_decode_matches_core_batched_video_output_shape():
    vae = FakeBatchedVAE()
    latent = {"samples": torch.zeros(1, 24, 7, 56, 76)}
    images, report = decode_minimax_h3_large_tile(vae, latent, tile_size=256, tile_overlap=128)
    assert images.shape == (2, 896, 1216, 3)
    assert "output=1216x896" in report
    assert (
        vae.first_stage_model.tile_size,
        vae.first_stage_model.tile_overlap_min,
        vae.first_stage_model.tiling,
    ) == (256, 64, True)


def test_large_tile_decode_is_call_scoped_and_restores_native_profile():
    vae = FakeVAE()
    latent = {"samples": torch.zeros(1, 24, 7, 56, 76)}
    images, report = decode_minimax_h3_large_tile(vae, latent, tile_size=256, tile_overlap=128)
    assert images.shape == (2, 896, 1216, 3)
    assert vae.seen_profile == (256, 128, True)
    assert (
        vae.first_stage_model.tile_size,
        vae.first_stage_model.tile_overlap_min,
        vae.first_stage_model.tiling,
    ) == (256, 64, True)
    assert "output=1216x896" in report
    assert "tiles=9x6" in report


def test_256_128_profile_changes_overlap_without_enlarging_decoder_tiles():
    model = MiniMaxH3VideoVAE()
    native_x, native_y = _tile_boundaries(model, 896, 1216)
    model.tile_overlap_min = 128
    overlap_x, overlap_y = _tile_boundaries(model, 896, 1216)
    assert native_x == [192, 384, 576, 768, 960]
    assert native_y == [160, 320, 480, 640]
    assert overlap_x == [112, 224, 336, 448, 576, 704, 832, 960]
    assert overlap_y == [128, 256, 384, 512, 640]
    assert model.tile_size == 256


def test_tile_profile_rejects_larger_tiles_and_invalid_overlap():
    assert _validate_profile(256, 128, 16) == (256, 128)
    for size, overlap in ((320, 128), (256, 70), (256, 256), (512, 128)):
        try:
            _validate_profile(size, overlap, 16)
        except ValueError:
            pass
        else:
            raise AssertionError((size, overlap))


def test_saved_320_widget_is_forced_back_to_released_256_tile(capsys):
    vae = FakeVAE()
    latent = {"samples": torch.zeros(1, 24, 7, 56, 76)}
    images, report = H3MiniMaxVAEDecodeLargeTile().decode(
        latent,
        vae,
        tile_size=320,
        tile_overlap=128,
    )
    assert images.shape == (2, 896, 1216, 3)
    assert vae.seen_profile == (256, 128, True)
    assert "tile=256px" in report
    output = capsys.readouterr().out
    assert "rejecting persisted tile_size=320px" in output
    assert "forcing released 256px tile extent" in output
