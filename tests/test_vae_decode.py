from __future__ import annotations

import math

import torch

from h3_flow_regenerate.vae_decode import (
    _remap_spatial_position_ids,
    _tile_boundaries,
    _validate_profile,
    decode_minimax_h3_global_spatial_position,
    decode_minimax_h3_large_tile,
)


def _token_ids(patch_dims, *, batch=1):
    axes = []
    for dim_size in patch_dims:
        coords = torch.arange(0.5, dim_size, dtype=torch.float32)
        axes.append(2.0 * coords / dim_size - 1.0)
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0).expand(batch, -1, -1).clone()


class FakePosEmbed:
    def __init__(self):
        self.seen = []

    def forward(self, img_ids):
        self.seen.append(img_ids.detach().clone())
        return img_ids


class FakeDecoder:
    def __init__(self):
        self.num_register_tokens = 4
        self.pos_embed = FakePosEmbed()


class MiniMaxH3VideoVAE:
    def __init__(self):
        self.tiling = True
        self.tile_size = 256
        self.tile_overlap_min = 64
        self.vae_ratio = 16
        self.decoder = FakeDecoder()
        self.seen_position_batches = []

    def split_tiles(self, input_len):
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

    def _decode_pixels(self, z):
        batch, _channels, temporal, height, width = z.shape
        ids = _token_ids((temporal, height, width), batch=batch)
        suffix = torch.zeros(batch, self.decoder.num_register_tokens + 1, 3)
        ids = torch.cat((ids, suffix), dim=1)
        self.decoder.pos_embed.forward(ids)
        return torch.zeros(batch, 3, temporal, height * 16, width * 16)

    def _decode_tile_row(self, z_row, x_idx, x_len):
        # Deliberately batch two neighboring tiles at a time so the diagnostic
        # has to map multiple tile offsets onto one decoder batch.
        slices = [
            z_row[..., pos // self.vae_ratio:(pos + length) // self.vae_ratio]
            for pos, length in zip(x_idx, x_len, strict=True)
        ]
        for k in range(0, len(slices), 2):
            group = slices[k:k + 2]
            decoded = self._decode_pixels(torch.cat(group))
            yield from decoded.chunk(len(group))

    def exercise_tiled_decoder(self, z):
        height = int(z.shape[-2]) * self.vae_ratio
        width = int(z.shape[-1]) * self.vae_ratio
        y_idx, y_len, _y_overlap = self.split_tiles(height)
        x_idx, x_len, _x_overlap = self.split_tiles(width)
        for y_pos, y_length in zip(y_idx, y_len, strict=True):
            z_row = z[
                ...,
                y_pos // self.vae_ratio:(y_pos + y_length) // self.vae_ratio,
                :,
            ]
            list(self._decode_tile_row(z_row, x_idx, x_len))


class FakeVAE:
    def __init__(self):
        self.first_stage_model = MiniMaxH3VideoVAE()
        self.seen_profile = None

    def _output(self, latent):
        model = self.first_stage_model
        frames = 2
        height = latent.shape[-2] * model.vae_ratio
        width = latent.shape[-1] * model.vae_ratio
        x = torch.linspace(0.0, 1.0, width).view(1, 1, width, 1)
        return x.expand(frames, height, width, 3).contiguous()

    def decode(self, latent):
        model = self.first_stage_model
        self.seen_profile = (model.tile_size, model.tile_overlap_min, model.tiling)
        return self._output(latent)


class FakeBatchedVAE(FakeVAE):
    def decode(self, latent):
        return super().decode(latent).unsqueeze(0)


class FakePositionVAE(FakeVAE):
    def decode(self, latent):
        model = self.first_stage_model
        self.seen_profile = (model.tile_size, model.tile_overlap_min, model.tiling)
        model.exercise_tiled_decoder(latent)
        return self._output(latent).unsqueeze(0)


def test_large_tile_decode_matches_core_batched_video_output_shape(capsys):
    vae = FakeBatchedVAE()
    latent = {"samples": torch.zeros(1, 24, 7, 56, 76)}
    images, report = decode_minimax_h3_large_tile(
        vae,
        latent,
        tile_size=320,
        tile_overlap=128,
    )
    assert images.shape == (2, 896, 1216, 3)
    assert "output=1216x896" in report
    assert report in capsys.readouterr().out
    assert (
        vae.first_stage_model.tile_size,
        vae.first_stage_model.tile_overlap_min,
        vae.first_stage_model.tiling,
    ) == (256, 64, True)


def test_large_tile_decode_is_call_scoped_and_restores_native_profile():
    vae = FakeVAE()
    latent = {"samples": torch.zeros(1, 24, 7, 56, 76)}
    images, report = decode_minimax_h3_large_tile(
        vae,
        latent,
        tile_size=320,
        tile_overlap=128,
    )
    assert images.shape == (2, 896, 1216, 3)
    assert vae.seen_profile == (320, 128, True)
    assert (
        vae.first_stage_model.tile_size,
        vae.first_stage_model.tile_overlap_min,
        vae.first_stage_model.tiling,
    ) == (256, 64, True)
    assert "output=1216x896" in report
    assert "tiles=6x4" in report


def test_320_128_profile_moves_grid_but_keeps_six_columns():
    model = MiniMaxH3VideoVAE()
    native_x, native_y = _tile_boundaries(model, 896, 1216)
    model.tile_size = 320
    model.tile_overlap_min = 128
    large_x, large_y = _tile_boundaries(model, 896, 1216)
    assert native_x == [192, 384, 576, 768, 960]
    assert native_y == [160, 320, 480, 640]
    assert large_x == [176, 352, 528, 704, 896]
    assert large_y == [192, 384, 576]
    assert len(large_x) == len(native_x)
    assert len(large_y) < len(native_y)


def test_global_position_remap_preserves_temporal_and_suffix_ids():
    image_ids = _token_ids((2, 2, 3), batch=2)
    suffix = torch.zeros(2, 5, 3)
    ids = torch.cat((image_ids, suffix), dim=1)
    remapped = _remap_spatial_position_ids(
        ids,
        image_token_count=12,
        local_hw=(2, 3),
        full_hw=(6, 8),
        spatial_offsets=[(1, 2), (3, 4)],
    )
    original_images = ids[:, :12].reshape(2, 2, 2, 3, 3)
    mapped_images = remapped[:, :12].reshape(2, 2, 2, 3, 3)

    assert torch.equal(mapped_images[..., 0], original_images[..., 0])
    assert torch.equal(remapped[:, 12:], ids[:, 12:])

    expected_y0 = 2.0 * (torch.tensor([1.5, 2.5]) / 6.0) - 1.0
    expected_x0 = 2.0 * (torch.tensor([2.5, 3.5, 4.5]) / 8.0) - 1.0
    assert torch.allclose(mapped_images[0, 0, :, 0, 1], expected_y0)
    assert torch.allclose(mapped_images[0, 0, 0, :, 2], expected_x0)

    expected_y1 = 2.0 * (torch.tensor([3.5, 4.5]) / 6.0) - 1.0
    expected_x1 = 2.0 * (torch.tensor([4.5, 5.5, 6.5]) / 8.0) - 1.0
    assert torch.allclose(mapped_images[1, 0, :, 0, 1], expected_y1)
    assert torch.allclose(mapped_images[1, 0, 0, :, 2], expected_x1)


def test_global_position_decode_uses_native_geometry_and_restores_overrides(capsys):
    vae = FakePositionVAE()
    model = vae.first_stage_model
    original_row = model._decode_tile_row
    original_pixels = model._decode_pixels
    original_pos_forward = model.decoder.pos_embed.forward
    latent = {"samples": torch.zeros(1, 24, 1, 56, 76)}

    images, report = decode_minimax_h3_global_spatial_position(vae, latent)

    assert images.shape == (2, 896, 1216, 3)
    assert "mode=global_spatial_rope" in report
    assert "tile=256px overlap>=64px" in report
    assert report in capsys.readouterr().out
    assert vae.seen_profile == (256, 64, True)
    assert (model.tile_size, model.tile_overlap_min, model.tiling) == (256, 64, True)

    # Instance overrides are removed, so normal class methods are active again.
    assert model._decode_tile_row.__func__ is original_row.__func__
    assert model._decode_pixels.__func__ is original_pixels.__func__
    assert model.decoder.pos_embed.forward.__func__ is original_pos_forward.__func__

    # First row: first decoder batch contains x tiles at latent offsets 0 and 12
    # (pixel starts 0 and 192). Both retain identical temporal IDs.
    first = model.decoder.pos_embed.seen[0]
    image_count = int(first.shape[1]) - 5
    first_images = first[:, :image_count].reshape(2, 1, 16, 16, 3)
    expected_first_x = 2.0 * (torch.arange(0.5, 16.0) / 76.0) - 1.0
    expected_second_x = 2.0 * (torch.arange(12.5, 28.0) / 76.0) - 1.0
    expected_y = 2.0 * (torch.arange(0.5, 16.0) / 56.0) - 1.0
    assert torch.allclose(first_images[0, 0, 0, :, 2], expected_first_x)
    assert torch.allclose(first_images[1, 0, 0, :, 2], expected_second_x)
    assert torch.allclose(first_images[0, 0, :, 0, 1], expected_y)
    assert torch.equal(first_images[0, ..., 0], first_images[1, ..., 0])



def test_global_position_decode_rejects_non_native_tile_profile():
    vae = FakePositionVAE()
    model = vae.first_stage_model
    model.tile_size = 320
    model.tile_overlap_min = 128
    latent = {"samples": torch.zeros(1, 24, 1, 56, 76)}

    try:
        decode_minimax_h3_global_spatial_position(vae, latent)
    except RuntimeError as exc:
        assert "requires the unchanged Core 256/64 tiled decode profile" in str(exc)
    else:
        raise AssertionError("expected non-native tile geometry to fail closed")

    assert (model.tile_size, model.tile_overlap_min, model.tiling) == (320, 128, True)

def test_tile_profile_rejects_unaligned_or_invalid_values():
    assert _validate_profile(320, 128, 16) == (320, 128)
    for size, overlap in ((300, 128), (320, 70), (256, 256), (528, 128)):
        try:
            _validate_profile(size, overlap, 16)
        except ValueError:
            pass
        else:
            raise AssertionError((size, overlap))
