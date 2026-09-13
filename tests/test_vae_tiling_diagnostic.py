from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.vae_tiling_diagnostic import decode_with_h3_overlap_mode


class MiniMaxH3VideoVAE:
    def __init__(self):
        self.tiling = True
        self.tile_size = 256
        self.tile_overlap_min = 64
        self.vae_ratio = 16

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

        for index in range(remaining // self.vae_ratio):
            overlaps[index % (count - 1)] += self.vae_ratio

        starts = [0]
        for index in range(count - 1):
            starts.append(starts[-1] + tile_size - overlaps[index])
        return starts, [tile_size] * count, overlaps


class FakeVAE:
    def __init__(self):
        self.first_stage_model = MiniMaxH3VideoVAE()
        self.calls = []
        self.raise_on_decode = False

    def decode(self, latent):
        model = self.first_stage_model
        self.calls.append((bool(model.tiling), int(model.tile_size), int(model.tile_overlap_min), latent.clone()))
        if self.raise_on_decode:
            raise RuntimeError("decode failed")
        return torch.zeros((1, 2, 8, 12, 3), dtype=torch.float32)


def test_current_mode_preserves_native_internal_tiling():
    vae = FakeVAE()
    latent = torch.randn(1, 24, 7, 64, 64)

    images, report = decode_with_h3_overlap_mode(vae, {"samples": latent}, mode="current")

    assert vae.calls[0][:3] == (True, 256, 64)
    assert images.shape == (2, 8, 12, 3)
    assert "mode=current" in report
    assert "semantic_tile_size=256" in report
    assert "x_starts=(0, 192, 384, 576, 768)" in report
    assert vae.first_stage_model.tiling is True
    assert vae.first_stage_model.tile_size == 256
    assert vae.first_stage_model.tile_overlap_min == 64


def test_overlap_probe_changes_only_overlap_and_stitch_positions():
    vae = FakeVAE()
    latent = torch.randn(1, 24, 7, 64, 64)

    _images, report = decode_with_h3_overlap_mode(
        vae,
        {"samples": latent},
        mode="overlap_probe",
        tile_overlap_min=128,
    )

    assert vae.calls[0][:3] == (True, 256, 128)
    assert "semantic_tile_size=256" in report
    assert "tile_overlap_min=128" in report
    assert "x_starts=(0, 128, 256, 384, 512, 640, 768)" in report
    assert vae.first_stage_model.tiling is True
    assert vae.first_stage_model.tile_size == 256
    assert vae.first_stage_model.tile_overlap_min == 64


def test_overlap_probe_restores_loaded_vae_when_decode_raises():
    vae = FakeVAE()
    vae.raise_on_decode = True
    latent = torch.randn(1, 24, 7, 64, 64)

    with pytest.raises(RuntimeError, match="decode failed"):
        decode_with_h3_overlap_mode(
            vae,
            {"samples": latent},
            mode="overlap_probe",
            tile_overlap_min=128,
        )

    assert vae.first_stage_model.tiling is True
    assert vae.first_stage_model.tile_size == 256
    assert vae.first_stage_model.tile_overlap_min == 64


def test_overlap_probe_requires_ratio_aligned_overlap():
    vae = FakeVAE()
    latent = torch.randn(1, 24, 7, 64, 64)

    with pytest.raises(ValueError, match="multiple"):
        decode_with_h3_overlap_mode(
            vae,
            {"samples": latent},
            mode="overlap_probe",
            tile_overlap_min=70,
        )


def test_unsafe_untiled_mode_is_rejected_before_decode():
    vae = FakeVAE()
    latent = torch.randn(1, 24, 7, 64, 64)

    with pytest.raises(ValueError, match="withdrawn"):
        decode_with_h3_overlap_mode(vae, {"samples": latent}, mode="untiled")

    assert vae.calls == []


def test_overlap_probe_does_not_mutate_input_latent():
    vae = FakeVAE()
    latent = torch.randn(1, 24, 7, 64, 64)
    original = latent.clone()

    decode_with_h3_overlap_mode(
        vae,
        {"samples": latent},
        mode="overlap_probe",
        tile_overlap_min=128,
    )

    assert torch.equal(latent, original)


def test_rejects_non_h3_vae():
    vae = SimpleNamespace(
        first_stage_model=SimpleNamespace(
            tiling=True,
            tile_size=256,
            tile_overlap_min=64,
            vae_ratio=16,
            split_tiles=lambda _: ([0], [256], []),
        ),
        decode=lambda latent: latent,
    )

    with pytest.raises(ValueError, match="MiniMaxH3VideoVAE"):
        decode_with_h3_overlap_mode(
            vae,
            {"samples": torch.zeros(1, 24, 7, 64, 64)},
            mode="overlap_probe",
            tile_overlap_min=128,
        )
