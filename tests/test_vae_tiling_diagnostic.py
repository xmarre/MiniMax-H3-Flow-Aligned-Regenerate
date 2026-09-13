from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.vae_tiling_diagnostic import decode_with_h3_internal_tiling_mode


class MiniMaxH3VideoVAE:
    def __init__(self):
        self.tiling = True
        self.tile_size = 256
        self.tile_overlap_min = 64
        self.vae_ratio = 16


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
    latent = torch.randn(1, 24, 7, 8, 12)

    images, report = decode_with_h3_internal_tiling_mode(vae, {"samples": latent}, mode="current")

    assert vae.calls[0][:3] == (True, 256, 64)
    assert images.shape == (2, 8, 12, 3)
    assert "mode=current" in report
    assert vae.first_stage_model.tiling is True
    assert vae.first_stage_model.tile_size == 256
    assert vae.first_stage_model.tile_overlap_min == 64


def test_untiled_mode_disables_only_for_decode_and_restores_afterward():
    vae = FakeVAE()
    latent = torch.randn(1, 24, 7, 8, 12)

    _images, report = decode_with_h3_internal_tiling_mode(vae, {"samples": latent}, mode="untiled")

    assert vae.calls[0][:3] == (False, 256, 64)
    assert "effective tiling=False" in report
    assert vae.first_stage_model.tiling is True
    assert vae.first_stage_model.tile_size == 256
    assert vae.first_stage_model.tile_overlap_min == 64


def test_custom_tiled_mode_uses_requested_geometry_then_restores():
    vae = FakeVAE()
    latent = torch.randn(1, 24, 7, 8, 12)

    _images, report = decode_with_h3_internal_tiling_mode(
        vae,
        {"samples": latent},
        mode="custom_tiled",
        tile_size=512,
        tile_overlap_min=128,
    )

    assert vae.calls[0][:3] == (True, 512, 128)
    assert "tile_size=512" in report
    assert "tile_overlap_min=128" in report
    assert vae.first_stage_model.tiling is True
    assert vae.first_stage_model.tile_size == 256
    assert vae.first_stage_model.tile_overlap_min == 64


def test_restore_happens_when_decode_raises():
    vae = FakeVAE()
    vae.raise_on_decode = True
    latent = torch.randn(1, 24, 7, 8, 12)

    with pytest.raises(RuntimeError, match="decode failed"):
        decode_with_h3_internal_tiling_mode(vae, {"samples": latent}, mode="untiled")

    assert vae.first_stage_model.tiling is True
    assert vae.first_stage_model.tile_size == 256
    assert vae.first_stage_model.tile_overlap_min == 64


def test_custom_geometry_must_align_to_h3_spatial_ratio():
    vae = FakeVAE()
    latent = torch.randn(1, 24, 7, 8, 12)

    with pytest.raises(ValueError, match="positive multiple"):
        decode_with_h3_internal_tiling_mode(
            vae,
            {"samples": latent},
            mode="custom_tiled",
            tile_size=250,
            tile_overlap_min=64,
        )


def test_rejects_non_h3_vae():
    vae = SimpleNamespace(
        first_stage_model=SimpleNamespace(tiling=True, tile_size=256, tile_overlap_min=64, vae_ratio=16),
        decode=lambda latent: latent,
    )

    with pytest.raises(ValueError, match="MiniMaxH3VideoVAE"):
        decode_with_h3_internal_tiling_mode(
            vae,
            {"samples": torch.zeros(1, 24, 7, 8, 12)},
            mode="untiled",
        )
