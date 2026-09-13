"""Diagnostic MiniMax-H3 VAE decode controls for spatial tiling A/B tests.

This module deliberately does not change the production H3 VAE defaults.  It
only overrides the already-loaded MiniMaxH3VideoVAE instance for the lifetime
of one decode call, then restores the original attributes in ``finally``.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

_DECODE_LOCK = threading.RLock()
_MODES = ("current", "untiled", "custom_tiled")


def _h3_first_stage_model(vae: Any) -> Any:
    model = getattr(vae, "first_stage_model", None)
    if model is None:
        raise ValueError("MiniMax H3 VAE diagnostic requires a ComfyUI VAE with first_stage_model")
    required = ("tiling", "tile_size", "tile_overlap_min", "vae_ratio")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"VAE does not expose MiniMax-H3 internal tiling attributes: {joined}")
    if model.__class__.__name__ != "MiniMaxH3VideoVAE":
        raise ValueError(
            "MiniMax H3 VAE diagnostic requires MiniMaxH3VideoVAE; "
            f"got {model.__class__.__module__}.{model.__class__.__name__}"
        )
    return model


def _validate_custom_geometry(model: Any, tile_size: int, tile_overlap_min: int) -> None:
    ratio = int(model.vae_ratio)
    if ratio <= 0:
        raise ValueError(f"invalid MiniMax-H3 VAE spatial ratio: {ratio}")
    if tile_size < ratio or tile_size % ratio:
        raise ValueError(f"tile_size must be a positive multiple of the H3 VAE spatial ratio ({ratio})")
    if tile_overlap_min < 0 or tile_overlap_min >= tile_size:
        raise ValueError("tile_overlap_min must satisfy 0 <= overlap < tile_size")
    if tile_overlap_min % ratio:
        raise ValueError(f"tile_overlap_min must be a multiple of the H3 VAE spatial ratio ({ratio})")


def decode_with_h3_internal_tiling_mode(
    vae: Any,
    samples: dict[str, Any],
    *,
    mode: str,
    tile_size: int = 256,
    tile_overlap_min: int = 64,
) -> tuple[torch.Tensor, str]:
    """Decode once while temporarily overriding H3's internal spatial tiler."""

    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if not isinstance(samples, dict) or "samples" not in samples:
        raise ValueError("expected a ComfyUI LATENT dictionary with a 'samples' tensor")
    latent = samples["samples"]
    if not torch.is_tensor(latent):
        raise ValueError("LATENT 'samples' must be a torch.Tensor")
    if latent.is_nested:
        latent = latent.unbind()[0]

    model = _h3_first_stage_model(vae)
    original_tiling = bool(model.tiling)
    original_tile_size = int(model.tile_size)
    original_overlap = int(model.tile_overlap_min)

    effective_tiling = original_tiling
    effective_tile_size = original_tile_size
    effective_overlap = original_overlap
    if mode == "untiled":
        effective_tiling = False
    elif mode == "custom_tiled":
        _validate_custom_geometry(model, int(tile_size), int(tile_overlap_min))
        effective_tiling = True
        effective_tile_size = int(tile_size)
        effective_overlap = int(tile_overlap_min)

    with _DECODE_LOCK:
        model.tiling = effective_tiling
        model.tile_size = effective_tile_size
        model.tile_overlap_min = effective_overlap
        try:
            images = vae.decode(latent)
        finally:
            model.tiling = original_tiling
            model.tile_size = original_tile_size
            model.tile_overlap_min = original_overlap

    if not torch.is_tensor(images):
        raise ValueError("VAE decode did not return a torch.Tensor")
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    report = (
        "MiniMax-H3 internal VAE tiling diagnostic: "
        f"mode={mode}; effective tiling={effective_tiling} tile_size={effective_tile_size} "
        f"tile_overlap_min={effective_overlap}; restored tiling={original_tiling} "
        f"tile_size={original_tile_size} tile_overlap_min={original_overlap}. "
        "Generation latents are unchanged; only this decode call is affected."
    )
    return images, report


class H3VAETilingDiagnosticDecode:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Decode the same MiniMax-H3 latent with the H3 VAE's own internal spatial tiler left unchanged, disabled, "
        "or temporarily given custom tile geometry. This is a decode-only A/B diagnostic: it restores the VAE's "
        "original attributes after every call and does not alter sampling or the latent."
    )
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "mode": (list(_MODES), {"default": "current"}),
                "tile_size": ("INT", {"default": 256, "min": 16, "max": 8192, "step": 16}),
                "tile_overlap_min": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 16}),
            }
        }

    def decode(self, samples, vae, mode, tile_size, tile_overlap_min):
        return decode_with_h3_internal_tiling_mode(
            vae,
            samples,
            mode=mode,
            tile_size=tile_size,
            tile_overlap_min=tile_overlap_min,
        )


NODE_CLASS_MAPPINGS = {"H3VAETilingDiagnosticDecode": H3VAETilingDiagnosticDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"H3VAETilingDiagnosticDecode": "MiniMax H3 VAE Internal Tiling Diagnostic Decode"}
