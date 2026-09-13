"""Diagnostic MiniMax-H3 VAE spatial-overlap A/B controls.

MiniMax-H3's ViT3D decoder uses spatial tiling as part of the reference decode
semantics, not merely as a memory fallback. This diagnostic therefore keeps
that semantic tile size intact and varies only the overlap/stitch geometry for
one decode call, then restores the loaded VAE in ``finally``.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

_DECODE_LOCK = threading.RLock()
_MODES = ("current", "overlap_probe")
_RETIRED_UNSAFE_MODES = ("untiled", "custom_tiled")


def _h3_first_stage_model(vae: Any) -> Any:
    model = getattr(vae, "first_stage_model", None)
    if model is None:
        raise ValueError("MiniMax H3 VAE diagnostic requires a ComfyUI VAE with first_stage_model")
    required = ("tiling", "tile_size", "tile_overlap_min", "vae_ratio", "split_tiles")
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


def _validate_overlap(model: Any, overlap_min: int) -> None:
    ratio = int(model.vae_ratio)
    tile_size = int(model.tile_size)
    if ratio <= 0:
        raise ValueError(f"invalid MiniMax-H3 VAE spatial ratio: {ratio}")
    if overlap_min < 0 or overlap_min >= tile_size:
        raise ValueError(f"tile_overlap_min must satisfy 0 <= overlap < semantic tile_size ({tile_size})")
    if overlap_min % ratio:
        raise ValueError(f"tile_overlap_min must be a multiple of the H3 VAE spatial ratio ({ratio})")


def _tile_layout(model: Any, latent: torch.Tensor) -> dict[str, Any]:
    ratio = int(model.vae_ratio)
    pixel_h = int(latent.shape[-2]) * ratio
    pixel_w = int(latent.shape[-1]) * ratio
    y_idx, _y_len, y_overlap = model.split_tiles(pixel_h)
    x_idx, _x_len, x_overlap = model.split_tiles(pixel_w)
    return {
        "pixel_hw": (pixel_h, pixel_w),
        "tile_size": int(model.tile_size),
        "tile_overlap_min": int(model.tile_overlap_min),
        "y_starts": tuple(int(value) for value in y_idx),
        "x_starts": tuple(int(value) for value in x_idx),
        "y_overlaps": tuple(int(value) for value in y_overlap),
        "x_overlaps": tuple(int(value) for value in x_overlap),
    }


def decode_with_h3_overlap_mode(
    vae: Any,
    samples: dict[str, Any],
    *,
    mode: str,
    tile_overlap_min: int = 128,
) -> tuple[torch.Tensor, str]:
    """Decode once while preserving H3's semantic tile size.

    ``current`` leaves the loaded decoder untouched. ``overlap_probe`` keeps
    internal tiling enabled and keeps the loaded ``tile_size`` unchanged, while
    temporarily changing only ``tile_overlap_min``. This moves stitch positions
    without changing the ViT3D decoder's semantic tile extent.
    """

    if mode in _RETIRED_UNSAFE_MODES:
        raise ValueError(
            f"mode={mode!r} was withdrawn: MiniMax-H3 Core explicitly treats internal tiling as reference "
            "decode semantics. Disabling tiling or changing tile_size is not a valid quality A/B. "
            "Use mode='overlap_probe'."
        )
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

    if mode == "overlap_probe":
        if not original_tiling:
            raise ValueError("overlap_probe requires the loaded MiniMax-H3 VAE's reference internal tiling to be enabled")
        _validate_overlap(model, int(tile_overlap_min))

    with _DECODE_LOCK:
        if mode == "overlap_probe":
            model.tiling = True
            model.tile_size = original_tile_size
            model.tile_overlap_min = int(tile_overlap_min)
        try:
            effective_layout = _tile_layout(model, latent)
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
        "MiniMax-H3 internal VAE overlap diagnostic: "
        f"mode={mode}; pixel_hw={effective_layout['pixel_hw']}; semantic_tile_size={effective_layout['tile_size']}; "
        f"tile_overlap_min={effective_layout['tile_overlap_min']}; "
        f"x_starts={effective_layout['x_starts']} y_starts={effective_layout['y_starts']}; "
        f"x_overlaps={effective_layout['x_overlaps']} y_overlaps={effective_layout['y_overlaps']}; "
        f"restored tiling={original_tiling} tile_size={original_tile_size} tile_overlap_min={original_overlap}. "
        "Generation latents are unchanged; only stitch overlap/positions for this decode call can differ."
    )
    return images, report


class H3VAETilingDiagnosticDecode:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Decode the same MiniMax-H3 latent while preserving the H3 VAE's reference semantic tile size. "
        "Current mode leaves the VAE unchanged; overlap_probe changes only the internal spatial overlap, moving "
        "stitch positions so visible banding can be tested without invalidating the ViT3D decoder geometry."
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
                "tile_overlap_min": ("INT", {"default": 128, "min": 0, "max": 240, "step": 16}),
            }
        }

    def decode(self, samples, vae, mode, tile_overlap_min):
        return decode_with_h3_overlap_mode(
            vae,
            samples,
            mode=mode,
            tile_overlap_min=tile_overlap_min,
        )


NODE_CLASS_MAPPINGS = {"H3VAETilingDiagnosticDecode": H3VAETilingDiagnosticDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"H3VAETilingDiagnosticDecode": "MiniMax H3 VAE Internal Tiling Diagnostic Decode"}
