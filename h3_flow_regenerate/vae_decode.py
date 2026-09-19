"""MiniMax-H3 VAE decode with a bounded larger spatial tile profile.

Core MiniMax-H3 decoding uses independent ViT decoder tiles. At high output
resolutions the native 256px / 64px-overlap profile can leave visible rectangular
context boundaries because each tile receives a different global-attention domain.
This node changes only the spatial decode tiling for one VAE call, restores the
original model attributes afterward, and reports measured seam discontinuities.
"""

from __future__ import annotations

import math
import threading

import torch


_DECODE_LOCK = threading.Lock()


def _h3_video_vae_model(vae: object):
    model = getattr(vae, "first_stage_model", None)
    required = ("tiling", "tile_size", "tile_overlap_min", "vae_ratio", "split_tiles")
    if model is None or not all(hasattr(model, name) for name in required):
        raise ValueError("expected a ComfyUI MiniMax-H3 video VAE")
    if model.__class__.__name__ != "MiniMaxH3VideoVAE":
        raise ValueError(
            "large-tile decode is restricted to ComfyUI MiniMaxH3VideoVAE, "
            f"got {model.__class__.__name__}"
        )
    return model


def _validate_profile(tile_size: int, tile_overlap: int, vae_ratio: int) -> tuple[int, int]:
    tile_size = int(tile_size)
    tile_overlap = int(tile_overlap)
    vae_ratio = int(vae_ratio)
    if vae_ratio <= 0:
        raise ValueError("MiniMax-H3 VAE spatial ratio must be positive")
    if tile_size < 256 or tile_size > 512 or tile_size % vae_ratio:
        raise ValueError(
            f"tile_size must be 256..512 and divisible by the VAE ratio ({vae_ratio})"
        )
    if tile_overlap < 64 or tile_overlap >= tile_size or tile_overlap % vae_ratio:
        raise ValueError(
            f"tile_overlap must be >=64, < tile_size, and divisible by {vae_ratio}"
        )
    return tile_size, tile_overlap


def _tile_boundaries(model: object, height: int, width: int) -> tuple[list[int], list[int]]:
    y_idx, _y_len, _y_overlap = model.split_tiles(int(height))
    x_idx, _x_len, _x_overlap = model.split_tiles(int(width))
    return [int(x) for x in x_idx[1:]], [int(y) for y in y_idx[1:]]


def _edge_ratio(images: torch.Tensor, *, axis: int, position: int, radius: int = 12) -> float:
    # Never cast/copy the complete decoded video for diagnostics: that can be
    # multiple GiB at Continuum resolutions. Promote only the two adjacent
    # pixel lines participating in each scalar edge measurement.
    rgb = images[..., :3]
    if axis == 2:
        length = int(rgb.shape[2])
        if not 1 <= position < length:
            return math.nan
        seam = (
            rgb[:, :, position].float() - rgb[:, :, position - 1].float()
        ).abs().mean()
        candidates = []
        for x in range(max(1, position - radius), min(length, position + radius + 1)):
            if x == position:
                continue
            candidates.append(
                (rgb[:, :, x].float() - rgb[:, :, x - 1].float()).abs().mean()
            )
    elif axis == 1:
        length = int(rgb.shape[1])
        if not 1 <= position < length:
            return math.nan
        seam = (
            rgb[:, position].float() - rgb[:, position - 1].float()
        ).abs().mean()
        candidates = []
        for y in range(max(1, position - radius), min(length, position + radius + 1)):
            if y == position:
                continue
            candidates.append(
                (rgb[:, y].float() - rgb[:, y - 1].float()).abs().mean()
            )
    else:
        raise ValueError("tile seam axis must be image H or W")
    if not candidates:
        return math.nan
    baseline = torch.stack(candidates).median()
    denominator = float(baseline.item())
    return float(seam.item()) / denominator if denominator > 0.0 else math.inf


def _format_seam_report(images: torch.Tensor, x_seams: list[int], y_seams: list[int]) -> str:
    x = [(pos, _edge_ratio(images, axis=2, position=pos)) for pos in x_seams]
    y = [(pos, _edge_ratio(images, axis=1, position=pos)) for pos in y_seams]
    def fmt(items):
        return ",".join(
            f"{pos}:{ratio:.3f}x" if math.isfinite(ratio) else f"{pos}:inf"
            for pos, ratio in items
        ) or "none"

    return f"x_seams=[{fmt(x)}] y_seams=[{fmt(y)}]"


def decode_minimax_h3_large_tile(
    vae: object,
    samples: dict[str, object],
    *,
    tile_size: int = 320,
    tile_overlap: int = 128,
) -> tuple[torch.Tensor, str]:
    if not isinstance(samples, dict) or not torch.is_tensor(samples.get("samples")):
        raise ValueError("samples must be a LATENT dictionary containing a tensor")
    latent = samples["samples"]
    if latent.ndim != 5 or int(latent.shape[1]) != 24:
        raise ValueError("MiniMax-H3 video latent must be [B,24,T,H,W]")

    model = _h3_video_vae_model(vae)
    tile_size, tile_overlap = _validate_profile(
        tile_size, tile_overlap, int(model.vae_ratio)
    )
    output_height = int(latent.shape[-2]) * int(model.vae_ratio)
    output_width = int(latent.shape[-1]) * int(model.vae_ratio)

    with _DECODE_LOCK:
        original = (bool(model.tiling), int(model.tile_size), int(model.tile_overlap_min))
        native_x_seams, native_y_seams = _tile_boundaries(
            model, output_height, output_width
        )
        try:
            model.tiling = True
            model.tile_size = tile_size
            model.tile_overlap_min = tile_overlap
            x_seams, y_seams = _tile_boundaries(model, output_height, output_width)
            images = vae.decode(latent)
        finally:
            model.tiling, model.tile_size, model.tile_overlap_min = original

    if not torch.is_tensor(images) or images.ndim != 4:
        raise RuntimeError(
            f"MiniMax-H3 VAE returned unexpected decoded shape {getattr(images, 'shape', None)}"
        )
    report = (
        "MiniMax-H3 large-tile decode: "
        f"tile={tile_size}px overlap>={tile_overlap}px output={output_width}x{output_height} "
        f"tiles={(len(x_seams)+1)}x{(len(y_seams)+1)}; active_"
        + _format_seam_report(images, x_seams, y_seams)
        + "; old_native_locations_"
        + _format_seam_report(images, native_x_seams, native_y_seams)
        + f"; restored_native_profile={original[1]}/{original[2]}"
    )
    return images, report


class H3MiniMaxVAEDecodeLargeTile:
    CATEGORY = "MiniMax H3/flow regenerate/experimental"
    DESCRIPTION = (
        "Decode MiniMax-H3 video latents with a larger spatial VAE tile and overlap "
        "to reduce rectangular tile-context artifacts. Replaces Core VAE Decode for "
        "the video branch only. The VAE's original tile settings are restored after "
        "every call. 320/128 is the bounded production test profile."
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
                "tile_size": (
                    "INT",
                    {"default": 320, "min": 256, "max": 512, "step": 16},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 128, "min": 64, "max": 256, "step": 16},
                ),
            }
        }

    def decode(self, samples, vae, tile_size, tile_overlap):
        return decode_minimax_h3_large_tile(
            vae,
            samples,
            tile_size=int(tile_size),
            tile_overlap=int(tile_overlap),
        )


NODE_CLASS_MAPPINGS = {
    "H3MiniMaxVAEDecodeLargeTile": H3MiniMaxVAEDecodeLargeTile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MiniMaxVAEDecodeLargeTile": "MiniMax H3 VAE Decode — Large Tile [Experimental]",
}
