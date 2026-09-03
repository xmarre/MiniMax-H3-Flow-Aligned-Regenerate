from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

H3_VIDEO_CHANNELS = 24
H3_AUDIO_CHANNELS = 32
H3_AUDIO_TRACKS = 2
H3_VAE_SPATIAL_DOWNSAMPLE = 16
H3_PATCH_H = 2
H3_PATCH_W = 2
H3_PIXEL_ALIGNMENT = H3_VAE_SPATIAL_DOWNSAMPLE * H3_PATCH_H
H3_NATIVE_BASE_SHORT_EDGE = 768
H3_NATIVE_CANVAS_MULTIPLE = 32
H3_NATIVE_MAX_PIXELS = 768 * 1344


def h3_native_reference_canvas(width: int, height: int) -> tuple[int, int]:
    """Mirror ComfyUI MiniMax H3 adapt_canvas() for an arbitrary target aspect ratio."""
    if isinstance(width, bool) or isinstance(height, bool):
        raise TypeError("canvas width/height must be integers")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("canvas width/height must be positive")
    ratio = width / height
    if ratio >= 1.0:
        nominal_w, nominal_h = H3_NATIVE_BASE_SHORT_EDGE * ratio, H3_NATIVE_BASE_SHORT_EDGE
    else:
        nominal_w, nominal_h = H3_NATIVE_BASE_SHORT_EDGE, H3_NATIVE_BASE_SHORT_EDGE / ratio
    if nominal_w * nominal_h > H3_NATIVE_MAX_PIXELS:
        scale = math.sqrt(H3_NATIVE_MAX_PIXELS / (nominal_w * nominal_h))
        nominal_w *= scale
        nominal_h *= scale
    multiple = H3_NATIVE_CANVAS_MULTIPLE
    return (
        max(multiple, round(nominal_w / multiple) * multiple),
        max(multiple, round(nominal_h / multiple) * multiple),
    )


@dataclass(frozen=True, slots=True)
class H3Geometry:
    batch: int
    latent_t: int
    latent_h: int
    latent_w: int
    pixel_h: int
    pixel_w: int
    padded_h: int
    padded_w: int

    @property
    def patch_safe(self) -> bool:
        return self.latent_h % H3_PATCH_H == 0 and self.latent_w % H3_PATCH_W == 0

    @property
    def video_rows(self) -> int:
        return self.latent_t * (self.padded_h // H3_PATCH_H) * (self.padded_w // H3_PATCH_W)


def geometry_from_video(video: torch.Tensor) -> H3Geometry:
    validate_video(video)
    batch, _, latent_t, latent_h, latent_w = map(int, video.shape)
    padded_h = latent_h + (-latent_h) % H3_PATCH_H
    padded_w = latent_w + (-latent_w) % H3_PATCH_W
    return H3Geometry(
        batch=batch,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        pixel_h=latent_h * H3_VAE_SPATIAL_DOWNSAMPLE,
        pixel_w=latent_w * H3_VAE_SPATIAL_DOWNSAMPLE,
        padded_h=padded_h,
        padded_w=padded_w,
    )


def normalize_target_geometry(
    *,
    source_h: int,
    source_w: int,
    scale: float | None = None,
    target_h: int | None = None,
    target_w: int | None = None,
    policy: str = "ceil",
) -> tuple[int, int]:
    """Resolve an even latent H/W without allowing native circular padding.

    Dimensions are latent-space dimensions. ``ceil`` never undershoots the requested
    size; ``nearest`` may move by one latent cell. Explicit odd targets are normalized
    deterministically and are always reported by the caller.
    """
    if source_h < 2 or source_w < 2:
        raise ValueError("source latent H/W must both be at least 2")
    if scale is not None:
        if target_h is not None or target_w is not None:
            raise ValueError("provide scale or explicit target H/W, not both")
        if not isinstance(scale, (float, int)) or isinstance(scale, bool) or float(scale) <= 0:
            raise ValueError("scale must be a positive finite number")
        raw_h, raw_w = source_h * float(scale), source_w * float(scale)
    else:
        if target_h is None or target_w is None:
            raise ValueError("explicit target_h and target_w are both required")
        raw_h, raw_w = float(target_h), float(target_w)
    if not torch.isfinite(torch.tensor([raw_h, raw_w])).all().item():
        raise ValueError("target geometry must be finite")
    if policy == "ceil":
        h = int(torch.ceil(torch.tensor(raw_h)).item())
        w = int(torch.ceil(torch.tensor(raw_w)).item())
        h += (-h) % H3_PATCH_H
        w += (-w) % H3_PATCH_W
    elif policy == "nearest":
        h = round(raw_h / H3_PATCH_H) * H3_PATCH_H
        w = round(raw_w / H3_PATCH_W) * H3_PATCH_W
    else:
        raise ValueError(f"unsupported geometry policy {policy!r}")
    if h < 2 or w < 2:
        raise ValueError("normalized target latent H/W must both be at least 2")
    return h, w


def pixel_to_safe_latent(height: int, width: int, *, policy: str = "ceil") -> tuple[int, int]:
    if height < H3_PIXEL_ALIGNMENT or width < H3_PIXEL_ALIGNMENT:
        raise ValueError(f"target pixels must be at least {H3_PIXEL_ALIGNMENT}x{H3_PIXEL_ALIGNMENT}")
    if policy == "ceil":
        h = (height + H3_PIXEL_ALIGNMENT - 1) // H3_PIXEL_ALIGNMENT
        w = (width + H3_PIXEL_ALIGNMENT - 1) // H3_PIXEL_ALIGNMENT
    elif policy == "nearest":
        h = round(height / H3_PIXEL_ALIGNMENT)
        w = round(width / H3_PIXEL_ALIGNMENT)
    else:
        raise ValueError(f"unsupported geometry policy {policy!r}")
    return int(h * H3_PATCH_H), int(w * H3_PATCH_W)


def validate_video(video: Any) -> torch.Tensor:
    if not isinstance(video, torch.Tensor):
        raise TypeError("H3 video state must be a torch.Tensor")
    if video.ndim != 5 or video.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(f"H3 video state must be Bx24xTxHxW, got {tuple(video.shape)}")
    if video.shape[0] != 1:
        raise ValueError("native MiniMax H3 currently requires batch size 1")
    if min(video.shape[2:]) < 1:
        raise ValueError("H3 video axes must be non-empty")
    if not video.is_floating_point():
        raise TypeError("H3 video state must use floating point")
    return video


def validate_audio(audio: Any) -> torch.Tensor:
    if not isinstance(audio, torch.Tensor):
        raise TypeError("H3 audio state must be a torch.Tensor")
    if audio.ndim != 4 or audio.shape[1] != H3_AUDIO_CHANNELS or audio.shape[2] != H3_AUDIO_TRACKS:
        raise ValueError(f"H3 audio state must be Bx32x2xT, got {tuple(audio.shape)}")
    if audio.shape[0] != 1 or audio.shape[-1] < 1:
        raise ValueError("H3 audio requires batch size 1 and a non-empty time axis")
    if not audio.is_floating_point():
        raise TypeError("H3 audio state must use floating point")
    return audio


def validate_av(video: Any, audio: Any, *, require_patch_safe: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    video = validate_video(video)
    audio = validate_audio(audio)
    if require_patch_safe and (video.shape[-2] % 2 or video.shape[-1] % 2):
        raise ValueError("H3 video latent H/W must be even; odd geometry would invoke native circular padding")
    return video, audio


def resize_spatial_5d(tensor: torch.Tensor, target_h: int, target_w: int, *, mode: str = "bicubic") -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5 or not tensor.is_floating_point():
        raise TypeError("spatial resize input must be a floating-point BxCxTxHxW tensor")
    if target_h % 2 or target_w % 2:
        raise ValueError("target latent H/W must be even")
    if mode not in {"nearest", "bilinear", "bicubic", "area"}:
        raise ValueError(f"unsupported spatial transfer mode {mode!r}")
    b, c, t, h, w = tensor.shape
    work = tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    kwargs: dict[str, Any] = {"size": (target_h, target_w), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = target_h < h or target_w < w
    out = F.interpolate(work, **kwargs)
    return out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).to(tensor)


def resize_video(video: torch.Tensor, target_h: int, target_w: int, *, mode: str = "bicubic") -> torch.Tensor:
    validate_video(video)
    return resize_spatial_5d(video, target_h, target_w, mode=mode)


def pack_streams(streams: Iterable[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    shapes: list[tuple[int, ...]] = []
    flattened: list[torch.Tensor] = []
    for tensor in streams:
        shapes.append(tuple(int(v) for v in tensor.shape))
        flattened.append(tensor.reshape(tensor.shape[0], 1, -1))
    if not flattened:
        raise ValueError("at least one stream is required")
    return torch.cat(flattened, dim=-1), shapes


def unpack_streams(packed: torch.Tensor, shapes: Iterable[tuple[int, ...]]) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    offset = 0
    for shape in shapes:
        count = 1
        for size in shape[1:]:
            count *= int(size)
        result.append(packed[..., offset : offset + count].reshape((packed.shape[0], *shape[1:])))
        offset += count
    if offset != packed.shape[-1]:
        raise ValueError(f"packed state has {packed.shape[-1] - offset} unmatched values")
    return result
