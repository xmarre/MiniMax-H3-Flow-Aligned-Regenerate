from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .geometry import normalize_target_geometry, pack_streams, unpack_streams, validate_av
from .guidance import conditional_renoise_alignment, conditional_renoise_target
from .sigma import H3_VIDEO_SHIFT, normalized_coordinate

H3_LATENT_UPSCALER_API_VERSION = 1
H3_LATENT_UPSCALER_KIND = "minimax_h3_learned_latent_upscaler"


def validate_learned_upscaler_provider(provider: Any) -> dict[str, Any]:
    if provider is None:
        raise ValueError("learned_3d handoff requires a connected H3_LATENT_UPSCALER provider")
    api_version = getattr(provider, "api_version", None)
    if api_version != H3_LATENT_UPSCALER_API_VERSION:
        raise ValueError(
            f"unsupported H3 latent-upscaler provider API {api_version!r}; expected {H3_LATENT_UPSCALER_API_VERSION}"
        )
    kind = getattr(provider, "kind", None)
    if kind != H3_LATENT_UPSCALER_KIND:
        raise ValueError(f"unsupported H3 latent-upscaler provider kind {kind!r}")
    upscale = getattr(provider, "upscale_clean_video", None)
    if not callable(upscale):
        raise TypeError("H3 latent-upscaler provider is missing callable upscale_clean_video")
    model_name = getattr(provider, "model_name", None)
    device = getattr(provider, "device", None)
    precision = getattr(provider, "precision", None)
    offload = getattr(provider, "offload_after_upscale", None)
    inference_device = getattr(provider, "inference_device", device)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("H3 latent-upscaler provider is missing a checkpoint identity")
    if device not in {"cuda", "cpu"}:
        raise ValueError("H3 latent-upscaler provider device must be cuda or cpu")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("H3 latent-upscaler provider precision must be fp32, fp16, or bf16")
    if not isinstance(offload, bool):
        raise TypeError("H3 latent-upscaler provider offload_after_upscale must be boolean")
    if inference_device not in {"cuda", "cpu"}:
        raise ValueError("H3 latent-upscaler provider inference_device must be cuda or cpu")
    return {
        "api_version": api_version,
        "kind": kind,
        "model_name": model_name,
        "device": device,
        "inference_device": inference_device,
        "precision": precision,
        "offload_after_upscale": offload,
        "upscale": upscale,
    }


@dataclass(frozen=True, slots=True)
class ProgressiveHandoffConfig:
    target_latent_h: int | None = None
    target_latent_w: int | None = None
    target_scale: float | None = None
    handoff_coordinate: float = 0.35
    handoff_selection: str = "fixed"
    auto_min_coordinate: float = 0.2
    auto_max_coordinate: float = 0.55
    transfer_mode: str = "bicubic"
    matching_mode: str = "conditional_renoise"
    seed_offset: int = 0x4833464C4F57
    min_high_steps: int = 2

    def __post_init__(self) -> None:
        explicit = self.target_latent_h is not None or self.target_latent_w is not None
        if explicit == (self.target_scale is not None):
            raise ValueError("provide either target latent H/W or target scale")
        if explicit:
            if self.target_latent_h is None or self.target_latent_h < 2 or self.target_latent_h % 2:
                raise ValueError("target latent H must be positive and even")
            if self.target_latent_w is None or self.target_latent_w < 2 or self.target_latent_w % 2:
                raise ValueError("target latent W must be positive and even")
        elif not math.isfinite(float(self.target_scale)) or float(self.target_scale) <= 1.0:
            raise ValueError("target scale must be finite and greater than 1")
        if not 0 < self.handoff_coordinate < 1 or not math.isfinite(self.handoff_coordinate):
            raise ValueError("handoff coordinate must be finite and inside (0, 1)")
        if self.handoff_selection not in {"fixed", "auto_compute"}:
            raise ValueError("handoff selection must be fixed or auto_compute")
        if not 0 < self.auto_min_coordinate <= self.auto_max_coordinate < 1:
            raise ValueError("automatic handoff bounds must lie inside (0, 1)")
        if self.matching_mode != "conditional_renoise":
            raise ValueError("only the derived conditional_renoise handoff is currently supported")
        if self.transfer_mode != "bicubic":
            raise ValueError("source-input progressive handoff currently supports bicubic transfer only")
        if self.min_high_steps < 1:
            raise ValueError("min_high_steps must be positive")

    def resolve_target(self, source_h: int, source_w: int) -> tuple[int, int]:
        if self.target_scale is not None:
            target_h, target_w = normalize_target_geometry(
                source_h=source_h,
                source_w=source_w,
                scale=self.target_scale,
                policy="nearest",
            )
        else:
            target_h, target_w = int(self.target_latent_h), int(self.target_latent_w)
        if target_h < int(source_h) or target_w < int(source_w):
            raise ValueError("progressive handoff target must not shrink either video axis")
        if target_h == int(source_h) and target_w == int(source_w):
            raise ValueError("progressive handoff target must increase at least one video axis")
        return target_h, target_w

    def resolve_coordinate(self, source_h: int, source_w: int, target_h: int, target_w: int) -> float:
        if self.handoff_selection == "fixed":
            return self.handoff_coordinate
        area_ratio = (target_h * target_w) / (source_h * source_w)
        estimate = self.handoff_coordinate / math.sqrt(area_ratio)
        return min(self.auto_max_coordinate, max(self.auto_min_coordinate, estimate))


@dataclass(frozen=True, slots=True)
class ProgressiveTargetInputConfig:
    """Run early denoising on a smaller grid while the workflow stays target-sized.

    This mode is designed for Continuum and other pipelines whose latent/session
    contract must remain on the final output grid. The wrapper derives a low-grid
    sampler invocation internally and returns to the original target grid at the
    handoff, so downstream spatial contracts never observe a geometry change.

    Exact Native Masked video protection defaults to the conservative target-grid
    fallback. ``target_sparse_lifter`` is an experimental continuation path that
    keeps the sampler state and protected rows on the exact target grid while
    reducing only the early H3 transformer token stream. ``mixed_grid_low_suffix``
    samples a real low-grid suffix while independently supplying the original
    target-grid prefix to H3, then performs learned suffix transfer.
    """

    source_latent_h: int | None = None
    source_latent_w: int | None = None
    source_scale: float | None = None
    handoff_coordinate: float = 0.35
    handoff_selection: str = "fixed"
    auto_min_coordinate: float = 0.2
    auto_max_coordinate: float = 0.55
    transfer_mode: str = "bicubic"
    matching_mode: str = "conditional_renoise"
    seed_offset: int = 0x4833464C4F57
    source_noise_offset: int = 0x48334C4F574C52
    min_high_steps: int = 2
    exact_prefix_mode: str = "fallback"
    suffix_dc_bridge: bool = True
    learned_upscaler: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        explicit = self.source_latent_h is not None or self.source_latent_w is not None
        if explicit == (self.source_scale is not None):
            raise ValueError("provide either source latent H/W or source scale")
        if explicit:
            if self.source_latent_h is None or self.source_latent_h < 2 or self.source_latent_h % 2:
                raise ValueError("source latent H must be positive and even")
            if self.source_latent_w is None or self.source_latent_w < 2 or self.source_latent_w % 2:
                raise ValueError("source latent W must be positive and even")
        elif not math.isfinite(float(self.source_scale)) or not 0.0 < float(self.source_scale) < 1.0:
            raise ValueError("source scale must be finite and inside (0, 1)")
        if not 0 < self.handoff_coordinate < 1 or not math.isfinite(self.handoff_coordinate):
            raise ValueError("handoff coordinate must be finite and inside (0, 1)")
        if self.handoff_selection not in {"fixed", "auto_compute"}:
            raise ValueError("handoff selection must be fixed or auto_compute")
        if not 0 < self.auto_min_coordinate <= self.auto_max_coordinate < 1:
            raise ValueError("automatic handoff bounds must lie inside (0, 1)")
        if self.matching_mode != "conditional_renoise":
            raise ValueError("only the derived conditional_renoise handoff is currently supported")
        if self.transfer_mode not in {"bicubic", "learned_3d"}:
            raise ValueError("target-input handoff transfer must be bicubic or learned_3d")
        if self.transfer_mode == "learned_3d":
            validate_learned_upscaler_provider(self.learned_upscaler)
        if self.exact_prefix_mode not in {"fallback", "target_sparse_lifter", "mixed_grid_low_suffix"}:
            raise ValueError("unsupported exact_prefix_mode")
        if self.exact_prefix_mode == "mixed_grid_low_suffix" and self.transfer_mode != "learned_3d":
            raise ValueError("mixed-grid continuation requires learned_3d transfer")
        if not isinstance(self.suffix_dc_bridge, bool):
            raise TypeError("suffix_dc_bridge must be boolean")
        if self.min_high_steps < 1:
            raise ValueError("min_high_steps must be positive")

    def resolve_source(self, target_h: int, target_w: int) -> tuple[int, int]:
        if self.source_scale is not None:
            source_h, source_w = normalize_target_geometry(
                source_h=target_h,
                source_w=target_w,
                scale=self.source_scale,
                policy="nearest",
            )
        else:
            source_h, source_w = int(self.source_latent_h), int(self.source_latent_w)
        if source_h > int(target_h) or source_w > int(target_w):
            raise ValueError("target-input progressive source must not exceed either target video axis")
        if source_h == int(target_h) and source_w == int(target_w):
            raise ValueError("target-input progressive source must reduce at least one video axis")
        return source_h, source_w

    def resolve_coordinate(self, source_h: int, source_w: int, target_h: int, target_w: int) -> float:
        if self.handoff_selection == "fixed":
            return self.handoff_coordinate
        area_ratio = (target_h * target_w) / (source_h * source_w)
        estimate = self.handoff_coordinate / math.sqrt(area_ratio)
        return min(self.auto_max_coordinate, max(self.auto_min_coordinate, estimate))


def select_handoff_index(
    sigmas: torch.Tensor,
    coordinate: float,
    *,
    min_high_steps: int = 2,
    video_shift: float = H3_VIDEO_SHIFT,
) -> int:
    if sigmas.ndim != 1 or sigmas.numel() < 4:
        raise ValueError("progressive handoff requires at least three sampling intervals")
    if not bool(torch.isfinite(sigmas).all().item()):
        raise ValueError("progressive handoff requires finite sigma values")
    if not math.isfinite(float(coordinate)) or not 0.0 < float(coordinate) < 1.0:
        raise ValueError("progressive handoff coordinate must be finite and inside (0, 1)")
    if not math.isfinite(float(video_shift)) or float(video_shift) <= 0.0:
        raise ValueError("progressive handoff video shift must be finite and positive")
    if bool((sigmas[1:] >= sigmas[:-1]).any()):
        raise ValueError("progressive handoff requires a strictly descending sigma schedule")
    candidates = torch.arange(1, sigmas.numel() - min_high_steps, device=sigmas.device)
    if candidates.numel() == 0:
        raise ValueError("sigma schedule leaves no valid handoff interval")
    base_coordinates = normalized_coordinate(sigmas[candidates], video_shift=video_shift)
    distances = (base_coordinates - float(coordinate)).abs()
    return int(candidates[int(distances.argmin())].item())


def deterministic_video_noise(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu").to(device=device, dtype=dtype)


def build_handoff_state(
    *,
    source_packed_state: torch.Tensor,
    source_x0_packed: torch.Tensor,
    source_shapes: list[tuple[int, ...]],
    sigma: float,
    target_h: int,
    target_w: int,
    seed: int,
    transfer_mode: str = "bicubic",
    learned_upscaler: Any | None = None,
    transfer_metrics: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    if len(source_shapes) != 2:
        raise ValueError("progressive H3 handoff requires exactly video and audio streams")
    if not 0 < sigma < 1:
        raise ValueError("handoff sigma must be strictly inside (0, 1)")
    source_video, source_audio = unpack_streams(source_packed_state, source_shapes)
    x0_video, _x0_audio = unpack_streams(source_x0_packed, source_shapes)
    validate_av(source_video, source_audio)
    if x0_video.shape != source_video.shape:
        raise ValueError("source x0 video geometry does not match the handoff state")
    if (target_h, target_w) == tuple(source_video.shape[-2:]):
        return source_packed_state.clone(), list(source_shapes)
    noise = deterministic_video_noise(
        (source_video.shape[0], source_video.shape[1], source_video.shape[2], target_h, target_w),
        seed=seed,
        device=source_video.device,
        dtype=source_video.dtype,
    )
    if transfer_mode == "bicubic":
        target_video = conditional_renoise_alignment(
            x0_video,
            target_h=target_h,
            target_w=target_w,
            sigma=float(sigma),
            noise=noise,
            transfer_mode="bicubic",
        )
        report = {"transfer_mode": "bicubic"}
    elif transfer_mode == "learned_3d":
        provider = validate_learned_upscaler_provider(learned_upscaler)
        learned_started = time.perf_counter()
        learned_x0 = provider["upscale"](
            x0_video,
            target_h=int(target_h),
            target_w=int(target_w),
        )
        learned_elapsed_ms = (time.perf_counter() - learned_started) * 1000.0
        if not isinstance(learned_x0, torch.Tensor):
            raise TypeError("H3 latent-upscaler provider returned a non-tensor value")
        expected_shape = (
            int(x0_video.shape[0]),
            int(x0_video.shape[1]),
            int(x0_video.shape[2]),
            int(target_h),
            int(target_w),
        )
        if tuple(learned_x0.shape) != expected_shape:
            raise RuntimeError(
                f"H3 latent-upscaler provider returned shape {tuple(learned_x0.shape)}; expected {expected_shape}"
            )
        if not learned_x0.is_floating_point():
            raise TypeError("H3 latent-upscaler provider returned a non-floating tensor")
        if not bool(torch.isfinite(learned_x0).all().item()):
            raise RuntimeError("H3 latent-upscaler provider returned NaN or Inf values")
        target_video = conditional_renoise_target(
            learned_x0,
            sigma=float(sigma),
            noise=noise,
        )
        report = {
            "transfer_mode": "learned_3d",
            "provider_api_version": provider["api_version"],
            "provider_kind": provider["kind"],
            "model_name": provider["model_name"],
            "source_hw": tuple(int(value) for value in x0_video.shape[-2:]),
            "target_hw": (int(target_h), int(target_w)),
            "temporal_length": int(x0_video.shape[2]),
            "input_dtype": str(x0_video.dtype),
            "input_device": str(x0_video.device),
            "inference_precision": provider["precision"],
            "configured_device": provider["device"],
            "inference_device": provider["inference_device"],
            "learned_upscale_elapsed_ms": learned_elapsed_ms,
            "offload_after_upscale": provider["offload_after_upscale"],
            "offloaded_after_upscale": bool(
                provider["offload_after_upscale"] and provider["inference_device"] == "cuda"
            ),
            "output_dtype": str(learned_x0.dtype),
            "output_device": str(learned_x0.device),
        }
    else:
        raise ValueError(f"unsupported progressive handoff transfer mode {transfer_mode!r}")
    if transfer_metrics is not None:
        transfer_metrics.update(report)
    # The packed sampler carries audio on the video sigma schedule. Its state is
    # preserved byte-for-byte across a purely spatial video transition.
    target_packed, target_shapes = pack_streams((target_video, source_audio.clone()))
    return target_packed, target_shapes
