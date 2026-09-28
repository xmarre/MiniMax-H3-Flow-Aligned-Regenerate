from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .geometry import normalize_target_geometry, pack_streams, unpack_streams, validate_av
from .guidance import conditional_renoise_alignment, conditional_renoise_target
from .sigma import H3_VIDEO_SHIFT, normalized_coordinate

H3_LATENT_UPSCALER_API_VERSION = 1
H3_LATENT_UPSCALER_KIND = "minimax_h3_learned_latent_upscaler"


@dataclass(frozen=True, slots=True)
class CleanVideoPostprocessResult:
    """Validated clean-video replacement returned by the handoff postprocess hook."""

    clean_video: torch.Tensor
    protected_prefix_t: int
    metadata: dict[str, Any] = field(default_factory=dict)


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
    suffix_dc_bridge: bool = False
    learned_upscaler: Any | None = field(default=None, repr=False, compare=False)
    suffix_geometric_bridge: bool = False
    frame_gauge_repair: bool = False
    frame_gauge_residual_mode: str = "off"

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
        if self.suffix_dc_bridge and self.exact_prefix_mode not in {"target_sparse_lifter", "mixed_grid_low_suffix"}:
            raise ValueError("suffix_dc_bridge is only supported by Continuum-specific exact-prefix modes")
        if not isinstance(self.suffix_geometric_bridge, bool):
            raise TypeError("suffix_geometric_bridge must be boolean")
        if self.suffix_geometric_bridge and self.exact_prefix_mode != "mixed_grid_low_suffix":
            raise ValueError("suffix_geometric_bridge requires mixed-grid Continuum")
        if not isinstance(self.frame_gauge_repair, bool):
            raise TypeError("frame_gauge_repair must be boolean")
        if self.frame_gauge_residual_mode not in {"off", "measure"}:
            raise ValueError("frame_gauge_residual_mode must be off or measure")
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


H3_HANDOFF_NOISE_INDEPENDENT = "independent"
H3_HANDOFF_NOISE_SOURCE_RESIDUAL = "source_residual_patch_refinement_v1"
H3_HANDOFF_NOISE_MODES = {
    H3_HANDOFF_NOISE_INDEPENDENT,
    H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
}


def _segment_mean_last(value: torch.Tensor, source_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Average contiguous target bins assigned by floor(target * source / target_size)."""

    target_size = int(value.shape[-1])
    source_size = int(source_size)
    if source_size <= 0 or target_size < source_size:
        raise ValueError("noise refinement segment geometry must be non-shrinking and positive")
    indices = torch.arange(source_size + 1, device=value.device, dtype=torch.int64)
    boundaries = torch.div(
        indices * target_size + source_size - 1,
        source_size,
        rounding_mode="floor",
    )
    starts = boundaries[:-1]
    stops = boundaries[1:]
    counts = stops - starts
    if bool((counts <= 0).any().item()):
        raise RuntimeError("noise refinement produced an empty source-cell support group")

    work = value.to(torch.float32)
    prefix = torch.cat((torch.zeros_like(work[..., :1]), work.cumsum(dim=-1)), dim=-1)
    sums = prefix.index_select(-1, stops) - prefix.index_select(-1, starts)
    return sums / counts.to(dtype=work.dtype), counts


def _patch_phase_group_mean(
    value: torch.Tensor,
    source_grid_h: int,
    source_grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rectangular group means and per-axis support counts."""

    x_mean, count_x = _segment_mean_last(value, source_grid_w)
    y_mean_t, count_y = _segment_mean_last(x_mean.transpose(-1, -2), source_grid_h)
    return y_mean_t.transpose(-1, -2), count_y, count_x


def refine_h3_patch_lattice_residual(
    source_residual: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Refine a source-grid flow residual onto a larger H3 patch lattice.

    Each of H3's four latent positions inside a 2x2 transformer patch is refined
    independently. Target cells are partitioned into contiguous physical patch-grid
    groups owned by one source cell. For a group of size n:

        target_j = source / sqrt(n) + eps_j - mean(eps_group).

    The normalized group sum is exactly the source residual while the added
    target-resolution innovation has zero coarse-group sum. If the source residual
    is iid standard Gaussian, the refined field is also iid standard Gaussian. A
    model-derived residual retains its measured coarse mode rather than being
    replaced with an independent realization.
    """

    if (
        not isinstance(source_residual, torch.Tensor)
        or source_residual.ndim != 5
        or not source_residual.is_floating_point()
    ):
        raise TypeError("H3 residual refinement requires floating BxCxTxHxW source video")
    if not bool(torch.isfinite(source_residual).all().item()):
        raise ValueError("H3 residual refinement requires finite source values")
    b, channels, temporal, source_h, source_w = map(int, source_residual.shape)
    target_h, target_w = int(target_h), int(target_w)
    if source_h % 2 or source_w % 2 or target_h % 2 or target_w % 2:
        raise ValueError("H3 residual refinement requires patch-safe even source/target H/W")
    if target_h < source_h or target_w < source_w:
        raise ValueError("H3 residual refinement target must not shrink either spatial axis")
    if target_h == source_h and target_w == source_w:
        source_rms = float(source_residual.float().square().mean().sqrt().item())
        return source_residual.clone(), {
            "policy": H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
            "source_hw": (source_h, source_w),
            "target_hw": (target_h, target_w),
            "identity": True,
            "projection_rms_error": 0.0,
            "projection_max_abs_error": 0.0,
            "source_residual_rms": source_rms,
            "target_residual_rms": source_rms,
            "extra_h3_nfe": 0,
        }

    innovation = deterministic_video_noise(
        (b, channels, temporal, target_h, target_w),
        seed=seed,
        device=source_residual.device,
        dtype=torch.float32,
    )
    refined = torch.empty_like(innovation)
    projection_error_sq = torch.zeros((), device=source_residual.device, dtype=torch.float64)
    projection_elements = 0
    projection_max = torch.zeros((), device=source_residual.device, dtype=torch.float32)
    min_group = None
    max_group = 0

    source_grid_h = source_h // 2
    source_grid_w = source_w // 2
    target_grid_h = target_h // 2
    target_grid_w = target_w // 2
    target_y_owner = torch.div(
        torch.arange(target_grid_h, device=source_residual.device, dtype=torch.int64) * source_grid_h,
        target_grid_h,
        rounding_mode="floor",
    )
    target_x_owner = torch.div(
        torch.arange(target_grid_w, device=source_residual.device, dtype=torch.int64) * source_grid_w,
        target_grid_w,
        rounding_mode="floor",
    )

    for phase_y in range(2):
        for phase_x in range(2):
            source_phase = source_residual[..., phase_y::2, phase_x::2].to(torch.float32)
            innovation_phase = innovation[..., phase_y::2, phase_x::2]
            innovation_group_mean, count_y, count_x = _patch_phase_group_mean(
                innovation_phase,
                source_grid_h,
                source_grid_w,
            )
            group_counts = count_y[:, None] * count_x[None, :]
            phase_min = int(group_counts.min().item())
            phase_max = int(group_counts.max().item())
            min_group = phase_min if min_group is None else min(min_group, phase_min)
            max_group = max(max_group, phase_max)

            common = source_phase / group_counts.sqrt().to(source_phase)
            common_target = common[..., target_y_owner[:, None], target_x_owner[None, :]]
            mean_target = innovation_group_mean[..., target_y_owner[:, None], target_x_owner[None, :]]
            refined_phase = common_target + innovation_phase - mean_target
            refined[..., phase_y::2, phase_x::2] = refined_phase

            projected_mean, projected_count_y, projected_count_x = _patch_phase_group_mean(
                refined_phase,
                source_grid_h,
                source_grid_w,
            )
            projected_counts = projected_count_y[:, None] * projected_count_x[None, :]
            projected = projected_mean * projected_counts.sqrt().to(projected_mean)
            error = projected - source_phase
            projection_error_sq += error.to(torch.float64).square().sum()
            projection_elements += int(error.numel())
            projection_max = torch.maximum(projection_max, error.abs().max())

    refined = refined.to(dtype=source_residual.dtype)
    projection_rms = float((projection_error_sq / max(projection_elements, 1)).sqrt().item())
    report = {
        "policy": H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
        "source_hw": (source_h, source_w),
        "target_hw": (target_h, target_w),
        "source_patch_grid": (source_grid_h, source_grid_w),
        "target_patch_grid": (target_grid_h, target_grid_w),
        "identity": False,
        "group_size_min": int(min_group or 0),
        "group_size_max": int(max_group),
        "projection_rms_error": projection_rms,
        "projection_max_abs_error": float(projection_max.item()),
        "source_residual_rms": float(source_residual.float().square().mean().sqrt().item()),
        "target_residual_rms": float(refined.float().square().mean().sqrt().item()),
        "source_residual_mean": float(source_residual.float().mean().item()),
        "target_residual_mean": float(refined.float().mean().item()),
        "innovation_seed": int(seed),
        "gaussian_marginal_if_source_standard": True,
        "innovation_coarse_group_sum_zero": True,
        "extra_h3_nfe": 0,
    }
    return refined, report


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
    clean_video_postprocess: Callable[[torch.Tensor], CleanVideoPostprocessResult] | None = None,
    noise_mode: str = H3_HANDOFF_NOISE_INDEPENDENT,
    noise_scale: float = 1.0,
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
    if noise_mode not in H3_HANDOFF_NOISE_MODES:
        raise ValueError(f"unsupported progressive handoff noise mode {noise_mode!r}")
    noise_scale = float(noise_scale)
    if not math.isfinite(noise_scale) or noise_scale <= 0.0:
        raise ValueError("progressive H3 handoff noise_scale must be finite and positive")
    noise_report: dict[str, Any]
    if noise_mode == H3_HANDOFF_NOISE_SOURCE_RESIDUAL:
        source_residual = (
            source_video.to(torch.float32) - (1.0 - float(sigma)) * x0_video.to(torch.float32)
        ) / (float(sigma) * noise_scale)
        noise, noise_report = refine_h3_patch_lattice_residual(
            source_residual,
            target_h=target_h,
            target_w=target_w,
            seed=seed,
        )
        noise = noise.to(source_video)
        reconstruction = (1.0 - float(sigma)) * x0_video.to(torch.float32) + float(sigma) * source_residual
        reconstruction_error = reconstruction - source_video.to(torch.float32)
        noise_report.update(
            source_state_reconstruction_rms_error=float(reconstruction_error.square().mean().sqrt().item()),
            source_state_reconstruction_max_abs_error=float(reconstruction_error.abs().max().item()),
            residual_source="same_sigma_source_state_minus_clean_probe",
            noise_scale=noise_scale,
        )
    else:
        noise = deterministic_video_noise(
            (source_video.shape[0], source_video.shape[1], source_video.shape[2], target_h, target_w),
            seed=seed,
            device=source_video.device,
            dtype=source_video.dtype,
        )
        noise_report = {
            "policy": H3_HANDOFF_NOISE_INDEPENDENT,
            "source_hw": tuple(int(value) for value in source_video.shape[-2:]),
            "target_hw": (int(target_h), int(target_w)),
            "innovation_seed": int(seed),
            "noise_scale": noise_scale,
            "extra_h3_nfe": 0,
        }
    if transfer_mode == "bicubic":
        target_video = conditional_renoise_alignment(
            x0_video,
            target_h=target_h,
            target_w=target_w,
            sigma=float(sigma),
            noise=noise,
            transfer_mode="bicubic",
        )
        report = {"transfer_mode": "bicubic", "handoff_noise": noise_report}
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
        if clean_video_postprocess is None:
            clean_operand = learned_x0
            postprocess_metadata: dict[str, Any] = {
                "enabled": False,
                "result": "baseline",
            }
        else:
            clean_operand = learned_x0.to(noise)
            if torch.is_inference(clean_operand):
                # Learned providers commonly run under ComfyUI's global
                # inference_mode. Inference tensors deliberately have no
                # version counter, but the hook's non-mutation contract relies
                # on one. Materialize only the opt-in postprocess operand as a
                # normal no-grad tensor; the provider output itself remains
                # untouched and the OFF path acquires no copy.
                with torch.inference_mode(False), torch.no_grad():
                    clean_operand = clean_operand.clone()
            input_version = clean_operand._version
            postprocess_result = clean_video_postprocess(clean_operand)
            if clean_operand._version != input_version:
                raise RuntimeError("clean-video postprocess hook mutated its input tensor")
            if not isinstance(postprocess_result, CleanVideoPostprocessResult):
                raise TypeError("clean-video postprocess hook returned an unsupported result")
            processed = postprocess_result.clean_video
            if not torch.is_tensor(processed) or tuple(processed.shape) != expected_shape:
                raise RuntimeError("clean-video postprocess hook changed learned handoff geometry")
            if processed.device != clean_operand.device or processed.dtype != clean_operand.dtype:
                raise RuntimeError("clean-video postprocess hook changed learned handoff device or dtype")
            if not processed.is_floating_point() or not bool(torch.isfinite(processed).all().item()):
                raise RuntimeError("clean-video postprocess hook returned non-finite clean video")
            prefix_t = postprocess_result.protected_prefix_t
            if type(prefix_t) is not int or not 0 <= prefix_t <= int(processed.shape[2]):
                raise RuntimeError("clean-video postprocess hook returned an invalid protected prefix length")
            if not torch.equal(processed[:, :, :prefix_t], clean_operand[:, :, :prefix_t]):
                raise RuntimeError("clean-video postprocess hook altered learned prefix ownership")
            if not isinstance(postprocess_result.metadata, dict):
                raise TypeError("clean-video postprocess hook metadata must be a dictionary")
            clean_operand = processed
            postprocess_metadata = dict(postprocess_result.metadata)
            postprocess_metadata.update(
                enabled=True,
                protected_prefix_t=prefix_t,
                clean_dtype=str(clean_operand.dtype),
                clean_device=str(clean_operand.device),
            )
        target_video = conditional_renoise_target(
            clean_operand,
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
            "clean_video_postprocess": postprocess_metadata,
            "handoff_noise": noise_report,
        }
    else:
        raise ValueError(f"unsupported progressive handoff transfer mode {transfer_mode!r}")
    if transfer_metrics is not None:
        transfer_metrics.update(report)
    # The packed sampler carries audio on the video sigma schedule. Its state is
    # preserved byte-for-byte across a purely spatial video transition.
    target_packed, target_shapes = pack_streams((target_video, source_audio.clone()))
    return target_packed, target_shapes
