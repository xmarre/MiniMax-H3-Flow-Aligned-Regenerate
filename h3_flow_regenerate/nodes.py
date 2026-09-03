from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .attention import AttentionConfig
from .comfy_compat import patch_flow_model
from .contracts import H3FlowTrajectory
from .geometry import h3_native_reference_canvas, h3_refine_scale_target_canvas, pixel_to_safe_latent
from .guidance import GuidanceConfig
from .handoff import ProgressiveHandoffConfig, ProgressiveTargetInputConfig
from .metrics import H3FlowMetrics
from .reference import apply_reference_budget
from .runtime import FLOW_BINDING_KEY, FlowBinding, conditioning_signature_from_conditioning
from .sigma import H3_VIDEO_SHIFT, resolution_aware_sigmas, resolution_shift_factor


class H3FlowTrajectoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storage": (["system_ram", "vram"], {"default": "system_ram"}),
                "max_runs": ("INT", {"default": 16, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("H3_FLOW_TRAJECTORY",)
    FUNCTION = "create"
    CATEGORY = "MiniMax H3/flow regenerate"

    @classmethod
    def IS_CHANGED(cls, storage, max_runs):
        # The trajectory is mutable execution state, not a reusable cached value.
        # Returning NaN makes Comfy create one fresh handle for each prompt while
        # still sharing that single handle across all downstream nodes in it.
        return float("nan")

    def create(self, storage, max_runs):
        return (H3FlowTrajectory(storage=storage, max_runs=max_runs),)


class H3TrajectoryCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "trajectory": ("H3_FLOW_TRAJECTORY",),
                "capture_forecasts": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "H3_FLOW_METRICS")
    RETURN_NAMES = ("model", "metrics")
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/flow regenerate"

    def patch(self, model, trajectory, capture_forecasts=False):
        metrics = H3FlowMetrics()
        patched, _ = patch_flow_model(
            model,
            trajectory=trajectory,
            guidance=GuidanceConfig(mode="off"),
            clear_progressive=True,
            capture_enabled=True,
            capture_forecasts=capture_forecasts,
            clear_guidance_conditioning_signature=True,
            clear_guidance_run_id=True,
            metrics=metrics,
        )
        return patched, metrics


class H3FlowAlignedRegenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "trajectory": ("H3_FLOW_TRAJECTORY",),
                "guidance_mode": (
                    ["off", "direction", "direction+acceleration", "direction+temporal", "downsample_consistency"],
                    {"default": "direction"},
                ),
                "direction_weight": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "acceleration_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "consistency_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "low_frequency_cutoff": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01}),
                "temporal_weight": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "source_conditioning": ("CONDITIONING",),
                "source_negative": ("CONDITIONING",),
                "metrics": ("H3_FLOW_METRICS",),
            },
        }

    RETURN_TYPES = ("MODEL", "H3_FLOW_METRICS")
    RETURN_NAMES = ("model", "metrics")
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/flow regenerate"

    def patch(
        self,
        model,
        trajectory,
        guidance_mode,
        direction_weight,
        acceleration_weight,
        consistency_weight,
        low_frequency_cutoff,
        source_conditioning=None,
        source_negative=None,
        metrics=None,
        temporal_weight=0.20,
    ):
        metrics = metrics or H3FlowMetrics()
        if source_negative is not None and source_conditioning is None:
            raise ValueError("source_negative requires source_conditioning")
        source_signature = (
            conditioning_signature_from_conditioning(source_conditioning, source_negative)
            if source_conditioning is not None
            else None
        )
        guidance = GuidanceConfig(
            mode=guidance_mode,
            direction_weight=direction_weight,
            acceleration_weight=acceleration_weight,
            temporal_weight=temporal_weight,
            consistency_weight=consistency_weight,
            cutoff=low_frequency_cutoff,
        )
        patched, _ = patch_flow_model(
            model,
            trajectory=trajectory,
            guidance=guidance,
            clear_progressive=True,
            capture_enabled=False,
            capture_forecasts=False,
            guidance_conditioning_signature=source_signature,
            clear_guidance_conditioning_signature=source_signature is None,
            clear_guidance_run_id=True,
            metrics=metrics,
        )
        return patched, metrics


class H3FlowAlignedRefineState:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refine_state": ("H3_CONTINUUM_REFINE_STATE",),
                "trajectory": ("H3_FLOW_TRAJECTORY",),
                "guidance_mode": (
                    ["off", "direction", "direction+acceleration", "direction+temporal", "downsample_consistency"],
                    {"default": "direction"},
                ),
                "direction_weight": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "acceleration_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "consistency_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "low_frequency_cutoff": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01}),
                "temporal_weight": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"metrics": ("H3_FLOW_METRICS",)},
        }

    RETURN_TYPES = ("H3_CONTINUUM_REFINE_STATE", "H3_FLOW_METRICS")
    RETURN_NAMES = ("refine_state", "metrics")
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/flow regenerate"

    def patch(
        self,
        refine_state,
        trajectory,
        guidance_mode,
        direction_weight,
        acceleration_weight,
        consistency_weight,
        low_frequency_cutoff,
        metrics=None,
        temporal_weight=0.20,
    ):
        if not isinstance(refine_state, dict):
            raise TypeError("H3 Continuum refine_state must be a dictionary")
        if type(refine_state.get("api")) is not int or int(refine_state["api"]) != 1:
            raise ValueError("H3 Flow-Aligned Refine State requires H3 Continuum refine_state API 1")
        model = refine_state.get("model")
        positive = refine_state.get("positive")
        if model is None or positive is None:
            raise ValueError("H3 Continuum refine_state is missing model or positive conditioning")
        guidance = GuidanceConfig(
            mode=guidance_mode,
            direction_weight=direction_weight,
            acceleration_weight=acceleration_weight,
            temporal_weight=temporal_weight,
            consistency_weight=consistency_weight,
            cutoff=low_frequency_cutoff,
        )
        metrics = metrics or H3FlowMetrics()
        source_signature = conditioning_signature_from_conditioning(positive)
        source_binding = (getattr(model, "model_options", None) or {}).get(FLOW_BINDING_KEY)
        source_run_id = source_binding.captured_run_id if isinstance(source_binding, FlowBinding) else None
        if source_run_id is None:
            raise RuntimeError(
                "H3 Continuum refine_state MODEL has no captured trajectory provenance. "
                "Ensure MiniMax H3 Trajectory Capture is the final MODEL patch before Continuum "
                "and that Flow-Aligned Refine State uses the same H3_FLOW_TRAJECTORY."
            )
        patched_model, _ = patch_flow_model(
            model,
            trajectory=trajectory,
            guidance=guidance,
            clear_progressive=True,
            capture_enabled=False,
            capture_forecasts=False,
            guidance_conditioning_signature=source_signature,
            guidance_run_id=source_run_id,
            metrics=metrics,
        )
        patched_state = dict(refine_state)
        patched_state["model"] = patched_model
        return patched_state, metrics


class H3ProgressiveHandoff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "trajectory": ("H3_FLOW_TRAJECTORY",),
                "target_mode": (["scale", "pixels"], {"default": "scale"}),
                "scale": ("FLOAT", {"default": 1.2, "min": 1.01, "max": 4.0, "step": 0.01}),
                "target_width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32}),
                "target_height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "handoff_coordinate": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 0.99, "step": 0.01}),
                "handoff_selection": (["fixed", "auto_compute"], {"default": "fixed"}),
                "guidance_mode": (
                    ["off", "direction", "direction+acceleration", "direction+temporal", "downsample_consistency"],
                    {"default": "direction"},
                ),
                "direction_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01}),
                "acceleration_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "consistency_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "low_frequency_cutoff": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01}),
                "temporal_weight": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"metrics": ("H3_FLOW_METRICS",)},
        }

    RETURN_TYPES = ("MODEL", "H3_FLOW_METRICS")
    RETURN_NAMES = ("model", "metrics")
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/flow regenerate"

    def patch(
        self,
        model,
        trajectory,
        target_mode,
        scale,
        target_width,
        target_height,
        handoff_coordinate,
        handoff_selection,
        guidance_mode,
        direction_weight,
        acceleration_weight,
        consistency_weight,
        low_frequency_cutoff,
        metrics=None,
        temporal_weight=0.20,
    ):
        if target_mode == "scale":
            progressive = ProgressiveHandoffConfig(
                target_scale=scale,
                handoff_coordinate=handoff_coordinate,
                handoff_selection=handoff_selection,
            )
        else:
            target_h, target_w = pixel_to_safe_latent(target_height, target_width)
            progressive = ProgressiveHandoffConfig(
                target_latent_h=target_h,
                target_latent_w=target_w,
                handoff_coordinate=handoff_coordinate,
                handoff_selection=handoff_selection,
            )
        guidance = GuidanceConfig(
            mode=guidance_mode,
            direction_weight=direction_weight,
            acceleration_weight=acceleration_weight,
            temporal_weight=temporal_weight,
            consistency_weight=consistency_weight,
            cutoff=low_frequency_cutoff,
        )
        metrics = metrics or H3FlowMetrics()
        patched, _ = patch_flow_model(
            model,
            trajectory=trajectory,
            guidance=guidance,
            progressive=progressive,
            capture_enabled=True,
            capture_forecasts=False,
            clear_guidance_conditioning_signature=True,
            clear_guidance_run_id=True,
            metrics=metrics,
        )
        return patched, metrics


class H3ProgressiveTargetInputHandoff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "trajectory": ("H3_FLOW_TRAJECTORY",),
                "source_mode": (["pixels", "scale"], {"default": "pixels"}),
                "source_scale": ("FLOAT", {"default": 0.84, "min": 0.1, "max": 0.99, "step": 0.01}),
                "source_width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
                "source_height": ("INT", {"default": 640, "min": 32, "max": 8192, "step": 32}),
                "handoff_coordinate": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 0.99, "step": 0.01}),
                "handoff_selection": (["fixed", "auto_compute"], {"default": "fixed"}),
                "guidance_mode": (
                    ["off", "direction", "direction+acceleration", "direction+temporal", "downsample_consistency"],
                    {"default": "direction"},
                ),
                "direction_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01}),
                "acceleration_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "consistency_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "low_frequency_cutoff": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01}),
                "temporal_weight": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"metrics": ("H3_FLOW_METRICS",)},
        }

    RETURN_TYPES = ("MODEL", "H3_FLOW_METRICS")
    RETURN_NAMES = ("model", "metrics")
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/flow regenerate"

    def patch(
        self,
        model,
        trajectory,
        source_mode,
        source_scale,
        source_width,
        source_height,
        handoff_coordinate,
        handoff_selection,
        guidance_mode,
        direction_weight,
        acceleration_weight,
        consistency_weight,
        low_frequency_cutoff,
        metrics=None,
        temporal_weight=0.20,
    ):
        if source_mode == "scale":
            progressive = ProgressiveTargetInputConfig(
                source_scale=source_scale,
                handoff_coordinate=handoff_coordinate,
                handoff_selection=handoff_selection,
            )
        else:
            source_h, source_w = pixel_to_safe_latent(source_height, source_width)
            progressive = ProgressiveTargetInputConfig(
                source_latent_h=source_h,
                source_latent_w=source_w,
                handoff_coordinate=handoff_coordinate,
                handoff_selection=handoff_selection,
            )
        guidance = GuidanceConfig(
            mode=guidance_mode,
            direction_weight=direction_weight,
            acceleration_weight=acceleration_weight,
            temporal_weight=temporal_weight,
            consistency_weight=consistency_weight,
            cutoff=low_frequency_cutoff,
        )
        metrics = metrics or H3FlowMetrics()
        patched, _ = patch_flow_model(
            model,
            trajectory=trajectory,
            guidance=guidance,
            progressive=progressive,
            capture_enabled=True,
            capture_forecasts=False,
            clear_guidance_conditioning_signature=True,
            clear_guidance_run_id=True,
            metrics=metrics,
        )
        return patched, metrics


class H3RefineTargetGeometry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_width": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "base_height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "scale": ("FLOAT", {"default": 1.20, "min": 1.0, "max": 4.0, "step": 0.05}),
                "align": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "keep_proportion": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "H3_FLOW_DIAGNOSTICS")
    RETURN_NAMES = ("target_width", "target_height", "diagnostics")
    FUNCTION = "resolve"
    CATEGORY = "MiniMax H3/flow regenerate/experimental"
    DESCRIPTION = (
        "Geometry-only mirror of the integrated MiniMax H3 Latent Upscaler + Refine "
        "scale-by-multiplier sizing. Feed the existing pre-Continuum MP width/height here, "
        "then use the outputs as the direct Continuum target for E0/E1. This node performs "
        "no latent upscale and no sampling."
    )

    def resolve(self, base_width, base_height, scale, align, keep_proportion):
        target_width, target_height = h3_refine_scale_target_canvas(
            base_width,
            base_height,
            scale,
            align=align,
            keep_proportion=keep_proportion,
        )
        return (
            target_width,
            target_height,
            {
                "base_width": int(base_width),
                "base_height": int(base_height),
                "scale": float(scale),
                "align": int(align),
                "keep_proportion": bool(keep_proportion),
                "target_width": int(target_width),
                "target_height": int(target_height),
                "semantics": "LBH refine scale-by-multiplier geometry only; no upscale/refine execution",
            },
        )


class H3ResolutionAwareSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "mode": (
                    ["off", "resolution_aware", "calibrated"],
                    {
                        "default": "off",
                        "tooltip": (
                            "off is exact SIGMAS parity. resolution_aware applies the SD3-derived "
                            "relative area map on H3's shared AV coordinate. calibrated is research-only."
                        ),
                    },
                ),
                "source_width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "tooltip": (
                            "0 together with source_height=0 derives the H3-native reference canvas "
                            "automatically from target aspect ratio (768px short edge, 768*1344 area cap, "
                            "32px alignment). Positive values select an explicit research reference."
                        ),
                    },
                ),
                "source_height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "tooltip": (
                            "0 together with source_width=0 derives the H3-native reference canvas "
                            "automatically from target aspect ratio. Positive values select an explicit "
                            "research reference."
                        ),
                    },
                ),
                "target_width": ("INT", {"default": 1024, "min": 32, "max": 8192}),
                "target_height": ("INT", {"default": 768, "min": 32, "max": 8192}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "0 = no relative resolution shift; 1 = analytic sqrt(target_area/reference_area). "
                            "Do not tune before a matched off/1.0 media pair."
                        ),
                    },
                ),
                "calibrated_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 8.0,
                        "step": 0.01,
                        "tooltip": "Direct relative shift factor for controlled research only.",
                    },
                ),
            },
            "optional": {"metrics": ("H3_FLOW_METRICS",)},
        }

    RETURN_TYPES = ("SIGMAS", "H3_FLOW_DIAGNOSTICS")
    RETURN_NAMES = ("sigmas", "diagnostics")
    FUNCTION = "map"
    CATEGORY = "MiniMax H3/flow regenerate/experimental"

    def map(
        self,
        sigmas,
        mode,
        source_width,
        source_height,
        target_width,
        target_height,
        strength,
        calibrated_factor,
        metrics=None,
    ):
        source_width = int(source_width)
        source_height = int(source_height)
        target_width = int(target_width)
        target_height = int(target_height)
        if source_width == 0 and source_height == 0:
            source_width, source_height = h3_native_reference_canvas(target_width, target_height)
            reference_mode = "h3_native_auto"
        elif source_width <= 0 or source_height <= 0:
            raise ValueError("source_width/source_height must both be 0 for auto or both be positive")
        else:
            reference_mode = "manual"
        source_area = source_width * source_height
        target_area = target_width * target_height
        mapped = resolution_aware_sigmas(
            sigmas,
            source_area=source_area,
            target_area=target_area,
            mode=mode,
            strength=strength,
            calibrated_factor=calibrated_factor,
        )
        if mode == "off":
            effective_factor = 1.0
        elif mode == "calibrated":
            effective_factor = float(calibrated_factor)
        else:
            effective_factor = resolution_shift_factor(source_area, target_area, strength)
        diagnostics = {
            "mode": mode,
            "reference_mode": reference_mode,
            "source_width": source_width,
            "source_height": source_height,
            "target_width": target_width,
            "target_height": target_height,
            "source_area": source_area,
            "target_area": target_area,
            "area_ratio": target_area / source_area,
            "extra_shift_factor": effective_factor,
            "base_video_shift": H3_VIDEO_SHIFT,
            "effective_video_shift": H3_VIDEO_SHIFT * effective_factor,
            "reference_semantics": (
                "0/0 source dimensions resolve automatically to ComfyUI's H3-native canvas for "
                "the target aspect ratio; positive source dimensions select an explicit research "
                "reference. Neither mode implies a low-resolution sampling pass."
            ),
            "shared_av_coordinate": True,
        }
        if metrics is not None:
            mapped_cpu = mapped.detach().to(device="cpu", dtype=torch.float64)
            input_cpu = sigmas.detach().to(device="cpu", dtype=torch.float64)
            delta = (mapped_cpu - input_cpu).abs()
            metrics.event(
                "resolution_sigma_map",
                mode=mode,
                reference_mode=reference_mode,
                source_width=int(source_width),
                source_height=int(source_height),
                target_width=int(target_width),
                target_height=int(target_height),
                strength=float(strength),
                calibrated_factor=float(calibrated_factor),
                source_area=int(source_area),
                target_area=int(target_area),
                area_ratio=float(diagnostics["area_ratio"]),
                extra_shift_factor=float(effective_factor),
                base_video_shift=float(H3_VIDEO_SHIFT),
                effective_video_shift=float(H3_VIDEO_SHIFT * effective_factor),
                shared_av_coordinate=True,
                sigma_points=int(sigmas.numel()),
                max_abs_sigma_delta=float(delta.max().item()) if delta.numel() else 0.0,
                exact_identity=bool(torch.equal(mapped, sigmas)),
            )
        return mapped, diagnostics

    DESCRIPTION = (
        "Experimental H3 shared-AV SIGMAS remap. Leave source width/height at 0/0 to derive the "
        "H3-native reference canvas automatically from the connected target dimensions. Positive source "
        "dimensions remain available for explicit research references. No low-resolution pass is run. "
        "The relative SD3 Eq.23 factor is composed with H3's native video shift rather than replacing it."
    )


class H3ReferenceBudget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mode": (["native", "diagnostic", "decoupled_direct_experimental"], {"default": "native"}),
                "max_direct_video_rows": ("INT", {"default": 2048, "min": 64, "max": 65536, "step": 64}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "H3_REFERENCE_REPORT")
    RETURN_NAMES = ("conditioning", "report")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/flow regenerate/experimental"

    def apply(self, conditioning, mode, max_direct_video_rows):
        result, report = apply_reference_budget(
            conditioning,
            mode=mode,
            max_direct_video_rows=max_direct_video_rows,
        )
        return result, asdict(report)


class H3AttentionExperiment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["native", "diagnostic", "experimental_sparse"], {"default": "native"}),
                "layers": ("STRING", {"default": "8,16,24,32,40"}),
                "sparse_window": ("INT", {"default": 4, "min": 1, "max": 32}),
                "global_heads": ("INT", {"default": 8, "min": 0, "max": 56}),
                "max_sequence": ("INT", {"default": 8192, "min": 256, "max": 65536}),
            },
            "optional": {"metrics": ("H3_FLOW_METRICS",)},
        }

    RETURN_TYPES = ("MODEL", "H3_FLOW_METRICS")
    RETURN_NAMES = ("model", "metrics")
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/flow regenerate/experimental"

    def patch(self, model, mode, layers, sparse_window, global_heads, max_sequence, metrics=None):
        try:
            selected = tuple(sorted({int(value.strip()) for value in layers.split(",") if value.strip()}))
        except ValueError as exc:
            raise ValueError("layers must be a comma-separated list of non-negative integers") from exc
        if not selected:
            raise ValueError("layers must select at least one H3 transformer block")
        config = AttentionConfig(
            mode=mode,
            layers=selected,
            sparse_window=sparse_window,
            global_heads=global_heads,
            max_sequence=max_sequence,
        )
        metrics = metrics or H3FlowMetrics()
        if mode == "native":
            return model, metrics
        patched, _ = patch_flow_model(model, attention=config, metrics=metrics)
        return patched, metrics


class H3RuntimeMetricsProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {"metrics": ("H3_FLOW_METRICS",)},
        }

    RETURN_TYPES = ("MODEL", "H3_FLOW_METRICS")
    RETURN_NAMES = ("model", "metrics")
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/flow regenerate/diagnostics"
    DESCRIPTION = (
        "Passive H3 sampler/model-call instrumentation. It installs the flow metrics wrappers without "
        "enabling trajectory capture, guidance, progressive handoff, or attention changes. Place it "
        "after Spectrum when exact/forecast provenance is required."
    )

    def patch(self, model, metrics=None):
        metrics = metrics or H3FlowMetrics()
        patched, _ = patch_flow_model(
            model,
            clear_trajectory=True,
            guidance=GuidanceConfig(mode="off"),
            clear_progressive=True,
            capture_enabled=False,
            capture_forecasts=False,
            clear_guidance_conditioning_signature=True,
            clear_guidance_run_id=True,
            metrics=metrics,
        )
        return patched, metrics


class H3MetricsJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"metrics": ("H3_FLOW_METRICS",)},
            "optional": {
                "filename_prefix": (
                    "STRING",
                    {"default": "h3_flow_regenerate/metrics"},
                )
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/flow regenerate"

    def render(self, metrics, filename_prefix="h3_flow_regenerate/metrics"):
        import folder_paths

        output_dir = Path(folder_paths.get_output_directory())
        target = metrics.autosave_path
        if target is None:
            full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
                filename_prefix,
                str(output_dir),
            )
            file_name = f"{filename}_{counter:05}_.json"
            target = Path(full_output_folder) / file_name
        target = metrics.enable_autosave(target)
        payload = metrics.to_json()
        try:
            relative_path = target.relative_to(output_dir).as_posix()
        except ValueError:
            relative_path = target.as_posix()
        return {
            "ui": {"text": [f"Saving metrics JSON: {relative_path}"]},
            "result": (payload,),
        }


NODE_CLASS_MAPPINGS = {
    "H3FlowTrajectory": H3FlowTrajectoryNode,
    "H3TrajectoryCapture": H3TrajectoryCapture,
    "H3FlowAlignedRegenerate": H3FlowAlignedRegenerate,
    "H3FlowAlignedRefineState": H3FlowAlignedRefineState,
    "H3ProgressiveHandoff": H3ProgressiveHandoff,
    "H3ProgressiveTargetInputHandoff": H3ProgressiveTargetInputHandoff,
    "H3RefineTargetGeometry": H3RefineTargetGeometry,
    "H3ResolutionAwareSigmas": H3ResolutionAwareSigmas,
    "H3ReferenceBudget": H3ReferenceBudget,
    "H3AttentionExperiment": H3AttentionExperiment,
    "H3RuntimeMetricsProbe": H3RuntimeMetricsProbe,
    "H3MetricsJSON": H3MetricsJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FlowTrajectory": "MiniMax H3 Flow Trajectory",
    "H3TrajectoryCapture": "MiniMax H3 Trajectory Capture",
    "H3FlowAlignedRegenerate": "MiniMax H3 Flow-Aligned Regenerate",
    "H3FlowAlignedRefineState": "MiniMax H3 Flow-Aligned Refine State",
    "H3ProgressiveHandoff": "MiniMax H3 Progressive Handoff",
    "H3ProgressiveTargetInputHandoff": "MiniMax H3 Progressive Handoff (Target Input)",
    "H3RefineTargetGeometry": "MiniMax H3 Refine Target Geometry [Experimental]",
    "H3ResolutionAwareSigmas": "MiniMax H3 Resolution-Aware Sigmas [Experimental]",
    "H3ReferenceBudget": "MiniMax H3 Reference Budget [Experimental]",
    "H3AttentionExperiment": "MiniMax H3 Attention Lab [Experimental]",
    "H3RuntimeMetricsProbe": "MiniMax H3 Runtime Metrics Probe",
    "H3MetricsJSON": "MiniMax H3 Metrics JSON",
}
