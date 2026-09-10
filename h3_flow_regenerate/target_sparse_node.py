from __future__ import annotations

from .comfy_compat import patch_flow_model
from .geometry import pixel_to_safe_latent
from .guidance import GuidanceConfig
from .handoff import ProgressiveTargetInputConfig
from .metrics import H3FlowMetrics
from .nodes import H3ProgressiveTargetInputHandoff


class H3ProgressiveTargetSparseHandoff(H3ProgressiveTargetInputHandoff):
    """Opt-in exact-prefix Continuum research path.

    Chunk 1 retains the normal Progressive Target Input implementation. For a
    Native Masked continuation chunk with exact video protection, the sampler
    latent/mask stay on the target grid while only the early H3 hidden-token
    stream is reduced and lifted back before the native final layer.
    """

    CATEGORY = "MiniMax H3/flow regenerate/experimental"
    EXACT_PREFIX_MODE = "target_sparse_lifter"
    DESCRIPTION = (
        "Experimental Continuum continuation path. Exact Native Masked video prefixes stay on the "
        "target grid; early H3 transformer work retains every protected video row plus a coarse "
        "target-grid anchor lattice for generated rows, then restores the full hidden grid before "
        "H3's native final layer. source_scale/source_width/source_height control anchor density on "
        "exact-prefix chunks, not sampler latent geometry. The one-token suffix DC bridge is "
        "Continuum-specific and enabled by default on canonical exact-prefix boundaries."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import copy

        inputs = copy.deepcopy(super().INPUT_TYPES())
        inputs["required"]["handoff_transfer"] = (
            ["bicubic", "learned_3d"],
            {
                "default": "bicubic",
                "tooltip": (
                    "Target-Sparse retains its dependency-free compatibility default. learned_3d is available "
                    "for controlled comparisons, while Mixed-Grid requires learned_3d."
                ),
            },
        )
        inputs["required"]["suffix_dc_bridge"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Continuum-only one-token per-channel DC seam correction. It calibrates from the first "
                    "actual full-grid H3 predicted-clean boundary, preserves the authoritative prefix, and "
                    "changes only the first generated suffix latent token."
                ),
            },
        )
        return inputs

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
        handoff_transfer=None,
        learned_upscaler=None,
        suffix_dc_bridge=True,
        suffix_geometric_bridge=None,
    ):
        # Keep direct Python calls consistent with each node's ComfyUI defaults.
        # Target-Sparse retains its compatibility defaults; Mixed-Grid requires
        # learned transfer and ships the validated seam-repair path enabled.
        if handoff_transfer is None:
            handoff_transfer = "learned_3d" if self.EXACT_PREFIX_MODE == "mixed_grid_low_suffix" else "bicubic"
        if suffix_geometric_bridge is None:
            suffix_geometric_bridge = self.EXACT_PREFIX_MODE == "mixed_grid_low_suffix"

        if source_mode == "scale":
            progressive = ProgressiveTargetInputConfig(
                source_scale=source_scale,
                handoff_coordinate=handoff_coordinate,
                handoff_selection=handoff_selection,
                transfer_mode=handoff_transfer,
                exact_prefix_mode=self.EXACT_PREFIX_MODE,
                suffix_dc_bridge=bool(suffix_dc_bridge),
                suffix_geometric_bridge=bool(suffix_geometric_bridge),
                learned_upscaler=learned_upscaler,
            )
        else:
            source_h, source_w = pixel_to_safe_latent(source_height, source_width)
            progressive = ProgressiveTargetInputConfig(
                source_latent_h=source_h,
                source_latent_w=source_w,
                handoff_coordinate=handoff_coordinate,
                handoff_selection=handoff_selection,
                transfer_mode=handoff_transfer,
                exact_prefix_mode=self.EXACT_PREFIX_MODE,
                suffix_dc_bridge=bool(suffix_dc_bridge),
                suffix_geometric_bridge=bool(suffix_geometric_bridge),
                learned_upscaler=learned_upscaler,
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


class H3ProgressiveMixedGridHandoff(H3ProgressiveTargetSparseHandoff):
    EXACT_PREFIX_MODE = "mixed_grid_low_suffix"
    DESCRIPTION = (
        "Recommended accelerated exact-prefix Continuum path. It keeps the authoritative target-grid "
        "protected-prefix conditioning, generates a genuine low-grid suffix, performs learned 3D latent "
        "transfer, restores the exact prefix, and starts fresh target-grid refinement. Requires an H3 "
        "latent-upscaler provider and VDN external-sequence API v2 when VDN is enabled. The validated "
        "one-token DC bridge and Mixed-Grid seam-repair path are enabled by default."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import copy

        inputs = copy.deepcopy(super().INPUT_TYPES())
        required = inputs["required"]

        # Canonical production defaults. Keep every visible value aligned with
        # the shipped workflows so a newly added node and an opened example do
        # not silently exercise different handoff policies.
        required["source_mode"] = (["pixels", "scale"], {"default": "scale"})
        required["source_scale"] = ("FLOAT", {"default": 0.70, "min": 0.1, "max": 0.99, "step": 0.01})
        required["source_width"] = ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32})
        required["source_height"] = ("INT", {"default": 640, "min": 32, "max": 8192, "step": 32})
        required["handoff_coordinate"] = ("FLOAT", {"default": 0.35, "min": 0.01, "max": 0.99, "step": 0.01})
        required["handoff_selection"] = (["fixed", "auto_compute"], {"default": "fixed"})
        required["guidance_mode"] = (
            ["off", "direction", "direction+acceleration", "direction+temporal", "downsample_consistency"],
            {"default": "direction+temporal"},
        )
        required["direction_weight"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01})
        required["acceleration_weight"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01})
        required["consistency_weight"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01})
        required["low_frequency_cutoff"] = ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01})
        required["temporal_weight"] = ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01})
        required["handoff_transfer"] = (
            ["learned_3d"],
            {
                "default": "learned_3d",
                "tooltip": "Mixed-Grid requires a connected H3 latent-upscaler provider for learned 3D transfer.",
            },
        )
        required["suffix_dc_bridge"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "One-token suffix-only per-channel DC bridge. Uses the discarded learned prefix as "
                    "calibration while keeping the authoritative Continuum prefix bit-exact. Enabled by "
                    "default after matched multi-boundary decoded-media validation removed the handoff flash."
                ),
            },
        )
        inputs.setdefault("optional", {})["suffix_geometric_bridge"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Enable the validated Mixed-Grid seam-repair path. It publishes the protected-prefix K/V "
                    "measure contract for compatible Sol-H3 backends and applies the independent target "
                    "exact-overlap representation reconciliation after learned transfer. The authoritative "
                    "prefix, generated suffix ownership, audio, masks, conditioning and H3 NFE count are preserved."
                ),
            },
        )
        return inputs


NODE_CLASS_MAPPINGS = {
    "H3ProgressiveTargetSparseHandoff": H3ProgressiveTargetSparseHandoff,
    "H3ProgressiveMixedGridHandoff": H3ProgressiveMixedGridHandoff,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ProgressiveTargetSparseHandoff": "MiniMax H3 Progressive Target-Sparse Continuum [Experimental]",
    "H3ProgressiveMixedGridHandoff": "MiniMax H3 Progressive Mixed-Grid Continuum",
}
