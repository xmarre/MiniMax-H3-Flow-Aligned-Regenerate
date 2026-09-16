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
        # Keep direct Python calls consistent with the inherited Target Input
        # handoff default. Mixed-Grid additionally enables its historical seam
        # repair by default for compatibility with serialized workflows.
        if handoff_transfer is None:
            handoff_transfer = "learned_3d"
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
    """Deprecated compatibility implementation for serialized Mixed-Grid workflows."""

    CATEGORY = "MiniMax H3/flow regenerate/deprecated"
    EXACT_PREFIX_MODE = "mixed_grid_low_suffix"
    DESCRIPTION = (
        "Deprecated compatibility path retained for one release so existing serialized Mixed-Grid "
        "workflows keep their original semantics. Do not use for new production workflows; use "
        "MiniMax H3 Progressive Handoff (Target Input). The old node ID is intentionally not remapped "
        "because remapping would silently change saved workflow behavior. Mixed-Grid is not a release gate."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import copy

        inputs = copy.deepcopy(super().INPUT_TYPES())
        required = inputs["required"]

        # Preserve the historical serialized-node contract unchanged during the
        # compatibility window. Deprecation changes discovery/documentation only.
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
                "tooltip": "Deprecated Mixed-Grid compatibility requires the connected H3 latent-upscaler provider.",
            },
        )
        required["suffix_dc_bridge"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Historical Mixed-Grid one-token suffix DC bridge. Retained unchanged for compatibility; "
                    "new workflows should use the Target Input exact-prefix fallback instead."
                ),
            },
        )
        inputs.setdefault("optional", {})["suffix_geometric_bridge"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Historical Mixed-Grid seam-repair control retained unchanged for serialized workflows. "
                    "It is deprecated and is not part of the current production acceptance matrix."
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
    "H3ProgressiveMixedGridHandoff": "MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]",
}
