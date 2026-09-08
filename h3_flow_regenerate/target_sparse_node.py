from __future__ import annotations

from .comfy_compat import patch_flow_model
from .geometric_provider import with_source_trajectory_context
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
        handoff_transfer="bicubic",
        learned_upscaler=None,
        suffix_dc_bridge=True,
        suffix_geometric_bridge=False,
    ):
        if self.EXACT_PREFIX_MODE == "mixed_grid_low_suffix":
            learned_upscaler = with_source_trajectory_context(learned_upscaler)
        if source_mode == "scale":
            progressive = ProgressiveTargetInputConfig(
                source_scale=source_scale,
                handoff_coordinate=handoff_coordinate,
                handoff_selection=handoff_selection,
                transfer_mode=handoff_transfer,
                exact_prefix_mode=self.EXACT_PREFIX_MODE,
                suffix_dc_bridge=bool(suffix_dc_bridge),
                suffix_geometric_bridge=suffix_geometric_bridge,
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
                suffix_geometric_bridge=suffix_geometric_bridge,
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
        "Real low-resolution Continuum suffix generation with original target-grid protected-prefix "
        "conditioning, learned 3D handoff, exact prefix restoration, and fresh target-grid refinement. "
        "Requires an H3 latent-upscaler provider and VDN external-sequence API v2 when VDN is enabled. "
        "The one-token suffix DC bridge is enabled by default because it removed the validated boundary "
        "flash. An optional cross-grid geometric bridge can correct persistent or measured transient "
        "framing residuals; it remains experimental and off by default."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import copy

        inputs = copy.deepcopy(super().INPUT_TYPES())
        # The inherited patch signature is retained for workflow compatibility.
        for group in ("required", "optional"):
            if "handoff_transfer" in inputs.get(group, {}):
                inputs[group]["handoff_transfer"] = (["learned_3d"], {"default": "learned_3d"})
        inputs["required"]["suffix_dc_bridge"] = (
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
                "default": False,
                "tooltip": (
                    "Experimental Mixed-Grid geometric seam bridge. It independently corroborates source- "
                    "and target-grid framing residuals, then uses genuine low-grid suffix motion to classify "
                    "each authorized axis as persistent, recovering, or unsafe. Persistent axes correct the "
                    "full suffix; measured recovery gets a monotonic transient envelope. Ambiguous axes are "
                    "left unchanged. Off by default pending decoded-media validation."
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
