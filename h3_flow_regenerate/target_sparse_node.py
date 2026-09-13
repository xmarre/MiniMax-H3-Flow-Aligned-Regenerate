from __future__ import annotations

from .comfy_compat import patch_flow_model
from .geometry import pixel_to_safe_latent
from .guidance import GuidanceConfig
from .handoff import ProgressiveTargetInputConfig
from .metrics import H3FlowMetrics
from .nodes import H3ProgressiveTargetInputHandoff


class H3ProgressiveTargetSparseHandoff(H3ProgressiveTargetInputHandoff):
    """Opt-in exact-prefix target-query pruning diagnostic.

    Chunk 1 retains the normal Progressive Target Input implementation. For a
    Native Masked continuation chunk with exact video protection, the sampler
    latent/mask stay on the target grid while only the early H3 hidden-token
    stream is reduced and lifted back before the native final layer.
    """

    CATEGORY = "MiniMax H3/flow regenerate/experimental"
    EXACT_PREFIX_MODE = "target_sparse_lifter"
    DESCRIPTION = (
        "Experimental Continuum target-query pruning path. Exact Native Masked video prefixes stay on the "
        "target grid; early H3 transformer work retains every protected video row plus a coarse target-grid "
        "anchor lattice for generated rows, then restores the full hidden grid before H3's native final layer. "
        "This is separate from Mixed-Grid weighted attention and is not used by the canonical Mixed-Grid path. "
        "The historical one-token suffix DC intervention remains retired."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import copy

        inputs = copy.deepcopy(super().INPUT_TYPES())
        inputs["required"]["suffix_dc_bridge"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Retired diagnostic input. Historical one-token per-channel DC transplantation is ignored; "
                    "the authoritative exact prefix is preserved without that seam intervention."
                ),
            },
        )
        # Keep this visible without shifting any existing serialized widget
        # positions. It is appended as the last optional widget and locked to
        # the single execution mode owned by this dedicated node class.
        inputs.setdefault("optional", {})["exact_prefix_mode"] = (
            [cls.EXACT_PREFIX_MODE],
            {
                "default": cls.EXACT_PREFIX_MODE,
                "tooltip": (
                    "Explicit exact-prefix execution contract. This dedicated node exposes its single valid mode "
                    "for workflow auditability instead of hiding it in the implementation."
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
        exact_prefix_mode=None,
        suffix_dc_bridge=False,
        suffix_geometric_bridge=None,
        attention_measure_profile=None,
        handoff_state_policy=None,
    ):
        if handoff_transfer is None:
            handoff_transfer = "learned_3d"
        if exact_prefix_mode is None:
            exact_prefix_mode = self.EXACT_PREFIX_MODE
        if exact_prefix_mode != self.EXACT_PREFIX_MODE:
            raise ValueError(
                f"{type(self).__name__} only supports exact_prefix_mode={self.EXACT_PREFIX_MODE!r}; "
                f"received {exact_prefix_mode!r}"
            )
        if suffix_geometric_bridge is None:
            suffix_geometric_bridge = exact_prefix_mode == "mixed_grid_low_suffix"

        # PR32 retired the historical one-token DC transplant for both the
        # Target-Sparse diagnostic and Mixed-Grid. Ignore old serialized true
        # values instead of silently reintroducing that decoded seam artifact.
        del suffix_dc_bridge
        effective_suffix_dc_bridge = False

        if source_mode == "scale":
            progressive = ProgressiveTargetInputConfig(
                source_scale=source_scale,
                handoff_coordinate=handoff_coordinate,
                handoff_selection=handoff_selection,
                transfer_mode=handoff_transfer,
                exact_prefix_mode=exact_prefix_mode,
                suffix_dc_bridge=effective_suffix_dc_bridge,
                suffix_geometric_bridge=bool(suffix_geometric_bridge),
                attention_measure_profile=attention_measure_profile,
                handoff_state_policy=handoff_state_policy,
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
                exact_prefix_mode=exact_prefix_mode,
                suffix_dc_bridge=effective_suffix_dc_bridge,
                suffix_geometric_bridge=bool(suffix_geometric_bridge),
                attention_measure_profile=attention_measure_profile,
                handoff_state_policy=handoff_state_policy,
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
        "Accelerated exact-prefix Continuum path. It keeps the authoritative target-grid protected-prefix "
        "conditioning, generates a genuine low-grid suffix, performs learned 3D clean-latent transfer, "
        "restores the exact prefix, and starts fresh target-grid refinement. The optional persistent suffix "
        "rebase can independently correct a held-out-validated per-channel tone-domain offset and a bounded "
        "constant affine frame offset measured directly from the learned upscaler's native prefix/suffix boundary. "
        "Any accepted component applies to the entire generated suffix; raw structural/chroma residual and one-token "
        "DC transplantation remain retired. Requires an H3 latent-upscaler provider and VDN external-sequence "
        "API v2 when VDN is enabled. weighted_measure_v1 is the canonical attention-measure route. State transport "
        "remains experimental and legacy re-noise remains the default."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import copy

        inputs = copy.deepcopy(super().INPUT_TYPES())
        required = inputs["required"]

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
                "default": False,
                "tooltip": (
                    "Retired for Mixed-Grid. The historical one-token per-channel DC injection can decode as "
                    "a coloured multi-frame pulse, so Mixed-Grid ignores older serialized true values."
                ),
            },
        )
        optional = inputs.setdefault("optional", {})
        # The inherited audit widget must be moved behind every pre-existing
        # Mixed-Grid widget so old workflow widget_values retain their indexes.
        optional.pop("exact_prefix_mode", None)
        optional["suffix_geometric_bridge"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Canonical Mixed-Grid suffix representation bridge for the current workflow. It may apply "
                    "a held-out-validated per-channel constant latent bias and/or a bounded constant affine "
                    "coordinate correction derived from the learned upscaler's own native prefix/suffix boundary. "
                    "Each component is independently gated by objective validation; the protected prefix is never "
                    "modified, raw structural/chroma residuals are never transplanted, and future motion is not "
                    "extrapolated. Keep enabled for the matched weighted Mixed-Grid full-stack run."
                ),
            },
        )
        optional["attention_measure_profile"] = (
            ["weighted_measure_v1", "legacy_representative_v1", "off"],
            {
                "default": "weighted_measure_v1",
                "tooltip": (
                    "Mixed-Grid attention measure. weighted_measure_v1 is the canonical all-Q/K/V operator and "
                    "publishes attention_measure_v1 independently of seam controls. legacy_representative_v1 "
                    "keeps the released reduced-K/V comparator; off is an explicit unweighted diagnostic control."
                ),
            },
        )
        optional["handoff_state_policy"] = (
            ["legacy_renoise", "velocity_bicubic_v1", "endpoint_residual_bicubic_v1"],
            {
                "default": "legacy_renoise",
                "tooltip": (
                    "Mixed-Grid handoff state policy. legacy_renoise preserves released state semantics and is the "
                    "current matched comparator. velocity_bicubic_v1 and endpoint_residual_bicubic_v1 remain "
                    "diagnostic candidates; matched runs 00385/00386 were worse than legacy on the current case."
                ),
            },
        )
        optional["exact_prefix_mode"] = (
            [cls.EXACT_PREFIX_MODE],
            {
                "default": cls.EXACT_PREFIX_MODE,
                "tooltip": (
                    "Mixed-Grid exact-prefix execution contract. Visible for workflow auditability and locked to "
                    "mixed_grid_low_suffix on this dedicated node."
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
