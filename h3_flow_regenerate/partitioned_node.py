"""Experimental production-shaped node for exact-prefix progressive continuation."""

from __future__ import annotations

import copy

from .comfy_compat import _put_wrapper_first, patch_flow_model
from .guidance import GuidanceConfig
from .handoff import ProgressiveTargetInputConfig
from .metrics import H3FlowMetrics
from .nodes import H3ProgressiveTargetInputHandoff, pixel_to_safe_latent
from .partitioned_outer import partitioned_outer_wrapper
from .partitioned_scheduler import PARTITIONED_PROGRESSIVE_KEY
from .partitioned_transformer import (
    PARTITIONED_WRAPPER_KEY,
    VDN_PARTITIONED_SEQUENCE_API,
    partitioned_diffusion_wrapper,
)
from .runtime import OUTER_WRAPPER_KEY


class H3PartitionedExactPrefixHandoff:
    """Run low/probe/high continuation without resizing exact protected context.

    This remains an experimental node until the real SM120 and decoded-media
    gates are closed. The released Progressive Target Input node is unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(H3ProgressiveTargetInputHandoff.INPUT_TYPES())
        spec["required"].pop("handoff_transfer", None)
        learned = spec["optional"].pop("learned_upscaler")
        spec["required"]["learned_upscaler"] = learned
        return spec

    RETURN_TYPES = ("MODEL", "H3_FLOW_METRICS")
    RETURN_NAMES = ("model", "metrics")
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/flow regenerate/experimental"
    DESCRIPTION = (
        "Experimental exact-prefix progressive continuation. The protected prefix stays on its "
        "target spatial grid during low/probe transformer attention while generated suffix rows "
        "stay on the lower source grid. Requires the matching Sol-H3 partitioned backend and "
        "VDN-H3-Plus partitioned transport development PRs."
    )

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
        learned_upscaler,
        metrics=None,
        temporal_weight=0.20,
    ):
        # Companion capabilities are imported lazily so ordinary Flow users do
        # not acquire cross-custom-node requirements at Comfy startup.
        try:
            from sol_h3.partitioned_history import install_partitioned_history_bridge
            from sol_h3.partitioned_request import PARTITIONED_REQUEST_ABI
            from vdn_h3.partitioned_runtime import install_partitioned_external_sequence_bridge
            from vdn_h3.partitioned_sequence import VDN_PARTITIONED_SEQUENCE_API as vdn_api
        except ImportError as exc:
            raise RuntimeError(
                "partitioned exact-prefix continuation requires the matching Sol-H3 and "
                "VDN-H3-Plus development branches"
            ) from exc
        if not isinstance(PARTITIONED_REQUEST_ABI, str) or not PARTITIONED_REQUEST_ABI:
            raise RuntimeError("Sol-H3 partitioned backend did not publish a valid ABI identity")
        if int(vdn_api) != VDN_PARTITIONED_SEQUENCE_API:
            raise RuntimeError("Flow and VDN partitioned external-sequence APIs do not match")

        common = dict(
            handoff_coordinate=handoff_coordinate,
            handoff_selection=handoff_selection,
            transfer_mode="learned_3d",
            learned_upscaler=learned_upscaler,
            # The released runtime sees only its conservative fallback mode. The
            # new partitioned scheduler is selected by a separate model-local key
            # below, so no deprecated Mixed-Grid mode is reinterpreted.
            exact_prefix_mode="fallback",
            suffix_dc_bridge=False,
            suffix_geometric_bridge=False,
        )
        if source_mode == "scale":
            progressive = ProgressiveTargetInputConfig(
                source_scale=source_scale,
                **common,
            )
        else:
            source_h, source_w = pixel_to_safe_latent(source_height, source_width)
            progressive = ProgressiveTargetInputConfig(
                source_latent_h=source_h,
                source_latent_w=source_w,
                **common,
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
        patched.model_options[PARTITIONED_PROGRESSIVE_KEY] = progressive

        # Extend only this cloned model's VDN object patches. Ordinary VDN calls
        # still delegate byte-for-byte to the released forward; only the explicit
        # partition contract selects heterogeneous grouped execution.
        install_partitioned_history_bridge()
        install_partitioned_external_sequence_bridge(patched)

        import comfy.patcher_extension

        _put_wrapper_first(
            patched,
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            PARTITIONED_WRAPPER_KEY,
            partitioned_diffusion_wrapper,
        )
        # Keep Flow outside Spectrum so low/probe/high sampler lifetimes each
        # traverse Spectrum independently. The partitioned outer wrapper performs
        # preflight-only fallback to the released exact target-grid path.
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            OUTER_WRAPPER_KEY,
        )
        _put_wrapper_first(
            patched,
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            OUTER_WRAPPER_KEY,
            partitioned_outer_wrapper,
        )
        metrics.event(
            "partitioned_exact_prefix_installed",
            sol_abi=PARTITIONED_REQUEST_ABI,
            vdn_external_sequence_api=int(vdn_api),
            scheduler_contract="partitioned_exact_prefix_v1",
            deprecated_mixed_grid_contract_active=False,
            vdn_grouped_softmax_preserved=True,
            vdn_variable_grid_linear_enabled=False,
            guided_audio_overlap=True,
            preflight_target_grid_fallback=True,
            production_default_changed=False,
        )
        return patched, metrics


NODE_CLASS_MAPPINGS = {
    "H3PartitionedExactPrefixHandoff": H3PartitionedExactPrefixHandoff,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PartitionedExactPrefixHandoff": "MiniMax H3 Partitioned Exact-Prefix Handoff [Experimental]",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
