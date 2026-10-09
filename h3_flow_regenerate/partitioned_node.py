"""Partitioned exact-prefix progressive continuation for the coordinated Continuum stack."""

from __future__ import annotations

import copy

from .boundary_witness import WITNESS_DIRECTORY_OPTION
from .comfy_compat import _put_wrapper_first, patch_flow_model
from .guidance import GuidanceConfig
from .handoff import ProgressiveTargetInputConfig
from .metrics import H3FlowMetrics
from .nodes import H3ProgressiveTargetInputHandoff, pixel_to_safe_latent
from .partitioned_attention import sol_attention_selected
from .partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_OPTIONS,
    PARTITIONED_AUDIO_POSITION_DOMAIN_OPTIONS,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AV_HANDOFF_SOURCE_OPTIONS,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_OPTIONS,
    PARTITIONED_HANDOFF_TRANSFER_LEARNED,
    PARTITIONED_HANDOFF_TRANSFER_OPTIONS,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS,
    PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OPTIONS,
    PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_OPTIONS,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK,
    PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS,
    PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
    PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCE,
    PARTITIONED_TARGET_BAND_CONTEXT_MIXED,
    PARTITIONED_TARGET_BAND_CONTEXT_OPTIONS,
    PARTITIONED_TARGET_BAND_HANDOFF_STATE_OPTIONS,
    PARTITIONED_TARGET_BAND_HANDOFF_STATE_RENOISE,
    PARTITIONED_TARGET_BAND_TOKENS_DEFAULT,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS,
    PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
    PARTITIONED_VDN_TEMPORAL_CARRIER_OPTIONS,
    apply_partitioned_diagnostic_controls,
)
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

    FRAME_GAUGE_REPAIR_DEFAULT = False
    PRODUCTION_DEFAULT_CHANGED = False

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
        "stay on the lower source grid. Uses the selected attention backend with "
        "VDN-H3-Plus partitioned transport; Sol-H3 attention is optional."
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
        frame_gauge_repair=False,
        frame_gauge_residual_mode="off",
    ):
        # Companion capabilities are imported lazily so ordinary Flow users do
        # not acquire cross-custom-node requirements at Comfy startup.
        try:
            from vdn_h3.partitioned_runtime import install_partitioned_external_sequence_bridge
            from vdn_h3.partitioned_sequence import VDN_PARTITIONED_SEQUENCE_API as vdn_api
        except ImportError as exc:
            raise RuntimeError(
                "partitioned exact-prefix continuation requires matching VDN-H3-Plus partitioned transport"
            ) from exc
        sol_abi = None
        if sol_attention_selected(model.model_options.get("transformer_options", {})):
            try:
                from sol_h3.partitioned_history import install_partitioned_history_bridge
                from sol_h3.partitioned_request import PARTITIONED_REQUEST_ABI
            except ImportError as exc:
                raise RuntimeError("selected Sol-H3 attention requires its partitioned backend") from exc
            if not isinstance(PARTITIONED_REQUEST_ABI, str) or not PARTITIONED_REQUEST_ABI:
                raise RuntimeError("Sol-H3 partitioned backend did not publish a valid ABI identity")
            sol_abi = PARTITIONED_REQUEST_ABI
            install_partitioned_history_bridge()
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
            # The generic bridge flag stays off: partitioned continuation owns
            # its boundary correction after learned transfer, before target-high.
            suffix_dc_bridge=False,
            suffix_geometric_bridge=False,
            frame_gauge_repair=frame_gauge_repair,
            frame_gauge_residual_mode=frame_gauge_residual_mode,
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
            sol_abi=sol_abi,
            vdn_external_sequence_api=int(vdn_api),
            scheduler_contract="partitioned_exact_prefix_v1",
            deprecated_mixed_grid_contract_active=False,
            vdn_grouped_softmax_preserved=True,
            vdn_variable_grid_linear_enabled=True,
            guided_audio_overlap=True,
            partitioned_suffix_dc_bridge=True,
            frame_gauge_repair=bool(frame_gauge_repair),
            frame_gauge_repair_default=self.FRAME_GAUGE_REPAIR_DEFAULT,
            frame_gauge_residual_mode=str(frame_gauge_residual_mode),
            frame_gauge_residual_mode_default="off",
            preflight_target_grid_fallback=True,
            production_default_changed=self.PRODUCTION_DEFAULT_CHANGED,
        )
        return patched, metrics


class H3PartitionedExactPrefixDiagnosticHandoff(H3PartitionedExactPrefixHandoff):
    """Production partitioned exact-prefix handoff.

    The historical class/node ID is retained so existing serialized workflows
    continue to load. The user-facing node is no longer labeled diagnostic.
    """

    FRAME_GAUGE_REPAIR_DEFAULT = False
    PRODUCTION_DEFAULT_CHANGED = True

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(super().INPUT_TYPES())
        spec["required"]["vdn_linear_diagnostic"] = (
            list(PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS),
            {
                "default": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
                "tooltip": (
                    "normal preserves the partitioned VDN learned linear complement; "
                    "bypass_partitioned_linear suppresses the complete complement; "
                    "suppress_cross_grid_temporal_taps keeps the complement active but "
                    "zeros only temporal short-conv taps that cross the target/source grid boundary; "
                    "raw_token_measure keeps both VDN paths active but disables only the matched "
                    "target-prefix density correction in softmax and learned-linear measure policy."
                ),
            },
        )
        spec["required"]["audio_guided_overlap_ticks"] = (
            "INT",
            {
                "default": 16,
                "min": 0,
                "step": 1,
                "tooltip": (
                    "40-Hz audio overlap width for sampler_mask or model_timestep_only. "
                    "exact_mask keeps the carried audio prefix protected and uses zero overlap "
                    "regardless of this width. Any non-negative integer is accepted. "
                    "The applied width uses at most the available carried audio prefix."
                ),
            },
        )
        # Append the new selector after the pre-existing tick widget so saved
        # Flow #54 diagnostic workflows keep their serialized widget positions.
        spec["required"]["audio_guided_overlap_mode"] = (
            list(PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS),
            {
                "default": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
                "tooltip": (
                    "exact_mask preserves the carried audio prefix with matching native sampler input, "
                    "timestep and velocity masks. sampler_mask_exact_timestep is a compatibility alias "
                    "for exact_mask. sampler_mask releases the configured overlap; model_timestep_only "
                    "is an intentionally mismatched timestep diagnostic."
                ),
            },
        )
        # Append after all Flow #54/#57 widgets so existing diagnostic workflow
        # widget positions remain stable.
        spec["required"]["prefix_transformer_context"] = (
            list(PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS),
            {
                "default": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
                "tooltip": (
                    "exact_target_partitioned preserves the production-shaped heterogeneous "
                    "target-prefix/source-suffix transformer. source_carrier_uniform is a bounded "
                    "diagnostic that leaves the low/probe transformer on its native uniform source "
                    "grid while caller-owned exact output restoration remains unchanged."
                ),
            },
        )
        spec["required"]["audio_position_domain"] = (
            list(PARTITIONED_AUDIO_POSITION_DOMAIN_OPTIONS),
            {
                "default": PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
                "tooltip": (
                    "source_carrier follows the low/probe video grid. On target-grid continuation its "
                    "audio spatial positions equal target-grid positions. "
                    "legacy_target remains available as the historical comparison domain."
                ),
            },
        )
        spec["required"]["audio_handoff_source"] = (
            list(PARTITIONED_AUDIO_HANDOFF_SOURCE_OPTIONS),
            {
                "default": PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
                "tooltip": (
                    "main_partitioned keeps the current low/probe audio state. "
                    "source_carrier_uniform_shadow runs an isolated source-grid low-stage shadow "
                    "lifetime and substitutes only its audio sampler state at the learned handoff; "
                    "the main exact-partitioned video state and learned video transfer are retained."
                ),
            },
        )
        # Append after the #61 audio-only selector so existing saved diagnostic
        # widget positions remain stable.
        spec["required"]["av_handoff_source"] = (
            list(PARTITIONED_AV_HANDOFF_SOURCE_OPTIONS),
            {
                "default": PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
                "tooltip": (
                    "main_partitioned preserves the #60/#61 handoff. "
                    "source_carrier_uniform_shadow runs a separate uniform source-grid low+probe "
                    "pair, selects its raw audio sampler state and clean generated video for the "
                    "learned handoff, but keeps the main exact-partitioned captured Flow trajectory."
                ),
            },
        )
        # Append after #62 so every previously serialized diagnostic widget keeps
        # the same positional index.
        spec["required"]["guidance_trajectory_source"] = (
            list(PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_OPTIONS),
            {
                "default": PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
                "tooltip": (
                    "main_exact_partitioned preserves #62. source_carrier_uniform_shadow captures "
                    "only the already-executed source-uniform shadow low+probe video trajectory "
                    "into an isolated temporary store and uses that trajectory for target-high "
                    "Flow guidance; handoff state, exact prefixes, and all audio controls stay #62."
                ),
            },
        )
        # Append after #64 so every previously serialized diagnostic widget keeps
        # the same positional index.
        spec["required"]["low_probe_execution_source"] = (
            list(PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS),
            {
                "default": PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
                "tooltip": (
                    "main_then_shadow keeps the selected exact-prefix low/probe path. Shadow lifetimes "
                    "run only when a shadow handoff selector is explicitly enabled. "
                    "source_carrier_uniform_only selects the alternative source-uniform low/probe path."
                ),
            },
        )
        # Append after every historical selector so existing serialized widget
        # positions remain stable.
        spec["required"]["frame_gauge_repair"] = (
            "BOOLEAN",
            {
                "default": cls.FRAME_GAUGE_REPAIR_DEFAULT,
                "tooltip": (
                    "Enable paired-prefix handoff checks. A same-grid handoff resolves to identity. "
                    "Heterogeneous registration remains subject to its runtime qualification."
                ),
            },
        )
        # Measurement milestone only: horizontal application remains unavailable
        # until the documented hardware/media gate is satisfied.
        spec["required"]["frame_gauge_residual_mode"] = (
            ["off", "measure"],
            {
                "default": "off",
                "tooltip": (
                    "measure exports existing low/high stage tensors without changing sampling. "
                    "Target-band runs also fit band/tail trajectories and export full video snapshots "
                    "within a 256 MiB CPU budget. "
                    "Accepted rigid transactions additionally record bounded regional residual geometry. "
                    "CPU copies and file I/O add diagnostic overhead; off disables these exports."
                ),
            },
        )
        # Append after all prior controls so saved workflow widget positions stay
        # stable. The historical soft_support_v1 value remains loadable, but hardware
        # validation invalidated its pre-high application; it is now diagnostic-only.
        spec["required"]["provider_boundary_stabilization"] = (
            list(PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OPTIONS),
            {
                "default": PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT,
                "tooltip": (
                    "soft_support_v1 leaves sampler state unchanged and emits a bounded post-high "
                    "boundary observation. Its historical pre-high correction is disabled. "
                    "off disables this observation."
                ),
            },
        )
        # Append only: never shift historical serialized widget positions.
        spec["required"]["capture_boundary_witness"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Capture boundary evidence for this execution. Target-band and progressive_uniform_source "
                    "continuation save scheduler-owned stage snapshots under "
                    "output/h3_flow_regenerate/residual_geometry for Local Boundary Audit; other partitioned "
                    "modes save VDN feature tensors under output/h3-flow-boundary-witness. "
                    "Stage copies have a CPU byte budget and add copy and disk-write time. "
                    "This is per-run and does not require an environment variable or ComfyUI restart."
                ),
            },
        )
        # Append after the witness toggle so every existing serialized widget index
        # remains stable. The candidate requires paired capability and is never the default.
        spec["required"]["vdn_temporal_carrier_policy"] = (
            list(PARTITIONED_VDN_TEMPORAL_CARRIER_OPTIONS),
            {
                "default": PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
                "tooltip": (
                    "native_grid_then_map_v1 preserves the current VDN short-conv arithmetic. "
                    "destination_grid_stencil_v1 supports exact-prefix and target-band continuation: cross-grid "
                    "temporal taps map the raw projected feature to the receiving frame lattice "
                    "before the checkpoint spatial stencil. Requires vdn_linear_diagnostic=normal "
                    "and paired VDN capability; same-grid work is unchanged."
                ),
            },
        )
        # Append after every existing selector so saved workflows retain all
        # historical widget positions. Absence/default preserves learned_3d.
        spec["required"]["handoff_transfer_control"] = (
            list(PARTITIONED_HANDOFF_TRANSFER_OPTIONS),
            {
                "default": PARTITIONED_HANDOFF_TRANSFER_LEARNED,
                "tooltip": (
                    "learned_3d preserves the current low->high learned latent handoff. "
                    "bicubic_same_source_control replaces only that clean-video transfer operator "
                    "with deterministic bicubic spatial resize while preserving the same source "
                    "low/probe state, residual/noise transport, exact-prefix restoration, "
                    "postprocess controls and target-high sampling. Diagnostic only."
                ),
            },
        )
        # Append after every existing selector so serialized widget positions remain stable.
        spec["required"]["spatial_stage_control"] = (
            list(PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS),
            {
                "default": PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCE,
                "tooltip": (
                    "progressive_uniform_source (default) runs continuation low/probe as one full-duration "
                    "reduced-grid clip for all generated frames, transfers the whole generated trajectory "
                    "through the learned upscaler, then restores the exact target-grid prefix before high "
                    "refinement. Use normal attention/VDN diagnostics and the default target-band selectors "
                    "with this mode. "
                    "same_grid_target_control runs continuation low/probe directly on the target grid, keeps "
                    "the same handoff split and downstream high stage, and uses an identity "
                    "clean-video transfer. All-generated first chunks retain progressive generation. "
                    "progressive_low_to_high selects the configured reduced continuation grid. "
                    "progressive_target_band keeps the first target_band_tokens generated tokens after the "
                    "protected prefix on the target grid, keeps their native clean prediction and re-noises "
                    "all generated tokens. It runs the remaining "
                    "continuation tokens on the reduced grid with learned transfer. "
                    "Modes other than progressive_uniform_source are retained for existing workflows and "
                    "comparisons."
                ),
            },
        )
        # Append-only after the 00726/00727 spatial selector. This discriminator
        # isolates suffix dense dispatch or target-query non-video key measure.
        spec["required"]["softmax_diagnostic"] = (
            list(PARTITIONED_SOFTMAX_DIAGNOSTIC_OPTIONS),
            {
                "default": PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
                "tooltip": (
                    "normal uses the selected attention backend for generated-suffix query groups. "
                    f"{PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX} forces only those same gathered "
                    "suffix groups through the weighted dense path while preserving the identical "
                    "Q/K/V domain, target-prefix bias, grouped ownership and VDN linear setting. "
                    f"{PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK} extends the target-grid query "
                    "key bias over text/reference/audio rows in low/probe. Reduced-grid and global "
                    "queries, gathered keys, linear measure policy and high-stage policy are preserved. "
                    "Requires paired VDN support and, when selected, Sol support. Diagnostic only."
                ),
            },
        )
        # Append-only historical control. Keep the uncapped integer in the node
        # so old workflows deserialize unchanged and matched hardware reruns can
        # preserve the requested width. Run 01093 retired the prefix-release
        # mutation; positive values are now provenance only.
        spec["required"]["video_guided_overlap_tokens"] = (
            "INT",
            {
                "default": 6,
                "min": 0,
                "step": 1,
                "tooltip": (
                    "Legacy video-overlap width in H3 temporal latent tokens. The value remains uncapped for "
                    "workflow compatibility and provenance. Positive values "
                    "do not feather or regenerate carried video tokens: target-high keeps the authoritative exact "
                    "video prefix for every evaluation and reports requested_tokens with applied_tokens=0 and "
                    "retired_prefix_release=true. Audio overlap is independent."
                ),
            },
        )
        # Append-only: never shift historical serialized widget positions.
        spec["required"]["suffix_dc_bridge"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Continuation only. When enabled, the first generated token that receives learned spatial "
                    "transfer gets the per-channel spatial-mean offset the transfer produced on the carried "
                    "context. Disable to leave that token exactly as transferred. Same-grid continuation "
                    "measures a zero offset either way."
                ),
            },
        )
        spec["required"]["target_band_tokens"] = (
            "INT",
            {
                "default": PARTITIONED_TARGET_BAND_TOKENS_DEFAULT,
                "min": 1,
                "step": 1,
                "tooltip": (
                    "Used only by spatial_stage_control=progressive_target_band: the number of generated H3 "
                    "temporal latent tokens directly after the protected prefix that stay on the target grid "
                    "through low/probe. It must leave at least one generated token on the reduced grid."
                ),
            },
        )
        # Append-only after target_band_tokens so serialized widget positions remain stable.
        spec["required"]["target_band_handoff_state"] = (
            list(PARTITIONED_TARGET_BAND_HANDOFF_STATE_OPTIONS),
            {
                "default": PARTITIONED_TARGET_BAND_HANDOFF_STATE_RENOISE,
                "tooltip": (
                    "Used only by progressive_target_band. renoise_clean re-noises the band's clean "
                    "prediction with the tail's independent handoff noise. carry_raw_band resumes the band "
                    "from its actual low/probe sampler state at the handoff sigma; the protected prefix, "
                    "tail and audio entry are unchanged. Experimental comparison control."
                ),
            },
        )
        spec["required"]["target_band_context"] = (
            list(PARTITIONED_TARGET_BAND_CONTEXT_OPTIONS),
            {
                "default": PARTITIONED_TARGET_BAND_CONTEXT_MIXED,
                "tooltip": (
                    "Used only by progressive_target_band. mixed_grid runs one transformer "
                    "sequence containing target-grid prefix/band and reduced-grid tail. domain_uniform_v1 "
                    "runs two uniform-grid hidden streams per model call: target prefix+band, and reduced "
                    "projected prefix+band plus tail. The streams share no hidden state or conditioning "
                    "rows. domain_uniform_all_stages_v1 also keeps the short head stream separate during "
                    "high refinement, with both streams on the target grid. The learned band/tail handoff "
                    "is unchanged. Experimental controls with additional compute; rendered tone continuity "
                    "is unvalidated."
                ),
            },
        )
        # Append-only after target_band_context so serialized widget positions remain stable.
        spec["required"]["uniform_source_detail_transport"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Used only by spatial_stage_control=progressive_uniform_source. The learned upscaler renders "
                    "fine static detail (text, patterns, texture) differently from the carried target-grid "
                    "frames, so without this the continuation can redraw such detail at the join. When enabled, "
                    "the difference between the carried last frame and the upscaler's rendering of it is added "
                    "to the generated frames before high refinement, weighted per location by how much the "
                    "content there has changed since that frame. The weighting is calibrated on the carried "
                    "frames of the same run and is skipped when it does not improve held-out carried frames."
                ),
            },
        )
        return spec

    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "Continuum handoff with exact caller-visible prefix restoration for the coordinated "
        "VDN-H3-Plus stack using the selected attention backend. Continuation defaults to target-grid "
        "low/probe and identity "
        "handoff; all-generated first chunks retain learned progressive transfer. Preserves "
        "coherent exact audio input/timestep/velocity masks, and exact "
        "caller-visible prefix restoration. Advanced selectors remain available for controlled comparisons."
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
        vdn_linear_diagnostic,
        audio_guided_overlap_mode,
        audio_guided_overlap_ticks,
        prefix_transformer_context,
        audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        audio_handoff_source=PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        av_handoff_source=PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
        guidance_trajectory_source=PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
        low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
        frame_gauge_repair=False,
        frame_gauge_residual_mode="off",
        provider_boundary_stabilization=PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT,
        capture_boundary_witness=False,
        vdn_temporal_carrier_policy=PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
        handoff_transfer_control=PARTITIONED_HANDOFF_TRANSFER_LEARNED,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCE,
        softmax_diagnostic=PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
        video_guided_overlap_tokens=6,
        suffix_dc_bridge=False,
        target_band_tokens=PARTITIONED_TARGET_BAND_TOKENS_DEFAULT,
        target_band_handoff_state=PARTITIONED_TARGET_BAND_HANDOFF_STATE_RENOISE,
        target_band_context=PARTITIONED_TARGET_BAND_CONTEXT_MIXED,
        uniform_source_detail_transport=True,
        metrics=None,
        temporal_weight=0.20,
    ):
        patched, metrics = super().patch(
            model=model,
            trajectory=trajectory,
            source_mode=source_mode,
            source_scale=source_scale,
            source_width=source_width,
            source_height=source_height,
            handoff_coordinate=handoff_coordinate,
            handoff_selection=handoff_selection,
            guidance_mode=guidance_mode,
            direction_weight=direction_weight,
            acceleration_weight=acceleration_weight,
            consistency_weight=consistency_weight,
            low_frequency_cutoff=low_frequency_cutoff,
            learned_upscaler=learned_upscaler,
            metrics=metrics,
            temporal_weight=temporal_weight,
            frame_gauge_repair=frame_gauge_repair,
            frame_gauge_residual_mode=frame_gauge_residual_mode,
        )
        witness_directory = ""
        if capture_boundary_witness:
            try:
                import folder_paths
            except ImportError as exc:
                raise RuntimeError("boundary witness capture requires ComfyUI folder_paths at node execution") from exc
            witness_subdirectory = (
                "/h3_flow_regenerate/residual_geometry"
                if spatial_stage_control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND
                else "/h3-flow-boundary-witness"
            )
            witness_directory = str(folder_paths.get_output_directory() + witness_subdirectory)
        # Store an explicit per-model value even when disabled so stale process
        # environment cannot silently override the node on subsequent runs.
        patched.model_options[WITNESS_DIRECTORY_OPTION] = witness_directory
        metrics.event(
            "partitioned_boundary_witness_control",
            enabled=bool(capture_boundary_witness),
            directory=witness_directory or None,
            source="node",
            restart_required=False,
            output_mutated=False,
        )
        return apply_partitioned_diagnostic_controls(
            patched,
            metrics,
            vdn_linear_diagnostic=vdn_linear_diagnostic,
            audio_guided_overlap_ticks=audio_guided_overlap_ticks,
            vdn_temporal_carrier_policy=vdn_temporal_carrier_policy,
            audio_guided_overlap_mode=audio_guided_overlap_mode,
            prefix_transformer_context=prefix_transformer_context,
            audio_position_domain=audio_position_domain,
            audio_handoff_source=audio_handoff_source,
            av_handoff_source=av_handoff_source,
            guidance_trajectory_source=guidance_trajectory_source,
            low_probe_execution_source=low_probe_execution_source,
            provider_boundary_stabilization=provider_boundary_stabilization,
            handoff_transfer_control=handoff_transfer_control,
            spatial_stage_control=spatial_stage_control,
            softmax_diagnostic=softmax_diagnostic,
            video_guided_overlap_tokens=video_guided_overlap_tokens,
            suffix_dc_bridge=suffix_dc_bridge,
            target_band_tokens=target_band_tokens,
            target_band_handoff_state=target_band_handoff_state,
            target_band_context=target_band_context,
            uniform_source_detail_transport=uniform_source_detail_transport,
        )


NODE_CLASS_MAPPINGS = {
    "H3PartitionedExactPrefixHandoff": H3PartitionedExactPrefixHandoff,
    "H3PartitionedExactPrefixDiagnosticHandoff": H3PartitionedExactPrefixDiagnosticHandoff,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PartitionedExactPrefixHandoff": "MiniMax H3 Partitioned Exact-Prefix Handoff [Compatibility]",
    "H3PartitionedExactPrefixDiagnosticHandoff": "MiniMax H3 Partitioned Exact-Prefix Handoff",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
