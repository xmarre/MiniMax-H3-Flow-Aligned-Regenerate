"""Bounded node-selectable diagnostics for partitioned exact-prefix continuation.

These controls are intentionally model-local. They do not change the released
Progressive Target Input node or the ordinary partitioned node defaults.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .audio_guided_overlap import (
    configured_audio_guided_overlap_ticks,
    validate_audio_guided_overlap_ticks,
)
from .video_guided_overlap import validate_video_guided_overlap_tokens

PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY = "h3_flow_partitioned_vdn_linear_diagnostic_v1"
PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL = "normal"
PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS = "bypass_partitioned_linear"
PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL = "suppress_cross_grid_temporal_taps"
PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE = "raw_token_measure"
PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS = (
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
)

PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY = "h3_flow_partitioned_audio_guided_overlap_ticks_v1"
PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY = "h3_flow_partitioned_video_guided_overlap_tokens_v1"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY = "h3_flow_partitioned_audio_guided_overlap_mode_v1"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER = "sampler_mask"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP = "model_timestep_only"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP = "sampler_mask_exact_timestep"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_EXACT = "exact_mask"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS = (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_EXACT,
)
PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY = "h3_flow_partitioned_audio_model_timestep_context_v1"

PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY = "h3_flow_partitioned_prefix_transformer_context_v1"
PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT = "exact_target_partitioned"
PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE = "source_carrier_uniform"
PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS = (
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
)

PARTITIONED_AUDIO_POSITION_DOMAIN_KEY = "h3_flow_partitioned_audio_position_domain_v1"
PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY = "legacy_target"
PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE = "source_carrier"
PARTITIONED_AUDIO_POSITION_DOMAIN_OPTIONS = (
    PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
)

PARTITIONED_AUDIO_HANDOFF_SOURCE_KEY = "h3_flow_partitioned_audio_handoff_source_v1"
PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN = "main_partitioned"
PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW = "source_carrier_uniform_shadow"
PARTITIONED_AUDIO_HANDOFF_SOURCE_OPTIONS = (
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW,
)

PARTITIONED_AV_HANDOFF_SOURCE_KEY = "h3_flow_partitioned_av_handoff_source_v1"
PARTITIONED_AV_HANDOFF_SOURCE_MAIN = "main_partitioned"
PARTITIONED_AV_HANDOFF_SOURCE_SHADOW = "source_carrier_uniform_shadow"
PARTITIONED_AV_HANDOFF_SOURCE_OPTIONS = (
    PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
)

PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY = "h3_flow_partitioned_guidance_trajectory_source_v1"
PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN = "main_exact_partitioned"
PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW = "source_carrier_uniform_shadow"
PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_OPTIONS = (
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
)

PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY = "h3_flow_partitioned_low_probe_execution_source_v1"
PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW = "main_then_shadow"
PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY = "source_carrier_uniform_only"
PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS = (
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
)

PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_KEY = "h3_flow_partitioned_provider_boundary_stabilization_v1"
PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF = "off"
PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT = "soft_support_v1"
PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OPTIONS = (
    PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF,
    PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT,
)

PARTITIONED_HANDOFF_TRANSFER_CONTROL_KEY = "h3_flow_partitioned_handoff_transfer_control_v1"
PARTITIONED_HANDOFF_TRANSFER_LEARNED = "learned_3d"
PARTITIONED_HANDOFF_TRANSFER_BICUBIC_CONTROL = "bicubic_same_source_control"
PARTITIONED_HANDOFF_TRANSFER_OPTIONS = (
    PARTITIONED_HANDOFF_TRANSFER_LEARNED,
    PARTITIONED_HANDOFF_TRANSFER_BICUBIC_CONTROL,
)

PARTITIONED_SPATIAL_STAGE_CONTROL_KEY = "h3_flow_partitioned_spatial_stage_control_v1"
PARTITIONED_SPATIAL_STAGE_PROGRESSIVE = "progressive_low_to_high"
PARTITIONED_SPATIAL_STAGE_SAME_GRID = "same_grid_target_control"
PARTITIONED_SPATIAL_STAGE_TARGET_BAND = "progressive_target_band"
PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS = (
    PARTITIONED_SPATIAL_STAGE_PROGRESSIVE,
    PARTITIONED_SPATIAL_STAGE_SAME_GRID,
    PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
)

# Number of generated H3 temporal latent tokens that stay on the target grid
# directly after the protected prefix under progressive_target_band.
PARTITIONED_TARGET_BAND_TOKENS_KEY = "h3_flow_partitioned_target_band_tokens_v1"
PARTITIONED_TARGET_BAND_TOKENS_DEFAULT = 4

# Absence is the historical contract: the one-token suffix DC bridge is active.
# The leaf is published only when a node explicitly disables the bridge.
PARTITIONED_SUFFIX_DC_BRIDGE_KEY = "h3_flow_partitioned_suffix_dc_bridge_v1"

PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY = "h3_flow_partitioned_softmax_diagnostic_v1"
PARTITIONED_SOFTMAX_DIAGNOSTIC_API = 1
PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL = "normal"
PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX = "dense_suffix_same_domain"
PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK = "target_query_sink_measure"
PARTITIONED_SOFTMAX_DIAGNOSTIC_OPTIONS = (
    PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK,
)

VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API = 1

PARTITIONED_VDN_TEMPORAL_CARRIER_KEY = "h3_flow_partitioned_vdn_temporal_carrier_v1"
PARTITIONED_VDN_TEMPORAL_CARRIER_API = 1
PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE = "native_grid_then_map_v1"
PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION = "destination_grid_stencil_v1"
PARTITIONED_VDN_TEMPORAL_CARRIER_OPTIONS = (
    PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
    PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,
)
PARTITIONED_VDN_TEMPORAL_CARRIER_MAPPING_POLICY = "h3_physical_bilinear_border_fp32_restore_dtype_v1"


@dataclass(slots=True)
class PartitionedAudioModelTimestepContext:
    """Call-scoped audio mask consumed only inside MiniMax-H3's inner forward."""

    audio_mask: Any
    metrics: Any
    ticks: int
    audio_prefix_ticks: int
    mask_kind: str = "guided_overlap"
    calls: int = 0
    verification_calls: int = 0

    def record_verification(self) -> None:
        self.calls += 1
        self.verification_calls += 1
        increment = getattr(self.metrics, "increment", None)
        if callable(increment):
            increment("coherent_exact_audio_model_mask_calls")

    def record_call(self) -> None:
        self.calls += 1
        increment = getattr(self.metrics, "increment", None)
        if callable(increment):
            increment("partitioned_audio_model_timestep_override_calls")


def normalize_spatial_stage_control(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS:
        raise ValueError(
            "partitioned spatial-stage control must be one of "
            f"{PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS!r}, got {value!r}"
        )
    return value


def validate_target_band_tokens(value: Any, *, source: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{source} must be a positive integer number of H3 temporal latent tokens, got {value!r}")
    return value


def resolve_partitioned_target_band_tokens(transformer_options: dict[str, Any]) -> int:
    return validate_target_band_tokens(
        transformer_options.get(PARTITIONED_TARGET_BAND_TOKENS_KEY, PARTITIONED_TARGET_BAND_TOKENS_DEFAULT),
        source=PARTITIONED_TARGET_BAND_TOKENS_KEY,
    )


def resolve_partitioned_suffix_dc_bridge(transformer_options: dict[str, Any]) -> bool:
    """Return whether the partitioned one-token suffix DC bridge is enabled."""
    if PARTITIONED_SUFFIX_DC_BRIDGE_KEY not in transformer_options:
        return True
    value = transformer_options[PARTITIONED_SUFFIX_DC_BRIDGE_KEY]
    if value is not False:
        raise ValueError("the partitioned suffix DC bridge leaf may only carry the explicit value False")
    return False


def normalize_partitioned_softmax_diagnostic(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_SOFTMAX_DIAGNOSTIC_OPTIONS:
        raise ValueError(
            f"partitioned softmax diagnostic must be one of {PARTITIONED_SOFTMAX_DIAGNOSTIC_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_handoff_transfer_control(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_HANDOFF_TRANSFER_OPTIONS:
        raise ValueError(
            "partitioned handoff transfer control must be one of "
            f"{PARTITIONED_HANDOFF_TRANSFER_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_vdn_linear_diagnostic(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS:
        raise ValueError(
            f"VDN linear diagnostic must be one of {PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_vdn_temporal_carrier_policy(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_VDN_TEMPORAL_CARRIER_OPTIONS:
        raise ValueError(
            f"VDN temporal-carrier policy must be one of {PARTITIONED_VDN_TEMPORAL_CARRIER_OPTIONS!r}, got {value!r}"
        )
    return value


def build_vdn_temporal_carrier_contract(
    *,
    policy: str,
    flow_semantic_digest: str,
    diagnostic_mode: str,
    short_conv_spec: str,
) -> dict[str, Any]:
    """Build the exact numerical-policy leaf consumed by paired VDN."""
    policy = normalize_vdn_temporal_carrier_policy(policy)
    if not isinstance(flow_semantic_digest, str) or len(flow_semantic_digest) != 64:
        raise ValueError("VDN temporal-carrier policy requires a Flow semantic digest")
    diagnostic_mode = normalize_vdn_linear_diagnostic(diagnostic_mode)
    if not isinstance(short_conv_spec, str) or not short_conv_spec:
        raise ValueError("VDN temporal-carrier policy requires a checkpoint short-conv specification")
    payload = {
        "api": PARTITIONED_VDN_TEMPORAL_CARRIER_API,
        "policy": policy,
        "flow_semantic_digest": flow_semantic_digest,
        "diagnostic_mode": diagnostic_mode,
        "short_conv_spec": short_conv_spec,
        "precision_mapping_policy": PARTITIONED_VDN_TEMPORAL_CARRIER_MAPPING_POLICY,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["numerical_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def normalize_audio_guided_overlap_mode(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS:
        raise ValueError(
            f"audio guided-overlap mode must be one of {PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_prefix_transformer_context(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS:
        raise ValueError(
            "prefix transformer context must be one of "
            f"{PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_audio_position_domain(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_AUDIO_POSITION_DOMAIN_OPTIONS:
        raise ValueError(
            f"audio position domain must be one of {PARTITIONED_AUDIO_POSITION_DOMAIN_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_audio_handoff_source(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_AUDIO_HANDOFF_SOURCE_OPTIONS:
        raise ValueError(
            f"audio handoff source must be one of {PARTITIONED_AUDIO_HANDOFF_SOURCE_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_av_handoff_source(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_AV_HANDOFF_SOURCE_OPTIONS:
        raise ValueError(f"AV handoff source must be one of {PARTITIONED_AV_HANDOFF_SOURCE_OPTIONS!r}, got {value!r}")
    return value


def normalize_guidance_trajectory_source(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_OPTIONS:
        raise ValueError(
            "guidance trajectory source must be one of "
            f"{PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_low_probe_execution_source(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS:
        raise ValueError(
            "low/probe execution source must be one of "
            f"{PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS!r}, got {value!r}"
        )
    return value


def normalize_provider_boundary_stabilization(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OPTIONS:
        raise ValueError(
            "provider-boundary stabilization must be one of "
            f"{PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OPTIONS!r}, got {value!r}"
        )
    return value


def resolve_partitioned_audio_guided_overlap_mode(model_options: dict[str, Any]) -> tuple[str, str]:
    if PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY not in model_options:
        return PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER, "default_sampler_mask"
    return (
        normalize_audio_guided_overlap_mode(model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY]),
        "diagnostic_node",
    )


def resolve_partitioned_audio_guided_overlap_ticks(model_options: dict[str, Any]) -> tuple[int, str]:
    if PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY not in model_options:
        return configured_audio_guided_overlap_ticks(), "environment_or_default"
    return (
        validate_audio_guided_overlap_ticks(
            model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY],
            source=PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
        ),
        "diagnostic_node",
    )


def resolve_partitioned_video_guided_overlap_tokens(model_options: dict[str, Any]) -> tuple[int, str]:
    if PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY not in model_options:
        return 0, "default_off"
    return (
        validate_video_guided_overlap_tokens(
            model_options[PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY],
            source=PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY,
        ),
        "diagnostic_node",
    )


def apply_partitioned_diagnostic_controls(
    model,
    metrics,
    *,
    vdn_linear_diagnostic: str,
    audio_guided_overlap_ticks: int,
    vdn_temporal_carrier_policy: str = PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
    audio_guided_overlap_mode: str = PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    prefix_transformer_context: str = PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    audio_position_domain: str = PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    audio_handoff_source: str = PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    av_handoff_source: str = PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    guidance_trajectory_source: str = PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    low_probe_execution_source: str = PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
    provider_boundary_stabilization: str = PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF,
    handoff_transfer_control: str = PARTITIONED_HANDOFF_TRANSFER_LEARNED,
    spatial_stage_control: str = PARTITIONED_SPATIAL_STAGE_PROGRESSIVE,
    softmax_diagnostic: str = PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    video_guided_overlap_tokens: int = 0,
    suffix_dc_bridge: bool = True,
    target_band_tokens: int = PARTITIONED_TARGET_BAND_TOKENS_DEFAULT,
):
    """Install diagnostic controls on one cloned MODEL only."""

    mode = normalize_vdn_linear_diagnostic(vdn_linear_diagnostic)
    temporal_carrier_policy = normalize_vdn_temporal_carrier_policy(vdn_temporal_carrier_policy)
    if (
        temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION
        and mode != PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL
    ):
        raise ValueError("destination-grid temporal stencil requires vdn_linear_diagnostic='normal'")
    ticks = validate_audio_guided_overlap_ticks(
        audio_guided_overlap_ticks,
        source="audio_guided_overlap_ticks",
    )
    audio_mode = normalize_audio_guided_overlap_mode(audio_guided_overlap_mode)
    prefix_context = normalize_prefix_transformer_context(prefix_transformer_context)
    position_domain = normalize_audio_position_domain(audio_position_domain)
    handoff_source = normalize_audio_handoff_source(audio_handoff_source)
    av_handoff = normalize_av_handoff_source(av_handoff_source)
    guidance_source = normalize_guidance_trajectory_source(guidance_trajectory_source)
    execution_source = normalize_low_probe_execution_source(low_probe_execution_source)
    boundary_stabilization = normalize_provider_boundary_stabilization(provider_boundary_stabilization)
    handoff_transfer = normalize_handoff_transfer_control(handoff_transfer_control)
    spatial_stage = normalize_spatial_stage_control(spatial_stage_control)
    softmax_mode = normalize_partitioned_softmax_diagnostic(softmax_diagnostic)
    if (
        softmax_mode == PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK
        and mode == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE
    ):
        raise ValueError("target-query sink measure cannot be combined with raw_token_measure")
    video_overlap_tokens = validate_video_guided_overlap_tokens(
        video_guided_overlap_tokens,
        source="video_guided_overlap_tokens",
    )
    if type(suffix_dc_bridge) is not bool:
        raise TypeError("suffix_dc_bridge must be a boolean")
    band_tokens = validate_target_band_tokens(target_band_tokens, source="target_band_tokens")
    if (
        spatial_stage == PARTITIONED_SPATIAL_STAGE_SAME_GRID
        and handoff_transfer != PARTITIONED_HANDOFF_TRANSFER_LEARNED
    ):
        raise ValueError("same-grid spatial-stage control requires handoff_transfer_control='learned_3d'")
    if spatial_stage == PARTITIONED_SPATIAL_STAGE_TARGET_BAND:
        if handoff_transfer != PARTITIONED_HANDOFF_TRANSFER_LEARNED:
            raise ValueError("progressive_target_band requires handoff_transfer_control='learned_3d'")
        if temporal_carrier_policy != PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE:
            raise ValueError("progressive_target_band requires vdn_temporal_carrier_policy='native_grid_then_map_v1'")
        if prefix_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
            raise ValueError("progressive_target_band requires prefix_transformer_context='exact_target_partitioned'")
        if execution_source != PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW:
            raise ValueError("progressive_target_band requires low_probe_execution_source='main_then_shadow'")
        if (
            handoff_source != PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN
            or av_handoff != PARTITIONED_AV_HANDOFF_SOURCE_MAIN
            or guidance_source != PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN
        ):
            raise ValueError("progressive_target_band supports only the main partitioned handoff and guidance sources")
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("partitioned diagnostics require mutable model_options")

    transformer_options = dict(model_options.get("transformer_options") or {})
    transformer_options[PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY] = mode
    if temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE:
        transformer_options.pop(PARTITIONED_VDN_TEMPORAL_CARRIER_KEY, None)
    else:
        # The fully bound numerical-policy leaf is constructed only after the
        # physical Flow plan has a semantic digest inside the transformer.
        transformer_options[PARTITIONED_VDN_TEMPORAL_CARRIER_KEY] = temporal_carrier_policy
    transformer_options[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY] = prefix_context
    if position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY:
        transformer_options.pop(PARTITIONED_AUDIO_POSITION_DOMAIN_KEY, None)
    else:
        transformer_options[PARTITIONED_AUDIO_POSITION_DOMAIN_KEY] = position_domain
    if handoff_source == PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN:
        transformer_options.pop(PARTITIONED_AUDIO_HANDOFF_SOURCE_KEY, None)
    else:
        transformer_options[PARTITIONED_AUDIO_HANDOFF_SOURCE_KEY] = handoff_source
    if av_handoff == PARTITIONED_AV_HANDOFF_SOURCE_MAIN:
        transformer_options.pop(PARTITIONED_AV_HANDOFF_SOURCE_KEY, None)
    else:
        transformer_options[PARTITIONED_AV_HANDOFF_SOURCE_KEY] = av_handoff
    if guidance_source == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN:
        transformer_options.pop(PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY, None)
    else:
        transformer_options[PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY] = guidance_source
    if execution_source == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW:
        transformer_options.pop(PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY, None)
    else:
        transformer_options[PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY] = execution_source
    if boundary_stabilization == PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF:
        transformer_options.pop(PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_KEY, None)
    else:
        transformer_options[PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_KEY] = boundary_stabilization
    if handoff_transfer == PARTITIONED_HANDOFF_TRANSFER_LEARNED:
        # Absence is the historical/default contract. Only the diagnostic
        # transfer control publishes a leaf, so old workflows remain identical.
        transformer_options.pop(PARTITIONED_HANDOFF_TRANSFER_CONTROL_KEY, None)
    else:
        transformer_options[PARTITIONED_HANDOFF_TRANSFER_CONTROL_KEY] = handoff_transfer
    if spatial_stage == PARTITIONED_SPATIAL_STAGE_PROGRESSIVE:
        transformer_options.pop(PARTITIONED_SPATIAL_STAGE_CONTROL_KEY, None)
    else:
        transformer_options[PARTITIONED_SPATIAL_STAGE_CONTROL_KEY] = spatial_stage
    if softmax_mode == PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
        transformer_options.pop(PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY, None)
    else:
        transformer_options[PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY] = softmax_mode
    if suffix_dc_bridge:
        transformer_options.pop(PARTITIONED_SUFFIX_DC_BRIDGE_KEY, None)
    else:
        transformer_options[PARTITIONED_SUFFIX_DC_BRIDGE_KEY] = False
    if spatial_stage == PARTITIONED_SPATIAL_STAGE_TARGET_BAND:
        transformer_options[PARTITIONED_TARGET_BAND_TOKENS_KEY] = band_tokens
    else:
        transformer_options.pop(PARTITIONED_TARGET_BAND_TOKENS_KEY, None)
    model_options["transformer_options"] = transformer_options
    model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY] = ticks
    model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY] = audio_mode
    if video_overlap_tokens:
        model_options[PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY] = video_overlap_tokens
    else:
        model_options.pop(PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY, None)

    event = getattr(metrics, "event", None)
    if callable(event):
        fields = {
            "vdn_linear_diagnostic": mode,
            "audio_guided_overlap_ticks": ticks,
            "audio_guided_overlap_mode": audio_mode,
            "prefix_transformer_context": prefix_context,
            "model_local": True,
            "native_vdn_unchanged": True,
            "production_default_changed": False,
        }
        if video_overlap_tokens:
            fields["video_guided_overlap_tokens"] = video_overlap_tokens
        if temporal_carrier_policy != PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE:
            fields["vdn_temporal_carrier_policy"] = temporal_carrier_policy
        if position_domain != PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY:
            fields["audio_position_domain"] = position_domain
        if handoff_source != PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN:
            fields["audio_handoff_source"] = handoff_source
        if av_handoff != PARTITIONED_AV_HANDOFF_SOURCE_MAIN:
            fields["av_handoff_source"] = av_handoff
        if guidance_source != PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN:
            fields["guidance_trajectory_source"] = guidance_source
        if execution_source != PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW:
            fields["low_probe_execution_source"] = execution_source
        if boundary_stabilization != PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF:
            fields["provider_boundary_stabilization"] = boundary_stabilization
        if handoff_transfer != PARTITIONED_HANDOFF_TRANSFER_LEARNED:
            fields["handoff_transfer_control"] = handoff_transfer
        if spatial_stage != PARTITIONED_SPATIAL_STAGE_PROGRESSIVE:
            fields["spatial_stage_control"] = spatial_stage
        if softmax_mode != PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
            fields["softmax_diagnostic"] = softmax_mode
        if not suffix_dc_bridge:
            fields["suffix_dc_bridge"] = False
        if spatial_stage == PARTITIONED_SPATIAL_STAGE_TARGET_BAND:
            fields["target_band_tokens"] = band_tokens
        event("partitioned_diagnostic_controls", **fields)
    return model, metrics


__all__ = [
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_EXACT",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY",
    "PARTITIONED_AUDIO_HANDOFF_SOURCE_KEY",
    "PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN",
    "PARTITIONED_AUDIO_HANDOFF_SOURCE_OPTIONS",
    "PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW",
    "PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY",
    "PARTITIONED_AUDIO_POSITION_DOMAIN_KEY",
    "PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY",
    "PARTITIONED_AUDIO_POSITION_DOMAIN_OPTIONS",
    "PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE",
    "PARTITIONED_AV_HANDOFF_SOURCE_KEY",
    "PARTITIONED_AV_HANDOFF_SOURCE_MAIN",
    "PARTITIONED_AV_HANDOFF_SOURCE_OPTIONS",
    "PARTITIONED_AV_HANDOFF_SOURCE_SHADOW",
    "PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY",
    "PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN",
    "PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_OPTIONS",
    "PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW",
    "PARTITIONED_HANDOFF_TRANSFER_BICUBIC_CONTROL",
    "PARTITIONED_HANDOFF_TRANSFER_CONTROL_KEY",
    "PARTITIONED_HANDOFF_TRANSFER_LEARNED",
    "PARTITIONED_HANDOFF_TRANSFER_OPTIONS",
    "PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY",
    "PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW",
    "PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS",
    "PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY",
    "PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT",
    "PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY",
    "PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS",
    "PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE",
    "PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_KEY",
    "PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OFF",
    "PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_OPTIONS",
    "PARTITIONED_PROVIDER_BOUNDARY_STABILIZATION_SOFT",
    "PARTITIONED_SOFTMAX_DIAGNOSTIC_API",
    "PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX",
    "PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY",
    "PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL",
    "PARTITIONED_SOFTMAX_DIAGNOSTIC_OPTIONS",
    "PARTITIONED_SOFTMAX_DIAGNOSTIC_TARGET_SINK",
    "PARTITIONED_SPATIAL_STAGE_CONTROL_KEY",
    "PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS",
    "PARTITIONED_SPATIAL_STAGE_PROGRESSIVE",
    "PARTITIONED_SPATIAL_STAGE_SAME_GRID",
    "PARTITIONED_SPATIAL_STAGE_TARGET_BAND",
    "PARTITIONED_SUFFIX_DC_BRIDGE_KEY",
    "PARTITIONED_TARGET_BAND_TOKENS_DEFAULT",
    "PARTITIONED_TARGET_BAND_TOKENS_KEY",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL",
    "PARTITIONED_VDN_TEMPORAL_CARRIER_API",
    "PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION",
    "PARTITIONED_VDN_TEMPORAL_CARRIER_KEY",
    "PARTITIONED_VDN_TEMPORAL_CARRIER_MAPPING_POLICY",
    "PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE",
    "PARTITIONED_VDN_TEMPORAL_CARRIER_OPTIONS",
    "PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API",
    "PartitionedAudioModelTimestepContext",
    "apply_partitioned_diagnostic_controls",
    "build_vdn_temporal_carrier_contract",
    "normalize_audio_guided_overlap_mode",
    "normalize_audio_handoff_source",
    "normalize_audio_position_domain",
    "normalize_av_handoff_source",
    "normalize_guidance_trajectory_source",
    "normalize_handoff_transfer_control",
    "normalize_low_probe_execution_source",
    "normalize_partitioned_softmax_diagnostic",
    "normalize_prefix_transformer_context",
    "normalize_provider_boundary_stabilization",
    "normalize_spatial_stage_control",
    "normalize_vdn_linear_diagnostic",
    "normalize_vdn_temporal_carrier_policy",
    "resolve_partitioned_audio_guided_overlap_mode",
    "resolve_partitioned_audio_guided_overlap_ticks",
    "resolve_partitioned_suffix_dc_bridge",
    "resolve_partitioned_target_band_tokens",
    "resolve_partitioned_video_guided_overlap_tokens",
    "validate_target_band_tokens",
]
