"""Bounded node-selectable diagnostics for partitioned exact-prefix continuation.

These controls are intentionally model-local. They do not change the released
Progressive Target Input node or the ordinary partitioned node defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio_guided_overlap import configured_audio_guided_overlap_ticks

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
MAX_PARTITIONED_DIAGNOSTIC_AUDIO_GUIDED_OVERLAP_TICKS = 32
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY = "h3_flow_partitioned_audio_guided_overlap_mode_v1"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER = "sampler_mask"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP = "model_timestep_only"
PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS = (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
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
VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API = 1


@dataclass(slots=True)
class PartitionedAudioModelTimestepContext:
    """Call-scoped audio timestep mask consumed only inside MiniMax-H3's inner forward."""

    audio_mask: Any
    metrics: Any
    ticks: int
    audio_prefix_ticks: int
    calls: int = 0

    def record_call(self) -> None:
        self.calls += 1
        increment = getattr(self.metrics, "increment", None)
        if callable(increment):
            increment("partitioned_audio_model_timestep_override_calls")


def normalize_vdn_linear_diagnostic(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS:
        raise ValueError(
            f"VDN linear diagnostic must be one of {PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS!r}, got {value!r}"
        )
    return value


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


def validate_partitioned_audio_guided_overlap_ticks(
    value: int,
    *,
    source: str = "partitioned audio guided overlap",
) -> int:
    """Validate the model-local diagnostic width without widening production env controls."""

    valid = type(value) is int and (0 <= value <= 16 or value == MAX_PARTITIONED_DIAGNOSTIC_AUDIO_GUIDED_OVERLAP_TICKS)
    if not valid:
        raise ValueError(f"{source} must be an integer in [0, 16] or 32, got {value!r}")
    return int(value)


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
        validate_partitioned_audio_guided_overlap_ticks(
            model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY],
            source=PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
        ),
        "diagnostic_node",
    )


def apply_partitioned_diagnostic_controls(
    model,
    metrics,
    *,
    vdn_linear_diagnostic: str,
    audio_guided_overlap_ticks: int,
    audio_guided_overlap_mode: str = PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    prefix_transformer_context: str = PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    audio_position_domain: str = PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    audio_handoff_source: str = PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    av_handoff_source: str = PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    guidance_trajectory_source: str = PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    low_probe_execution_source: str = PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
):
    """Install diagnostic controls on one cloned MODEL only."""

    mode = normalize_vdn_linear_diagnostic(vdn_linear_diagnostic)
    ticks = validate_partitioned_audio_guided_overlap_ticks(
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
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("partitioned diagnostics require mutable model_options")

    transformer_options = dict(model_options.get("transformer_options") or {})
    transformer_options[PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY] = mode
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
    model_options["transformer_options"] = transformer_options
    model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY] = ticks
    model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY] = audio_mode

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
        event("partitioned_diagnostic_controls", **fields)
    return model, metrics


__all__ = [
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER",
    "MAX_PARTITIONED_DIAGNOSTIC_AUDIO_GUIDED_OVERLAP_TICKS",
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
    "PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY",
    "PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW",
    "PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS",
    "PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY",
    "PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT",
    "PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY",
    "PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS",
    "PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL",
    "VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API",
    "PartitionedAudioModelTimestepContext",
    "apply_partitioned_diagnostic_controls",
    "normalize_audio_guided_overlap_mode",
    "normalize_audio_handoff_source",
    "normalize_audio_position_domain",
    "normalize_av_handoff_source",
    "normalize_guidance_trajectory_source",
    "normalize_low_probe_execution_source",
    "normalize_prefix_transformer_context",
    "normalize_vdn_linear_diagnostic",
    "resolve_partitioned_audio_guided_overlap_mode",
    "resolve_partitioned_audio_guided_overlap_ticks",
    "validate_partitioned_audio_guided_overlap_ticks",
]
