"""Bounded node-selectable diagnostics for partitioned exact-prefix continuation.

These controls are intentionally model-local. They do not change the released
Progressive Target Input node or the ordinary partitioned node defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audio_guided_overlap import (
    configured_audio_guided_overlap_ticks,
    validate_audio_guided_overlap_ticks,
)

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


def apply_partitioned_diagnostic_controls(
    model,
    metrics,
    *,
    vdn_linear_diagnostic: str,
    audio_guided_overlap_ticks: int,
    audio_guided_overlap_mode: str = PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    prefix_transformer_context: str = PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
):
    """Install diagnostic controls on one cloned MODEL only."""

    mode = normalize_vdn_linear_diagnostic(vdn_linear_diagnostic)
    ticks = validate_audio_guided_overlap_ticks(
        audio_guided_overlap_ticks,
        source="audio_guided_overlap_ticks",
    )
    audio_mode = normalize_audio_guided_overlap_mode(audio_guided_overlap_mode)
    prefix_context = normalize_prefix_transformer_context(prefix_transformer_context)
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("partitioned diagnostics require mutable model_options")

    transformer_options = dict(model_options.get("transformer_options") or {})
    transformer_options[PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY] = mode
    transformer_options[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY] = prefix_context
    model_options["transformer_options"] = transformer_options
    model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY] = ticks
    model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY] = audio_mode

    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_diagnostic_controls",
            vdn_linear_diagnostic=mode,
            audio_guided_overlap_ticks=ticks,
            audio_guided_overlap_mode=audio_mode,
            prefix_transformer_context=prefix_context,
            model_local=True,
            native_vdn_unchanged=True,
            production_default_changed=False,
        )
    return model, metrics


__all__ = [
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_OPTIONS",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER",
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY",
    "PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY",
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
    "normalize_prefix_transformer_context",
    "normalize_vdn_linear_diagnostic",
    "resolve_partitioned_audio_guided_overlap_mode",
    "resolve_partitioned_audio_guided_overlap_ticks",
]
