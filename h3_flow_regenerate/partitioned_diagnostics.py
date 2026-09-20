"""Bounded node-selectable diagnostics for partitioned exact-prefix continuation.

These controls are intentionally model-local. They do not change the released
Progressive Target Input node or the ordinary partitioned node defaults.
"""

from __future__ import annotations

from typing import Any

from .audio_guided_overlap import (
    configured_audio_guided_overlap_ticks,
    validate_audio_guided_overlap_ticks,
)

PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY = "h3_flow_partitioned_vdn_linear_diagnostic_v1"
PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL = "normal"
PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS = "bypass_partitioned_linear"
PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS = (
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
)

PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY = "h3_flow_partitioned_audio_guided_overlap_ticks_v1"


def normalize_vdn_linear_diagnostic(value: str) -> str:
    value = str(value)
    if value not in PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS:
        raise ValueError(
            f"VDN linear diagnostic must be one of {PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS!r}, got {value!r}"
        )
    return value


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
):
    """Install diagnostic controls on one cloned MODEL only."""

    mode = normalize_vdn_linear_diagnostic(vdn_linear_diagnostic)
    ticks = validate_audio_guided_overlap_ticks(
        audio_guided_overlap_ticks,
        source="audio_guided_overlap_ticks",
    )
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("partitioned diagnostics require mutable model_options")

    transformer_options = dict(model_options.get("transformer_options") or {})
    transformer_options[PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY] = mode
    model_options["transformer_options"] = transformer_options
    model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY] = ticks

    event = getattr(metrics, "event", None)
    if callable(event):
        event(
            "partitioned_diagnostic_controls",
            vdn_linear_diagnostic=mode,
            audio_guided_overlap_ticks=ticks,
            model_local=True,
            native_vdn_unchanged=True,
            production_default_changed=False,
        )
    return model, metrics


__all__ = [
    "PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL",
    "PARTITIONED_VDN_LINEAR_DIAGNOSTIC_OPTIONS",
    "apply_partitioned_diagnostic_controls",
    "normalize_vdn_linear_diagnostic",
    "resolve_partitioned_audio_guided_overlap_ticks",
]
