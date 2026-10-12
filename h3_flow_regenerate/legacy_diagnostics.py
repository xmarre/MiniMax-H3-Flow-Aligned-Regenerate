"""Compatibility for serialized controls from the retired live replay diagnostics."""

from __future__ import annotations

RETIRED_LOW_OPTIONS = (
    "h3_flow_low_sampler_aba_v1",
    "h3_flow_low_native_trace_step_v1",
    "h3_flow_low_native_full_pair_block_v1",
    "h3_flow_low_native_module_detail_v1",
    "h3_flow_low_native_projection_detail_v1",
    "h3_flow_low_native_chunked_qkv_detail_v1",
    "h3_flow_low_native_token_admission_v1",
)


def retire_low_replay_controls(model, metrics, **controls):
    """Clear inherited live-observer options and report requested legacy controls.

    Saved widget slots remain loadable. The retired values cannot schedule
    shadow samplers, install hooks, retain observer state, or change inference.
    """
    requested = []
    for name, value in controls.items():
        if name in ("low_native_trace_step", "low_native_trace_pair_block"):
            if type(value) is not int or not -1 <= value <= 511:
                raise ValueError(f"{name} must be an integer from -1 to 511")
            enabled = value >= 0
        else:
            if type(value) is not bool:
                raise TypeError(f"{name} must be boolean")
            enabled = value
        if enabled:
            requested.append(name)
    inherited = [key for key in RETIRED_LOW_OPTIONS if key in model.model_options]
    for key in RETIRED_LOW_OPTIONS:
        model.model_options.pop(key, None)
    if requested or inherited:
        metrics.event(
            "retired_low_replay_controls",
            requested=requested,
            cleared_inherited_options=inherited,
            live_replay_executed=False,
            kernel_observers_installed=False,
            extra_h3_nfe=0,
        )
