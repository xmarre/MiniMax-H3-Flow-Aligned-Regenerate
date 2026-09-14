"""Validation overlay for isolating the sole high-stage Spectrum forecast.

For the short learned-3D progressive schedule, request two exact high-stage
prefix calls through the existing ``h3_refinement`` contract. This turns the
high-stage A/F/A sequence into A/A/A without changing the low stage or probe.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Any

from . import runtime as _runtime
from .handoff import ProgressiveTargetInputConfig

_STATE_ATTR = "_h3_high_stage_forecast_isolation_state_v1"
_WRAPPER_MARK = "_h3_high_stage_forecast_isolation_v1"
_REQUESTED_PREFIX = 2


@dataclass(slots=True)
class _IsolationRecord:
    enabled: bool
    binding: Any
    event_start: int
    high_contract_entries: int = 0


class _IsolationState:
    def __init__(self) -> None:
        self.tls = threading.local()


def _state() -> _IsolationState:
    state = getattr(_runtime, _STATE_ATTR, None)
    if state is None:
        state = _IsolationState()
        setattr(_runtime, _STATE_ATTR, state)
    return state


def _active() -> _IsolationRecord | None:
    return getattr(_state().tls, "record", None)


def _eligible(config: Any, binding: Any) -> bool:
    guidance = getattr(binding, "guidance", None)
    return (
        isinstance(config, ProgressiveTargetInputConfig)
        and config.transfer_mode == "learned_3d"
        and config.exact_prefix_mode == "fallback"
        and guidance is not None
        and getattr(guidance, "mode", "off") != "off"
    )


@contextlib.contextmanager
def _high_stage_contract_isolation_wrapper(guider: Any):
    record = _active()
    options = getattr(guider, "model_options", None)
    transformer = options.get("transformer_options") if isinstance(options, dict) else None
    stage = transformer.get(_runtime.FLOW_STAGE_KEY) if isinstance(transformer, dict) else None

    if record is None or not record.enabled or stage != "high":
        with _ORIGINAL_HIGH_STAGE_CONTRACT(guider):
            yield
        return

    if not isinstance(options, dict):
        raise RuntimeError("high-stage forecast isolation requires mutable model options")
    if not isinstance(transformer, dict):
        raise RuntimeError("high-stage forecast isolation requires mutable transformer options")

    previous = transformer.get("h3_refinement")
    request = {
        "api": 1,
        "active": True,
        "min_actual_prefix_steps": _REQUESTED_PREFIX,
        "sigma_reference": 1.0,
        "source": "h3_flow_high_stage_forecast_isolation",
    }
    if previous is not None:
        if not isinstance(previous, dict):
            raise RuntimeError("existing h3_refinement contract is not a dictionary")
        conflicts = {
            key: (previous[key], value) for key, value in request.items() if key in previous and previous[key] != value
        }
        if conflicts:
            message = "existing h3_refinement contract conflicts with high-stage forecast isolation"
            raise RuntimeError(f"{message}: {conflicts}")
        request = {**previous, **request}

    record.high_contract_entries += 1
    record.binding.metrics.event(
        "high_stage_forecast_isolation",
        applied=True,
        stage="high",
        min_actual_prefix_steps=_REQUESTED_PREFIX,
        replaced_schedule_role="first_high_forecast",
        production_status="validation_only",
    )
    transformer["h3_refinement"] = request
    try:
        yield
    finally:
        if previous is None:
            transformer.pop("h3_refinement", None)
        else:
            transformer["h3_refinement"] = previous


setattr(_high_stage_contract_isolation_wrapper, _WRAPPER_MARK, True)


def _run_progressive_isolation_wrapper(
    executor,
    guider,
    binding,
    config,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask,
    callback,
    disable_pbar,
    seed,
    latent_shapes,
):
    state = _state()
    previous = getattr(state.tls, "record", None)
    event_start = len(binding.metrics.events)
    record = _IsolationRecord(_eligible(config, binding), binding, event_start)
    state.tls.record = record
    try:
        result = _ORIGINAL_RUN_PROGRESSIVE(
            executor,
            guider,
            binding,
            config,
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes,
        )
        if not record.enabled:
            return result

        high_calls = [
            event
            for event in binding.metrics.events[event_start:]
            if event.kind == "model_call" and event.fields.get("stage") == "high"
        ]
        if record.high_contract_entries != 1:
            raise RuntimeError(
                "high-stage forecast isolation did not observe exactly one high-stage refinement lifetime; "
                f"observed {record.high_contract_entries}"
            )
        if len(high_calls) < 2:
            raise RuntimeError(
                "high-stage forecast isolation requires at least two high-stage logical calls; "
                f"observed {len(high_calls)}"
            )

        first_actual = bool(high_calls[0].fields.get("actual"))
        second_actual = bool(high_calls[1].fields.get("actual"))
        if not first_actual or not second_actual:
            raise RuntimeError(
                "Spectrum did not honor the requested two-call exact high-stage prefix: "
                f"first_actual={first_actual} second_actual={second_actual}"
            )

        actual = sum(bool(event.fields.get("actual")) for event in high_calls)
        binding.metrics.event(
            "high_stage_forecast_isolation_result",
            applied=True,
            high_logical_calls=len(high_calls),
            high_actual_calls=actual,
            high_forecast_calls=len(high_calls) - actual,
            first_high_actual=first_actual,
            second_high_actual=second_actual,
            requested_actual_prefix_steps=_REQUESTED_PREFIX,
            production_status="validation_only",
        )
        return result
    finally:
        state.tls.record = previous


setattr(_run_progressive_isolation_wrapper, _WRAPPER_MARK, True)

_ORIGINAL_HIGH_STAGE_CONTRACT = _runtime._high_stage_contract
_ORIGINAL_RUN_PROGRESSIVE = _runtime._run_progressive


def install_high_stage_forecast_isolation() -> None:
    global _ORIGINAL_HIGH_STAGE_CONTRACT, _ORIGINAL_RUN_PROGRESSIVE

    if not getattr(_runtime._high_stage_contract, _WRAPPER_MARK, False):
        _ORIGINAL_HIGH_STAGE_CONTRACT = _runtime._high_stage_contract
        _runtime._high_stage_contract = _high_stage_contract_isolation_wrapper
    if not getattr(_runtime._run_progressive, _WRAPPER_MARK, False):
        _ORIGINAL_RUN_PROGRESSIVE = _runtime._run_progressive
        _runtime._run_progressive = _run_progressive_isolation_wrapper


__all__ = ["install_high_stage_forecast_isolation"]
