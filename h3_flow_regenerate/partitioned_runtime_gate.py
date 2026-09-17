"""Post-run validation for partitioned exact-prefix production evidence.

The validator is deliberately offline. It consumes the ordinary Flow metrics JSON
and the ordinary ComfyUI log so the hardware acceptance run does not acquire a
second diagnostic execution path or mutate sampler/backend state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

PARTITIONED_SOL_ABI = "sol-h3-partitioned-single-union-v1"
VDN_LINEAR_ACTIVE_MARKER = (
    "partitioned exact-prefix: grouped VDN softmax active; "
    "variable-grid linear complement active"
)
AUDIO_OVERLAP_MARKER = "partitioned audio guided overlap active"
_SOL_PREFIX = "Sol-H3 "


class RuntimeGateError(RuntimeError):
    """Raised when supplied production evidence violates an acceptance invariant."""


@dataclass(frozen=True, slots=True)
class RuntimeGateReport:
    logical_calls: int
    actual_calls: int
    forecast_calls: int
    low_logical: int
    low_actual: int
    probe_logical: int
    probe_actual: int
    high_logical: int
    high_actual: int
    partitioned_transformer_events: int
    sol_partitioned_requests: int
    sol_mapped_calls: int
    sol_rectangular_calls: int
    sol_requested_q_rows: int
    sol_kernel_q_rows: int
    vdn_variable_grid_linear_active: bool
    audio_guided_overlap_active: bool

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeGateError(message)


def _event_kind(event: Any) -> str:
    return str(event.get("kind", "")) if isinstance(event, dict) else ""


def _event_fields(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    fields = event.get("fields", {})
    return fields if isinstance(fields, dict) else {}


def _latest_partitioned_window(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts = [
        index
        for index, event in enumerate(events)
        if _event_kind(event) == "partitioned_stage_plan"
    ]
    if not starts:
        raise RuntimeGateError("Flow metrics contain no partitioned_stage_plan event")
    start = starts[-1]
    stop = len(events)
    for index in range(start + 1, len(events)):
        if _event_kind(events[index]) == "partitioned_stage_plan":
            stop = index
            break
    return events[start:stop]


def _parse_sol_records(log_text: str) -> list[dict[str, Any]]:
    records = []
    for raw_line in log_text.splitlines():
        marker = raw_line.find(_SOL_PREFIX)
        if marker < 0:
            continue
        payload = raw_line[marker + len(_SOL_PREFIX) :].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _partitioned_sol_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        gates = record.get("arithmetic_gates", [])
        if not isinstance(gates, list):
            continue
        if any(
            isinstance(gate, dict) and gate.get("route") == PARTITIONED_SOL_ABI
            for gate in gates
        ):
            selected.append(record)
    return selected


def _stage_counts(model_calls: list[dict[str, Any]], stage: str) -> tuple[int, int]:
    calls = [
        event
        for event in model_calls
        if str(_event_fields(event).get("stage")) == stage
    ]
    actual = sum(bool(_event_fields(event).get("actual")) for event in calls)
    return len(calls), actual


def validate_partitioned_runtime_evidence(
    metrics: dict[str, Any],
    log_text: str,
    *,
    expected_vdn_api: int = 4,
    expected_logical: int | None = None,
    expected_actual: int | None = None,
    expected_forecast: int | None = None,
    require_spectrum: bool = True,
    require_audio_overlap: bool = True,
    require_vdn_linear: bool = True,
) -> RuntimeGateReport:
    """Validate one latest partitioned chunk plus its matching process log evidence."""
    _require(isinstance(metrics, dict), "Flow metrics root must be an object")
    events = metrics.get("events")
    counters = metrics.get("counters")
    _require(isinstance(events, list), "Flow metrics events must be a list")
    _require(isinstance(counters, dict), "Flow metrics counters must be an object")

    window = _latest_partitioned_window(events)
    kinds = [_event_kind(event) for event in window]
    _require(
        "partitioned_exact_prefix_fallback" not in kinds,
        "partitioned run fell back to target-grid execution",
    )

    plan = next(
        event for event in window if _event_kind(event) == "partitioned_stage_plan"
    )
    plan_fields = _event_fields(plan)
    _require(
        plan_fields.get("input_mode") == "partitioned_exact_prefix",
        "partitioned stage input mode drifted",
    )
    _require(
        plan_fields.get("prefix_exact_latent_resized_for_transformer") is False,
        "exact prefix was resized for H3",
    )
    _require(
        plan_fields.get("deprecated_mixed_grid_contract_active") is False,
        "deprecated Mixed-Grid contract became active",
    )

    transformer_events = [
        event
        for event in window
        if _event_kind(event) == "partitioned_exact_prefix_transformer"
    ]
    _require(
        bool(transformer_events),
        "partitioned transformer emitted no physical-domain evidence",
    )
    for event in transformer_events:
        fields = _event_fields(event)
        _require(
            fields.get("vdn_external_sequence_api") == expected_vdn_api,
            "partitioned transformer used the wrong VDN API",
        )
        _require(
            fields.get("prefix_exact_latent_resized") is False,
            "partitioned transformer resized the exact prefix",
        )
        _require(
            fields.get("prefix_target_grid_rope") is True,
            "target-prefix RoPE contract was not active",
        )
        _require(
            fields.get("suffix_source_grid_rope") is True,
            "source-suffix RoPE contract was not active",
        )
        _require(
            fields.get("low_suffix_real_latent") is True,
            "low suffix was not represented by the real source-grid latent",
        )
        _require(
            fields.get("deprecated_mixed_grid_contract_active") is False,
            "deprecated Mixed-Grid transformer state became active",
        )

    transfers = [
        event for event in window if _event_kind(event) == "partitioned_transfer"
    ]
    _require(bool(transfers), "partitioned handoff emitted no transfer evidence")
    transfer = _event_fields(transfers[-1])
    _require(
        transfer.get("learned_transfer_performed") is True,
        "learned_3d transfer did not execute",
    )
    _require(
        transfer.get("upscaler_prefix_output_discarded") is True,
        "upscaled prefix output was not discarded",
    )
    _require(
        transfer.get("authoritative_target_prefix_restored") is True,
        "authoritative target prefix was not restored",
    )
    _require(
        transfer.get("target_prefix_resized_for_transformer") is False,
        "target prefix was resized for transformer use",
    )
    _require(
        transfer.get("deprecated_mixed_grid_repairs_applied") is False,
        "deprecated Mixed-Grid seam repair executed",
    )

    completes = [
        event
        for event in window
        if _event_kind(event) == "partitioned_exact_prefix_complete"
    ]
    _require(bool(completes), "partitioned run emitted no completion proof")
    complete = _event_fields(completes[-1])
    _require(
        complete.get("final_prefix_exact") is True,
        "final protected prefix was not bitwise exact",
    )
    _require(
        complete.get("high_stage_first_call_actual") is True,
        "high stage did not begin with an actual H3 evaluation",
    )
    _require(
        complete.get("deprecated_mixed_grid_contract_active") is False,
        "deprecated Mixed-Grid contract survived completion",
    )

    handoffs = [
        event for event in window if _event_kind(event) == "handoff_complete"
    ]
    _require(bool(handoffs), "partitioned run emitted no handoff completion evidence")
    handoff = _event_fields(handoffs[-1])
    _require(
        handoff.get("input_mode") == "partitioned_exact_prefix",
        "handoff completion mode drifted",
    )
    _require(
        handoff.get("exact_probe_performed") is True,
        "exact handoff probe did not execute",
    )
    _require(
        handoff.get("sampler_invocation_count") == 3,
        "partitioned chunk did not use exactly low/probe/high sampler lifetimes",
    )
    _require(
        handoff.get("history_boundary_count") == 2,
        "partitioned chunk did not publish exactly two history boundaries",
    )
    _require(
        handoff.get("high_stage_first_call_actual") is True,
        "handoff completion did not prove first-high actual",
    )

    model_calls = [event for event in window if _event_kind(event) == "model_call"]
    logical = len(model_calls)
    actual = sum(bool(_event_fields(event).get("actual")) for event in model_calls)
    forecast = logical - actual
    _require(logical > 0, "partitioned chunk contains no model-call telemetry")
    if require_spectrum:
        _require(forecast > 0, "Spectrum-enabled gate expected at least one forecast call")
    if expected_logical is not None:
        _require(
            logical == expected_logical,
            f"logical calls {logical} != expected {expected_logical}",
        )
    if expected_actual is not None:
        _require(
            actual == expected_actual,
            f"actual calls {actual} != expected {expected_actual}",
        )
    if expected_forecast is not None:
        _require(
            forecast == expected_forecast,
            f"forecast calls {forecast} != expected {expected_forecast}",
        )

    low_logical, low_actual = _stage_counts(model_calls, "low")
    probe_logical, probe_actual = _stage_counts(model_calls, "probe")
    high_logical, high_actual = _stage_counts(model_calls, "high")
    _require(
        low_logical > 0 and high_logical > 0,
        "partitioned chunk is missing low or high model calls",
    )
    _require(
        probe_logical == 1 and probe_actual == 1,
        "handoff probe must be exactly one actual H3 model call",
    )
    first_high = next(
        event
        for event in model_calls
        if str(_event_fields(event).get("stage")) == "high"
    )
    _require(
        _event_fields(first_high).get("actual") is True,
        "first high-stage model call was forecast",
    )

    sol_records = _partitioned_sol_records(_parse_sol_records(log_text))
    _require(
        bool(sol_records),
        "ComfyUI log contains no partitioned Sol arithmetic-gate record",
    )
    _require(
        all(record.get("success") is True for record in sol_records),
        "a partitioned Sol request failed",
    )
    for record in sol_records:
        _require(
            int(record.get("external_mixed_sol_calls", 0)) == 0,
            "deprecated external Mixed-Grid Sol route executed",
        )
        _require(
            int(record.get("external_mixed_measure_calls", 0)) == 0,
            "deprecated Mixed-Grid measure route executed",
        )
        _require(
            int(record.get("external_mixed_weighted_measure_calls", 0)) == 0,
            "deprecated weighted Mixed-Grid route executed",
        )
        _require(
            int(record.get("vdn_square_expanded_calls", 0)) == 0,
            "VDN square-Q expansion executed",
        )
        _require(
            int(record.get("vdn_square_requested_rows", 0)) == 0,
            "VDN square-Q requested-row accounting is nonzero",
        )
        _require(
            int(record.get("vdn_square_kernel_rows", 0)) == 0,
            "VDN square-Q kernel-row accounting is nonzero",
        )

    sol_mapped = sum(
        int(record.get("vdn_mapped_sol_calls", 0)) for record in sol_records
    )
    sol_rectangular = sum(
        int(record.get("vdn_rectangular_sol_calls", 0)) for record in sol_records
    )
    sol_requested = sum(
        int(record.get("vdn_requested_q_rows", 0)) for record in sol_records
    )
    sol_kernel = sum(
        int(record.get("vdn_kernel_q_rows", 0)) for record in sol_records
    )
    _require(
        sol_mapped > 0,
        "partitioned production run exercised no mapped-neighbor Sol local call",
    )
    _require(
        sol_rectangular > 0,
        "partitioned production run exercised no rectangular Sol local call",
    )
    _require(
        sol_requested > 0 and sol_requested == sol_kernel,
        "partitioned Sol expanded or lost requested Q rows",
    )

    mapped_bias_gate = False
    for record in sol_records:
        for gate in record.get("arithmetic_gates", []):
            if not isinstance(gate, dict) or gate.get("route") != PARTITIONED_SOL_ABI:
                continue
            if gate.get("mapped_neighbor_abi") is True and gate.get("key_measure_bias") is True:
                mapped_bias_gate = True
                break
        if mapped_bias_gate:
            break
    _require(
        mapped_bias_gate,
        "SM120 arithmetic gate did not cover mapped-neighbor + key-measure execution",
    )

    vdn_linear = VDN_LINEAR_ACTIVE_MARKER in log_text
    audio_overlap = bool(
        AUDIO_OVERLAP_MARKER in log_text
        and re.search(r"partitioned audio guided overlap active ticks=4\b", log_text)
    )
    if require_vdn_linear:
        _require(
            vdn_linear,
            "VDN API-4 variable-grid linear complement was not observed active",
        )
    if require_audio_overlap:
        _require(
            audio_overlap,
            "four-tick partitioned audio guided overlap was not observed",
        )

    return RuntimeGateReport(
        logical_calls=logical,
        actual_calls=actual,
        forecast_calls=forecast,
        low_logical=low_logical,
        low_actual=low_actual,
        probe_logical=probe_logical,
        probe_actual=probe_actual,
        high_logical=high_logical,
        high_actual=high_actual,
        partitioned_transformer_events=len(transformer_events),
        sol_partitioned_requests=len(sol_records),
        sol_mapped_calls=sol_mapped,
        sol_rectangular_calls=sol_rectangular,
        sol_requested_q_rows=sol_requested,
        sol_kernel_q_rows=sol_kernel,
        vdn_variable_grid_linear_active=vdn_linear,
        audio_guided_overlap_active=audio_overlap,
    )


__all__ = [
    "PARTITIONED_SOL_ABI",
    "RuntimeGateError",
    "RuntimeGateReport",
    "validate_partitioned_runtime_evidence",
]
