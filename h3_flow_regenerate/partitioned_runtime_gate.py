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
    "partitioned exact-prefix: grouped VDN softmax active; variable-grid linear complement active"
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
    partitioned_provider_creations: int
    partitioned_provider_reuses: int
    partitioned_provider_equivalent_rebindings: int
    partitioned_provider_semantic_transitions: int
    sol_partitioned_requests: int
    sol_mapped_calls: int
    sol_rectangular_calls: int
    sol_requested_q_rows: int
    sol_kernel_q_rows: int
    vdn_variable_grid_linear_active: bool
    audio_guided_overlap_active: bool
    request_id: str | None
    correlated_model_calls: int
    stage_accounting_rows: int
    stage_accounting_unknown_rows: int
    partitioned_component_summary_events: int
    partitioned_host_component_names: tuple[str, ...]
    sol_correlation_records: int
    vdn_cuda_samples: int
    vdn_cuda_component_names: tuple[str, ...]

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
    starts = [index for index, event in enumerate(events) if _event_kind(event) == "partitioned_stage_plan"]
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
        if any(isinstance(gate, dict) and gate.get("route") == PARTITIONED_SOL_ABI for gate in gates):
            selected.append(record)
    return selected


def _stage_counts(model_calls: list[dict[str, Any]], stage: str) -> tuple[int, int]:
    calls = [event for event in model_calls if str(_event_fields(event).get("stage")) == stage]
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
    require_performance_accounting: bool = False,
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

    plan = next(event for event in window if _event_kind(event) == "partitioned_stage_plan")
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

    transformer_events = [event for event in window if _event_kind(event) == "partitioned_exact_prefix_transformer"]
    _require(
        bool(transformer_events),
        "partitioned transformer emitted no physical-domain evidence",
    )

    # Provider stability is part of the numerical-history contract, not merely a
    # performance counter. Provider selection happens when the model wrapper is
    # entered, before Spectrum may satisfy that logical model call from a forecast.
    # Therefore provider bindings correspond to logical low/probe calls, whereas
    # partitioned_transformer_calls counts only actual H3 executions. 00508 proved
    # this distinction with 6 bindings (2 creations + 4 reuses) but only 5 actual
    # partitioned transformer executions.
    partitioned_calls = int(counters.get("partitioned_transformer_calls", 0))
    provider_creations = int(counters.get("partitioned_attention_provider_creations", 0))
    provider_reuses = int(counters.get("partitioned_attention_provider_reuses", 0))
    provider_rebindings = int(counters.get("partitioned_attention_equivalent_provider_rebindings", 0))
    provider_transitions = int(counters.get("partitioned_attention_inherited_provider_transitions", 0))
    _require(
        partitioned_calls > 0,
        "partitioned transformer-call counter is missing or zero",
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

    transfers = [event for event in window if _event_kind(event) == "partitioned_transfer"]
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

    completes = [event for event in window if _event_kind(event) == "partitioned_exact_prefix_complete"]
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

    handoffs = [event for event in window if _event_kind(event) == "handoff_complete"]
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

    partitioned_logical = low_logical + probe_logical
    partitioned_actual = low_actual + probe_actual
    _require(
        partitioned_calls == partitioned_actual,
        "partitioned transformer-call accounting does not match actual low/probe calls",
    )
    _require(
        provider_creations + provider_reuses == partitioned_logical,
        "partitioned provider binding accounting does not match logical low/probe calls",
    )
    if partitioned_logical > 1:
        _require(
            provider_reuses > 0,
            "partitioned provider identity changed on every logical low/probe call",
        )

    first_high = next(event for event in model_calls if str(_event_fields(event).get("stage")) == "high")
    _require(
        _event_fields(first_high).get("actual") is True,
        "first high-stage model call was forecast",
    )

    request_ids = {
        fields.get("request_id")
        for fields in (_event_fields(event) for event in model_calls)
        if isinstance(fields.get("request_id"), str) and fields.get("request_id")
    }
    request_id = next(iter(request_ids)) if len(request_ids) == 1 else None
    correlated_model_calls = sum(
        isinstance(_event_fields(event).get("request_id"), str)
        and isinstance(_event_fields(event).get("stage_id"), str)
        and isinstance(_event_fields(event).get("evaluation_id"), str)
        for event in model_calls
    )
    stage_accounting = metrics.get("stage_accounting")
    accounting_rows = []
    if isinstance(stage_accounting, list) and request_id is not None:
        accounting_rows = [
            row for row in stage_accounting if isinstance(row, dict) and row.get("request_id") == request_id
        ]
    accounting_unknown = sum(
        row.get("wall_ms") is None or row.get("model_ms") is None or row.get("remainder_ms") is None
        for row in accounting_rows
    )
    if require_performance_accounting:
        _require(
            len(request_ids) == 1,
            "partitioned model calls do not share one Flow request correlation ID",
        )
        _require(
            correlated_model_calls == logical,
            "partitioned model-call request/stage/evaluation correlation is incomplete",
        )
        _require(
            len(accounting_rows) == 3,
            "partitioned stage accounting does not contain exactly low/probe/high rows",
        )
        _require(
            accounting_unknown == 0,
            "partitioned stage accounting contains unknown timing; missing values are not zero",
        )
        stage_kinds = {row.get("kind") for row in accounting_rows}
        _require(
            stage_kinds == {"low_stage_wall", "handoff_probe_wall", "high_stage_wall"},
            "partitioned stage accounting does not identify low/probe/high walls",
        )

    component_summaries = [
        _event_fields(event)
        for event in window
        if _event_kind(event) == "partitioned_stage_runtime_summary"
        and _event_fields(event).get("request_id") == request_id
    ]
    host_component_names = set()
    for fields in component_summaries:
        components = fields.get("host_component_s")
        if isinstance(components, dict):
            host_component_names.update(
                name
                for name, value in components.items()
                if isinstance(name, str) and isinstance(value, (int, float))
            )
    if require_performance_accounting:
        _require(
            bool(component_summaries),
            "partitioned Flow metrics contain no correlated VDN component summary",
        )
        required_host_components = {
            "vdn_gather_host_wall_s",
            "vdn_softmax_host_wall_s",
            "vdn_linear_readout_total_host_wall_s",
            "vdn_linear_gate_host_wall_s",
            "vdn_linear_epsilon_scalar_host_wall_s",
        }
        _require(
            required_host_components.issubset(host_component_names),
            "partitioned VDN host attribution is incomplete",
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

    sol_correlation_records = 0
    if request_id is not None:
        for record in sol_records:
            validation = record.get("validation")
            examples = validation.get("examples") if isinstance(validation, dict) else None
            if not isinstance(examples, list):
                continue
            seen = {
                (entry.get("context") or {}).get("flow_request_id")
                for entry in examples
                if isinstance(entry, dict) and isinstance(entry.get("context"), dict)
            }
            seen.discard(None)
            if request_id in seen:
                sol_correlation_records += 1
            if require_performance_accounting and seen:
                _require(
                    seen == {request_id},
                    "Sol arithmetic-gate correlation references a different Flow request",
                )
    if require_performance_accounting:
        _require(
            sol_correlation_records > 0,
            "no Sol arithmetic-validation record correlates to the partitioned Flow request",
        )

    vdn_cuda_samples = []
    vdn_cuda_component_names = set()
    if request_id is not None:
        for record in sol_records:
            diagnostics = record.get("cuda_diagnostics")
            details = diagnostics.get("details") if isinstance(diagnostics, dict) else None
            if not isinstance(details, list):
                continue
            for sample in details:
                if not isinstance(sample, dict) or sample.get("kind") != "vdn_partitioned_components":
                    continue
                context = sample.get("context")
                if not isinstance(context, dict) or context.get("flow_request_id") != request_id:
                    continue
                spans = sample.get("cuda_event_ms")
                if not isinstance(spans, dict):
                    continue
                vdn_cuda_samples.append(sample)
                vdn_cuda_component_names.update(
                    name for name, value in spans.items()
                    if isinstance(name, str) and isinstance(value, (int, float))
                )
    if require_performance_accounting:
        _require(
            bool(vdn_cuda_samples),
            "Sol CUDA diagnostics contain no correlated VDN partitioned component sample",
        )
        required_cuda_components = {
            "vdn_preprocess",
            "vdn_gather",
            "vdn_softmax",
            "vdn_weights",
            "vdn_softmax_epilogue",
            "vdn_linear_features",
            "vdn_linear_statistics",
            "vdn_linear_scans",
            "vdn_linear_gate",
            "vdn_linear_gather",
            "vdn_linear_epsilon_scalar",
            "vdn_linear_output",
            "vdn_linear_projection",
        }
        _require(
            required_cuda_components.issubset(vdn_cuda_component_names),
            "partitioned VDN CUDA attribution is incomplete",
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

    sol_mapped = sum(int(record.get("vdn_mapped_sol_calls", 0)) for record in sol_records)
    sol_rectangular = sum(int(record.get("vdn_rectangular_sol_calls", 0)) for record in sol_records)
    sol_requested = sum(int(record.get("vdn_requested_q_rows", 0)) for record in sol_records)
    sol_kernel = sum(int(record.get("vdn_kernel_q_rows", 0)) for record in sol_records)
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
        AUDIO_OVERLAP_MARKER in log_text and re.search(r"partitioned audio guided overlap active ticks=4\b", log_text)
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
        partitioned_provider_creations=provider_creations,
        partitioned_provider_reuses=provider_reuses,
        partitioned_provider_equivalent_rebindings=provider_rebindings,
        partitioned_provider_semantic_transitions=provider_transitions,
        sol_partitioned_requests=len(sol_records),
        sol_mapped_calls=sol_mapped,
        sol_rectangular_calls=sol_rectangular,
        sol_requested_q_rows=sol_requested,
        sol_kernel_q_rows=sol_kernel,
        vdn_variable_grid_linear_active=vdn_linear,
        audio_guided_overlap_active=audio_overlap,
        request_id=request_id,
        correlated_model_calls=correlated_model_calls,
        stage_accounting_rows=len(accounting_rows),
        stage_accounting_unknown_rows=accounting_unknown,
        partitioned_component_summary_events=len(component_summaries),
        partitioned_host_component_names=tuple(sorted(host_component_names)),
        sol_correlation_records=sol_correlation_records,
        vdn_cuda_samples=len(vdn_cuda_samples),
        vdn_cuda_component_names=tuple(sorted(vdn_cuda_component_names)),
    )


__all__ = [
    "PARTITIONED_SOL_ABI",
    "RuntimeGateError",
    "RuntimeGateReport",
    "validate_partitioned_runtime_evidence",
]
