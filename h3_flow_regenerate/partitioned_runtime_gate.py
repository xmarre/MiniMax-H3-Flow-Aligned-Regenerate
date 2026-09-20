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
AUDIO_POSITION_DOMAIN_LEGACY = "legacy_target"
AUDIO_POSITION_DOMAIN_SOURCE = "source_carrier"
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
    audio_position_domain: str | None
    audio_position_candidate_verified: bool
    audio_position_candidate_block0_calls: int
    audio_position_model_timestep_override_calls: int

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


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_audio_position_policy(
    window: list[dict[str, Any]],
    transformer_events: list[dict[str, Any]],
    plan_fields: dict[str, Any],
    *,
    expected_audio_position_domain: str | None,
) -> tuple[str | None, bool, int, int]:
    if expected_audio_position_domain is None:
        return None, False, 0, 0
    _require(
        expected_audio_position_domain in {AUDIO_POSITION_DOMAIN_LEGACY, AUDIO_POSITION_DOMAIN_SOURCE},
        f"unsupported expected audio-position domain {expected_audio_position_domain!r}",
    )
    observed_plan_domain = plan_fields.get("audio_position_domain")
    if expected_audio_position_domain == AUDIO_POSITION_DOMAIN_LEGACY:
        _require(
            observed_plan_domain in {None, AUDIO_POSITION_DOMAIN_LEGACY},
            "legacy control selected a non-legacy target-audio position domain",
        )
        return AUDIO_POSITION_DOMAIN_LEGACY, False, 0, 0

    _require(
        observed_plan_domain == AUDIO_POSITION_DOMAIN_SOURCE,
        "source-carrier candidate was not selected in the partitioned stage plan",
    )
    for event in transformer_events:
        fields = _event_fields(event)
        _require(
            fields.get("audio_position_domain") == AUDIO_POSITION_DOMAIN_SOURCE,
            "source-carrier candidate did not reach every actual low/probe transformer call",
        )
        _require(
            fields.get("audio_position_policy_active") is True,
            "source-carrier candidate did not publish an active position policy",
        )
        _require(
            fields.get("audio_position_temporal_equal") is True,
            "source-carrier candidate changed target-audio temporal coordinates",
        )
        before = fields.get("audio_position_non_audio_before_digest")
        after = fields.get("audio_position_non_audio_after_digest")
        _require(
            _sha256(before) and before == after,
            "source-carrier candidate changed non-target-audio position rows",
        )
        _require(
            _sha256(fields.get("prefix_rope_position_digest")),
            "source-carrier candidate is missing the target-prefix RoPE receipt",
        )
        _require(
            _sha256(fields.get("suffix_rope_position_digest")),
            "source-carrier candidate is missing the source-suffix RoPE receipt",
        )
        _require(
            _sha256(fields.get("position_digest")),
            "source-carrier candidate is missing the complete position digest",
        )
        _require(
            bool(fields.get("audio_position_policy_signature")),
            "source-carrier candidate is missing its numerical-history policy signature",
        )
        owner = fields.get("audio_position_stage_owner_generation")
        _require(
            type(owner) is int and owner > 0,
            "source-carrier candidate is missing its stage-owner position-policy receipt",
        )

    verified_events = [
        _event_fields(event)
        for event in window
        if _event_kind(event) == "partitioned_audio_position_domain_verified"
    ]
    _require(bool(verified_events), "source-carrier candidate emitted no execution-verification receipt")
    verified = verified_events[-1]
    _require(verified.get("mode") == AUDIO_POSITION_DOMAIN_SOURCE, "candidate execution receipt mode drifted")
    block0_calls = int(verified.get("actual_block0_calls", 0))
    wrapper_entries = int(verified.get("wrapper_entries", 0))
    model_timestep_calls = int(verified.get("model_timestep_override_calls", 0))
    _require(block0_calls > 0, "source-carrier candidate executed no actual block-zero calls")
    _require(
        wrapper_entries >= block0_calls,
        "source-carrier candidate wrapper/block execution accounting drifted",
    )
    _require(
        model_timestep_calls > 0,
        "source-carrier candidate gate requires observed model-timestep-only audio guidance",
    )

    integrity_events = [
        _event_fields(event)
        for event in window
        if _event_kind(event) == "partitioned_audio_position_candidate_integrity"
    ]
    _require(bool(integrity_events), "source-carrier candidate emitted no final exact-AV integrity receipt")
    integrity = integrity_events[-1]
    _require(integrity.get("mode") == AUDIO_POSITION_DOMAIN_SOURCE, "candidate integrity receipt mode drifted")
    _require(
        integrity.get("final_exact_video_prefix") is True
        and integrity.get("final_exact_audio_prefix") is True,
        "source-carrier candidate did not preserve the caller-owned exact AV prefix",
    )
    _require(
        integrity.get("sampler_masks_unchanged") is True,
        "source-carrier candidate changed sampler-mask ownership",
    )
    _require(
        _sha256(integrity.get("low_mask_digest")) and _sha256(integrity.get("high_mask_digest")),
        "source-carrier candidate is missing sampler-mask receipts",
    )
    return AUDIO_POSITION_DOMAIN_SOURCE, True, block0_calls, model_timestep_calls


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
    expected_audio_position_domain: str | None = None,
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
    audio_position_domain, candidate_verified, candidate_block0_calls, model_timestep_calls = (
        _validate_audio_position_policy(
            window,
            transformer_events,
            plan_fields,
            expected_audio_position_domain=expected_audio_position_domain,
        )
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
        audio_position_domain=audio_position_domain,
        audio_position_candidate_verified=candidate_verified,
        audio_position_candidate_block0_calls=candidate_block0_calls,
        audio_position_model_timestep_override_calls=model_timestep_calls,
    )


__all__ = [
    "AUDIO_POSITION_DOMAIN_LEGACY",
    "AUDIO_POSITION_DOMAIN_SOURCE",
    "PARTITIONED_SOL_ABI",
    "RuntimeGateError",
    "RuntimeGateReport",
    "validate_partitioned_runtime_evidence",
]
