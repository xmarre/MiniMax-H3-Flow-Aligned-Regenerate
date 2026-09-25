"""Post-run validation for partitioned exact-prefix production evidence.

The validator is deliberately offline. It consumes the ordinary Flow metrics JSON
and the ordinary ComfyUI log so the hardware acceptance run does not acquire a
second diagnostic execution path or mutate sampler/backend state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .frame_gauge import (
    FRAME_GAUGE_POLICY_VERSION,
    GUIDANCE_REFERENCE_POLICY,
    LEARNED_VIDEO_POLICY,
    FrameGaugePolicy,
)
from .residual_geometry import (
    DEFAULT_RESIDUAL_POLICY,
    RESIDUAL_GEOMETRY_POLICY_VERSION,
    diagnostic_model_fits,
    replay_horizontal_telemetry,
)

PARTITIONED_SOL_ABI = "sol-h3-partitioned-single-union-v1"
VDN_LINEAR_ACTIVE_MARKER = (
    "partitioned exact-prefix: grouped VDN softmax active; variable-grid linear complement active"
)
AUDIO_OVERLAP_MARKER = "partitioned audio guided overlap mode="
AUDIO_POSITION_DOMAIN_LEGACY = "legacy_target"
AUDIO_POSITION_DOMAIN_SOURCE = "source_carrier"
PARTITIONED_EXACT_OVERLAP_POLICY = "partitioned_exact_overlap_structural_plus_dc_v1"
FRAME_GAUGE_EXACT_OVERLAP_FALLBACK_REASONS = frozenset(
    {
        "boundary_upper45_not_improved",
        "boundary_full_not_improved",
        "boundary_upper45_insufficient_improvement",
        "boundary_full_insufficient_improvement",
    }
)
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
    frame_gauge_mode: str | None
    frame_gauge_result: str | None
    frame_gauge_verified: bool
    frame_gauge_video_dx: float | None
    frame_gauge_video_dy: float | None
    frame_gauge_guidance_dx: float | None
    frame_gauge_guidance_dy: float | None
    residual_geometry_mode: str | None
    residual_geometry_result: str | None
    residual_geometry_verified: bool
    residual_geometry_horizontal_eligible: bool
    residual_geometry_evidence_bundle: str | None
    auto_strength_verified_off: bool
    auto_strength_report_digests: tuple[str, ...]

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
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


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
        _event_fields(event) for event in window if _event_kind(event) == "partitioned_audio_position_domain_verified"
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
        integrity.get("final_exact_video_prefix") is True and integrity.get("final_exact_audio_prefix") is True,
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


def _finite_number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeGateError(f"frame-gauge receipt contains non-numeric value {value!r}") from exc
    _require(math.isfinite(result), "frame-gauge receipt contains a non-finite numeric value")
    return result


def _validate_registration_receipt(
    fields: Any,
    *,
    label: str,
    policy: FrameGaugePolicy,
) -> str:
    _require(isinstance(fields, dict), f"{label} registration receipt is missing")
    status = str(fields.get("status", ""))
    _require(status in {"accepted", "identity"}, f"{label} registration was not accepted/identity")
    _require(fields.get("policy_version") == FRAME_GAUGE_POLICY_VERSION, f"{label} registration policy version drifted")
    _require(fields.get("units") == "target_latent_cells", f"{label} registration units drifted")
    dx = _finite_number(fields.get("dx"))
    dy = _finite_number(fields.get("dy"))
    _require(
        abs(dx) <= policy.accepted_bound and abs(dy) <= policy.accepted_bound,
        f"{label} registration exceeds the displacement bound",
    )
    invalid_fraction = _finite_number(fields.get("invalid_fraction", 0.0))
    _require(
        0.0 <= invalid_fraction <= policy.max_invalid_fraction,
        f"{label} registration invalid-area bound failed",
    )
    if status == "identity":
        _require(dx == 0.0 and dy == 0.0, f"{label} identity registration is not an exact no-op")
        return status

    _require(
        _finite_number(fields.get("validation_ncc")) >= policy.min_ncc,
        f"{label} registration held-out NCC gate failed",
    )
    _require(
        _finite_number(fields.get("rms_improvement")) >= policy.min_rms_improvement,
        f"{label} registration held-out RMS-improvement gate failed",
    )
    runner_margin = _finite_number(fields.get("runner_margin_ratio"))
    _require(
        runner_margin >= 0.0,
        f"{label} registration runner-up margin is invalid",
    )
    if policy.require_global_runner_margin:
        _require(
            runner_margin >= policy.min_runner_margin,
            f"{label} registration runner-up margin gate failed",
        )
    _require(
        _finite_number(fields.get("last_holdout_ncc")) >= policy.min_ncc,
        f"{label} registration last-frame NCC gate failed",
    )
    _require(
        _finite_number(fields.get("last_holdout_improvement")) >= policy.min_rms_improvement,
        f"{label} registration last-frame improvement gate failed",
    )

    frame_checks = fields.get("frame_checks")
    _require(isinstance(frame_checks, list), f"{label} registration held-out frame receipts are missing")
    informative_frames = [item for item in frame_checks if isinstance(item, dict) and item.get("informative")]
    _require(len(informative_frames) >= 2, f"{label} registration has insufficient held-out frame support")
    _require(
        all(item.get("supports_global") is True for item in informative_frames),
        f"{label} registration held-out frame agreement failed",
    )

    region_checks = fields.get("region_checks")
    _require(isinstance(region_checks, list), f"{label} registration regional receipts are missing")
    by_name = {str(item.get("name")): item for item in region_checks if isinstance(item, dict)}
    _require(
        not any(item.get("strong_conflict") is True for item in by_name.values()),
        f"{label} registration has a strong regional conflict",
    )
    vertical = any(
        by_name.get(name, {}).get("informative") is True and by_name.get(name, {}).get("supports_global") is True
        for name in ("upper", "lower")
    )
    horizontal = any(
        by_name.get(name, {}).get("informative") is True and by_name.get(name, {}).get("supports_global") is True
        for name in ("left", "right")
    )
    _require(vertical and horizontal, f"{label} registration lacks independent regional support")

    parity_checks = fields.get("parity_checks")
    _require(isinstance(parity_checks, list), f"{label} registration patch-phase receipts are missing")
    _require(
        all(
            not item.get("informative") or item.get("supports_global") is True
            for item in parity_checks
            if isinstance(item, dict)
        ),
        f"{label} registration patch-phase agreement failed",
    )
    return status


def _validate_boundary_motion_receipt(fields: Any) -> None:
    _require(isinstance(fields, dict), "accepted frame-gauge transaction is missing boundary-motion evidence")
    _require(fields.get("status") == "accepted", "accepted frame-gauge boundary-motion gate did not accept")
    policy = fields.get("policy")
    _require(
        policy in ("native_boundary_motion_preservation_v1", "native_boundary_motion_preservation_v2"),
        "frame-gauge boundary-motion policy drifted",
    )
    min_error = _finite_number(fields.get("min_error_cells"))
    min_improvement = _finite_number(fields.get("min_improvement_ratio"))
    min_response = _finite_number(fields.get("min_response"))
    _require(min_error == 0.125, "frame-gauge boundary-motion error threshold drifted")
    _require(min_improvement == 0.25, "frame-gauge boundary-motion improvement threshold drifted")
    _require(min_response == 3.0, "frame-gauge boundary-motion response threshold drifted")

    checks = fields.get("checks")
    _require(isinstance(checks, dict), "frame-gauge boundary-motion ROI receipts are missing")
    _require(set(checks) == {"upper45", "full"}, "frame-gauge boundary-motion ROI set drifted")
    for name in ("upper45", "full"):
        check = checks[name]
        _require(isinstance(check, dict), f"frame-gauge boundary-motion {name} receipt is malformed")
        before_error = _finite_number(check.get("before_error_cells"))
        after_error = _finite_number(check.get("after_error_cells"))
        improvement = _finite_number(check.get("error_improvement_ratio"))
        _require(before_error >= 0.0 and after_error >= 0.0, f"frame-gauge boundary-motion {name} error is negative")
        _require(
            after_error <= before_error,
            f"frame-gauge boundary-motion {name} candidate did not improve",
        )
        expected_informative = before_error >= min_error
        _require(
            check.get("informative") is expected_informative,
            f"frame-gauge boundary-motion {name} informative classification drifted",
        )
        if expected_informative:
            _require(
                improvement >= min_improvement,
                f"frame-gauge boundary-motion {name} improvement gate failed",
            )
        variants = ("native", "exact_restored", "candidate")
        if policy == "native_boundary_motion_preservation_v2":
            variants += ("transformed_native",)
        for variant in variants:
            receipt = check.get(variant)
            _require(isinstance(receipt, dict), f"frame-gauge boundary-motion {name}/{variant} receipt is missing")
            _finite_number(receipt.get("dx"))
            _finite_number(receipt.get("dy"))
            response = _finite_number(receipt.get("response"))
            _require(receipt.get("clipped") is False, f"frame-gauge boundary-motion {name}/{variant} clipped")
            _require(
                response >= min_response,
                f"frame-gauge boundary-motion {name}/{variant} response gate failed",
            )
        if policy == "native_boundary_motion_preservation_v2":
            # Replay the coordinate-domain comparisons rather than trusting
            # summary errors that could have been computed against native v1.
            def distance(left, right):
                return math.hypot(
                    _finite_number(left["dx"]) - _finite_number(right["dx"]),
                    _finite_number(left["dy"]) - _finite_number(right["dy"]),
                )

            expected_before = distance(check["exact_restored"], check["native"])
            expected_after = distance(check["candidate"], check["transformed_native"])
            expected_improvement = (expected_before - expected_after) / max(expected_before, 1e-12)
            for label, actual, expected in (
                ("before error", before_error, expected_before),
                ("after error", after_error, expected_after),
                ("improvement", improvement, expected_improvement),
            ):
                _require(
                    math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12),
                    f"frame-gauge boundary-motion {name} {label} does not reproduce",
                )


def _validate_frame_gauge_transfer(
    window: list[dict[str, Any]],
    *,
    mode: str,
    result: str,
) -> None:
    transfers = [_event_fields(event) for event in window if _event_kind(event) == "partitioned_transfer"]
    _require(len(transfers) == 1, "partitioned run must emit exactly one transfer receipt")
    transfer = transfers[0]
    enabled = mode == "on"
    accepted = enabled and result == "accepted"
    _require(
        transfer.get("frame_gauge_repair_enabled") is enabled,
        "frame-gauge transfer toggle differs from the transaction receipt",
    )
    _require(
        str(transfer.get("frame_gauge_result", "")) == result,
        "frame-gauge transfer result differs from the transaction receipt",
    )
    _require(
        transfer.get("authoritative_target_prefix_restored") is True,
        "frame-gauge transfer did not restore the caller-owned exact prefix",
    )
    _require(
        transfer.get("deprecated_mixed_grid_repairs_applied") is False,
        "frame-gauge transfer activated deprecated mixed-grid repairs",
    )
    _require(
        transfer.get("suffix_dc_bridge_policy") == "one_token_spatial_mean_v1",
        "frame-gauge transfer changed the existing one-token DC policy",
    )
    _require(
        int(transfer.get("suffix_dc_bridge_corrected_tokens", 0)) == 1,
        "frame-gauge transfer did not apply exactly one DC-corrected suffix token",
    )
    expected_mapping = "pre_renoise_clean_operand" if accepted else "conditional_renoise_affine"
    expected_clean_source = "actual_provider" if accepted else "inverse_recovered"
    _require(
        transfer.get("suffix_dc_bridge_state_mapping") == expected_mapping,
        "frame-gauge transfer changed DC routing for the selected arm",
    )
    _require(
        transfer.get("splice_clean_source") == expected_clean_source,
        "frame-gauge transfer used the wrong clean-state source",
    )
    _require(
        "suffix_gauge_bridge_policy" not in transfer and "suffix_exact_prefix_gauge_bridge_policy" not in transfer,
        "retired full-field residual/gauge bridge became active",
    )

    # Current partitioned heads emit this nested receipt on every path.  It is
    # optional here so historical evidence remains replayable.  When present,
    # validate the partitioned-local exact-overlap fallback independently from
    # the rigid frame-gauge result; it is not a deprecated Mixed-Grid route.
    overlap = transfer.get("partitioned_exact_overlap_bridge")
    if overlap is not None:
        _require(isinstance(overlap, dict), "partitioned exact-overlap receipt is malformed")
        _require(
            overlap.get("policy") == PARTITIONED_EXACT_OVERLAP_POLICY,
            "partitioned exact-overlap policy version drifted",
        )
        requested = overlap.get("requested") is True
        applied = overlap.get("applied") is True
        _require(
            overlap.get("authoritative_prefix_modified") is False,
            "partitioned exact-overlap repair altered authoritative prefix ownership",
        )
        _require(
            overlap.get("later_suffix_extrapolated") is False,
            "partitioned exact-overlap repair extrapolated into unmeasured later suffix tokens",
        )
        if requested:
            _require(
                enabled and result == "rejected",
                "partitioned exact-overlap repair was requested outside a rejected frame-gauge arm",
            )
            _require(
                str(overlap.get("trigger", "")) in FRAME_GAUGE_EXACT_OVERLAP_FALLBACK_REASONS,
                "partitioned exact-overlap repair used an ineligible frame-gauge rejection",
            )
        if applied:
            _require(requested, "partitioned exact-overlap repair applied without being requested")
            _require(
                overlap.get("suffix_representation_bridge_accepted") is True
                and overlap.get("suffix_representation_bridge_enabled") is True,
                "partitioned exact-overlap repair lacks an accepted structural-overlap receipt",
            )
            _require(
                int(overlap.get("suffix_representation_bridge_corrected_tokens", 0)) == 1,
                "partitioned exact-overlap repair changed more than the first suffix token",
            )
            _require(
                overlap.get("state_mapping") == "conditional_renoise_affine",
                "partitioned exact-overlap repair used the wrong conditional-state mapping",
            )
        else:
            _require(
                overlap.get("state_mapping") == "disabled_or_noop",
                "inactive partitioned exact-overlap repair reports an active state mapping",
            )


def _validate_frame_gauge(
    window: list[dict[str, Any]],
    *,
    expected_mode: str | None,
) -> tuple[str | None, str | None, bool, float | None, float | None, float | None, float | None]:
    if expected_mode is None:
        return None, None, False, None, None, None, None
    allowed = {"off", "on-accepted", "on-rejected", "on-identity"}
    _require(expected_mode in allowed, f"unsupported expected frame-gauge mode {expected_mode!r}")
    receipts = [_event_fields(event) for event in window if _event_kind(event) == "partitioned_frame_gauge"]
    _require(len(receipts) == 1, "partitioned run must emit exactly one frame-gauge receipt")
    receipt = receipts[0]
    mode = str(receipt.get("mode", ""))
    result = str(receipt.get("result", ""))
    if expected_mode == "off":
        _require(mode == "off" and receipt.get("enabled") is False, "frame-gauge OFF control was not actually disabled")
        _require(result == "off", f"frame-gauge OFF control reported unexpected result {result!r}")
        _require(receipt.get("spatial_warp_applied") is False, "frame-gauge OFF control applied a spatial warp")
    else:
        expected_result = expected_mode.removeprefix("on-")
        _require(mode == "on" and receipt.get("enabled") is True, "frame-gauge ON arm was not actually enabled")
        _require(result == expected_result, f"frame-gauge ON arm result {result!r} != expected {expected_result!r}")
        if expected_result == "accepted":
            _require(
                receipt.get("spatial_warp_applied") is True,
                "accepted frame-gauge transaction applied no spatial warp",
            )
            guidance_mode = str(receipt.get("guidance_mode", "off"))
            if guidance_mode != "off":
                _require(
                    receipt.get("registered_guidance_reference") is True,
                    "accepted frame-gauge transaction did not publish its independently registered Flow reference",
                )
        else:
            _require(
                receipt.get("spatial_warp_applied") is False,
                "non-accepted frame-gauge transaction applied a spatial warp",
            )
            _require(
                receipt.get("registered_guidance_reference") is False,
                "non-accepted frame-gauge transaction published a registered Flow reference",
            )

    _require(
        receipt.get("authoritative_prefix_modified") is False,
        "frame-gauge transaction altered exact-prefix ownership",
    )
    handoff_receipts = [_event_fields(event) for event in window if _event_kind(event) == "handoff_transfer_wall"]
    _require(
        bool(handoff_receipts) and handoff_receipts[-1].get("protected_video_noise_exact") is True,
        "frame-gauge evidence does not prove protected video-noise ownership",
    )
    _require(
        _sha256(receipt.get("exact_prefix_sha256")),
        "frame-gauge transaction is missing the exact-prefix identity receipt",
    )
    expected_registration_domain = "actual_clean_target_video" if mode == "on" else "off"
    expected_transform_domain = "actual_clean_target_video" if mode == "on" and result == "accepted" else "none"
    _require(
        receipt.get("registration_domain") == expected_registration_domain,
        "frame-gauge transaction used the wrong registration domain",
    )
    _require(
        receipt.get("transform_domain") == expected_transform_domain,
        "frame-gauge transaction used the wrong correction domain",
    )
    if mode == "on":
        _require(
            receipt.get("policy_version") == FRAME_GAUGE_POLICY_VERSION,
            "frame-gauge transaction policy version drifted",
        )
        if result == "accepted":
            _require(
                _validate_registration_receipt(
                    receipt.get("video_registration"),
                    label="video",
                    policy=LEARNED_VIDEO_POLICY,
                )
                == "accepted",
                "accepted transaction lacks an accepted video registration",
            )
            _validate_boundary_motion_receipt(receipt.get("boundary_motion"))
            if str(receipt.get("guidance_mode", "off")) != "off":
                _validate_registration_receipt(
                    receipt.get("guidance_registration"),
                    label="guidance",
                    policy=GUIDANCE_REFERENCE_POLICY,
                )
        elif result == "identity":
            _require(
                _validate_registration_receipt(
                    receipt.get("video_registration"),
                    label="video",
                    policy=LEARNED_VIDEO_POLICY,
                )
                == "identity",
                "identity transaction lacks an identity video registration",
            )
    _require(receipt.get("extra_h3_nfe") == 0, "frame-gauge transaction added H3 NFE")
    _require(receipt.get("extra_sampler_lifetimes") == 0, "frame-gauge transaction added a sampler lifetime")
    _require(receipt.get("extra_history_boundaries") == 0, "frame-gauge transaction added a history boundary")
    _require(receipt.get("extra_provider_calls") == 0, "frame-gauge transaction added a learned-provider call")
    _require(receipt.get("extra_vae_calls") == 0, "frame-gauge transaction added a VAE call")
    _require(
        receipt.get("auto_strength_validation_required") is True,
        "frame-gauge receipt did not mark auto-strength provenance as an acceptance prerequisite",
    )
    _validate_frame_gauge_transfer(window, mode=mode, result=result)
    if "exact_overlap_fallback_policy" in receipt:
        _require(
            receipt.get("exact_overlap_fallback_policy") == PARTITIONED_EXACT_OVERLAP_POLICY,
            "frame-gauge exact-overlap fallback policy version drifted",
        )
        fallback_requested = receipt.get("exact_overlap_fallback_requested") is True
        fallback_applied = receipt.get("exact_overlap_fallback_applied") is True
        if fallback_requested:
            _require(
                mode == "on" and result == "rejected",
                "frame-gauge exact-overlap fallback was requested outside the rejected ON arm",
            )
            _require(
                str(receipt.get("exact_overlap_fallback_trigger", ""))
                in FRAME_GAUGE_EXACT_OVERLAP_FALLBACK_REASONS,
                "frame-gauge exact-overlap fallback trigger is not eligible",
            )
        _require(
            not fallback_applied or fallback_requested,
            "frame-gauge exact-overlap fallback applied without a request",
        )
    video_dx = _finite_number(receipt.get("video_dx", 0.0))
    video_dy = _finite_number(receipt.get("video_dy", 0.0))
    guidance_dx = _finite_number(receipt.get("guidance_dx", 0.0))
    guidance_dy = _finite_number(receipt.get("guidance_dy", 0.0))
    if mode == "on" and result == "accepted":
        video_registration = receipt.get("video_registration")
        _require(isinstance(video_registration, dict), "accepted transaction lacks video registration fields")
        _require(
            video_dx == _finite_number(video_registration.get("dx"))
            and video_dy == _finite_number(video_registration.get("dy")),
            "reported video displacement differs from the validated video registration",
        )
        if str(receipt.get("guidance_mode", "off")) != "off":
            guidance_registration = receipt.get("guidance_registration")
            _require(
                isinstance(guidance_registration, dict),
                "accepted transaction lacks guidance registration fields",
            )
            _require(
                guidance_dx == _finite_number(guidance_registration.get("dx"))
                and guidance_dy == _finite_number(guidance_registration.get("dy")),
                "reported guidance displacement differs from the validated guidance registration",
            )
    return (
        mode,
        result,
        True,
        video_dx,
        video_dy,
        guidance_dx,
        guidance_dy,
    )


def _close_number(left: Any, right: Any, *, atol: float = 1e-9) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=atol)
    )


def _validate_residual_observation(
    observation: Any,
    *,
    fit_indices: set[int],
    holdout_indices: set[int],
) -> None:
    _require(isinstance(observation, dict), "residual geometry contains a malformed regional observation")
    status = str(observation.get("status", ""))
    _require(
        status in {"accepted", "rejected", "unavailable", "identity"},
        f"unsupported residual regional status {status!r}",
    )
    frame_index = int(observation.get("frame_index"))
    _require(
        frame_index in fit_indices or frame_index in holdout_indices,
        "residual regional observation references a frame outside fit/holdout support",
    )
    expected_phase = "fit" if frame_index in fit_indices else "holdout"
    _require(
        observation.get("phase") == expected_phase,
        "residual regional observation fit/holdout phase drifted",
    )
    tile_id = str(observation.get("tile_id", ""))
    _require(
        tile_id in {"TL", "TM", "TR", "ML", "C", "MR", "BL", "BM", "BR"},
        "residual regional observation tile identity drifted",
    )
    if "search_count" in observation:
        search_count = int(observation["search_count"])
        _require(1 <= search_count <= 162, "residual regional search exceeded the bounded score budget")
    if status in {"accepted", "identity"}:
        _require(
            int(observation.get("valid_channels", 0)) >= DEFAULT_RESIDUAL_POLICY.min_textured_channels,
            "residual regional accepted/identity observation lacks textured channels",
        )
        _require(
            int(observation.get("support_y", 0)) >= DEFAULT_RESIDUAL_POLICY.min_tile_axis
            and int(observation.get("support_x", 0)) >= DEFAULT_RESIDUAL_POLICY.min_tile_axis,
            "residual regional accepted/identity observation lacks spatial support",
        )
        for key in (
            "zero_loss",
            "zero_rms",
            "best_loss",
            "best_rms",
            "ncc",
            "ux",
            "uy",
            "uncertainty_half_width_x",
            "uncertainty_half_width_y",
        ):
            _finite_number(observation.get(key))
    if status == "accepted":
        _require(
            _finite_number(observation.get("ncc")) >= DEFAULT_RESIDUAL_POLICY.min_ncc,
            "residual regional NCC gate drifted",
        )
        _require(observation.get("saturated") is False, "accepted residual regional search saturated")
        _require(
            abs(_finite_number(observation.get("ux"))) <= DEFAULT_RESIDUAL_POLICY.search_radius
            and abs(_finite_number(observation.get("uy"))) <= DEFAULT_RESIDUAL_POLICY.search_radius,
            "accepted residual regional displacement exceeded its bounded search",
        )
        runner_margin = observation.get("runner_margin_ratio")
        if runner_margin is not None:
            _require(
                _finite_number(runner_margin) >= DEFAULT_RESIDUAL_POLICY.min_runner_margin,
                "accepted residual regional runner-up margin gate drifted",
            )
    elif status == "identity":
        _require(
            _finite_number(observation.get("zero_loss")) <= DEFAULT_RESIDUAL_POLICY.zero_loss_floor,
            "residual identity observation is above the rigid-residual floor",
        )
        _require(
            _finite_number(observation.get("ux")) == 0.0 and _finite_number(observation.get("uy")) == 0.0,
            "residual identity observation is not an exact no-correction receipt",
        )


def _validate_residual_model_fit(
    runtime_fit: Any,
    recomputed_fit: Any,
    *,
    label: str,
    parameters: tuple[str, ...],
) -> None:
    _require(isinstance(runtime_fit, dict), f"{label} runtime fit is missing")
    _require(isinstance(recomputed_fit, dict), f"{label} recomputed fit is missing")
    _require(
        str(runtime_fit.get("status")) == str(recomputed_fit.get("status")),
        f"{label} fit status does not reproduce from regional receipts",
    )
    if runtime_fit.get("status") != "accepted":
        return
    for parameter in parameters:
        _require(
            _close_number(runtime_fit.get(parameter), recomputed_fit.get(parameter)),
            f"{label} parameter {parameter} does not reproduce from regional receipts",
        )
    if "rank" in runtime_fit:
        _require(
            int(runtime_fit["rank"]) == int(recomputed_fit["rank"]),
            f"{label} fit rank does not reproduce",
        )
    if runtime_fit.get("condition") is not None:
        _require(
            _close_number(runtime_fit.get("condition"), recomputed_fit.get("condition"), atol=1e-7),
            f"{label} fit condition does not reproduce",
        )


def _validate_residual_measurement_receipt(fields: Any, *, label: str) -> bool:
    _require(isinstance(fields, dict), f"{label} residual geometry receipt is missing")
    _require(
        fields.get("policy") == RESIDUAL_GEOMETRY_POLICY_VERSION,
        f"{label} residual geometry policy version drifted",
    )
    _require(fields.get("status") == "measured", f"{label} residual geometry measurement did not complete")
    support = fields.get("support")
    _require(isinstance(support, dict), f"{label} residual geometry support receipt is missing")
    height = int(support.get("height", 0))
    width = int(support.get("width", 0))
    _require(height > 0 and width > 0, f"{label} residual geometry dimensions are invalid")
    _require(
        int(support.get("valid_channels", 0)) <= 24,
        f"{label} residual geometry exceeded the 24-channel bound",
    )
    _require(
        int(support.get("max_support_axis", 0)) == 96,
        f"{label} residual geometry support-axis policy drifted",
    )

    fit_indices_raw = fields.get("fit_indices")
    holdout_indices_raw = fields.get("holdout_indices")
    _require(
        isinstance(fit_indices_raw, list) and isinstance(holdout_indices_raw, list),
        f"{label} residual geometry fit/holdout indices are missing",
    )
    fit_indices = {int(value) for value in fit_indices_raw}
    holdout_indices = {int(value) for value in holdout_indices_raw}
    _require(
        len(fit_indices) >= 2 and len(holdout_indices) >= 2 and fit_indices.isdisjoint(holdout_indices),
        f"{label} residual geometry fit/holdout partition is invalid",
    )
    _require(
        max(fit_indices | holdout_indices) in holdout_indices,
        f"{label} residual geometry last selected prefix frame is not held out",
    )

    tile_bounds = fields.get("tile_bounds")
    _require(isinstance(tile_bounds, dict), f"{label} residual geometry tile map is missing")
    _require(
        set(tile_bounds) == {"TL", "TM", "TR", "ML", "C", "MR", "BL", "BM", "BR"},
        f"{label} residual geometry tile partition drifted",
    )
    for bounds in tile_bounds.values():
        _require(
            isinstance(bounds, list) and len(bounds) == 4,
            f"{label} residual geometry contains malformed tile bounds",
        )
        y0, y1, x0, x1 = (int(value) for value in bounds)
        _require(
            0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width,
            f"{label} residual geometry tile lies outside the target latent",
        )

    observations = fields.get("observations")
    _require(isinstance(observations, list), f"{label} residual regional observations are missing")
    expected_observations = 9 * len(fit_indices | holdout_indices)
    _require(
        len(observations) == expected_observations <= 54,
        f"{label} residual geometry observation count exceeded the bounded prefix/tile budget",
    )
    identities: set[tuple[int, str]] = set()
    for observation in observations:
        _validate_residual_observation(
            observation,
            fit_indices=fit_indices,
            holdout_indices=holdout_indices,
        )
        key = (int(observation["frame_index"]), str(observation["tile_id"]))
        _require(key not in identities, f"{label} residual geometry duplicated a frame/tile observation")
        identities.add(key)

    resource = fields.get("resource")
    _require(isinstance(resource, dict), f"{label} residual geometry resource receipt is missing")
    _require(
        int(resource.get("estimated_cpu_scratch_bytes", 0))
        <= int(resource.get("max_cpu_scratch_bytes", 0))
        <= DEFAULT_RESIDUAL_POLICY.max_cpu_scratch_bytes,
        f"{label} residual geometry exceeded the CPU scratch bound",
    )

    models = fields.get("models")
    _require(isinstance(models, dict), f"{label} residual model receipts are missing")
    recomputed = diagnostic_model_fits(
        observations,
        width=width,
        height=height,
        fit_indices=sorted(fit_indices),
        holdout_indices=sorted(holdout_indices),
    )
    recomputed_fits = recomputed["fits"]
    _validate_residual_model_fit(
        models.get("residual_constant"),
        recomputed_fits["residual_constant"],
        label=f"{label} residual-constant",
        parameters=("bx", "by"),
    )
    _validate_residual_model_fit(
        models.get("horizontal"),
        recomputed_fits["horizontal"],
        label=f"{label} horizontal",
        parameters=("a", "b"),
    )
    diagnostic_fits = models.get("diagnostic_fits")
    _require(isinstance(diagnostic_fits, dict), f"{label} diagnostic model ladder is missing")
    for model_name, parameters in (
        ("axis_scales", ("a", "bx", "e", "by")),
        ("similarity", ("scale_term", "rotation_term", "bx", "by")),
        ("affine", ("a", "h", "bx", "k", "e", "by")),
    ):
        _validate_residual_model_fit(
            diagnostic_fits.get(model_name),
            recomputed_fits[model_name],
            label=f"{label} {model_name}",
            parameters=parameters,
        )
    local_model = diagnostic_fits.get("local_projective")
    _require(
        isinstance(local_model, dict) and local_model.get("status") == "not_implemented",
        f"{label} residual geometry unexpectedly enabled a local/projective optimizer",
    )

    rigid_dx = _finite_number(fields.get("rigid_dx"))
    rigid_dy = _finite_number(fields.get("rigid_dy"))
    replay = replay_horizontal_telemetry(
        observations,
        width=width,
        height=height,
        fit_indices=sorted(fit_indices),
        holdout_indices=sorted(holdout_indices),
        rigid_dx=rigid_dx,
        rigid_dy=rigid_dy,
    )
    runtime_envelope = models.get("deletion_sensitivity_envelope")
    replay_envelope = replay.get("deletion_sensitivity_envelope")
    _require(
        isinstance(runtime_envelope, dict) and isinstance(replay_envelope, dict),
        f"{label} residual deletion envelope is missing",
    )
    _require(
        runtime_envelope.get("status") == replay_envelope.get("status"),
        f"{label} residual deletion envelope status does not reproduce",
    )
    if runtime_envelope.get("status") == "accepted":
        for key in ("a", "b"):
            runtime_range = runtime_envelope.get(key)
            replay_range = replay_envelope.get(key)
            _require(
                isinstance(runtime_range, list)
                and isinstance(replay_range, list)
                and len(runtime_range) == len(replay_range) == 2,
                f"{label} residual deletion {key} envelope is malformed",
            )
            _require(
                all(
                    _close_number(runtime_value, replay_value, atol=1e-8)
                    for runtime_value, replay_value in zip(runtime_range, replay_range, strict=True)
                ),
                f"{label} residual deletion {key} envelope does not reproduce",
            )

    replay_holdout = replay.get("holdout")
    runtime_holdout = models.get("holdout")
    if isinstance(runtime_holdout, dict) or isinstance(replay_holdout, dict):
        _require(
            isinstance(runtime_holdout, dict) and isinstance(replay_holdout, dict),
            f"{label} residual holdout summary does not reproduce",
        )
        for key in (
            "horizontal_rms",
            "horizontal_max",
            "constant_rms",
            "constant_max",
            "improvement_ratio",
        ):
            _require(
                _close_number(runtime_holdout.get(key), replay_holdout.get(key), atol=1e-8),
                f"{label} residual holdout {key} does not reproduce",
            )

    runtime_transform = models.get("transform")
    replay_transform = replay.get("transform")
    if isinstance(runtime_transform, dict) or isinstance(replay_transform, dict):
        _require(
            isinstance(runtime_transform, dict) and isinstance(replay_transform, dict),
            f"{label} residual transform algebra does not reproduce",
        )
        for key in ("a", "b", "forward_sx", "matrix_roundtrip_abs_max"):
            _require(
                _close_number(runtime_transform.get(key), replay_transform.get(key), atol=1e-10),
                f"{label} residual transform {key} does not reproduce",
            )
        for matrix_key in ("forward_matrix", "inverse_matrix"):
            runtime_matrix = runtime_transform.get(matrix_key)
            replay_matrix = replay_transform.get(matrix_key)
            _require(
                isinstance(runtime_matrix, list)
                and isinstance(replay_matrix, list)
                and len(runtime_matrix) == len(replay_matrix) == 3,
                f"{label} residual {matrix_key} is malformed",
            )
            for runtime_row, replay_row in zip(runtime_matrix, replay_matrix, strict=True):
                _require(
                    isinstance(runtime_row, list)
                    and isinstance(replay_row, list)
                    and len(runtime_row) == len(replay_row) == 3
                    and all(
                        _close_number(runtime_value, replay_value, atol=1e-10)
                        for runtime_value, replay_value in zip(runtime_row, replay_row, strict=True)
                    ),
                    f"{label} residual {matrix_key} does not reproduce",
                )
        _require(
            _finite_number(runtime_transform.get("matrix_roundtrip_abs_max")) <= 1e-12,
            f"{label} residual forward/inverse matrices do not round-trip",
        )

    for runtime_key, replay_key in (
        ("forward_scale_envelope", "forward_scale_envelope"),
        ("max_corner_residual", "max_corner_residual"),
        ("candidate_invalid_fraction", "candidate_invalid_fraction"),
    ):
        runtime_value = models.get(runtime_key)
        replay_value = replay.get(replay_key)
        if runtime_value is None and replay_value is None:
            continue
        if isinstance(runtime_value, list) or isinstance(replay_value, list):
            _require(
                isinstance(runtime_value, list)
                and isinstance(replay_value, list)
                and len(runtime_value) == len(replay_value)
                and all(
                    _close_number(left, right, atol=1e-8)
                    for left, right in zip(runtime_value, replay_value, strict=True)
                ),
                f"{label} residual {runtime_key} does not reproduce",
            )
        else:
            _require(
                _close_number(runtime_value, replay_value, atol=1e-8),
                f"{label} residual {runtime_key} does not reproduce",
            )

    accepted_observations = [
        observation
        for observation in observations
        if observation.get("status") == "accepted"
        and _finite_number(observation.get("uncertainty_half_width_x"))
        <= DEFAULT_RESIDUAL_POLICY.max_uncertainty_half_width
        and _finite_number(observation.get("uncertainty_half_width_y"))
        <= DEFAULT_RESIDUAL_POLICY.max_uncertainty_half_width
    ]
    required_tiles = {"TL", "TM", "TR", "BL", "BM", "BR"}
    last_frame = max(fit_indices | holdout_indices)
    last_ids = {
        str(observation["tile_id"])
        for observation in accepted_observations
        if int(observation["frame_index"]) == last_frame
    }
    _require(
        bool(models.get("last_holdout_global_support")) == required_tiles.issubset(last_ids),
        f"{label} residual last-holdout support classification does not reproduce",
    )

    horizontal = models.get("horizontal")
    eligible = bool(models.get("eligible"))
    failures = models.get("eligibility_failures")
    _require(isinstance(failures, list), f"{label} residual eligibility failures are missing")
    if eligible:
        _require(not failures, f"{label} residual geometry claims eligibility with recorded failures")
        _require(
            isinstance(horizontal, dict) and horizontal.get("status") == "accepted",
            f"{label} residual geometry claims eligibility without a horizontal fit",
        )
        transform = models.get("transform")
        _require(isinstance(transform, dict), f"{label} eligible horizontal fit lacks transform algebra")
        a = _finite_number(transform.get("a"))
        b = _finite_number(transform.get("b"))
        _require(
            _close_number(a, horizontal.get("a")) and _close_number(b, horizontal.get("b")),
            f"{label} transform algebra does not match its horizontal fit",
        )
        forward_scale = _finite_number(transform.get("forward_sx"))
        _require(
            DEFAULT_RESIDUAL_POLICY.min_forward_scale <= forward_scale <= DEFAULT_RESIDUAL_POLICY.max_forward_scale,
            f"{label} eligible horizontal forward scale exceeds the policy bound",
        )
        _require(
            _finite_number(models.get("max_corner_residual")) <= DEFAULT_RESIDUAL_POLICY.max_residual_displacement,
            f"{label} eligible horizontal corner displacement exceeds the policy bound",
        )
        scale_envelope = models.get("forward_scale_envelope")
        _require(
            isinstance(scale_envelope, list)
            and len(scale_envelope) == 2
            and DEFAULT_RESIDUAL_POLICY.min_forward_scale
            <= _finite_number(scale_envelope[0])
            <= _finite_number(scale_envelope[1])
            <= DEFAULT_RESIDUAL_POLICY.max_forward_scale,
            f"{label} eligible horizontal deletion-scale envelope exceeds the policy bound",
        )
        _require(
            _finite_number(models.get("candidate_invalid_fraction")) <= DEFAULT_RESIDUAL_POLICY.max_invalid_fraction,
            f"{label} eligible horizontal candidate invalid area exceeds the policy bound",
        )
        _require(
            models.get("last_holdout_global_support") is True,
            f"{label} eligible horizontal fit lacks last-frame global support",
        )
        direct = models.get("direct_feature_checks")
        parity = models.get("parity_checks")
        _require(
            isinstance(direct, list)
            and bool(direct)
            and all(isinstance(item, dict) and item.get("nondegrading") is True for item in direct),
            f"{label} eligible horizontal fit failed direct feature checks",
        )
        _require(
            isinstance(parity, list)
            and len(parity) == 4
            and all(
                isinstance(item, dict) and item.get("status") == "accepted" and item.get("nondegrading") is True
                for item in parity
            ),
            f"{label} eligible horizontal fit failed H3 parity checks",
        )
    return eligible


def _validate_residual_geometry(
    window: list[dict[str, Any]],
    *,
    expected_mode: str | None,
    expected_result: str | None,
) -> tuple[str | None, str | None, bool, bool, str | None]:
    if expected_mode is None and expected_result is None:
        return None, None, False, False, None
    _require(expected_mode in {"off", "measure"}, f"unsupported residual mode {expected_mode!r}")
    _require(
        expected_result in {"off", "not-evaluated", "measured-only"},
        f"unsupported residual result {expected_result!r}",
    )
    frame_receipts = [_event_fields(event) for event in window if _event_kind(event) == "partitioned_frame_gauge"]
    _require(len(frame_receipts) == 1, "residual gate requires exactly one partitioned frame-gauge receipt")
    frame_receipt = frame_receipts[0]
    receipt = frame_receipt.get("residual_geometry")
    _require(isinstance(receipt, dict), "partitioned frame-gauge receipt is missing residual geometry")
    _require(
        receipt.get("policy") == RESIDUAL_GEOMETRY_POLICY_VERSION,
        "residual geometry policy version drifted",
    )
    mode = str(receipt.get("requested_mode", ""))
    _require(mode == expected_mode, f"residual mode {mode!r} != expected {expected_mode!r}")
    _require(receipt.get("decision") == "not_evaluated", "measurement milestone made an application decision")
    _require(receipt.get("applied") is False, "measurement milestone applied a residual transform")
    _require(
        frame_receipt.get("residual_geometry_telemetry_bytes", 2**31) <= 128 * 1024,
        "residual geometry telemetry exceeded the 128 KiB bound",
    )

    stage_events = [
        _event_fields(event) for event in window if _event_kind(event) == "partitioned_residual_geometry_stage"
    ]
    evidence_events = [
        _event_fields(event) for event in window if _event_kind(event) == "partitioned_residual_geometry_evidence"
    ]
    if mode == "off":
        _require(expected_result == "off", "residual OFF control expected a non-OFF result")
        _require(receipt.get("measured") is False, "residual OFF control ran the regional estimator")
        _require(not stage_events, "residual OFF control emitted measurement-stage receipts")
        _require(not evidence_events, "residual OFF control exported measurement tensor evidence")
        return mode, "off", True, False, None

    frame_result = str(frame_receipt.get("result", ""))
    if frame_result != "accepted":
        _require(
            expected_result == "not-evaluated",
            "residual measurement was expected despite rigid v2 not accepting",
        )
        _require(receipt.get("measured") is False, "residual estimator ran before rigid-v2 acceptance")
        _require(receipt.get("measurement_status") == "not_evaluated", "residual not-evaluated status drifted")
        _require(not stage_events and not evidence_events, "not-evaluated residual arm emitted measurement evidence")
        return mode, "not-evaluated", True, False, None

    _require(expected_result == "measured-only", "accepted rigid-v2 arm expected the wrong residual result")
    _require(receipt.get("measured") is True, "residual measure arm did not run after rigid-v2 acceptance")
    _require(
        receipt.get("measurement_status") == "measured",
        "residual measure arm did not complete its bounded regional measurement",
    )
    _require(receipt.get("final_path") == "rigid_v2", "residual measurement changed the final correction path")
    video_eligible = _validate_residual_measurement_receipt(receipt.get("video"), label="video")
    guidance_mode = str(frame_receipt.get("guidance_mode", "off"))
    guidance_receipt = receipt.get("guidance")
    if guidance_mode == "off":
        _require(
            isinstance(guidance_receipt, dict) and guidance_receipt.get("status") == "off",
            "guidance-off residual arm emitted a guidance fit",
        )
    else:
        _validate_residual_measurement_receipt(guidance_receipt, label="guidance")

    boundary_regional = receipt.get("boundary_regional")
    _require(
        isinstance(boundary_regional, dict)
        and boundary_regional.get("policy") == "paired_prefix_residual_boundary_regions_v1",
        "residual measure arm is missing regional native-motion controls",
    )
    tiles = boundary_regional.get("tiles")
    _require(
        isinstance(tiles, dict) and len(tiles) == 9,
        "residual regional boundary control did not cover all nine disjoint tiles",
    )
    for tile in tiles.values():
        _require(isinstance(tile, dict), "residual regional boundary tile is malformed")
        for variant in (
            "native",
            "exact_unregistered",
            "transformed_native",
            "exact_rigid_pre_dc",
            "exact_rigid_post_dc",
        ):
            variant_receipt = tile.get(variant)
            _require(isinstance(variant_receipt, dict), "residual regional boundary variant is missing")
            _finite_number(variant_receipt.get("dx"))
            _finite_number(variant_receipt.get("dy"))
            _finite_number(variant_receipt.get("response"))
        _finite_number(tile.get("pre_dc_native_error"))
        _finite_number(tile.get("post_dc_native_error"))

    required_stages = {
        "learned_native_same_time",
        "learned_rigid_aligned_same_time",
        "exact_restored_pre_high_dc",
        "final_post_high_internal_clean",
        "final_post_high_caller_domain",
    }
    if guidance_mode != "off":
        required_stages |= {"guidance_native_same_time", "guidance_rigid_aligned_same_time"}
    by_stage = {str(event.get("stage")): event for event in stage_events}
    _require(required_stages.issubset(by_stage), "residual common-domain stage evidence is incomplete")
    session_ids = {str(event.get("session_id")) for event in stage_events}
    chunk_ids = {str(event.get("chunk_id")) for event in stage_events}
    _require(len(session_ids) == 1 and len(chunk_ids) == 1, "residual stage invocation identity drifted")
    for stage, stage_receipt in by_stage.items():
        _require(
            stage_receipt.get("policy") == RESIDUAL_GEOMETRY_POLICY_VERSION,
            f"residual stage {stage} policy version drifted",
        )
        _require(_sha256(stage_receipt.get("tensor_sha256")), f"residual stage {stage} lacks a tensor digest")
        domain = stage_receipt.get("domain")
        if stage == "final_post_high_caller_domain":
            _require(domain == "caller_output_latent", "caller-domain final receipt domain drifted")
        else:
            _require(domain == "model_internal_clean", f"residual stage {stage} is not in the common internal domain")

    final_internal = by_stage["final_post_high_internal_clean"]
    _require(
        final_internal.get("owner_before") == "authoritative_exact_prefix_E",
        "post-high internal comparison did not preserve authoritative exact-prefix ownership",
    )

    _require(len(evidence_events) == 1, "residual measure arm must emit exactly one evidence bundle receipt")
    evidence = evidence_events[0]
    _require(evidence.get("policy") == RESIDUAL_GEOMETRY_POLICY_VERSION, "residual evidence policy drifted")
    _require(evidence.get("requested_mode") == "measure", "residual evidence mode drifted")
    _require(evidence.get("decision") == "not_evaluated", "residual evidence made an application decision")
    _require(evidence.get("applied") is False, "residual evidence claims a transform was applied")
    _require(evidence.get("horizontal_application_enabled") is False, "horizontal correction was enabled prematurely")
    _require(evidence.get("status") == "exported", "formal residual gate did not export raw tensor evidence")
    _require(evidence.get("extra_vae_calls") == 0, "residual evidence export added a VAE call")
    bundle = evidence.get("bundle")
    _require(isinstance(bundle, str) and bool(bundle), "residual evidence bundle identity is missing")
    return mode, "measured-only", True, video_eligible, bundle


def compare_residual_measurement_pair(
    control_metrics: dict[str, Any],
    measure_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare matched rigid-v2 OFF and residual MEASURE metrics.

    This comparison is deliberately limited to evidence already emitted by both
    arms. Exact tensor bytes from the MEASURE bundle remain separate hardware
    evidence; the OFF arm is not instrumented with extra tensor readbacks.
    """

    for label, metrics in (("control", control_metrics), ("measure", measure_metrics)):
        _require(isinstance(metrics, dict), f"{label} pair metrics root must be an object")
        _require(isinstance(metrics.get("events"), list), f"{label} pair metrics events are missing")
    control = _latest_partitioned_window(control_metrics["events"])
    measure = _latest_partitioned_window(measure_metrics["events"])

    def one(window: list[dict[str, Any]], kind: str) -> dict[str, Any]:
        rows = [_event_fields(event) for event in window if _event_kind(event) == kind]
        _require(len(rows) == 1, f"matched pair requires exactly one {kind} receipt per arm")
        return rows[0]

    control_frame = one(control, "partitioned_frame_gauge")
    measure_frame = one(measure, "partitioned_frame_gauge")
    _require(
        control_frame.get("mode") == measure_frame.get("mode") == "on"
        and control_frame.get("result") == measure_frame.get("result") == "accepted",
        "matched residual pair requires accepted rigid-v2 ON transactions in both arms",
    )
    control_residual = control_frame.get("residual_geometry")
    measure_residual = measure_frame.get("residual_geometry")
    _require(
        isinstance(control_residual, dict)
        and control_residual.get("requested_mode") == "off"
        and control_residual.get("applied") is False,
        "matched control is not residual OFF",
    )
    _require(
        isinstance(measure_residual, dict)
        and measure_residual.get("requested_mode") == "measure"
        and measure_residual.get("measured") is True
        and measure_residual.get("applied") is False
        and measure_residual.get("final_path") == "rigid_v2",
        "matched candidate is not residual measurement-only",
    )

    exact_fields = (
        "policy_version",
        "split_coordinate",
        "video_dx",
        "video_dy",
        "guidance_dx",
        "guidance_dy",
        "exact_prefix_sha256",
        "guidance_mode",
        "provider_api_version",
        "provider_kind",
        "provider_model_name",
        "deterministic_noise_seed",
        "deterministic_noise_seed_offset",
        "mask_classification",
    )
    for field in exact_fields:
        _require(
            control_frame.get(field) == measure_frame.get(field),
            f"matched residual pair differs in frame-gauge field {field}",
        )

    control_transfer = one(control, "partitioned_transfer")
    measure_transfer = one(measure, "partitioned_transfer")
    transfer_fields = (
        "provider_api_version",
        "provider_kind",
        "model_name",
        "source_hw",
        "target_hw",
        "temporal_length",
        "suffix_dc_bridge_policy",
        "suffix_dc_bridge_corrected_tokens",
        "suffix_dc_bridge_state_mapping",
        "authoritative_target_prefix_restored",
        "splice_clean_source",
    )
    for field in transfer_fields:
        _require(
            control_transfer.get(field) == measure_transfer.get(field),
            f"matched residual pair differs in transfer field {field}",
        )

    control_noise = one(control, "handoff_transfer_wall")
    measure_noise = one(measure, "handoff_transfer_wall")
    _require(
        control_noise.get("protected_video_noise_exact") is True
        and measure_noise.get("protected_video_noise_exact") is True,
        "matched residual pair does not preserve protected video noise",
    )

    def call_sequence(window: list[dict[str, Any]]) -> tuple[tuple[str, bool], ...]:
        return tuple(
            (str(_event_fields(event).get("stage")), bool(_event_fields(event).get("actual")))
            for event in window
            if _event_kind(event) == "model_call"
        )

    control_calls = call_sequence(control)
    measure_calls = call_sequence(measure)
    _require(control_calls == measure_calls, "matched residual pair changed logical/actual H3 call topology")

    control_handoff = one(control, "handoff_complete")
    measure_handoff = one(measure, "handoff_complete")
    for field in (
        "sampler_invocation_count",
        "history_boundary_count",
        "high_stage_model_calls",
        "high_stage_first_call_actual",
        "input_mode",
        "transfer_mode",
    ):
        _require(
            control_handoff.get(field) == measure_handoff.get(field),
            f"matched residual pair differs in no-extra-work field {field}",
        )

    control_complete = one(control, "partitioned_exact_prefix_complete")
    measure_complete = one(measure, "partitioned_exact_prefix_complete")
    _require(
        control_complete.get("final_prefix_exact") is True and measure_complete.get("final_prefix_exact") is True,
        "matched residual pair lost exact final prefix ownership",
    )

    return {
        "status": "matched",
        "control_residual_mode": "off",
        "measure_residual_mode": "measure",
        "rigid_policy": control_frame.get("policy_version"),
        "exact_prefix_sha256": control_frame.get("exact_prefix_sha256"),
        "video_dx": control_frame.get("video_dx"),
        "video_dy": control_frame.get("video_dy"),
        "guidance_dx": control_frame.get("guidance_dx"),
        "guidance_dy": control_frame.get("guidance_dy"),
        "model_call_topology": control_calls,
        "protected_video_noise_exact": True,
        "final_prefix_exact": True,
        "numerical_output_identity": (
            "established structurally/unit-wise; formal hardware tensor/media comparison "
            "uses the retained MEASURE evidence bundle plus matched workflow outputs"
        ),
    }


def _normalize_auto_strength_report(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeGateError(f"invalid DoRA auto-strength report JSON: {exc}") from exc
    _require(isinstance(value, dict), "DoRA auto-strength report must be a JSON object")
    return value


def _validate_auto_strength_off(
    reports: Iterable[dict[str, Any] | str] | None,
    *,
    required: bool,
    expected_digests: Iterable[str] | None,
) -> tuple[bool, tuple[str, ...]]:
    normalized = [] if reports is None else [_normalize_auto_strength_report(value) for value in reports]
    if required:
        _require(bool(normalized), "hardware evidence requires at least one resolved DoRA auto-strength report")
    digests: list[str] = []
    for report in normalized:
        _require(report.get("schema") == 1, "unsupported DoRA auto-strength report schema")
        _require(
            report.get("kind") == "dora_power_lora_auto_strength_stack_report",
            "unexpected DoRA auto-strength report kind",
        )
        _require(
            report.get("auto_strength_enabled") is False,
            "DoRA auto-strength was not resolved OFF",
        )
        rows = report.get("rows")
        _require(isinstance(rows, list), "DoRA auto-strength report rows are missing")
        for row in rows:
            _require(isinstance(row, dict), "DoRA auto-strength report contains a malformed row")
            status = str(row.get("status", ""))
            _require(
                status not in {"analyzed", "auto_strength_skipped"},
                f"DoRA row {row.get('row_index')} executed or attempted auto-strength analysis",
            )
            if status == "applied_without_auto_strength":
                _require(
                    row.get("report") is None,
                    f"DoRA row {row.get('row_index')} retained an auto-strength analysis report while OFF",
                )
        canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digests.append(hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    observed = tuple(sorted(digests))
    if expected_digests is not None:
        expected = tuple(sorted(str(value) for value in expected_digests))
        _require(
            observed == expected,
            "DoRA auto-strength report identity differs from the matched control arm",
        )
    return bool(normalized) and all(report.get("auto_strength_enabled") is False for report in normalized), observed


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
    expected_frame_gauge_mode: str | None = None,
    expected_residual_mode: str | None = None,
    expected_residual_result: str | None = None,
    auto_strength_reports: Iterable[dict[str, Any] | str] | None = None,
    require_auto_strength_off: bool = False,
    expected_auto_strength_digests: Iterable[str] | None = None,
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

    auto_strength_verified_off, auto_strength_report_digests = _validate_auto_strength_off(
        auto_strength_reports,
        required=require_auto_strength_off,
        expected_digests=expected_auto_strength_digests,
    )
    (
        frame_gauge_mode,
        frame_gauge_result,
        frame_gauge_verified,
        frame_gauge_video_dx,
        frame_gauge_video_dy,
        frame_gauge_guidance_dx,
        frame_gauge_guidance_dy,
    ) = _validate_frame_gauge(
        window,
        expected_mode=expected_frame_gauge_mode,
    )
    (
        residual_geometry_mode,
        residual_geometry_result,
        residual_geometry_verified,
        residual_geometry_horizontal_eligible,
        residual_geometry_evidence_bundle,
    ) = _validate_residual_geometry(
        window,
        expected_mode=expected_residual_mode,
        expected_result=expected_residual_result,
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
        AUDIO_OVERLAP_MARKER in log_text
        and re.search(
            r"partitioned audio guided overlap mode=\S+ ticks=4\b applied=True\b",
            log_text,
        )
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
        frame_gauge_mode=frame_gauge_mode,
        frame_gauge_result=frame_gauge_result,
        frame_gauge_verified=frame_gauge_verified,
        frame_gauge_video_dx=frame_gauge_video_dx,
        frame_gauge_video_dy=frame_gauge_video_dy,
        frame_gauge_guidance_dx=frame_gauge_guidance_dx,
        frame_gauge_guidance_dy=frame_gauge_guidance_dy,
        residual_geometry_mode=residual_geometry_mode,
        residual_geometry_result=residual_geometry_result,
        residual_geometry_verified=residual_geometry_verified,
        residual_geometry_horizontal_eligible=residual_geometry_horizontal_eligible,
        residual_geometry_evidence_bundle=residual_geometry_evidence_bundle,
        auto_strength_verified_off=auto_strength_verified_off,
        auto_strength_report_digests=auto_strength_report_digests,
    )


__all__ = [
    "AUDIO_POSITION_DOMAIN_LEGACY",
    "AUDIO_POSITION_DOMAIN_SOURCE",
    "PARTITIONED_SOL_ABI",
    "RuntimeGateError",
    "RuntimeGateReport",
    "compare_residual_measurement_pair",
    "validate_partitioned_runtime_evidence",
]
