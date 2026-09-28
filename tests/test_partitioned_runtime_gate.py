from __future__ import annotations

import json

import pytest
import torch

from h3_flow_regenerate.boundary_content_diagnostics import measure_learned_transfer_residual_diagnostic
from h3_flow_regenerate.frame_gauge import FRAME_GAUGE_POLICY_VERSION
from h3_flow_regenerate.partitioned_runtime_gate import (
    AUDIO_POSITION_DOMAIN_LEGACY,
    AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_SOL_ABI,
    RuntimeGateError,
    compare_residual_measurement_pair,
    validate_partitioned_runtime_evidence,
)
from h3_flow_regenerate.residual_geometry import measure_residual_geometry


def _event(kind: str, **fields):
    return {"kind": kind, "fields": fields}


def _metrics() -> dict:
    events = [
        _event(
            "partitioned_stage_plan",
            input_mode="partitioned_exact_prefix",
            prefix_exact_latent_resized_for_transformer=False,
            deprecated_mixed_grid_contract_active=False,
        ),
        _event(
            "partitioned_exact_prefix_transformer",
            vdn_external_sequence_api=4,
            prefix_exact_latent_resized=False,
            prefix_target_grid_rope=True,
            suffix_source_grid_rope=True,
            low_suffix_real_latent=True,
            deprecated_mixed_grid_contract_active=False,
        ),
        _event("model_call", stage="low", actual=True),
        _event("model_call", stage="low", actual=False),
        _event("model_call", stage="probe", actual=True),
        _event(
            "partitioned_transfer",
            learned_transfer_performed=True,
            upscaler_prefix_output_discarded=True,
            authoritative_target_prefix_restored=True,
            target_prefix_resized_for_transformer=False,
            deprecated_mixed_grid_repairs_applied=False,
        ),
        _event(
            "handoff_transfer_wall",
            protected_video_noise_exact=True,
        ),
        _event("model_call", stage="high", actual=True),
        _event("model_call", stage="high", actual=False),
        _event(
            "partitioned_exact_prefix_complete",
            final_prefix_exact=True,
            high_stage_first_call_actual=True,
            deprecated_mixed_grid_contract_active=False,
        ),
        _event(
            "handoff_complete",
            input_mode="partitioned_exact_prefix",
            exact_probe_performed=True,
            sampler_invocation_count=3,
            history_boundary_count=2,
            high_stage_first_call_actual=True,
        ),
    ]
    return {
        "schema_version": 1,
        "counters": {
            "partitioned_transformer_calls": 2,
            "partitioned_attention_provider_creations": 2,
            "partitioned_attention_provider_reuses": 1,
            "partitioned_attention_equivalent_provider_rebindings": 1,
            "partitioned_attention_inherited_provider_transitions": 0,
        },
        "events": events,
    }


def _sol_record(**overrides) -> dict:
    record = {
        "success": True,
        "external_mixed_sol_calls": 0,
        "external_mixed_measure_calls": 0,
        "external_mixed_weighted_measure_calls": 0,
        "vdn_square_expanded_calls": 0,
        "vdn_square_requested_rows": 0,
        "vdn_square_kernel_rows": 0,
        "vdn_mapped_sol_calls": 3,
        "vdn_rectangular_sol_calls": 3,
        "vdn_requested_q_rows": 192,
        "vdn_kernel_q_rows": 192,
        "arithmetic_gates": [
            {
                "route": PARTITIONED_SOL_ABI,
                "mapped_neighbor_abi": True,
                "key_measure_bias": True,
                "max_abs": 0.001,
            }
        ],
    }
    record.update(overrides)
    return record


def _log(record: dict | None = None) -> str:
    record = _sol_record() if record is None else record
    return "\n".join(
        (
            "partitioned exact-prefix: grouped VDN softmax active; variable-grid linear complement active",
            "partitioned audio guided overlap mode=model_timestep_only ticks=4 applied=True "
            "source=diagnostic_node reason=guided_prefix exact_prefix=4 "
            "ramp=[0.203125, 0.40234375, 0.6015625, 0.80078125] "
            "final_exact_restore=true",
            "INFO comfy.sol_h3 Sol-H3 " + json.dumps(record, sort_keys=True),
        )
    )


def test_partitioned_runtime_gate_accepts_complete_evidence():
    report = validate_partitioned_runtime_evidence(
        _metrics(),
        _log(),
        expected_logical=5,
        expected_actual=3,
        expected_forecast=2,
    )

    assert report.logical_calls == 5
    assert report.actual_calls == 3
    assert report.forecast_calls == 2
    assert report.probe_logical == 1
    assert report.probe_actual == 1
    assert report.high_logical == 2
    assert report.high_actual == 1
    assert report.partitioned_provider_creations == 2
    assert report.partitioned_provider_reuses == 1
    assert report.partitioned_provider_equivalent_rebindings == 1
    assert report.partitioned_provider_semantic_transitions == 0
    assert report.sol_mapped_calls == 3
    assert report.sol_rectangular_calls == 3
    assert report.sol_requested_q_rows == report.sol_kernel_q_rows == 192
    assert report.vdn_variable_grid_linear_active is True
    assert report.audio_guided_overlap_active is True
    assert report.audio_guided_overlap_mode == "model_timestep_only"
    assert report.audio_guided_overlap_ticks == 4


def _accepted_post_high_video_receipt(**overrides):
    receipt = {
        "policy": "post_high_vertical_residual_release_v1",
        "eligible": True,
        "accepted": True,
        "applied": True,
        "output_mutated": True,
        "authoritative_prefix_modified": False,
        "audio_modified": False,
        "safe_video_topology": True,
        "reason": "accepted_bounded_vertical_release",
        "horizontal_application_enabled": False,
        "operator_dx_cells": 0.0,
        "caller_audio_exact": True,
        "caller_prefix_exact": True,
        "internal_audio_exact": True,
        "internal_prefix_exact": True,
        "canonical_candidate_accepted": True,
        "selected_dy_cells": -0.1143016587,
        "correction_magnitude_cells": 0.1143016587,
        "max_induced_step_cells": 0.040411,
        "release_weights": [1.0, 0.8535533905932737, 0.5, 0.14644660940672627],
        "corrected_tokens": 4,
        "caller_reentry_validation": {
            "before_mean_abs_error_cells": 0.14,
            "after_mean_abs_error_cells": 0.04,
            "mean_improvement_ratio": 0.7142857143,
            "no_material_roi_regression": True,
            "horizontal_phase_stable": True,
        },
        "extra_h3_nfe": 0,
        "extra_sampler_lifetimes": 0,
        "extra_history_boundaries": 0,
        "extra_provider_calls": 0,
        "extra_vae_calls": 0,
    }
    receipt.update(overrides)
    return receipt


def test_partitioned_runtime_gate_accepts_bounded_post_high_video_repair():
    metrics = _metrics()
    metrics["events"].insert(
        -2,
        _event(
            "partitioned_post_high_video_residual_repair",
            **_accepted_post_high_video_receipt(),
        ),
    )

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_logical=5,
        expected_actual=3,
        expected_forecast=2,
    )
    assert report.logical_calls == 5


def test_partitioned_runtime_gate_rejects_malformed_applied_post_high_video_repair():
    metrics = _metrics()
    metrics["events"].insert(
        -2,
        _event(
            "partitioned_post_high_video_residual_repair",
            **_accepted_post_high_video_receipt(
                canonical_candidate_accepted=False,
            ),
        ),
    )
    with pytest.raises(RuntimeGateError, match="round-trip validation"):
        validate_partitioned_runtime_evidence(metrics, _log())


def test_partitioned_runtime_gate_accepts_known_fail_closed_post_high_video_repair():
    metrics = _metrics()
    metrics["events"].insert(
        -2,
        _event(
            "partitioned_post_high_video_residual_repair",
            policy="post_high_vertical_residual_release_v1",
            eligible=False,
            accepted=False,
            applied=False,
            output_mutated=False,
            authoritative_prefix_modified=False,
            audio_modified=False,
            reason="target_high_vertical_delta_not_coherent",
            extra_h3_nfe=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
        ),
    )

    report = validate_partitioned_runtime_evidence(metrics, _log())
    assert report.logical_calls == 5


def test_partitioned_runtime_gate_rejects_unapplied_or_wrong_width_audio_overlap():
    with pytest.raises(RuntimeGateError, match="latest partitioned audio guided overlap"):
        validate_partitioned_runtime_evidence(
            _metrics(),
            _log().replace("ticks=4 applied=True", "ticks=3 applied=True"),
        )
    with pytest.raises(RuntimeGateError, match="latest partitioned audio guided overlap"):
        validate_partitioned_runtime_evidence(
            _metrics(),
            _log().replace("ticks=4 applied=True", "ticks=4 applied=False"),
        )


def test_partitioned_runtime_gate_uses_latest_audio_overlap_receipt_and_exact_expectations():
    stale = _log().splitlines()[1]
    current = stale.replace(
        "mode=model_timestep_only ticks=4",
        "mode=sampler_mask ticks=16",
    )
    log_text = "\n".join(
        (
            "partitioned exact-prefix: grouped VDN softmax active; variable-grid linear complement active",
            stale,
            current,
            "INFO comfy.sol_h3 Sol-H3 " + json.dumps(_sol_record(), sort_keys=True),
        )
    )

    with pytest.raises(RuntimeGateError, match="latest partitioned audio guided overlap"):
        validate_partitioned_runtime_evidence(_metrics(), log_text)

    exact_log = _log().replace(
        "mode=model_timestep_only",
        "mode=sampler_mask_exact_timestep",
    )
    report = validate_partitioned_runtime_evidence(
        _metrics(),
        exact_log,
        expected_audio_guided_overlap_mode="sampler_mask_exact_timestep",
        expected_audio_guided_overlap_ticks=4,
    )
    assert report.audio_guided_overlap_mode == "sampler_mask_exact_timestep"
    assert report.audio_guided_overlap_ticks == 4

    with pytest.raises(RuntimeGateError, match="mode differs from expectation"):
        validate_partitioned_runtime_evidence(
            _metrics(),
            _log(),
            expected_audio_guided_overlap_mode="sampler_mask_exact_timestep",
        )
    with pytest.raises(RuntimeGateError, match="width differs from expectation"):
        validate_partitioned_runtime_evidence(
            _metrics(),
            _log(),
            expected_audio_guided_overlap_ticks=16,
        )


def test_partitioned_runtime_gate_rejects_provider_recreation_on_every_call():
    metrics = _metrics()
    metrics["counters"].update(
        partitioned_attention_provider_creations=3,
        partitioned_attention_provider_reuses=0,
        partitioned_attention_equivalent_provider_rebindings=0,
    )
    with pytest.raises(RuntimeGateError, match="provider identity changed on every logical low/probe call"):
        validate_partitioned_runtime_evidence(metrics, _log())


def test_partitioned_runtime_gate_rejects_provider_binding_accounting_drift():
    metrics = _metrics()
    metrics["counters"]["partitioned_attention_provider_reuses"] = 0
    with pytest.raises(RuntimeGateError, match="provider binding accounting"):
        validate_partitioned_runtime_evidence(metrics, _log())


def test_partitioned_runtime_gate_distinguishes_forecast_binding_from_actual_transformer_execution():
    metrics = _metrics()

    # Three logical low/probe calls bind a provider, but one low call is a
    # Spectrum forecast and therefore only two of them execute H3.
    assert metrics["counters"]["partitioned_transformer_calls"] == 2
    assert metrics["counters"]["partitioned_attention_provider_creations"] == 2
    assert metrics["counters"]["partitioned_attention_provider_reuses"] == 1

    report = validate_partitioned_runtime_evidence(metrics, _log())
    assert report.low_logical + report.probe_logical == 3
    assert report.low_actual + report.probe_actual == 2


def test_partitioned_runtime_gate_rejects_transformer_actual_accounting_drift():
    metrics = _metrics()
    metrics["counters"]["partitioned_transformer_calls"] = 3
    with pytest.raises(RuntimeGateError, match="transformer-call accounting"):
        validate_partitioned_runtime_evidence(metrics, _log())


def test_partitioned_runtime_gate_rejects_square_q_expansion():
    with pytest.raises(RuntimeGateError, match="square-Q expansion"):
        validate_partitioned_runtime_evidence(
            _metrics(),
            _log(_sol_record(vdn_square_expanded_calls=1)),
        )


def test_partitioned_runtime_gate_rejects_missing_combined_sm120_gate():
    record = _sol_record()
    record["arithmetic_gates"] = [
        {
            "route": PARTITIONED_SOL_ABI,
            "mapped_neighbor_abi": True,
            "key_measure_bias": False,
        }
    ]
    with pytest.raises(RuntimeGateError, match="mapped-neighbor \+ key-measure"):
        validate_partitioned_runtime_evidence(_metrics(), _log(record))


def test_partitioned_runtime_gate_uses_latest_partitioned_window():
    metrics = _metrics()
    stale = _metrics()["events"]
    stale[0] = _event(
        "partitioned_stage_plan",
        input_mode="partitioned_exact_prefix",
        prefix_exact_latent_resized_for_transformer=True,
        deprecated_mixed_grid_contract_active=False,
    )
    metrics["events"] = [*stale, *metrics["events"]]

    report = validate_partitioned_runtime_evidence(metrics, _log())

    assert report.logical_calls == 5


def _candidate_metrics() -> dict:
    metrics = _metrics()
    events = metrics["events"]
    events[0]["fields"]["audio_position_domain"] = AUDIO_POSITION_DOMAIN_SOURCE
    transformer = events[1]["fields"]
    transformer.update(
        audio_position_domain=AUDIO_POSITION_DOMAIN_SOURCE,
        audio_position_policy_active=True,
        audio_position_policy_signature=(
            "h3_flow_partitioned_position_policy_v1",
            AUDIO_POSITION_DOMAIN_SOURCE,
            17,
        ),
        audio_position_stage_owner_generation=17,
        audio_position_temporal_equal=True,
        audio_position_non_audio_before_digest="a" * 64,
        audio_position_non_audio_after_digest="a" * 64,
        prefix_rope_position_digest="b" * 64,
        suffix_rope_position_digest="c" * 64,
        position_digest="d" * 64,
    )
    probe_transformer = _event(
        "partitioned_exact_prefix_transformer",
        **{
            **transformer,
            "audio_position_policy_signature": (
                "h3_flow_partitioned_position_policy_v1",
                AUDIO_POSITION_DOMAIN_SOURCE,
                23,
            ),
            "audio_position_stage_owner_generation": 23,
        },
    )
    probe_index = next(
        index
        for index, event in enumerate(events)
        if event["kind"] == "model_call" and event["fields"].get("stage") == "probe"
    )
    events.insert(probe_index, probe_transformer)
    events.insert(
        3,
        _event(
            "partitioned_audio_position_domain_verified",
            mode=AUDIO_POSITION_DOMAIN_SOURCE,
            wrapper_entries=3,
            actual_block0_calls=2,
            model_timestep_override_calls=2,
            target_audio_rows_only=True,
            sampler_mask_mutated=False,
            fail_closed=True,
        ),
    )
    events.insert(
        -2,
        _event(
            "partitioned_audio_position_candidate_integrity",
            mode=AUDIO_POSITION_DOMAIN_SOURCE,
            low_mask_digest="e" * 64,
            high_mask_digest="f" * 64,
            final_exact_video_prefix=True,
            final_exact_audio_prefix=True,
            sampler_masks_unchanged=True,
        ),
    )
    return metrics


def test_partitioned_runtime_gate_accepts_source_carrier_candidate_receipts():
    report = validate_partitioned_runtime_evidence(
        _candidate_metrics(),
        _log(),
        expected_audio_position_domain=AUDIO_POSITION_DOMAIN_SOURCE,
    )

    assert report.audio_position_domain == AUDIO_POSITION_DOMAIN_SOURCE
    assert report.audio_position_candidate_verified is True
    assert report.audio_position_candidate_block0_calls == 2
    assert report.audio_position_model_timestep_override_calls == 2


def test_partitioned_runtime_gate_keeps_legacy_gate_backward_compatible():
    report = validate_partitioned_runtime_evidence(
        _metrics(),
        _log(),
        expected_audio_position_domain=AUDIO_POSITION_DOMAIN_LEGACY,
    )

    assert report.audio_position_domain == AUDIO_POSITION_DOMAIN_LEGACY
    assert report.audio_position_candidate_verified is False


def test_partitioned_runtime_gate_rejects_candidate_position_or_av_receipt_drift():
    metrics = _candidate_metrics()
    metrics["events"][1]["fields"]["audio_position_non_audio_after_digest"] = "0" * 64
    with pytest.raises(RuntimeGateError, match="non-target-audio"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_audio_position_domain=AUDIO_POSITION_DOMAIN_SOURCE,
        )

    metrics = _candidate_metrics()
    integrity = next(
        event for event in metrics["events"] if event["kind"] == "partitioned_audio_position_candidate_integrity"
    )
    integrity["fields"]["final_exact_audio_prefix"] = False
    with pytest.raises(RuntimeGateError, match="exact AV prefix"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_audio_position_domain=AUDIO_POSITION_DOMAIN_SOURCE,
        )


def _accepted_registration(
    dx=0.5,
    dy=-0.25,
    *,
    rms_improvement=0.31,
    last_holdout_improvement=0.29,
):
    return {
        "status": "accepted",
        "reason": "accepted",
        "policy_version": FRAME_GAUGE_POLICY_VERSION,
        "dx": dx,
        "dy": dy,
        "units": "target_latent_cells",
        "invalid_fraction": 0.04,
        "validation_ncc": 0.91,
        "rms_improvement": rms_improvement,
        "runner_margin_ratio": 0.12,
        "last_holdout_ncc": 0.90,
        "last_holdout_improvement": last_holdout_improvement,
        "frame_checks": [
            {"informative": True, "supports_global": True},
            {"informative": True, "supports_global": True},
        ],
        "region_checks": [
            {"name": "upper", "informative": True, "supports_global": True, "strong_conflict": False},
            {"name": "lower", "informative": False, "supports_global": False, "strong_conflict": False},
            {"name": "left", "informative": True, "supports_global": True, "strong_conflict": False},
            {"name": "right", "informative": False, "supports_global": False, "strong_conflict": False},
        ],
        "parity_checks": [
            {"informative": True, "supports_global": True},
            {"informative": False, "supports_global": False},
        ],
    }


def _identity_registration():
    return {
        "status": "identity",
        "reason": "already_aligned",
        "policy_version": FRAME_GAUGE_POLICY_VERSION,
        "dx": 0.0,
        "dy": 0.0,
        "units": "target_latent_cells",
        "invalid_fraction": 0.0,
    }


def _accepted_boundary_motion():
    return {
        "status": "accepted",
        "reason": "accepted",
        "policy": "native_boundary_motion_preservation_v1",
        "min_error_cells": 0.125,
        "min_improvement_ratio": 0.25,
        "min_response": 3.0,
        "checks": {
            "upper45": {
                "informative": True,
                "before_error_cells": 0.8,
                "after_error_cells": 0.2,
                "error_improvement_ratio": 0.75,
                "native": {"dx": 0.0, "dy": 0.0, "response": 12.0, "clipped": False},
                "exact_restored": {"dx": 0.6, "dy": 0.5, "response": 9.0, "clipped": False},
                "candidate": {"dx": 0.1, "dy": 0.1, "response": 11.0, "clipped": False},
            },
            "full": {
                "informative": True,
                "before_error_cells": 0.7,
                "after_error_cells": 0.25,
                "error_improvement_ratio": 0.6428571428571429,
                "native": {"dx": 0.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "exact_restored": {"dx": 0.5, "dy": 0.45, "response": 8.0, "clipped": False},
                "candidate": {"dx": 0.15, "dy": 0.1, "response": 9.0, "clipped": False},
            },
        },
    }


def _rejected_v2_boundary_motion():
    return {
        "status": "rejected",
        "reason": "boundary_upper45_insufficient_improvement",
        "policy": "native_boundary_motion_preservation_v2",
        "min_error_cells": 0.125,
        "min_improvement_ratio": 0.25,
        "min_response": 3.0,
        "checks": {
            "upper45": {
                "informative": True,
                "before_error_cells": 1.0,
                "after_error_cells": 0.8,
                "error_improvement_ratio": 0.2,
                "native": {"dx": 0.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "transformed_native": {"dx": 0.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "exact_restored": {"dx": 1.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "candidate": {"dx": 0.8, "dy": 0.0, "response": 10.0, "clipped": False},
            },
            "full": {
                "informative": True,
                "before_error_cells": 1.0,
                "after_error_cells": 0.5,
                "error_improvement_ratio": 0.5,
                "native": {"dx": 0.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "transformed_native": {"dx": 0.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "exact_restored": {"dx": 1.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "candidate": {"dx": 0.5, "dy": 0.0, "response": 10.0, "clipped": False},
            },
        },
    }


def _accepted_v2_boundary_motion():
    return {
        "status": "accepted",
        "reason": "accepted",
        "policy": "native_boundary_motion_preservation_v2",
        "min_error_cells": 0.125,
        "min_improvement_ratio": 0.25,
        "min_response": 3.0,
        "checks": {
            "upper45": {
                "informative": True,
                "before_error_cells": 1.0,
                "after_error_cells": 0.3,
                "error_improvement_ratio": 0.7,
                "native": {"dx": 0.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "transformed_native": {"dx": 0.2, "dy": 0.0, "response": 10.0, "clipped": False},
                "exact_restored": {"dx": 1.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "candidate": {"dx": 0.5, "dy": 0.0, "response": 10.0, "clipped": False},
            },
            "full": {
                "informative": True,
                "before_error_cells": 0.8,
                "after_error_cells": 0.2,
                "error_improvement_ratio": 0.75,
                "native": {"dx": 0.0, "dy": 0.0, "response": 10.0, "clipped": False},
                "transformed_native": {"dx": 0.1, "dy": 0.0, "response": 10.0, "clipped": False},
                "exact_restored": {"dx": 0.8, "dy": 0.0, "response": 10.0, "clipped": False},
                "candidate": {"dx": 0.3, "dy": 0.0, "response": 10.0, "clipped": False},
            },
        },
    }


def _frame_gauge_event(
    *,
    mode="off",
    result="off",
    guidance_mode="off",
    video_rms_improvement=0.31,
    video_last_holdout_improvement=0.29,
):
    accepted = mode == "on" and result == "accepted"
    shadow_only = mode == "on" and result == "shadow_only"
    candidate = accepted or shadow_only
    identity = mode == "on" and result == "identity"
    if candidate:
        video_registration = _accepted_registration(
            rms_improvement=video_rms_improvement,
            last_holdout_improvement=video_last_holdout_improvement,
        )
    elif identity:
        video_registration = _identity_registration()
    else:
        video_registration = {"status": result, "reason": result}
    guidance_registration = (
        _accepted_registration(0.375, -0.125)
        if candidate and guidance_mode != "off"
        else {"status": "off", "reason": "guidance_off"}
    )
    return _event(
        "partitioned_frame_gauge",
        mode=mode,
        enabled=mode == "on",
        result=result,
        policy_version=FRAME_GAUGE_POLICY_VERSION,
        spatial_warp_applied=accepted,
        candidate_accepted=shadow_only,
        candidate_spatial_warp_computed=shadow_only,
        candidate_guidance_reference_computed=shadow_only and guidance_mode != "off",
        production_mutation_allowed=accepted,
        hardware_invalidation=("00687_visible_frame_shift_after_applied_rigid_v4" if shadow_only else None),
        authoritative_prefix_modified=False,
        exact_prefix_sha256="9" * 64,
        registration_domain="actual_clean_target_video" if mode == "on" else "off",
        transform_domain="actual_clean_target_video" if accepted else "none",
        video_registration=video_registration,
        guidance_registration=guidance_registration,
        boundary_motion=(_accepted_boundary_motion() if candidate else {"status": "not_evaluated"}),
        registered_guidance_reference=accepted and guidance_mode != "off",
        guidance_mode=guidance_mode,
        video_dx=0.5 if candidate else 0.0,
        video_dy=-0.25 if candidate else 0.0,
        guidance_dx=0.375 if candidate and guidance_mode != "off" else 0.0,
        guidance_dy=-0.125 if candidate and guidance_mode != "off" else 0.0,
        extra_h3_nfe=0,
        extra_sampler_lifetimes=0,
        extra_history_boundaries=0,
        extra_provider_calls=0,
        extra_vae_calls=0,
        auto_strength_validation_required=True,
    )


def _install_frame_gauge_transfer(metrics, *, mode="off", result="off"):
    accepted = mode == "on" and result == "accepted"
    transfer = next(event for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["fields"].update(
        frame_gauge_repair_enabled=mode == "on",
        frame_gauge_result=result,
        suffix_dc_bridge_state_mapping=("pre_renoise_clean_operand" if accepted else "conditional_renoise_affine"),
        suffix_dc_bridge_policy="one_token_spatial_mean_v1",
        suffix_dc_bridge_corrected_tokens=1,
        splice_clean_source="actual_provider" if accepted else "inverse_recovered",
    )
    return metrics


def _dora_off_report():
    return {
        "schema": 1,
        "kind": "dora_power_lora_auto_strength_stack_report",
        "auto_strength_enabled": False,
        "auto_strength_device": "gpu",
        "ratio_floor": 0.3,
        "ratio_ceiling": 1.5,
        "rows": [
            {
                "row_index": 0,
                "enabled": True,
                "lora_name": "example.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "status": "applied_without_auto_strength",
                "report": None,
            }
        ],
    }


def test_runtime_gate_verifies_frame_gauge_off_and_resolved_auto_strength_off():
    metrics = _install_frame_gauge_transfer(_metrics())
    metrics["events"].insert(-2, _frame_gauge_event())
    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_frame_gauge_mode="off",
        auto_strength_reports=[_dora_off_report()],
        require_auto_strength_off=True,
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_mode == "off"
    assert report.frame_gauge_result == "off"
    assert report.auto_strength_verified_off is True
    assert len(report.auto_strength_report_digests) == 1


def test_runtime_gate_verifies_accepted_frame_gauge_transaction_with_guidance():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    metrics["events"].insert(
        -2,
        _frame_gauge_event(mode="on", result="accepted", guidance_mode="direction+temporal"),
    )
    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_frame_gauge_mode="on-accepted",
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_result == "accepted"
    assert report.frame_gauge_video_dx == pytest.approx(0.5)
    assert report.frame_gauge_guidance_dx == pytest.approx(0.375)


def test_runtime_gate_verifies_00687_fail_closed_rigid_shadow_with_guidance():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="shadow_only")
    metrics["events"].insert(
        -2,
        _frame_gauge_event(mode="on", result="shadow_only", guidance_mode="direction+temporal"),
    )

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_frame_gauge_mode="on-shadow_only",
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_result == "shadow_only"
    assert report.frame_gauge_video_dx == pytest.approx(0.5)
    assert report.frame_gauge_guidance_dx == pytest.approx(0.375)


def test_runtime_gate_accepts_v2_video_below_legacy_rms_threshold():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    metrics["events"].insert(
        -2,
        _frame_gauge_event(
            mode="on",
            result="accepted",
            video_rms_improvement=0.069,
            video_last_holdout_improvement=0.05,
        ),
    )

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_frame_gauge_mode="on-accepted",
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_result == "accepted"


def test_runtime_gate_allows_guidance_below_legacy_rms_threshold_after_strong_checks():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted", guidance_mode="direction+temporal")
    receipt["fields"]["guidance_registration"]["rms_improvement"] = 0.14931248733225388
    receipt["fields"]["guidance_registration"]["last_holdout_improvement"] = 0.05
    metrics["events"].insert(-2, receipt)

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_frame_gauge_mode="on-accepted",
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_result == "accepted"


def test_runtime_gate_rejects_guidance_that_degrades_aggregate_rms():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted", guidance_mode="direction+temporal")
    receipt["fields"]["guidance_registration"]["rms_improvement"] = -0.01
    metrics["events"].insert(-2, receipt)

    with pytest.raises(RuntimeGateError, match="guidance registration held-out RMS-improvement"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )


def test_runtime_gate_rejects_guidance_that_degrades_last_holdout():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted", guidance_mode="direction+temporal")
    receipt["fields"]["guidance_registration"]["last_holdout_improvement"] = -0.01
    metrics["events"].insert(-2, receipt)

    with pytest.raises(RuntimeGateError, match="guidance registration last-frame improvement"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )


def test_runtime_gate_allows_learned_video_global_runner_ambiguity_after_strong_checks():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted")
    receipt["fields"]["video_registration"]["runner_margin_ratio"] = 0.04789114198525804
    metrics["events"].insert(-2, receipt)

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_frame_gauge_mode="on-accepted",
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_result == "accepted"


def test_runtime_gate_keeps_guidance_global_runner_margin_strict():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(
        mode="on",
        result="accepted",
        guidance_mode="direction+temporal",
    )
    receipt["fields"]["guidance_registration"]["runner_margin_ratio"] = 0.04789114198525804
    metrics["events"].insert(-2, receipt)

    with pytest.raises(RuntimeGateError, match="guidance registration runner-up margin"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )


def test_runtime_gate_requires_accepted_boundary_motion_receipt():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted")
    receipt["fields"]["boundary_motion"]["checks"]["upper45"]["after_error_cells"] = 0.9
    metrics["events"].insert(-2, receipt)

    with pytest.raises(RuntimeGateError, match="upper45 candidate did not improve"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )


def test_runtime_gate_accepts_equal_noninformative_boundary_error():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted")
    for name in ("upper45", "full"):
        check = receipt["fields"]["boundary_motion"]["checks"][name]
        check["before_error_cells"] = 0.0
        check["after_error_cells"] = 0.0
        check["error_improvement_ratio"] = 0.0
        check["informative"] = False
    metrics["events"].insert(-2, receipt)

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_frame_gauge_mode="on-accepted",
    )

    assert report.frame_gauge_verified is True


def test_runtime_gate_validates_identity_video_registration():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="identity")
    metrics["events"].insert(-2, _frame_gauge_event(mode="on", result="identity"))

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_frame_gauge_mode="on-identity",
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_result == "identity"


def test_runtime_gate_rejects_missing_or_enabled_auto_strength_evidence():
    metrics = _metrics()
    metrics["events"].insert(-2, _frame_gauge_event())

    with pytest.raises(RuntimeGateError, match="at least one resolved"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="off",
            require_auto_strength_off=True,
        )

    enabled = _dora_off_report()
    enabled["auto_strength_enabled"] = True
    with pytest.raises(RuntimeGateError, match="resolved OFF"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            auto_strength_reports=[enabled],
            require_auto_strength_off=True,
        )


def test_runtime_gate_can_pin_auto_strength_report_identity_across_pair():
    metrics = _metrics()
    metrics["events"].insert(-2, _frame_gauge_event())
    first = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        auto_strength_reports=[_dora_off_report()],
        require_auto_strength_off=True,
    )
    validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        auto_strength_reports=[_dora_off_report()],
        require_auto_strength_off=True,
        expected_auto_strength_digests=first.auto_strength_report_digests,
    )

    changed = _dora_off_report()
    changed["rows"][0]["strength_model"] = 0.9
    with pytest.raises(RuntimeGateError, match="identity differs"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            auto_strength_reports=[changed],
            require_auto_strength_off=True,
            expected_auto_strength_digests=first.auto_strength_report_digests,
        )


def test_runtime_gate_rejects_accepted_registration_below_heldout_threshold():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted")
    receipt["fields"]["video_registration"]["validation_ncc"] = 0.74
    metrics["events"].insert(-2, receipt)

    with pytest.raises(RuntimeGateError, match="held-out NCC"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )


def test_runtime_gate_rejects_unbound_or_nonfinite_reported_frame_gauge_displacement():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted", guidance_mode="direction+temporal")
    receipt["fields"]["video_dx"] = 0.625
    metrics["events"].insert(-2, receipt)

    with pytest.raises(RuntimeGateError, match="reported video displacement differs"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )

    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    receipt = _frame_gauge_event(mode="on", result="accepted", guidance_mode="direction+temporal")
    receipt["fields"]["guidance_dx"] = -0.25
    metrics["events"].insert(-2, receipt)

    with pytest.raises(RuntimeGateError, match="reported guidance displacement differs"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )

    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="rejected")
    receipt = _frame_gauge_event(mode="on", result="rejected")
    receipt["fields"]["video_dx"] = float("nan")
    metrics["events"].insert(-2, receipt)

    with pytest.raises(RuntimeGateError, match="non-finite numeric value"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-rejected",
        )


def test_runtime_gate_requires_protected_video_noise_ownership_receipt():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    metrics["events"].insert(-2, _frame_gauge_event(mode="on", result="accepted"))
    handoff = next(event for event in metrics["events"] if event["kind"] == "handoff_transfer_wall")
    handoff["fields"]["protected_video_noise_exact"] = False

    with pytest.raises(RuntimeGateError, match="protected video-noise ownership"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )


def test_runtime_gate_rejects_wrong_clean_domain_and_retired_residual_routing():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    metrics["events"].insert(-2, _frame_gauge_event(mode="on", result="accepted"))
    transfer = next(event for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["fields"]["splice_clean_source"] = "inverse_recovered"

    with pytest.raises(RuntimeGateError, match="wrong clean-state source"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )

    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    metrics["events"].insert(-2, _frame_gauge_event(mode="on", result="accepted"))
    transfer = next(event for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["fields"]["suffix_gauge_bridge_policy"] = "retired"

    with pytest.raises(RuntimeGateError, match="retired full-field"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-accepted",
        )


def _install_residual_receipt(event, *, mode="off", measured=False, final_path="rigid_v2"):
    event["fields"]["residual_geometry"] = {
        "policy": "paired_prefix_residual_geometry_v1",
        "requested_mode": mode,
        "measured": measured,
        "measurement_status": "measured" if measured else ("off" if mode == "off" else "not_evaluated"),
        "reason": "measurement_only" if measured else ("disabled" if mode == "off" else "rigid_not_accepted"),
        "selected_model": "none",
        "decision": "not_evaluated",
        "applied": False,
        "final_path": final_path,
        "video": {"status": "off" if mode == "off" else "not_evaluated"},
        "guidance": {"status": "off", "reason": "guidance_off"},
    }
    event["fields"]["residual_geometry_telemetry_bytes"] = 1024
    return event


def test_runtime_gate_preserves_historical_rigid_v2_when_residual_mode_is_off():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    frame = _install_residual_receipt(
        _frame_gauge_event(mode="on", result="accepted"),
        mode="off",
    )
    metrics["events"].insert(-2, frame)

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_residual_mode="off",
        expected_residual_result="off",
    )

    assert report.residual_geometry_verified is True
    assert report.residual_geometry_mode == "off"
    assert report.residual_geometry_result == "off"
    assert report.residual_geometry_horizontal_eligible is False


def test_runtime_gate_rejects_measurement_receipt_that_claims_application():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="rejected")
    frame = _install_residual_receipt(
        _frame_gauge_event(mode="on", result="rejected"),
        mode="measure",
        final_path="baseline",
    )
    frame["fields"]["residual_geometry"]["applied"] = True
    metrics["events"].insert(-2, frame)

    with pytest.raises(RuntimeGateError, match="applied a residual transform"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_residual_mode="measure",
            expected_residual_result="not-evaluated",
        )


def test_runtime_gate_accepts_measure_mode_as_not_evaluated_when_rigid_v2_rejects():
    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="rejected")
    frame = _install_residual_receipt(
        _frame_gauge_event(mode="on", result="rejected"),
        mode="measure",
        final_path="baseline",
    )
    metrics["events"].insert(-2, frame)

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_residual_mode="measure",
        expected_residual_result="not-evaluated",
    )

    assert report.residual_geometry_verified is True
    assert report.residual_geometry_result == "not-evaluated"
    assert report.residual_geometry_evidence_bundle is None


def test_runtime_gate_accepts_producer_fed_measured_only_receipt():
    torch.manual_seed(321)
    latent = torch.randn(1, 24, 4, 56, 74)
    video_measurement = measure_residual_geometry(
        latent,
        latent.clone(),
        rigid_dx=0.0,
        rigid_dy=0.0,
    )
    assert video_measurement["status"] == "measured"

    metrics = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    frame = _install_residual_receipt(
        _frame_gauge_event(mode="on", result="accepted"),
        mode="measure",
        measured=True,
    )
    frame["fields"]["video_dx"] = 0.0
    frame["fields"]["video_dy"] = 0.0
    frame["fields"]["video_registration"]["dx"] = 0.0
    frame["fields"]["video_registration"]["dy"] = 0.0
    frame["fields"]["residual_geometry"].update(
        video=video_measurement,
        guidance={"status": "off", "reason": "guidance_off"},
        boundary_regional={
            "policy": "paired_prefix_residual_boundary_regions_v1",
            "tiles": {
                tile_id: {
                    variant: {"dx": 0.0, "dy": 0.0, "response": 1.0, "clipped": False}
                    for variant in (
                        "native",
                        "exact_unregistered",
                        "transformed_native",
                        "exact_rigid_pre_dc",
                        "exact_rigid_post_dc",
                    )
                }
                | {
                    "pre_dc_native_error": 0.0,
                    "post_dc_native_error": 0.0,
                }
                for tile_id in ("TL", "TM", "TR", "ML", "C", "MR", "BL", "BM", "BR")
            },
        },
    )

    stages = []
    for stage in (
        "learned_native_same_time",
        "learned_rigid_aligned_same_time",
        "exact_restored_pre_high_dc",
        "final_post_high_internal_clean",
        "final_post_high_caller_domain",
    ):
        fields = {
            "policy": "paired_prefix_residual_geometry_v1",
            "stage": stage,
            "session_id": "session",
            "chunk_id": "chunk",
            "tensor_sha256": "a" * 64,
            "prefix_boundary_index": 5,
            "domain": ("caller_output_latent" if stage == "final_post_high_caller_domain" else "model_internal_clean"),
        }
        if stage == "final_post_high_internal_clean":
            fields["owner_before"] = "authoritative_exact_prefix_E"
        stages.append(_event("partitioned_residual_geometry_stage", **fields))

    bicubic_shadow = [
        _event(
            "partitioned_multiframe_trajectory",
            stage="source_low_exact_context_bicubic_shadow",
            roi=roi,
            source_hw=(44, 44),
            target_hw=(64, 64),
            transfer_mode="bicubic",
            diagnostic_only=True,
            production_gate=False,
            extra_h3_nfe=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
        )
        for roi in ("upper45", "full")
    ]

    spatial_shadow = torch.zeros(1, 24, 6, 32, 32, dtype=torch.float32)
    learned_provider = spatial_shadow.clone()
    learned_residual_receipt = measure_learned_transfer_residual_diagnostic(
        learned_provider,
        spatial_shadow,
        5,
    )
    learned_residual = _event(
        "partitioned_learned_transfer_residual",
        domain="model_internal_clean",
        owner_before="source_low_exact_context_bicubic_shadow",
        owner_after="learned_provider_native",
        elapsed_ms=1.0,
        extra_h3_nfe=0,
        extra_provider_calls=0,
        extra_vae_calls=0,
        extra_sampler_lifetimes=0,
        extra_history_boundaries=0,
        **learned_residual_receipt,
    )

    evidence = _event(
        "partitioned_residual_geometry_evidence",
        policy="paired_prefix_residual_geometry_v1",
        requested_mode="measure",
        decision="not_evaluated",
        applied=False,
        horizontal_application_enabled=False,
        status="exported",
        extra_vae_calls=0,
        bundle="h3_flow_regenerate/residual_geometry/test-bundle",
    )
    insert_at = len(metrics["events"]) - 2
    metrics["events"][insert_at:insert_at] = [
        frame,
        *stages,
        *bicubic_shadow,
        learned_residual,
        evidence,
    ]

    report = validate_partitioned_runtime_evidence(
        metrics,
        _log(),
        expected_residual_mode="measure",
        expected_residual_result="measured-only",
    )

    assert report.residual_geometry_verified is True
    assert report.residual_geometry_result == "measured-only"
    assert report.residual_geometry_evidence_bundle == "h3_flow_regenerate/residual_geometry/test-bundle"

    broken = json.loads(json.dumps(metrics))
    shadow = next(
        event
        for event in broken["events"]
        if event["kind"] == "partitioned_multiframe_trajectory"
        and event["fields"].get("stage") == "source_low_exact_context_bicubic_shadow"
    )
    shadow["fields"]["extra_provider_calls"] = 1
    with pytest.raises(RuntimeGateError, match="bicubic transfer shadow added work"):
        validate_partitioned_runtime_evidence(
            broken,
            _log(),
            expected_residual_mode="measure",
            expected_residual_result="measured-only",
        )

    broken_residual = json.loads(json.dumps(metrics))
    learned_event = next(
        event for event in broken_residual["events"] if event["kind"] == "partitioned_learned_transfer_residual"
    )
    learned_event["fields"]["output_mutated"] = True
    with pytest.raises(RuntimeGateError, match="claims output mutation"):
        validate_partitioned_runtime_evidence(
            broken_residual,
            _log(),
            expected_residual_mode="measure",
            expected_residual_result="measured-only",
        )


def test_matched_residual_pair_requires_identical_rigid_v2_and_work_topology():
    control = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    control_frame = _install_residual_receipt(
        _frame_gauge_event(mode="on", result="accepted"),
        mode="off",
    )
    control["events"].insert(-2, control_frame)

    measure = _install_frame_gauge_transfer(_metrics(), mode="on", result="accepted")
    measure_frame = _install_residual_receipt(
        _frame_gauge_event(mode="on", result="accepted"),
        mode="measure",
        measured=True,
    )
    measure["events"].insert(-2, measure_frame)

    pair = compare_residual_measurement_pair(control, measure)
    assert pair["status"] == "matched"
    assert pair["model_call_topology"] == (
        ("low", True),
        ("low", False),
        ("probe", True),
        ("high", True),
        ("high", False),
    )

    measure["events"][3]["fields"]["actual"] = True
    with pytest.raises(RuntimeGateError, match="call topology"):
        compare_residual_measurement_pair(control, measure)


def _install_exact_overlap_fallback_receipt(metrics):
    reason = "boundary_upper45_insufficient_improvement"
    metrics = _install_frame_gauge_transfer(metrics, mode="on", result="rejected")
    transfer = next(event for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["fields"]["frame_gauge_reason"] = reason
    transfer["fields"]["splice_clean_source"] = "actual_provider_boundary_pair_plus_inverse_recovered"
    transfer["fields"]["partitioned_exact_overlap_bridge"] = {
        "policy": "partitioned_exact_overlap_structural_plus_dc_v1",
        "requested": True,
        "trigger": reason,
        "applied": True,
        "state_mapping": "conditional_renoise_affine",
        "source": "actual_provider_boundary_pair",
        "authoritative_prefix_modified": False,
        "later_suffix_extrapolated": False,
        "suffix_representation_bridge_enabled": True,
        "suffix_representation_bridge_accepted": True,
        "suffix_representation_bridge_corrected_tokens": 1,
    }
    frame = _frame_gauge_event(mode="on", result="rejected")
    frame["fields"].update(
        reason=reason,
        video_registration=_accepted_registration(),
        boundary_motion=_rejected_v2_boundary_motion(),
        exact_overlap_fallback_policy="partitioned_exact_overlap_structural_plus_dc_v1",
        exact_overlap_fallback_requested=True,
        exact_overlap_fallback_trigger=reason,
        exact_overlap_fallback_applied=True,
        exact_overlap_fallback_source="actual_provider_boundary_pair",
    )
    metrics["events"].insert(-2, frame)
    return metrics


def test_runtime_gate_accepts_partitioned_exact_overlap_fallback_after_rigid_boundary_veto():
    report = validate_partitioned_runtime_evidence(
        _install_exact_overlap_fallback_receipt(_metrics()),
        _log(),
        expected_frame_gauge_mode="on-rejected",
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_result == "rejected"


def _install_shadow_exact_overlap_fallback_receipt(metrics):
    reason = "hardware_invalidated_global_rigid_application_00687"
    metrics = _install_frame_gauge_transfer(metrics, mode="on", result="shadow_only")
    transfer = next(event for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["fields"].update(
        frame_gauge_reason=reason,
        splice_clean_source="actual_provider_boundary_pair_plus_inverse_recovered",
        suffix_dc_bridge_policy="successor_safe_linear_v2",
        suffix_dc_bridge_corrected_tokens=4,
        partitioned_exact_overlap_bridge={
            "policy": "partitioned_exact_overlap_successor_safe_v2",
            "requested": True,
            "trigger": reason,
            "applied": True,
            "state_mapping": "conditional_renoise_affine",
            "source": "actual_provider_boundary_pair",
            "authoritative_prefix_modified": False,
            "later_suffix_extrapolated": True,
            "suffix_support_policy": "bounded_linear_return_v2",
            "suffix_support_tokens": 4,
            "suffix_outside_support_modified": False,
            "suffix_representation_bridge_enabled": True,
            "suffix_representation_bridge_accepted": True,
            "suffix_representation_bridge_corrected_tokens": 4,
            "suffix_representation_bridge_successor_safe": True,
            "suffix_representation_bridge_temporal_weights": [1.0, 0.75, 0.5, 0.25],
            "suffix_representation_bridge_max_weight_step": 0.25,
        },
    )
    frame = _frame_gauge_event(mode="on", result="shadow_only")
    frame["fields"].update(
        reason=reason,
        boundary_motion=_accepted_v2_boundary_motion(),
        exact_overlap_fallback_policy="partitioned_exact_overlap_successor_safe_v2",
        exact_overlap_fallback_requested=True,
        exact_overlap_fallback_trigger=reason,
        exact_overlap_fallback_applied=True,
        exact_overlap_fallback_source="actual_provider_boundary_pair",
    )
    metrics["events"].insert(-2, frame)
    return metrics


def test_runtime_gate_accepts_exact_overlap_after_hardware_invalidated_rigid_shadow():
    report = validate_partitioned_runtime_evidence(
        _install_shadow_exact_overlap_fallback_receipt(_metrics()),
        _log(),
        expected_frame_gauge_mode="on-shadow_only",
    )

    assert report.frame_gauge_verified is True
    assert report.frame_gauge_result == "shadow_only"


def test_runtime_gate_rejects_successor_safe_shadow_overlap_with_weight_drift():
    metrics = _install_shadow_exact_overlap_fallback_receipt(_metrics())
    transfer = next(event for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["fields"]["partitioned_exact_overlap_bridge"]["suffix_representation_bridge_temporal_weights"][2] = 0.6

    with pytest.raises(RuntimeGateError, match="weights drifted"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-shadow_only",
        )


def test_runtime_gate_rejects_shadow_overlap_if_rigid_mutation_is_reenabled():
    metrics = _install_shadow_exact_overlap_fallback_receipt(_metrics())
    frame = next(event for event in metrics["events"] if event["kind"] == "partitioned_frame_gauge")
    frame["fields"]["production_mutation_allowed"] = True

    with pytest.raises(RuntimeGateError, match="allowed a production mutation"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-shadow_only",
        )


def test_runtime_gate_rejects_exact_overlap_fallback_not_bound_to_transaction_reason():
    metrics = _install_exact_overlap_fallback_receipt(_metrics())
    transfer = next(event for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["fields"]["partitioned_exact_overlap_bridge"]["trigger"] = "boundary_full_insufficient_improvement"

    with pytest.raises(RuntimeGateError, match="trigger differs from the frame-gauge"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-rejected",
        )


def test_runtime_gate_rejects_exact_overlap_fallback_that_extrapolates_later_suffix():
    metrics = _install_exact_overlap_fallback_receipt(_metrics())
    transfer = next(event for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["fields"]["partitioned_exact_overlap_bridge"]["later_suffix_extrapolated"] = True

    with pytest.raises(RuntimeGateError, match="historical exact-overlap repair extrapolated into later suffix tokens"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(),
            expected_frame_gauge_mode="on-rejected",
        )
