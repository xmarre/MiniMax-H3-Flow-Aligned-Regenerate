from __future__ import annotations

import json

import pytest

from h3_flow_regenerate.partitioned_runtime_gate import (
    AUDIO_POSITION_DOMAIN_LEGACY,
    AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_SOL_ABI,
    RuntimeGateError,
    validate_partitioned_runtime_evidence,
)


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


def test_partitioned_runtime_gate_rejects_unapplied_or_wrong_width_audio_overlap():
    with pytest.raises(RuntimeGateError, match="four-tick partitioned audio guided overlap"):
        validate_partitioned_runtime_evidence(
            _metrics(),
            _log().replace("ticks=4 applied=True", "ticks=3 applied=True"),
        )
    with pytest.raises(RuntimeGateError, match="four-tick partitioned audio guided overlap"):
        validate_partitioned_runtime_evidence(
            _metrics(),
            _log().replace("ticks=4 applied=True", "ticks=4 applied=False"),
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

def _frame_gauge_event(*, mode="off", result="off", guidance_mode="off"):
    accepted = mode == "on" and result == "accepted"
    return _event(
        "partitioned_frame_gauge",
        mode=mode,
        enabled=mode == "on",
        result=result,
        spatial_warp_applied=accepted,
        authoritative_prefix_modified=False,
        registered_guidance_reference=accepted and guidance_mode != "off",
        guidance_mode=guidance_mode,
        video_dx=0.5 if accepted else 0.0,
        video_dy=-0.25 if accepted else 0.0,
        guidance_dx=0.375 if accepted and guidance_mode != "off" else 0.0,
        guidance_dy=-0.125 if accepted and guidance_mode != "off" else 0.0,
        extra_h3_nfe=0,
        extra_sampler_lifetimes=0,
        extra_history_boundaries=0,
        extra_provider_calls=0,
        extra_vae_calls=0,
        auto_strength_validation_required=True,
    )


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
    metrics = _metrics()
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
    metrics = _metrics()
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

