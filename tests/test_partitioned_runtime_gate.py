from __future__ import annotations

import json

import pytest
import torch

from h3_flow_regenerate.boundary_content_diagnostics import measure_learned_transfer_residual_diagnostic
from h3_flow_regenerate.frame_gauge import FRAME_GAUGE_POLICY_VERSION
from h3_flow_regenerate.high_stage_boundary import (
    HIGH_PREDICTION_GAUGE_BRIDGE_POLICY,
    HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS,
)
from h3_flow_regenerate.partitioned_runtime_gate import (
    AUDIO_POSITION_DOMAIN_LEGACY,
    AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_EXACT_OVERLAP_COUPLED_POLICY,
    PARTITIONED_SOL_ABI,
    RuntimeGateError,
    compare_residual_measurement_pair,
    validate_coherent_exact_audio_evidence,
    validate_partitioned_runtime_evidence,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PARTITIONED_EXACT_OVERLAP_POLICY,
    _apply_partitioned_exact_overlap_bridge,
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
            transferred_prefix_output_discarded=True,
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


@pytest.mark.parametrize(
    "receipts",
    [
        {"transferred_prefix_output_discarded": True},
        {"upscaler_prefix_output_discarded": True},
        {"transferred_prefix_output_discarded": True, "upscaler_prefix_output_discarded": True},
    ],
)
def test_runtime_gate_accepts_current_and_historical_prefix_discard_receipts(receipts):
    metrics = _metrics()
    transfer = next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer.pop("transferred_prefix_output_discarded")
    transfer.update(receipts)
    assert validate_partitioned_runtime_evidence(metrics, _log()).high_actual == 1


@pytest.mark.parametrize(
    "receipts",
    [
        {},
        {"transferred_prefix_output_discarded": False},
        {"upscaler_prefix_output_discarded": False},
        {"transferred_prefix_output_discarded": 1},
        {"upscaler_prefix_output_discarded": "true"},
        {"transferred_prefix_output_discarded": True, "upscaler_prefix_output_discarded": False},
        {"transferred_prefix_output_discarded": False, "upscaler_prefix_output_discarded": True},
        {"transferred_prefix_output_discarded": None, "upscaler_prefix_output_discarded": True},
    ],
)
def test_runtime_gate_rejects_missing_false_or_conflicting_prefix_discard_receipts(receipts):
    metrics = _metrics()
    transfer = next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer.pop("transferred_prefix_output_discarded")
    transfer.update(receipts)
    with pytest.raises(RuntimeGateError, match="transferred prefix output was not discarded or its receipts disagree"):
        validate_partitioned_runtime_evidence(metrics, _log())


def _metrics_with_high_video_guard(**setup_overrides):
    metrics = _metrics()
    policy = "first_generated_token_exact_high_context_v1"
    for event in metrics["events"]:
        if event["kind"] == "partitioned_transfer":
            event["fields"].update(
                video_high_guard_enabled=True,
                video_high_guard_policy=policy,
                video_high_guard_tokens=1,
            )
        elif event["kind"] == "partitioned_exact_prefix_complete":
            event["fields"].update(
                high_video_guard_enabled=True,
                high_video_guard_policy=policy,
                high_video_guard_tokens=1,
                high_video_guard_final_exact=True,
                high_video_guard_endpoint_canonicalized=False,
                high_video_guard_release_boundary_t=13,
                high_boundary_context_tokens=13,
            )
        elif event["kind"] == "handoff_complete":
            event["fields"]["high_stage_video_guard_requested"] = 1

    setup = {
        "policy": policy,
        "enabled": True,
        "applied": True,
        "guard_tokens": 1,
        "guard_token_index": 12,
        "caller_exact_prefix_tokens": 12,
        "high_protected_video_tokens": 13,
        "guard_source": "exact_restored_pre_high_first_suffix_clean",
        "guard_source_exact": True,
        "caller_prefix_modified": False,
        "caller_prefix_exact": True,
        "audio_modified": False,
        "audio_exact": True,
        "audio_mask_modified": False,
        "audio_mask_exact": True,
        "later_suffix_modified": False,
        "later_suffix_exact": True,
        "mask_outside_guard_modified": False,
        "mask_outside_guard_exact": True,
        "original_guard_mask_exact_one": True,
        "high_guard_mask_exact_zero": True,
        "internal_guard_exact": True,
        "internal_audio_exact": True,
        "caller_internal_round_trip_verified": True,
        "flow_guidance_protected_video_tokens": 13,
        "target_state_reconstruction_verified": True,
        "target_state_reconstruction_max_abs_delta": 1e-7,
        "target_state_reconstruction_rms_delta": 1e-8,
        "preserved_noise_scope": "caller_original_exact_mask_only",
        "extra_h3_nfe": 0,
        "extra_sampler_lifetimes": 0,
        "extra_history_boundaries": 0,
        "extra_provider_calls": 0,
        "extra_vae_calls": 0,
    }
    setup.update(setup_overrides)
    complete = {
        "policy": policy,
        "enabled": True,
        "applied": True,
        "guard_token_index": 12,
        "guard_tokens": 1,
        "final_guard_exact": True,
        "endpoint_canonicalized": False,
        "endpoint_max_abs_delta": 0.0,
        "endpoint_rms_delta": 0.0,
        "caller_prefix_modified": False,
        "audio_modified": False,
        "extra_h3_nfe": 0,
        "extra_sampler_lifetimes": 0,
        "extra_history_boundaries": 0,
        "extra_provider_calls": 0,
        "extra_vae_calls": 0,
    }
    metrics["events"].insert(
        -4,
        _event(
            "partitioned_high_boundary_reference_plan",
            video_enabled=False,
            video_support_tokens=0,
            high_video_guard_enabled=True,
            high_video_guard_policy=policy,
            high_boundary_context_tokens=13,
            high_boundary_prefix_witness="protected_first_generated_guard_tail_after_inpaint_restore",
        ),
    )
    metrics["events"].insert(
        -3,
        _event(
            "partitioned_high_boundary_reference_verified",
            video_enabled=False,
            anchor_calls=0,
            high_video_guard_enabled=True,
            high_boundary_context_tokens=13,
            high_boundary_prefix_witness="protected_first_generated_guard_tail_after_inpaint_restore",
        ),
    )
    metrics["events"].insert(
        -2,
        _event(
            "guidance",
            protected_prefix_t=13,
        ),
    )
    metrics["events"].insert(-2, _event("partitioned_high_video_guard", **setup))
    for roi in ("upper45", "full"):
        metrics["events"].insert(
            -1,
            _event(
                "partitioned_multiframe_trajectory",
                stage="final_post_high_guard_release",
                roi=roi,
                guard_policy=policy,
                guard_tokens=1,
                original_boundary_t=12,
                release_boundary_t=13,
                diagnostic_only=True,
                trajectory_boundary_t=13,
                pairwise_dx=[0.0],
                pairwise_dy=[0.0],
                pairwise_response=[10.0],
                pairwise_clipped=[False],
                pre_pairwise_median_dx=0.0,
                pre_pairwise_median_dy=0.0,
            ),
        )
    metrics["events"].insert(-1, _event("partitioned_high_video_guard_complete", **complete))
    return metrics


def test_partitioned_runtime_gate_accepts_one_token_high_video_guard():
    report = validate_partitioned_runtime_evidence(
        _metrics_with_high_video_guard(),
        _log(),
        expected_logical=5,
        expected_actual=3,
        expected_forecast=2,
    )
    assert report.logical_calls == 5


def test_partitioned_runtime_gate_rejects_high_video_guard_ownership_drift():
    with pytest.raises(RuntimeGateError, match="audio_mask_exact"):
        validate_partitioned_runtime_evidence(
            _metrics_with_high_video_guard(audio_mask_exact=False),
            _log(),
        )


def test_partitioned_runtime_gate_rejects_high_clean_reference_with_guard():
    metrics = _metrics_with_high_video_guard()
    reference_verified = next(
        event for event in metrics["events"] if event["kind"] == "partitioned_high_boundary_reference_verified"
    )
    reference_verified["fields"]["anchor_calls"] = 1
    with pytest.raises(RuntimeGateError, match="anchor executed"):
        validate_partitioned_runtime_evidence(metrics, _log())


def test_partitioned_runtime_gate_rejects_missing_high_guard_release_frontier():
    metrics = _metrics_with_high_video_guard()
    metrics["events"] = [
        event
        for event in metrics["events"]
        if not (
            event["kind"] == "partitioned_multiframe_trajectory"
            and event["fields"].get("stage") == "final_post_high_guard_release"
            and event["fields"].get("roi") == "full"
        )
    ]
    with pytest.raises(RuntimeGateError, match="both release-frontier ROI"):
        validate_partitioned_runtime_evidence(metrics, _log())


def test_partitioned_runtime_gate_rejects_material_high_guard_endpoint_rewrite():
    metrics = _metrics_with_high_video_guard()
    guard_complete = next(
        event for event in metrics["events"] if event["kind"] == "partitioned_high_video_guard_complete"
    )
    guard_complete["fields"]["endpoint_max_abs_delta"] = 0.01
    with pytest.raises(RuntimeGateError, match="numerical roundoff"):
        validate_partitioned_runtime_evidence(metrics, _log())


def _metrics_with_high_prediction_gauge_bridge(**receipt_overrides):
    metrics = _metrics()
    policy = HIGH_PREDICTION_GAUGE_BRIDGE_POLICY
    weights = list(HIGH_PREDICTION_GAUGE_BRIDGE_WEIGHTS)
    support = len(weights)
    prefix_t = 12
    for event in metrics["events"]:
        if event["kind"] == "partitioned_transfer":
            event["fields"].update(
                video_high_prediction_gauge_bridge_enabled=True,
                video_high_prediction_gauge_bridge_policy=policy,
                video_high_prediction_gauge_bridge_support_tokens=support,
            )
        elif event["kind"] == "partitioned_exact_prefix_complete":
            event["fields"].update(
                high_prediction_gauge_bridge_enabled=True,
                high_prediction_gauge_bridge_policy=policy,
                high_prediction_gauge_bridge_support_tokens=support,
                high_prediction_gauge_bridge_weights=weights,
                high_prediction_gauge_bridge_release_boundary_t=prefix_t + support,
                high_boundary_context_tokens=prefix_t,
            )
        elif event["kind"] == "handoff_complete":
            event["fields"].update(
                high_stage_video_guard_requested=0,
                high_stage_prediction_gauge_bridge_requested=support,
            )

    plan = _event(
        "partitioned_high_boundary_reference_plan",
        video_enabled=False,
        video_support_tokens=0,
        high_prediction_gauge_bridge_enabled=True,
        high_prediction_gauge_bridge_policy=policy,
        high_prediction_gauge_bridge_support_tokens=support,
        high_prediction_gauge_bridge_weights=weights,
        high_prediction_gauge_bridge_mask_fully_generated=True,
        high_boundary_context_tokens=prefix_t,
        high_boundary_prefix_witness="authoritative_exact_tail_after_inpaint_restore",
    )
    reference_verified = _event(
        "partitioned_high_boundary_reference_verified",
        video_enabled=False,
        anchor_calls=0,
        high_prediction_gauge_bridge_enabled=True,
        high_prediction_gauge_bridge_policy=policy,
        high_prediction_gauge_bridge_calls=2,
        high_prediction_gauge_bridge_all_model_calls_covered=True,
        high_boundary_context_tokens=prefix_t,
        high_boundary_prefix_witness="authoritative_exact_tail_after_inpaint_restore",
    )
    per_call = []
    for call_index, actual in enumerate((True, False)):
        fields = {
            "policy": policy,
            "call_index": call_index,
            "sigma": 0.8 - 0.2 * call_index,
            "actual": actual,
            "prefix_t": prefix_t,
            "support_tokens": support,
            "temporal_weights": weights,
            "max_release_weight_step": 0.3535533906,
            "model_prefix_to_exact_delta_rms": 0.25,
            "model_prefix_to_exact_delta_abs_max": 0.75,
            "model_native_first_transition_rms": 0.4,
            "exact_rebased_first_transition_rms": 0.4,
            "first_transition_error_rms": 1e-7,
            "first_transition_error_max_abs": 1e-6,
            "caller_prefix_modified": False,
            "predicted_prefix_exact": True,
            "audio_modified": False,
            "audio_exact": True,
            "suffix_outside_support_modified": False,
            "suffix_outside_support_exact": True,
            "extra_h3_nfe": 0,
            "extra_sampler_lifetimes": 0,
            "extra_history_boundaries": 0,
            "extra_provider_calls": 0,
            "extra_vae_calls": 0,
        }
        fields.update(receipt_overrides)
        per_call.append(_event("partitioned_high_prediction_gauge_bridge", **fields))
    bridge_complete = _event(
        "partitioned_high_prediction_gauge_bridge_complete",
        policy=policy,
        calls=2,
        support_tokens=support,
        temporal_weights=weights,
        extra_h3_nfe=0,
        extra_provider_calls=0,
        extra_vae_calls=0,
        extra_sampler_lifetimes=0,
        extra_history_boundaries=0,
    )
    bridge_verified = _event(
        "partitioned_high_prediction_gauge_bridge_verified",
        policy=policy,
        expected=True,
        model_calls=2,
        bridge_calls=2,
        all_model_calls_covered=True,
        support_tokens=support,
        temporal_weights=weights,
        sampler_mask_modified=False,
        sampler_entry_state_modified=False,
        fixed_clean_reference_used=False,
        extra_h3_nfe=0,
        extra_provider_calls=0,
        extra_vae_calls=0,
        extra_sampler_lifetimes=0,
        extra_history_boundaries=0,
    )
    release = [
        _event(
            "partitioned_multiframe_trajectory",
            stage="final_post_high_prediction_gauge_release",
            roi=roi,
            prediction_gauge_bridge_policy=policy,
            bridge_support_tokens=support,
            bridge_temporal_weights=weights,
            original_boundary_t=prefix_t,
            release_boundary_t=prefix_t + support,
            diagnostic_only=True,
            trajectory_boundary_t=prefix_t + support,
        )
        for roi in ("upper45", "full")
    ]

    insert_at = next(
        i
        for i, event in enumerate(metrics["events"])
        if event["kind"] == "model_call" and event["fields"].get("stage") == "high"
    )
    metrics["events"][insert_at:insert_at] = [plan]
    # Per-call receipts are output-side and may be interleaved with model calls in
    # production; ordering is not material to the offline ownership validator.
    metrics["events"].extend(per_call)
    metrics["events"].extend([reference_verified, bridge_complete, bridge_verified, *release])
    return metrics


def test_partitioned_runtime_gate_accepts_high_prediction_gauge_bridge():
    report = validate_partitioned_runtime_evidence(
        _metrics_with_high_prediction_gauge_bridge(),
        _log(),
        expected_logical=5,
        expected_actual=3,
        expected_forecast=2,
    )
    assert report.high_logical == 2


def test_partitioned_runtime_gate_rejects_prediction_bridge_transition_drift():
    with pytest.raises(RuntimeGateError, match="model-native first transition"):
        validate_partitioned_runtime_evidence(
            _metrics_with_high_prediction_gauge_bridge(first_transition_error_max_abs=0.01),
            _log(),
        )


def test_partitioned_runtime_gate_rejects_prediction_bridge_fixed_anchor_reactivation():
    metrics = _metrics_with_high_prediction_gauge_bridge()
    verified = next(
        event for event in metrics["events"] if event["kind"] == "partitioned_high_prediction_gauge_bridge_verified"
    )
    verified["fields"]["fixed_clean_reference_used"] = True
    with pytest.raises(RuntimeGateError, match="fixed anchor"):
        validate_partitioned_runtime_evidence(metrics, _log())


def test_partitioned_runtime_gate_rejects_missing_prediction_bridge_release_roi():
    metrics = _metrics_with_high_prediction_gauge_bridge()
    metrics["events"] = [
        event
        for event in metrics["events"]
        if not (
            event["kind"] == "partitioned_multiframe_trajectory"
            and event["fields"].get("stage") == "final_post_high_prediction_gauge_release"
            and event["fields"].get("roi") == "full"
        )
    ]
    with pytest.raises(RuntimeGateError, match="both release-frontier ROI"):
        validate_partitioned_runtime_evidence(metrics, _log())


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

    selected_log = _log().replace(
        "mode=model_timestep_only",
        "mode=sampler_mask",
    )
    report = validate_partitioned_runtime_evidence(
        _metrics(),
        selected_log,
        expected_audio_guided_overlap_mode="sampler_mask",
        expected_audio_guided_overlap_ticks=4,
    )
    assert report.audio_guided_overlap_mode == "sampler_mask"
    assert report.audio_guided_overlap_ticks == 4

    with pytest.raises(RuntimeGateError, match="mode differs from expectation"):
        validate_partitioned_runtime_evidence(
            _metrics(),
            _log(),
            expected_audio_guided_overlap_mode="sampler_mask",
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


def _exact_audio_metrics():
    metrics = _candidate_metrics()
    verified = next(e["fields"] for e in metrics["events"] if e["kind"] == "partitioned_audio_position_domain_verified")
    verified["model_timestep_override_calls"] = 0
    metrics["events"].append(
        _event(
            "partitioned_exact_audio_mask_verified",
            policy="coherent_exact_audio_mask_v1",
            mode="exact_mask",
            requested_overlap_ticks=16,
            verified_model_entries=3,
            effective_overlap_ticks=0,
            sampler_input_mask_exact=True,
            model_timestep_mask_exact=True,
            model_velocity_mask_exact=True,
            final_prefix_exact=True,
            fail_closed=True,
            timestep_override_applied=False,
            regenerated_prefix_restored=False,
            extra_h3_nfe=0,
            extra_sampler_lifetimes=0,
            extra_history_boundaries=0,
            extra_provider_calls=0,
            extra_vae_calls=0,
        )
    )
    return metrics


def test_exact_audio_evidence_accepts_verified_native_masks_without_label_overrides():
    metrics = _exact_audio_metrics()
    receipt = validate_coherent_exact_audio_evidence(metrics)
    assert receipt["verified_model_entries"] == 3
    log = _log().replace("mode=model_timestep_only ticks=4 applied=True", "mode=exact_mask ticks=16 applied=False")
    report = validate_partitioned_runtime_evidence(
        metrics,
        log,
        require_audio_overlap=False,
        expected_audio_guided_overlap_mode="exact_mask",
        expected_audio_guided_overlap_ticks=16,
        expected_audio_position_domain=AUDIO_POSITION_DOMAIN_SOURCE,
    )
    assert report.audio_position_candidate_verified is True
    assert report.audio_position_model_timestep_override_calls == 0


@pytest.mark.parametrize(
    "name,value",
    [
        ("verified_model_entries", 0),
        ("effective_overlap_ticks", 4),
        ("model_velocity_mask_exact", False),
        ("sampler_input_mask_exact", False),
        ("model_timestep_mask_exact", False),
        ("timestep_override_applied", True),
        ("regenerated_prefix_restored", True),
        ("extra_h3_nfe", 1),
        ("extra_vae_calls", 1),
        ("extra_provider_calls", 1),
    ],
)
def test_exact_audio_evidence_rejects_incoherence_and_added_work(name, value):
    metrics = _exact_audio_metrics()
    metrics["events"][-1]["fields"][name] = value
    with pytest.raises(RuntimeGateError):
        validate_coherent_exact_audio_evidence(metrics)


def test_exact_audio_evidence_cannot_reuse_an_earlier_continuation_receipt():
    metrics = _exact_audio_metrics()
    metrics["events"].append(_event("partitioned_stage_plan"))
    with pytest.raises(RuntimeGateError, match="one coherent"):
        validate_coherent_exact_audio_evidence(metrics)


def test_exact_audio_expectation_rejects_legacy_ramp_without_coherent_receipt():
    log = _log().replace("mode=model_timestep_only", "mode=sampler_mask_exact_timestep")
    with pytest.raises(RuntimeGateError, match="one coherent"):
        validate_partitioned_runtime_evidence(
            _candidate_metrics(),
            log,
            expected_audio_guided_overlap_mode="sampler_mask_exact_timestep",
        )


def test_exact_audio_expectation_rejects_applied_ramp_even_with_receipt():
    log = _log().replace("mode=model_timestep_only", "mode=exact_mask")
    with pytest.raises(RuntimeGateError, match="must not apply"):
        validate_partitioned_runtime_evidence(
            _exact_audio_metrics(),
            log,
            expected_audio_guided_overlap_mode="exact_mask",
        )


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


def _install_current_coupled_overlap_receipt(arm):
    if arm == "inactive":
        metrics = _install_frame_gauge_transfer(_metrics(), mode="off", result="off")
        metrics["events"].insert(-2, _frame_gauge_event())
    elif arm in {"veto", "guidance"}:
        metrics = _install_exact_overlap_fallback_receipt(_metrics())
    else:
        metrics = _install_shadow_exact_overlap_fallback_receipt(_metrics())
    transfer = next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    frame = next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_frame_gauge")
    requested = arm != "inactive"
    applied = arm not in {"inactive", "noop"}
    weights = [1.0, 0.75, 0.5, 0.25] if requested else [1.0]
    transfer.update(
        splice_clean_source="actual_clean_postprocess",
        suffix_dc_bridge_policy="successor_safe_linear_v2" if requested else "one_token_spatial_mean_v1",
        suffix_dc_bridge_corrected_tokens=len(weights),
        suffix_dc_bridge_first_weight=weights[0],
        suffix_dc_bridge_last_weight=weights[-1],
        suffix_dc_bridge_delta_rms=0.0 if arm == "noop" else 0.23,
    )
    overlap = transfer.setdefault("partitioned_exact_overlap_bridge", {})
    overlap.update(
        policy=PARTITIONED_EXACT_OVERLAP_COUPLED_POLICY,
        requested=requested,
        applied=applied,
        state_mapping="conditional_renoise_affine" if applied else "disabled_or_noop",
        authoritative_prefix_modified=False,
        later_suffix_extrapolated=applied,
        suffix_support_policy="bounded_linear_return_v2" if applied else "first_suffix_only_v1",
        suffix_support_tokens=4 if applied else 0,
        suffix_outside_support_modified=False,
        suffix_representation_bridge_enabled=applied,
        suffix_representation_bridge_accepted=applied,
        suffix_representation_bridge_corrected_tokens=4 if applied else 0,
        suffix_representation_bridge_structural_rms=0.4 if applied else 0.0,
        suffix_representation_bridge_reason="exact_overlap_structural_residual_successor_distributed"
        if applied
        else "structural_overlap_already_matched",
        suffix_representation_bridge_centered_error_before=0.4 if applied else 0.0,
        suffix_representation_bridge_successor_safe=True,
        suffix_representation_bridge_temporal_weights=weights,
        suffix_representation_bridge_max_weight_step=0.25 if requested else 1.0,
        dc_support_policy="bounded_linear_return_v2" if requested else "first_suffix_only_v1",
        dc_support_tokens=len(weights),
        dc_temporal_weights=weights.copy(),
    )
    frame.update(
        exact_overlap_fallback_policy=PARTITIONED_EXACT_OVERLAP_COUPLED_POLICY,
        exact_overlap_fallback_requested=requested,
        exact_overlap_fallback_applied=applied,
    )
    if arm == "guidance":
        reason = "unsupported_sampler_contract"
        trigger = f"guidance_rejected_after_video_boundary_acceptance:{reason}"
        frame.update(
            reason=reason,
            boundary_motion=_accepted_v2_boundary_motion(),
            guidance_registration={"status": "rejected", "reason": reason},
            exact_overlap_fallback_trigger=trigger,
        )
        transfer["frame_gauge_reason"] = reason
        overlap["trigger"] = trigger
    if arm == "dc_only":
        overlap.update(
            suffix_representation_bridge_enabled=False,
            suffix_representation_bridge_accepted=False,
            suffix_representation_bridge_corrected_tokens=0,
            suffix_representation_bridge_structural_rms=0.0,
            suffix_representation_bridge_reason="structural_overlap_already_matched",
            suffix_representation_bridge_centered_error_before=0.0,
        )
    return metrics


@pytest.mark.parametrize("arm", ["shadow", "veto", "guidance", "inactive", "dc_only", "noop"])
def test_runtime_gate_accepts_current_coupled_overlap_receipts(arm):
    expected_mode = "off" if arm == "inactive" else "on-rejected" if arm in {"veto", "guidance"} else "on-shadow_only"
    report = validate_partitioned_runtime_evidence(
        _install_current_coupled_overlap_receipt(arm), _log(), expected_frame_gauge_mode=expected_mode
    )
    assert report.frame_gauge_verified is True


@pytest.mark.parametrize(
    ("location", "field", "value", "error"),
    [
        ("overlap", "dc_temporal_weights", [1.0], "DC weights drifted"),
        ("overlap", "dc_support_tokens", 1, "DC support drifted"),
        ("transfer", "suffix_dc_bridge_corrected_tokens", 1, "DC support does not match"),
        ("transfer", "suffix_dc_bridge_policy", "one_token_spatial_mean_v1", "DC policy does not match"),
        ("transfer", "suffix_dc_bridge_first_weight", 0.75, "DC endpoint weights drifted"),
        ("transfer", "suffix_dc_bridge_last_weight", 1.0, "DC endpoint weights drifted"),
        ("overlap", "suffix_representation_bridge_temporal_weights", [1.0, 0.75, 0.6, 0.25], "weights drifted"),
        (
            "frame",
            "exact_overlap_fallback_policy",
            "partitioned_exact_overlap_structural_plus_dc_v1",
            "receipts differ",
        ),
        ("overlap", "policy", "partitioned_exact_overlap_structural_plus_dc_v1", "receipts differ"),
    ],
)
def test_runtime_gate_rejects_uncoupled_current_overlap_receipts(location, field, value, error):
    metrics = _install_current_coupled_overlap_receipt("guidance")
    transfer = next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    targets = {
        "transfer": transfer,
        "overlap": transfer["partitioned_exact_overlap_bridge"],
        "frame": next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_frame_gauge"),
    }
    targets[location][field] = value
    with pytest.raises(RuntimeGateError, match=error):
        validate_partitioned_runtime_evidence(metrics, _log(), expected_frame_gauge_mode="on-rejected")


def test_runtime_gate_rejects_current_transaction_without_transfer_overlap():
    metrics = _install_current_coupled_overlap_receipt("guidance")
    transfer = next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    del transfer["partitioned_exact_overlap_bridge"]
    with pytest.raises(RuntimeGateError, match="coupled exact-overlap transaction and transfer receipts differ"):
        validate_partitioned_runtime_evidence(metrics, _log(), expected_frame_gauge_mode="on-rejected")


def test_runtime_gate_rejects_current_noop_overlap_with_nonzero_dc():
    metrics = _install_current_coupled_overlap_receipt("noop")
    transfer = next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_transfer")
    transfer["suffix_dc_bridge_delta_rms"] = 0.23
    with pytest.raises(RuntimeGateError, match="inactive coupled exact-overlap receipt reports a nonzero correction"):
        validate_partitioned_runtime_evidence(metrics, _log(), expected_frame_gauge_mode="on-shadow_only")


def test_runtime_gate_rejects_current_guidance_overlap_with_failed_video_boundary():
    metrics = _install_current_coupled_overlap_receipt("guidance")
    frame = next(event["fields"] for event in metrics["events"] if event["kind"] == "partitioned_frame_gauge")
    frame["boundary_motion"]["status"] = "rejected"
    with pytest.raises(RuntimeGateError, match="outside an eligible frame-gauge arm"):
        validate_partitioned_runtime_evidence(metrics, _log(), expected_frame_gauge_mode="on-rejected")


def _install_dc_overlap_receipt(arm):
    metrics = _install_current_coupled_overlap_receipt(arm)
    transfer = next(e["fields"] for e in metrics["events"] if e["kind"] == "partitioned_transfer")
    frame = next(e["fields"] for e in metrics["events"] if e["kind"] == "partitioned_frame_gauge")
    learned = torch.zeros(1, 24, 7, 8, 10)
    exact = learned[:, :, :2].clone()
    exact[:, :, -1] += (torch.arange(10) - 4.5) / 16 + (0.0 if arm == "noop" else 0.25)
    _, _, representation, dc = _apply_partitioned_exact_overlap_bridge(learned, learned, exact, sigma=0.8)
    transfer.update(**dc, suffix_dc_bridge_policy="one_token_spatial_mean_v1")
    overlap = transfer["partitioned_exact_overlap_bridge"]
    overlap.update(
        **representation,
        policy=PARTITIONED_EXACT_OVERLAP_POLICY,
        later_suffix_extrapolated=False,
        suffix_support_policy="first_suffix_only_v1",
        suffix_support_tokens=1 if overlap["applied"] else 0,
        dc_support_policy="first_suffix_only_v1",
        dc_support_tokens=1,
        dc_temporal_weights=[1.0],
        structural_bridge_retired=True,
        structural_support_tokens=0,
        production_correction="one_token_per_channel_spatial_mean_only",
    )
    frame["exact_overlap_fallback_policy"] = PARTITIONED_EXACT_OVERLAP_POLICY
    return metrics


@pytest.mark.parametrize("arm", ["shadow", "veto", "guidance", "inactive", "noop"])
def test_runtime_gate_accepts_dc_only_production_receipts(arm):
    expected = "off" if arm == "inactive" else "on-rejected" if arm in {"veto", "guidance"} else "on-shadow_only"
    report = validate_partitioned_runtime_evidence(
        _install_dc_overlap_receipt(arm), _log(), expected_frame_gauge_mode=expected
    )
    assert report.frame_gauge_verified


@pytest.mark.parametrize(
    ("location", "field", "value", "error"),
    [
        ("transfer", "suffix_dc_bridge_policy", "successor_safe_linear_v2", "DC policy does not match"),
        ("transfer", "suffix_dc_bridge_corrected_tokens", 4, "DC support does not match"),
        ("transfer", "suffix_dc_bridge_first_weight", 0.75, "channel-mean support drifted"),
        ("transfer", "suffix_dc_bridge_last_weight", 0.25, "channel-mean support drifted"),
        ("transfer", "suffix_dc_bridge_enabled", False, "channel-mean support drifted"),
        ("transfer", "suffix_dc_bridge_delta_rms", 0.0, "application differs"),
        ("transfer", "splice_clean_source", "foreign", "wrong clean-state source"),
        ("overlap", "dc_temporal_weights", [1.0, 0.75], "channel-mean support drifted"),
        ("overlap", "dc_support_tokens", 4, "channel-mean support drifted"),
        ("overlap", "suffix_representation_bridge_requested", True, "reactivated structural transport"),
        ("overlap", "suffix_representation_bridge_enabled", True, "reactivated structural transport"),
        ("overlap", "suffix_representation_bridge_accepted", True, "reactivated structural transport"),
        ("overlap", "suffix_representation_bridge_corrected_tokens", 1, "reactivated structural transport"),
        ("overlap", "structural_bridge_retired", False, "reactivated structural transport"),
        ("overlap", "structural_support_tokens", 1, "reactivated structural transport"),
        ("overlap", "later_suffix_extrapolated", True, "changed later suffix tokens"),
        ("overlap", "suffix_support_tokens", 4, "changed later suffix tokens"),
        ("overlap", "suffix_outside_support_modified", True, "changed later suffix tokens"),
        ("overlap", "state_mapping", "pre_renoise_clean_operand", "wrong conditional-state mapping"),
        ("frame", "exact_overlap_fallback_policy", PARTITIONED_EXACT_OVERLAP_COUPLED_POLICY, "receipts differ"),
    ],
)
def test_runtime_gate_rejects_dc_only_ownership_or_support_drift(location, field, value, error):
    metrics = _install_dc_overlap_receipt("guidance")
    transfer = next(e["fields"] for e in metrics["events"] if e["kind"] == "partitioned_transfer")
    targets = {
        "transfer": transfer,
        "overlap": transfer["partitioned_exact_overlap_bridge"],
        "frame": next(e["fields"] for e in metrics["events"] if e["kind"] == "partitioned_frame_gauge"),
    }
    targets[location][field] = value
    with pytest.raises(RuntimeGateError, match=error):
        validate_partitioned_runtime_evidence(metrics, _log(), expected_frame_gauge_mode="on-rejected")


def test_runtime_gate_rejects_dc_only_transaction_without_transfer_receipt():
    metrics = _install_dc_overlap_receipt("guidance")
    transfer = next(e["fields"] for e in metrics["events"] if e["kind"] == "partitioned_transfer")
    del transfer["partitioned_exact_overlap_bridge"]
    with pytest.raises(RuntimeGateError, match="receipts differ"):
        validate_partitioned_runtime_evidence(metrics, _log(), expected_frame_gauge_mode="on-rejected")


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
