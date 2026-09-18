from __future__ import annotations

import json

import pytest

from h3_flow_regenerate.partitioned_runtime_gate import (
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
            "partitioned audio guided overlap active ticks=4 exact_prefix=4 "
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


def _correlated_metrics() -> dict:
    metrics = _metrics()
    request_id = "flow-request-1"
    serial = 0
    stage_ids = {"low": "low-1", "probe": "probe-1", "high": "high-1"}
    for event in metrics["events"]:
        if event["kind"] == "partitioned_stage_plan":
            event["fields"]["request_id"] = request_id
        if event["kind"] == "model_call":
            stage = event["fields"]["stage"]
            event["fields"].update(
                request_id=request_id,
                stage_id=stage_ids[stage],
                evaluation_id=f"{request_id}:{serial}",
            )
            serial += 1
    for stage in ("low", "probe"):
        metrics["events"].append(
            {
                "kind": "partitioned_stage_runtime_summary",
                "fields": {
                    "request_id": request_id,
                    "stage_id": stage_ids[stage],
                    "owner_generation": stage_ids[stage],
                    "host_component_s": {
                        "vdn_gather_host_wall_s": 0.01,
                        "vdn_softmax_host_wall_s": 0.02,
                        "vdn_linear_readout_total_host_wall_s": 0.03,
                    },
                    "host_component_calls": {
                        "vdn_gather_host_wall_s": 1,
                        "vdn_softmax_host_wall_s": 1,
                        "vdn_linear_readout_total_host_wall_s": 1,
                    },
                },
            }
        )
    metrics["stage_accounting"] = [
        {
            "kind": "low_stage_wall",
            "request_id": request_id,
            "stage_id": stage_ids["low"],
            "wall_ms": 20.0,
            "model_ms": 18.0,
            "model_calls": 2,
            "remainder_ms": 2.0,
        },
        {
            "kind": "handoff_probe_wall",
            "request_id": request_id,
            "stage_id": stage_ids["probe"],
            "wall_ms": 10.0,
            "model_ms": 9.0,
            "model_calls": 1,
            "remainder_ms": 1.0,
        },
        {
            "kind": "high_stage_wall",
            "request_id": request_id,
            "stage_id": stage_ids["high"],
            "wall_ms": 21.0,
            "model_ms": 19.0,
            "model_calls": 2,
            "remainder_ms": 2.0,
        },
    ]
    return metrics


def _correlated_sol_record(request_id="flow-request-1") -> dict:
    record = _sol_record()
    record["validation"] = {
        "examples": [
            {
                "mode": "partitioned_mapped_weighted_v1",
                "context": {
                    "flow_request_id": request_id,
                    "flow_stage_id": "low-1",
                    "flow_evaluation_id": f"{request_id}:0",
                },
            }
        ]
    }
    record["cuda_diagnostics"] = {
        "enabled": True,
        "details": [
            {
                "kind": "vdn_partitioned_components",
                "context": {
                    "flow_request_id": request_id,
                    "flow_stage": "low",
                    "flow_stage_id": "low-1",
                    "flow_evaluation_id": f"{request_id}:0",
                    "block_index": 0,
                    "plan_digest": "a" * 64,
                },
                "cuda_event_ms": {
                    "vdn_preprocess": 0.1,
                    "vdn_gather": 0.2,
                    "vdn_softmax": 0.3,
                    "vdn_weights": 0.1,
                    "vdn_softmax_epilogue": 0.2,
                    "vdn_linear_features": 0.2,
                    "vdn_linear_statistics": 0.2,
                    "vdn_linear_scans": 0.2,
                    "vdn_linear_gather": 0.2,
                    "vdn_linear_output": 0.2,
                    "vdn_linear_projection": 0.1,
                },
            }
        ],
    }
    return record


def test_partitioned_runtime_gate_accepts_complete_performance_correlation():
    report = validate_partitioned_runtime_evidence(
        _correlated_metrics(),
        _log(_correlated_sol_record()),
        require_performance_accounting=True,
    )
    assert report.request_id == "flow-request-1"
    assert report.correlated_model_calls == report.logical_calls
    assert report.stage_accounting_rows == 3
    assert report.stage_accounting_unknown_rows == 0
    assert report.partitioned_component_summary_events == 2
    assert "vdn_linear_readout_total_host_wall_s" in report.partitioned_host_component_names
    assert report.sol_correlation_records == 1
    assert report.vdn_cuda_samples == 1
    assert "vdn_linear_output" in report.vdn_cuda_component_names


def test_partitioned_runtime_gate_never_treats_missing_timing_as_zero():
    metrics = _correlated_metrics()
    metrics["stage_accounting"][1]["model_ms"] = None
    metrics["stage_accounting"][1]["remainder_ms"] = None
    with pytest.raises(RuntimeGateError, match="unknown timing"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(_correlated_sol_record()),
            require_performance_accounting=True,
        )


def test_partitioned_runtime_gate_rejects_cross_request_sol_correlation():
    with pytest.raises(RuntimeGateError, match="different Flow request"):
        validate_partitioned_runtime_evidence(
            _correlated_metrics(),
            _log(_correlated_sol_record("different-request")),
            require_performance_accounting=True,
        )


def test_partitioned_runtime_gate_rejects_missing_vdn_cuda_attribution():
    record = _correlated_sol_record()
    record["cuda_diagnostics"]["details"] = []
    with pytest.raises(RuntimeGateError, match="no correlated VDN"):
        validate_partitioned_runtime_evidence(
            _correlated_metrics(),
            _log(record),
            require_performance_accounting=True,
        )


def test_partitioned_runtime_gate_rejects_incomplete_vdn_host_attribution():
    metrics = _correlated_metrics()
    summary = next(
        event
        for event in metrics["events"]
        if event["kind"] == "partitioned_stage_runtime_summary"
    )
    del summary["fields"]["host_component_s"]["vdn_linear_readout_total_host_wall_s"]
    for event in metrics["events"]:
        if event["kind"] == "partitioned_stage_runtime_summary":
            event["fields"]["host_component_s"].pop(
                "vdn_linear_readout_total_host_wall_s",
                None,
            )
    with pytest.raises(RuntimeGateError, match="host attribution is incomplete"):
        validate_partitioned_runtime_evidence(
            metrics,
            _log(_correlated_sol_record()),
            require_performance_accounting=True,
        )
