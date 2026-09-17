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
    return {"schema_version": 1, "counters": {}, "events": events}


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
            "partitioned exact-prefix: grouped VDN softmax active; "
            "variable-grid linear complement active",
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
    assert report.sol_mapped_calls == 3
    assert report.sol_rectangular_calls == 3
    assert report.sol_requested_q_rows == report.sol_kernel_q_rows == 192
    assert report.vdn_variable_grid_linear_active is True
    assert report.audio_guided_overlap_active is True


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
