from h3_flow_regenerate.metrics import H3FlowMetrics


def test_stage_accounting_correlates_model_time_without_inventing_missing_values():
    metrics = H3FlowMetrics()
    metrics.event(
        "model_call",
        request_id="request-1",
        stage="low",
        stage_id="low-1",
        evaluation_id="request-1:0",
        elapsed_ms=10.0,
    )
    metrics.event(
        "model_call",
        request_id="request-1",
        stage="low",
        stage_id="low-1",
        evaluation_id="request-1:1",
        elapsed_ms=12.0,
    )
    metrics.event(
        "low_stage_wall",
        request_id="request-1",
        stage_id="low-1",
        elapsed_ms=25.0,
    )
    metrics.event("handoff_probe_wall", elapsed_ms=7.0)

    accounting = metrics.snapshot()["stage_accounting"]
    assert accounting[0] == {
        "kind": "low_stage_wall",
        "request_id": "request-1",
        "stage_id": "low-1",
        "wall_ms": 25.0,
        "model_ms": 22.0,
        "model_calls": 2,
        "remainder_ms": 3.0,
    }
    assert accounting[1]["model_ms"] is None
    assert accounting[1]["remainder_ms"] is None
