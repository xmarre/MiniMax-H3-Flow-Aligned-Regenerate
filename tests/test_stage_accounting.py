from types import SimpleNamespace

import torch

from h3_flow_regenerate import runtime
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


def test_predict_correlation_uses_actual_empty_transformer_options(monkeypatch):
    binding = runtime.FlowBinding()
    binding.active_request_id = "flow-test"
    binding.evaluation_serial = 0
    guider = SimpleNamespace(model_options={})
    seen = {}

    class Executor:
        class_obj = guider

        def __call__(self, x, timestep, model_options, seed):
            del timestep, seed
            transformer = model_options["transformer_options"]
            seen["evaluation_id"] = transformer.get(runtime.FLOW_EVALUATION_ID_KEY)
            return x

    monkeypatch.setattr(runtime, "_resolve_binding", lambda _guider: binding)
    model_options = {"transformer_options": {}}
    x = torch.zeros((1, 1), dtype=torch.float32)
    timestep = torch.tensor([0.5], dtype=torch.float32)

    result = runtime.flow_predict_wrapper(
        Executor(),
        x,
        timestep,
        model_options=model_options,
        seed=1,
    )

    assert result is x
    assert seen["evaluation_id"] == "flow-test:0"
    assert runtime.FLOW_EVALUATION_ID_KEY not in model_options["transformer_options"]
    model_call = next(event for event in binding.metrics.events if event.kind == "model_call")
    assert model_call.fields["evaluation_id"] == "flow-test:0"
