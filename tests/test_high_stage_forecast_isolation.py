from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from h3_flow_regenerate import high_stage_forecast_isolation as iso
from h3_flow_regenerate import runtime
from h3_flow_regenerate.guidance import GuidanceConfig
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig


class _Metrics:
    def __init__(self):
        self.events = []

    def event(self, kind, **fields):
        self.events.append(SimpleNamespace(kind=kind, fields=fields))


class _Provider:
    api_version = 1
    kind = "minimax_h3_learned_latent_upscaler"
    model_name = "dummy.safetensors"
    device = "cpu"
    inference_device = "cpu"
    precision = "fp32"
    offload_after_upscale = False

    def upscale_clean_video(self, *_args, **_kwargs):
        raise AssertionError("provider must not execute in isolation unit tests")


def _binding(mode: str = "direction"):
    return SimpleNamespace(guidance=GuidanceConfig(mode=mode), metrics=_Metrics())


def _config():
    return ProgressiveTargetInputConfig(
        source_scale=0.7,
        handoff_coordinate=0.35,
        handoff_selection="fixed",
        transfer_mode="learned_3d",
        exact_prefix_mode="fallback",
        learned_upscaler=_Provider(),
    )


def test_eligibility_is_narrow():
    binding = _binding()
    config = _config()
    assert iso._eligible(config, binding)
    assert not iso._eligible(replace(config, transfer_mode="bicubic", learned_upscaler=None), binding)
    assert not iso._eligible(replace(config, exact_prefix_mode="target_sparse_lifter"), binding)
    assert not iso._eligible(config, _binding("off"))


def test_probe_stage_keeps_native_one_call_refinement_contract(monkeypatch):
    seen = {}

    @iso.contextlib.contextmanager
    def original(guider):
        transformer = guider.model_options.setdefault("transformer_options", {})
        previous = transformer.get("h3_refinement")
        transformer["h3_refinement"] = {
            "api": 1,
            "active": True,
            "min_actual_prefix_steps": 1,
            "sigma_reference": 1.0,
            "source": "native",
        }
        try:
            seen.update(transformer["h3_refinement"])
            yield
        finally:
            if previous is None:
                transformer.pop("h3_refinement", None)
            else:
                transformer["h3_refinement"] = previous

    monkeypatch.setattr(iso, "_ORIGINAL_HIGH_STAGE_CONTRACT", original)
    state = iso._state()
    state.tls.record = iso._IsolationRecord(True, _binding(), 0)
    guider = SimpleNamespace(model_options={"transformer_options": {runtime.FLOW_STAGE_KEY: "probe"}})
    try:
        with iso._high_stage_contract_isolation_wrapper(guider):
            request = guider.model_options["transformer_options"]["h3_refinement"]
            assert request["min_actual_prefix_steps"] == 1
    finally:
        state.tls.record = None

    assert seen["min_actual_prefix_steps"] == 1


def test_high_stage_requests_two_actual_prefix_calls(monkeypatch):
    @iso.contextlib.contextmanager
    def should_not_run(_guider):
        raise AssertionError("native high-stage contract should not be entered for active isolation")
        yield

    monkeypatch.setattr(iso, "_ORIGINAL_HIGH_STAGE_CONTRACT", should_not_run)
    binding = _binding()
    state = iso._state()
    record = iso._IsolationRecord(True, binding, 0)
    state.tls.record = record
    guider = SimpleNamespace(model_options={"transformer_options": {runtime.FLOW_STAGE_KEY: "high"}})
    transformer = guider.model_options["transformer_options"]
    try:
        with iso._high_stage_contract_isolation_wrapper(guider):
            request = transformer["h3_refinement"]
            assert request["api"] == 1
            assert request["active"] is True
            assert request["min_actual_prefix_steps"] == 2
            assert request["sigma_reference"] == 1.0
            assert request["source"] == "h3_flow_high_stage_forecast_isolation"
        assert "h3_refinement" not in transformer
    finally:
        state.tls.record = None

    assert record.high_contract_entries == 1
    assert binding.metrics.events[-1].kind == "high_stage_forecast_isolation"


def test_high_stage_conflicting_existing_contract_fails_closed():
    binding = _binding()
    state = iso._state()
    state.tls.record = iso._IsolationRecord(True, binding, 0)
    guider = SimpleNamespace(
        model_options={
            "transformer_options": {
                runtime.FLOW_STAGE_KEY: "high",
                "h3_refinement": {
                    "api": 1,
                    "active": True,
                    "min_actual_prefix_steps": 1,
                    "sigma_reference": 1.0,
                },
            }
        }
    )
    try:
        with (
            pytest.raises(RuntimeError, match="conflicts with high-stage forecast isolation"),
            iso._high_stage_contract_isolation_wrapper(guider),
        ):
            pass
    finally:
        state.tls.record = None


def test_result_validation_requires_second_high_call_actual(monkeypatch):
    binding = _binding()

    def original(*_args, **_kwargs):
        record = iso._active()
        assert record is not None
        record.high_contract_entries = 1
        binding.metrics.events.extend(
            [
                SimpleNamespace(kind="model_call", fields={"stage": "high", "actual": True}),
                SimpleNamespace(kind="model_call", fields={"stage": "high", "actual": False}),
            ]
        )
        return "result"

    monkeypatch.setattr(iso, "_ORIGINAL_RUN_PROGRESSIVE", original)
    with pytest.raises(RuntimeError, match="did not honor"):
        iso._run_progressive_isolation_wrapper(
            None,
            SimpleNamespace(),
            binding,
            _config(),
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            0,
            [],
        )


def test_result_validation_records_observed_topology(monkeypatch):
    binding = _binding()

    def original(*_args, **_kwargs):
        record = iso._active()
        assert record is not None
        record.high_contract_entries = 1
        binding.metrics.events.extend(
            [
                SimpleNamespace(kind="model_call", fields={"stage": "high", "actual": True}),
                SimpleNamespace(kind="model_call", fields={"stage": "high", "actual": True}),
                SimpleNamespace(kind="model_call", fields={"stage": "high", "actual": True}),
            ]
        )
        return "result"

    monkeypatch.setattr(iso, "_ORIGINAL_RUN_PROGRESSIVE", original)
    result = iso._run_progressive_isolation_wrapper(
        None,
        SimpleNamespace(),
        binding,
        _config(),
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        0,
        [],
    )
    assert result == "result"
    event = binding.metrics.events[-1]
    assert event.kind == "high_stage_forecast_isolation_result"
    assert event.fields["high_logical_calls"] == 3
    assert event.fields["high_actual_calls"] == 3
    assert event.fields["high_forecast_calls"] == 0
    assert event.fields["second_high_actual"] is True
