from __future__ import annotations

from types import SimpleNamespace

import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.guidance import GuidanceConfig
from h3_flow_regenerate.high_stage_diagnostics import (
    HIGH_STAGE_DIAGNOSTIC_KEY,
    high_stage_diagnostic_context,
    make_high_stage_diagnostic_contract,
    next_call_fields,
    record_callback_boundary,
    record_packed_boundary,
    record_video_boundary,
)
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.runtime import (
    FLOW_BINDING_KEY,
    FLOW_STAGE_KEY,
    SPECTRUM_ACTUAL_KEY,
    SPECTRUM_OUTER_STEP_KEY,
    SPECTRUM_PHASE_KEY,
    FlowBinding,
    flow_predict_wrapper,
)


def _packed_fixture():
    video = torch.linspace(-1.0, 1.0, 1 * 24 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 24, 3, 4, 4)
    audio = torch.zeros((1, 32, 2, 3), dtype=torch.float32)
    packed, shapes = pack_streams((video, audio))
    return video, audio, packed, shapes


def _contract(shapes):
    return make_high_stage_diagnostic_contract(
        prefix_t=1,
        shapes=shapes,
        phases=((0, "single"), (1, "single"), (2, "single")),
        sampler="sample_res_multistep",
    )


def test_high_stage_diagnostic_context_is_transient():
    _video, _audio, _packed, shapes = _packed_fixture()
    contract = _contract(shapes)
    guider = SimpleNamespace(model_options={"transformer_options": {}})

    with high_stage_diagnostic_context(guider, contract) as active:
        assert active is contract
        assert guider.model_options["transformer_options"][HIGH_STAGE_DIAGNOSTIC_KEY] is contract

    assert HIGH_STAGE_DIAGNOSTIC_KEY not in guider.model_options["transformer_options"]


def test_call_provenance_uses_solver_metadata_and_fallbacks():
    _video, _audio, _packed, shapes = _packed_fixture()
    contract = _contract(shapes)

    actual = next_call_fields(
        contract,
        sigma=0.5,
        coordinate=0.375,
        actual=True,
        solver_phase=None,
        solver_outer_step=None,
        spectrum_step_id=17,
    )
    forecast = next_call_fields(
        contract,
        sigma=0.25,
        coordinate=0.25,
        actual=False,
        solver_phase="forecast_phase",
        solver_outer_step=7,
        spectrum_step_id=18,
    )

    assert actual["logical_step"] == 0
    assert actual["provenance"] == "actual"
    assert actual["solver_phase"] == "single"
    assert actual["solver_outer_step"] == 0
    assert forecast["logical_step"] == 1
    assert forecast["provenance"] == "forecast"
    assert forecast["solver_phase"] == "forecast_phase"
    assert forecast["solver_outer_step"] == 7
    assert contract["call_index"] == 2
    assert contract["last_call"] == forecast
    assert contract["call_history"] == [actual, forecast]


def test_boundary_recorders_are_read_only_and_callback_semantics_are_explicit():
    video, _audio, packed, shapes = _packed_fixture()
    contract = _contract(shapes)
    previous = next_call_fields(
        contract,
        sigma=0.5,
        coordinate=0.375,
        actual=True,
        solver_phase="single",
        solver_outer_step=0,
        spectrum_step_id=11,
    )
    fields = next_call_fields(
        contract,
        sigma=0.25,
        coordinate=0.2,
        actual=False,
        solver_phase="single",
        solver_outer_step=1,
        spectrum_step_id=12,
    )
    metrics = H3FlowMetrics()
    packed_before = packed.clone()
    video_before = video.clone()

    record_packed_boundary(metrics, "mixed_grid_high_prediction_boundary", packed, contract, fields)
    record_video_boundary(metrics, "mixed_grid_high_guided_boundary", video, contract, fields)
    record_callback_boundary(
        metrics,
        step=1,
        global_step=12,
        x0=packed,
        x=packed,
        contract=contract,
    )

    assert torch.equal(packed, packed_before)
    assert torch.equal(video, video_before)
    assert [event.kind for event in metrics.events] == [
        "mixed_grid_high_prediction_boundary",
        "mixed_grid_high_guided_boundary",
        "mixed_grid_high_step_boundary",
    ]
    callback = metrics.events[-1].fields
    assert callback["event_call_fields_semantics"] == "current_callback_x0_prediction"
    assert callback["state_semantics"] == "pre_current_solver_update_post_previous_outer_update"
    assert callback["state_after_previous_solver_step"] is True
    assert callback["completed_solver_step"] == 0
    assert callback["state_source_semantics"] == "last_model_call_of_previous_solver_outer"
    assert callback["x0_semantics"] == "sampler_callback_denoised_after_model_wrappers"
    assert callback["provenance"] == "forecast"
    assert callback["x0_call_provenance"] == "forecast"
    assert callback["x0_call_spectrum_step_id"] == 12
    assert callback["state_source_call_provenance"] == "actual"
    assert callback["state_source_call_logical_step"] == previous["logical_step"]
    assert callback["state_source_call_spectrum_step_id"] == 11
    for name in (
        "state_seam_rms",
        "state_seam_lowpass_rms",
        "state_seam_spatial_mean_rms",
        "x0_seam_rms",
        "x0_seam_lowpass_rms",
        "x0_seam_spatial_mean_rms",
    ):
        assert torch.isfinite(torch.tensor(callback[name]))


def test_callback_state_source_uses_last_call_of_previous_pece_outer():
    _video, _audio, packed, shapes = _packed_fixture()
    contract = make_high_stage_diagnostic_contract(
        prefix_t=1,
        shapes=shapes,
        phases=((0, "predicted"), (1, "predicted"), (1, "corrected"), (2, "predicted")),
        sampler="sample_sa_solver_pece",
    )
    next_call_fields(
        contract,
        sigma=0.8,
        coordinate=0.7,
        actual=True,
        solver_phase="predicted",
        solver_outer_step=0,
        spectrum_step_id=20,
    )
    next_call_fields(
        contract,
        sigma=0.6,
        coordinate=0.5,
        actual=False,
        solver_phase="predicted",
        solver_outer_step=1,
        spectrum_step_id=21,
    )
    corrected = next_call_fields(
        contract,
        sigma=0.6,
        coordinate=0.5,
        actual=True,
        solver_phase="corrected",
        solver_outer_step=1,
        spectrum_step_id=22,
    )
    current = next_call_fields(
        contract,
        sigma=0.4,
        coordinate=0.3,
        actual=False,
        solver_phase="predicted",
        solver_outer_step=2,
        spectrum_step_id=23,
    )
    metrics = H3FlowMetrics()

    record_callback_boundary(
        metrics,
        step=2,
        global_step=14,
        x0=packed,
        x=packed,
        contract=contract,
    )

    callback = metrics.events[-1].fields
    assert callback["provenance"] == current["provenance"]
    assert callback["x0_call_solver_phase"] == "predicted"
    assert callback["x0_call_spectrum_step_id"] == 23
    assert callback["completed_solver_step"] == 1
    assert callback["state_source_call_logical_step"] == corrected["logical_step"]
    assert callback["state_source_call_solver_phase"] == "corrected"
    assert callback["state_source_call_provenance"] == "actual"
    assert callback["state_source_call_spectrum_step_id"] == 22


def test_first_callback_state_has_no_previous_outer_source():
    _video, _audio, packed, shapes = _packed_fixture()
    contract = _contract(shapes)
    next_call_fields(
        contract,
        sigma=0.5,
        coordinate=0.375,
        actual=True,
        solver_phase="single",
        solver_outer_step=0,
        spectrum_step_id=None,
    )
    metrics = H3FlowMetrics()

    record_callback_boundary(
        metrics,
        step=0,
        global_step=11,
        x0=packed,
        x=packed,
        contract=contract,
    )

    callback = metrics.events[-1].fields
    assert callback["state_semantics"] == "high_stage_input_before_first_solver_update"
    assert callback["state_after_previous_solver_step"] is False
    assert callback["completed_solver_step"] is None
    assert callback["state_source_semantics"] == "no_previous_solver_outer"
    assert callback["state_source_call_logical_step"] is None
    assert callback["x0_call_logical_step"] == 0


class _Executor:
    def __init__(self, guider, result):
        self.class_obj = guider
        self._result = result

    def __call__(self, _x, _timestep, _model_options, _seed):
        return self._result.clone()


def _run_predict(monkeypatch, *, diagnostics: bool, actual: bool):
    _video, _audio, packed, shapes = _packed_fixture()
    binding = FlowBinding(guidance=GuidanceConfig(mode="direction"))
    binding.active_guidance_run = object()
    guider = SimpleNamespace(
        model_options={FLOW_BINDING_KEY: binding},
        inner_model=SimpleNamespace(latent_shapes=shapes),
    )
    executor = _Executor(guider, packed)
    transformer = {
        FLOW_STAGE_KEY: "high",
        SPECTRUM_ACTUAL_KEY: actual,
        SPECTRUM_PHASE_KEY: "single",
        SPECTRUM_OUTER_STEP_KEY: 0,
    }
    if diagnostics:
        transformer[HIGH_STAGE_DIAGNOSTIC_KEY] = _contract(shapes)

    def fake_guidance(video_x0, **_kwargs):
        guided = video_x0.clone()
        guided[:, :, 1:] += 0.125
        return guided

    monkeypatch.setattr("h3_flow_regenerate.runtime.apply_guidance", fake_guidance)
    x = packed.clone()
    x_before = x.clone()
    result = flow_predict_wrapper(
        executor,
        x,
        torch.tensor([0.5], dtype=torch.float32),
        {"transformer_options": transformer},
        123,
    )
    assert torch.equal(x, x_before)
    return result, binding.metrics


def test_predict_diagnostics_do_not_change_guidance_result_or_nfe_counters(monkeypatch):
    diagnostic_result, diagnostic_metrics = _run_predict(
        monkeypatch,
        diagnostics=True,
        actual=True,
    )
    baseline_result, baseline_metrics = _run_predict(
        monkeypatch,
        diagnostics=False,
        actual=True,
    )

    assert torch.equal(diagnostic_result, baseline_result)
    assert diagnostic_metrics.counters == baseline_metrics.counters
    assert diagnostic_metrics.counters["transformer_actual_nfe"] == 1
    assert diagnostic_metrics.counters["sampler_logical_calls"] == 1

    diagnostic_kinds = [event.kind for event in diagnostic_metrics.events]
    assert diagnostic_kinds == [
        "model_call",
        "mixed_grid_high_prediction_boundary",
        "mixed_grid_high_guided_boundary",
        "guidance",
    ]
    prediction = diagnostic_metrics.events[1].fields
    guided = diagnostic_metrics.events[2].fields
    assert prediction["provenance"] == "actual"
    assert guided["provenance"] == "actual"
    assert guided["seam_rms"] != prediction["seam_rms"]


def test_predict_diagnostics_record_forecast_without_changing_nfe_accounting(monkeypatch):
    _result, metrics = _run_predict(
        monkeypatch,
        diagnostics=True,
        actual=False,
    )

    assert metrics.counters.get("transformer_actual_nfe", 0) == 0
    assert metrics.counters["spectrum_forecast_calls"] == 1
    assert metrics.counters["sampler_logical_calls"] == 1
    prediction = next(event for event in metrics.events if event.kind == "mixed_grid_high_prediction_boundary")
    guided = next(event for event in metrics.events if event.kind == "mixed_grid_high_guided_boundary")
    assert prediction.fields["provenance"] == "forecast"
    assert prediction.fields["actual"] is False
    assert guided.fields["provenance"] == "forecast"
