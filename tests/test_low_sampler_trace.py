"""First-divergence localization and read-only observer lifetime contracts."""

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.low_sampler_trace import (
    POLICY,
    LowReplayTrace,
    _delta,
    _digest,
    active_trace,
    low_trace_scope,
    native_trace_wrapper,
    trace_stage,
)


@pytest.fixture
def trace():
    observer = LowReplayTrace()
    try:
        yield observer
    finally:
        observer.close()


def emit(trace, arm, *, changed=None, actual=True, stage="low", delta=0.125):
    trace.arms.setdefault(arm, [])
    trace.arm, trace.stage = arm, stage
    for step in range(3):
        trace.evaluation = step
        for boundary in ("solver_input", "predict_input", "timestep", "predict_output"):
            value = torch.full((1, 4), 0.5 if boundary == "timestep" else float(step))
            if (step, boundary) == changed:
                value += delta
            metadata = {"actual": actual, "phase": "model", "outer_step": step} if boundary == "predict_output" else {}
            trace.record(boundary, value, **metadata)


@pytest.mark.parametrize(
    ("boundary", "actual", "classification"),
    [
        ("predict_output", True, "actual_prediction"),
        ("predict_output", False, "forecast_prediction"),
        ("solver_input", True, "solver_interval_before_mask"),
        ("predict_input", True, "post_mask_model_input"),
    ],
)
def test_first_prediction_matches_but_later_boundary_diverges(trace, boundary, actual, classification):
    emit(trace, "A", actual=actual)
    emit(trace, "B_target_projection", actual=actual)
    emit(trace, "A_replay", changed=(1, boundary), actual=actual)
    report = trace.summary()["comparisons"]["A_replay"]
    assert report["first_observed_divergence"]["evaluation"] == 1
    assert report["first_observed_divergence"]["classification"] == classification
    assert report["first_observed_divergence"]["difference"]["rms"] == pytest.approx(0.125)
    assert not report["reproduction_within_tolerance"]
    operands = trace.divergent_operands()
    assert torch.equal(
        operands["first_divergence_A_replay"] - operands["first_divergence_A"], torch.full((1, 4), 0.125)
    )


def test_probe_only_difference_is_separate(trace):
    for arm in ("A", "A_replay"):
        emit(trace, arm)
        emit(trace, arm, stage="probe", changed=(2, "predict_output") if arm == "A_replay" else None)
    first = trace.summary()["comparisons"]["A_replay"]["first_observed_divergence"]
    assert first["classification"] == "probe_predict_output"
    assert first["evaluation"] == 2


def test_changed_forecast_identity_prevents_false_alignment(trace):
    emit(trace, "A", actual=True)
    emit(trace, "A_replay", actual=False, changed=(0, "predict_output"))
    report = trace.summary()["comparisons"]["A_replay"]
    assert report["first_observed_divergence"] is None
    assert report["alignment_failure"]["reason"] == "actual_forecast_phase_or_outer_step"
    assert not report["fully_observed_records_exact"]
    assert not trace.divergent_operands()


@pytest.mark.parametrize("field", ["cfg", "conditioning"])
def test_later_semantic_change_rejects_replay_after_tolerated_tensor_difference(trace, field):
    emit(trace, "A")
    emit(trace, "A_replay", changed=(0, "predict_input"), delta=1e-6)
    row = next(r for r in trace.arms["A_replay"] if r["evaluation"] == 1 and r["boundary"] == "predict_input")
    row["metadata"][field] = 2.0 if field == "cfg" else {"sha256": "changed"}
    report = trace.summary()["comparisons"]["A_replay"]
    assert report["first_observed_divergence"]["evaluation"] == 0
    assert not report["reproduction_within_tolerance"]


def test_conditioning_difference_prevents_mislabeling_later_tensor_export(trace):
    emit(trace, "A")
    emit(trace, "A_replay", changed=(1, "predict_output"))
    trace.arms["A_replay"][1]["metadata"]["conditioning"] = {"sha256": "changed"}
    report = trace.summary()["comparisons"]["A_replay"]
    assert report["first_observed_divergence"]["boundary"] == "native_conditioning"
    assert not report["reproduction_within_tolerance"]
    assert not trace.divergent_operands()


def test_timestep_order_and_missing_records_reject_alignment(trace):
    emit(trace, "A")
    emit(trace, "A_replay", changed=(1, "timestep"))
    assert trace.summary()["comparisons"]["A_replay"]["alignment_failure"]["reason"] == "boundary_order_or_timestep"
    trace.arms["A_replay"] = trace.arms["A_replay"][:4]
    assert trace.summary()["comparisons"]["A_replay"]["alignment_failure"]["reason"] == "different_record_count"


def test_storage_bound_marks_incomplete_and_never_retains_gpu(trace):
    trace.max_bytes = 1
    emit(trace, "A")
    emit(trace, "A_replay")
    report = trace.summary()
    assert report["baseline_disk_bytes"] == 0
    assert not report["recorded_tensor_coverage_complete"]
    assert not report["comparisons"]["A_replay"]["reproduction_within_tolerance"]
    assert not trace.divergent_operands()


def test_exact_bytes_are_distinct_from_numeric_tolerance(trace):
    emit(trace, "A")
    emit(trace, "A_replay", changed=(1, "predict_output"), delta=1e-6)
    report = trace.summary()["comparisons"]["A_replay"]
    assert report["reproduction_within_tolerance"]
    assert not report["fully_observed_records_exact"]
    assert report["first_observed_divergence"]["difference"]["exact"] is False


def test_nonfinite_and_shape_mismatch_do_not_qualify():
    assert _delta(torch.ones(2), torch.full((2,), float("nan")))["finite"] is False
    assert _delta(torch.ones(2), torch.ones(3))["compatible"] is False
    assert _delta(torch.tensor([0.0]), torch.tensor([-0.0]))["exact"] is False


def test_digest_is_full_or_explicitly_unobserved():
    a = torch.arange(8.0)
    b = a.clone()
    b[3] += 1
    assert _digest(a)["sha256"] != _digest(b)["sha256"]
    bounded = _digest(a, budget=1)
    assert bounded["unobserved"] == ["root"]
    assert "sha256" not in bounded["tree"]


def test_scope_restores_options_and_context_on_exception(trace):
    original = {"diffusion_model": {"original": [object()]}, "apply_model": {"old": [object()]}}
    guider = SimpleNamespace(model_options={"transformer_options": {"wrappers": original}})
    sampler = SimpleNamespace(sampler_function=lambda: None)
    with pytest.raises(RuntimeError, match="injected"), trace_stage(trace, "A", "low", sampler, guider):
        assert active_trace() is trace
        assert POLICY in guider.model_options["transformer_options"]["wrappers"]["diffusion_model"]
        assert guider.model_options["transformer_options"]["wrappers"]["apply_model"] is original["apply_model"]
        raise RuntimeError("injected")
    assert guider.model_options["transformer_options"]["wrappers"] is original
    assert active_trace() is None
    with pytest.raises(ValueError), low_trace_scope() as owned:
        path = owned.root
        raise ValueError
    assert not path.exists()


def test_native_full_av_output_catches_audio_only_difference(trace):
    original = torch.ones(1, 2)
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    sampler = SimpleNamespace(sampler_function=lambda: None)
    for arm in ("A", "A_replay"):
        with trace_stage(trace, arm, "low", sampler, guider):
            trace.evaluation = 0
            result = native_trace_wrapper(
                lambda x, *args, selected=arm, **kwargs: [x[0].clone(), x[1] + (0.25 if selected == "A_replay" else 0)],
                [original, original],
                torch.tensor([500.0]),
                original,
            )
            assert torch.equal(result[0], original)
    first = trace.summary()["comparisons"]["A_replay"]["first_observed_divergence"]
    assert first["boundary"] == "native_output_audio"
    assert torch.equal(original, torch.ones_like(original))


def test_native_multiple_calls_have_distinct_ordinals(trace):
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    sampler = SimpleNamespace(sampler_function=lambda: None)
    with trace_stage(trace, "A", "low", sampler, guider):
        for _ in range(2):
            native_trace_wrapper(lambda x, *a, **kw: x, [torch.ones(1), torch.ones(1)], torch.ones(1), torch.ones(1))
    native_inputs = [r for r in trace.arms["A"] if r["boundary"] == "native_input_video"]
    assert [r["native_call"] for r in native_inputs] == [1, 2]
    assert not trace.truncated


def test_read_only_live_solver_frame_and_rng(trace):
    guider = SimpleNamespace(model_options={"transformer_options": {}})

    def solver(x):
        before = x.clone()
        trace.begin_prediction(guider, x * 0.5, torch.tensor([0.75]), {})
        assert torch.equal(x, before)
        return x + 1

    sampler = SimpleNamespace(sampler_function=solver)
    state = torch.get_rng_state().clone()
    with trace_stage(trace, "A", "low", sampler, guider):
        result = solver(torch.ones(1, 4))
    assert torch.equal(result, torch.full((1, 4), 2.0))
    assert torch.equal(torch.get_rng_state(), state)
    rows = trace.arms["A"]
    assert rows[0]["boundary"] == "solver_input"
    assert rows[0]["sha256"] != rows[1]["sha256"]


def test_native_core_euler_is_observed_without_callable_replacement(trace):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    from comfy.k_diffusion.sampling import sample_euler

    guider = SimpleNamespace(model_options={"transformer_options": {}})
    sampler = SimpleNamespace(sampler_function=sample_euler)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    initial = torch.ones(1, 4)

    def model(x, sigma, **kwargs):
        trace.begin_prediction(guider, x, sigma, {})
        result = x * 0.25
        trace.end_prediction(result, actual=True, phase="model", outer=trace.evaluation)
        return result

    expected = sample_euler(lambda x, sigma, **kw: x * 0.25, initial.clone(), sigmas, disable=True)
    for arm in ("A", "A_replay"):
        with trace_stage(trace, arm, "low", sampler, guider):
            actual = sample_euler(model, initial.clone(), sigmas, disable=True)
            assert torch.equal(actual, expected)
            assert sampler.sampler_function is sample_euler
    assert trace.summary()["comparisons"]["A_replay"]["fully_observed_records_exact"]


def test_flow_predict_and_native_wrapper_preserve_actual_forecast_provenance(trace):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    from comfy.patcher_extension import WrapperExecutor, get_all_wrappers

    from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FlowBinding, flow_predict_wrapper

    options = {"transformer_options": {"h3_flow_stage": "low"}}
    binding = FlowBinding()
    options[FLOW_BINDING_KEY] = binding
    guider = SimpleNamespace(model_options=options)
    sampler = SimpleNamespace(sampler_function=lambda: None)
    control = {"step": 0}
    actuals = [True, False, True]

    def predictor(x, timestep, model_options=None, seed=None):
        transformer = model_options["transformer_options"]
        step = control["step"]
        transformer["spectrum_h3_actual"] = actuals[step]
        transformer["spectrum_h3_outer_step_id"] = step
        transformer["spectrum_h3_solver_phase"] = "model"
        if actuals[step]:
            executor = WrapperExecutor.new_executor(
                lambda values, *a, **kw: [values[0] * 0.25, values[1] * 0.5],
                get_all_wrappers("diffusion_model", transformer),
            )
            values = executor.execute([x[:, :2], x[:, 2:]], timestep, x, transformer)
            return torch.cat(values, dim=1)
        return x * 0.125

    observed = []
    for arm in ("A", "A_replay"):
        values = []
        with trace_stage(trace, arm, "low", sampler, guider):
            for step in range(3):
                control["step"] = step
                call = WrapperExecutor.new_class_executor(predictor, guider, [flow_predict_wrapper])
                values.append(call.execute(torch.ones(1, 4), torch.tensor([1.0 - step * 0.25]), options, 42))
        observed.append(values)
    assert all(torch.equal(a, b) for a, b in zip(*observed, strict=True))
    calls = [r for r in trace.arms["A"] if r["boundary"] == "predict_output"]
    assert [r["metadata"]["actual"] for r in calls] == actuals
    assert [r["metadata"]["outer_step"] for r in calls] == [0, 1, 2]
    assert len([r for r in trace.arms["A"] if r["boundary"] == "native_output_audio"]) == 2
    assert binding.metrics.counters["spectrum_forecast_calls"] == 2
    assert active_trace() is None
