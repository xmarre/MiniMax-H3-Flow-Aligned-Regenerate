"""First-divergence localization and read-only observer lifetime contracts."""

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.low_sampler_trace import (
    POLICY,
    LowReplayTrace,
    _delta,
    _digest,
    _full_sha,
    active_trace,
    backend_receipt,
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
    assert report["first_observed_divergence"]["boundary"] == "predictor_conditioning"
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


@pytest.mark.parametrize("missing", ["both", "replay"])
def test_missing_logical_outputs_never_qualify_even_when_other_rows_match(trace, missing):
    emit(trace, "A")
    emit(trace, "A_replay")
    for arm in ("A", "A_replay") if missing == "both" else ("A_replay",):
        trace.arms[arm] = [r for r in trace.arms[arm] if r["boundary"] != "predict_output"]
    report = trace.summary()
    assert not report["recorded_tensor_coverage_complete"]
    assert not report["comparisons"]["A_replay"]["fully_observed_records_exact"]
    assert not report["comparisons"]["A_replay"]["reproduction_within_tolerance"]


def test_missing_end_on_stage_exit_is_explicit_and_context_restores(trace):
    guider = SimpleNamespace(model_options={"transformer_options": {}})

    def solver(x):
        trace.begin_prediction(guider, x, torch.ones(1), {})

    with trace_stage(trace, "A", "low", SimpleNamespace(sampler_function=solver), guider):
        solver(torch.ones(1, 4))
    assert active_trace() is None
    assert trace._pending_prediction is None
    assert any("missing_predict_output_at_stage_exit" in r for r in trace.truncated)
    assert not trace.summary()["recorded_tensor_coverage_complete"]


@pytest.mark.parametrize("field", ["spectrum_h3_solver_phase", "spectrum_h3_outer_step_id"])
def test_malformed_prediction_identity_is_unobserved_without_changing_return(trace, field):
    guider = SimpleNamespace(model_options={"transformer_options": {}})

    def solver(x):
        trace.begin_prediction(guider, x, torch.ones(1), {})
        native_trace_wrapper(lambda values, *a, **kw: values, [x[:, :2], x[:, 2:]], torch.ones(1), x, {})
        result = trace.finish_prediction(x, guider, {field: None})
        assert result is x

    with trace_stage(trace, "A", "low", SimpleNamespace(sampler_function=solver), guider):
        solver(torch.ones(1, 4))
    assert any("incomplete_prediction_identity" in r for r in trace.truncated)
    assert not trace.summary()["recorded_tensor_coverage_complete"]


def native_pair(trace, payload, *, options=None):
    guider = SimpleNamespace(model_options={"transformer_options": {}})

    def solver(x, arm):
        trace.begin_prediction(guider, x, torch.ones(1), {})
        values = native_trace_wrapper(
            lambda x, *a, **kw: x,
            [x[:, :2], x[:, 2:]],
            torch.ones(1),
            torch.ones(1),
            options(arm) if options else {},
            payload=payload(arm),
        )
        trace.end_prediction(torch.cat(values, dim=1), actual=True, phase="model", outer=0)

    for arm in ("A", "A_replay"):
        with trace_stage(trace, arm, "low", SimpleNamespace(sampler_function=solver), guider):
            solver(torch.ones(1, 4), arm)
    return trace.summary()


@pytest.mark.parametrize("kind", ["opaque", "over_budget"])
def test_identically_omitted_but_different_mandatory_payload_cannot_qualify(trace, kind):
    from h3_flow_regenerate.low_sampler_trace import CONDITIONING_BYTES

    def payload(arm):
        value = 0 if arm == "A" else 1
        if kind == "opaque":
            return SimpleNamespace(mutable_conditioning=value)
        return torch.full((CONDITIONING_BYTES // 4 + 1,), float(value))

    report = native_pair(trace, payload)
    assert not report["mandatory_conditioning_coverage_complete"]
    assert not report["recorded_tensor_coverage_complete"]
    assert not report["comparisons"]["A_replay"]["reproduction_within_tolerance"]
    assert not report["comparisons"]["A_replay"]["fully_observed_records_exact"]


def test_changed_positional_native_mask_is_a_conditioning_divergence(trace):
    report = native_pair(
        trace,
        lambda arm: None,
        options=lambda arm: {
            "denoise_mask": torch.tensor([0.0 if arm == "A" else 1.0]),
        },
    )
    first = report["comparisons"]["A_replay"]["first_observed_divergence"]
    assert first["boundary"] == "native_conditioning"
    assert not report["comparisons"]["A_replay"]["reproduction_within_tolerance"]


def audio_context_options(arm):
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.partitioned_diagnostics import PartitionedAudioModelTimestepContext

    return {
        "h3_flow_partitioned_audio_model_timestep_context_v1": PartitionedAudioModelTimestepContext(
            audio_mask=torch.ones(1, 1, 2, 4),
            metrics=H3FlowMetrics(),
            ticks=0,
            audio_prefix_ticks=2,
            mask_kind="exact_authoritative",
            calls=0 if arm == "A" else 22,
            verification_calls=0 if arm == "A" else 22,
        ),
        "callbacks": {"on_clone": {"test_owner": [lambda *args: None]}},
    }


def test_audio_context_telemetry_and_executable_callbacks_do_not_change_mandatory_data(trace):
    report = native_pair(trace, lambda arm: None, options=audio_context_options)
    assert report["mandatory_conditioning_coverage_complete"]
    assert report["recorded_tensor_coverage_complete"]
    assert report["comparisons"]["A_replay"]["reproduction_within_tolerance"]
    native = next(r for r in report["arms"]["A_replay"] if r["boundary"] == "native_input_video")
    assert "callbacks" in native["metadata"]["unobserved_backend_options"]
    assert not report["physical_state_isolation_verified"]


@pytest.mark.parametrize("field", ["audio_mask", "ticks", "audio_prefix_ticks", "mask_kind"])
def test_audio_context_semantic_fields_remain_mandatory(trace, field):
    def options(arm):
        result = audio_context_options(arm)
        context = result["h3_flow_partitioned_audio_model_timestep_context_v1"]
        if arm == "A_replay":
            setattr(
                context,
                field,
                {
                    "audio_mask": torch.zeros(1, 1, 2, 4),
                    "ticks": 1,
                    "audio_prefix_ticks": 3,
                    "mask_kind": "guided_overlap",
                }[field],
            )
        return result

    report = native_pair(trace, lambda arm: None, options=options)
    assert report["mandatory_conditioning_coverage_complete"]
    assert report["comparisons"]["A_replay"]["first_observed_divergence"]["boundary"] == "native_conditioning"
    assert not report["comparisons"]["A_replay"]["reproduction_within_tolerance"]


def test_audio_counter_drift_does_not_hide_first_native_av_divergence_operands(trace):
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    contexts = []

    def solver(x, arm):
        options = audio_context_options(arm)
        contexts.append(options["h3_flow_partitioned_audio_model_timestep_context_v1"])
        trace.begin_prediction(guider, x, torch.ones(1), options)
        values = native_trace_wrapper(
            lambda values, *args, **kwargs: [v + (0 if arm == "A" else 0.125) for v in values],
            [x[:, :2], x[:, 2:]],
            torch.ones(1),
            torch.ones(1),
            options,
        )
        trace.end_prediction(torch.cat(values, dim=1), actual=True, phase="model", outer=5)

    for arm in ("A", "A_replay"):
        with trace_stage(trace, arm, "low", SimpleNamespace(sampler_function=solver), guider):
            solver(torch.ones(1, 4), arm)
    report = trace.summary()
    assert report["mandatory_conditioning_coverage_complete"]
    assert report["comparisons"]["A_replay"]["first_observed_divergence"]["boundary"] == "native_output_video"
    assert not report["comparisons"]["A_replay"]["reproduction_within_tolerance"]
    operands = trace.divergent_operands()
    assert torch.equal(operands["first_divergence_A_replay"], operands["first_divergence_A"] + 0.125)
    assert [(c.calls, c.verification_calls) for c in contexts] == [(0, 0), (22, 22)]


def test_forecast_option_difference_is_labeled_at_predictor_not_native_boundary(trace):
    guider = SimpleNamespace(model_options={"transformer_options": {}})

    def solver(x, arm):
        options = {"denoise_mask": torch.tensor([0.0 if arm == "A" else 1.0])}
        trace.begin_prediction(guider, x, torch.ones(1), options)
        trace.end_prediction(x.clone(), actual=False, phase="model", outer=0)

    for arm in ("A", "A_replay"):
        with trace_stage(trace, arm, "low", SimpleNamespace(sampler_function=solver), guider):
            solver(torch.ones(1, 4), arm)
    report = trace.summary()["comparisons"]["A_replay"]
    first = report["first_observed_divergence"]
    assert first["boundary"] == "predictor_conditioning"
    assert first["record_boundary"] == "predict_input"
    assert not any(r["boundary"].startswith("native_") for r in trace.arms["A"])
    assert not report["reproduction_within_tolerance"]


def test_backend_objects_are_caveated_without_obscuring_mandatory_options(trace):
    import uuid

    report = native_pair(
        trace,
        lambda arm: None,
        options=lambda arm: {
            "wrappers": {"opaque": SimpleNamespace(changed=arm)},
            "spectrum_h3_runtime": SimpleNamespace(cache=arm),
            "optimized_attention_override": object(),
            "minimax_h3_sigma_shift_video": 3.0,
            "uuids": [uuid.UUID(int=1)],
        },
    )
    assert report["comparisons"]["A_replay"]["reproduction_within_tolerance"]
    assert report["mandatory_conditioning_coverage_complete"]
    assert not report["physical_state_isolation_verified"]
    row = next(r for r in report["arms"]["A"] if r["boundary"] == "native_input_video")
    assert "wrappers" in row["metadata"]["unobserved_backend_options"]


def test_changed_core_conditioning_uuid_is_observed_as_data(trace):
    import uuid

    report = native_pair(
        trace,
        lambda arm: None,
        options=lambda arm: {
            "uuids": [uuid.UUID(int=1 if arm == "A" else 2)],
        },
    )
    assert report["mandatory_conditioning_coverage_complete"]
    assert report["comparisons"]["A_replay"]["first_observed_divergence"]["boundary"] == "native_conditioning"
    assert not report["comparisons"]["A_replay"]["reproduction_within_tolerance"]


def test_partitioned_prefix_and_noise_are_semantic_data_and_metrics_are_not(trace):
    from h3_flow_regenerate.partitioned_stage import PartitionedStagePlan, PartitionedStageRuntime

    def options(arm):
        prefix = torch.full((1, 24, 1, 4, 4), 0.0 if arm == "A" else 1.0)
        owner = PartitionedStageRuntime(
            PartitionedStagePlan(prefix, 3, 2, 2, torch.zeros_like(prefix)),
            SimpleNamespace(counter=arm),
        )
        return {"h3_flow_partitioned_stage_v1": owner}

    report = native_pair(trace, lambda arm: None, options=options)
    assert report["mandatory_conditioning_coverage_complete"]
    assert report["comparisons"]["A_replay"]["first_observed_divergence"]["boundary"] == "native_conditioning"


def test_digest_traversal_and_noncontiguous_copy_chunks_are_bounded(monkeypatch):
    import h3_flow_regenerate.low_sampler_trace as module

    bounded = _digest(list(range(10000)), max_nodes=10)
    assert not bounded["complete"]
    assert bounded["omitted_count"] == 1
    assert not _digest({"opaque": object()})["complete"]
    view = torch.arange(16.0).reshape(4, 4).t()
    assert _full_sha(view) == _full_sha(view.contiguous())
    monkeypatch.setattr(module, "HASH_CHUNK_BYTES", 16)
    copied = []
    original_to = torch.Tensor.to

    def observe_to(tensor, *args, **kwargs):
        copied.append(tensor.numel() * tensor.element_size())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observe_to)
    assert _full_sha(view) == _full_sha(view.contiguous())
    assert max(copied) <= 16
    device_view = torch.empty(4, 4, device="meta").t()
    with pytest.raises(ValueError, match="no observable strided storage"):
        _full_sha(device_view)
    assert not _digest(device_view)["complete"]


@pytest.mark.parametrize("layout", ["broadcast", "permuted", "stepped", "scalar", "empty"])
def test_strided_hash_preserves_full_logical_byte_order(monkeypatch, layout):
    import hashlib

    import h3_flow_regenerate.low_sampler_trace as module

    monkeypatch.setattr(module, "HASH_CHUNK_BYTES", 16)
    values = torch.arange(64, dtype=torch.bfloat16).reshape(2, 4, 8)
    value = {
        "broadcast": values[:1].expand(3, 4, 8),
        "permuted": values.permute(2, 0, 1),
        "stepped": values[:, ::2, ::3],
        "scalar": torch.tensor(-0.0),
        "empty": values[:, :0],
    }[layout]
    expected = hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    assert _full_sha(value) == expected


def test_backend_history_per_receipt_and_invocation_budgets_are_hard_limits(trace, monkeypatch):
    import h3_flow_regenerate.low_sampler_trace as module

    monkeypatch.setattr(module, "BACKEND_RECEIPT_BYTES", 64)
    trace.backend_bytes_remaining = 96
    history = [SimpleNamespace(coordinate=0, feature_flat=torch.ones(16))]
    runtime = SimpleNamespace(forecaster=SimpleNamespace(_history=history))
    guider = SimpleNamespace(
        model_options={"spectrum_h3_binding": SimpleNamespace(runtime=runtime), "transformer_options": {}}
    )
    sampler = SimpleNamespace(sampler_function=lambda: None)
    with trace_stage(trace, "A", "low", sampler, guider):
        receipts = [backend_receipt(guider, {}) for _ in range(4)]
    assert sum(r["inspection_budget"]["hashed_bytes"] for r in receipts) <= 96
    assert all(r["inspection_budget"]["hashed_bytes"] <= 64 for r in receipts)
    assert receipts[0]["spectrum"]["state_digest_complete"]
    assert not receipts[-1]["spectrum"]["state_digest_complete"]
    assert trace.mandatory_conditioning_complete
    assert not trace.truncated


def test_backend_budget_reserves_observation_for_each_replay_arm(monkeypatch):
    import h3_flow_regenerate.low_sampler_trace as module

    monkeypatch.setattr(module, "BACKEND_RECEIPT_BYTES", 64)
    monkeypatch.setattr(module, "BACKEND_TOTAL_BYTES", 192)
    runtime = SimpleNamespace(
        forecaster=SimpleNamespace(_history=[SimpleNamespace(coordinate=0, feature_flat=torch.ones(16))])
    )
    guider = SimpleNamespace(
        model_options={"spectrum_h3_binding": SimpleNamespace(runtime=runtime), "transformer_options": {}}
    )
    observer = LowReplayTrace()
    try:
        for arm in ("A", "B_target_projection", "A_replay"):
            with trace_stage(observer, arm, "low", SimpleNamespace(sampler_function=lambda: None), guider):
                assert backend_receipt(guider, {})["inspection_budget"]["hashed_bytes"] == 64
                assert backend_receipt(guider, {})["inspection_budget"]["hashed_bytes"] == 0
        assert observer.backend_bytes_remaining == 0
    finally:
        observer.close()


@pytest.mark.parametrize("effect", ["none", "rng", "tensor"])
def test_user_callback_effects_preserve_execution_and_do_not_assert_history_equivalence(trace, effect):
    emit(trace, "A")
    emit(trace, "A_replay")
    trace.arm, trace.stage, trace.evaluation = "A", "low", 2
    x = torch.ones(1, 4)
    state = torch.get_rng_state().clone()
    try:
        with trace.user_callback(2, x.clone(), x):
            if effect == "rng":
                torch.rand(1)
            elif effect == "tensor":
                x.add_(1)
        report = trace.summary()
        receipt = report["callback_provenance"]["effects"][0]
        assert receipt["rng_changed"] == (effect == "rng")
        assert receipt["sampler_tensors_changed"] == (effect == "tensor")
        assert not report["callback_provenance"]["execution_history_equivalence_verified"]
        assert report["comparisons"]["A_replay"]["reproduction_within_tolerance"] == (effect == "none")
        assert torch.equal(torch.get_rng_state(), state) == (effect != "rng")
        assert torch.equal(x, torch.full_like(x, 2.0 if effect == "tensor" else 1.0))
    finally:
        torch.set_rng_state(state)


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
        values = native_trace_wrapper(
            lambda values, *a, **kw: [v * 0.25 for v in values],
            [x[:, :2], x[:, 2:]],
            sigma,
            x,
            {},
        )
        result = torch.cat(values, dim=1)
        trace.end_prediction(result, actual=True, phase="model", outer=trace.evaluation)
        return result

    expected = sample_euler(lambda x, sigma, **kw: x * 0.25, initial.clone(), sigmas, disable=True)
    for arm in ("A", "A_replay"):
        with trace_stage(trace, arm, "low", sampler, guider):
            actual = sample_euler(model, initial.clone(), sigmas, disable=True)
            assert torch.equal(actual, expected)
            assert sampler.sampler_function is sample_euler
    assert trace.summary()["comparisons"]["A_replay"]["fully_observed_records_exact"]


@pytest.mark.parametrize("guidance_mode", [None, "direction", "unbound"])
def test_flow_predict_and_native_wrapper_preserve_actual_forecast_provenance(trace, guidance_mode):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    from comfy.patcher_extension import WrapperExecutor, get_all_wrappers

    from h3_flow_regenerate.guidance import GuidanceConfig
    from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FlowBinding, flow_predict_wrapper

    options = {"transformer_options": {"h3_flow_stage": "low"}}
    binding = FlowBinding(guidance=GuidanceConfig(mode=guidance_mode) if guidance_mode == "direction" else None)
    assert binding.active_guidance_run is None
    if guidance_mode != "unbound":
        options[FLOW_BINDING_KEY] = binding
    guider = SimpleNamespace(model_options=options)
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

    def solver(x, timestep):
        call = WrapperExecutor.new_class_executor(predictor, guider, [flow_predict_wrapper])
        return call.execute(x, timestep, options, 42)

    sampler = SimpleNamespace(sampler_function=solver)
    observed = []
    for arm in ("A", "A_replay"):
        values = []
        for stage in ("low", "probe"):
            options["transformer_options"]["h3_flow_stage"] = stage
            options["transformer_options"].pop("h3_flow_exact_probe_context", None)
            if stage == "probe":
                options["transformer_options"]["h3_flow_exact_probe_context"] = {"outer_step": 7}
            with trace_stage(trace, arm, stage, sampler, guider):
                for step in range(3 if stage == "low" else 1):
                    control["step"] = step
                    values.append(solver(torch.ones(1, 4), torch.tensor([1.0 - step * 0.25])))
        observed.append(values)
    assert all(torch.equal(a, b) for a, b in zip(*observed, strict=True))
    calls = [r for r in trace.arms["A"] if r["boundary"] == "predict_output"]
    assert [r["metadata"]["actual"] for r in calls] == [*actuals, True]
    assert [r["metadata"]["outer_step"] for r in calls] == [0, 1, 2, 7]
    assert [r["metadata"]["phase"] for r in calls] == ["model", "model", "model", "handoff_probe"]
    assert len([r for r in trace.arms["A"] if r["boundary"] == "native_output_audio"]) == 3
    if guidance_mode != "unbound":
        assert binding.metrics.counters["spectrum_forecast_calls"] == 2
        assert binding.metrics.counters["transformer_actual_nfe"] == 6
        assert binding.metrics.counters["sampler_logical_calls"] == 8
    else:
        assert not binding.metrics.counters
    assert torch.equal(observed[0][0], torch.tensor([[0.25, 0.25, 0.5, 0.5]]))
    assert torch.equal(observed[0][1], torch.full((1, 4), 0.125))
    report = trace.summary()
    assert report["recorded_tensor_coverage_complete"]
    assert report["comparisons"]["A_replay"]["fully_observed_records_exact"]
    assert active_trace() is None
