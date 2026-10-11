"""Native observer selection, full-byte coverage and real VDN lease ownership."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.low_sampler_trace import LowReplayTrace, _full_sha
from h3_flow_regenerate.native_execution_trace import NativeExecutionTrace, _metadata, validate_step


def block():
    module = torch.nn.Identity()
    module.attn = torch.nn.Identity()
    return module


def emit(observer, arm, *, changed=None, layers=5):
    trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
    module = block()
    for layer in range(layers):
        for boundary in ("input", "output"):
            value = torch.arange(24.0).reshape(4, 6).t()
            if (layer, boundary) == changed:
                value = value.clone()
                value[-1, -1] += 0.125
            observer.observe(trace, layer, layers, boundary, value, module, {}, None)


@pytest.mark.parametrize("layers", [1, 2, 5])
def test_complete_selected_full_byte_hashes_and_coarse_difference(layers):
    observer = NativeExecutionTrace(5, _full_sha)
    emit(observer, "A", layers=layers)
    emit(observer, "A_replay", layers=layers, changed=(layers // 2, "output"))
    report = observer.summary()
    comparison = report["comparisons"]["A_replay"]
    assert comparison["selected_boundaries_complete"]
    assert not comparison["selected_boundaries_exact"]
    assert comparison["first_observed_difference"]["key"] == ["low", 5, 1, layers // 2, "output"]
    assert all("sha256" in row for row in report["arms"]["A"])
    assert report["physical_state_isolation_verified"] is False
    assert all(not torch.is_tensor(v) for row in observer.arms["A"] for v in row.values())


def test_missing_same_boundaries_on_both_arms_never_claim_complete():
    observer = NativeExecutionTrace(5, _full_sha)
    emit(observer, "A")
    emit(observer, "A_replay")
    assert observer.summary()["comparisons"]["A_replay"]["selected_boundaries_exact"]
    for rows in observer.arms.values():
        rows.pop(2)
    assert not observer.summary()["comparisons"]["A_replay"]["selected_boundaries_complete"]
    assert not observer.summary()["comparisons"]["A_replay"]["alignment_complete"]


@pytest.mark.parametrize("limits", [{"arm_bytes": 3 * 24 * 4}, {"tensor_bytes": 1}, {"max_records": 2}])
def test_budget_and_record_omissions_are_explicit(limits):
    observer = NativeExecutionTrace(5, _full_sha, **limits)
    emit(observer, "A")
    emit(observer, "A_replay")
    assert not observer.summary()["comparisons"]["A_replay"]["selected_boundaries_complete"]
    assert observer.bytes.get("A", 0) <= observer.arm_bytes
    assert len(observer.arms["A"]) <= observer.max_records


def test_target_scope_forecast_disabled_and_unknown_arm_do_not_hash():
    def forbidden(_value):
        raise AssertionError("unselected observation hashed a tensor")

    observer = NativeExecutionTrace(5, forbidden)
    for changes in ({"evaluation": 4}, {"native_call": 0}, {"native_call": 2}, {"stage": "probe"}, {"arm": "unknown"}):
        trace = SimpleNamespace(**({"arm": "A", "stage": "low", "evaluation": 5, "native_call": 1} | changes))
        observer.observe(trace, 0, 5, "input", torch.ones(1), block(), {}, None)
    assert not observer.arms
    assert observer.summary()["comparisons"]["A_replay"]["missing_selected_native_call"]


def test_real_vdn_primary_nested_and_outside_lease_metadata_are_read_only():
    runtime = pytest.importorskip("vdn_h3.runtime")
    owner = runtime.RuntimeBufferOwner(True)
    state = SimpleNamespace(runtime=owner)

    def forward(*args, **kwargs):
        return state

    forward._vdn_forward = True
    module = block()
    module.attn.forward = forward
    observer = NativeExecutionTrace(5, _full_sha)
    hidden = torch.ones(3, 4)
    assert not observer.backend(hidden, module, {}, None)["vdn"]["active_lease"]
    with owner.execution() as primary:
        banks = primary.scan_banks(3, (2, 2), torch.float32, "cpu")
        banks[0].fill_(float("nan"))
        versions = [t._version for t in banks]
        primary._prefetcher = SimpleNamespace(_generation=17, _index=(2, "cuda:0", torch.bfloat16))
        receipt = observer.backend(hidden, module, {}, None)["vdn"]
        assert receipt["active_lease"] and receipt["primary_pool"] and receipt["owner_pool_matches"]
        assert receipt["owner_id"] == id(owner) and receipt["buffers_id"] == id(primary)
        assert receipt["caches"]["_scan"]["count"] == 1
        assert receipt["prefetch_generation"] == 17
        assert receipt["prefetch_result_and_weights"] == "unobserved"
        assert versions == [t._version for t in banks]
        assert runtime.current_runtime_buffers() is primary
        with owner.execution() as transient:
            nested = observer.backend(hidden, module, {}, None)["vdn"]
            assert nested["buffers_id"] == id(transient) and not nested["primary_pool"]
            assert nested["owner_pool_matches"] and nested["retain"] is False
        assert runtime.current_runtime_buffers() is primary
        primary._prefetcher = None
    assert runtime.current_runtime_buffers() is None


def test_metadata_never_reads_tensor_values_and_bounds_nested_values():
    assert _metadata(torch.empty(100))["shape"] == [100]
    result = _metadata([[list(range(128)) for _ in range(128)] for _ in range(128)])
    assert result["entries_omitted"] > 0


@pytest.mark.parametrize("step", [-2, 512, True, 1.5])
def test_invalid_selector_is_rejected_before_sampling(step):

    with pytest.raises(ValueError, match="low_native_trace_step"):
        validate_step(step)
    with pytest.raises(ValueError, match="low_native_trace_step"):
        node_call(low_native_trace_step=step)


def test_observer_default_is_off_and_requires_frozen_replay():
    from h3_flow_regenerate.partitioned_node import H3PartitionedExactPrefixDiagnosticHandoff

    spec = H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]
    assert spec["low_native_trace_step"][1]["default"] == -1
    with pytest.raises(ValueError, match="requires low_sampler_aba"):
        node_call(low_native_trace_step=0)
    trace = LowReplayTrace()
    try:
        assert trace.native_execution is None
    finally:
        trace.close()


def node_call(**changes):
    from h3_flow_regenerate.partitioned_node import H3PartitionedExactPrefixDiagnosticHandoff

    node = H3PartitionedExactPrefixDiagnosticHandoff()
    spec = node.INPUT_TYPES()["required"]
    values = {"model": None, "trajectory": None}
    for name, parameter in inspect.signature(node.patch).parameters.items():
        if name not in values and parameter.default is inspect.Parameter.empty:
            definition = spec[name]
            settings = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}
            values[name] = settings.get("default")
    return node.patch(**(values | changes))


def test_real_core_flow_block_chain_preserves_av_outputs_and_captures_inside_lease(monkeypatch):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    runtime = pytest.importorskip("vdn_h3.runtime")
    pytest.importorskip("sol_h3.runtime")
    import test_partitioned_target_band_domain as fixture

    from h3_flow_regenerate.low_sampler_trace import active_trace, low_trace_scope, native_trace_wrapper, trace_stage

    fixture._sol(monkeypatch)
    case = fixture._Case()
    baseline = case.source_reference(case.video)
    owner = runtime.RuntimeBufferOwner(True)
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    sampler = SimpleNamespace(sampler_function=lambda: None)
    case.wrappers = [native_trace_wrapper]
    rng = torch.get_rng_state().clone()
    with low_trace_scope(5) as trace:
        for arm in ("A", "A_replay"):
            with trace_stage(trace, arm, "low", sampler, guider), owner.execution():
                trace.evaluation = 5
                result = case.source_reference(case.video)
                for actual, expected in zip(result, baseline, strict=True):
                    assert torch.equal(actual, expected)
        report = trace.native_execution.summary()
        assert report["comparisons"]["A_replay"]["selected_boundaries_exact"]
        assert all(r["backend"]["vdn"]["active_lease"] for r in report["arms"]["A"])
        assert report["arms"]["A"][0]["key"] == ["low", 5, 1, 0, "input"]
        assert report["arms"]["A"][-1]["key"][-1] == "output"
        assert trace.summary()["native_execution_trace"] == report
    assert active_trace() is None and runtime.current_runtime_buffers() is None
    assert torch.equal(torch.get_rng_state(), rng)


def test_01860_late_block_refinement_finds_earliest_sampled_difference():
    observer = NativeExecutionTrace(5, _full_sha)
    module = block()
    layers = 50
    for arm in ("A", "A_replay"):
        trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
        for layer in range(layers):
            value = torch.arange(128.0).reshape(32, 4)
            if arm == "A_replay" and layer >= 37:
                value = value.clone()
                value[0, 0] += 0.125
            observer.observe(trace, layer, layers, "output", value, module, {}, None)
            if layer == 0:
                observer.observe(trace, 0, layers, "input", value, module, {}, None)
    result = observer.summary()
    first = result["comparisons"]["A_replay"]["first_observed_difference"]
    assert result["comparisons"]["A_replay"]["selected_boundaries_complete"]
    assert first == {
        "key": ["low", 5, 1, 37, "output"],
        "classification": "packed_hidden_output_sampled",
        "coverage": "token_rows_stride_32",
    }
    assert len(result["arms"]["A"]) == 27
    assert sum(row["coverage"] == "token_rows_stride_32" for row in result["arms"]["A"]) == 23
    assert observer.bytes["A"] == 4 * 128 * 4 + 23 * 4 * 4


def test_strided_probe_never_claims_full_block_coverage():
    observer = NativeExecutionTrace(5, _full_sha)
    module = block()
    for arm in ("A", "A_replay"):
        trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
        for layer in range(5):
            value = torch.arange(128.0).reshape(32, 4)
            if arm == "A_replay" and layer == 3:
                value = value.clone()
                value[1, 0] += 0.125  # Not among sampled token rows.
            observer.observe(trace, layer, 5, "output", value, module, {}, None)
            if layer == 0:
                observer.observe(trace, 0, 5, "input", value, module, {}, None)
    report = observer.summary()
    assert report["comparisons"]["A_replay"]["selected_boundaries_exact"]
    assert report["arms"]["A"][3]["coverage"] == "token_rows_stride_32"
    assert any("Sampled equality does not establish" in note for note in report["limitations"])
