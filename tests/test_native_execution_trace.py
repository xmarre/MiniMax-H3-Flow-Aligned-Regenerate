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
            if layer == 0:
                observer.observe(trace, 0, layers, "input", value, module, {}, None)
            observer.observe(trace, layer, layers, "output", value, module, {}, None)
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
            if layer == 0:
                observer.observe(trace, 0, 5, "input", value, module, {}, None)
            observer.observe(trace, layer, 5, "output", value, module, {}, None)
    report = observer.summary()
    assert report["comparisons"]["A_replay"]["selected_boundaries_exact"]
    assert report["arms"]["A"][3]["coverage"] == "token_rows_stride_32"
    assert any("Sampled equality does not establish" in note for note in report["limitations"])


def test_01867_full_pair_hashes_complete_neighboring_blocks_with_exact_coverage():
    observer = NativeExecutionTrace(5, _full_sha, pair_block=45)
    module = block()
    for arm in ("A", "A_replay"):
        trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
        for layer in range(50):
            rows = torch.arange(128.0).reshape(32, 4)
            observer.observe(trace, layer, 50, "input", rows, module, {}, None)
            if arm == "A_replay" and layer == 45:
                rows = rows.clone()
                rows[1, 0] += 0.125  # Missed by one-in-32 sampled probe.
            observer.observe(trace, layer, 50, "output", rows, module, {}, None)
    report = observer.summary()
    a_rows = report["arms"]["A"]
    assert [row["key"][-2:] for row in a_rows] == [
        [0, "input"],
        [44, "output"],
        [45, "input"],
        [45, "output"],
        [49, "output"],
    ]
    assert all(row["coverage"] == "full" for row in a_rows)
    assert all(row["source_shape"] == row["shape"] for row in a_rows)
    assert report["comparisons"]["A_replay"]["selected_boundaries_complete"]
    assert report["comparisons"]["A_replay"]["first_observed_difference"] == {
        "key": ["low", 5, 1, 45, "output"],
        "classification": "packed_hidden_output",
        "coverage": "full",
    }
    assert observer.bytes["A"] == 5 * 128 * 4


@pytest.mark.parametrize("block", [-2, 512, True, 1.5])
def test_invalid_pair_selector_rejected_before_sampling(block):
    from h3_flow_regenerate.native_execution_trace import validate_pair_block

    with pytest.raises(ValueError, match="low_native_trace_pair_block"):
        validate_pair_block(block)
    with pytest.raises(ValueError, match="low_native_trace_pair_block"):
        node_call(low_native_trace_pair_block=block)


def test_pair_selector_default_off_and_requires_selected_native_prediction():
    from h3_flow_regenerate.partitioned_node import H3PartitionedExactPrefixDiagnosticHandoff

    spec = H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]
    assert spec["low_native_trace_pair_block"][1]["default"] == -1
    with pytest.raises(ValueError, match="low_native_trace_pair_block requires"):
        node_call(low_native_trace_pair_block=45)
    with pytest.raises(ValueError, match="low_native_trace_pair_block requires"):
        LowReplayTrace(native_pair_block=45)
    with pytest.raises(ValueError, match="middle\\+2"):
        NativeExecutionTrace(5, _full_sha, pair_block=25).selected_outputs(50)


def test_full_pair_missing_or_over_budget_fails_closed():
    module = block()
    trace = SimpleNamespace(arm="A", stage="low", evaluation=5, native_call=1)
    observer = NativeExecutionTrace(5, _full_sha, pair_block=45, arm_bytes=4 * 128 * 4)
    for layer in range(50):
        value = torch.arange(128.0).reshape(32, 4)
        observer.observe(trace, layer, 50, "input", value, module, {}, None)
        observer.observe(trace, layer, 50, "output", value, module, {}, None)
    for arm in ("B_target_projection", "A_replay"):
        other = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
        for layer in range(50):
            value = torch.arange(128.0).reshape(32, 4)
            observer.observe(other, layer, 50, "input", value, module, {}, None)
            observer.observe(other, layer, 50, "output", value, module, {}, None)
    assert not observer.summary()["comparisons"]["A_replay"]["selected_boundaries_complete"]
    assert observer.summary()["arms"]["A"][-1]["unobserved"] == "tensor_or_arm_byte_budget"


def test_01881_module_detail_brackets_attention_residual_and_mlp():
    observer = NativeExecutionTrace(5, _full_sha, pair_block=45, module_detail=True)
    module = torch.nn.Module()
    module.attn = torch.nn.Module()
    module.attn.out_proj = torch.nn.Identity()
    module.norm2 = torch.nn.Identity()
    module.mlp = torch.nn.Identity()
    for arm in ("A", "B_target_projection", "A_replay"):
        trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
        value = torch.ones(6, 4)
        observer.observe(trace, 45, 50, "input", value, module, {}, None)
        with observer.internal_block_hooks(trace, 45, 50, module, {}, None):
            projected = module.attn.out_proj(value + (0.125 if arm == "A_replay" else 0.0))
            normalized = module.norm2(projected)
            mlp_out = module.mlp(normalized)
        observer.observe(trace, 45, 50, "output", mlp_out, module, {}, None)
    report = observer.summary()
    assert report["internal_module_detail"]
    for arm in ("A", "B_target_projection", "A_replay"):
        rows = report["arms"][arm]
        assert [row["key"][-1] for row in rows] == [
            "input",
            "attention_out_proj",
            "post_attention_norm2",
            "mlp_output",
            "output",
        ]
        assert all(row["coverage"] == "full" and row["sha256"] for row in rows)
    assert report["comparisons"]["A_replay"]["alignment_complete"]
    assert report["comparisons"]["A_replay"]["selected_boundaries_complete"]
    assert report["comparisons"]["A_replay"]["first_observed_difference"]["key"][-1] == "attention_out_proj"
    assert not module.attn.out_proj._forward_hooks
    assert not module.norm2._forward_hooks
    assert not module.mlp._forward_hooks


def test_module_detail_fails_closed_if_module_skipped():
    observer = NativeExecutionTrace(5, _full_sha, pair_block=45, module_detail=True)
    module = torch.nn.Module()
    module.attn = torch.nn.Module()
    module.attn.out_proj = torch.nn.Identity()
    module.norm2 = torch.nn.Identity()
    module.mlp = torch.nn.Identity()
    for arm in ("A", "A_replay"):
        trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
        value = torch.ones(6, 4)
        observer.observe(trace, 45, 50, "input", value, module, {}, None)
        with observer.internal_block_hooks(trace, 45, 50, module, {}, None):
            module.attn.out_proj(value)
            module.mlp(value)  # missing norm2 must not be marked complete
        observer.observe(trace, 45, 50, "output", value, module, {}, None)
    assert not observer.summary()["comparisons"]["A_replay"]["selected_boundaries_complete"]


def test_module_detail_opt_in_validation():
    from h3_flow_regenerate.native_execution_trace import validate_module_detail
    from h3_flow_regenerate.partitioned_node import H3PartitionedExactPrefixDiagnosticHandoff

    spec = H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]
    assert spec["low_native_trace_module_detail"][1]["default"] is False
    with pytest.raises(ValueError, match="low_native_trace_module_detail"):
        validate_module_detail(1)
    with pytest.raises(ValueError, match="low_native_trace_module_detail"):
        node_call(low_native_trace_module_detail=True)
    with pytest.raises(ValueError, match="low_native_trace_module_detail"):
        LowReplayTrace(native_step=5, native_module_detail=True)
    assert NativeExecutionTrace(5, _full_sha, pair_block=45).module_detail is False


def test_01891_projection_detail_brackets_raw_and_post_forward_lora():
    class Projection(torch.nn.Module):
        def forward(self, value):
            return value[:, :4] + 0.5

    module = torch.nn.Module()
    module.attn = torch.nn.Module()
    module.attn.out_proj = Projection()
    active_adapter = {"offset": 0.25}

    def existing_lora(_module, _inputs, value):
        return value + active_adapter["offset"]

    lora_handle = module.attn.out_proj.register_forward_hook(existing_lora)
    observer = NativeExecutionTrace(5, _full_sha, pair_block=45, module_detail=True, projection_detail=True)
    try:
        for arm in ("A", "B_target_projection", "A_replay"):
            trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
            value = torch.ones(6, 4)
            observer.observe(trace, 45, 50, "input", value, module, {}, None)
            if arm == "A_replay":
                active_adapter["offset"] = 0.5  # unchanged input/raw projection, changed adapter output
            with observer.internal_block_hooks(trace, 45, 50, module, {}, None):
                result = module.attn.out_proj(torch.ones(6, 8))
            observer.observe(trace, 45, 50, "output", result, module, {}, None)
            assert tuple(module.attn.out_proj._forward_hooks.values()) == (existing_lora,)
            assert not module.attn.out_proj._forward_pre_hooks
        report = observer.summary()
        assert report["internal_projection_detail"]
        assert report["comparisons"]["A_replay"]["selected_boundaries_complete"]
        assert report["comparisons"]["A_replay"]["first_observed_difference"]["key"][-1] == "out_proj_post_adapter"
        for rows in report["arms"].values():
            assert [r["key"][-1] for r in rows] == [
                "input",
                "out_proj_input",
                "out_proj_raw",
                "out_proj_post_adapter",
                "output",
            ]
            assert all(r["coverage"] == "full" for r in rows)
    finally:
        lora_handle.remove()


def test_01891_projection_detail_identifies_changed_attention_input_and_cleans_on_exception():
    class Projection(torch.nn.Module):
        def forward(self, value):
            return value[:, :4]

    module = torch.nn.Module()
    module.attn = torch.nn.Module()
    module.attn.out_proj = Projection()
    observer = NativeExecutionTrace(5, _full_sha, pair_block=45, module_detail=True, projection_detail=True)
    for arm in ("A", "A_replay"):
        trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
        value = torch.ones(6, 4)
        observer.observe(trace, 45, 50, "input", value, module, {}, None)
        with observer.internal_block_hooks(trace, 45, 50, module, {}, None):
            result = module.attn.out_proj(torch.ones(6, 8) + (0.125 if arm == "A_replay" else 0.0))
        observer.observe(trace, 45, 50, "output", result, module, {}, None)
    assert observer.summary()["comparisons"]["A_replay"]["first_observed_difference"]["key"][-1] == "out_proj_input"
    trace = SimpleNamespace(arm="A", stage="low", evaluation=5, native_call=1)
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        observer.internal_block_hooks(trace, 45, 50, module, {}, None),
    ):
        raise RuntimeError("deliberate")
    assert not module.attn.out_proj._forward_hooks
    assert not module.attn.out_proj._forward_pre_hooks


def test_01891_core_filesystem_module_identity_recovers_matching_pooled_stats():
    code = compile(
        "def bind(patch):\n    def previous(args, extra):\n        return patch.pooled\n    return previous\n",
        "/home/toor/ComfyUI/comfy_extras/nodes_sparse_attention.py",
        "exec",
    )
    ns = {}
    exec(code, ns)
    cls = type(
        "SparseAttnPatch",
        (),
        {"__module__": "/home/toor/ComfyUI/comfy_extras/nodes_sparse_attention"},
    )
    patch = cls()
    patch.pooled = {(45, 7, ("stable-uuid",)): (torch.ones(56, 128), torch.ones(56, 128))}
    observer = NativeExecutionTrace(5, _full_sha, pair_block=45, module_detail=True, projection_detail=True)
    receipt = observer.core_sparse_stats(torch.ones(7, 4), 45, {"uuids": ("stable-uuid",)}, ns["bind"](patch))
    assert receipt["available"] is True
    assert receipt["present"] is True
    assert receipt["total_cached_keys"] == 1
    assert len(receipt["statistics"]) == 2
    assert all(s["sha256"] for s in receipt["statistics"])
    assert not observer.core_sparse_stats(torch.ones(8, 4), 45, {"uuids": ("stable-uuid",)}, ns["bind"](patch))[
        "present"
    ]


def test_01891_projection_selector_fails_closed_and_geometry_stays_within_budget():
    from h3_flow_regenerate.native_execution_trace import validate_projection_detail
    from h3_flow_regenerate.partitioned_node import H3PartitionedExactPrefixDiagnosticHandoff

    assert (
        H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]["low_native_trace_projection_detail"][1][
            "default"
        ]
        is False
    )
    with pytest.raises(ValueError, match="projection_detail"):
        validate_projection_detail(1)
    with pytest.raises(ValueError, match="projection_detail"):
        node_call(low_native_trace_projection_detail=True)
    with pytest.raises(ValueError, match="projection_detail"):
        LowReplayTrace(native_step=5, native_pair_block=45, native_projection_detail=True)
    with pytest.raises(ValueError, match="projection_detail"):
        NativeExecutionTrace(5, _full_sha, pair_block=45, projection_detail=True)
    n_rows = 36445
    bf16_bytes = 2
    hidden = 5376
    out_proj_input = 7168
    total = bf16_bytes * n_rows * (4 * hidden + out_proj_input)
    assert total == 2089902080
    assert total < 2 * 1024**3
    assert bf16_bytes * n_rows * out_proj_input < 512 * 1024**2


def test_01892_chunked_qkv_full_coverage_localizes_changed_adapter_output():
    class QKV(torch.nn.Module):
        def forward(self, value):
            return torch.cat((value[:, :2], value[:, :2], value[:, :2]), dim=-1)

    module = torch.nn.Module()
    module.attn = torch.nn.Module()
    module.attn.heads = 1
    module.attn.head_dim = 2
    module.attn.qkv_proj = QKV()
    adapter = {"strength": 0}

    def original_adapter(_module, _args, value):
        return value + adapter["strength"] * 0.125

    handle = module.attn.qkv_proj.register_forward_hook(original_adapter)
    observer = NativeExecutionTrace(
        5, _full_sha, pair_block=45, module_detail=True, projection_detail=True, qkv_detail=True
    )
    observer.core_sparse_stats = lambda *_args, **_kwargs: {
        "available": True, "key_observed": True, "present": True,
        "statistics": [{"sha256": "k"}, {"sha256": "v"}],
    }
    try:
        for arm in ("A", "B_target_projection", "A_replay"):
            if arm == "A_replay":
                adapter["strength"] = 1
            trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
            source = torch.ones((9000, 4), dtype=torch.bfloat16)
            observer.observe(trace, 45, 50, "input", source, module, {}, None)
            with observer.internal_block_hooks(trace, 45, 50, module, {}, None):
                for start in range(0, len(source), 4096):
                    module.attn.qkv_proj(source[start : start + 4096])
            assert tuple(module.attn.qkv_proj._forward_hooks.values()) == (original_adapter,)
            assert not module.attn.qkv_proj._forward_pre_hooks
        report = observer.summary()
        assert report["internal_qkv_detail"]
        assert report["chunked_qkv_expected"]["A"]["rows"] == 9000
        assert report["comparisons"]["A_replay"]["selected_boundaries_complete"]
        assert report["comparisons"]["A_replay"]["first_observed_difference"]["key"][-1] == "qkv_output_0"
        assert [row["key"][-1] for row in report["arms"]["A"]] == [
            "qkv_input_0",
            "qkv_output_0",
            "qkv_input_1",
            "qkv_output_1",
            "qkv_input_2",
            "qkv_output_2",
        ]
        assert all(row["coverage"] == "full" for row in report["arms"]["A"])
        assert report["inspection_bytes_charged_per_arm"]["A"] == 9000 * (4 + 6) * 2
    finally:
        handle.remove()


def test_01892_chunked_qkv_missing_chunk_and_budget_fail_closed():
    module = torch.nn.Module()
    module.attn = torch.nn.Module()
    module.attn.heads = 1
    module.attn.head_dim = 2
    module.attn.qkv_proj = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    for omit_last, budget in ((True, 2 * 1024**3), (False, 100)):
        observer = NativeExecutionTrace(
            5,
            _full_sha,
            pair_block=45,
            module_detail=True,
            projection_detail=True,
            qkv_detail=True,
            arm_bytes=budget,
        )
        observer.core_sparse_stats = lambda *_args, **_kwargs: {
            "available": True, "key_observed": True, "present": True,
            "statistics": [{"sha256": "k"}, {"sha256": "v"}],
        }
        for arm in ("A", "A_replay"):
            trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
            value = torch.ones(5000, 4, dtype=torch.bfloat16)
            observer.observe(trace, 45, 50, "input", value, module, {}, None)
            with observer.internal_block_hooks(trace, 45, 50, module, {}, None):
                module.attn.qkv_proj(value[:4096])
                if not omit_last:
                    module.attn.qkv_proj(value[4096:])
        assert not observer.summary()["comparisons"]["A_replay"]["selected_boundaries_complete"]
        assert not module.attn.qkv_proj._forward_hooks
        assert not module.attn.qkv_proj._forward_pre_hooks
    with pytest.raises(ValueError, match="qkv_detail"):
        NativeExecutionTrace(5, _full_sha, pair_block=45, qkv_detail=True)


def test_01892_qkv_selector_validation_and_cleanup_exception():
    from h3_flow_regenerate.native_execution_trace import validate_qkv_detail
    from h3_flow_regenerate.partitioned_node import H3PartitionedExactPrefixDiagnosticHandoff

    assert (
        H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]["low_native_trace_qkv_detail"][1]["default"]
        is False
    )
    with pytest.raises(ValueError, match="qkv_detail"):
        validate_qkv_detail(1)
    with pytest.raises(ValueError, match="qkv_detail"):
        node_call(low_native_trace_qkv_detail=True)
    with pytest.raises(ValueError, match="qkv_detail"):
        LowReplayTrace(native_step=5, native_pair_block=45, native_qkv_detail=True)
    module = torch.nn.Module()
    module.attn = torch.nn.Module()
    module.attn.qkv_proj = torch.nn.Identity()
    observer = NativeExecutionTrace(
        5, _full_sha, pair_block=45, module_detail=True, projection_detail=True, qkv_detail=True
    )
    trace = SimpleNamespace(arm="A", stage="low", evaluation=5, native_call=1)
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        observer.internal_block_hooks(trace, 45, 50, module, {}, None),
    ):
        raise RuntimeError("deliberate")
    assert not module.attn.qkv_proj._forward_hooks
    assert not module.attn.qkv_proj._forward_pre_hooks
    assert 36445 * (5376 + 21504) * 2 == 1959283200
    assert 1959283200 < 2 * 1024**3


def test_01893_qkv_cold_bootstrap_two_passes_capture_only_attention_producer():
    class QKV(torch.nn.Module):
        def forward(self, value):
            return torch.cat((value[:, :2], value[:, :2], value[:, :2]), dim=-1)

    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.heads = 1
    model.attn.head_dim = 2
    model.attn.qkv_proj = QKV()
    adapter = {"strength": 0}

    def existing_vdn_hook(_module, _args, result):
        return result + adapter["strength"] * 0.125

    existing_handle = model.attn.qkv_proj.register_forward_hook(existing_vdn_hook)
    observer = NativeExecutionTrace(
        5, _full_sha, pair_block=45, module_detail=True, projection_detail=True, qkv_detail=True
    )
    observer.core_sparse_stats = lambda *_args, **_kwargs: {
        "available": True, "key_observed": True, "present": False, "statistics": []
    }
    source = torch.ones((9000, 4), dtype=torch.bfloat16)
    try:
        for arm in ("A", "B_target_projection", "A_replay"):
            trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
            observer.observe(trace, 45, 50, "input", source, model, {}, None)
            assert observer.qkv_expected[arm]["producer_passes_expected"] == 2
            with observer.internal_block_hooks(trace, 45, 50, model, {}, None):
                for bootstrap in (True, False):
                    adapter["strength"] = int(arm == "A_replay" and not bootstrap)
                    for start in range(0, len(source), 4096):
                        model.attn.qkv_proj(source[start : start + 4096])
            assert tuple(model.attn.qkv_proj._forward_hooks.values()) == (existing_vdn_hook,)
            assert not model.attn.qkv_proj._forward_pre_hooks
        summary = observer.summary()
        assert summary["comparisons"]["A_replay"]["selected_boundaries_complete"]
        assert summary["comparisons"]["A_replay"]["first_observed_difference"]["key"][-1] == "qkv_output_0"
        assert summary["comparisons"]["B_target_projection"]["selected_boundaries_exact"]
        assert len(summary["arms"]["A"]) == 6
        assert summary["chunked_qkv_observed_calls"]["A"] == {"inputs": 6, "outputs": 6}
        assert all(r["sha256"] for r in summary["arms"]["A"])
        assert summary["inspection_bytes_charged_per_arm"]["A"] == 9000 * (4 + 6) * 2
        assert summary["limits"]["qkv_capture_policy"] == "last_core_sparse_producer_pass_v1"
    finally:
        existing_handle.remove()


def test_01893_qkv_cold_single_or_three_passes_and_unknown_state_fail_closed():
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.heads = 1
    model.attn.head_dim = 2
    model.attn.qkv_proj = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    source = torch.ones((5000, 4), dtype=torch.bfloat16)
    for producer_passes in (1, 3):
        observer = NativeExecutionTrace(
            5, _full_sha, pair_block=45, module_detail=True, projection_detail=True, qkv_detail=True
        )
        observer.core_sparse_stats = lambda *_args, **_kwargs: {
            "available": True, "key_observed": True, "present": False, "statistics": []
        }
        for arm in ("A", "A_replay"):
            trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
            observer.observe(trace, 45, 50, "input", source, model, {}, None)
            with observer.internal_block_hooks(trace, 45, 50, model, {}, None):
                for _ in range(producer_passes):
                    model.attn.qkv_proj(source[:4096])
                    model.attn.qkv_proj(source[4096:])
        report = observer.summary()
        assert not report["comparisons"]["A_replay"]["selected_boundaries_complete"]
        assert report["chunked_qkv_observed_calls"]["A"] == {
            "inputs": 2 * producer_passes, "outputs": 2 * producer_passes
        }
        assert not model.attn.qkv_proj._forward_hooks
        assert not model.attn.qkv_proj._forward_pre_hooks

    observer = NativeExecutionTrace(
        5, _full_sha, pair_block=45, module_detail=True, projection_detail=True, qkv_detail=True
    )
    observer.core_sparse_stats = lambda *_args, **_kwargs: {
        "available": False, "key_observed": False
    }
    for arm in ("A", "A_replay"):
        trace = SimpleNamespace(arm=arm, stage="low", evaluation=5, native_call=1)
        observer.observe(trace, 45, 50, "input", source, model, {}, None)
        with observer.internal_block_hooks(trace, 45, 50, model, {}, None):
            for start in range(0, len(source), 4096):
                model.attn.qkv_proj(source[start : start + 4096])
    report = observer.summary()
    assert not report["comparisons"]["A_replay"]["selected_boundaries_complete"]
    assert report["chunked_qkv_expected"]["A"]["producer_passes_expected"] is None
    assert report["arms"].get("A", []) == []
