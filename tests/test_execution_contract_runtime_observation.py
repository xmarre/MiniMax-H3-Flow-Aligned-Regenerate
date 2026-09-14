from __future__ import annotations

import contextvars
import json
import sys
from collections import deque
from types import SimpleNamespace

import torch

from h3_flow_regenerate import execution_contract_diagnostics as diag
from h3_flow_regenerate import execution_contract_runtime_observation as observation


def _state():
    return diag._State(
        max_bytes=1 << 20,
        strict_provenance=False,
        manifest={"gate_complete": True},
    )


def test_model_lifecycle_manifest_tracks_block_hook_identity():
    class Block(torch.nn.Module):
        def forward(self, value):
            return value

    class Diffusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

        def forward(self, value):
            return value

    diffusion = Diffusion()

    def post_hook(_module, _args, output):
        return output

    handle = diffusion.blocks[1].register_forward_hook(post_hook)
    try:
        patcher = SimpleNamespace(
            model=SimpleNamespace(diffusion_model=diffusion),
            is_injected=True,
            load_device="cuda:0",
            offload_device="cpu",
            injections={},
            object_patches={},
            model_options={},
        )
        manifest = observation.model_lifecycle_manifest(
            patcher,
            model_options={"transformer_options": {"wrappers": {}, "callbacks": {}}},
        )
    finally:
        handle.remove()

    assert manifest["patcher_is_injected"] is True
    assert manifest["block_count"] == 2
    assert manifest["block_pre_hook_count"] == 0
    assert manifest["block_post_hook_count"] == 1
    assert manifest["blocks"][1]["post_hooks"][0]["qualname"].endswith("post_hook")
    assert len(manifest["block_runtime_digest"]) == 64
    assert manifest["modified_modules"]["count"] == 1
    assert manifest["modified_modules"]["truncated"] is False
    assert manifest["runtime_lifecycle_digest"]


def test_model_lifecycle_manifest_tracks_nested_forward_replacement_owner():
    class Leaf(torch.nn.Module):
        def forward(self, value):
            return value

    class Diffusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Leaf()])
            self.proj = Leaf()

        def forward(self, value):
            return self.proj(value)

    class FakeAdapter:
        pass

    class ForwardHook:
        def __init__(self, module):
            self.module = module
            self.adapter = FakeAdapter()
            self.multiplier = 0.75
            self.original_forward = module.forward

        def bypass(self, value):
            return self.original_forward(value)

    diffusion = Diffusion()
    hook = ForwardHook(diffusion.proj)
    diffusion.proj.forward = hook.bypass
    patcher = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=diffusion),
        is_injected=True,
        load_device="cuda:0",
        offload_device="cpu",
        injections={},
        object_patches={},
        model_options={},
    )

    manifest = observation.model_lifecycle_manifest(patcher)
    modified = manifest["modified_modules"]
    assert modified["count"] == 1
    assert modified["truncated"] is False
    entry = modified["modules"][0]
    assert entry["path"] == "proj"
    assert entry["instance_forward_override"] is True
    owner = entry["forward"]["bound_owner"]
    assert owner["type"].endswith("ForwardHook")
    assert owner["adapter_type"].endswith("FakeAdapter")
    assert owner["module_type"].endswith("Leaf")
    assert owner["multiplier"] == 0.75


def test_prepare_sampling_observer_records_post_load_state(monkeypatch):
    state = _state()
    record = diag._Record(state=state)
    active_token = diag._ACTIVE.set(record)
    stage_token = diag._STAGE.set("high")
    model = SimpleNamespace(is_injected=False)
    calls = 0

    def lifecycle(value, *, model_options=None):
        return {
            "patcher_is_injected": value.is_injected,
            "active_injections": {"adapter": [{}]} if value.is_injected else {},
            "block_runtime_digest": "digest",
            "runtime_lifecycle_digest": "runtime-digest",
            "modified_modules": {"count": 1, "truncated": False},
            "marker": (model_options or {}).get("marker"),
        }

    monkeypatch.setattr(observation, "model_lifecycle_manifest", lifecycle)

    class Executor:
        def __call__(self, value, noise_shape, conds, **_kwargs):
            nonlocal calls
            calls += 1
            assert noise_shape == (1, 2, 3)
            value.is_injected = True
            return "real", conds, [SimpleNamespace(is_injected=True)]

    try:
        result = observation._prepare_sampling_wrapper(
            Executor(),
            model,
            (1, 2, 3),
            {"positive": []},
            model_options={"marker": "effective"},
        )
    finally:
        diag._STAGE.reset(stage_token)
        diag._ACTIVE.reset(active_token)

    assert calls == 1
    assert result[0] == "real"
    extension = state.manifest[observation._EXTENSION_KEY]
    assert extension["stage_lifecycles"] == [
        {
            "stage": "high",
            "capture_phase": "post_prepare_sampling_load",
            "model": {
                "patcher_is_injected": True,
                "active_injections": {"adapter": [{}]},
                "block_runtime_digest": "digest",
                "runtime_lifecycle_digest": "runtime-digest",
                "modified_modules": {"count": 1, "truncated": False},
                "marker": "effective",
            },
            "additional_models": [{"type": "types.SimpleNamespace", "patcher_is_injected": True}],
        }
    ]


def test_prepare_sampling_observation_failure_does_not_mask_loaded_result(monkeypatch):
    state = _state()
    record = diag._Record(state=state)
    active_token = diag._ACTIVE.set(record)
    stage_token = diag._STAGE.set("probe")
    model = SimpleNamespace(is_injected=False)

    def fail_lifecycle(*_args, **_kwargs):
        raise RuntimeError("observer failed")

    monkeypatch.setattr(observation, "model_lifecycle_manifest", fail_lifecycle)

    class Executor:
        def __call__(self, value, _noise_shape, conds, **_kwargs):
            value.is_injected = True
            return "real", conds, []

    try:
        result = observation._prepare_sampling_wrapper(
            Executor(),
            model,
            (1, 2, 3),
            {"positive": []},
        )
    finally:
        diag._STAGE.reset(stage_token)
        diag._ACTIVE.reset(active_token)

    assert result[0] == "real"
    extension = state.manifest[observation._EXTENSION_KEY]
    assert extension["observation_errors"] == [
        "post_prepare_sampling_load: RuntimeError: observer failed"
    ]
    lifecycle = extension["stage_lifecycles"][0]
    assert lifecycle["stage"] == "probe"
    assert lifecycle["model"] == {}
    assert lifecycle["observation_error"] == extension["observation_errors"][0]


def test_diffusion_observer_records_companion_state_before_and_after(monkeypatch):
    state = _state()
    record = diag._Record(state=state)
    active_token = diag._ACTIVE.set(record)
    stage_token = diag._STAGE.set("high")
    calls = 0

    class Executor:
        wrappers = ()

        def __call__(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return "result"

    def companion(_executor, _transformer, _root_model_options):
        return {"executor_calls": calls}

    monkeypatch.setattr(observation, "_active_companion_snapshot", companion)
    wrapper = observation._make_diffusion_wrapper({})
    try:
        result = wrapper(
            Executor(),
            object(),
            torch.tensor([0.4]),
            None,
            {diag._runtime.FLOW_STAGE_KEY: "high"},
        )
    finally:
        diag._STAGE.reset(stage_token)
        diag._ACTIVE.reset(active_token)

    assert result == "result"
    assert calls == 1
    extension = state.manifest[observation._EXTENSION_KEY]
    first = extension["companion_calls"]["high_first"]
    assert first["actual_index"] == 1
    assert first["before"] == {"executor_calls": 0}
    assert first["after"] == {"executor_calls": 1}
    assert first["observation_errors"] == []
    assert extension["companion_calls"]["high_last"] == first


def test_diffusion_observation_failure_does_not_add_or_replace_model_call(monkeypatch):
    state = _state()
    record = diag._Record(state=state)
    active_token = diag._ACTIVE.set(record)
    stage_token = diag._STAGE.set("high")
    calls = 0

    class Executor:
        wrappers = ()

        def __call__(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return "model-result"

    def fail_snapshot(*_args, **_kwargs):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(observation, "_active_companion_snapshot", fail_snapshot)
    wrapper = observation._make_diffusion_wrapper({})
    try:
        result = wrapper(
            Executor(),
            object(),
            torch.tensor([0.4]),
            None,
            {diag._runtime.FLOW_STAGE_KEY: "high"},
        )
    finally:
        diag._STAGE.reset(stage_token)
        diag._ACTIVE.reset(active_token)

    assert result == "model-result"
    assert calls == 1
    extension = state.manifest[observation._EXTENSION_KEY]
    assert len(extension["observation_errors"]) == 2
    assert all("RuntimeError: snapshot failed" in item for item in extension["observation_errors"])
    first = extension["companion_calls"]["high_first"]
    assert first["before"] == {}
    assert first["after"] == {}
    assert len(first["observation_errors"]) == 2


def test_vdn_runtime_snapshot_reads_closure_owned_layout_and_pool():
    VDNState = type("VDNState", (), {"__module__": "vdn_h3.hybrid"})
    state = VDNState()
    state.layout = SimpleNamespace(
        video_start=10,
        video_end=42,
        audio_start=4,
        audio_end=10,
        num_frames=2,
        tokens_per_frame=16,
        frame_size=(4, 4),
        text_start=0,
        text_len=4,
        bounds=((0, 2),),
        full_cover=True,
        seq_len=42,
        anchor_frames=1,
    )
    resources = SimpleNamespace(retain=True, retained_counts=lambda: {"kv": 1})
    state.runtime = SimpleNamespace(current=lambda: resources)
    state.retain_buffers = True

    def wrapper(*_args, **_kwargs):
        return state

    snapshot = observation._vdn_runtime_snapshot(SimpleNamespace(wrappers=[wrapper]))
    assert len(snapshot) == 1
    assert snapshot[0]["layout_active"] is True
    assert snapshot[0]["layout"]["tokens_per_frame"] == 16
    assert snapshot[0]["runtime_pool_active"] is True
    assert snapshot[0]["retained_counts"] == {"kv": 1}


def test_sol_runtime_snapshot_reads_active_request_without_import(monkeypatch):
    request = SimpleNamespace(
        evaluations=2,
        eligible_calls=7,
        sparse_calls=3,
        dense_calls=1,
        backend_transitions=0,
        last_routes=((0, "sol"),),
        gates=[{"block": 0}],
        fallbacks={},
        dense_provider_failures={},
        dense_attention_backends={"sage"},
        kernel=SimpleNamespace(backend_name="sol-attn", source_tree_verified=True),
    )
    request_var = contextvars.ContextVar("request", default=request)
    forward_var = contextvars.ContextVar(
        "forward",
        default=(object(), request, 1, {0, 1}, [(0, "sol")]),
    )
    module = SimpleNamespace(_REQUEST=request_var, _FORWARD=forward_var)
    monkeypatch.setitem(sys.modules, "sol_h3.runtime", module)

    snapshot = observation._sol_runtime_snapshot()
    assert snapshot["request_active"] is True
    assert snapshot["forward_active"] is True
    assert snapshot["evaluations"] == 2
    assert snapshot["dense_attention_backends"] == ["sage"]
    assert snapshot["forward"]["seen_blocks"] == [0, 1]
    assert snapshot["kernel"] == {
        "backend_name": "sol-attn",
        "source_tree_verified": True,
    }


def _complete_runtime_observation():
    lifecycle = {
        "capture_phase": "post_prepare_sampling_load",
        "model": {
            "patcher_is_injected": True,
            "active_injections": {"adapter": [{}]},
            "runtime_lifecycle_digest": "same",
            "modified_modules": {"count": 2, "truncated": False},
        },
        "additional_models": [],
    }
    companion = {
        "before": {
            "sol": {"request_active": True, "forward_active": True},
            "vdn": [{"layout_active": True, "runtime_pool_active": True}],
            "spectrum": {"active_run_id": 3, "active_step_id": 0},
        },
        "after": {
            "sol": {"request_active": True, "forward_active": True},
            "vdn": [{"layout_active": True, "runtime_pool_active": True}],
            "spectrum": {"active_run_id": 3, "active_step_id": 0},
        },
        "observation_errors": [],
    }
    return {
        "schema_version": 2,
        "stage_lifecycles": [
            {"stage": "low", **lifecycle},
            {"stage": "probe", **lifecycle},
            {"stage": "high", **lifecycle},
        ],
        "companion_calls": {
            "low_last": {"stage": "low", "actual_index": 4, **companion},
            "probe": {"stage": "probe", "actual_index": 1, **companion},
            "high_first": {"stage": "high", "actual_index": 1, **companion},
            "high_last": {"stage": "high", "actual_index": 2, **companion},
        },
        "observation_errors": [],
    }


def test_report_downgrades_structural_candidate_when_runtime_observation_is_incomplete():
    state = _state()
    state.complete = deque(
        [
            {
                "provenance": {},
                "observation_gate": {"structural_candidate": True},
                "promotion": {"production_fix_authorized": False},
            }
        ],
        maxlen=4,
    )

    report_text = observation.H3ExecutionContractReport().extract(state, None)[0]
    report = json.loads(report_text)
    gate = report["observation_gate"]
    assert gate["base_structural_candidate"] is True
    assert gate["runtime_contract"]["runtime_contract_complete"] is False
    assert gate["structural_candidate"] is False


def test_report_keeps_structural_candidate_only_with_complete_runtime_contract():
    state = _state()
    state.complete = deque(
        [
            {
                "provenance": {observation._EXTENSION_KEY: _complete_runtime_observation()},
                "observation_gate": {"structural_candidate": True},
                "promotion": {"production_fix_authorized": False},
            }
        ],
        maxlen=4,
    )

    report_text = observation.H3ExecutionContractReport().extract(state, None)[0]
    report = json.loads(report_text)
    gate = report["observation_gate"]
    runtime = gate["runtime_contract"]
    assert runtime["stage_lifecycle_complete"] is True
    assert runtime["injection_state_consistent"] is True
    assert runtime["hook_runtime_stable"] is True
    assert runtime["modified_runtime_visible"] is True
    assert runtime["observation_error_free"] is True
    assert runtime["runtime_contract_complete"] is True
    assert gate["structural_candidate"] is True


def test_report_rejects_observation_errors_even_when_other_runtime_evidence_is_complete():
    runtime_observation = _complete_runtime_observation()
    runtime_observation["observation_errors"] = ["high:before_diffusion_model: RuntimeError: failed"]
    state = _state()
    state.complete = deque(
        [
            {
                "provenance": {observation._EXTENSION_KEY: runtime_observation},
                "observation_gate": {"structural_candidate": True},
                "promotion": {"production_fix_authorized": False},
            }
        ],
        maxlen=4,
    )

    report_text = observation.H3ExecutionContractReport().extract(state, None)[0]
    report = json.loads(report_text)
    runtime = report["observation_gate"]["runtime_contract"]
    assert runtime["observation_error_free"] is False
    assert runtime["runtime_contract_complete"] is False
    assert report["observation_gate"]["structural_candidate"] is False


def test_report_rejects_truncated_modified_module_observation():
    runtime_observation = _complete_runtime_observation()
    runtime_observation["stage_lifecycles"][1]["model"]["modified_modules"] = {
        "count": observation._MAX_MODIFIED_MODULES + 1,
        "truncated": True,
    }
    state = _state()
    state.complete = deque(
        [
            {
                "provenance": {observation._EXTENSION_KEY: runtime_observation},
                "observation_gate": {"structural_candidate": True},
                "promotion": {"production_fix_authorized": False},
            }
        ],
        maxlen=4,
    )

    report_text = observation.H3ExecutionContractReport().extract(state, None)[0]
    report = json.loads(report_text)
    runtime = report["observation_gate"]["runtime_contract"]
    assert runtime["modified_runtime_visible"] is False
    assert runtime["runtime_contract_complete"] is False
    assert report["observation_gate"]["structural_candidate"] is False
