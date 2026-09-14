from __future__ import annotations

from types import SimpleNamespace

import torch

from h3_flow_regenerate import execution_contract_diagnostics as diag
from h3_flow_regenerate import execution_contract_provenance as provenance
from h3_flow_regenerate.geometry import pack_streams


def _state(max_bytes: int = 1 << 20):
    return diag._State(max_bytes=max_bytes, strict_provenance=False, manifest={"gate_complete": True})


def test_snapshot_budget_is_bounded_and_replace_does_not_accumulate_bytes():
    record = diag._Record(state=_state(256))
    first = torch.zeros(16, dtype=torch.float32)
    second = torch.ones(16, dtype=torch.float32)

    record.capture("last", first)
    assert record.byte_count == 64
    record.capture("last", second, replace=True)
    assert record.byte_count == 64
    assert torch.equal(record.snapshots["last"].tensor, second)

    record.capture("too_large", torch.zeros(128, dtype=torch.float32))
    assert "too_large" not in record.snapshots
    assert record.byte_count == 64
    assert any("capture budget exceeded" in item for item in record.incomplete)


def test_relative_wrapper_insertion_preserves_existing_order_and_anchor():
    flow_key = diag._runtime.OUTER_WRAPPER_KEY

    def before(*_args, **_kwargs):
        return None

    def flow(*_args, **_kwargs):
        return None

    def after(*_args, **_kwargs):
        return None

    class FakeModel:
        def __init__(self):
            self.wrappers = {"outer": {"before": [before], flow_key: [flow], "after": [after]}}

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.wrappers.setdefault(wrapper_type, {}).pop(key, None)

    model = FakeModel()
    diag._insert_relative(model, "outer", flow_key, "observer_before", before, before=True)
    diag._insert_relative(model, "outer", flow_key, "observer_after", after, before=False)

    assert list(model.wrappers["outer"]) == [
        "before",
        "observer_before",
        flow_key,
        "observer_after",
        "after",
    ]
    assert model.wrappers["outer"][flow_key] == [flow]


def test_sampler_condition_compare_observes_processed_conditions_without_replaying_preprocess():
    shared = torch.arange(4.0)
    pristine = {"positive": [{"model_conds": {"shared": shared}, "cross_attn": shared}]}
    digest, summary = provenance.fingerprint(pristine)
    record = diag._Record(
        state=_state(),
        pristine_conds=pristine,
        pristine_cond_digest=digest,
        pristine_cond_summary=summary,
        pristine_cond_alias_graph=provenance.alias_graph(pristine),
        high_pre_core_digest=digest,
        high_pre_core_summary=summary,
        high_pre_core_alias_graph=provenance.alias_graph(pristine),
    )
    token = diag._ACTIVE.set(record)
    calls = []

    class Executor:
        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "sentinel"

    model_wrap = SimpleNamespace(conds=pristine)
    extra_args = {"model_options": {"transformer_options": {diag._runtime.FLOW_STAGE_KEY: "high"}}}
    try:
        result = diag._sampler_wrapper(
            Executor(),
            model_wrap,
            torch.tensor([0.8, 0.0]),
            extra_args,
            None,
            torch.zeros(1, 2),
        )
    finally:
        diag._ACTIVE.reset(token)

    assert result == "sentinel"
    assert len(calls) == 1
    assert record.condition_compare["pre_core_equal"] is True
    assert record.condition_compare["no_extra_condition_preprocess"] is True
    assert record.condition_compare["no_extra_h3_evaluation"] is True
    assert record.condition_compare["no_extra_upscaler_call"] is True
    assert record.condition_compare["pristine_target_alias_graph"]
    assert record.condition_compare["actual_processed_alias_graph"]


def test_sampler_wrapper_captures_raw_high_state_before_inpaint_rewrite():
    video = torch.randn(1, 24, 5, 4, 4)
    audio = torch.randn(1, 32, 2, 9)
    noise_video = torch.randn_like(video)
    noise_audio = torch.randn_like(audio)
    latent_video = torch.randn_like(video)
    latent_audio = torch.randn_like(audio)
    noise, shapes = pack_streams((noise_video, noise_audio))
    latent, _ = pack_streams((latent_video, latent_audio))
    sigmas = torch.tensor([0.4, 0.0])
    noise_scale = 1.25
    record = diag._Record(state=_state(8 << 20))
    token = diag._ACTIVE.set(record)

    class Executor:
        def __call__(self, *_args, **_kwargs):
            return "sentinel"

    model_wrap = SimpleNamespace(
        conds={},
        inner_model=SimpleNamespace(
            latent_shapes=shapes,
            model_sampling=SimpleNamespace(noise_scale=noise_scale),
        ),
    )
    extra_args = {"model_options": {"transformer_options": {diag._runtime.FLOW_STAGE_KEY: "high"}}}
    try:
        result = diag._sampler_wrapper(Executor(), model_wrap, sigmas, extra_args, None, noise, latent)
    finally:
        diag._ACTIVE.reset(token)

    expected = sigmas[0] * (noise_scale * noise) + (1.0 - sigmas[0]) * latent
    expected_video, expected_audio = diag.unpack_streams(expected, shapes)
    assert result == "sentinel"
    assert torch.equal(record.snapshots["first_high_sampler_input_video"].tensor, expected_video)
    assert torch.equal(record.snapshots["first_high_sampler_input_audio"].tensor, expected_audio)


def _first_high_fixture(*, exact_bridge: bool = False):
    video = torch.randn(1, 24, 5, 4, 4)
    audio = torch.randn(1, 32, 2, 9)
    _packed, shapes = pack_streams((video, audio))
    sigma = 0.4
    seed = 17
    seed_offset = 1000
    noise = diag.deterministic_video_noise(
        tuple(video.shape),
        seed=seed + seed_offset,
        device=video.device,
        dtype=video.dtype,
    )
    clean = torch.randn_like(video)
    state_video = (1.0 - sigma) * clean + sigma * noise
    _state_packed, _ = pack_streams((state_video, audio))
    model_input_video = torch.randn_like(video)
    model_input_audio = torch.randn_like(audio)
    model_input_packed, _ = pack_streams((model_input_video, model_input_audio))
    raw_video = torch.randn_like(video)
    raw_packed, _ = pack_streams((raw_video, audio))

    record = diag._Record(
        state=_state(8 << 20),
        seed=seed,
        config=SimpleNamespace(seed_offset=seed_offset),
    )
    record.capture("first_high_sampler_input_video", state_video)
    record.capture("first_high_sampler_input_audio", audio)
    token = diag._ACTIVE.set(record)

    class Executor:
        class_obj = SimpleNamespace(inner_model=SimpleNamespace(latent_shapes=shapes))

        def __call__(self, *_args, **_kwargs):
            return raw_packed

    transformer = {
        diag._runtime.FLOW_STAGE_KEY: "high",
        "h3_refinement": {
            "api": 1,
            "active": True,
            "min_actual_prefix_steps": 1,
            "sigma_reference": 1.0,
            "source": "h3_flow_progressive_handoff",
        },
    }
    if exact_bridge:
        transformer[diag._runtime.EXACT_PREFIX_BRIDGE_KEY] = {
            "exact_prefix": torch.zeros_like(video[:, :, :1]),
            "source": "test",
            "applied": False,
        }
    model_options = {"transformer_options": transformer}
    try:
        result = diag._predict_raw_wrapper(Executor(), model_input_packed, torch.tensor([sigma]), model_options, seed)
    finally:
        diag._ACTIVE.reset(token)
    return record, result, raw_packed, raw_video, model_input_video, clean


def test_first_high_raw_capture_recovers_existing_learned_clean_without_upscaler_call():
    record, result, raw_packed, raw_video, model_input_video, clean = _first_high_fixture()

    assert torch.equal(result, raw_packed)
    assert torch.equal(record.snapshots["first_high_model_input_video"].tensor, model_input_video)
    assert torch.equal(record.snapshots["first_high_model_raw_video"].tensor, raw_video)
    assert torch.equal(record.snapshots["first_high_pre_guidance_video"].tensor, raw_video)
    assert torch.equal(record.snapshots["last_high_pre_guidance_video"].tensor, raw_video)
    assert torch.allclose(record.snapshots["learned_transfer_clean_video"].tensor, clean, atol=1e-5, rtol=1e-5)
    assert record.first_high_contract["refinement"]["min_actual_prefix_steps"] == 1


def test_exact_prefix_bridge_does_not_mislabel_raw_output_as_pre_guidance():
    record, _result, _raw_packed, raw_video, _model_input_video, _clean = _first_high_fixture(exact_bridge=True)

    assert torch.equal(record.snapshots["first_high_model_raw_video"].tensor, raw_video)
    assert torch.equal(record.snapshots["last_high_model_raw_video"].tensor, raw_video)
    assert "first_high_pre_guidance_video" not in record.snapshots
    assert "last_high_pre_guidance_video" not in record.snapshots
    assert any("raw equivalence is invalid" in item for item in record.incomplete)


def test_diffusion_observer_captures_native_av_streams_once_without_changing_result():
    video = torch.randn(1, 24, 5, 4, 4)
    audio = torch.randn(1, 32, 2, 9)
    out_video = torch.randn_like(video)
    out_audio = torch.randn_like(audio)
    record = diag._Record(state=_state(8 << 20))
    active_token = diag._ACTIVE.set(record)
    stage_token = diag._STAGE.set("high")
    calls = 0

    class Executor:
        def __call__(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return [out_video, out_audio]

    try:
        result = diag._diffusion_wrapper(
            Executor(),
            [video, audio],
            torch.tensor([0.4]),
            None,
            {diag._runtime.FLOW_STAGE_KEY: "high"},
        )
    finally:
        diag._STAGE.reset(stage_token)
        diag._ACTIVE.reset(active_token)

    assert calls == 1
    assert result[0] is out_video
    assert result[1] is out_audio
    assert torch.equal(record.snapshots["first_high_h3_input_video"].tensor, video)
    assert torch.equal(record.snapshots["first_high_h3_input_audio"].tensor, audio)
    assert torch.equal(record.snapshots["first_high_h3_velocity_video"].tensor, out_video)
    assert torch.equal(record.snapshots["first_high_h3_velocity_audio"].tensor, out_audio)


def test_controlled_o_requires_exact_per_stage_topology_not_only_matching_totals():
    exact = {
        "logical": 9,
        "actual": 7,
        "forecast": 2,
        "learned_upscale_events": 1,
        "per_stage": {
            "low": {"logical": 5, "actual": 4, "forecast": 1},
            "probe": {"logical": 1, "actual": 1, "forecast": 0},
            "high": {"logical": 3, "actual": 2, "forecast": 1},
        },
    }
    shifted = {
        **exact,
        "per_stage": {
            "low": {"logical": 5, "actual": 3, "forecast": 2},
            "probe": {"logical": 1, "actual": 1, "forecast": 0},
            "high": {"logical": 3, "actual": 3, "forecast": 0},
        },
    }

    assert diag._matches_controlled_o(exact) is True
    assert diag._matches_controlled_o(shifted) is False


def test_research_receipt_absence_is_not_treated_as_zero_activity():
    assert diag._all_research_receipts_zero([]) is None
    receipts = diag._research_receipts({"sol": {"weighted_calls": 0, "mixed_grid_calls": 0}, "unrelated": 4})
    assert receipts
    assert diag._all_research_receipts_zero(receipts) is True
    nonzero = diag._research_receipts({"sol": {"weighted_calls": 1}})
    assert diag._all_research_receipts_zero(nonzero) is False


def test_alias_graph_uses_stable_labels_not_raw_object_ids():
    shared = torch.tensor([1.0])
    graph = provenance.alias_graph({"left": shared, "right": [shared]})

    assert graph == [
        {
            "alias": "alias_1",
            "type": "torch.Tensor",
            "paths": ["$.left", "$.right[0]"],
        }
    ]


def test_path_marker_matching_is_case_and_separator_insensitive():
    assert provenance._path_matches(
        r"C:\ComfyUI\custom_nodes\ComfyUI-Sol-H3\sol_h3\runtime.py",
        ("comfyui-sol-h3",),
    )
    assert not provenance._path_matches("/tmp/unrelated/runtime.py", ("comfyui-sol-h3",))


def test_fingerprint_ignores_execution_only_uuid():
    left = {"positive": [{"uuid": "a", "value": torch.tensor([1.0, 2.0])}]}
    right = {"positive": [{"uuid": "b", "value": torch.tensor([1.0, 2.0])}]}
    assert provenance.fingerprint(left)[0] == provenance.fingerprint(right)[0]


def test_callable_identity_resolves_callable_instance_source_file():
    class CallableObserver:
        def __call__(self, value):
            return value

    identity = provenance.callable_identity(CallableObserver())
    assert identity["module"] == __name__
    assert identity["file"] is not None
    assert identity["file"]["sha256"]


def test_callable_identity_records_nested_closure_choice():
    def make_outer(choice):
        def selected(value):
            return value + choice

        def outer(value):
            return selected(value)

        return outer

    left = provenance.callable_identity(make_outer(1))
    right = provenance.callable_identity(make_outer(2))

    assert left["closure"]["selected"]["callable"]["closure"]["choice"] == 1
    assert right["closure"]["selected"]["callable"]["closure"]["choice"] == 2


def test_finalize_guard_does_not_mask_sampling_error(monkeypatch):
    state = _state()
    record = diag._Record(state=state, error="RuntimeError: sampler failed")
    original = RuntimeError("sampler failed")

    def fail_finalize(_record, _guider):
        raise ValueError("diagnostic failed")

    monkeypatch.setattr(diag, "_finalize_record", fail_finalize)
    diag._finalize_record_guarded(record, object(), sampling_error=original)

    report = state.complete[-1]
    assert report["error"] == "RuntimeError: sampler failed"
    assert report["diagnostic_finalize_error"] == "ValueError: diagnostic failed"
    assert report["promotion"]["production_fix_authorized"] is False


def test_finalize_guard_raises_diagnostic_failure_when_sampling_succeeded(monkeypatch):
    state = _state()
    record = diag._Record(state=state)

    def fail_finalize(_record, _guider):
        raise ValueError("diagnostic failed")

    monkeypatch.setattr(diag, "_finalize_record", fail_finalize)
    try:
        diag._finalize_record_guarded(record, object(), sampling_error=None)
    except ValueError as exc:
        assert str(exc) == "diagnostic failed"
    else:
        raise AssertionError("diagnostic finalization failure was swallowed")


def test_audio_roundtrip_gate_accepts_float_rounding_but_rejects_material_change():
    low = torch.tensor([1.0, -2.0, 0.25], dtype=torch.float32)
    high = low.clone()
    high[0] = torch.nextafter(high[0], torch.tensor(float("inf"), dtype=high.dtype))
    rounded = diag._audio_roundtrip_check(low, high)
    assert rounded["carried_audio_exact"] is False
    assert rounded["carried_audio_bitwise_exact"] is False
    assert rounded["carried_audio_roundtrip_within_dtype_tolerance"] is True
    assert rounded["carried_audio_roundtrip_max_abs"] > 0.0
    changed = low.clone()
    changed[0] += 1e-3
    assert diag._audio_roundtrip_check(low, changed)["carried_audio_roundtrip_within_dtype_tolerance"] is False


def test_execution_contract_default_capture_budget_is_512_mib():
    required = diag.H3ExecutionContractDiagnostics.INPUT_TYPES()["required"]
    assert required["capture_mib"][1]["default"] == 512
