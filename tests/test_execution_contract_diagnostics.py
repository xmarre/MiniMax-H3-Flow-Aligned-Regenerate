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
    pristine = {"positive": [{"model_conds": {}, "cross_attn": torch.arange(4.0)}]}
    digest, summary = provenance.fingerprint(pristine)
    record = diag._Record(
        state=_state(),
        pristine_conds=pristine,
        pristine_cond_digest=digest,
        pristine_cond_summary=summary,
        high_pre_core_digest=digest,
        high_pre_core_summary=summary,
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


def test_first_high_raw_capture_recovers_existing_learned_clean_without_upscaler_call():
    video = torch.randn(1, 24, 5, 4, 4)
    audio = torch.randn(1, 32, 2, 9)
    packed, shapes = pack_streams((video, audio))
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
    state_packed, _ = pack_streams((state_video, audio))
    raw_video = torch.randn_like(video)
    raw_packed, _ = pack_streams((raw_video, audio))

    record = diag._Record(
        state=_state(8 << 20),
        seed=seed,
        config=SimpleNamespace(seed_offset=seed_offset),
    )
    token = diag._ACTIVE.set(record)

    class Executor:
        class_obj = SimpleNamespace(inner_model=SimpleNamespace(latent_shapes=shapes))

        def __call__(self, *_args, **_kwargs):
            return raw_packed

    model_options = {
        "transformer_options": {
            diag._runtime.FLOW_STAGE_KEY: "high",
            "h3_refinement": {
                "api": 1,
                "active": True,
                "min_actual_prefix_steps": 1,
                "sigma_reference": 1.0,
                "source": "h3_flow_progressive_handoff",
            },
        }
    }
    try:
        result = diag._predict_raw_wrapper(
            Executor(), state_packed, torch.tensor([sigma]), model_options, seed
        )
    finally:
        diag._ACTIVE.reset(token)

    assert torch.equal(result, raw_packed)
    assert torch.equal(record.snapshots["first_high_model_raw_video"].tensor, raw_video)
    assert torch.equal(record.snapshots["first_high_pre_guidance_video"].tensor, raw_video)
    assert torch.allclose(record.snapshots["learned_transfer_clean_video"].tensor, clean, atol=1e-5, rtol=1e-5)
    assert record.first_high_contract["refinement"]["min_actual_prefix_steps"] == 1


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
