from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import handoff_checkpoint_diagnostic as diag
from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.guidance import GuidanceConfig, GuidanceState


def _record():
    record = diag._CaptureRecord()
    diag._state().tls.record = record
    return record


def _clear_state():
    state = diag._state()
    state.tls.record = None
    state.tls.flow_stage = None
    with state.lock:
        state.complete.clear()


def test_build_handoff_wrapper_captures_probe_and_recovers_learned_clean(monkeypatch):
    _clear_state()
    record = _record()
    source_video = torch.randn(1, 24, 5, 4, 4)
    source_audio = torch.randn(1, 32, 2, 9)
    source_x0, source_shapes = pack_streams((source_video, source_audio))
    source_state = torch.randn_like(source_x0)
    sigma = 0.4
    seed = 123
    learned = torch.randn(1, 24, 5, 6, 6)

    def fake_build(**kwargs):
        noise = diag.deterministic_video_noise(
            tuple(learned.shape),
            seed=kwargs["seed"],
            device=learned.device,
            dtype=learned.dtype,
        )
        target_video = (1.0 - kwargs["sigma"]) * learned + kwargs["sigma"] * noise
        return pack_streams((target_video, source_audio.clone()))

    monkeypatch.setattr(diag, "_ORIGINAL_BUILD_HANDOFF_STATE", fake_build)
    target_packed, target_shapes = diag._build_handoff_state_capture_wrapper(
        source_packed_state=source_state,
        source_x0_packed=source_x0,
        source_shapes=source_shapes,
        sigma=sigma,
        target_h=6,
        target_w=6,
        seed=seed,
        transfer_mode="learned_3d",
    )

    assert tuple(target_shapes[0]) == tuple(learned.shape)
    assert target_packed.ndim == 3
    assert torch.equal(record.exact_probe_clean_video, source_video)
    assert torch.allclose(record.learned_transfer_clean_video, learned, atol=1e-6, rtol=1e-6)
    assert record.handoff_sigma == sigma
    diag._state().tls.record = None


def test_predict_wrapper_splits_raw_model_from_flow_postprocessing(monkeypatch):
    _clear_state()
    record = _record()
    video = torch.zeros(1, 24, 5, 4, 4)
    audio = torch.zeros(1, 32, 2, 9)
    packed, shapes = pack_streams((video, audio))

    class Executor:
        class_obj = SimpleNamespace(inner_model=SimpleNamespace(latent_shapes=shapes))

        def __call__(self, x, _timestep, _model_options=None, _seed=None):
            return x + 10.0

    executor = Executor()

    def fake_predict(observed_executor, x, timestep, model_options=None, seed=None):
        return observed_executor(x, timestep, model_options, seed) + 1.0

    monkeypatch.setattr(diag, "_ORIGINAL_FLOW_PREDICT", fake_predict)
    low_options = {"transformer_options": {diag._runtime.FLOW_STAGE_KEY: "low"}}
    high_options = {
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

    diag._flow_predict_capture_wrapper(executor, packed, torch.tensor([0.8]), low_options, None)
    second_low = diag._flow_predict_capture_wrapper(executor, packed + 3.0, torch.tensor([0.6]), low_options, None)
    first_high = diag._flow_predict_capture_wrapper(executor, packed + 6.0, torch.tensor([0.4]), high_options, None)
    diag._flow_predict_capture_wrapper(executor, packed + 9.0, torch.tensor([0.2]), high_options, None)

    expected_low, _ = diag.unpack_streams(second_low, shapes)
    expected_high, _ = diag.unpack_streams(first_high, shapes)
    expected_raw, _ = diag.unpack_streams(packed + 16.0, shapes)
    assert record.low_model_calls == 2
    assert torch.equal(record.low_last_model_clean_video, expected_low)
    assert torch.equal(record.first_high_clean_video, expected_high)
    assert torch.equal(record.first_high_model_raw_video, expected_raw)
    assert record.low_last_sigma == pytest.approx(0.6)
    assert record.first_high_sigma == pytest.approx(0.4)
    assert record.first_high_contract["flow_stage"] == "high"
    assert record.first_high_contract["h3_refinement"]["min_actual_prefix_steps"] == 1
    diag._state().tls.record = None


def test_guidance_wrapper_captures_pre_direction_full_split_and_reference(monkeypatch):
    _clear_state()
    record = _record()
    diag._state().tls.flow_stage = "high"
    high_x0 = torch.randn(1, 24, 5, 6, 6)
    source_ref = torch.randn(1, 24, 5, 4, 4)
    config = GuidanceConfig(mode="direction+temporal", direction_weight=0.25, temporal_weight=0.20)
    actual_state = GuidanceState()

    def fake_apply(value, *, config, state, **_kwargs):
        if config.mode == "direction":
            return value + 2.0
        state.last_schedule = 0.75
        state.last_correction_rms = 3.0
        state.last_baseline_rms = 6.0
        state.last_correction_rms_ratio = 0.5
        state.last_clamp_scale = 1.0
        state.last_direction_rms_ratio = 0.25
        state.last_temporal_rms_ratio = 0.1
        state.last_temporal_confidence_mean = 0.02
        state.last_temporal_valid_fraction = 0.01
        state.last_temporal_disocclusion_fraction = 0.99
        state.last_temporal_similarity_mean = 0.4
        state.last_temporal_margin_mean = 0.03
        state.last_temporal_cache_hit = False
        return value + 3.0

    monkeypatch.setattr(diag, "_ORIGINAL_APPLY_GUIDANCE", fake_apply)
    monkeypatch.setattr(diag, "time_matched_reference_info", lambda _run, _coordinate: (source_ref, 0.3, False))
    monkeypatch.setattr(
        diag,
        "resize_video",
        lambda value, h, w, mode: (
            torch.nn.functional.interpolate(
                value.permute(0, 2, 1, 3, 4).reshape(-1, value.shape[1], value.shape[3], value.shape[4]),
                size=(h, w),
                mode="nearest",
            )
            .reshape(value.shape[0], value.shape[2], value.shape[1], h, w)
            .permute(0, 2, 1, 3, 4)
        ),
    )

    result = diag._apply_guidance_capture_wrapper(
        high_x0,
        run=object(),
        coordinate=0.4,
        config=config,
        state=actual_state,
        high_state=torch.zeros_like(high_x0),
        sigma=0.4,
    )

    assert torch.equal(result, high_x0 + 3.0)
    assert torch.equal(record.first_high_pre_guidance_video, high_x0)
    assert torch.equal(record.first_high_direction_only_video, high_x0 + 2.0)
    assert record.first_high_guidance_reference_video.shape == high_x0.shape
    assert record.first_high_guidance["mode"] == "direction+temporal"
    assert record.first_high_guidance["temporal_valid_fraction"] == pytest.approx(0.01)
    diag._state().tls.record = None
    diag._state().tls.flow_stage = None


def test_extract_node_returns_and_consumes_latest_complete_capture():
    _clear_state()
    record = diag._CaptureRecord()
    record.source_shapes = [(1, 24, 5, 4, 4), (1, 32, 2, 9)]
    record.target_shapes = [(1, 24, 5, 6, 6), (1, 32, 2, 9)]
    record.handoff_sigma = 0.4
    record.low_last_model_clean_video = torch.zeros(1, 24, 5, 4, 4)
    record.exact_probe_clean_video = torch.ones(1, 24, 5, 4, 4)
    record.learned_transfer_clean_video = torch.full((1, 24, 5, 6, 6), 2.0)
    record.first_high_clean_video = torch.full((1, 24, 5, 6, 6), 3.0)
    record.final_high_video = torch.full((1, 24, 5, 6, 6), 4.0)
    record.first_high_model_raw_video = torch.full((1, 24, 5, 6, 6), 2.2)
    record.first_high_pre_guidance_video = torch.full((1, 24, 5, 6, 6), 2.3)
    record.first_high_direction_only_video = torch.full((1, 24, 5, 6, 6), 2.6)
    record.first_high_guidance_reference_video = torch.full((1, 24, 5, 6, 6), 1.5)
    record.first_high_input_state_rms = 0.7
    record.first_high_contract = {"flow_stage": "high"}
    record.first_high_guidance = {"mode": "direction+temporal"}
    record.completed_ns = 12345
    with diag._state().lock:
        diag._state().complete.append(record)

    class MiniMaxH3AV:
        scale_factor = 1.0

    class MiniMaxH3:
        latent_format = MiniMaxH3AV()

    model = SimpleNamespace(model=MiniMaxH3())
    trigger = {"samples": torch.zeros(1, 24, 5, 6, 6)}
    outputs = diag.H3HandoffCheckpointDiagnostic().extract(model, trigger)

    assert len(outputs) == 10
    assert outputs[0]["samples"].shape == (1, 24, 5, 4, 4)
    assert outputs[4]["samples"].shape == (1, 24, 5, 6, 6)
    assert '"capture_completed_ns": 12345' in outputs[5]
    assert outputs[6]["samples"].shape == (1, 24, 5, 6, 6)
    assert outputs[9]["samples"].shape == (1, 24, 5, 6, 6)
    assert '"model_raw_to_pre_guidance"' in outputs[5]
    with pytest.raises(RuntimeError, match="no complete learned_3d"):
        diag.H3HandoffCheckpointDiagnostic().extract(model, trigger)


def test_model_contract_fails_closed_if_video_latent_scaling_changes():
    class MiniMaxH3AV:
        scale_factor = 0.5

    class MiniMaxH3:
        latent_format = MiniMaxH3AV()

    with pytest.raises(RuntimeError, match="scaling changed"):
        diag._validate_model_video_latent_contract(SimpleNamespace(model=MiniMaxH3()))
