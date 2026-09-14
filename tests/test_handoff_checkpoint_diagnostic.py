from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import handoff_checkpoint_diagnostic as diag
from h3_flow_regenerate.geometry import pack_streams


def _record():
    record = diag._CaptureRecord()
    diag._state().tls.record = record
    return record


def _clear_state():
    state = diag._state()
    state.tls.record = None
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


def test_predict_wrapper_keeps_last_low_and_first_high(monkeypatch):
    _clear_state()
    record = _record()
    video = torch.zeros(1, 24, 5, 4, 4)
    audio = torch.zeros(1, 32, 2, 9)
    packed, shapes = pack_streams((video, audio))

    class_obj = SimpleNamespace(inner_model=SimpleNamespace(latent_shapes=shapes))
    executor = SimpleNamespace(class_obj=class_obj)

    def fake_predict(_executor, x, _timestep, _model_options=None, _seed=None):
        return x + 1.0

    monkeypatch.setattr(diag, "_ORIGINAL_FLOW_PREDICT", fake_predict)
    low_options = {"transformer_options": {diag._runtime.FLOW_STAGE_KEY: "low"}}
    high_options = {"transformer_options": {diag._runtime.FLOW_STAGE_KEY: "high"}}

    diag._flow_predict_capture_wrapper(executor, packed, torch.tensor([0.8]), low_options, None)
    second_low = diag._flow_predict_capture_wrapper(executor, packed + 3.0, torch.tensor([0.6]), low_options, None)
    first_high = diag._flow_predict_capture_wrapper(executor, packed + 6.0, torch.tensor([0.4]), high_options, None)
    diag._flow_predict_capture_wrapper(executor, packed + 9.0, torch.tensor([0.2]), high_options, None)

    expected_low, _ = diag.unpack_streams(second_low, shapes)
    expected_high, _ = diag.unpack_streams(first_high, shapes)
    assert record.low_model_calls == 2
    assert torch.equal(record.low_last_model_clean_video, expected_low)
    assert torch.equal(record.first_high_clean_video, expected_high)
    assert record.low_last_sigma == pytest.approx(0.6)
    assert record.first_high_sigma == pytest.approx(0.4)
    diag._state().tls.record = None


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

    assert len(outputs) == 6
    assert outputs[0]["samples"].shape == (1, 24, 5, 4, 4)
    assert outputs[4]["samples"].shape == (1, 24, 5, 6, 6)
    assert '"capture_completed_ns": 12345' in outputs[5]
    with pytest.raises(RuntimeError, match="no complete learned_3d"):
        diag.H3HandoffCheckpointDiagnostic().extract(model, trigger)


def test_model_contract_fails_closed_if_video_latent_scaling_changes():
    class MiniMaxH3AV:
        scale_factor = 0.5

    class MiniMaxH3:
        latent_format = MiniMaxH3AV()

    with pytest.raises(RuntimeError, match="scaling changed"):
        diag._validate_model_video_latent_contract(SimpleNamespace(model=MiniMaxH3()))
