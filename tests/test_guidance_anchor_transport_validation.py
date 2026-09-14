from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import guidance_anchor_transport_validation as validation
from h3_flow_regenerate.contracts import TrajectoryRun, TrajectorySample
from h3_flow_regenerate.geometry import geometry_from_video, pack_streams
from h3_flow_regenerate.guidance import GuidanceConfig, GuidanceState, time_matched_reference_info


def _run(source_anchor: torch.Tensor) -> TrajectoryRun:
    delta = torch.full_like(source_anchor, 0.25)
    samples = (
        TrajectorySample(
            coordinate=0.6,
            video_sigma=0.95,
            audio_sigma=0.8,
            outer_step=3,
            call_index=3,
            phase="single",
            provenance="actual",
            video_x0=source_anchor - delta,
        ),
        TrajectorySample(
            coordinate=0.4,
            video_sigma=0.88,
            audio_sigma=0.64,
            outer_step=5,
            call_index=5,
            phase="handoff_probe",
            provenance="actual",
            video_x0=source_anchor,
        ),
        TrajectorySample(
            coordinate=0.2,
            video_sigma=0.7,
            audio_sigma=0.4,
            outer_step=6,
            call_index=6,
            phase="single",
            provenance="actual",
            video_x0=source_anchor + delta,
        ),
    )
    return TrajectoryRun(
        schema_version=1,
        run_id="run",
        session_id="session",
        chunk_id="chunk",
        sampler="sample_euler",
        scheduler="sched",
        geometry=geometry_from_video(source_anchor),
        audio_shape=(1, 32, 2, 9),
        layout_signature="source",
        conditioning_signature="cond",
        storage="vram",
        samples=samples,
        started_ns=1,
        completed_ns=2,
        complete=True,
    )


def _clear_state() -> None:
    state = validation._state()
    state.tls.record = None
    with state.lock:
        state.complete.clear()


def test_transport_run_is_exact_at_handoff_and_transports_deltas() -> None:
    source_anchor = torch.randn(1, 24, 5, 4, 4)
    target_anchor = torch.randn(1, 24, 5, 6, 6)
    run = _run(source_anchor)

    transformed, facts = validation._transport_run_to_target(
        run,
        captured_source_anchor=source_anchor.clone(),
        target_anchor=target_anchor,
    )

    at_handoff, coordinate, clamped = time_matched_reference_info(transformed, 0.4)
    assert coordinate == pytest.approx(0.4)
    assert clamped is False
    assert torch.equal(at_handoff, target_anchor)
    assert facts["anchor_source_match_max_abs"] == 0.0
    assert facts["source_sample_count"] == 3
    assert transformed.geometry.latent_h == 6
    assert transformed.geometry.latent_w == 6

    later, _, _ = time_matched_reference_info(transformed, 0.2)
    expected_delta = validation.resize_video(
        torch.full_like(source_anchor, 0.25),
        6,
        6,
        mode="bicubic",
    )
    assert torch.allclose(later, target_anchor + expected_delta)


def test_transport_fails_closed_when_committed_probe_differs_from_transfer_anchor() -> None:
    source_anchor = torch.randn(1, 24, 5, 4, 4)
    target_anchor = torch.randn(1, 24, 5, 6, 6)
    run = _run(source_anchor)

    with pytest.raises(RuntimeError, match="does not match the clean state"):
        validation._transport_run_to_target(
            run,
            captured_source_anchor=source_anchor + 0.01,
            target_anchor=target_anchor,
        )


def test_build_wrapper_recovers_existing_learned_target_without_second_upscaler(monkeypatch) -> None:
    _clear_state()
    source_video = torch.randn(1, 24, 5, 4, 4)
    source_audio = torch.randn(1, 32, 2, 9)
    source_packed, source_shapes = pack_streams((source_video, source_audio))
    learned = torch.randn(1, 24, 5, 6, 6)
    sigma = 0.4
    seed = 99
    calls = {"count": 0}

    def fake_build(**kwargs):
        calls["count"] += 1
        noise = validation.deterministic_video_noise(
            tuple(learned.shape),
            seed=kwargs["seed"],
            device=learned.device,
            dtype=learned.dtype,
        )
        target_video = (1.0 - kwargs["sigma"]) * learned + kwargs["sigma"] * noise
        return pack_streams((target_video, source_audio.clone()))

    monkeypatch.setattr(validation, "_ORIGINAL_BUILD_HANDOFF_STATE", fake_build)
    record = validation._ValidationRecord(enabled=True)
    validation._state().tls.record = record
    validation._build_handoff_state_anchor_wrapper(
        source_packed_state=source_packed,
        source_x0_packed=source_packed,
        source_shapes=source_shapes,
        sigma=sigma,
        target_h=6,
        target_w=6,
        seed=seed,
        transfer_mode="learned_3d",
    )

    assert calls["count"] == 1
    assert torch.equal(record.source_anchor, source_video)
    assert torch.allclose(record.target_anchor, learned, atol=1e-6, rtol=1e-6)
    validation._state().tls.record = None


def test_apply_wrapper_passes_target_representation_to_existing_guidance(monkeypatch) -> None:
    _clear_state()
    source_anchor = torch.randn(1, 24, 5, 4, 4)
    target_anchor = torch.randn(1, 24, 5, 6, 6)
    run = _run(source_anchor)
    record = validation._ValidationRecord(
        enabled=True,
        source_anchor=source_anchor.clone(),
        target_anchor=target_anchor.clone(),
    )
    validation._state().tls.record = record
    observed = {}

    def fake_apply(high_x0, *, run, coordinate, config, state, high_state=None, sigma=None):
        reference, _, _ = time_matched_reference_info(run, coordinate)
        observed["reference"] = reference.detach().clone()
        return high_x0 + reference.to(high_x0) * 0.1

    monkeypatch.setattr(validation, "_ORIGINAL_APPLY_GUIDANCE", fake_apply)
    high = torch.zeros_like(target_anchor)
    result = validation._apply_guidance_anchor_wrapper(
        high,
        run=run,
        coordinate=0.4,
        config=GuidanceConfig(mode="direction"),
        state=GuidanceState(),
        high_state=high,
        sigma=0.88,
    )

    assert torch.equal(observed["reference"], target_anchor)
    assert torch.equal(record.first_reference, target_anchor)
    assert record.first_reference_to_target_max_abs == 0.0
    assert record.guidance_calls == 1
    assert torch.equal(record.last_pre_guidance, high)
    assert torch.allclose(record.last_full_guidance, target_anchor * 0.1)
    assert torch.allclose(result, target_anchor * 0.1)
    validation._state().tls.record = None


def test_progressive_wrapper_queues_only_complete_validation_capture(monkeypatch) -> None:
    _clear_state()
    source_anchor = torch.randn(1, 24, 5, 4, 4)
    target_anchor = torch.randn(1, 24, 5, 6, 6)

    config = SimpleNamespace(transfer_mode="learned_3d", exact_prefix_mode="fallback")
    binding = SimpleNamespace(guidance=SimpleNamespace(mode="direction+temporal"))

    def fake_run(*_args, **_kwargs):
        record = validation._active()
        assert record is not None
        record.source_anchor = source_anchor.clone()
        record.target_anchor = target_anchor.clone()
        record.first_reference = target_anchor.clone()
        record.last_pre_guidance = target_anchor.clone()
        record.last_full_guidance = target_anchor.clone()
        record.guidance_calls = 3
        return torch.tensor(1.0)

    monkeypatch.setattr(validation, "_ORIGINAL_RUN_PROGRESSIVE", fake_run)
    monkeypatch.setattr(
        validation,
        "_eligible",
        lambda observed_config, observed_binding: observed_config is config and observed_binding is binding,
    )
    result = validation._run_progressive_anchor_wrapper(
        None,
        None,
        binding,
        config,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        0,
        [],
    )

    assert result.item() == 1.0
    with validation._state().lock:
        queued = validation._state().complete.pop()
    assert queued.completed_ns is not None
    assert queued.guidance_calls == 3
    assert queued.target_anchor.device.type == "cpu"


def test_validation_node_outputs_last_high_checkpoints() -> None:
    _clear_state()
    target = torch.randn(1, 24, 5, 6, 6)
    record = validation._ValidationRecord(
        enabled=True,
        source_anchor=torch.randn(1, 24, 5, 4, 4),
        target_anchor=target,
        first_reference=target.clone(),
        last_pre_guidance=target + 1.0,
        last_full_guidance=target + 2.0,
        guidance_calls=3,
        completed_ns=123,
    )
    record.move_outputs_to_cpu()
    with validation._state().lock:
        validation._state().complete.append(record)

    outputs = validation.H3GuidanceAnchorTransportValidation().extract({"samples": target})
    assert len(outputs) == 5
    assert torch.equal(outputs[0]["samples"], target)
    assert torch.equal(outputs[1]["samples"], target)
    assert torch.equal(outputs[2]["samples"], target + 1.0)
    assert torch.equal(outputs[3]["samples"], target + 2.0)
    assert '"no_extra_h3_evaluations": true' in outputs[4]
