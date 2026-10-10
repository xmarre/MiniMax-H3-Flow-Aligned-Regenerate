"""Frozen low-source A/B/A ownership, RNG and replay qualification tests."""

from __future__ import annotations

import random
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from h3_flow_regenerate import partitioned_scheduler as scheduler
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.low_sampler_ab import (
    capture_rng,
    compare_aba,
    counterfactual_low_inputs,
    frozen_rng,
)
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_stage import PartitionedStagePlan


@pytest.fixture
def inputs():
    g = torch.Generator().manual_seed(987)
    video = torch.randn((1, 24, 17, 4, 4), generator=g)
    audio = torch.randn((1, 32, 2, 4, 4), generator=g)
    packed, shapes = pack_streams((video, audio))
    return video, audio, packed, shapes


def test_frozen_source_B_changes_only_prefix_and_owns_output(inputs):
    video, audio, original, shapes = inputs
    alternate = video[:, :, :7].clone() + 0.03
    original_bytes = original.detach().clone()
    candidate = counterfactual_low_inputs(original, shapes, video[:, :, :7], alternate, 7)
    b_video, b_audio = unpack_streams(candidate, shapes)
    assert torch.equal(b_video[:, :, :7], alternate)
    assert torch.equal(b_video[:, :, 7:], video[:, :, 7:])
    assert torch.equal(b_audio, audio)
    assert torch.equal(original, original_bytes)
    assert candidate.data_ptr() != original.data_ptr()


def test_mismatched_actual_input_or_native_phase_is_rejected(inputs):
    video, _audio, original, shapes = inputs
    with pytest.raises(ValueError, match="actual sampler input"):
        counterfactual_low_inputs(original, shapes, video[:, :, :7] + 1, video[:, :, :7], 7)
    with pytest.raises(ValueError, match="native phase"):
        counterfactual_low_inputs(original, shapes, video[:, :, :6], video[:, :, :6] + 1, 6)


def test_rng_rewinds_identically_and_restores_all_global_states():
    before = capture_rng()
    try:
        random.seed(127)
        np.random.seed(127)
        torch.manual_seed(127)
        frozen = capture_rng()
        samples = []
        for _ in range(2):
            with frozen_rng(frozen):
                samples.append((random.random(), np.random.rand(), torch.randn(2)))
        assert samples[0][0] == samples[1][0]
        assert samples[0][1] == samples[1][1]
        assert torch.equal(samples[0][2], samples[1][2])
        following = (random.random(), np.random.rand(), torch.randn(2))
        with frozen_rng(frozen):
            _ = random.random(), np.random.rand(), torch.randn(2)
        assert random.random() != following[0]
    finally:
        from h3_flow_regenerate.low_sampler_ab import restore_rng

        restore_rng(before)


def test_reproduction_requires_first_prediction_and_source_suffix_match(inputs):
    video, _audio, _packed, _shapes = inputs
    first = torch.randn((1, 24, 17, 4, 4))
    b = video.clone()
    b[:, :, 7:] += 0.09
    b_first = first.clone()
    b_first[:, :, 7:] += 0.09
    result = compare_aba(video, b, video.clone(), first, b_first, first.clone(), prefix_t=7)
    assert result["reproduction_verified"] is True
    assert result["causal_low_input_effect_qualified"] is True
    assert result["changed_b_clean_suffix"]["rms"] > 0.08
    drifted_first = first.clone()
    drifted_first[:, :, -1] += 0.002
    unqualified = compare_aba(video, b, video.clone(), first, b_first, drifted_first, prefix_t=7)
    assert unqualified["causal_low_input_effect_qualified"] is False
    assert unqualified["repeat_a_first_actual_prediction"]["abs_max"] > 0.001


def test_later_model_state_leakage_rejects_a2(inputs):
    video, _audio, _packed, _shapes = inputs
    first = video.clone()
    candidate = video + 0.5
    replay = video.clone()
    replay[:, :, 8] += 0.0002
    report = compare_aba(video, candidate, replay, first, candidate, first, prefix_t=7)
    assert report["reproduction_verified"] is False
    assert report["model_execution_history_strictly_isolated"] is False


def test_real_scheduler_aba_shadow_uses_two_independent_low_probe_lifetimes(inputs, monkeypatch):
    video, audio, packed, shapes = inputs
    noise, _ = pack_streams((torch.zeros_like(video), torch.zeros_like(audio)))
    video_mask = torch.cat((torch.zeros_like(video[:, :, :7]), torch.ones_like(video[:, :, 7:])), dim=2)
    mask, _ = pack_streams((video_mask, torch.ones_like(audio)))
    original_prefix = video[:, :, :7].clone()
    alternate_prefix = original_prefix + 0.125
    low_plan = PartitionedStagePlan(
        prefix=original_prefix.clone().float(),
        temporal=17,
        source_h=4,
        source_w=4,
        prefix_noise=torch.zeros_like(original_prefix),
    )
    low_stage_calls = []
    capture_stack = []
    binding = SimpleNamespace(
        metrics=H3FlowMetrics(),
        trajectory=object(),
        capture_enabled=True,
        active_capture=None,
        active_guidance_run=None,
        captured_run_id="main-trajectory",
    )
    main_metrics = binding.metrics
    original_trajectory = binding.trajectory
    base = SimpleNamespace()
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=base),
        conds={"positive": [{"original": True}]},
    )
    original_conds = guider.conds
    state = {"stage": None}

    @contextmanager
    def stage(_guider, name):
        previous = state["stage"]
        state["stage"] = name
        try:
            yield
        finally:
            state["stage"] = previous

    @contextmanager
    def passthrough(*_args, **_kwargs):
        yield

    @contextmanager
    def recorder(_binding, _guider, _sampler, _sigmas, _shapes, *, enabled):
        assert enabled and _binding is binding
        holder = {}
        record = {}
        capture_stack.append(record)
        yield holder
        holder["run"] = SimpleNamespace(
            exact_samples=lambda: [
                SimpleNamespace(
                    phase="predicted",
                    video_x0=record["first"],
                )
            ]
        )

    def fake_executor(_noise, latents, _sampler, _sigmas, _mask, _callback, _pbar, _seed, *, latent_shapes):
        latent_video, latent_audio = unpack_streams(latents, latent_shapes)
        if state["stage"] == "low":
            low_stage_calls.append(latent_video[:, :, :7].clone())
            capture_stack[-1]["first"] = latent_video.clone()
            return latents.clone()
        assert state["stage"] == "probe"
        changed_video = latent_video.clone()
        changed_video[:, :, 7:] += latent_video[:, :, :7].mean() * 0.02
        return pack_streams((changed_video, latent_audio))[0]

    monkeypatch.setattr(scheduler, "_isolated_shadow_trajectory_capture", recorder)
    monkeypatch.setattr(scheduler, "_flow_stage_contract", stage)
    monkeypatch.setattr(scheduler, "_high_stage_contract", passthrough)
    monkeypatch.setattr(scheduler, "_partitioned_stage_contract", passthrough)
    monkeypatch.setattr(scheduler, "_reset_guider_conds", lambda g, *, template: setattr(g, "conds", template))
    monkeypatch.setattr(scheduler, "_raw_sampler_state", lambda _base, result, _shapes, _sigma: result)
    monkeypatch.setattr(scheduler, "_process_latent_in", lambda _base, value, _shapes: value)
    monkeypatch.setattr(scheduler, "_noise_argument", lambda _base, raw, _sigma, _latent: raw)
    monkeypatch.setattr(scheduler, "_make_probe_sampler", lambda sampler: sampler)
    monkeypatch.setattr(scheduler, "sampler_name", lambda sampler: "sample_euler")
    monkeypatch.setattr(scheduler, "_conditioning_signature", lambda _guider: "identical")
    monkeypatch.setattr(scheduler, "_interop_identity", lambda _options: ("fake-session", "2"))
    monkeypatch.setattr(
        scheduler,
        "export_residual_geometry_evidence",
        lambda tensors, **_kwargs: {"status": "mock", "names": sorted(tensors)},
    )
    first_video = video.clone()
    original_clean = video.clone()
    original_clean[:, :, 7:] += video[:, :, :7].mean() * 0.02
    original_clean_packed = pack_streams((original_clean, audio))[0]
    frozen = capture_rng()
    result = scheduler._run_frozen_source_low_aba(
        fake_executor,
        guider,
        binding,
        frozen_pre_rng=frozen,
        conditioning_template=original_conds,
        sampler=object(),
        low_sigmas=torch.tensor([1.0, 0.5]),
        probe_sigmas=torch.tensor([0.5]),
        low_shapes=shapes,
        low_noise=noise,
        original_latent=packed,
        low_mask=mask,
        seed=42,
        disable_pbar=True,
        low_plan=low_plan,
        original_prefix=original_prefix,
        candidate_prefix=alternate_prefix,
        authoritative_prefix=torch.randn((1, 24, 7, 8, 8)),
        baseline_probe_internal=original_clean_packed,
        baseline_first_actual=first_video,
        sigma=0.5,
        index=1,
        source_h=4,
        source_w=4,
        target_h=8,
        target_w=8,
    )
    assert result["causal_low_input_effect_qualified"] is True
    assert result["known_input_pairing"]["input_pair_eligible"] is True
    assert result["changed_b_clean_suffix"]["rms"] > 0
    assert result["changed_b_first_actual_prediction"]["rms"] > 0
    assert len(low_stage_calls) == 2
    assert torch.equal(low_stage_calls[0], alternate_prefix)
    assert torch.equal(low_stage_calls[1], original_prefix)
    assert binding.metrics is main_metrics
    assert binding.trajectory is original_trajectory
    assert guider.conds is original_conds
    assert binding.captured_run_id == "main-trajectory"
    assert result["evidence"]["status"] == "mock"
