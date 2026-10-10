"""Frozen low-source A/B/A ownership, RNG and replay qualification tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.low_sampler_ab import (
    capture_rng,
    compare_aba,
    counterfactual_low_inputs,
    frozen_rng,
)


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
