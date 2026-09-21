from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.audio_guided_overlap import apply_audio_guided_overlap_mask
from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PartitionedPreflightUnsupported,
    _validate_fast_main_width16_candidate_configuration,
    _verify_fast_main_width16_sampler_mask,
)


def _candidate(**overrides):
    values = {
        "low_probe_execution_source": PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
        "prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        "audio_position_domain": PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        "audio_handoff_source": PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        "av_handoff_source": PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
        "guidance_trajectory_source": PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
        "audio_guided_overlap_mode": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        "audio_guided_overlap_ticks": 16,
    }
    values.update(overrides)
    return _validate_fast_main_width16_candidate_configuration(**values)


def test_fast_main_width16_candidate_requires_single_exact_main_path():
    assert _candidate() is True

    with pytest.raises(PartitionedPreflightUnsupported, match="av_handoff_source='main_partitioned'"):
        _candidate(av_handoff_source=PARTITIONED_AV_HANDOFF_SOURCE_SHADOW)
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_position_domain='source_carrier'"):
        _candidate(audio_position_domain="legacy_target")
    with pytest.raises(PartitionedPreflightUnsupported, match="guidance_trajectory_source='main_exact_partitioned'"):
        _candidate(guidance_trajectory_source="source_carrier_uniform_shadow")


def test_fast_main_width16_candidate_does_not_capture_other_execution_modes():
    assert (
        _candidate(
            low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
        )
        is False
    )
    assert (
        _candidate(
            audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
            audio_guided_overlap_ticks=4,
        )
        is False
    )
    assert _candidate(audio_guided_overlap_ticks=15) is False


def test_fast_main_width16_verifies_canonical_sampler_mask_ramp():
    video_mask = torch.ones(1, 1, 4, 2, 2)
    audio_mask = torch.ones(1, 1, 2, 24)
    audio_mask[..., :20] = 0
    exact_mask, shapes = pack_streams((video_mask, audio_mask))
    runtime_mask, report = apply_audio_guided_overlap_mask(exact_mask, list(shapes), ticks=16)
    assert report["applied"] is True

    receipt = _verify_fast_main_width16_sampler_mask(
        runtime_mask,
        list(shapes),
        ticks=16,
    )
    assert receipt["sampler_mask_fractional_ticks"] == 16
    assert receipt["sampler_mask_ramp_verified"] is True
    assert receipt["sampler_mask_ramp"] == [
        0.0625,
        0.12109375,
        0.1796875,
        0.23828125,
        0.296875,
        0.35546875,
        0.4140625,
        0.47265625,
        0.53125,
        0.58984375,
        0.6484375,
        0.70703125,
        0.765625,
        0.82421875,
        0.8828125,
        0.94140625,
    ]


def test_fast_main_width16_rejects_wrong_overlap_width():
    video_mask = torch.ones(1, 1, 4, 2, 2)
    audio_mask = torch.ones(1, 1, 2, 24)
    audio_mask[..., :20] = 0
    exact_mask, shapes = pack_streams((video_mask, audio_mask))
    runtime_mask, _report = apply_audio_guided_overlap_mask(exact_mask, list(shapes), ticks=4)

    with pytest.raises(RuntimeError, match="expected 16, observed 4"):
        _verify_fast_main_width16_sampler_mask(
            runtime_mask,
            list(shapes),
            ticks=16,
        )
