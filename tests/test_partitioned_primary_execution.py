from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.audio_guided_overlap import apply_audio_guided_overlap_mask
from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_AV_HANDOFF_SOURCE_KEY,
    PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PartitionedPreflightUnsupported,
    _source_uniform_primary_execution_controls,
    _validate_low_probe_execution_source_configuration,
    _verify_source_uniform_primary_overlap_mask,
)


def _validate_candidate(**overrides):
    values = {
        "low_probe_execution_source": PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
        "prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        "audio_position_domain": PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        "audio_handoff_source": PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        "av_handoff_source": PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
        "guidance_trajectory_source": PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
        "audio_guided_overlap_mode": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        "audio_guided_overlap_ticks": 32,
    }
    values.update(overrides)
    _validate_low_probe_execution_source_configuration(**values)


def test_source_uniform_primary_execution_requires_width32_sampler_mask_control_tuple():
    _validate_candidate()

    with pytest.raises(PartitionedPreflightUnsupported, match="av_handoff_source"):
        _validate_candidate(av_handoff_source="source_carrier_uniform_shadow")
    with pytest.raises(PartitionedPreflightUnsupported, match="guidance_trajectory_source"):
        _validate_candidate(guidance_trajectory_source="source_carrier_uniform_shadow")
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_guided_overlap_mode"):
        _validate_candidate(audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP)
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_guided_overlap_mode"):
        _validate_candidate(audio_guided_overlap_mode="unsupported")
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_position_domain"):
        _validate_candidate(audio_position_domain="legacy_target")
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_guided_overlap_ticks"):
        _validate_candidate(audio_guided_overlap_ticks=4)
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_guided_overlap_ticks"):
        _validate_candidate(audio_guided_overlap_ticks=16)
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_guided_overlap_ticks"):
        _validate_candidate(audio_guided_overlap_ticks=31)


def test_width32_overlap_is_rejected_on_main_then_shadow_execution():
    with pytest.raises(
        PartitionedPreflightUnsupported,
        match="reserved for source_carrier_uniform_only",
    ):
        _validate_candidate(
            low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
        )


def test_width32_overlap_is_allowed_only_for_collapsed_source_primary_inner_call():
    _validate_low_probe_execution_source_configuration(
        PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
        prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        audio_position_domain="legacy_target",
        audio_handoff_source=PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        av_handoff_source=PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
        guidance_trajectory_source=PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
        audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        audio_guided_overlap_ticks=32,
        source_uniform_primary_inner=True,
    )


def test_source_uniform_width32_runtime_mask_is_verified_exactly():
    video = torch.zeros(1, 24, 4, 4, 4)
    audio = torch.zeros(1, 32, 2, 80)
    _packed, shapes = pack_streams((video, audio))

    video_mask = torch.ones_like(video)
    video_mask[:, :, :1] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :65] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]
    runtime_mask, guided = apply_audio_guided_overlap_mask(exact_mask, list(shapes), ticks=32)
    assert guided["applied"] is True

    report = _verify_source_uniform_primary_overlap_mask(
        runtime_mask,
        exact_mask,
        list(shapes),
        ticks=32,
    )
    assert report["audio_prefix_ticks"] == 65
    assert report["fractional_audio_ticks"] == 32
    assert report["ramp_start_tick"] == 33
    assert report["ramp_stop_tick"] == 65
    assert report["video_mask_unchanged"] is True
    assert report["outside_ramp_audio_mask_unchanged"] is True


def test_source_uniform_width32_runtime_mask_rejects_wrong_width():
    video = torch.zeros(1, 24, 4, 4, 4)
    audio = torch.zeros(1, 32, 2, 80)
    _packed, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    video_mask[:, :, :1] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :65] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]
    runtime_mask, guided = apply_audio_guided_overlap_mask(exact_mask, list(shapes), ticks=16)
    assert guided["applied"] is True

    with pytest.raises(RuntimeError, match="exactly the requested contiguous fractional audio tail"):
        _verify_source_uniform_primary_overlap_mask(
            runtime_mask,
            exact_mask,
            list(shapes),
            ticks=32,
        )


def test_source_uniform_primary_controls_are_bounded_and_restore_exactly():
    keep = object()
    transformer = {
        "keep": keep,
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY: PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        PARTITIONED_AUDIO_POSITION_DOMAIN_KEY: PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY: PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    }
    before = dict(transformer)

    with _source_uniform_primary_execution_controls(transformer):
        assert transformer["keep"] is keep
        assert transformer[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY] == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
        assert PARTITIONED_AUDIO_POSITION_DOMAIN_KEY not in transformer
        assert PARTITIONED_AV_HANDOFF_SOURCE_KEY not in transformer
        assert PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY not in transformer
        assert PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY not in transformer

    assert transformer == before
    assert transformer["keep"] is keep
