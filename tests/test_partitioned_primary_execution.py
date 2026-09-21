from __future__ import annotations

import pytest

from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_AV_HANDOFF_SOURCE_KEY,
    PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY,
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
)


def _validate_candidate(**overrides):
    values = {
        "low_probe_execution_source": PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
        "prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        "audio_position_domain": PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        "audio_handoff_source": PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        "av_handoff_source": PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
        "guidance_trajectory_source": PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
        "audio_guided_overlap_mode": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
        "audio_guided_overlap_ticks": 4,
    }
    values.update(overrides)
    _validate_low_probe_execution_source_configuration(**values)


def test_source_uniform_primary_execution_requires_exact_pr64_control_tuple():
    _validate_candidate()

    with pytest.raises(PartitionedPreflightUnsupported, match="guidance_trajectory_source"):
        _validate_candidate(guidance_trajectory_source="main_exact_partitioned")
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_position_domain"):
        _validate_candidate(audio_position_domain="legacy_target")
    with pytest.raises(PartitionedPreflightUnsupported, match="audio_guided_overlap_ticks"):
        _validate_candidate(audio_guided_overlap_ticks=3)


def test_source_uniform_primary_controls_are_bounded_and_restore_exactly():
    keep = object()
    transformer = {
        "keep": keep,
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY: PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        PARTITIONED_AUDIO_POSITION_DOMAIN_KEY: PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        PARTITIONED_AV_HANDOFF_SOURCE_KEY: PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
        PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY: PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
        PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY: PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    }
    before = dict(transformer)

    with _source_uniform_primary_execution_controls(transformer):
        assert transformer["keep"] is keep
        assert (
            transformer[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY]
            == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
        )
        assert PARTITIONED_AUDIO_POSITION_DOMAIN_KEY not in transformer
        assert PARTITIONED_AV_HANDOFF_SOURCE_KEY not in transformer
        assert PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY not in transformer
        assert PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY not in transformer

    assert transformer == before
    assert transformer["keep"] is keep
