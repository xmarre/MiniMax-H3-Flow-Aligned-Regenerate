from __future__ import annotations

import pytest

from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_AV_HANDOFF_SOURCE_KEY,
    PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
)
from h3_flow_regenerate.partitioned_node import H3PartitionedExactPrefixDiagnosticHandoff
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
        "av_handoff_source": PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
        "guidance_trajectory_source": PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
        "audio_guided_overlap_mode": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        "audio_guided_overlap_ticks": 16,
    }
    values.update(overrides)
    _validate_low_probe_execution_source_configuration(**values)


def test_exact_onepass_execution_requires_main_owners_and_exact_context():
    for ticks in (0, 4, 16):
        _validate_candidate(
            low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY,
            audio_guided_overlap_ticks=ticks,
        )
        _validate_candidate(
            low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY,
            audio_guided_overlap_ticks=ticks,
            audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
        )

    with pytest.raises(PartitionedPreflightUnsupported, match="prefix_transformer_context"):
        _validate_candidate(
            low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY,
            prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
        )
    with pytest.raises(PartitionedPreflightUnsupported, match="av_handoff_source"):
        _validate_candidate(
            low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY,
            av_handoff_source="source_carrier_uniform_shadow",
        )


def test_exact_onepass_node_rejects_unsupported_diagnostics_before_patch():
    node = H3PartitionedExactPrefixDiagnosticHandoff()
    base = {
        "model": None,
        "trajectory": None,
        "source_mode": "scale",
        "source_scale": 0.7,
        "source_width": 864,
        "source_height": 640,
        "handoff_coordinate": 0.35,
        "handoff_selection": "fixed",
        "guidance_mode": "direction+temporal",
        "direction_weight": 0.25,
        "acceleration_weight": 0.25,
        "consistency_weight": 0.25,
        "low_frequency_cutoff": 0.25,
        "learned_upscaler": None,
        "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        "audio_guided_overlap_mode": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        "audio_guided_overlap_ticks": 16,
        "prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        "audio_position_domain": PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        "audio_handoff_source": PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        "av_handoff_source": PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
        "guidance_trajectory_source": PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
        "low_probe_execution_source": PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY,
    }

    with pytest.raises(RuntimeError, match="vdn_linear_diagnostic='normal'"):
        node.patch(**{**base, "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS})

    with pytest.raises(RuntimeError, match="audio_position_domain='source_carrier'"):
        node.patch(**{**base, "audio_position_domain": PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY})


def test_source_uniform_primary_execution_accepts_configured_sampler_owned_overlap_width():
    for ticks in (0, 1, 4, 15, 16):
        _validate_candidate(audio_guided_overlap_ticks=ticks)
        _validate_candidate(
            audio_guided_overlap_ticks=ticks,
            audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
        )

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
