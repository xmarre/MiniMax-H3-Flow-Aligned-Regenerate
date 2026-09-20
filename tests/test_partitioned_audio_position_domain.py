from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    normalize_audio_position_domain,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PartitionedPreflightUnsupported,
    _validate_audio_position_candidate_configuration,
    _verify_audio_position_domain_diagnostic,
)
from h3_flow_regenerate.partitioned_stage import (
    PARTITIONED_POSITION_POLICY_TAG,
    PartitionedStagePlan,
    PartitionedStageRuntime,
    partitioned_positions,
    partitioned_positions_for_runtime,
    partitioned_positions_with_audio_policy,
)


class _Native:
    @staticmethod
    def _frame_grid(h, w):
        rows = torch.zeros((h * w // 4, 2), dtype=torch.float64)
        width = torch.linspace(-float(w), float(w), w // 2, dtype=torch.float64)
        return rows, width

    @staticmethod
    def _video_grid(vt, frame, cursor):
        out = torch.zeros((vt, frame.shape[0], 3), dtype=torch.float64)
        out[:, :, 0] = torch.arange(vt, dtype=torch.float64)[:, None] + float(cursor)
        out[:, :, 1:] = frame
        return out.reshape(-1, 3)

    @staticmethod
    def _audio_grid(cursor, t, w_low, w_high):
        out = torch.zeros((t * 2, 3), dtype=torch.float64)
        out[:, 0] = (float(cursor) + torch.arange(t, dtype=torch.float64)).repeat(2)
        out[:t, 2] = float(w_low)
        out[t:, 2] = float(w_high)
        return out


def _plan():
    prefix = torch.zeros((1, 24, 2, 8, 8), dtype=torch.float32)
    return PartitionedStagePlan(
        prefix=prefix,
        temporal=5,
        source_h=4,
        source_w=4,
        prefix_noise=prefix.clone(),
    )


def _layout():
    plan = _plan()
    text = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    ref_audio = _Native._audio_grid(2.0, 2, -77.0, 77.0)
    target_audio = _Native._audio_grid(4.0, 3, -8.0, 8.0)
    video = _Native._video_grid(plan.temporal, torch.zeros((plan.source_rows, 2), dtype=torch.float64), 4.0)
    position_ids = torch.cat((text, ref_audio, target_audio, video))
    audio_start = len(text) + len(ref_audio)
    video_start = audio_start + len(target_audio)
    return SimpleNamespace(
        position_ids=position_ids,
        segments=[
            (0, len(text), "text"),
            (len(text), audio_start, "ref_audio"),
            (audio_start, video_start, "audio"),
            (video_start, len(position_ids), "video"),
        ],
    )


def test_legacy_audio_position_domain_is_exact_existing_arithmetic():
    plan = _plan()
    layout = _layout()
    expected = partitioned_positions(_Native, plan, layout)
    actual, policy = partitioned_positions_with_audio_policy(
        _Native, plan, layout, PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY
    )
    assert policy is None
    assert torch.equal(actual, expected)


def test_source_carrier_changes_only_target_audio_spatial_coordinates():
    plan = _plan()
    layout = _layout()
    incoming_position_ids = layout.position_ids.clone()
    legacy = partitioned_positions(_Native, plan, layout)
    candidate, policy = partitioned_positions_with_audio_policy(
        _Native, plan, layout, PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE
    )
    assert policy is not None
    assert torch.equal(layout.position_ids, incoming_position_ids)
    assert policy.signature[0] == PARTITIONED_POSITION_POLICY_TAG
    audio_start, audio_stop = policy.audio_range
    assert torch.equal(candidate[audio_start:audio_stop, 0], legacy[audio_start:audio_stop, 0])
    assert not torch.equal(candidate[audio_start:audio_stop, 2], legacy[audio_start:audio_stop, 2])
    assert torch.equal(candidate[:audio_start], legacy[:audio_start])
    assert torch.equal(candidate[audio_stop:], legacy[audio_stop:])
    ref_start, ref_stop, _ = layout.segments[1]
    assert torch.equal(candidate[ref_start:ref_stop], legacy[ref_start:ref_stop])
    assert policy.target_audio_spatial_endpoints == ((0.0, -8.0), (0.0, 8.0))
    assert policy.source_audio_spatial_endpoints == ((0.0, -4.0), (0.0, 4.0))
    assert policy.audio_temporal_equal is True
    assert policy.non_audio_before_digest == policy.non_audio_after_digest


def test_candidate_policy_and_positions_are_stage_owned_and_immutable():
    runtime = PartitionedStageRuntime(
        plan=_plan(),
        metrics=SimpleNamespace(),
        audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    )
    layout = _layout()
    first, policy = partitioned_positions_for_runtime(_Native, runtime, layout)
    second, repeated = partitioned_positions_for_runtime(_Native, runtime, layout)
    assert repeated is policy
    assert second is first
    assert policy.stage_owner_generation == id(runtime)

    drifted = _layout()
    audio_start, _, _ = drifted.segments[2]
    drifted.position_ids[audio_start, 2] += 1.0
    with pytest.raises(RuntimeError, match="drifted"):
        partitioned_positions_for_runtime(_Native, runtime, drifted)


def test_audio_position_selector_is_bounded():
    assert normalize_audio_position_domain("legacy_target") == "legacy_target"
    assert normalize_audio_position_domain("source_carrier") == "source_carrier"
    with pytest.raises(ValueError, match="audio position domain"):
        normalize_audio_position_domain("invented")


def test_candidate_execution_gate_fails_closed_then_accepts_actual_block_zero():
    class Metrics:
        def __init__(self):
            self.counters = {}
            self.events = []

        def increment(self, name, value=1):
            self.counters[name] = self.counters.get(name, 0) + value

        def event(self, kind, **fields):
            self.events.append((kind, fields))

    metrics = Metrics()
    with pytest.raises(RuntimeError, match="no actual low/probe"):
        _verify_audio_position_domain_diagnostic(
            metrics,
            PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
            block0_calls_before=0,
            wrapper_entries_before=0,
            model_timestep_calls_before=0,
        )
    metrics.increment("partitioned_audio_position_candidate_wrapper_entries")
    metrics.increment("partitioned_audio_position_source_carrier_block0_calls")
    _verify_audio_position_domain_diagnostic(
        metrics,
        PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        block0_calls_before=0,
        wrapper_entries_before=0,
        model_timestep_calls_before=0,
    )
    assert metrics.events[-1][0] == "partitioned_audio_position_domain_verified"
    assert metrics.events[-1][1]["actual_block0_calls"] == 1


def test_candidate_contract_requires_heterogeneous_context_and_normal_vdn():
    _validate_audio_position_candidate_configuration(
        PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    )
    with pytest.raises(PartitionedPreflightUnsupported):
        _validate_audio_position_candidate_configuration(
            PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
            PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        )
    with pytest.raises(PartitionedPreflightUnsupported):
        _validate_audio_position_candidate_configuration(
            PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
            PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        )
