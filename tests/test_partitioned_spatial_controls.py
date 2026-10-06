"""Node and model-local controls for the suffix DC bridge and target-band continuation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_HANDOFF_TRANSFER_BICUBIC_CONTROL,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_SPATIAL_STAGE_CONTROL_KEY,
    PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS,
    PARTITIONED_SPATIAL_STAGE_PROGRESSIVE,
    PARTITIONED_SPATIAL_STAGE_SAME_GRID,
    PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
    PARTITIONED_SUFFIX_DC_BRIDGE_KEY,
    PARTITIONED_TARGET_BAND_TOKENS_DEFAULT,
    PARTITIONED_TARGET_BAND_TOKENS_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    PARTITIONED_VDN_TEMPORAL_CARRIER_KEY,
    PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,
    apply_partitioned_diagnostic_controls,
    resolve_partitioned_suffix_dc_bridge,
    resolve_partitioned_target_band_tokens,
)
from h3_flow_regenerate.partitioned_node import (
    H3PartitionedExactPrefixDiagnosticHandoff,
    H3PartitionedExactPrefixHandoff,
)
from h3_flow_regenerate.partitioned_scheduler import (
    _apply_partitioned_exact_overlap_bridge,
    _apply_partitioned_suffix_dc_bridge,
)


class _Metrics:
    def __init__(self):
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))


def _apply(**overrides):
    model = SimpleNamespace(model_options={"transformer_options": {"keep": "value"}})
    metrics = _Metrics()
    kwargs = dict(vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL, audio_guided_overlap_ticks=4)
    kwargs.update(overrides)
    apply_partitioned_diagnostic_controls(model, metrics, **kwargs)
    return model.model_options["transformer_options"], metrics.events[-1][1]


def test_new_controls_are_appended_after_every_historical_widget():
    ordinary = H3PartitionedExactPrefixHandoff.INPUT_TYPES()["required"]
    production = H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]
    assert "suffix_dc_bridge" not in ordinary
    assert "target_band_tokens" not in ordinary

    keys = list(production)
    assert keys[-3:] == ["video_guided_overlap_tokens", "suffix_dc_bridge", "target_band_tokens"]
    assert production["suffix_dc_bridge"][0] == "BOOLEAN"
    assert production["suffix_dc_bridge"][1]["default"] is True
    assert production["target_band_tokens"][0] == "INT"
    assert production["target_band_tokens"][1]["default"] == PARTITIONED_TARGET_BAND_TOKENS_DEFAULT
    assert production["target_band_tokens"][1]["min"] == 1
    # The historical option order is preserved; the band arm is appended.
    assert production["spatial_stage_control"][0] == list(PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS)
    assert PARTITIONED_SPATIAL_STAGE_CONTROL_OPTIONS == (
        PARTITIONED_SPATIAL_STAGE_PROGRESSIVE,
        PARTITIONED_SPATIAL_STAGE_SAME_GRID,
        PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
    )
    assert production["spatial_stage_control"][1]["default"] == PARTITIONED_SPATIAL_STAGE_SAME_GRID


def test_suffix_dc_bridge_leaf_is_absent_by_default_and_explicit_only_when_disabled():
    transformer, fields = _apply()
    assert PARTITIONED_SUFFIX_DC_BRIDGE_KEY not in transformer
    assert "suffix_dc_bridge" not in fields
    assert resolve_partitioned_suffix_dc_bridge(transformer) is True

    transformer, fields = _apply(suffix_dc_bridge=False)
    assert transformer[PARTITIONED_SUFFIX_DC_BRIDGE_KEY] is False
    assert fields["suffix_dc_bridge"] is False
    assert transformer["keep"] == "value"
    assert resolve_partitioned_suffix_dc_bridge(transformer) is False

    with pytest.raises(TypeError, match="boolean"):
        _apply(suffix_dc_bridge=1)
    with pytest.raises(ValueError, match="explicit value False"):
        resolve_partitioned_suffix_dc_bridge({PARTITIONED_SUFFIX_DC_BRIDGE_KEY: True})


def test_target_band_tokens_are_published_only_for_the_band_arm():
    transformer, fields = _apply(target_band_tokens=7)
    assert PARTITIONED_TARGET_BAND_TOKENS_KEY not in transformer
    assert "target_band_tokens" not in fields

    transformer, fields = _apply(spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND, target_band_tokens=7)
    assert transformer[PARTITIONED_SPATIAL_STAGE_CONTROL_KEY] == PARTITIONED_SPATIAL_STAGE_TARGET_BAND
    assert transformer[PARTITIONED_TARGET_BAND_TOKENS_KEY] == 7
    assert fields["target_band_tokens"] == 7
    assert resolve_partitioned_target_band_tokens(transformer) == 7
    assert resolve_partitioned_target_band_tokens({}) == PARTITIONED_TARGET_BAND_TOKENS_DEFAULT

    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="positive integer"):
            _apply(spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND, target_band_tokens=bad)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"handoff_transfer_control": PARTITIONED_HANDOFF_TRANSFER_BICUBIC_CONTROL}, "learned_3d"),
        ({"prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE}, "exact_target_partitioned"),
        ({"low_probe_execution_source": PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY}, "main_then_shadow"),
        ({"av_handoff_source": PARTITIONED_AV_HANDOFF_SOURCE_SHADOW}, "main partitioned"),
    ],
)
def test_target_band_rejects_unsupported_companion_selectors(override, error):
    with pytest.raises(ValueError, match=error):
        _apply(spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND, **override)


def test_target_band_destination_stencil_preserves_other_controls_and_rejects_suppression():
    transformer, fields = _apply(
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        vdn_temporal_carrier_policy=PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,
        target_band_tokens=7,
        suffix_dc_bridge=False,
    )
    assert transformer[PARTITIONED_VDN_TEMPORAL_CARRIER_KEY] == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION
    assert transformer[PARTITIONED_TARGET_BAND_TOKENS_KEY] == 7
    assert transformer[PARTITIONED_SUFFIX_DC_BRIDGE_KEY] is False
    assert transformer["keep"] == "value"
    assert fields["vdn_temporal_carrier_policy"] == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION
    with pytest.raises(ValueError, match="vdn_linear_diagnostic='normal'"):
        _apply(
            spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
            vdn_temporal_carrier_policy=PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,
            vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
        )


def _bridge_operands():
    torch.manual_seed(3)
    learned = torch.randn(1, 24, 6, 4, 6)
    state = torch.randn_like(learned)
    exact = learned[:, :, :2].clone() + 0.3
    return state, learned, exact


@pytest.mark.parametrize("bridge", ["plain", "exact_overlap"])
def test_disabled_suffix_dc_bridge_leaves_state_and_clean_untouched(bridge):
    state, learned, exact = _bridge_operands()
    if bridge == "plain":
        enabled_state, _, enabled_metrics = _apply_partitioned_suffix_dc_bridge(
            state, learned, exact, sigma=0.8, enabled=True
        )
        disabled_state, disabled_clean, metrics = _apply_partitioned_suffix_dc_bridge(
            state, learned, exact, sigma=0.8, enabled=False
        )
    else:
        enabled_state, _, _, enabled_metrics = _apply_partitioned_exact_overlap_bridge(state, learned, exact, sigma=0.8)
        disabled_state, disabled_clean, _, metrics = _apply_partitioned_exact_overlap_bridge(
            state, learned, exact, sigma=0.8, dc_enabled=False
        )
    # The enabled bridge changes exactly the first generated token; disabling it changes nothing.
    assert enabled_metrics["suffix_dc_bridge_corrected_tokens"] == 1
    assert not torch.equal(enabled_state[:, :, 2], state[:, :, 2])
    assert torch.equal(enabled_state[:, :, 3:], state[:, :, 3:])
    assert torch.equal(disabled_state, state)
    assert torch.equal(disabled_clean, learned)
    assert metrics["suffix_dc_bridge_enabled"] is False
    assert metrics["suffix_dc_bridge_corrected_tokens"] == 0
    assert metrics["suffix_dc_bridge_delta_rms"] == 0.0
    assert metrics["suffix_dc_bridge_prefix_t"] == 2
