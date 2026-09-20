from __future__ import annotations

from types import SimpleNamespace

import pytest

from h3_flow_regenerate.audio_guided_overlap import AUDIO_GUIDED_OVERLAP_ENV
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    apply_partitioned_diagnostic_controls,
    resolve_partitioned_audio_guided_overlap_ticks,
)
from h3_flow_regenerate.partitioned_node import (
    H3PartitionedExactPrefixDiagnosticHandoff,
    H3PartitionedExactPrefixHandoff,
)


class _Metrics:
    def __init__(self):
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))


def test_diagnostic_node_exposes_bounded_ab_controls_without_changing_ordinary_node():
    ordinary = H3PartitionedExactPrefixHandoff.INPUT_TYPES()["required"]
    diagnostic = H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]

    assert "vdn_linear_diagnostic" not in ordinary
    assert "audio_guided_overlap_ticks" not in ordinary

    assert diagnostic["vdn_linear_diagnostic"][0] == [
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    ]
    assert diagnostic["vdn_linear_diagnostic"][1]["default"] == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL
    assert diagnostic["audio_guided_overlap_ticks"][1]["default"] == 4
    assert diagnostic["audio_guided_overlap_ticks"][1]["min"] == 0
    assert diagnostic["audio_guided_overlap_ticks"][1]["max"] == 16


def test_apply_partitioned_diagnostic_controls_is_model_local_and_preserves_existing_transformer_options():
    model = SimpleNamespace(model_options={"transformer_options": {"keep": "value"}})
    metrics = _Metrics()

    returned_model, returned_metrics = apply_partitioned_diagnostic_controls(
        model,
        metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        audio_guided_overlap_ticks=0,
    )

    assert returned_model is model
    assert returned_metrics is metrics
    assert model.model_options["transformer_options"]["keep"] == "value"
    assert (
        model.model_options["transformer_options"][PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY]
        == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS
    )
    assert model.model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY] == 0
    assert metrics.events == [
        (
            "partitioned_diagnostic_controls",
            {
                "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
                "audio_guided_overlap_ticks": 0,
                "model_local": True,
                "native_vdn_unchanged": True,
                "production_default_changed": False,
            },
        )
    ]


def test_node_local_audio_overlap_override_wins_over_process_environment(monkeypatch):
    monkeypatch.setenv(AUDIO_GUIDED_OVERLAP_ENV, "11")
    model_options = {PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY: 0}

    ticks, source = resolve_partitioned_audio_guided_overlap_ticks(model_options)

    assert ticks == 0
    assert source == "diagnostic_node"


def test_ordinary_partitioned_audio_overlap_still_uses_existing_environment_or_default(monkeypatch):
    monkeypatch.setenv(AUDIO_GUIDED_OVERLAP_ENV, "7")

    ticks, source = resolve_partitioned_audio_guided_overlap_ticks({})

    assert ticks == 7
    assert source == "environment_or_default"


@pytest.mark.parametrize("bad", [-1, 17, True, 4.0, "4"])
def test_diagnostic_audio_overlap_rejects_noncanonical_values(bad):
    model = SimpleNamespace(model_options={"transformer_options": {}})
    with pytest.raises(ValueError):
        apply_partitioned_diagnostic_controls(
            model,
            _Metrics(),
            vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
            audio_guided_overlap_ticks=bad,
        )
