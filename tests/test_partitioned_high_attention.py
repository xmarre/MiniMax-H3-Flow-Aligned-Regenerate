from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
    PARTITIONED_VDN_TEMPORAL_CARRIER_KEY,
)
from h3_flow_regenerate.partitioned_scheduler import _partitioned_high_stage_contract
from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStagePlan
from h3_flow_regenerate.runtime import _flow_stage_contract


def _plan(source_hw=(4, 6)):
    prefix = torch.randn(1, 24, 2, 8, 12)
    return PartitionedStagePlan(
        prefix=prefix,
        prefix_noise=torch.randn_like(prefix),
        temporal=7,
        source_h=source_hw[0],
        source_w=source_hw[1],
    )


@pytest.mark.parametrize("source_hw", [(4, 6), (8, 12)])
@pytest.mark.parametrize("failure", [False, True])
def test_high_owns_target_grid_context_and_restores_low_controls(source_hw, failure, monkeypatch, tmp_path):
    monkeypatch.setenv("H3_FLOW_BOUNDARY_WITNESS_DIR", str(tmp_path))
    plan = _plan(source_hw)
    prefix = plan.prefix.clone()
    prefix_noise = plan.prefix_noise.clone()
    previous_refinement = {"provider_note": "keep"}
    options = {
        "h3_refinement": previous_refinement,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY: "bypass_partitioned_linear",
        PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY: "dense_suffix_same_domain",
        PARTITIONED_VDN_TEMPORAL_CARRIER_KEY: "destination_grid_stencil_v1",
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY: "exact_target_partitioned",
        PARTITIONED_AUDIO_POSITION_DOMAIN_KEY: "source_carrier",
    }
    original = options.copy()
    guider = SimpleNamespace(model_options={"transformer_options": options})
    metrics = H3FlowMetrics()

    def execute():
        with _flow_stage_contract(guider, "high"), _partitioned_high_stage_contract(guider, plan, metrics):
            owner = options[PARTITIONED_STAGE_KEY]
            assert owner.plan.source_grid == owner.plan.target_grid == (4, 6)
            assert owner.plan.prefix is plan.prefix
            assert owner.plan.prefix_noise is plan.prefix_noise
            assert owner.vdn_linear_diagnostic == owner.softmax_diagnostic == "normal"
            assert owner.vdn_temporal_carrier_policy == "native_grid_then_map_v1"
            assert owner.audio_position_domain == "legacy_target"
            assert owner.boundary_witness is None
            assert options["h3_refinement"]["min_actual_prefix_steps"] == 1
            assert options["h3_refinement"]["source"] == "h3_flow_partitioned_refinement"
            assert options["h3_refinement"]["provider_note"] == "keep"
            if failure:
                raise ValueError("downstream model failure")

    if failure:
        with pytest.raises(ValueError, match="downstream model failure"):
            execute()
    else:
        execute()
    assert options == original
    assert options["h3_refinement"] is previous_refinement
    assert torch.equal(plan.prefix, prefix)
    assert torch.equal(plan.prefix_noise, prefix_noise)


def test_source_uniform_comparison_keeps_native_high_and_new_attention_startup():
    options = {PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY: "bypass_partitioned_linear"}
    original = options.copy()
    guider = SimpleNamespace(model_options={"transformer_options": options})
    metrics = H3FlowMetrics()
    with (
        _flow_stage_contract(guider, "high"),
        _partitioned_high_stage_contract(guider, _plan(), metrics, exact_prefix_attention=False),
    ):
        assert PARTITIONED_STAGE_KEY not in options
        assert options[PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY] == original[PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY]
        assert options["h3_refinement"]["source"] == "h3_flow_partitioned_refinement"
    assert options == original


def test_high_cannot_replace_an_existing_partition_owner():
    existing = object()
    options = {PARTITIONED_STAGE_KEY: existing, PARTITIONED_AUDIO_POSITION_DOMAIN_KEY: "source_carrier"}
    original = options.copy()
    guider = SimpleNamespace(model_options={"transformer_options": options})
    with (
        pytest.raises(RuntimeError, match="nested partitioned"),
        _partitioned_high_stage_contract(guider, _plan(), H3FlowMetrics()),
    ):
        pass
    assert options == original
    assert options[PARTITIONED_STAGE_KEY] is existing
