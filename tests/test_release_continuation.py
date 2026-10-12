"""Release interface and saved-workflow compatibility regressions."""

from types import SimpleNamespace

import pytest

from h3_flow_regenerate.legacy_diagnostics import RETIRED_LOW_OPTIONS, retire_low_replay_controls
from h3_flow_regenerate.partitioned_node import (
    H3NativeSourceContinuationHandoff,
    H3PartitionedExactPrefixDiagnosticHandoff,
    H3PartitionedExactPrefixHandoff,
)
from h3_flow_regenerate.source_prefix_carry import SOURCE_PREFIX_CARRY_KEY


class Metrics:
    def __init__(self):
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))


def test_retired_controls_clear_cached_model_observers_without_touching_other_owners():
    keep = object()
    model = SimpleNamespace(model_options={**dict.fromkeys(RETIRED_LOW_OPTIONS, object()), "keep": keep})
    metrics = Metrics()
    retire_low_replay_controls(model, metrics, low_sampler_aba=True, low_native_trace_step=5)
    assert model.model_options == {"keep": keep}
    kind, receipt = metrics.events[0]
    assert kind == "retired_low_replay_controls"
    assert receipt["cleared_inherited_options"] == list(RETIRED_LOW_OPTIONS)
    assert receipt["requested"] == ["low_sampler_aba", "low_native_trace_step"]
    assert receipt["extra_h3_nfe"] == 0
    assert receipt["live_replay_executed"] is receipt["kernel_observers_installed"] is False
    retire_low_replay_controls(model, metrics, low_sampler_aba=False, low_native_trace_step=-1)
    assert len(metrics.events) == 1


@pytest.mark.parametrize(
    "controls,error",
    [
        ({"low_sampler_aba": 1}, TypeError),
        ({"low_native_trace_step": True}, ValueError),
        ({"low_native_trace_pair_block": 512}, ValueError),
    ],
)
def test_retired_controls_reject_malformed_serialized_values(controls, error):
    with pytest.raises(error):
        retire_low_replay_controls(SimpleNamespace(model_options={}), Metrics(), **controls)


@pytest.mark.parametrize("projection", ["native_source_carry", "latent_bicubic"])
@pytest.mark.parametrize("detail", [False, True])
def test_production_interface_uses_existing_runtime_contract(monkeypatch, projection, detail):
    def base_patch(self, **kwargs):
        return SimpleNamespace(model_options={"transformer_options": {}}), Metrics()

    monkeypatch.setattr(H3PartitionedExactPrefixHandoff, "patch", base_patch)
    spec = H3NativeSourceContinuationHandoff.INPUT_TYPES()
    kwargs = {k: v[1]["default"] for k, v in spec["required"].items() if len(v) > 1}
    kwargs.update(
        model=object(),
        trajectory=object(),
        learned_upscaler=object(),
        source_prefix_projection=projection,
        uniform_source_detail_transport=detail,
    )
    new, _ = H3NativeSourceContinuationHandoff().patch(**kwargs)
    old, _ = H3PartitionedExactPrefixDiagnosticHandoff().patch(
        **kwargs,
        vdn_linear_diagnostic="normal",
        prefix_transformer_context="exact_target_partitioned",
        spatial_stage_control="progressive_uniform_source",
        low_sampler_aba=True,
        low_native_trace_step=5,
        low_native_trace_token_admission=True,
    )
    assert (SOURCE_PREFIX_CARRY_KEY in new.model_options) == (projection == "native_source_carry")
    assert (SOURCE_PREFIX_CARRY_KEY in old.model_options) == (projection == "native_source_carry")
    assert {k: v for k, v in new.model_options.items() if k != SOURCE_PREFIX_CARRY_KEY} == {
        k: v for k, v in old.model_options.items() if k != SOURCE_PREFIX_CARRY_KEY
    }
    assert not set(RETIRED_LOW_OPTIONS).intersection(new.model_options)
    assert not set(RETIRED_LOW_OPTIONS).intersection(old.model_options)


def test_production_interface_excludes_live_experiments_and_mismatched_audio():
    spec = H3NativeSourceContinuationHandoff.INPUT_TYPES()
    assert not H3NativeSourceContinuationHandoff.DEPRECATED
    assert H3PartitionedExactPrefixDiagnosticHandoff.DEPRECATED
    assert not any(k.startswith("low_native_trace") or k == "low_sampler_aba" for k in spec["required"])
    assert "spatial_stage_control" not in spec["required"]
    assert "model_timestep_only" not in spec["required"]["audio_guided_overlap_mode"][0]
    kwargs = {k: v[1]["default"] for k, v in spec["required"].items() if len(v) > 1}
    kwargs.update(
        model=object(), trajectory=object(), learned_upscaler=object(), audio_guided_overlap_mode="model_timestep_only"
    )
    with pytest.raises(ValueError, match="coherent audio"):
        H3NativeSourceContinuationHandoff().patch(**kwargs)
