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


@pytest.mark.parametrize("projection", ["native_source_carry", "latent_bicubic", "native_source_carry_target_band"])
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
    carried = projection != "latent_bicubic"
    assert (SOURCE_PREFIX_CARRY_KEY in new.model_options) == carried
    assert (SOURCE_PREFIX_CARRY_KEY in old.model_options) == carried
    if carried:
        target_band = projection == "native_source_carry_target_band"
        assert new.model_options[SOURCE_PREFIX_CARRY_KEY].target_band is target_band
        assert old.model_options[SOURCE_PREFIX_CARRY_KEY].target_band is target_band
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


def test_production_defaults_match_release_profile_and_preserve_legacy_detail_default():
    spec = H3NativeSourceContinuationHandoff.INPUT_TYPES()["required"]
    defaults = {
        "source_mode": "scale",
        "source_scale": 0.70,
        "source_width": 864,
        "source_height": 640,
        "handoff_coordinate": 0.35,
        "handoff_selection": "fixed",
        "guidance_mode": "direction+temporal",
        "direction_weight": 0.25,
        "acceleration_weight": 0.25,
        "consistency_weight": 0.25,
        "low_frequency_cutoff": 0.25,
        "temporal_weight": 0.20,
        "audio_guided_overlap_mode": "sampler_mask_exact_timestep",
        "audio_guided_overlap_ticks": 16,
        "uniform_source_detail_transport": True,
        "source_prefix_projection": "native_source_carry",
    }
    assert {k: spec[k][1]["default"] for k in defaults} == defaults
    assert (
        H3NativeSourceContinuationHandoff.INPUT_TYPES()["optional"]["capture_boundary_witness"][1]["default"] is False
    )
    assert (
        H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]["uniform_source_detail_transport"][1][
            "default"
        ]
        is False
    )


@pytest.mark.parametrize("mask_present", [False, True])
def test_unprotected_single_chunk_routes_to_ordinary_progressive_runtime(monkeypatch, mask_present):
    import torch

    from h3_flow_regenerate import partitioned_outer as outer
    from h3_flow_regenerate.geometry import pack_streams
    from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FlowBinding
    from h3_flow_regenerate.source_prefix_carry import SourcePrefixCarry

    provider = SimpleNamespace(
        api_version=1,
        kind="minimax_h3_learned_latent_upscaler",
        model_name="fixture",
        device="cpu",
        precision="fp32",
        offload_after_upscale=False,
        upscale_clean_video=lambda *a, **k: None,
    )
    progressive = ProgressiveTargetInputConfig(source_scale=0.70, transfer_mode="learned_3d", learned_upscaler=provider)
    owner = SourcePrefixCarry()
    model_options = {
        FLOW_BINDING_KEY: FlowBinding(metrics=H3FlowMetrics()),
        outer.PARTITIONED_PROGRESSIVE_KEY: progressive,
        SOURCE_PREFIX_CARRY_KEY: owner,
    }
    guider = SimpleNamespace(model_options=model_options)
    video, audio = torch.zeros(1, 24, 7, 8, 12), torch.zeros(1, 32, 2, 12)
    packed, shapes = pack_streams((video, audio))
    mask = torch.ones_like(packed) if mask_present else None
    marker = object()
    received = []

    def ordinary(*args, **kwargs):
        received.append((args, kwargs))
        assert args[0].class_obj.model_options is model_options
        return marker

    def forbidden(*args, **kwargs):
        pytest.fail("single-chunk request entered protected-prefix continuation")

    monkeypatch.setattr(outer, "flow_outer_wrapper_with_exact_mask", ordinary)
    monkeypatch.setattr(outer, "run_partitioned_progressive", forbidden)
    args = (
        SimpleNamespace(class_obj=guider),
        torch.ones_like(packed),
        packed,
        object(),
        torch.tensor([1.0, 0.0]),
        mask,
        object(),
        True,
        7,
    )
    assert outer.partitioned_outer_wrapper(*args, latent_shapes=shapes) is marker
    assert len(received) == 1
    assert all(a is b for a, b in zip(received[0][0], args, strict=True))
    assert received[0][1]["latent_shapes"] is shapes
    assert model_options[SOURCE_PREFIX_CARRY_KEY] is owner
