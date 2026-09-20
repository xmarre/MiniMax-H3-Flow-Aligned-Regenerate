from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.audio_guided_overlap import AUDIO_GUIDED_OVERLAP_ENV
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
    PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    PartitionedAudioModelTimestepContext,
    apply_partitioned_diagnostic_controls,
    resolve_partitioned_audio_guided_overlap_mode,
    resolve_partitioned_audio_guided_overlap_ticks,
)
from h3_flow_regenerate.partitioned_node import (
    H3PartitionedExactPrefixDiagnosticHandoff,
    H3PartitionedExactPrefixHandoff,
)
from h3_flow_regenerate.partitioned_outer import (
    _source_has_audio_velocity_mask_contract,
    partitioned_outer_wrapper,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PARTITIONED_PROGRESSIVE_KEY,
    PartitionedPreflightUnsupported,
    _validate_partitioned_vdn_compat,
    _verify_partitioned_vdn_linear_diagnostic,
)
from h3_flow_regenerate.partitioned_transformer import _audio_model_timestep_kwargs
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FlowBinding


class _Metrics:
    def __init__(self):
        self.events = []
        self._counters = {}

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def increment(self, name, value=1):
        self._counters[name] = self._counters.get(name, 0) + value

    @property
    def counters(self):
        return dict(self._counters)


def test_diagnostic_node_exposes_bounded_ab_controls_without_changing_ordinary_node():
    ordinary = H3PartitionedExactPrefixHandoff.INPUT_TYPES()["required"]
    diagnostic = H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]

    assert "vdn_linear_diagnostic" not in ordinary
    assert "audio_guided_overlap_ticks" not in ordinary
    assert "audio_guided_overlap_mode" not in ordinary

    assert diagnostic["vdn_linear_diagnostic"][0] == [
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    ]
    assert diagnostic["vdn_linear_diagnostic"][1]["default"] == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL
    assert diagnostic["audio_guided_overlap_mode"][0] == [
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    ]
    assert diagnostic["audio_guided_overlap_mode"][1]["default"] == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER
    assert diagnostic["audio_guided_overlap_ticks"][1]["default"] == 4
    assert diagnostic["audio_guided_overlap_ticks"][1]["min"] == 0
    assert diagnostic["audio_guided_overlap_ticks"][1]["max"] == 16
    keys = list(diagnostic)
    assert keys.index("audio_guided_overlap_ticks") < keys.index("audio_guided_overlap_mode")


def test_apply_partitioned_diagnostic_controls_is_model_local_and_preserves_existing_transformer_options():
    model = SimpleNamespace(model_options={"transformer_options": {"keep": "value"}})
    metrics = _Metrics()

    returned_model, returned_metrics = apply_partitioned_diagnostic_controls(
        model,
        metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        audio_guided_overlap_ticks=0,
        audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    )

    assert returned_model is model
    assert returned_metrics is metrics
    assert model.model_options["transformer_options"]["keep"] == "value"
    assert (
        model.model_options["transformer_options"][PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY]
        == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS
    )
    assert model.model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY] == 0
    assert (
        model.model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY]
        == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP
    )
    assert metrics.events == [
        (
            "partitioned_diagnostic_controls",
            {
                "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
                "audio_guided_overlap_ticks": 0,
                "audio_guided_overlap_mode": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
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
            audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        )


def test_audio_guided_overlap_mode_defaults_and_node_override():
    mode, source = resolve_partitioned_audio_guided_overlap_mode({})
    assert mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER
    assert source == "default_sampler_mask"
    mode, source = resolve_partitioned_audio_guided_overlap_mode(
        {PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY: PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP}
    )
    assert mode == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP
    assert source == "diagnostic_node"


def test_model_timestep_audio_context_changes_only_inner_forward_mask():
    metrics = _Metrics()
    exact = torch.zeros(1, 32, 2, 8)
    guided = exact.clone()
    guided[..., 2:6] = torch.tensor([0.2, 0.4, 0.6, 0.8]).view(1, 1, 1, 4)
    context = PartitionedAudioModelTimestepContext(
        audio_mask=guided,
        metrics=metrics,
        ticks=4,
        audio_prefix_ticks=6,
    )
    kwargs = {"audio_denoise_mask": exact}
    forwarded = _audio_model_timestep_kwargs(
        {PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY: context},
        kwargs,
    )
    assert forwarded is not kwargs
    assert forwarded["audio_denoise_mask"] is not exact
    assert torch.equal(forwarded["audio_denoise_mask"], guided)
    assert torch.count_nonzero(kwargs["audio_denoise_mask"]).item() == 0
    assert context.calls == 1
    assert metrics.counters["partitioned_audio_model_timestep_override_calls"] == 1


def test_vdn_bypass_preflight_rejects_stale_bridge_without_capability_api():
    stale = SimpleNamespace(_vdn_forward=True, _vdn_external_sequence_api=4)
    patcher = SimpleNamespace(object_patches={"diffusion_model.blocks.0.attn.forward": stale})
    _validate_partitioned_vdn_compat(
        patcher,
        required_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    )
    with pytest.raises(PartitionedPreflightUnsupported, match="diagnostic API"):
        _validate_partitioned_vdn_compat(
            patcher,
            required_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        )
    current = SimpleNamespace(
        _vdn_forward=True,
        _vdn_external_sequence_api=4,
        _vdn_partitioned_linear_diagnostic_api=1,
    )
    patcher.object_patches["diffusion_model.blocks.0.attn.forward"] = current
    _validate_partitioned_vdn_compat(
        patcher,
        required_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    )


def test_vdn_bypass_verification_fails_closed_when_no_bypass_executed():
    metrics = _Metrics()
    with pytest.raises(RuntimeError, match="zero bypass calls"):
        _verify_partitioned_vdn_linear_diagnostic(
            metrics,
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
            bypass_calls_before=0,
        )
    metrics.increment("partitioned_vdn_linear_bypass_calls", 3)
    metrics.increment("partitioned_vdn_linear_bypass_video_rows", 99)
    _verify_partitioned_vdn_linear_diagnostic(
        metrics,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        bypass_calls_before=0,
    )
    assert metrics.events[-1][0] == "partitioned_vdn_linear_diagnostic_verified"
    assert metrics.events[-1][1]["bypass_calls"] == 3



def test_model_timestep_only_outer_keeps_sampler_mask_exact_and_restores_context(monkeypatch):
    monkeypatch.setattr(
        "h3_flow_regenerate.partitioned_outer._core_has_audio_velocity_mask_contract",
        lambda: True,
    )
    video = torch.randn(1, 24, 5, 8, 12)
    audio = torch.randn(1, 32, 2, 12)
    packed, shapes = pack_streams((video, audio))
    shapes = list(shapes)
    video_mask = torch.ones_like(video)
    video_mask[:, :, :2] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :6] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]

    metrics = H3FlowMetrics()
    binding = FlowBinding(metrics=metrics)
    progressive = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=6,
        transfer_mode="learned_3d",
        learned_upscaler=object(),
    )
    transformer_options = {}
    guider = SimpleNamespace(
        model_options={
            FLOW_BINDING_KEY: binding,
            PARTITIONED_PROGRESSIVE_KEY: progressive,
            PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY: 4,
            PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY: PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
            "transformer_options": transformer_options,
        }
    )

    class Executor:
        class_obj = guider

    observed = {}

    def fake_partitioned(
        adapted,
        call_guider,
        call_binding,
        config,
        noise,
        latent_image,
        sampler,
        sigmas,
        call_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes,
    ):
        del adapted, call_guider, call_binding, config, noise, sampler, sigmas, callback, disable_pbar, seed
        assert latent_shapes == shapes
        assert torch.equal(call_mask, exact_mask)
        context = transformer_options.get(PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY)
        assert isinstance(context, PartitionedAudioModelTimestepContext)
        exact_audio = unpack_streams(call_mask, latent_shapes)[1]
        forwarded = _audio_model_timestep_kwargs(
            transformer_options,
            {"audio_denoise_mask": exact_audio},
        )
        assert torch.equal(exact_audio, audio_mask)
        assert not torch.equal(forwarded["audio_denoise_mask"], exact_audio)
        observed["inner_audio_mask"] = forwarded["audio_denoise_mask"].clone()
        return latent_image.clone()

    monkeypatch.setattr(
        "h3_flow_regenerate.partitioned_outer.run_partitioned_progressive",
        fake_partitioned,
    )
    result = partitioned_outer_wrapper(
        Executor(),
        torch.randn_like(packed),
        packed,
        SimpleNamespace(),
        torch.tensor([1.0, 0.0]),
        exact_mask,
        None,
        True,
        7,
        latent_shapes=shapes,
    )

    assert torch.equal(result, packed)
    assert PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY not in transformer_options
    assert "inner_audio_mask" in observed
    assert metrics.counters["partitioned_audio_model_timestep_override_calls"] == 1
    context_events = [
        event for event in metrics.events if event.kind == "partitioned_audio_model_timestep_context"
    ]
    assert len(context_events) == 1
    assert context_events[0].fields["override_calls"] == 1
    assert context_events[0].fields["sampler_mask_modified"] is False
    assert context_events[0].fields["exact_sampler_prefix_preserved"] is True
    assert context_events[0].fields["core_audio_velocity_mask_contract"] is True



def test_audio_model_timestep_mode_requires_post_wrapper_velocity_mask_contract():
    old_source = """
out = WrapperExecutor(...).execute(x, audio_denoise_mask=audio_denoise_mask)
return out
"""
    fixed_source = """
out = WrapperExecutor(...).execute(x, audio_denoise_mask=audio_denoise_mask)
if audio_denoise_mask is not None:
    out[1] = out[1] * audio_denoise_mask
return out
"""
    assert _source_has_audio_velocity_mask_contract(old_source) is False
    assert _source_has_audio_velocity_mask_contract(fixed_source) is True
