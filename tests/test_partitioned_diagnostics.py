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
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_OPTIONS,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_OPTIONS,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AV_HANDOFF_SOURCE_OPTIONS,
    PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_OPTIONS,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS,
    PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    PartitionedAudioModelTimestepContext,
    apply_partitioned_diagnostic_controls,
    normalize_audio_handoff_source,
    normalize_av_handoff_source,
    normalize_guidance_trajectory_source,
    normalize_low_probe_execution_source,
    normalize_prefix_transformer_context,
    resolve_partitioned_audio_guided_overlap_mode,
    resolve_partitioned_audio_guided_overlap_ticks,
)
from h3_flow_regenerate.partitioned_node import (
    NODE_DISPLAY_NAME_MAPPINGS,
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
    _verify_prefix_transformer_context_diagnostic,
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


def test_partitioned_production_node_exposes_advanced_controls_without_changing_compatibility_node():
    ordinary = H3PartitionedExactPrefixHandoff.INPUT_TYPES()["required"]
    diagnostic = H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]

    assert "vdn_linear_diagnostic" not in ordinary
    assert "audio_guided_overlap_ticks" not in ordinary
    assert "audio_guided_overlap_mode" not in ordinary
    assert "prefix_transformer_context" not in ordinary
    assert "audio_position_domain" not in ordinary
    assert "audio_handoff_source" not in ordinary
    assert "av_handoff_source" not in ordinary
    assert "guidance_trajectory_source" not in ordinary
    assert "low_probe_execution_source" not in ordinary

    assert diagnostic["vdn_linear_diagnostic"][0] == [
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
    ]
    assert diagnostic["vdn_linear_diagnostic"][1]["default"] == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL
    assert diagnostic["audio_guided_overlap_mode"][0] == [
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    ]
    assert (
        diagnostic["audio_guided_overlap_mode"][1]["default"]
        == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP
    )
    assert diagnostic["audio_guided_overlap_ticks"][1]["default"] == 4
    assert diagnostic["audio_guided_overlap_ticks"][1]["min"] == 0
    assert diagnostic["audio_guided_overlap_ticks"][1]["max"] == 16
    assert diagnostic["prefix_transformer_context"][0] == list(PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_OPTIONS)
    assert diagnostic["prefix_transformer_context"][1]["default"] == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT
    assert diagnostic["audio_position_domain"][0] == list(PARTITIONED_AUDIO_POSITION_DOMAIN_OPTIONS)
    assert diagnostic["audio_position_domain"][1]["default"] == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE
    assert diagnostic["audio_handoff_source"][0] == list(PARTITIONED_AUDIO_HANDOFF_SOURCE_OPTIONS)
    assert diagnostic["audio_handoff_source"][1]["default"] == PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN
    assert diagnostic["av_handoff_source"][0] == list(PARTITIONED_AV_HANDOFF_SOURCE_OPTIONS)
    assert diagnostic["av_handoff_source"][1]["default"] == PARTITIONED_AV_HANDOFF_SOURCE_MAIN
    assert diagnostic["guidance_trajectory_source"][0] == list(PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_OPTIONS)
    assert diagnostic["guidance_trajectory_source"][1]["default"] == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN
    assert diagnostic["low_probe_execution_source"][0] == list(PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_OPTIONS)
    assert diagnostic["low_probe_execution_source"][1]["default"] == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY
    assert diagnostic["source_mode"][1]["default"] == "scale"
    assert diagnostic["source_scale"][1]["default"] == 0.70
    assert diagnostic["source_width"][1]["default"] == 864
    assert diagnostic["source_height"][1]["default"] == 640
    assert diagnostic["handoff_coordinate"][1]["default"] == 0.35
    assert diagnostic["handoff_selection"][1]["default"] == "fixed"
    assert diagnostic["guidance_mode"][1]["default"] == "direction+temporal"
    assert diagnostic["direction_weight"][1]["default"] == 0.25
    assert diagnostic["acceleration_weight"][1]["default"] == 0.25
    assert diagnostic["consistency_weight"][1]["default"] == 0.25
    assert diagnostic["low_frequency_cutoff"][1]["default"] == 0.25
    assert diagnostic["temporal_weight"][1]["default"] == 0.20
    assert H3PartitionedExactPrefixDiagnosticHandoff.CATEGORY == "MiniMax H3/flow regenerate"
    assert (
        NODE_DISPLAY_NAME_MAPPINGS["H3PartitionedExactPrefixDiagnosticHandoff"]
        == "MiniMax H3 Partitioned Exact-Prefix Handoff"
    )
    assert "[Diagnostic]" not in NODE_DISPLAY_NAME_MAPPINGS["H3PartitionedExactPrefixDiagnosticHandoff"]

    keys = list(diagnostic)
    assert keys.index("audio_guided_overlap_ticks") < keys.index("audio_guided_overlap_mode")
    assert keys.index("audio_guided_overlap_mode") < keys.index("prefix_transformer_context")
    assert keys.index("prefix_transformer_context") < keys.index("audio_position_domain")
    assert keys.index("audio_position_domain") < keys.index("audio_handoff_source")
    assert keys.index("audio_handoff_source") < keys.index("av_handoff_source")
    assert keys.index("av_handoff_source") < keys.index("guidance_trajectory_source")
    assert keys.index("guidance_trajectory_source") < keys.index("low_probe_execution_source")


def test_apply_partitioned_diagnostic_controls_is_model_local_and_preserves_existing_transformer_options():
    model = SimpleNamespace(model_options={"transformer_options": {"keep": "value"}})
    metrics = _Metrics()

    returned_model, returned_metrics = apply_partitioned_diagnostic_controls(
        model,
        metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        audio_guided_overlap_ticks=0,
        audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
        prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    )

    assert returned_model is model
    assert returned_metrics is metrics
    assert model.model_options["transformer_options"]["keep"] == "value"
    assert (
        model.model_options["transformer_options"][PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY]
        == PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS
    )
    assert (
        model.model_options["transformer_options"]["h3_flow_partitioned_prefix_transformer_context_v1"]
        == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
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
                "prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
                "model_local": True,
                "native_vdn_unchanged": True,
                "production_default_changed": False,
            },
        )
    ]


def test_prefix_transformer_context_normalization_is_bounded():
    assert (
        normalize_prefix_transformer_context(PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT)
        == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT
    )
    assert (
        normalize_prefix_transformer_context(PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE)
        == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
    )
    with pytest.raises(ValueError, match="prefix transformer context"):
        normalize_prefix_transformer_context("invented")


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
    exact = torch.zeros(1, 1, 2, 8)
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


def test_audio_timestep_context_can_restore_exact_labels_over_fractional_sampler_mask():
    metrics = _Metrics()
    runtime = torch.ones(1, 1, 2, 8)
    runtime[..., 2:6] = torch.tensor([0.2, 0.4, 0.6, 0.8]).view(1, 1, 1, 4)
    exact = torch.ones_like(runtime)
    exact[..., :6] = 0
    context = PartitionedAudioModelTimestepContext(
        audio_mask=exact,
        metrics=metrics,
        ticks=16,
        audio_prefix_ticks=6,
        mask_kind="exact_authoritative",
    )
    kwargs = {"audio_denoise_mask": runtime}
    forwarded = _audio_model_timestep_kwargs(
        {PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY: context},
        kwargs,
    )
    assert torch.equal(kwargs["audio_denoise_mask"], runtime)
    assert torch.equal(forwarded["audio_denoise_mask"], exact)
    assert context.mask_kind == "exact_authoritative"
    assert context.calls == 1


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
    with pytest.raises(PartitionedPreflightUnsupported, match="diagnostic capability"):
        _validate_partitioned_vdn_compat(
            patcher,
            required_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
        )
    current._vdn_partitioned_linear_diagnostic_modes = (
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    )
    _validate_partitioned_vdn_compat(
        patcher,
        required_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
    )
    with pytest.raises(PartitionedPreflightUnsupported, match="diagnostic capability"):
        _validate_partitioned_vdn_compat(
            patcher,
            required_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
        )
    current._vdn_partitioned_linear_diagnostic_modes += (PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,)
    _validate_partitioned_vdn_compat(
        patcher,
        required_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
    )


def test_source_carrier_transformer_verification_fails_closed_then_reports_counts():
    metrics = _Metrics()
    metrics.increment("partitioned_source_carrier_uniform_transformer_calls", 2)
    metrics.increment("partitioned_source_carrier_uniform_prefix_frames", 24)

    with pytest.raises(RuntimeError, match="no verified low/probe execution"):
        _verify_prefix_transformer_context_diagnostic(
            metrics,
            PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
            calls_before=2,
            prefix_frames_before=24,
        )

    metrics.increment("partitioned_source_carrier_uniform_transformer_calls", 3)
    metrics.increment("partitioned_source_carrier_uniform_prefix_frames", 36)
    _verify_prefix_transformer_context_diagnostic(
        metrics,
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
        calls_before=2,
        prefix_frames_before=24,
    )
    kind, fields = metrics.events[-1]
    assert kind == "partitioned_prefix_transformer_context_verified"
    assert fields["mode"] == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
    assert fields["transformer_calls"] == 3
    assert fields["prefix_frames"] == 36
    assert fields["heterogeneous_partition_contract_published"] is False
    assert fields["fail_closed"] is True


def test_vdn_bypass_verification_fails_closed_when_no_bypass_executed():
    metrics = _Metrics()
    metrics.increment("partitioned_vdn_linear_bypass_calls", 2)
    metrics.increment("partitioned_vdn_linear_bypass_video_rows", 50)
    with pytest.raises(RuntimeError, match="zero bypass calls"):
        _verify_partitioned_vdn_linear_diagnostic(
            metrics,
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
            bypass_calls_before=2,
            bypass_video_rows_before=50,
        )
    metrics.increment("partitioned_vdn_linear_bypass_calls", 3)
    metrics.increment("partitioned_vdn_linear_bypass_video_rows", 99)
    _verify_partitioned_vdn_linear_diagnostic(
        metrics,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS,
        bypass_calls_before=2,
        bypass_video_rows_before=50,
    )
    assert metrics.events[-1][0] == "partitioned_vdn_linear_diagnostic_verified"
    assert metrics.events[-1][1]["bypass_calls"] == 3
    assert metrics.events[-1][1]["bypass_video_rows"] == 99


def test_vdn_cross_grid_temporal_verification_fails_closed_then_reports_counts():
    metrics = _Metrics()
    metrics.increment("partitioned_vdn_cross_grid_temporal_suppression_calls", 4)
    metrics.increment("partitioned_vdn_cross_grid_temporal_suppressed_taps", 20)
    metrics.increment("partitioned_vdn_cross_grid_temporal_suppressed_rows", 200)
    with pytest.raises(RuntimeError, match="no verified cross-grid short-conv suppression"):
        _verify_partitioned_vdn_linear_diagnostic(
            metrics,
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
            bypass_calls_before=0,
            bypass_video_rows_before=0,
            suppression_calls_before=4,
            suppressed_taps_before=20,
            suppressed_rows_before=200,
        )

    metrics.increment("partitioned_vdn_cross_grid_temporal_suppression_calls", 3)
    metrics.increment("partitioned_vdn_cross_grid_temporal_suppressed_taps", 18)
    metrics.increment("partitioned_vdn_cross_grid_temporal_suppressed_rows", 144)
    _verify_partitioned_vdn_linear_diagnostic(
        metrics,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
        bypass_calls_before=0,
        bypass_video_rows_before=0,
        suppression_calls_before=4,
        suppressed_taps_before=20,
        suppressed_rows_before=200,
    )
    kind, fields = metrics.events[-1]
    assert kind == "partitioned_vdn_linear_diagnostic_verified"
    assert fields["cross_grid_temporal_suppression_calls"] == 3
    assert fields["suppressed_taps"] == 18
    assert fields["suppressed_rows"] == 144


def test_vdn_raw_token_measure_verification_fails_closed_then_reports_counts():
    metrics = _Metrics()
    metrics.increment("partitioned_vdn_raw_token_measure_calls", 5)
    metrics.increment("partitioned_vdn_raw_token_measure_prefix_frames", 60)
    with pytest.raises(RuntimeError, match="no verified raw-token measure execution"):
        _verify_partitioned_vdn_linear_diagnostic(
            metrics,
            PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
            bypass_calls_before=0,
            bypass_video_rows_before=0,
            raw_measure_calls_before=5,
            raw_measure_prefix_frames_before=60,
        )

    metrics.increment("partitioned_vdn_raw_token_measure_calls", 3)
    metrics.increment("partitioned_vdn_raw_token_measure_prefix_frames", 36)
    _verify_partitioned_vdn_linear_diagnostic(
        metrics,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
        bypass_calls_before=0,
        bypass_video_rows_before=0,
        raw_measure_calls_before=5,
        raw_measure_prefix_frames_before=60,
    )
    kind, fields = metrics.events[-1]
    assert kind == "partitioned_vdn_linear_diagnostic_verified"
    assert fields["raw_token_measure_calls"] == 3
    assert fields["raw_token_measure_prefix_frames"] == 36


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
    learned_upscaler = SimpleNamespace(
        api_version=1,
        kind="minimax_h3_learned_latent_upscaler",
        model_name="diagnostic-test-provider",
        device="cpu",
        inference_device="cpu",
        precision="fp32",
        offload_after_upscale=False,
        upscale_clean_video=lambda *args, **kwargs: None,
    )
    progressive = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=6,
        transfer_mode="learned_3d",
        learned_upscaler=learned_upscaler,
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
        exact_denoise_mask=None,
    ):
        del adapted, call_guider, call_binding, config, noise, sampler, sigmas, callback, disable_pbar, seed
        assert latent_shapes == shapes
        assert torch.equal(call_mask, exact_mask)
        assert torch.equal(exact_denoise_mask, exact_mask)
        context = transformer_options.get(PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY)
        assert isinstance(context, PartitionedAudioModelTimestepContext)
        exact_audio = unpack_streams(call_mask, latent_shapes)[1]
        exact_audio_cond = exact_audio.amax(dim=1, keepdim=True)
        forwarded = _audio_model_timestep_kwargs(
            transformer_options,
            {"audio_denoise_mask": exact_audio_cond},
        )
        assert torch.equal(exact_audio, audio_mask)
        assert tuple(context.audio_mask.shape) == tuple(exact_audio_cond.shape)
        assert context.audio_mask.shape[1] == 1
        assert not torch.equal(forwarded["audio_denoise_mask"], exact_audio_cond)
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
    context_events = [event for event in metrics.events if event.kind == "partitioned_audio_model_timestep_context"]
    assert len(context_events) == 1
    assert context_events[0].fields["override_calls"] == 1
    assert context_events[0].fields["sampler_mask_modified"] is False
    assert context_events[0].fields["exact_sampler_prefix_preserved"] is True
    assert context_events[0].fields["core_audio_velocity_mask_contract"] is True


def test_sampler_mask_outer_keeps_runtime_overlap_separate_from_exact_diagnostic_mask(monkeypatch):
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
        learned_upscaler=SimpleNamespace(
            api_version=1,
            kind="minimax_h3_learned_latent_upscaler",
            model_name="diagnostic-test-provider",
            device="cpu",
            inference_device="cpu",
            precision="fp32",
            offload_after_upscale=False,
            upscale_clean_video=lambda *args, **kwargs: None,
        ),
    )
    guider = SimpleNamespace(
        model_options={
            FLOW_BINDING_KEY: binding,
            PARTITIONED_PROGRESSIVE_KEY: progressive,
            PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY: 4,
            PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY: PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
            "transformer_options": {},
        }
    )

    class Executor:
        class_obj = guider

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
        exact_denoise_mask=None,
    ):
        del adapted, call_guider, call_binding, config, noise, sampler, sigmas, callback, disable_pbar, seed
        assert latent_shapes == shapes
        assert torch.equal(exact_denoise_mask, exact_mask)
        assert not torch.equal(call_mask, exact_mask)
        runtime_audio = unpack_streams(call_mask, latent_shapes)[1]
        assert bool(((runtime_audio > 0) & (runtime_audio < 1)).any().item())
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
    overlap_events = [event for event in metrics.events if event.kind == "audio_guided_overlap"]
    assert len(overlap_events) == 1
    assert overlap_events[0].fields["mode"] == PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER
    assert overlap_events[0].fields["sampler_mask_modified"] is True
    assert overlap_events[0].fields["sampler_exact_audio_prefix_preserved"] is False


def test_sampler_mask_exact_timestep_keeps_fractional_sampler_mask_but_exact_inner_labels(monkeypatch):
    monkeypatch.setattr(
        "h3_flow_regenerate.partitioned_outer._core_has_audio_velocity_mask_contract",
        lambda: True,
    )
    video = torch.randn(1, 24, 5, 8, 12)
    audio = torch.randn(1, 32, 2, 24)
    packed, shapes = pack_streams((video, audio))
    shapes = list(shapes)
    video_mask = torch.ones_like(video)
    video_mask[:, :, :2] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :20] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]

    metrics = H3FlowMetrics()
    binding = FlowBinding(metrics=metrics)
    progressive = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=6,
        transfer_mode="learned_3d",
        learned_upscaler=SimpleNamespace(
            api_version=1,
            kind="minimax_h3_learned_latent_upscaler",
            model_name="diagnostic-test-provider",
            device="cpu",
            inference_device="cpu",
            precision="fp32",
            offload_after_upscale=False,
            upscale_clean_video=lambda *args, **kwargs: None,
        ),
    )
    transformer_options = {}
    guider = SimpleNamespace(
        model_options={
            FLOW_BINDING_KEY: binding,
            PARTITIONED_PROGRESSIVE_KEY: progressive,
            PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY: 16,
            PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_KEY: PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
            "transformer_options": transformer_options,
        }
    )

    class Executor:
        class_obj = guider

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
        exact_denoise_mask=None,
    ):
        del adapted, call_guider, call_binding, config, noise, sampler, sigmas, callback, disable_pbar, seed
        assert latent_shapes == shapes
        assert torch.equal(exact_denoise_mask, exact_mask)
        assert not torch.equal(call_mask, exact_mask)
        runtime_audio = unpack_streams(call_mask, latent_shapes)[1].amax(dim=1, keepdim=True)
        assert bool(((runtime_audio > 0) & (runtime_audio < 1)).any().item())
        exact_audio = unpack_streams(exact_mask, latent_shapes)[1].amax(dim=1, keepdim=True)
        context = transformer_options.get(PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY)
        assert isinstance(context, PartitionedAudioModelTimestepContext)
        assert context.mask_kind == "exact_authoritative"
        forwarded = _audio_model_timestep_kwargs(
            transformer_options,
            {"audio_denoise_mask": runtime_audio},
        )
        assert torch.equal(forwarded["audio_denoise_mask"], exact_audio)
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
    overlap = [event for event in metrics.events if event.kind == "audio_guided_overlap"][-1]
    assert overlap.fields["sampler_mask_modified"] is True
    assert overlap.fields["model_timestep_mask_kind"] == "exact_authoritative"
    assert overlap.fields["model_timestep_mask_modified"] is False
    assert overlap.fields["inner_exact_audio_prefix_preserved"] is True
    assert overlap.fields["sampler_exact_audio_prefix_preserved"] is False
    context_event = [event for event in metrics.events if event.kind == "partitioned_audio_model_timestep_context"][-1]
    assert context_event.fields["mask_kind"] == "exact_authoritative"
    assert context_event.fields["sampler_mask_modified"] is True
    assert context_event.fields["exact_sampler_prefix_preserved"] is False
    assert context_event.fields["inner_exact_audio_prefix_preserved"] is True


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


def test_source_carrier_audio_position_control_is_opt_in_and_model_local():
    model = SimpleNamespace(model_options={"transformer_options": {"keep": "value"}})
    metrics = _Metrics()
    apply_partitioned_diagnostic_controls(
        model,
        metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        audio_guided_overlap_ticks=4,
        audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
        prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    )
    assert (
        model.model_options["transformer_options"]["h3_flow_partitioned_audio_position_domain_v1"]
        == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE
    )
    assert metrics.events[-1][1]["audio_position_domain"] == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE

    legacy = SimpleNamespace(model_options={"transformer_options": {"keep": "value"}})
    legacy_metrics = _Metrics()
    apply_partitioned_diagnostic_controls(
        legacy,
        legacy_metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        audio_guided_overlap_ticks=4,
        audio_guided_overlap_mode=PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER,
        prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    )
    assert "h3_flow_partitioned_audio_position_domain_v1" not in legacy.model_options["transformer_options"]
    assert "audio_position_domain" not in legacy_metrics.events[-1][1]


def test_audio_handoff_source_is_bounded_and_defaults_to_main_path():
    assert (
        normalize_audio_handoff_source(PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN) == PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN
    )
    assert (
        normalize_audio_handoff_source(PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW)
        == PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW
    )
    with pytest.raises(ValueError, match="audio handoff source"):
        normalize_audio_handoff_source("invented")


def test_av_handoff_source_is_bounded_and_defaults_to_main_path():
    assert normalize_av_handoff_source(PARTITIONED_AV_HANDOFF_SOURCE_MAIN) == PARTITIONED_AV_HANDOFF_SOURCE_MAIN
    assert normalize_av_handoff_source(PARTITIONED_AV_HANDOFF_SOURCE_SHADOW) == PARTITIONED_AV_HANDOFF_SOURCE_SHADOW
    with pytest.raises(ValueError, match="AV handoff source"):
        normalize_av_handoff_source("invented")


def test_guidance_trajectory_source_is_bounded_and_defaults_to_main_path():
    assert (
        normalize_guidance_trajectory_source(PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN)
        == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN
    )
    assert (
        normalize_guidance_trajectory_source(PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW)
        == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW
    )
    with pytest.raises(ValueError, match="guidance trajectory source"):
        normalize_guidance_trajectory_source("invented")


def test_guidance_trajectory_source_is_model_local_and_default_absent():
    default_model = SimpleNamespace(model_options={"transformer_options": {}})
    default_metrics = _Metrics()
    apply_partitioned_diagnostic_controls(
        default_model,
        default_metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        audio_guided_overlap_ticks=4,
    )
    assert PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY not in default_model.model_options["transformer_options"]

    shadow_model = SimpleNamespace(model_options={"transformer_options": {}})
    shadow_metrics = _Metrics()
    apply_partitioned_diagnostic_controls(
        shadow_model,
        shadow_metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        audio_guided_overlap_ticks=4,
        guidance_trajectory_source=PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
    )
    assert (
        shadow_model.model_options["transformer_options"][PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_KEY]
        == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW
    )
    assert shadow_metrics.events[-1][1]["guidance_trajectory_source"] == PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW


def test_low_probe_execution_source_is_bounded_and_model_local():
    assert (
        normalize_low_probe_execution_source(PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW)
        == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_MAIN_THEN_SHADOW
    )
    assert (
        normalize_low_probe_execution_source(PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY)
        == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY
    )
    assert (
        normalize_low_probe_execution_source(PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY)
        == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY
    )
    with pytest.raises(ValueError, match="low/probe execution source"):
        normalize_low_probe_execution_source("invented")

    default_model = SimpleNamespace(model_options={"transformer_options": {}})
    default_metrics = _Metrics()
    apply_partitioned_diagnostic_controls(
        default_model,
        default_metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        audio_guided_overlap_ticks=4,
    )
    assert PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY not in default_model.model_options["transformer_options"]

    candidate_model = SimpleNamespace(model_options={"transformer_options": {}})
    candidate_metrics = _Metrics()
    apply_partitioned_diagnostic_controls(
        candidate_model,
        candidate_metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        audio_guided_overlap_ticks=4,
        low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY,
    )
    assert (
        candidate_model.model_options["transformer_options"][PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY]
        == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY
    )
    assert (
        candidate_metrics.events[-1][1]["low_probe_execution_source"]
        == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_SOURCE_ONLY
    )

    exact_model = SimpleNamespace(model_options={"transformer_options": {}})
    exact_metrics = _Metrics()
    apply_partitioned_diagnostic_controls(
        exact_model,
        exact_metrics,
        vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        audio_guided_overlap_ticks=4,
        low_probe_execution_source=PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY,
    )
    assert (
        exact_model.model_options["transformer_options"][PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_KEY]
        == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY
    )
    exact_source = exact_metrics.events[-1][1]["low_probe_execution_source"]
    assert exact_source == PARTITIONED_LOW_PROBE_EXECUTION_SOURCE_EXACT_ONLY
