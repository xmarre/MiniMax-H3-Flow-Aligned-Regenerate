"""Regression for direction guidance undoing an exact-prefix handoff."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import h3_flow_regenerate.guidance as guidance
from h3_flow_regenerate.contracts import TrajectoryRun, TrajectorySample
from h3_flow_regenerate.geometry import geometry_from_video, pack_streams, resize_video, unpack_streams
from h3_flow_regenerate.guidance import ExactPrefixGuidanceGauge, GuidanceConfig, GuidanceState, apply_guidance
from h3_flow_regenerate.high_stage_boundary import high_boundary_contract
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_scheduler import (
    _apply_partitioned_exact_overlap_bridge,
    _apply_partitioned_suffix_dc_bridge,
)
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FLOW_STAGE_KEY, FlowBinding, flow_predict_wrapper


def _case(dtype=torch.float32):
    prefix_t = 3
    # A source scene with motion; the authoritative prefix also carries a
    # spatially varying representation residual and a nonzero channel mean.
    y, x = torch.meshgrid(torch.arange(8), torch.arange(10), indexing="ij")
    source = 1 + x / 20 + y / 30 + torch.arange(10)[:, None, None] / 40
    source = source[None, None].expand(1, 24, -1, -1, -1).to(dtype).clone()
    native = resize_video(source, 12, 14)
    exact = native[:, :, :prefix_t].clone()
    exact[:, :, -1] += 0.2 + torch.linspace(-0.1, 0.1, 14).to(dtype)
    _, corrected, _, _ = _apply_partitioned_exact_overlap_bridge(
        native.clone(), native, exact, sigma=0.8, weights=(1.0, 0.75, 0.5, 0.25)
    )
    corrected[:, :, :prefix_t] = exact
    samples = tuple(
        TrajectorySample(c, c, c, i, i, "handoff_probe", "actual", source.clone()) for i, c in enumerate((0.8, 0.2))
    )
    run = TrajectoryRun(
        1,
        "r",
        "s",
        "0",
        "sample_euler",
        "sched",
        geometry_from_video(source),
        (1, 32, 2, 8),
        "layout",
        "cond",
        "system_ram",
        samples,
        0,
        1,
        True,
    )
    return prefix_t, source, native, exact, corrected, run


def _dc_case(dtype=torch.float32):
    p, source, native, exact, _, run = _case(dtype)
    _, corrected, metrics = _apply_partitioned_suffix_dc_bridge(native.clone(), native, exact, sigma=0.8, enabled=True)
    corrected[:, :, :p] = exact
    return p, source, native, exact, corrected, run, metrics


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cutoff", [0.25, 1.0])
def test_dc_handoff_guidance_preserves_mean_without_reintroducing_rejected_spatial_residual(dtype, cutoff):
    p, source, native, exact, high, run, metrics = _dc_case(dtype)
    original_high, original_source = high.clone(), source.clone()
    config = GuidanceConfig(direction_weight=0.35, cutoff=cutoff, max_correction_rms_ratio=10.0)
    legacy = apply_guidance(high, run=run, coordinate=0.2, config=config, state=GuidanceState(), protected_prefix_t=p)
    assert (legacy[:, :, p] - high[:, :, p]).mean().item() == pytest.approx(-0.07, abs=2e-7)

    binding = FlowBinding(guidance=config)
    with high_boundary_contract(
        binding, exact, [tuple(high.shape)], measure=False, guidance_reference_dc_metrics=metrics
    ):
        gauge = binding.guidance_reference_gauge
        assert gauge is not None and gauge.weights == (1.0,) and gauge.spatial_mean_only
        # Retain channel means rather than the complete prefix/frame activation.
        assert gauge.anchor.shape == (1, 24, 1, 1, 1)
        assert gauge.anchor.untyped_storage().nbytes() == 24 * 4
        state = GuidanceState()
        guided = apply_guidance(
            high, run=run, coordinate=0.2, config=config, state=state, protected_prefix_t=p, reference_gauge=gauge
        )
        residual = gauge.residual(native, prefix_t=p)
        assert residual.shape == (1, 24, 1, 1, 1)
    torch.testing.assert_close(guided, high, atol=4e-7, rtol=0)
    assert torch.equal(guided[:, :, :p], exact)
    assert torch.equal(guided[:, :, p + 1 :], high[:, :, p + 1 :])
    assert torch.equal(high, original_high) and torch.equal(source, original_source)
    assert all(torch.equal(sample.video_x0, source) for sample in run.samples)
    assert state.last_reference_gauge_policy == "exact_prefix_guidance_reference_dc_v1"
    receipt = binding.metrics.events[-1].fields
    assert receipt["policy"] == state.last_reference_gauge_policy and receipt["support_tokens"] == 1
    assert receipt["spatial_mean_only"] is True
    assert gauge.anchor is None and binding.guidance_reference_gauge is None


@pytest.mark.parametrize("actual", [True, False])
def test_production_wrapper_applies_dc_reference_and_preserves_audio(actual):
    p, _, _, exact, high, run, metrics = _dc_case()
    audio = torch.randn(1, 32, 2, 8)
    packed, shapes = pack_streams((high, audio))
    binding = FlowBinding(guidance=GuidanceConfig(cutoff=1.0), active_guidance_run=run)
    guider = SimpleNamespace(
        model_options={FLOW_BINDING_KEY: binding}, inner_model=SimpleNamespace(latent_shapes=list(shapes))
    )

    class Executor:
        class_obj = guider

        def __call__(self, x, timestep, model_options, seed):
            return x.clone()

    options = {"transformer_options": {FLOW_STAGE_KEY: "high", "spectrum_h3_actual": actual}}
    with high_boundary_contract(binding, exact, shapes, measure=False, guidance_reference_dc_metrics=metrics):
        result = flow_predict_wrapper(Executor(), packed, torch.tensor([0.7]), model_options=options, seed=17)
    video_out, audio_out = unpack_streams(result, shapes)
    torch.testing.assert_close(video_out, high, atol=4e-7, rtol=0)
    assert torch.equal(audio_out, audio) and torch.equal(video_out[:, :, :p], exact)
    receipt = next(e.fields for e in binding.metrics.events if e.kind == "guidance")
    assert receipt["reference_gauge_used"] and receipt["actual"] is actual
    assert receipt["reference_gauge_policy"] == "exact_prefix_guidance_reference_dc_v1"


@pytest.mark.parametrize(
    "mode", ["direction", "direction+temporal", "direction+acceleration", "off", "downsample_consistency"]
)
@pytest.mark.parametrize("bridge_state", ["applied", "zero", "disabled", "registered", "wrong_prefix"])
def test_dc_reference_activation_requires_actual_compatible_handoff(mode, bridge_state):
    _, _, _, exact, high, _, metrics = _dc_case()
    metrics = dict(metrics)
    if bridge_state == "zero":
        metrics["suffix_dc_bridge_delta_rms"] = 0.0
    elif bridge_state == "disabled":
        metrics["suffix_dc_bridge_enabled"] = False
    elif bridge_state == "wrong_prefix":
        metrics["suffix_dc_bridge_prefix_t"] += 1
    binding = FlowBinding(guidance=GuidanceConfig(mode=mode))
    if bridge_state == "registered":
        binding.registered_guidance_reference = object()
    with high_boundary_contract(
        binding, exact, [tuple(high.shape)], measure=False, guidance_reference_dc_metrics=metrics
    ):
        expected = mode in {"direction", "direction+temporal", "direction+acceleration"} and bridge_state == "applied"
        assert (binding.guidance_reference_gauge is not None) is expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cutoff", [0.25, 1.0])
def test_guidance_preserves_an_already_reconciled_native_transition(dtype, cutoff):
    p, source, native, exact, high, run = _case(dtype)
    original_high, original_source = high.clone(), source.clone()
    config = GuidanceConfig(direction_weight=0.35, cutoff=cutoff, max_correction_rms_ratio=10.0)
    legacy = apply_guidance(high, run=run, coordinate=0.2, config=config, state=GuidanceState(), protected_prefix_t=p)
    native_transition = native[:, :, p] - native[:, :, p - 1]
    legacy_error = (legacy[:, :, p] - exact[:, :, -1] - native_transition).abs().max()
    assert legacy_error > 0.03

    state = GuidanceState()
    gauge = ExactPrefixGuidanceGauge(exact, (1.0, 0.75, 0.5, 0.25))
    guided = apply_guidance(
        high, run=run, coordinate=0.2, config=config, state=state, protected_prefix_t=p, reference_gauge=gauge
    )
    assert torch.allclose(guided[:, :, p] - exact[:, :, -1], native_transition, atol=5e-7, rtol=1e-6)
    assert torch.allclose(guided, high, atol=3e-7, rtol=1e-6)
    assert torch.equal(guided[:, :, :p], exact)
    assert torch.equal(guided[:, :, p + 4 :], high[:, :, p + 4 :])
    assert torch.equal(high, original_high) and torch.equal(source, original_source)
    assert all(torch.equal(sample.video_x0, source) for sample in run.samples)
    assert state.last_reference_gauge_used and gauge.calls == 1


@pytest.mark.parametrize("dc_only", [False, True])
def test_reference_gauge_leaves_source_temporal_correspondence_unchanged(monkeypatch, dc_only):
    p, source, _, exact, high, run = _case()
    observed = []

    def correspondence(reference, **kwargs):
        observed.append((reference.clone(), kwargs["prefix_t"]))
        return None, False

    monkeypatch.setattr(guidance, "_temporal_correspondence", correspondence)
    apply_guidance(
        high,
        run=run,
        coordinate=0.2,
        config=GuidanceConfig(mode="direction+temporal"),
        state=GuidanceState(),
        protected_prefix_t=p,
        reference_gauge=ExactPrefixGuidanceGauge(
            exact, (1.0,) if dc_only else (1.0, 0.75, 0.5, 0.25), spatial_mean_only=dc_only
        ),
    )
    assert len(observed) == 1
    assert torch.equal(observed[0][0], source) and observed[0][1] == p


@pytest.mark.parametrize("dc_only", [False, True])
@pytest.mark.parametrize("moving", [False, True])
def test_temporal_transport_respects_the_reconciled_representation(dc_only, moving):
    p, _, _, _, _, run = _case()
    generator = torch.Generator().manual_seed(713)
    texture = torch.randn(1, 24, 1, 8, 10, generator=generator)
    source = torch.cat([torch.roll(texture, i if moving else 0, dims=-1) for i in range(10)], dim=2)
    original_source = source.clone()
    native = resize_video(source, 12, 14)
    exact = native[:, :, :p].clone()
    y, x = torch.meshgrid(torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 14), indexing="ij")
    exact[:, :, -1] += 0.3 + 0.12 * x + 0.08 * y
    weights = (1.0,) if dc_only else (1.0, 0.75, 0.5, 0.25)
    if dc_only:
        _, high, _ = _apply_partitioned_suffix_dc_bridge(native.clone(), native, exact, sigma=0.8, enabled=True)
    else:
        _, high, _, _ = _apply_partitioned_exact_overlap_bridge(
            native.clone(), native, exact, sigma=0.8, weights=weights
        )
    high[:, :, :p] = exact
    original_high = high.clone()
    baseline = native.clone()
    baseline[:, :, :p] = exact
    run = replace(
        run,
        samples=tuple(
            TrajectorySample(c, c, c, i, i, "handoff_probe", "actual", source.clone()) for i, c in enumerate((0.8, 0.2))
        ),
    )
    config = GuidanceConfig(
        mode="direction+temporal",
        # Isolate temporal transport in the moving case. On the stationary
        # scene both direction and temporal guidance must preserve the handoff.
        direction_weight=0.0 if moving else 0.35,
        temporal_weight=0.3,
        temporal_min_similarity=0.1,
        temporal_min_margin=0.001,
        temporal_search_radius=2,
        cutoff=1.0,
        max_correction_rms_ratio=10.0,
    )
    state = GuidanceState()
    baseline_state = GuidanceState()
    gauge = ExactPrefixGuidanceGauge(exact, weights, spatial_mean_only=dc_only)
    cached = None
    for coordinate in (0.2, 0.1):
        baseline_result = apply_guidance(
            baseline, run=run, coordinate=coordinate, config=config, state=baseline_state, protected_prefix_t=p
        )
        result = apply_guidance(
            high,
            run=run,
            coordinate=coordinate,
            config=config,
            state=state,
            protected_prefix_t=p,
            reference_gauge=gauge,
        )
        # A representation change cannot create a different temporal correction.
        assert torch.allclose(
            result[:, :, p:] - high[:, :, p:], baseline_result[:, :, p:] - baseline[:, :, p:], atol=4e-7
        )
        if not moving:
            assert torch.allclose(result, high, atol=2e-6, rtol=1e-6)
        assert torch.equal(result[:, :, :p], exact)
        assert state.last_temporal_valid_fraction > 0.0
        assert state.last_temporal_reference_gauge_used
        assert torch.equal(state.temporal_cache.backward_flow, baseline_state.temporal_cache.backward_flow)
        assert torch.equal(state.temporal_cache.backward_confidence, baseline_state.temporal_cache.backward_confidence)
        if cached is not None:
            assert state.temporal_cache is cached and state.last_temporal_cache_hit
        cached = state.temporal_cache
    assert torch.equal(high, original_high) and torch.equal(source, original_source)
    assert all(torch.equal(sample.video_x0, source) for sample in run.samples)


@pytest.mark.parametrize("actual", [True, False])
def test_production_prediction_wrapper_uses_the_gauge_and_preserves_audio(actual):
    p, _, native, exact, high, run = _case()
    audio = torch.randn(1, 32, 2, 8)
    packed, shapes = pack_streams((high, audio))
    binding = FlowBinding(guidance=GuidanceConfig(cutoff=1.0), active_guidance_run=run)
    guider = SimpleNamespace(
        model_options={FLOW_BINDING_KEY: binding}, inner_model=SimpleNamespace(latent_shapes=list(shapes))
    )

    class Executor:
        class_obj = guider

        def __call__(self, x, timestep, model_options, seed):
            return x.clone()

    options = {"transformer_options": {FLOW_STAGE_KEY: "high", "spectrum_h3_actual": actual}}
    with high_boundary_contract(
        binding, exact, shapes, measure=False, guidance_reference_gauge_weights=(1.0, 0.75, 0.5, 0.25)
    ):
        result = flow_predict_wrapper(Executor(), packed, torch.tensor([0.7]), model_options=options, seed=17)
    video_out, audio_out = unpack_streams(result, shapes)
    assert torch.equal(audio_out, audio)
    assert torch.equal(video_out[:, :, :p], exact)
    assert torch.allclose(video_out[:, :, p] - exact[:, :, -1], native[:, :, p] - native[:, :, p - 1], atol=5e-7)
    receipt = next(e.fields for e in binding.metrics.events if e.kind == "guidance")
    assert receipt["reference_gauge_used"] and receipt["actual"] is actual
    assert receipt["reference_gauge_policy"] == "exact_prefix_guidance_reference_coupled_v1"
    assert binding.guidance_reference_gauge is None


@pytest.mark.parametrize("dc_only", [False, True])
def test_acceleration_uses_the_same_reconciled_reference_velocity(dc_only):
    if dc_only:
        p, source, _, exact, high, run, _ = _dc_case()
    else:
        p, source, _, exact, high, run = _case()
    samples = tuple(
        TrajectorySample(c, c, c, i, i, "corrected", "actual", source.clone()) for i, c in enumerate((0.8, 0.5, 0.2))
    )
    run = replace(run, samples=samples)
    state = GuidanceState()
    gauge = ExactPrefixGuidanceGauge(exact, (1.0,) if dc_only else (1.0, 0.75, 0.5, 0.25), spatial_mean_only=dc_only)
    for coordinate in (0.2, 0.1):
        result = apply_guidance(
            high,
            run=run,
            coordinate=coordinate,
            config=GuidanceConfig(mode="direction+acceleration", acceleration_weight=0.2, cutoff=1.0),
            state=state,
            high_state=high + 0.7,
            sigma=0.7,
            protected_prefix_t=p,
            reference_gauge=gauge,
        )
        assert torch.allclose(result, high, atol=5e-7, rtol=1e-6)
    assert state.last_acceleration_applied
    assert torch.allclose(state.current_reference_velocity[:, :, p:], torch.ones_like(high[:, :, p:]), atol=5e-7)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("dc_only", [False, True])
def test_high_contract_owns_one_frame_and_cleans_recursive_clones(failure, dc_only):
    p, _, _, exact, _, _, metrics = _dc_case()
    binding = FlowBinding(guidance=GuidanceConfig())
    shapes = [(1, 24, 10, 12, 14), (1, 32, 2, 8)]
    try:
        with high_boundary_contract(
            binding,
            exact,
            shapes,
            measure=False,
            guidance_reference_gauge_weights=None if dc_only else (1.0, 0.75, 0.5, 0.25),
            guidance_reference_dc_metrics=metrics if dc_only else None,
        ):
            owner = binding.guidance_reference_gauge
            clone = copy.deepcopy({"transformer_options": {"owner": owner}})["transformer_options"]["owner"]
            assert clone is owner
            assert owner.anchor.shape == ((1, 24, 1, 1, 1) if dc_only else (1, 24, 1, 12, 14))
            assert owner.anchor.untyped_storage().nbytes() == owner.anchor.numel() * owner.anchor.element_size()
            assert binding.guidance_protected_prefix_t == p
            with (
                pytest.raises(RuntimeError, match="nested high-stage"),
                high_boundary_contract(binding, exact, shapes, measure=False),
            ):
                pass
            if failure:
                raise RuntimeError("sampler failure")
    except RuntimeError as exc:
        assert failure and str(exc) == "sampler failure"
    assert binding.guidance_reference_gauge is None and binding.guidance_protected_prefix_t == 0
    assert owner.anchor is None
    with pytest.raises(RuntimeError, match="outside its high lifetime"):
        clone.residual(torch.zeros(1, 24, 10, 12, 14), prefix_t=p)
    receipt = binding.metrics.events[-1].fields
    assert receipt["support_tokens"] == (1 if dc_only else 4) and receipt["spatial_warp_applied"] is False


@pytest.mark.parametrize("dc_only", [False, True])
def test_high_contract_releases_anchor_before_receipt_failure(dc_only):
    _, _, _, exact, _, _, metrics = _dc_case()

    class FailedMetrics(H3FlowMetrics):
        def event(self, *args, **kwargs):
            raise RuntimeError("receipt failure")

    binding = FlowBinding(guidance=GuidanceConfig(), metrics=FailedMetrics())
    with (
        pytest.raises(RuntimeError, match="receipt failure"),
        high_boundary_contract(
            binding,
            exact,
            [(1, 24, 10, 12, 14)],
            measure=False,
            guidance_reference_gauge_weights=None if dc_only else (1.0, 0.75, 0.5, 0.25),
            guidance_reference_dc_metrics=metrics if dc_only else None,
        ),
    ):
        owner = binding.guidance_reference_gauge
    assert owner.anchor is None and binding.guidance_reference_gauge is None
    assert binding.guidance_protected_prefix_t == 0


def test_reference_gauge_rejects_wrong_ownership_and_releases_short_support():
    p, _, native, exact, _, _ = _case()
    gauge = ExactPrefixGuidanceGauge(exact, (1.0, 0.75, 0.5, 0.25))
    with pytest.raises(RuntimeError, match="prefix ownership"):
        gauge.residual(native, prefix_t=p - 1)
    with pytest.raises(RuntimeError, match="target geometry"):
        gauge.residual(native[..., :-1], prefix_t=p)
    short = native[:, :, : p + 1].clone()
    residual = gauge.residual(short, prefix_t=p)
    result = torch.zeros_like(short)
    gauge.add_to(result, residual)
    assert torch.equal(result[:, :, p:], residual)
    assert torch.count_nonzero(result[:, :, :p]) == 0
    gauge.close()


def test_inactive_high_contract_retains_baseline_binding_surface():
    binding = SimpleNamespace(
        guidance_protected_prefix_t=0,
        high_boundary_trace=None,
        high_boundary_anchor=None,
        high_prediction_bridge=None,
        metrics=H3FlowMetrics(),
    )
    with high_boundary_contract(binding, torch.ones(1, 24, 3, 8, 8), [], measure=False):
        assert not hasattr(binding, "guidance_reference_gauge")
    assert not hasattr(binding, "guidance_reference_gauge")
