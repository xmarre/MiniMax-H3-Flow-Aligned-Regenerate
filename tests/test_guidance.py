import pytest
import torch

from h3_flow_regenerate.contracts import TrajectoryRun, TrajectorySample
from h3_flow_regenerate.geometry import geometry_from_video
from h3_flow_regenerate.guidance import (
    GuidanceConfig,
    GuidanceState,
    _local_correspondence,
    apply_guidance,
    conditional_renoise_alignment,
    low_frequency_projection,
    time_matched_reference,
)


def run(values=(1.0, 0.0), coords=(0.8, 0.2), provenance=("actual", "actual")):
    samples = tuple(
        TrajectorySample(c, c, c, i, i, "corrected", p, torch.full((1, 24, 1, 4, 4), v))
        for i, (v, c, p) in enumerate(zip(values, coords, provenance, strict=True))
    )
    return TrajectoryRun(
        1,
        "r",
        "s",
        "0",
        "sa",
        "sched",
        geometry_from_video(torch.zeros(1, 24, 1, 4, 4)),
        (1, 32, 2, 8),
        "layout",
        "cond",
        "system_ram",
        samples,
        0,
        1,
        True,
    )


def run_video(video, coords=(0.8, 0.2)):
    samples = tuple(TrajectorySample(c, c, c, i, i, "corrected", "actual", video.clone()) for i, c in enumerate(coords))
    return TrajectoryRun(
        1,
        "r",
        "s",
        "0",
        "sa",
        "sched",
        geometry_from_video(video),
        (video.shape[0], 32, 2, 8),
        "layout",
        "cond",
        "system_ram",
        samples,
        0,
        1,
        True,
    )


def test_guidance_config_rejects_nonfinite_correction_bound():
    with pytest.raises(ValueError, match="finite and positive"):
        GuidanceConfig(max_correction_rms_ratio=float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        GuidanceConfig(max_correction_rms_ratio=float("inf"))


def test_time_matching_interpolates_coordinate_not_step():
    ref = time_matched_reference(run(), 0.5)
    assert torch.allclose(ref, torch.full_like(ref, 0.5))


def test_forecast_only_trajectory_is_not_trustworthy():
    with pytest.raises(RuntimeError, match="exact anchors"):
        time_matched_reference(run(provenance=("forecast", "forecast")), 0.5)


def test_duplicate_pece_coordinate_prefers_corrected_anchor():
    values = [torch.full((1, 24, 1, 4, 4), value) for value in (1.0, 2.0, 3.0, 4.0)]
    samples = (
        TrajectorySample(0.8, 0.8, 0.7, 0, 0, "predicted", "actual", values[0]),
        TrajectorySample(0.5, 0.5, 0.4, 1, 1, "predicted", "actual", values[1]),
        TrajectorySample(0.5, 0.5, 0.4, 1, 2, "corrected", "actual", values[2]),
        TrajectorySample(0.2, 0.2, 0.1, 2, 3, "predicted", "actual", values[3]),
    )
    trajectory = TrajectoryRun(
        1,
        "r",
        "s",
        "0",
        "sa",
        "sched",
        geometry_from_video(torch.zeros(1, 24, 1, 4, 4)),
        (1, 32, 2, 8),
        "layout",
        "cond",
        "system_ram",
        samples,
        0,
        1,
        True,
    )
    assert torch.equal(time_matched_reference(trajectory, 0.5), values[2])


def test_zero_weight_is_exact_baseline_parity():
    high = torch.randn(1, 24, 1, 8, 8)
    config = GuidanceConfig(mode="direction", direction_weight=0.0)
    assert torch.equal(apply_guidance(high, run=run(), coordinate=0.5, config=config, state=GuidanceState()), high)


def test_direction_schedule_is_normalized_to_refine_start_coordinate():
    high = torch.full((1, 24, 1, 4, 4), 0.2)
    state = GuidanceState()
    config = GuidanceConfig(
        mode="direction",
        direction_weight=1.0,
        cutoff=1.0,
        max_correction_rms_ratio=10.0,
    )
    trajectory = run(values=(1.0, 0.0), coords=(0.8, 0.2))

    first = apply_guidance(high, run=trajectory, coordinate=0.5, config=config, state=state)
    assert torch.allclose(first, torch.full_like(first, 0.5))

    second = apply_guidance(high, run=trajectory, coordinate=0.25, config=config, state=state)
    reference = torch.full_like(second, (0.25 - 0.2) / (0.8 - 0.2))
    expected = high + 0.5 * (reference - high)
    assert torch.allclose(second, expected)


def test_guidance_state_records_bounded_correction_telemetry():
    high = torch.full((1, 24, 1, 4, 4), 0.2)
    state = GuidanceState()
    config = GuidanceConfig(
        mode="direction",
        direction_weight=10.0,
        cutoff=1.0,
        max_correction_rms_ratio=0.25,
    )
    apply_guidance(high, run=run(values=(1.0, 0.0), coords=(0.8, 0.2)), coordinate=0.5, config=config, state=state)
    assert state.last_schedule == pytest.approx(1.0)
    assert state.last_correction_rms is not None
    assert state.last_baseline_rms is not None
    assert state.last_correction_rms_ratio == pytest.approx(0.25, rel=1e-5)
    assert state.last_clamp_scale is not None
    assert 0.0 < state.last_clamp_scale < 1.0


def test_non_acceleration_guidance_does_not_retain_full_video_state():
    high = torch.randn(1, 24, 1, 8, 8)
    state = GuidanceState()
    apply_guidance(
        high,
        run=run(),
        coordinate=0.5,
        config=GuidanceConfig(mode="direction"),
        state=state,
    )
    assert state.current_coordinate is None
    assert state.current_high_velocity is None
    assert state.current_reference_velocity is None
    assert state.previous_coordinate is None
    assert state.previous_high_velocity is None
    assert state.previous_reference_velocity is None


def test_low_frequency_projection_does_not_change_constant():
    constant = torch.ones(1, 24, 1, 8, 8)
    assert torch.allclose(low_frequency_projection(constant, 0.25), constant)


def test_acceleration_requires_state_and_stays_bounded():
    high = torch.zeros(1, 24, 1, 8, 8)
    state = GuidanceState()
    config = GuidanceConfig(mode="direction+acceleration", acceleration_weight=0.2)
    trajectory = run(values=(1.0, 0.5, 0.0), coords=(0.8, 0.5, 0.2), provenance=("actual",) * 3)
    high_state = torch.ones_like(high)
    first = apply_guidance(
        high,
        run=trajectory,
        coordinate=0.7,
        config=config,
        state=state,
        high_state=high_state,
        sigma=0.9,
    )
    second = apply_guidance(
        first,
        run=trajectory,
        coordinate=0.4,
        config=config,
        state=state,
        high_state=high_state,
        sigma=0.7,
    )
    assert torch.isfinite(second).all()


def test_acceleration_requires_sampler_state_and_sigma():
    high = torch.zeros(1, 24, 1, 4, 4)
    trajectory = run(values=(1.0, 0.5, 0.0), coords=(0.8, 0.5, 0.2), provenance=("actual",) * 3)
    config = GuidanceConfig(mode="direction+acceleration", direction_weight=0.0, acceleration_weight=0.2)

    with pytest.raises(ValueError, match="sampler state"):
        apply_guidance(high, run=trajectory, coordinate=0.7, config=config, state=GuidanceState(), sigma=0.9)
    with pytest.raises(ValueError, match="sampler sigma"):
        apply_guidance(
            high,
            run=trajectory,
            coordinate=0.7,
            config=config,
            state=GuidanceState(),
            high_state=torch.ones_like(high),
        )


def test_acceleration_matches_hiflow_velocity_alignment():
    trajectory = run(values=(0.9, 0.5, 0.1), coords=(0.8, 0.5, 0.2), provenance=("actual",) * 3)
    config = GuidanceConfig(
        mode="direction+acceleration",
        direction_weight=0.0,
        acceleration_weight=0.4,
        cutoff=0.25,
        max_correction_rms_ratio=10.0,
    )
    state = GuidanceState()

    first_x0 = torch.full((1, 24, 1, 4, 4), 0.25)
    first_state = torch.full_like(first_x0, 0.8)
    first_sigma = 0.9
    first_coordinate = 0.7
    first_ref = time_matched_reference(trajectory, first_coordinate)
    first = apply_guidance(
        first_x0,
        run=trajectory,
        coordinate=first_coordinate,
        config=config,
        state=state,
        high_state=first_state,
        sigma=first_sigma,
    )
    assert torch.equal(first, first_x0)

    previous_high_velocity = (first_state - first) / first_sigma
    previous_reference_velocity = (first_state - first_ref) / first_sigma

    current_x0 = torch.full_like(first_x0, 0.15)
    current_state = torch.full_like(first_x0, 0.6)
    current_sigma = 0.7
    current_coordinate = 0.4
    current_ref = time_matched_reference(trajectory, current_coordinate)
    current_high_velocity = (current_state - current_x0) / current_sigma
    current_reference_velocity = (current_state - current_ref) / current_sigma
    schedule = current_coordinate / first_coordinate
    expected_velocity = current_high_velocity + schedule * config.acceleration_weight * (
        current_reference_velocity - previous_reference_velocity - current_high_velocity + previous_high_velocity
    )
    expected_x0 = current_state - current_sigma * expected_velocity

    actual = apply_guidance(
        current_x0,
        run=trajectory,
        coordinate=current_coordinate,
        config=config,
        state=state,
        high_state=current_state,
        sigma=current_sigma,
    )
    assert torch.allclose(actual, expected_x0)


def test_pece_corrector_reuses_previous_distinct_velocity_anchor():
    trajectory = run(values=(0.9, 0.5, 0.1), coords=(0.8, 0.5, 0.2), provenance=("actual",) * 3)
    config = GuidanceConfig(
        mode="direction+acceleration",
        direction_weight=0.0,
        acceleration_weight=0.4,
        max_correction_rms_ratio=10.0,
    )
    state = GuidanceState()

    first_x0 = torch.full((1, 24, 1, 4, 4), 0.25)
    first_state = torch.full_like(first_x0, 0.8)
    first_sigma = 0.9
    first_coordinate = 0.7
    first_ref = time_matched_reference(trajectory, first_coordinate)
    first = apply_guidance(
        first_x0,
        run=trajectory,
        coordinate=first_coordinate,
        config=config,
        state=state,
        high_state=first_state,
        sigma=first_sigma,
    )
    previous_high_velocity = (first_state - first) / first_sigma
    previous_reference_velocity = (first_state - first_ref) / first_sigma

    predictor_x0 = torch.full_like(first_x0, 0.15)
    predictor_state = torch.full_like(first_x0, 0.6)
    current_sigma = 0.7
    current_coordinate = 0.4
    current_ref = time_matched_reference(trajectory, current_coordinate)
    predictor_high_velocity = (predictor_state - predictor_x0) / current_sigma
    predictor_reference_velocity = (predictor_state - current_ref) / current_sigma
    schedule = current_coordinate / first_coordinate
    expected_predictor_velocity = predictor_high_velocity + schedule * config.acceleration_weight * (
        predictor_reference_velocity - previous_reference_velocity - predictor_high_velocity + previous_high_velocity
    )
    expected_predictor_x0 = predictor_state - current_sigma * expected_predictor_velocity
    predictor = apply_guidance(
        predictor_x0,
        run=trajectory,
        coordinate=current_coordinate,
        config=config,
        state=state,
        high_state=predictor_state,
        sigma=current_sigma,
    )
    assert torch.allclose(predictor, expected_predictor_x0)
    assert state.last_same_coordinate_refinement is False
    assert state.last_acceleration_applied is True

    corrector_x0 = torch.full_like(first_x0, 0.10)
    corrector_state = torch.full_like(first_x0, 0.55)
    corrector_high_velocity = (corrector_state - corrector_x0) / current_sigma
    corrector_reference_velocity = (corrector_state - current_ref) / current_sigma
    expected_corrector_velocity = corrector_high_velocity + schedule * config.acceleration_weight * (
        corrector_reference_velocity - previous_reference_velocity - corrector_high_velocity + previous_high_velocity
    )
    expected_corrector_x0 = corrector_state - current_sigma * expected_corrector_velocity
    corrector = apply_guidance(
        corrector_x0,
        run=trajectory,
        coordinate=current_coordinate,
        config=config,
        state=state,
        high_state=corrector_state,
        sigma=current_sigma,
    )
    assert torch.allclose(corrector, expected_corrector_x0)
    assert state.last_same_coordinate_refinement is True
    assert state.last_acceleration_applied is True

    next_x0 = torch.full_like(first_x0, 0.05)
    next_state = torch.full_like(first_x0, 0.4)
    next_sigma = 0.5
    next_coordinate = 0.2
    next_ref = time_matched_reference(trajectory, next_coordinate)
    next_high_velocity = (next_state - next_x0) / next_sigma
    next_reference_velocity = (next_state - next_ref) / next_sigma
    corrected_current_velocity = (corrector_state - corrector) / current_sigma
    corrected_reference_velocity = (corrector_state - current_ref) / current_sigma
    next_schedule = next_coordinate / first_coordinate
    expected_next_velocity = next_high_velocity + next_schedule * config.acceleration_weight * (
        next_reference_velocity - corrected_reference_velocity - next_high_velocity + corrected_current_velocity
    )
    expected_next_x0 = next_state - next_sigma * expected_next_velocity
    next_result = apply_guidance(
        next_x0,
        run=trajectory,
        coordinate=next_coordinate,
        config=config,
        state=state,
        high_state=next_state,
        sigma=next_sigma,
    )
    assert torch.allclose(next_result, expected_next_x0)


def test_acceleration_fails_closed_without_three_exact_anchors():
    high = torch.zeros(1, 24, 1, 8, 8)
    config = GuidanceConfig(mode="direction+acceleration", acceleration_weight=0.2)
    with pytest.raises(RuntimeError, match="three exact"):
        apply_guidance(high, run=run(), coordinate=0.5, config=config, state=GuidanceState())


def test_conditional_renoise_is_rectified_flow_law():
    x0 = torch.ones(1, 24, 1, 4, 4)
    noise = torch.zeros(1, 24, 1, 8, 8)
    state = conditional_renoise_alignment(x0, target_h=8, target_w=8, sigma=0.25, noise=noise)
    assert torch.allclose(state, torch.full_like(state, 0.75))


def test_temporal_guidance_is_exact_parity_when_high_matches_reference():
    torch.manual_seed(11)
    reference = torch.randn(1, 24, 3, 8, 8)
    trajectory = run_video(reference)
    high = reference.clone()
    state = GuidanceState()
    config = GuidanceConfig(
        mode="direction+temporal",
        direction_weight=0.0,
        temporal_weight=0.2,
        temporal_search_radius=2,
        temporal_min_similarity=0.1,
        temporal_min_margin=0.01,
        max_correction_rms_ratio=10.0,
    )

    result = apply_guidance(high, run=trajectory, coordinate=0.5, config=config, state=state)

    assert torch.allclose(result, high, atol=1e-5, rtol=1e-5)
    assert state.last_temporal_valid_fraction is not None
    assert state.last_temporal_valid_fraction > 0.0
    assert state.last_temporal_rms_ratio == pytest.approx(0.0, abs=1e-6)
    assert state.last_temporal_cache_hit is False


def test_temporal_guidance_reuses_same_coordinate_correspondence():
    torch.manual_seed(12)
    reference = torch.randn(1, 24, 3, 8, 8)
    trajectory = run_video(reference)
    high = reference.clone()
    state = GuidanceState()
    config = GuidanceConfig(
        mode="direction+temporal",
        direction_weight=0.0,
        temporal_weight=0.2,
        temporal_search_radius=2,
        temporal_min_similarity=0.1,
        temporal_min_margin=0.01,
        max_correction_rms_ratio=10.0,
    )

    apply_guidance(high, run=trajectory, coordinate=0.5, config=config, state=state)
    cached = state.temporal_cache
    assert cached is not None

    apply_guidance(high, run=trajectory, coordinate=0.5, config=config, state=state)
    assert state.temporal_cache is cached
    assert state.last_temporal_cache_hit is True

    apply_guidance(high, run=trajectory, coordinate=0.4, config=config, state=state)
    assert state.temporal_cache is not cached
    assert state.last_temporal_cache_hit is False


def test_temporal_guidance_falls_back_on_disoccluded_unmatched_content():
    torch.manual_seed(13)
    reference = torch.randn(1, 24, 2, 8, 8)
    reference[:, :, 1] = torch.randn_like(reference[:, :, 1])
    trajectory = run_video(reference)
    high = torch.randn_like(reference)
    state = GuidanceState()
    config = GuidanceConfig(
        mode="direction+temporal",
        direction_weight=0.0,
        temporal_weight=0.5,
        temporal_search_radius=1,
        temporal_min_similarity=0.9999,
        temporal_min_margin=0.5,
        temporal_cycle_tolerance=0.0,
        max_correction_rms_ratio=10.0,
    )

    result = apply_guidance(high, run=trajectory, coordinate=0.5, config=config, state=state)

    assert torch.equal(result, high)
    assert state.last_temporal_valid_fraction == pytest.approx(0.0)
    assert state.last_temporal_disocclusion_fraction == pytest.approx(1.0)
    assert state.last_temporal_confidence_mean == pytest.approx(0.0)


def test_temporal_guidance_recovers_known_adjacent_translation():
    torch.manual_seed(14)
    frame0 = torch.randn(1, 24, 1, 10, 10)
    frame1 = torch.zeros_like(frame0)
    frame1[:, :, :, :, 1:] = frame0[:, :, :, :, :-1]
    reference = torch.cat((frame0, frame1), dim=2)
    trajectory = run_video(reference)
    state = GuidanceState()
    config = GuidanceConfig(
        mode="direction+temporal",
        direction_weight=0.0,
        temporal_weight=0.2,
        temporal_search_radius=2,
        temporal_min_similarity=0.1,
        temporal_min_margin=0.01,
        temporal_cycle_tolerance=0.0,
        max_correction_rms_ratio=10.0,
    )

    apply_guidance(reference, run=trajectory, coordinate=0.5, config=config, state=state)

    cached = state.temporal_cache
    assert cached is not None
    interior_dx = cached.backward_flow[:, :, 0, 2:-2, 2:-2]
    interior_dy = cached.backward_flow[:, :, 1, 2:-2, 2:-2]
    assert torch.median(interior_dx).item() == pytest.approx(-1.0)
    assert torch.median(interior_dy).item() == pytest.approx(0.0)
    assert state.last_temporal_flow_magnitude_mean is not None
    assert state.last_temporal_flow_magnitude_mean > 0.5


def test_temporal_min_margin_is_a_hard_rejection_threshold():
    torch.manual_seed(15)
    reference = torch.randn(1, 24, 3, 8, 8)
    trajectory = run_video(reference)
    high = torch.randn_like(reference)
    state = GuidanceState()
    config = GuidanceConfig(
        mode="direction+temporal",
        direction_weight=0.0,
        temporal_weight=0.5,
        temporal_search_radius=2,
        temporal_min_similarity=-1.0,
        temporal_min_margin=10.0,
        temporal_cycle_tolerance=0.0,
        max_correction_rms_ratio=10.0,
    )

    result = apply_guidance(high, run=trajectory, coordinate=0.5, config=config, state=state)

    assert torch.equal(result, high)
    assert state.last_temporal_valid_fraction == pytest.approx(0.0)
    assert state.last_temporal_confidence_mean == pytest.approx(0.0)
    assert state.last_temporal_rms_ratio == pytest.approx(0.0)


def test_temporal_cache_keys_resolved_clamped_reference_coordinate():
    torch.manual_seed(16)
    reference = torch.randn(1, 24, 3, 8, 8)
    trajectory = run_video(reference, coords=(0.8, 0.4))
    high = reference.clone()
    state = GuidanceState()
    config = GuidanceConfig(
        mode="direction+temporal",
        direction_weight=0.0,
        temporal_weight=0.2,
        temporal_search_radius=2,
        temporal_min_similarity=0.1,
        temporal_min_margin=0.01,
        max_correction_rms_ratio=10.0,
    )

    apply_guidance(high, run=trajectory, coordinate=0.3, config=config, state=state)
    cached = state.temporal_cache
    assert cached is not None
    assert cached.coordinate == pytest.approx(0.4)
    assert state.last_temporal_reference_coordinate == pytest.approx(0.4)
    assert state.last_temporal_reference_clamped is True
    assert state.last_temporal_cache_hit is False

    apply_guidance(high, run=trajectory, coordinate=0.2, config=config, state=state)
    assert state.temporal_cache is cached
    assert state.last_temporal_reference_coordinate == pytest.approx(0.4)
    assert state.last_temporal_reference_clamped is True
    assert state.last_temporal_cache_hit is True


def test_temporal_default_requires_exact_reverse_cycle():
    assert GuidanceConfig().temporal_cycle_tolerance == pytest.approx(0.0)


def test_temporal_correspondence_handles_single_spatial_candidate():
    torch.manual_seed(17)
    query = torch.randn(1, 24, 1, 1)

    flow, confidence, similarity, margin = _local_correspondence(
        query,
        query.clone(),
        radius=1,
        min_similarity=0.1,
        min_margin=0.01,
    )

    assert torch.equal(flow, torch.zeros_like(flow))
    assert torch.isfinite(confidence).all()
    assert torch.isfinite(similarity).all()
    assert torch.isinf(margin).all()
    assert torch.all(confidence > 0)

