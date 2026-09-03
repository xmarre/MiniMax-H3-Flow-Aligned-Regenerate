import pytest
import torch

from h3_flow_regenerate.contracts import TrajectoryRun, TrajectorySample
from h3_flow_regenerate.geometry import geometry_from_video
from h3_flow_regenerate.guidance import (
    GuidanceConfig,
    GuidanceState,
    apply_guidance,
    initialization_alignment,
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


def test_low_frequency_projection_does_not_change_constant():
    constant = torch.ones(1, 24, 1, 8, 8)
    assert torch.allclose(low_frequency_projection(constant, 0.25), constant)


def test_acceleration_requires_state_and_stays_bounded():
    high = torch.zeros(1, 24, 1, 8, 8)
    state = GuidanceState()
    config = GuidanceConfig(mode="direction+acceleration", acceleration_weight=0.2)
    trajectory = run(values=(1.0, 0.5, 0.0), coords=(0.8, 0.5, 0.2), provenance=("actual",) * 3)
    first = apply_guidance(high, run=trajectory, coordinate=0.7, config=config, state=state)
    second = apply_guidance(first, run=trajectory, coordinate=0.4, config=config, state=state)
    assert torch.isfinite(second).all()


def test_acceleration_fails_closed_without_three_exact_anchors():
    high = torch.zeros(1, 24, 1, 8, 8)
    config = GuidanceConfig(mode="direction+acceleration", acceleration_weight=0.2)
    with pytest.raises(RuntimeError, match="three exact"):
        apply_guidance(high, run=run(), coordinate=0.5, config=config, state=GuidanceState())


def test_initialization_is_rectified_flow_conditional_law():
    x0 = torch.ones(1, 24, 1, 4, 4)
    noise = torch.zeros(1, 24, 1, 8, 8)
    state = initialization_alignment(x0, target_h=8, target_w=8, sigma=0.25, noise=noise)
    assert torch.allclose(state, torch.full_like(state, 0.75))
