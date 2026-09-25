import pytest
import torch

from h3_flow_regenerate.frame_gauge import (
    DEFAULT_POLICY,
    FRAME_GAUGE_POLICY_VERSION,
    LEARNED_VIDEO_POLICY,
    estimate_paired_prefix_translation,
    translate_video_cells,
)


def _analytic_pair(
    *,
    dx: float,
    dy: float,
    frames: int = 4,
    height: int = 26,
    width: int = 26,
    per_frame_shift: tuple[tuple[float, float], ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    exact_frames = []
    learned_frames = []
    for frame in range(frames):
        exact_channels = []
        learned_channels = []
        frame_dx, frame_dy = per_frame_shift[frame] if per_frame_shift is not None else (dx, dy)
        for channel in range(24):
            phase = 0.11 * frame + 0.067 * channel

            def field(y, x, phase=phase):
                return (
                    torch.sin(0.31 * x + phase)
                    + 0.73 * torch.cos(0.27 * y - 0.5 * phase)
                    + 0.31 * torch.sin(0.17 * (x + y) + 1.7 * phase)
                    + 0.19 * torch.cos(0.13 * (x - 2.0 * y) - phase)
                )

            exact_channels.append(field(yy, xx))
            learned_channels.append(field(yy + frame_dy, xx + frame_dx))
        exact_frames.append(torch.stack(exact_channels))
        learned_frames.append(torch.stack(learned_channels))
    exact = torch.stack(exact_frames, dim=1).unsqueeze(0).float()
    learned = torch.stack(learned_frames, dim=1).unsqueeze(0).float()
    return learned, exact


def test_v2_video_fit_relaxation_does_not_weaken_default_guidance_policy():
    assert FRAME_GAUGE_POLICY_VERSION == "paired_prefix_rigid_v2"
    assert LEARNED_VIDEO_POLICY.min_rms_improvement == 0.0
    assert DEFAULT_POLICY.min_rms_improvement == 0.15
    assert LEARNED_VIDEO_POLICY.min_runner_margin == DEFAULT_POLICY.min_runner_margin == 0.05
    assert LEARNED_VIDEO_POLICY.require_global_runner_margin is False
    assert DEFAULT_POLICY.require_global_runner_margin is True


def test_exact_equality_is_bitwise_identity():
    torch.manual_seed(10)
    exact = torch.randn(1, 24, 4, 26, 26)
    estimate = estimate_paired_prefix_translation(exact, exact)
    assert estimate.identity
    assert estimate.reason == "already_aligned"
    assert estimate.dx == 0.0
    assert estimate.dy == 0.0


@pytest.mark.parametrize(
    ("dx", "dy"),
    [
        (0.5, -0.375),
        (-0.625, 0.4375),
    ],
)
def test_registration_recovers_independent_fractional_translation(dx, dy):
    learned, exact = _analytic_pair(dx=dx, dy=dy)
    rng_before = torch.random.get_rng_state().clone()
    estimate = estimate_paired_prefix_translation(learned, exact)
    rng_after = torch.random.get_rng_state()

    assert estimate.accepted, estimate.telemetry()
    assert torch.equal(rng_before, rng_after)
    assert estimate.dx == pytest.approx(dx, abs=0.125)
    assert estimate.dy == pytest.approx(dy, abs=0.125)
    assert estimate.metrics["validation_ncc"] >= 0.75
    assert estimate.metrics["rms_improvement"] >= 0.15
    assert estimate.metrics["runner_margin_ratio"] >= 0.05


@pytest.mark.parametrize(
    ("dx", "dy"),
    [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, -1.0),
    ],
)
def test_registration_recovers_integer_axis_and_combined_shifts(dx, dy):
    size = 40 if dx != 0.0 and dy != 0.0 else 26
    learned, exact = _analytic_pair(dx=dx, dy=dy, height=size, width=size)
    estimate = estimate_paired_prefix_translation(learned, exact)

    assert estimate.accepted, estimate.telemetry()
    assert estimate.dx == pytest.approx(dx, abs=0.125)
    assert estimate.dy == pytest.approx(dy, abs=0.125)


def test_registration_rejects_constant_and_overbound_fields():
    constant = torch.ones(1, 24, 4, 26, 26)
    ambiguous = estimate_paired_prefix_translation(
        constant + 0.1,
        constant,
    )
    assert ambiguous.rejected
    assert ambiguous.reason in {
        "insufficient_valid_channels",
        "ambiguous_low_energy",
        "ambiguous_low_gradient",
    }

    learned, exact = _analytic_pair(dx=2.5, dy=0.0)
    overbound = estimate_paired_prefix_translation(learned, exact)
    assert overbound.rejected
    assert overbound.reason in {
        "over_bound_shift",
        "search_saturated",
    }


def test_split_frame_translation_is_rejected_instead_of_lower_mad_acceptance():
    shifts = (
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.5, 0.0),
        (-0.5, 0.0),
    )
    learned, exact = _analytic_pair(
        dx=0.0,
        dy=0.0,
        per_frame_shift=shifts,
    )
    estimate = estimate_paired_prefix_translation(learned, exact)
    video_estimate = estimate_paired_prefix_translation(
        learned,
        exact,
        policy=LEARNED_VIDEO_POLICY,
    )
    assert estimate.rejected
    assert estimate.reason != "accepted"
    assert video_estimate.rejected
    assert video_estimate.reason != "accepted"


def test_translation_sign_border_prefix_and_input_ownership():
    video = torch.zeros(1, 24, 3, 7, 7)
    video[:, :, 0] = 3.0
    video[:, :, 1:, 3, 3] = 1.0
    video[:, :, 1:, :, 0] = 2.0
    before = video.clone()

    applied = translate_video_cells(
        video,
        dx=1.0,
        dy=0.0,
        start_frame=1,
        batch_frames=1,
    )

    assert torch.equal(video, before)
    assert torch.equal(applied.video[:, :, 0], before[:, :, 0])
    assert torch.equal(
        applied.video[:, :, 1:, 3, 4],
        torch.ones_like(applied.video[:, :, 1:, 3, 4]),
    )
    assert torch.equal(
        applied.video[:, :, 1:, :, 0],
        torch.full_like(applied.video[:, :, 1:, :, 0], 2.0),
    )
    assert applied.invalid_fraction == pytest.approx(1.0 / 7.0)
    assert applied.transformed_frames == 2


def test_zero_translation_is_exact_noop_and_does_not_consume_rng():
    torch.manual_seed(99)
    video = torch.randn(1, 24, 4, 8, 8)
    state_before = torch.random.get_rng_state().clone()
    applied = translate_video_cells(
        video,
        dx=0.0,
        dy=0.0,
        start_frame=2,
    )
    state_after = torch.random.get_rng_state()

    assert applied.video is video
    assert torch.equal(applied.video, video)
    assert torch.equal(state_before, state_after)
    assert applied.transformed_frames == 0


def test_nonfinite_registration_is_a_hard_error():
    learned, exact = _analytic_pair(dx=0.5, dy=0.0)
    learned[:, :, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        estimate_paired_prefix_translation(learned, exact)


def test_exact_equality_still_requires_minimum_temporal_support():
    exact = torch.randn(1, 24, 3, 26, 26)
    estimate = estimate_paired_prefix_translation(exact, exact)
    assert estimate.rejected
    assert estimate.reason == "insufficient_prefix_support"


def test_registration_is_invariant_to_per_channel_positive_gain_and_dc():
    learned, exact = _analytic_pair(dx=0.5, dy=-0.375)
    channel = torch.arange(24, dtype=learned.dtype).view(1, 24, 1, 1, 1)
    learned = learned * (0.8 + 0.015 * channel) + (-0.6 + 0.05 * channel)

    estimate = estimate_paired_prefix_translation(learned, exact)

    assert estimate.accepted, estimate.telemetry()
    assert estimate.dx == pytest.approx(0.5, abs=0.125)
    assert estimate.dy == pytest.approx(-0.375, abs=0.125)


def test_registration_rejects_local_nonrigid_spatial_conflict():
    learned_right, exact = _analytic_pair(dx=0.5, dy=0.0, height=48, width=48)
    learned_left, _ = _analytic_pair(dx=-0.5, dy=0.0, height=48, width=48)
    learned = learned_right.clone()
    learned[..., : learned.shape[-1] // 2] = learned_left[..., : learned.shape[-1] // 2]

    estimate = estimate_paired_prefix_translation(learned, exact)
    video_estimate = estimate_paired_prefix_translation(
        learned,
        exact,
        policy=LEARNED_VIDEO_POLICY,
    )

    assert estimate.rejected
    assert estimate.reason in {
        "regional_conflict",
        "regional_disagreement",
        "insufficient_regional_support",
        "ambiguous_runner_up",
        "insufficient_validation_improvement",
    }
    assert video_estimate.rejected
    assert video_estimate.reason != "accepted"


def test_registration_rejects_informative_patch_phase_disagreement():
    learned_global, exact = _analytic_pair(dx=0.5, dy=0.0)
    learned_phase, _ = _analytic_pair(dx=-3.0, dy=0.0)
    learned = learned_global.clone()
    learned[..., 0::2, 0::2] = learned_phase[..., 0::2, 0::2]

    estimate = estimate_paired_prefix_translation(learned, exact)
    video_estimate = estimate_paired_prefix_translation(
        learned,
        exact,
        policy=LEARNED_VIDEO_POLICY,
    )

    assert estimate.rejected, estimate.telemetry()
    assert estimate.reason in {
        "phase_dependent_disagreement",
        "phase_dependent_conflict",
        "regional_disagreement",
        "regional_conflict",
    }
    assert video_estimate.rejected, video_estimate.telemetry()
    assert video_estimate.reason != "accepted"


def test_last_holdout_disagreement_cannot_be_averaged_away():
    shifts = (
        (0.5, 0.0),
        (0.5, 0.0),
        (0.5, 0.0),
        (-0.5, 0.0),
    )
    learned, exact = _analytic_pair(
        dx=0.0,
        dy=0.0,
        per_frame_shift=shifts,
    )

    estimate = estimate_paired_prefix_translation(learned, exact)

    assert estimate.rejected
    assert estimate.reason != "accepted"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_translation_preserves_supported_dtype_and_bounded_batch_equivalence(dtype):
    video = torch.randn(1, 24, 5, 12, 14, dtype=dtype)
    before = video.clone()

    one = translate_video_cells(
        video,
        dx=0.5,
        dy=-0.25,
        start_frame=2,
        batch_frames=1,
    )
    four = translate_video_cells(
        video,
        dx=0.5,
        dy=-0.25,
        start_frame=2,
        batch_frames=4,
    )

    assert one.video.dtype == dtype
    assert four.video.dtype == dtype
    assert torch.equal(video, before)
    assert torch.equal(one.video[:, :, :2], before[:, :, :2])
    tolerance = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(one.video.float(), four.video.float(), atol=tolerance, rtol=tolerance)
    assert torch.isfinite(one.video).all()


def test_registration_rejects_repeated_stripes_with_multiple_equal_solutions():
    x = torch.arange(26, dtype=torch.float32)
    stripe = torch.cos(torch.pi * x).view(1, 1, 1, 1, 26)
    exact = stripe.expand(1, 24, 4, 26, 26).clone()
    learned = -exact

    estimate = estimate_paired_prefix_translation(learned, exact)
    video_estimate = estimate_paired_prefix_translation(
        learned,
        exact,
        policy=LEARNED_VIDEO_POLICY,
    )

    assert estimate.rejected
    assert estimate.reason == "ambiguous_runner_up"
    assert video_estimate.rejected
    assert video_estimate.reason != "accepted"


def test_registration_rejects_when_fewer_than_eight_channels_are_informative():
    learned, exact = _analytic_pair(dx=0.5, dy=0.0)
    learned[:, 7:] = 0.0
    exact[:, 7:] = 0.0

    estimate = estimate_paired_prefix_translation(learned, exact)

    assert estimate.rejected
    assert estimate.reason == "insufficient_valid_channels"


def test_sub_identity_bound_estimate_is_an_exact_noop():
    learned, exact = _analytic_pair(dx=0.03125, dy=-0.03125)
    estimate = estimate_paired_prefix_translation(learned, exact)

    assert estimate.identity, estimate.telemetry()
    assert estimate.dx == 0.0
    assert estimate.dy == 0.0
