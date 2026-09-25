from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import h3_flow_regenerate.partitioned_scheduler as partitioned_scheduler
from h3_flow_regenerate.contracts import TrajectoryRun, TrajectorySample
from h3_flow_regenerate.frame_gauge import GUIDANCE_REFERENCE_POLICY, translate_video_cells
from h3_flow_regenerate.geometry import geometry_from_video
from h3_flow_regenerate.guidance import GuidanceConfig
from h3_flow_regenerate.partitioned_scheduler import (
    _frame_gauge_boundary_motion_check,
    _frame_gauge_clean_postprocess,
    _prepare_registered_guidance_reference,
)
from h3_flow_regenerate.sigma import H3_VIDEO_SHIFT, normalized_coordinate


def _field_video(*, dx: float, dy: float, frames: int = 6, height: int = 26, width: int = 26) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    output = []
    for frame in range(frames):
        channels = []
        for channel in range(24):
            phase = 0.09 * frame + 0.071 * channel
            y = yy + dy
            x = xx + dx
            channels.append(
                torch.sin(0.29 * x + phase)
                + 0.71 * torch.cos(0.23 * y - 0.4 * phase)
                + 0.27 * torch.sin(0.19 * (x + y) + 1.3 * phase)
                + 0.17 * torch.cos(0.11 * (x - 2.0 * y) - phase)
            )
        output.append(torch.stack(channels))
    return torch.stack(output, dim=1).unsqueeze(0).float()


def _rigid_textured_video(*, shift_x: int = 0, shift_y: int = 0, frames: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(12345)
    base = torch.randn(1, 24, 1, 26, 26, generator=generator)
    video = base.repeat(1, 1, frames, 1, 1)
    if shift_x or shift_y:
        video = torch.roll(video, shifts=(shift_y, shift_x), dims=(-2, -1))
    return video


def _run(reference: torch.Tensor, coordinate: float) -> TrajectoryRun:
    sample = TrajectorySample(
        coordinate,
        0.8,
        0.8,
        0,
        0,
        "handoff_probe",
        "actual",
        reference.clone(),
    )
    return TrajectoryRun(
        1,
        "frame-gauge-run",
        "session",
        "chunk",
        "sample_res_multistep",
        "schedule",
        geometry_from_video(reference),
        (1, 32, 2, 8),
        "layout",
        "conditioning",
        "system_ram",
        (sample,),
        0,
        1,
        True,
    )


def _schedule():
    high_sigmas = torch.tensor([0.8, 0.5, 0.0], dtype=torch.float32)
    split_coordinate = float(normalized_coordinate(0.8, video_shift=H3_VIDEO_SHIFT))
    return high_sigmas, split_coordinate


def test_frame_gauge_transaction_calibrates_video_and_guidance_independently():
    exact_full = _rigid_textured_video()
    learned = _rigid_textured_video(shift_x=-1)
    guidance_reference = _rigid_textured_video(shift_x=1)
    prefix_t = 4
    high_sigmas, split_coordinate = _schedule()
    run = _run(guidance_reference, split_coordinate)

    postprocess, registered, witnesses, transaction = _frame_gauge_clean_postprocess(
        learned,
        exact_prefix=exact_full[:, :, :prefix_t],
        guidance_run=run,
        guidance=GuidanceConfig(mode="direction"),
        target_h=26,
        target_w=26,
        prefix_t=prefix_t,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert transaction["result"] == "accepted", transaction
    assert transaction["video_registration"]["dx"] == pytest.approx(1.0, abs=0.125)
    assert transaction["video_registration"]["dy"] == pytest.approx(0.0, abs=0.125)
    assert transaction["guidance_registration"]["dx"] == pytest.approx(-1.0, abs=0.125)
    assert transaction["guidance_registration"]["dy"] == pytest.approx(0.0, abs=0.125)
    assert registered is not None
    assert transaction["boundary_motion"]["status"] == "accepted"
    assert registered.dx != pytest.approx(transaction["video_registration"]["dx"], abs=0.125)
    assert torch.equal(postprocess.clean_video[:, :, :prefix_t], learned[:, :, :prefix_t])
    assert set(witnesses) == {
        "learned_native",
        "paired_prefix_aligned_witness",
        "corrected_clean",
    }


def test_boundary_motion_gate_compact_witness_matches_full_translation():
    exact_full = _rigid_textured_video()
    learned = _rigid_textured_video(shift_x=-1)
    full = translate_video_cells(
        learned,
        dx=1.0,
        dy=0.0,
        start_frame=0,
    ).video
    compact = translate_video_cells(
        learned[:, :, 3:5],
        dx=1.0,
        dy=0.0,
        start_frame=0,
        batch_frames=2,
    ).video

    full_result = _frame_gauge_boundary_motion_check(
        learned,
        exact_full[:, :, :4],
        full,
        prefix_t=4,
    )
    compact_result = _frame_gauge_boundary_motion_check(
        learned,
        exact_full[:, :, :4],
        compact,
        prefix_t=4,
    )

    assert compact_result[0] == full_result[0]
    assert compact_result[2] == full_result[2]
    for name in ("upper45", "full"):
        for key in ("before_error_cells", "after_error_cells", "error_improvement_ratio"):
            assert compact_result[1]["checks"][name][key] == pytest.approx(full_result[1]["checks"][name][key])


def test_boundary_motion_gate_accepts_rigid_correction_that_restores_native_transition():
    exact_full = _rigid_textured_video()
    learned = _rigid_textured_video(shift_x=-1)
    aligned = translate_video_cells(
        learned,
        dx=1.0,
        dy=0.0,
        start_frame=0,
    ).video

    accepted, fields, reason = _frame_gauge_boundary_motion_check(
        learned,
        exact_full[:, :, :4],
        aligned,
        prefix_t=4,
    )

    assert accepted, fields
    assert reason == "accepted"
    assert fields["status"] == "accepted"
    for name in ("upper45", "full"):
        check = fields["checks"][name]
        assert check["after_error_cells"] < check["before_error_cells"]
        assert check["error_improvement_ratio"] >= 0.25


def test_boundary_motion_gate_accepts_equal_zero_error_as_nondegrading():
    exact_full = _rigid_textured_video()
    learned = exact_full.clone()

    accepted, fields, reason = _frame_gauge_boundary_motion_check(
        learned,
        exact_full[:, :, :4],
        learned.clone(),
        prefix_t=4,
    )

    assert accepted, fields
    assert reason == "accepted"
    for name in ("upper45", "full"):
        check = fields["checks"][name]
        assert check["informative"] is False
        assert check["after_error_cells"] == pytest.approx(check["before_error_cells"])
        assert check["after_error_cells"] == pytest.approx(0.0)


def test_boundary_motion_gate_rejects_translation_that_moves_away_from_native_transition():
    exact_full = _rigid_textured_video()
    learned = _rigid_textured_video(shift_x=-1)
    wrong = translate_video_cells(
        learned,
        dx=-1.0,
        dy=0.0,
        start_frame=0,
    ).video

    accepted, fields, reason = _frame_gauge_boundary_motion_check(
        learned,
        exact_full[:, :, :4],
        wrong,
        prefix_t=4,
    )

    assert not accepted
    assert fields["status"] == "rejected"
    assert reason != "accepted"


def test_guidance_registration_rejection_aborts_the_whole_spatial_transaction():
    exact_full = _rigid_textured_video()
    learned = _rigid_textured_video(shift_x=-1)
    ambiguous_guidance = torch.ones_like(exact_full)
    prefix_t = 4
    high_sigmas, split_coordinate = _schedule()
    run = _run(ambiguous_guidance, split_coordinate)

    postprocess, registered, witnesses, transaction = _frame_gauge_clean_postprocess(
        learned,
        exact_prefix=exact_full[:, :, :prefix_t],
        guidance_run=run,
        guidance=GuidanceConfig(mode="direction"),
        target_h=26,
        target_w=26,
        prefix_t=prefix_t,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert transaction["result"] == "rejected"
    assert str(transaction["reason"]).startswith("guidance_")
    assert registered is None
    assert witnesses == {}
    assert postprocess.clean_video is learned


def test_identity_guidance_calibration_never_resamples_suffix():
    exact_full = _field_video(dx=0.0, dy=0.0)
    prefix_t = 4
    high_sigmas, split_coordinate = _schedule()
    run = _run(exact_full, split_coordinate)

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="direction"),
        exact_prefix=exact_full[:, :, :prefix_t],
        target_h=26,
        target_w=26,
        prefix_t=prefix_t,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert error is None
    assert registered is not None
    assert fields["status"] == "identity"
    assert fields["identity_no_resample"] is True
    assert fields["dx"] == 0.0
    assert fields["dy"] == 0.0
    assert registered.dx == 0.0
    assert registered.dy == 0.0
    assert torch.equal(registered.video, exact_full)


def test_guidance_registration_uses_evidence_based_guidance_policy(monkeypatch):
    exact_full = _field_video(dx=0.0, dy=0.0)
    high_sigmas, split_coordinate = _schedule()
    run = _run(exact_full, split_coordinate)
    observed_policies = []
    original = partitioned_scheduler.estimate_paired_prefix_translation

    def capture_policy(learned, exact, *, policy):
        observed_policies.append(policy)
        return original(learned, exact, policy=policy)

    monkeypatch.setattr(
        partitioned_scheduler,
        "estimate_paired_prefix_translation",
        capture_policy,
    )

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="direction"),
        exact_prefix=exact_full[:, :, :4],
        target_h=26,
        target_w=26,
        prefix_t=4,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert error is None
    assert registered is not None
    assert fields["status"] == "identity"
    assert observed_policies == [GUIDANCE_REFERENCE_POLICY]
    assert GUIDANCE_REFERENCE_POLICY.min_rms_improvement == 0.0
    assert GUIDANCE_REFERENCE_POLICY.require_global_runner_margin is True


def test_guidance_registration_rejection_resizes_only_prefix(monkeypatch):
    exact_full = _field_video(dx=0.0, dy=0.0)
    ambiguous_guidance = torch.ones_like(exact_full)
    high_sigmas, split_coordinate = _schedule()
    run = _run(ambiguous_guidance, split_coordinate)
    resize_temporal = []
    original = partitioned_scheduler.resize_video

    def capture_resize(video, target_h, target_w, *, mode):
        resize_temporal.append(int(video.shape[2]))
        return original(video, target_h, target_w, mode=mode)

    monkeypatch.setattr(partitioned_scheduler, "resize_video", capture_resize)

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="direction"),
        exact_prefix=exact_full[:, :, :4],
        target_h=26,
        target_w=26,
        prefix_t=4,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert registered is None
    assert error is not None
    assert fields["status"] == "rejected"
    assert resize_temporal == [4]


def test_guidance_registration_acceptance_resizes_prefix_and_suffix_once(monkeypatch):
    exact_full = _field_video(dx=0.0, dy=0.0)
    high_sigmas, split_coordinate = _schedule()
    run = _run(exact_full, split_coordinate)
    resize_temporal = []
    original = partitioned_scheduler.resize_video

    def capture_resize(video, target_h, target_w, *, mode):
        resize_temporal.append(int(video.shape[2]))
        return original(video, target_h, target_w, mode=mode)

    monkeypatch.setattr(partitioned_scheduler, "resize_video", capture_resize)

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="direction"),
        exact_prefix=exact_full[:, :, :4],
        target_h=26,
        target_w=26,
        prefix_t=4,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert error is None
    assert registered is not None
    assert fields["status"] == "identity"
    assert resize_temporal == [4, 2]
    assert fields["prefix_resize_elapsed_ms"] >= 0.0
    assert fields["suffix_resize_elapsed_ms"] >= 0.0
    assert fields["translation_elapsed_ms"] >= 0.0


def test_guidance_registration_rejects_unsupported_sampler_contract():
    exact_full = _field_video(dx=0.0, dy=0.0)
    high_sigmas, split_coordinate = _schedule()
    run = replace(_run(exact_full, split_coordinate), sampler="sample_euler")

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="direction"),
        exact_prefix=exact_full[:, :, :4],
        target_h=26,
        target_w=26,
        prefix_t=4,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert registered is None
    assert error == "unsupported_sampler_contract"
    assert fields["status"] == "rejected"
    assert fields["sampler"] == "sample_euler"
    assert fields["supported_samplers"] == ("sample_res_multistep",)


def test_guidance_registration_rejects_downsample_consistency_operator():
    exact_full = _field_video(dx=0.0, dy=0.0)
    high_sigmas, split_coordinate = _schedule()
    run = _run(exact_full, split_coordinate)

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="downsample_consistency"),
        exact_prefix=exact_full[:, :, :4],
        target_h=26,
        target_w=26,
        prefix_t=4,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert registered is None
    assert error == "unsupported_guidance_operator"
    assert fields == {"status": "rejected", "reason": "unsupported_guidance_operator"}


def test_guidance_registration_rejects_high_schedule_that_changes_reference_identity():
    exact_full = _field_video(dx=0.0, dy=0.0)
    high_sigmas, split_coordinate = _schedule()
    run = _run(exact_full, split_coordinate)
    lower = TrajectorySample(
        0.02,
        0.2,
        0.2,
        1,
        1,
        "corrected",
        "actual",
        exact_full.clone(),
    )
    run = replace(run, samples=(run.samples[0], lower))

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="direction"),
        exact_prefix=exact_full[:, :, :4],
        target_h=26,
        target_w=26,
        prefix_t=4,
        split_coordinate=split_coordinate,
        high_sigmas=high_sigmas,
        video_shift=H3_VIDEO_SHIFT,
    )

    assert registered is None
    assert error == "high_schedule_changes_reference_identity"
    assert fields["status"] == "rejected"
    assert fields["reference_coordinate"] == pytest.approx(split_coordinate)
    assert any(value != pytest.approx(split_coordinate) for value in fields["resolved_high_coordinates"])


def test_guidance_registration_prefers_exact_handoff_probe_at_duplicate_coordinate():
    exact_full = _field_video(dx=0.0, dy=0.0)
    wrong = _field_video(dx=1.0, dy=0.0)
    _, split_coordinate = _schedule()
    samples = (
        TrajectorySample(
            split_coordinate,
            0.8,
            0.8,
            0,
            0,
            "corrected",
            "actual",
            wrong,
        ),
        TrajectorySample(
            split_coordinate,
            0.8,
            0.8,
            0,
            1,
            "handoff_probe",
            "actual",
            exact_full.clone(),
        ),
    )
    run = TrajectoryRun(
        1,
        "frame-gauge-duplicate",
        "session",
        "chunk",
        "sample_res_multistep",
        "schedule",
        geometry_from_video(exact_full),
        (1, 32, 2, 8),
        "layout",
        "conditioning",
        "system_ram",
        samples,
        0,
        1,
        True,
    )

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="direction"),
        exact_prefix=exact_full[:, :, :4],
        target_h=26,
        target_w=26,
        prefix_t=4,
        split_coordinate=split_coordinate,
        high_sigmas=torch.tensor([0.8, 0.5, 0.0]),
        video_shift=H3_VIDEO_SHIFT,
    )

    assert error is None
    assert registered is not None
    assert fields["status"] == "identity"
    assert torch.equal(registered.video, exact_full)


def test_guidance_registration_rejects_high_schedule_above_probe_endpoint():
    exact_full = _field_video(dx=0.0, dy=0.0)
    _, split_coordinate = _schedule()
    run = _run(exact_full, split_coordinate)

    registered, fields, error = _prepare_registered_guidance_reference(
        run=run,
        guidance=GuidanceConfig(mode="direction"),
        exact_prefix=exact_full[:, :, :4],
        target_h=26,
        target_w=26,
        prefix_t=4,
        split_coordinate=split_coordinate,
        high_sigmas=torch.tensor([0.9, 0.5, 0.0]),
        video_shift=H3_VIDEO_SHIFT,
    )

    assert registered is None
    assert error == "high_schedule_exceeds_probe_endpoint"
    assert fields["status"] == "rejected"
