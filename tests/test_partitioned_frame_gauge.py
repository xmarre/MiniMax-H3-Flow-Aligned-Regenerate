from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from h3_flow_regenerate.contracts import TrajectoryRun, TrajectorySample
from h3_flow_regenerate.geometry import geometry_from_video
from h3_flow_regenerate.guidance import GuidanceConfig
from h3_flow_regenerate.partitioned_scheduler import (
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
    exact_full = _field_video(dx=0.0, dy=0.0)
    learned = _field_video(dx=0.5, dy=-0.375)
    guidance_reference = _field_video(dx=-0.625, dy=0.4375)
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
    assert transaction["video_registration"]["dx"] == pytest.approx(0.5, abs=0.125)
    assert transaction["video_registration"]["dy"] == pytest.approx(-0.375, abs=0.125)
    assert transaction["guidance_registration"]["dx"] == pytest.approx(-0.625, abs=0.125)
    assert transaction["guidance_registration"]["dy"] == pytest.approx(0.4375, abs=0.125)
    assert registered is not None
    assert registered.dx != pytest.approx(transaction["video_registration"]["dx"], abs=0.125)
    assert torch.equal(postprocess.clean_video[:, :, :prefix_t], learned[:, :, :prefix_t])
    assert set(witnesses) == {
        "learned_native",
        "paired_prefix_aligned_witness",
        "corrected_clean",
    }


def test_guidance_registration_rejection_aborts_the_whole_spatial_transaction():
    exact_full = _field_video(dx=0.0, dy=0.0)
    learned = _field_video(dx=0.5, dy=-0.375)
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
