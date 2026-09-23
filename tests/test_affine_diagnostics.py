from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from h3_flow_regenerate.seam_diagnostics import measure_affine_trajectory


def _base_frame(height: int = 56, width: int = 74) -> torch.Tensor:
    generator = torch.Generator().manual_seed(224)
    frame = torch.rand(
        (1, 12, height, width),
        generator=generator,
        dtype=torch.float32,
    )
    return F.avg_pool2d(frame, kernel_size=3, stride=1, padding=1)


def _affine_frame(
    frame: torch.Tensor,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
) -> torch.Tensor:
    _, _, height, width = frame.shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    source_x = (x - float(translation_x)) / float(scale_x)
    source_y = (y - float(translation_y)) / float(scale_y)
    grid = torch.stack(
        (
            2.0 * source_x / float(width - 1) - 1.0,
            2.0 * source_y / float(height - 1) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0)
    return F.grid_sample(
        frame,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _video(*frames: torch.Tensor) -> torch.Tensor:
    return torch.stack([frame[0] for frame in frames], dim=1).unsqueeze(0)


def test_affine_trajectory_recovers_boundary_scale_without_mutation():
    base = _base_frame()
    expanded = _affine_frame(
        base,
        scale_x=1.01,
        scale_y=1.03,
        translation_x=0.25,
        translation_y=-0.50,
    )
    video = _video(base, base, expanded, expanded, expanded)
    original = video.clone()

    result = measure_affine_trajectory(
        video,
        2,
        forward_steps=2,
        backward_steps=1,
        roi_fraction=1.0,
        max_shift=4,
    )

    assert torch.equal(video, original)
    assert result["affine_diagnostic_version"] == 1
    assert result["boundary_scale_x"] == pytest.approx(1.01, abs=0.015)
    assert result["boundary_scale_y"] == pytest.approx(1.03, abs=0.015)
    assert result["boundary_translation_x"] == pytest.approx(0.25, abs=0.40)
    assert result["boundary_translation_y"] == pytest.approx(-0.50, abs=0.40)
    assert result["boundary_scale_confidence_x"] > 0.25
    assert result["boundary_scale_confidence_y"] > 0.25
    assert result["pre_median_scale_x"] == pytest.approx(1.0, abs=0.005)
    assert result["pre_median_scale_y"] == pytest.approx(1.0, abs=0.005)


def test_affine_trajectory_does_not_misclassify_translation_as_scale():
    base = _base_frame()
    shifted = _affine_frame(
        base,
        translation_x=1.5,
        translation_y=-2.0,
    )
    video = _video(base, base, shifted, shifted)

    result = measure_affine_trajectory(
        video,
        2,
        forward_steps=1,
        backward_steps=1,
        roi_fraction=1.0,
        max_shift=4,
    )

    assert result["boundary_scale_x"] == pytest.approx(1.0, abs=0.005)
    assert result["boundary_scale_y"] == pytest.approx(1.0, abs=0.005)
    assert result["boundary_translation_x"] == pytest.approx(1.5, abs=0.20)
    assert result["boundary_translation_y"] == pytest.approx(-2.0, abs=0.20)
    assert result["boundary_scale_confidence_x"] < 0.25
    assert result["boundary_scale_confidence_y"] < 0.25
