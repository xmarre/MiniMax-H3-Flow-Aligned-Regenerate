from pathlib import Path

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.partitioned_stage import (
    PartitionedStagePlan,
    build_partitioned_stage_plan,
    partitioned_mod_segments,
)


def _fixture():
    generator = torch.Generator(device="cpu").manual_seed(1401)
    video = torch.randn((1, 24, 5, 8, 8), generator=generator)
    audio = torch.randn((1, 32, 2, 11), generator=generator)
    noise_video = torch.randn(video.shape, generator=generator)
    noise_audio = torch.randn(audio.shape, generator=generator)
    video_mask = torch.ones_like(video)
    video_mask[:, :, :2] = 0
    audio_mask = torch.ones_like(audio)
    internal, shapes = pack_streams((video, audio))
    noise, _ = pack_streams((noise_video, noise_audio))
    mask, _ = pack_streams((video_mask, audio_mask))
    return video, noise_video, internal, noise, mask, shapes


def test_partitioned_stage_keeps_authoritative_target_prefix_and_source_suffix_geometry():
    video, noise_video, internal, noise, mask, shapes = _fixture()
    plan = build_partitioned_stage_plan(
        mask,
        shapes,
        internal,
        noise,
        source_h=4,
        source_w=4,
    )
    assert isinstance(plan, PartitionedStagePlan)
    assert plan.prefix_t == 2
    assert plan.target_hw == (8, 8)
    assert plan.source_grid == (2, 2)
    assert plan.target_grid == (4, 4)
    assert plan.source_rows == 4
    assert plan.target_rows == 16
    assert plan.prefix_rows == 32
    assert plan.suffix_rows == 12
    assert plan.partitioned_rows == 44
    assert torch.equal(plan.prefix, video[:, :, :2])
    assert torch.equal(plan.prefix_noise, noise_video[:, :, :2])


def test_partitioned_stage_rejects_noncanonical_or_nonreducing_geometry():
    _video, _noise_video, internal, noise, mask, shapes = _fixture()
    bad_mask = mask.clone()
    # Turn one value in the generated suffix into a fractional blend. The new
    # mode accepts only a whole-frame exact prefix followed by whole generated frames.
    video_values = 24 * 5 * 8 * 8
    bad_mask[..., 2 * 24 * 8 * 8] = 0.5
    assert video_values < bad_mask.shape[-1]
    with pytest.raises(ValueError, match="contiguous whole-frame"):
        build_partitioned_stage_plan(
            bad_mask,
            shapes,
            internal,
            noise,
            source_h=4,
            source_w=4,
        )

    with pytest.raises(ValueError, match="strictly reduce"):
        build_partitioned_stage_plan(
            mask,
            shapes,
            internal,
            noise,
            source_h=8,
            source_w=8,
        )


def test_partitioned_mod_segments_expands_only_exact_prefix_rows():
    video, noise_video, _internal, _noise, _mask, _shapes = _fixture()
    plan = PartitionedStagePlan(
        prefix=video[:, :, :2].clone(),
        temporal=5,
        source_h=4,
        source_w=4,
        prefix_noise=noise_video[:, :, :2].clone(),
    )
    video_start = 3
    carrier_rows = plan.temporal * plan.source_rows
    row = torch.ones(carrier_rows)
    result = partitioned_mod_segments(
        [(0, video_start, torch.zeros(video_start)), (video_start, video_start + carrier_rows, row)],
        plan,
        video_start,
        video_start + carrier_rows,
    )
    assert result[0][0:2] == (0, video_start)
    assert result[1][0:2] == (video_start, video_start + plan.partitioned_rows)
    assert result[1][2].numel() == plan.partitioned_rows


def test_partitioned_runtime_does_not_reinterpret_retired_mixed_grid_mode():
    root = Path(__file__).resolve().parents[1] / "h3_flow_regenerate"
    node = (root / "partitioned_node.py").read_text(encoding="utf-8")
    transformer = (root / "partitioned_transformer.py").read_text(encoding="utf-8")
    scheduler = (root / "partitioned_scheduler.py").read_text(encoding="utf-8")

    assert 'exact_prefix_mode="mixed_grid_low_suffix"' not in node
    assert "from .mixed_grid import" not in node
    assert "from .mixed_grid import" not in transformer
    assert "from .mixed_grid import" not in scheduler
    assert not (root / "partitioned_mixed.py").exists()
