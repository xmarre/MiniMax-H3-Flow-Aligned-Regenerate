"""Runtime-only geometry for partitioned exact-prefix progressive stages.

This is deliberately separate from the retired Mixed-Grid contract.  The sampler
uses a uniform low-grid carrier, while the transformer receives an explicit
[target-grid exact prefix | source-grid generated suffix] video domain.  The
cross-repo attention contract is published separately by ``partitioned_prefix``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .geometry import unpack_streams

PARTITIONED_STAGE_KEY = "h3_flow_partitioned_stage_v1"


@dataclass(frozen=True, slots=True)
class PartitionedStagePlan:
    prefix: torch.Tensor
    temporal: int
    source_h: int
    source_w: int
    prefix_noise: torch.Tensor

    @property
    def prefix_t(self) -> int:
        return int(self.prefix.shape[2])

    @property
    def target_hw(self) -> tuple[int, int]:
        return tuple(map(int, self.prefix.shape[-2:]))

    @property
    def source_grid(self) -> tuple[int, int]:
        return self.source_h // 2, self.source_w // 2

    @property
    def target_grid(self) -> tuple[int, int]:
        return self.target_hw[0] // 2, self.target_hw[1] // 2

    @property
    def source_rows(self) -> int:
        return self.source_h * self.source_w // 4

    @property
    def target_rows(self) -> int:
        return self.target_hw[0] * self.target_hw[1] // 4

    @property
    def prefix_rows(self) -> int:
        return self.prefix_t * self.target_rows

    @property
    def suffix_rows(self) -> int:
        return (self.temporal - self.prefix_t) * self.source_rows

    @property
    def partitioned_rows(self) -> int:
        return self.prefix_rows + self.suffix_rows


@dataclass(slots=True)
class PartitionedStageRuntime:
    """One mutable owner whose identity is stable for one sampler-stage lifetime.

    ComfyUI recursively copies nested transformer-option dictionaries between
    model calls.  Publishing the stage itself as a dictionary therefore cannot
    own identity-sensitive runtime state.  This object is intentionally a
    non-dict leaf so ComfyUI's options cloning preserves the same owner while the
    surrounding low/probe sampler lifetime is active.
    """

    plan: PartitionedStagePlan
    metrics: object
    # Keyed by the same semantic provider identity used by Sol history-v1:
    # ordered preprocess implementation names plus terminal provider name/object
    # identity. Generic preprocess wrappers may legitimately be reconstructed per
    # model call, so raw wrapper-object identity is not a stable numerical key.
    attention_provider_cache: dict[tuple[object, ...], object] = field(default_factory=dict)
    attention_provider_transforms: tuple[object, ...] = ()
    attention_provider_terminal: object = None
    attention_provider_identity: tuple[object, ...] | None = None
    # Captured once at stage entry and republished explicitly to each VDN block call.
    vdn_linear_diagnostic: str = "normal"


def build_partitioned_stage_plan(
    mask,
    shapes,
    internal_latent,
    sampler_noise,
    *,
    source_h: int,
    source_w: int,
) -> PartitionedStagePlan:
    """Build the exact-prefix stage plan before any split sampler lifetime starts."""
    if mask is None:
        raise ValueError("partitioned exact-prefix requires an explicit prepared H3 mask")
    if len(shapes) != 2:
        raise ValueError("partitioned exact-prefix requires native packed H3 video/audio shapes")
    video_mask, _ = unpack_streams(mask, shapes)
    video, _ = unpack_streams(internal_latent, shapes)
    video_noise, _ = unpack_streams(sampler_noise, shapes)
    if video.ndim != 5 or video.shape[:2] != (1, 24):
        raise ValueError("partitioned exact-prefix requires batch-one native H3 video")
    if tuple(video_noise.shape) != tuple(video.shape):
        raise ValueError("partitioned exact-prefix sampler noise does not match target video geometry")
    if not bool(torch.isfinite(video_mask).all().item()):
        raise ValueError("partitioned exact-prefix video mask must be finite")
    if not bool(torch.isfinite(video_noise).all().item()):
        raise ValueError("partitioned exact-prefix sampler noise must be finite")

    frames = video_mask.permute(2, 0, 1, 3, 4).reshape(video.shape[2], -1)
    protected = (frames == 0).all(1)
    generated = (frames == 1).all(1)
    prefix_t = int(protected.sum().item())
    if not 0 < prefix_t < int(video.shape[2]):
        raise ValueError("partitioned exact-prefix requires a nonempty exact prefix and generated suffix")
    if not bool(protected[:prefix_t].all().item() and generated[prefix_t:].all().item()):
        raise ValueError("partitioned exact-prefix requires a contiguous whole-frame zero-prefix/one-suffix mask")

    target_h, target_w = map(int, video.shape[-2:])
    if any(n < 2 or n % 2 for n in (source_h, source_w, target_h, target_w)):
        raise ValueError("partitioned exact-prefix spatial axes must be positive and H3 patch-safe")
    if source_h > target_h or source_w > target_w or (source_h, source_w) == (target_h, target_w):
        raise ValueError("partitioned exact-prefix source must strictly reduce the target spatial grid")

    return PartitionedStagePlan(
        prefix=video[:, :, :prefix_t].detach().clone(),
        temporal=int(video.shape[2]),
        source_h=int(source_h),
        source_w=int(source_w),
        prefix_noise=video_noise[:, :, :prefix_t].detach().clone(),
    )


def partitioned_carrier_layout(native, plan, text_len: int, audio_t: int, payload: dict):
    """Construct the native low-grid carrier without altering target conditioning rows."""
    layout = native.PackedLayout(
        text_len,
        plan.temporal,
        *plan.target_hw,
        audio_t,
        keyframes=payload.get("keyframes"),
        refs=payload.get("refs"),
    )
    video_start, _, _ = layout.segments[-1]
    origin = float(layout.position_ids[video_start, 0])
    frame, _ = native._frame_grid(plan.source_h, plan.source_w)
    positions = native._video_grid(plan.temporal, frame, origin)
    layout.position_ids = torch.cat((layout.position_ids[:video_start], positions))
    layout.seq_len = video_start + len(positions)
    layout.segments = [*layout.segments[:-1], (video_start, layout.seq_len, "video")]
    keep = layout.img_pos < video_start
    layout.img_pos = torch.cat((layout.img_pos[keep], torch.arange(video_start, layout.seq_len)))
    layout.img_update = torch.cat((layout.img_update[keep], torch.ones(len(positions), dtype=torch.bool)))
    layout.signature = (text_len, plan.temporal, plan.source_h, plan.source_w, audio_t)
    return layout


def partitioned_positions(native, plan: PartitionedStagePlan, layout):
    """Return physical RoPE rows for target-prefix and source-suffix domains."""
    video_start, _, _ = layout.segments[-1]
    origin = float(layout.position_ids[video_start, 0])
    frame, _ = native._frame_grid(*plan.target_hw)
    prefix = native._video_grid(plan.temporal, frame, origin)[: plan.prefix_rows]
    # The suffix remains on the globally constructed low-grid timeline.  Never
    # restart temporal phase at the prefix/suffix boundary.
    suffix = layout.position_ids[video_start + plan.prefix_t * plan.source_rows :]
    return torch.cat((layout.position_ids[:video_start], prefix, suffix))


def partitioned_mod_segments(segments, plan: PartitionedStagePlan, video_start: int, video_end: int):
    result = []
    for start, stop, row in segments:
        if stop <= video_start:
            result.append((start, stop, row))
        elif (start, stop) == (video_start, video_end):
            if not torch.is_tensor(row) or row.ndim != 1 or row.numel() != video_end - video_start:
                raise RuntimeError("partitioned exact-prefix requires native per-video-row timestep indices")
            carrier_prefix_rows = plan.prefix_t * plan.source_rows
            if not bool((row[:carrier_prefix_rows] == row[0]).all().item()):
                raise RuntimeError("partitioned exact-prefix protected prefix has inconsistent timestep labels")
            expanded = torch.cat((row[:1].expand(plan.prefix_rows), row[carrier_prefix_rows:]))
            result.append((video_start, video_start + plan.partitioned_rows, expanded))
        else:
            raise RuntimeError("partitioned exact-prefix requires target video to be the final native segment")
    return result


__all__ = [
    "PARTITIONED_STAGE_KEY",
    "PartitionedStagePlan",
    "PartitionedStageRuntime",
    "build_partitioned_stage_plan",
    "partitioned_carrier_layout",
    "partitioned_mod_segments",
    "partitioned_positions",
]
