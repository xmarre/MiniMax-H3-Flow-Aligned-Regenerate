"""Runtime-only geometry for partitioned exact-prefix progressive stages.

This is deliberately separate from the retired Mixed-Grid contract.  The sampler
uses a uniform low-grid carrier, while the transformer receives an explicit
[target-grid exact prefix | source-grid generated suffix] video domain.  The
cross-repo attention contract is published separately by ``partitioned_prefix``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from .geometry import unpack_streams
from .partitioned_diagnostics import (
    PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
)

PARTITIONED_STAGE_KEY = "h3_flow_partitioned_stage_v1"
PARTITIONED_POSITION_POLICY_TAG = "h3_flow_partitioned_position_policy_v1"


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PartitionedPositionPolicy:
    mode: str
    source_hw: tuple[int, int]
    target_hw: tuple[int, int]
    stage_owner_generation: int
    audio_range: tuple[int, int]
    target_audio_spatial_endpoints: tuple[tuple[float, float], tuple[float, float]]
    source_audio_spatial_endpoints: tuple[tuple[float, float], tuple[float, float]]
    audio_temporal_equal: bool
    temporal_digest: str
    non_audio_before_digest: str
    non_audio_after_digest: str
    prefix_rope_position_digest: str
    suffix_rope_position_digest: str
    position_digest: str

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            PARTITIONED_POSITION_POLICY_TAG,
            self.mode,
            self.source_hw,
            self.target_hw,
            self.stage_owner_generation,
            self.audio_range,
            self.target_audio_spatial_endpoints,
            self.source_audio_spatial_endpoints,
            self.audio_temporal_equal,
            self.temporal_digest,
            self.non_audio_before_digest,
            self.non_audio_after_digest,
            self.prefix_rope_position_digest,
            self.suffix_rope_position_digest,
            self.position_digest,
        )


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
    # Diagnostic-only structural A/B. The ordinary node always uses the exact
    # target-grid prefix inside the heterogeneous low/probe transformer.
    prefix_transformer_context: str = "exact_target_partitioned"
    audio_position_domain: str = PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY
    position_policy: PartitionedPositionPolicy | None = None
    position_policy_positions: torch.Tensor | None = None


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


def _target_audio_range(layout) -> tuple[int, int]:
    matches = [
        (int(start), int(stop))
        for start, stop, kind in layout.segments
        if kind == "audio"
    ]
    if len(matches) != 1:
        raise RuntimeError("source-carrier audio-position candidate requires exactly one target audio segment")
    start, stop = matches[0]
    if start < 0 or stop <= start or stop > int(layout.position_ids.shape[0]) or (stop - start) % 2:
        raise RuntimeError("source-carrier audio-position candidate found malformed target audio rows")
    return start, stop


def partitioned_positions_with_audio_policy(
    native,
    plan: PartitionedStagePlan,
    layout,
    audio_position_domain: str,
    *,
    stage_owner_generation: int = 0,
) -> tuple[torch.Tensor, PartitionedPositionPolicy | None]:
    """Apply the opt-in target-audio spatial coordinate policy.

    The legacy path is exactly the pre-candidate arithmetic. The candidate changes
    only the spatial columns of the target audio segment; reference and conditioning
    audio rows are not selected by this segment lookup.
    """
    positions = partitioned_positions(native, plan, layout)
    if audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_LEGACY:
        return positions, None
    if audio_position_domain != PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
        raise RuntimeError(f"unsupported partitioned audio position domain {audio_position_domain!r}")
    if not callable(getattr(native, "_frame_grid", None)) or not callable(getattr(native, "_audio_grid", None)):
        raise RuntimeError("installed MiniMax-H3 Core does not expose the required native audio/grid constructors")

    audio_start, audio_stop = _target_audio_range(layout)
    audio_rows = positions[audio_start:audio_stop]
    audio_t = (audio_stop - audio_start) // 2
    _source_frame, source_w_grid = native._frame_grid(plan.source_h, plan.source_w)
    if source_w_grid.numel() < 1:
        raise RuntimeError("source-carrier audio-position candidate produced an empty source width grid")
    source_audio = native._audio_grid(
        float(audio_rows[0, 0]),
        audio_t,
        float(source_w_grid[0]),
        float(source_w_grid[-1]),
    )
    if tuple(source_audio.shape) != tuple(audio_rows.shape):
        raise RuntimeError("source-carrier native audio grid does not match target audio row geometry")
    if not torch.equal(source_audio[:, 0], audio_rows[:, 0]):
        raise RuntimeError("source-carrier audio-position candidate changed target audio temporal coordinates")

    candidate = positions.clone()
    candidate[audio_start:audio_stop, 1:] = source_audio[:, 1:].to(candidate)
    if not torch.equal(candidate[audio_start:audio_stop, 0], positions[audio_start:audio_stop, 0]):
        raise RuntimeError("source-carrier audio-position candidate changed target audio temporal coordinates")

    unchanged_before = torch.cat((positions[:audio_start], positions[audio_stop:]))
    unchanged_after = torch.cat((candidate[:audio_start], candidate[audio_stop:]))
    if not torch.equal(unchanged_before, unchanged_after):
        raise RuntimeError("source-carrier audio-position candidate mutated non-target-audio position rows")

    video_start = int(layout.segments[-1][0])
    prefix_stop = video_start + plan.prefix_rows
    target_spatial_endpoints = (
        tuple(map(float, audio_rows[0, 1:].tolist())),
        tuple(map(float, audio_rows[audio_t, 1:].tolist())),
    )
    source_spatial_endpoints = (
        tuple(map(float, source_audio[0, 1:].tolist())),
        tuple(map(float, source_audio[audio_t, 1:].tolist())),
    )
    non_audio_before_digest = tensor_sha256(unchanged_before)
    non_audio_after_digest = tensor_sha256(unchanged_after)
    if non_audio_before_digest != non_audio_after_digest:
        raise RuntimeError("source-carrier audio-position candidate changed non-audio position digest")
    policy = PartitionedPositionPolicy(
        mode=audio_position_domain,
        source_hw=(int(plan.source_h), int(plan.source_w)),
        target_hw=tuple(map(int, plan.target_hw)),
        stage_owner_generation=int(stage_owner_generation),
        audio_range=(audio_start, audio_stop),
        target_audio_spatial_endpoints=target_spatial_endpoints,
        source_audio_spatial_endpoints=source_spatial_endpoints,
        audio_temporal_equal=True,
        temporal_digest=tensor_sha256(candidate[audio_start:audio_stop, :1]),
        non_audio_before_digest=non_audio_before_digest,
        non_audio_after_digest=non_audio_after_digest,
        prefix_rope_position_digest=tensor_sha256(candidate[video_start:prefix_stop]),
        suffix_rope_position_digest=tensor_sha256(candidate[prefix_stop:]),
        position_digest=tensor_sha256(candidate),
    )
    return candidate, policy


def partitioned_positions_for_runtime(native, runtime: PartitionedStageRuntime, layout):
    """Resolve one immutable position policy/position tensor per sampler-stage owner."""
    mode = str(runtime.audio_position_domain)
    positions, policy = partitioned_positions_with_audio_policy(
        native,
        runtime.plan,
        layout,
        mode,
        stage_owner_generation=id(runtime),
    )
    if policy is None:
        if runtime.position_policy is not None or runtime.position_policy_positions is not None:
            raise RuntimeError("legacy audio-position stage unexpectedly retained candidate position state")
        return positions, None

    if runtime.position_policy is None:
        runtime.position_policy = policy
        runtime.position_policy_positions = positions
        return positions, policy

    if runtime.position_policy_positions is None:
        raise RuntimeError("partitioned audio-position policy lost its validated position tensor")
    if runtime.position_policy.signature != policy.signature:
        raise RuntimeError("partitioned audio-position policy drifted within one sampler-stage lifetime")
    if not torch.equal(runtime.position_policy_positions, positions):
        raise RuntimeError("partitioned audio-position rows drifted within one sampler-stage lifetime")
    return runtime.position_policy_positions, runtime.position_policy


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
    "PARTITIONED_POSITION_POLICY_TAG",
    "PARTITIONED_STAGE_KEY",
    "PartitionedPositionPolicy",
    "PartitionedStagePlan",
    "PartitionedStageRuntime",
    "build_partitioned_stage_plan",
    "partitioned_carrier_layout",
    "partitioned_mod_segments",
    "partitioned_positions",
    "partitioned_positions_for_runtime",
    "partitioned_positions_with_audio_policy",
    "tensor_sha256",
]
