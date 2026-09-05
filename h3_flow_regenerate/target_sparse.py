from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .geometry import unpack_streams
from .metrics import H3FlowMetrics

TARGET_SPARSE_CONTRACT_KEY = "h3_flow_target_sparse_v1"
TARGET_SPARSE_API_VERSION = 1
VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
VDN_EXTERNAL_SEQUENCE_API_VERSION = 1
VDN_EXTERNAL_SEQUENCE_MODE = "dense_gate_no_linear"


@dataclass(frozen=True, slots=True)
class TargetSparsePlan:
    """Immutable early-stage token plan for exact-prefix target-grid sampling.

    The sampler latent remains on the full target grid. Only the H3 transformer
    token stream is reduced: all non-video rows and every exactly protected video
    patch row are retained, while generated video rows are represented by a
    regular target-grid anchor lattice. The final block lifts the anchor features
    back to the full target video grid before H3's native final layer.
    """

    target_t: int
    target_h: int
    target_w: int
    source_h: int
    source_w: int
    selected_video_rows: torch.Tensor
    anchor_video_rows: torch.Tensor
    protected_video_rows: torch.Tensor

    @property
    def target_patch_h(self) -> int:
        return self.target_h // 2

    @property
    def target_patch_w(self) -> int:
        return self.target_w // 2

    @property
    def source_patch_h(self) -> int:
        return self.source_h // 2

    @property
    def source_patch_w(self) -> int:
        return self.source_w // 2

    @property
    def target_video_rows(self) -> int:
        return self.target_t * self.target_patch_h * self.target_patch_w

    @property
    def selected_video_row_count(self) -> int:
        return int(self.selected_video_rows.numel())

    @property
    def anchor_video_row_count(self) -> int:
        return int(self.anchor_video_rows.numel())

    @property
    def protected_video_row_count(self) -> int:
        return int(self.protected_video_rows.numel())

    @property
    def video_row_fraction(self) -> float:
        return self.selected_video_row_count / self.target_video_rows


def _axis_anchor_indices(target_count: int, source_count: int) -> torch.Tensor:
    if target_count < 1 or source_count < 1:
        raise ValueError("target/source patch axes must be non-empty")
    if source_count > target_count:
        raise ValueError("target-sparse source patch axis cannot exceed target patch axis")
    if source_count == target_count:
        return torch.arange(target_count, dtype=torch.long)
    if source_count == 1:
        return torch.tensor([target_count // 2], dtype=torch.long)
    values = torch.linspace(0, target_count - 1, source_count, dtype=torch.float64).round().to(torch.long)
    if int(torch.unique(values).numel()) != source_count:
        raise RuntimeError("target-sparse anchor mapping produced duplicate target positions")
    return values


def _anchor_video_rows(
    *,
    target_t: int,
    target_patch_h: int,
    target_patch_w: int,
    source_patch_h: int,
    source_patch_w: int,
) -> torch.Tensor:
    ys = _axis_anchor_indices(target_patch_h, source_patch_h)
    xs = _axis_anchor_indices(target_patch_w, source_patch_w)
    spatial = (ys[:, None] * target_patch_w + xs[None, :]).reshape(-1)
    rows_per_frame = target_patch_h * target_patch_w
    frames = torch.arange(target_t, dtype=torch.long)[:, None] * rows_per_frame
    return (frames + spatial[None, :]).reshape(-1)


def exact_protected_video_patch_rows(
    denoise_mask: torch.Tensor,
    shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    """Mirror native H3's 2x2 max-mask row semantics for exact protection."""

    if len(shapes) != 2:
        raise ValueError("target-sparse protection requires packed video and audio shapes")
    expected = (shapes[0][0], 1, sum(torch.tensor(shape[1:]).prod().item() for shape in shapes))
    if tuple(denoise_mask.shape) != tuple(int(value) for value in expected):
        raise ValueError("prepared H3 denoise mask does not match packed AV geometry")
    video_mask, _audio_mask = unpack_streams(denoise_mask, shapes)
    if video_mask.ndim != 5 or video_mask.shape[1] < 1:
        raise ValueError("prepared H3 video denoise mask has invalid shape")
    _batch, _channels, target_t, target_h, target_w = map(int, video_mask.shape)
    if target_h % 2 or target_w % 2:
        raise ValueError("target-sparse H3 video geometry must be patch-safe")

    # Native MiniMax H3 consumes denoise_mask[0, 0] and takes a 2x2 spatial
    # maximum for each packed video row. A row is therefore exactly protected
    # only when every pixel in that 2x2 patch is exactly zero.
    rows = video_mask[0, 0].to(device="cpu", dtype=torch.float32)
    row_values = rows.reshape(target_t, target_h // 2, 2, target_w // 2, 2).amax(dim=(2, 4)).reshape(-1)
    return torch.nonzero(row_values == 0, as_tuple=False).reshape(-1).to(torch.long)


def build_target_sparse_plan(
    denoise_mask: torch.Tensor,
    shapes: list[tuple[int, ...]],
    *,
    source_h: int,
    source_w: int,
) -> TargetSparsePlan:
    if len(shapes) != 2:
        raise ValueError("target-sparse plan requires native packed H3 AV geometry")
    target_shape = shapes[0]
    if len(target_shape) != 5 or int(target_shape[1]) != 24:
        raise ValueError("target-sparse plan requires Bx24xTxHxW video geometry")
    target_t, target_h, target_w = map(int, target_shape[-3:])
    source_h = int(source_h)
    source_w = int(source_w)
    if source_h < 2 or source_w < 2 or source_h % 2 or source_w % 2:
        raise ValueError("target-sparse source latent H/W must be positive and even")
    if source_h > target_h or source_w > target_w:
        raise ValueError("target-sparse source latent H/W cannot exceed target geometry")
    if source_h == target_h and source_w == target_w:
        raise ValueError("target-sparse source must reduce at least one spatial axis")

    anchor_rows = _anchor_video_rows(
        target_t=target_t,
        target_patch_h=target_h // 2,
        target_patch_w=target_w // 2,
        source_patch_h=source_h // 2,
        source_patch_w=source_w // 2,
    )
    protected_rows = exact_protected_video_patch_rows(denoise_mask, shapes)
    if protected_rows.numel() == 0:
        raise ValueError("target-sparse exact-prefix mode requires at least one exactly protected video row")
    selected_rows = torch.unique(torch.cat((anchor_rows, protected_rows)), sorted=True)
    return TargetSparsePlan(
        target_t=target_t,
        target_h=target_h,
        target_w=target_w,
        source_h=source_h,
        source_w=source_w,
        selected_video_rows=selected_rows.contiguous(),
        anchor_video_rows=anchor_rows.contiguous(),
        protected_video_rows=protected_rows.contiguous(),
    )


def target_sparse_contract(plan: TargetSparsePlan) -> dict[str, Any]:
    return {
        "api": TARGET_SPARSE_API_VERSION,
        "active": True,
        "plan": plan,
        "device_cache": {},
    }


def _video_segment(layout: Any) -> tuple[int, int]:
    segments = getattr(layout, "segments", None)
    if not isinstance(segments, (list, tuple)):
        raise RuntimeError("target-sparse H3 path requires PackedLayout.segments")
    video = [(int(a), int(b)) for a, b, kind in segments if str(kind) == "video"]
    if len(video) != 1:
        raise RuntimeError(f"target-sparse H3 path requires exactly one target video segment, got {len(video)}")
    return video[0]


def _device_indices(contract: dict[str, Any], plan: TargetSparsePlan, device: torch.device) -> dict[str, torch.Tensor]:
    cache = contract.setdefault("device_cache", {})
    key = str(device)
    cached = cache.get(key)
    if isinstance(cached, dict):
        return cached
    selected = plan.selected_video_rows.to(device=device)
    anchors = plan.anchor_video_rows.to(device=device)
    anchor_positions = torch.searchsorted(selected, anchors)
    if not torch.equal(selected.index_select(0, anchor_positions), anchors):
        raise RuntimeError("target-sparse anchor rows are not a subset of selected rows")
    cached = {
        "selected_video": selected,
        "anchors_video": anchors,
        "anchor_positions": anchor_positions,
    }
    cache[key] = cached
    return cached


def _reduce_mod_segments(
    mod_segments: Any,
    *,
    video_start: int,
    video_stop: int,
    selected_video: torch.Tensor,
) -> list[tuple[int, int, Any]]:
    if not isinstance(mod_segments, (list, tuple)):
        raise RuntimeError("target-sparse H3 path requires native mod_segments")
    reduced: list[tuple[int, int, Any]] = []
    selected_count = int(selected_video.numel())
    found_video = False
    for segment in mod_segments:
        if not isinstance(segment, (list, tuple)) or len(segment) < 3:
            raise RuntimeError("target-sparse H3 mod_segments contains an invalid segment")
        start, stop, row = int(segment[0]), int(segment[1]), segment[2]
        if stop <= video_start:
            reduced.append((start, stop, row))
            continue
        if start == video_start and stop == video_stop:
            found_video = True
            if torch.is_tensor(row) and row.numel() != 1:
                if row.ndim != 1 or int(row.numel()) != video_stop - video_start:
                    raise RuntimeError("target-sparse H3 video modulation rows do not match the full video segment")
                row = row.index_select(0, selected_video.to(device=row.device))
            reduced.append((video_start, video_start + selected_count, row))
            continue
        raise RuntimeError(
            "target-sparse H3 path encountered packed rows after or overlapping the target video segment"
        )
    if not found_video:
        raise RuntimeError("target-sparse H3 path could not locate target video modulation metadata")
    return reduced


def _lift_video_hidden(
    compact_video: torch.Tensor,
    *,
    plan: TargetSparsePlan,
    selected_video: torch.Tensor,
    anchor_positions: torch.Tensor,
) -> torch.Tensor:
    if compact_video.ndim != 2 or int(compact_video.shape[0]) != plan.selected_video_row_count:
        raise RuntimeError("target-sparse compact video hidden shape does not match the selected-row plan")
    anchors = compact_video.index_select(0, anchor_positions)
    hidden = int(compact_video.shape[-1])
    anchors = anchors.reshape(plan.target_t, plan.source_patch_h, plan.source_patch_w, hidden)
    work = anchors.permute(0, 3, 1, 2).to(torch.float32)
    align_corners = plan.source_patch_h > 1 or plan.source_patch_w > 1
    lifted = F.interpolate(
        work,
        size=(plan.target_patch_h, plan.target_patch_w),
        mode="bilinear",
        align_corners=align_corners,
    )
    lifted = lifted.permute(0, 2, 3, 1).reshape(plan.target_video_rows, hidden).to(compact_video)
    # Retained rows are authoritative transformer outputs. Interpolation is only
    # a lifter for rows that were intentionally absent from the sparse stream.
    lifted.index_copy_(0, selected_video, compact_video)
    return lifted


def make_target_sparse_block_wrapper(
    layer: int,
    num_layers: int,
    metrics: H3FlowMetrics,
    previous=None,
):
    """Reduce target-video tokens for an early stage and restore them at the last block."""

    def call_next(args, extra):
        if previous is not None:
            return previous(args, extra)
        return extra["original_block"](args)

    def wrapper(args, extra):
        transformer = args.get("transformer_options", {}) or {}
        contract = transformer.get(TARGET_SPARSE_CONTRACT_KEY)
        if not isinstance(contract, dict) or contract.get("active") is not True:
            return call_next(args, extra)
        if int(contract.get("api", -1)) != TARGET_SPARSE_API_VERSION:
            raise RuntimeError("unsupported H3 target-sparse runtime contract")
        plan = contract.get("plan")
        if not isinstance(plan, TargetSparsePlan):
            raise RuntimeError("H3 target-sparse runtime contract is missing its plan")
        layout = args.get("layout")
        video_start, video_stop = _video_segment(layout)
        if video_stop - video_start != plan.target_video_rows:
            raise RuntimeError(
                "target-sparse plan/video layout mismatch: "
                f"plan={plan.target_video_rows} layout={video_stop - video_start}"
            )
        img = args.get("img")
        rope = args.get("rope_freqs")
        if not torch.is_tensor(img) or img.ndim != 2:
            raise RuntimeError("target-sparse H3 block wrapper requires a 2D packed hidden stream")
        if not torch.is_tensor(rope) or rope.ndim < 2 or int(rope.shape[1]) != video_stop:
            raise RuntimeError("target-sparse H3 block wrapper requires full target-grid RoPE rows")

        indices = _device_indices(contract, plan, img.device)
        selected_video = indices["selected_video"]
        full_prefix = torch.arange(video_start, device=img.device, dtype=torch.long)
        selected_global = torch.cat((full_prefix, video_start + selected_video))
        reduced_len = video_start + plan.selected_video_row_count
        if int(img.shape[0]) == video_stop:
            reduced_img = img.index_select(0, selected_global)
        elif int(img.shape[0]) == reduced_len:
            reduced_img = img
        else:
            raise RuntimeError(
                "target-sparse H3 hidden length "
                f"{int(img.shape[0])} is neither full {video_stop} nor reduced {reduced_len}"
            )

        new_transformer = dict(transformer)
        existing_vdn_contract = new_transformer.get(VDN_EXTERNAL_SEQUENCE_KEY)
        if existing_vdn_contract is not None:
            raise RuntimeError("target-sparse H3 path found an existing VDN external-sequence contract")
        new_transformer[VDN_EXTERNAL_SEQUENCE_KEY] = {
            "api": VDN_EXTERNAL_SEQUENCE_API_VERSION,
            "mode": VDN_EXTERNAL_SEQUENCE_MODE,
            "full_sequence_rows": video_stop,
            "reduced_sequence_rows": reduced_len,
        }

        new_args = dict(args)
        new_args["img"] = reduced_img
        new_args["rope_freqs"] = rope.index_select(1, selected_global.to(device=rope.device))
        new_args["mod_segments"] = _reduce_mod_segments(
            args.get("mod_segments"),
            video_start=video_start,
            video_stop=video_stop,
            selected_video=selected_video,
        )
        new_args["transformer_options"] = new_transformer
        output = call_next(new_args, extra)
        if not isinstance(output, dict) or "img" not in output or not torch.is_tensor(output["img"]):
            raise RuntimeError("target-sparse wrapped H3 block must return {'img': tensor}")
        result = output["img"]
        if result.ndim != 2 or int(result.shape[0]) != reduced_len:
            raise RuntimeError("target-sparse wrapped H3 block returned an unexpected hidden shape")

        if layer == 0:
            metrics.increment("target_sparse_actual_calls")
            metrics.event(
                "target_sparse_transformer",
                layer=layer,
                full_sequence_rows=video_stop,
                reduced_sequence_rows=reduced_len,
                full_video_rows=plan.target_video_rows,
                selected_video_rows=plan.selected_video_row_count,
                anchor_video_rows=plan.anchor_video_row_count,
                protected_video_rows=plan.protected_video_row_count,
                video_row_fraction=plan.video_row_fraction,
                source_hw=(plan.source_h, plan.source_w),
                target_hw=(plan.target_h, plan.target_w),
                target_t=plan.target_t,
                exact_protected_rows_retained=True,
                target_grid_rope_retained=True,
                vdn_external_sequence_mode=VDN_EXTERNAL_SEQUENCE_MODE,
            )

        if layer != num_layers - 1:
            return output

        compact_video = result[video_start:]
        lifted_video = _lift_video_hidden(
            compact_video,
            plan=plan,
            selected_video=selected_video,
            anchor_positions=indices["anchor_positions"],
        )
        full = torch.cat((result[:video_start], lifted_video), dim=0)
        if int(full.shape[0]) != video_stop:
            raise RuntimeError("target-sparse lifter failed to restore the full packed H3 sequence")
        metrics.event(
            "target_sparse_lift",
            layer=layer,
            restored_sequence_rows=int(full.shape[0]),
            restored_video_rows=plan.target_video_rows,
            retained_rows_overwritten_exactly=plan.selected_video_row_count,
            interpolation="bilinear_hidden",
        )
        restored = dict(output)
        restored["img"] = full
        return restored

    wrapper._h3_flow_target_sparse_wrapper = True
    wrapper._h3_flow_target_sparse_previous = previous
    wrapper._h3_flow_target_sparse_metrics = metrics
    return wrapper
