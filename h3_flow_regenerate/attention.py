from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .metrics import H3FlowMetrics


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    mode: str = "native"
    layers: tuple[int, ...] = (8, 16, 24, 32, 40)
    diagnostic_head: int = 0
    diagnostic_queries: int = 8
    sparse_window: int = 4
    global_heads: int = 8
    query_chunk: int = 64
    max_sequence: int = 8192
    vdn_chunk_size: int = 5
    vdn_chunk_radius: int = 1
    vdn_anchor_mode: str = "both"
    continuum_seam_anchor: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"native", "diagnostic", "vdn_reference_dense", "experimental_sparse"}:
            raise ValueError(f"unsupported H3 attention mode {self.mode!r}")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("H3 diagnostic/sparse layers must be non-negative")
        if self.diagnostic_head < 0 or self.diagnostic_queries < 1:
            raise ValueError("attention diagnostic selectors must be non-negative")
        if self.sparse_window < 1 or self.global_heads < 0 or self.query_chunk < 1:
            raise ValueError("sparse attention controls must be positive")
        if self.max_sequence < 1:
            raise ValueError("max_sequence must be positive")
        if self.vdn_chunk_size < 1 or self.vdn_chunk_radius < 0:
            raise ValueError("VDN chunk size must be positive and radius must be non-negative")
        if self.vdn_anchor_mode not in {"none", "columns", "rows", "both"}:
            raise ValueError(f"unsupported VDN anchor mode {self.vdn_anchor_mode!r}")


def layout_summary(layout: Any) -> dict[str, Any]:
    segments = tuple((int(a), int(b), str(kind)) for a, b, kind in layout.segments)
    counts: dict[str, int] = {}
    for start, stop, kind in segments:
        counts[kind] = counts.get(kind, 0) + (stop - start)
    signature = tuple(layout.signature)
    if not signature or signature[0] != "h3_flow_mixed_grid_v1":
        signature = tuple(int(v) for v in signature)
    return {
        "signature": signature,
        "segments": segments,
        "text_rows": counts.get("text", 0),
        "reference_rows": sum(counts.get(k, 0) for k in ("cond", "cond_audio", "ref_img", "ref_audio")),
        "audio_rows": counts.get("audio", 0),
        "video_rows": counts.get("video", 0),
        "sequence_rows": int(layout.seq_len),
    }


def video_local_mask(
    layout: Any,
    query_indices: torch.Tensor,
    *,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a query-chunk mask with global non-video and temporal video reach.

    Video-to-video keys are spatially local on H3's 2x2 patch grid while all
    temporal positions inside the local spatial column remain visible. Text,
    reference and audio keys remain global in both directions.
    """
    if radius < 1:
        raise ValueError("video attention radius must be positive")
    seq_len = int(layout.seq_len)
    query_indices = query_indices.to(device=device, dtype=torch.long)
    if bool((query_indices < 0).any()) or bool((query_indices >= seq_len).any()):
        raise IndexError("attention query index is outside the packed sequence")
    mask = torch.ones((query_indices.numel(), seq_len), dtype=torch.bool, device=device)
    va, vb, _ = next(segment for segment in layout.segments if segment[2] == "video")
    video_queries = (query_indices >= va) & (query_indices < vb)
    if not bool(video_queries.any()):
        return mask
    _, latent_t, latent_h, latent_w, _ = map(int, layout.signature)
    patch_h, patch_w = latent_h // 2, latent_w // 2
    expected = latent_t * patch_h * patch_w
    if vb - va != expected:
        raise ValueError("H3 video segment does not match the layout signature")
    query_local = query_indices[video_queries] - va
    q_spatial = query_local.remainder(patch_h * patch_w)
    qh = torch.div(q_spatial, patch_w, rounding_mode="floor")
    qw = q_spatial.remainder(patch_w)
    keys = torch.arange(expected, device=device)
    key_spatial = keys.remainder(patch_h * patch_w)
    kh = torch.div(key_spatial, patch_w, rounding_mode="floor")
    kw = key_spatial.remainder(patch_w)
    local = (qh[:, None] - kh[None, :]).abs().le(radius) & (qw[:, None] - kw[None, :]).abs().le(radius)
    mask[video_queries, va:vb] = local
    return mask


def vdn_chunk_bounds(num_frames: int, *, chunk_size: int = 5, radius: int = 1) -> tuple[tuple[int, int], ...]:
    """Return clamped chunk-aligned temporal bounds for each latent frame.

    This independently implements the public OpenVDN c1 topology: a query sees
    complete neighbouring chunks rather than a frame-centred partial window.
    """
    if num_frames < 1 or chunk_size < 1 or radius < 0:
        raise ValueError("VDN frame count/chunk size must be positive and radius non-negative")
    last = num_frames - 1
    return tuple(
        (
            max(0, (frame // chunk_size - radius) * chunk_size),
            min(last, (frame // chunk_size + radius + 1) * chunk_size - 1),
        )
        for frame in range(num_frames)
    )


def _video_layout(layout: Any) -> tuple[int, int, int, int]:
    va, vb, _ = next(segment for segment in layout.segments if segment[2] == "video")
    _, latent_t, latent_h, latent_w, _ = map(int, layout.signature)
    tokens_per_frame = (latent_h // 2) * (latent_w // 2)
    if latent_t < 1 or tokens_per_frame < 1 or vb - va != latent_t * tokens_per_frame:
        raise ValueError("H3 video segment does not match the layout signature")
    return int(va), int(vb), latent_t, tokens_per_frame


def _vdn_allowed_frames(
    frame: int,
    *,
    num_frames: int,
    bounds: tuple[tuple[int, int], ...],
    anchor_mode: str,
    seam_frame: int | None,
) -> set[int]:
    lo, hi = bounds[frame]
    allowed = set(range(lo, hi + 1))
    if anchor_mode in {"columns", "both"}:
        allowed.update((0, num_frames - 1))
    if anchor_mode in {"rows", "both"} and frame in {0, num_frames - 1}:
        allowed.update(range(num_frames))
    if seam_frame is not None:
        # The Continuum seam uses the same symmetric anchor rule as the first
        # and last boundary anchors: all video queries see the seam frame, and
        # seam-frame queries retain global video reach.
        allowed.add(seam_frame)
        if frame == seam_frame:
            allowed.update(range(num_frames))
    return allowed


def vdn_reference_mask(
    layout: Any,
    query_indices: torch.Tensor,
    *,
    chunk_size: int = 5,
    radius: int = 1,
    anchor_mode: str = "both",
    seam_frame: int | None = None,
    device: torch.device,
) -> torch.Tensor:
    """Build the exact dense-mask reference for VDN's temporal topology.

    Every pair involving a non-video token remains global. Video/video pairs
    use complete chunk-aligned temporal windows across all spatial positions,
    with optional first/last boundary anchors and a symmetric Continuum seam
    anchor. This is a correctness oracle, not a sparse-compute implementation.
    """
    if anchor_mode not in {"none", "columns", "rows", "both"}:
        raise ValueError(f"unsupported VDN anchor mode {anchor_mode!r}")
    seq_len = int(layout.seq_len)
    query_indices = query_indices.to(device=device, dtype=torch.long)
    if bool((query_indices < 0).any()) or bool((query_indices >= seq_len).any()):
        raise IndexError("attention query index is outside the packed sequence")
    va, vb, num_frames, tokens_per_frame = _video_layout(layout)
    if seam_frame is not None and not 0 <= seam_frame < num_frames:
        raise ValueError("Continuum seam frame is outside the H3 video segment")
    bounds = vdn_chunk_bounds(num_frames, chunk_size=chunk_size, radius=radius)
    mask = torch.ones((query_indices.numel(), seq_len), dtype=torch.bool, device=device)
    video_queries = (query_indices >= va) & (query_indices < vb)
    video_rows = torch.nonzero(video_queries, as_tuple=False).flatten()
    if video_rows.numel():
        query_frames = torch.div(
            query_indices[video_rows] - va,
            tokens_per_frame,
            rounding_mode="floor",
        )
        key_frames = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
        bound_tensor = torch.tensor(bounds, device=device, dtype=torch.long)
        lo = bound_tensor[query_frames, 0]
        hi = bound_tensor[query_frames, 1]
        allowed = (key_frames[None, :] >= lo[:, None]) & (key_frames[None, :] <= hi[:, None])
        if anchor_mode in {"columns", "both"}:
            allowed |= (key_frames == 0) | (key_frames == num_frames - 1)
        if anchor_mode in {"rows", "both"}:
            boundary_queries = (query_frames == 0) | (query_frames == num_frames - 1)
            allowed |= boundary_queries[:, None]
        if seam_frame is not None:
            allowed |= key_frames == seam_frame
            allowed |= (query_frames == seam_frame)[:, None]
        mask[video_rows, va:vb] = allowed
    return mask


def vdn_attention_density(
    layout: Any,
    *,
    chunk_size: int = 5,
    radius: int = 1,
    anchor_mode: str = "both",
    seam_frame: int | None = None,
) -> dict[str, int | float]:
    """Return exact allowed pair counts without materialising a square mask."""
    va, vb, num_frames, tokens_per_frame = _video_layout(layout)
    sequence = int(layout.seq_len)
    video_rows = vb - va
    global_rows = sequence - video_rows
    bounds = vdn_chunk_bounds(num_frames, chunk_size=chunk_size, radius=radius)
    allowed_pairs = global_rows * sequence
    for frame in range(num_frames):
        allowed_frames = _vdn_allowed_frames(
            frame,
            num_frames=num_frames,
            bounds=bounds,
            anchor_mode=anchor_mode,
            seam_frame=seam_frame,
        )
        allowed_keys = global_rows + len(allowed_frames) * tokens_per_frame
        allowed_pairs += tokens_per_frame * allowed_keys
    total_pairs = sequence * sequence
    return {
        "allowed_pairs": allowed_pairs,
        "total_pairs": total_pairs,
        "density": allowed_pairs / total_pairs,
    }


def _continuum_seam_frame(transformer: dict[str, Any], num_frames: int) -> tuple[int | None, str]:
    request = transformer.get("h3_continuum")
    if not isinstance(request, dict) or request.get("active") is not True:
        return None, "continuum_inactive"
    slots = request.get("protected_video_prefix_latent_slots")
    if isinstance(slots, bool) or not isinstance(slots, int):
        return None, "exact_prefix_slots_unavailable"
    if not 1 <= slots <= num_frames:
        return None, "exact_prefix_slots_invalid"
    return slots - 1, "exact_contract"


def _call_backend(backend, q, k, v, heads: int, kwargs: dict[str, Any], *, mask=None):
    call_kwargs = dict(kwargs)
    call_kwargs["_inside_attn_wrapper"] = True
    call_kwargs["mask"] = mask
    call_kwargs["skip_reshape"] = True
    return backend(q, k, v, heads, **call_kwargs)


def _sparse_attention(backend, q, k, v, heads, layout, config, kwargs):
    sequence = int(q.shape[-2])
    global_heads = min(config.global_heads, heads)
    pieces: list[torch.Tensor] = []
    if global_heads:
        global_out = _call_backend(
            backend,
            q[:, :global_heads],
            k[:, :global_heads],
            v[:, :global_heads],
            global_heads,
            kwargs,
        )
        pieces.append(global_out.reshape(q.shape[0], sequence, global_heads, q.shape[-1]))
    sparse_heads = heads - global_heads
    sparse_chunks: list[torch.Tensor] = []
    for start in range(0, sequence, config.query_chunk):
        stop = min(sequence, start + config.query_chunk)
        indices = torch.arange(start, stop, device=q.device)
        allowed = video_local_mask(layout, indices, radius=config.sparse_window, device=q.device)
        mask_value = -torch.finfo(q.dtype).max
        local_mask = torch.zeros(allowed.shape, dtype=q.dtype, device=q.device)
        local_mask.masked_fill_(~allowed, mask_value)
        out = _call_backend(
            backend,
            q[:, global_heads:, start:stop],
            k[:, global_heads:],
            v[:, global_heads:],
            sparse_heads,
            kwargs,
            mask=local_mask,
        )
        sparse_chunks.append(out.reshape(q.shape[0], stop - start, sparse_heads, q.shape[-1]))
    pieces.append(torch.cat(sparse_chunks, dim=1))
    return torch.cat(pieces, dim=2).reshape(q.shape[0], sequence, heads * q.shape[-1])


def _vdn_reference_attention(backend, q, k, v, heads, layout, config, kwargs, *, seam_frame):
    sequence = int(q.shape[-2])
    chunks: list[torch.Tensor] = []
    for start in range(0, sequence, config.query_chunk):
        stop = min(sequence, start + config.query_chunk)
        indices = torch.arange(start, stop, device=q.device)
        allowed = vdn_reference_mask(
            layout,
            indices,
            chunk_size=config.vdn_chunk_size,
            radius=config.vdn_chunk_radius,
            anchor_mode=config.vdn_anchor_mode,
            seam_frame=seam_frame,
            device=q.device,
        )
        additive_mask = torch.zeros(allowed.shape, dtype=q.dtype, device=q.device)
        additive_mask.masked_fill_(~allowed, -torch.finfo(q.dtype).max)
        out = _call_backend(
            backend,
            q[:, :, start:stop],
            k,
            v,
            heads,
            kwargs,
            mask=additive_mask,
        )
        chunks.append(out.reshape(q.shape[0], stop - start, heads, q.shape[-1]))
    return torch.cat(chunks, dim=1).reshape(q.shape[0], sequence, heads * q.shape[-1])


def make_attention_override(
    config: AttentionConfig,
    metrics: H3FlowMetrics,
    *,
    previous_override=None,
):
    def override(backend, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
        transformer = kwargs.get("transformer_options") or {}
        context = transformer.get("h3_flow_attention_context")
        if context is None or mask is not None or not skip_reshape:
            if previous_override is not None:
                return previous_override(backend, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
            return backend(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
        layer = int(context["layer"])
        layout = context["layout"]
        if layer not in config.layers or q.shape[-2] != int(layout.seq_len):
            if previous_override is not None:
                return previous_override(backend, q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
            return backend(q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
        if config.mode == "diagnostic":
            _record_attention_diagnostic(q, k, layout, layer, config, metrics, transformer)
            if previous_override is not None:
                return previous_override(backend, q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
            return backend(q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
        if config.mode == "vdn_reference_dense":
            if previous_override is not None:
                metrics.event("attention_fallback", layer=layer, reason="existing_attention_override")
                return previous_override(backend, q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
            sequence = int(q.shape[-2])
            if sequence > config.max_sequence:
                metrics.event("attention_fallback", layer=layer, reason="sequence_guard")
                return backend(q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
            _, _, num_frames, _ = _video_layout(layout)
            seam_frame, seam_status = (None, "disabled")
            if config.continuum_seam_anchor:
                seam_frame, seam_status = _continuum_seam_frame(transformer, num_frames)
            try:
                result = _vdn_reference_attention(
                    backend,
                    q,
                    k,
                    v,
                    heads,
                    layout,
                    config,
                    kwargs,
                    seam_frame=seam_frame,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                metrics.event(
                    "attention_fallback",
                    layer=layer,
                    reason="backend_rejected_vdn_reference_mask",
                    error=type(exc).__name__,
                )
                return backend(q, k, v, heads, mask=None, skip_reshape=True, **kwargs)
            density = vdn_attention_density(
                layout,
                chunk_size=config.vdn_chunk_size,
                radius=config.vdn_chunk_radius,
                anchor_mode=config.vdn_anchor_mode,
                seam_frame=seam_frame,
            )
            metrics.event(
                "attention_vdn_reference",
                layer=layer,
                sequence=sequence,
                chunk_size=config.vdn_chunk_size,
                chunk_radius=config.vdn_chunk_radius,
                anchor_mode=config.vdn_anchor_mode,
                seam_frame=seam_frame,
                seam_status=seam_status,
                implementation="dense_additive_mask",
                acceleration_claim=False,
                **density,
            )
            return result
        if config.mode != "experimental_sparse":
            return backend(q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
        if previous_override is not None:
            metrics.event("attention_fallback", layer=layer, reason="existing_attention_override")
            return previous_override(backend, q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
        sequence = int(q.shape[-2])
        if sequence > config.max_sequence or config.global_heads >= heads:
            reason = "sequence_guard" if sequence > config.max_sequence else "all_heads_global"
            metrics.event("attention_fallback", layer=layer, reason=reason)
            return backend(q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
        try:
            result = _sparse_attention(backend, q, k, v, heads, layout, config, kwargs)
        except (RuntimeError, TypeError, ValueError) as exc:
            metrics.event(
                "attention_fallback",
                layer=layer,
                reason="backend_rejected_sparse_mask",
                error=type(exc).__name__,
            )
            return backend(q, k, v, heads, mask=None, skip_reshape=True, **kwargs)
        metrics.event(
            "attention_sparse",
            layer=layer,
            sequence=sequence,
            global_heads=min(config.global_heads, heads),
            local_heads=heads - min(config.global_heads, heads),
            window=config.sparse_window,
        )
        return result

    override._h3_flow_attention_override = True
    override._h3_flow_previous_override = previous_override
    override._h3_flow_attention_config = config
    override._h3_flow_metrics = metrics
    return override


def _record_attention_diagnostic(q, k, layout, layer, config, metrics, transformer) -> None:
    head = min(config.diagnostic_head, q.shape[1] - 1)
    va, vb, num_frames, tokens_per_frame = _video_layout(layout)
    count = min(config.diagnostic_queries, vb - va)
    indices = torch.linspace(va, vb - 1, count, device=q.device).round().long()
    logits = torch.matmul(q[0, head, indices].float(), k[0, head].float().transpose(0, 1)) / math.sqrt(q.shape[-1])
    probabilities = logits.softmax(dim=-1)
    seam_frame, seam_status = (None, "disabled")
    if config.continuum_seam_anchor:
        seam_frame, seam_status = _continuum_seam_frame(transformer, num_frames)
    allowed = vdn_reference_mask(
        layout,
        indices,
        chunk_size=config.vdn_chunk_size,
        radius=config.vdn_chunk_radius,
        anchor_mode=config.vdn_anchor_mode,
        seam_frame=seam_frame,
        device=q.device,
    )
    retained_mass = (probabilities * allowed).sum(dim=-1)
    outside_mass = (probabilities * ~allowed).sum(dim=-1)
    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1).mean()
    masses: dict[str, float] = {}
    for a, b, kind in layout.segments:
        masses[kind] = masses.get(kind, 0.0) + float(probabilities[:, a:b].sum(dim=-1).mean().item())
    density = vdn_attention_density(
        layout,
        chunk_size=config.vdn_chunk_size,
        radius=config.vdn_chunk_radius,
        anchor_mode=config.vdn_anchor_mode,
        seam_frame=seam_frame,
    )
    boundary_mass = {
        "first": float(probabilities[:, va : va + tokens_per_frame].sum(dim=-1).mean().item()),
        "last": float(probabilities[:, vb - tokens_per_frame : vb].sum(dim=-1).mean().item()),
    }
    seam_mass = None
    if seam_frame is not None:
        seam_start = va + seam_frame * tokens_per_frame
        seam_mass = float(probabilities[:, seam_start : seam_start + tokens_per_frame].sum(dim=-1).mean().item())
    metrics.event(
        "attention_diagnostic",
        layer=layer,
        head=head,
        queries=count,
        query_scope="video",
        entropy=float(entropy.item()),
        modality_mass=masses,
        vdn_retained_mass_mean=float(retained_mass.mean().item()),
        vdn_retained_mass_min=float(retained_mass.min().item()),
        vdn_outside_mass_mean=float(outside_mass.mean().item()),
        vdn_chunk_size=config.vdn_chunk_size,
        vdn_chunk_radius=config.vdn_chunk_radius,
        vdn_anchor_mode=config.vdn_anchor_mode,
        boundary_mass=boundary_mass,
        continuum_seam_frame=seam_frame,
        continuum_seam_status=seam_status,
        continuum_seam_mass=seam_mass,
        output_neutral=True,
        **density,
    )


def make_layout_block_wrapper(
    layer: int,
    metrics: H3FlowMetrics,
    previous=None,
    *,
    record_layout: bool = True,
):
    def wrapper(args, extra):
        transformer = args["transformer_options"]
        old = transformer.get("h3_flow_attention_context")
        transformer["h3_flow_attention_context"] = {"layout": args["layout"], "layer": layer}
        if layer == 0 and record_layout:
            metrics.event("packed_layout", **layout_summary(args["layout"]))
        try:
            if previous is not None:
                return previous(args, extra)
            return extra["original_block"](args)
        finally:
            if old is None:
                transformer.pop("h3_flow_attention_context", None)
            else:
                transformer["h3_flow_attention_context"] = old

    return wrapper


def mark_layout_wrapper(
    wrapper,
    *,
    metrics: H3FlowMetrics,
    previous=None,
    scope: str = "layout",
):
    if scope not in {"layout", "attention"}:
        raise ValueError(f"unsupported H3 flow layout-wrapper scope {scope!r}")
    wrapper._h3_flow_layout_wrapper = True
    wrapper._h3_flow_layout_scope = scope
    wrapper._h3_flow_metrics = metrics
    wrapper._h3_flow_previous = previous
    return wrapper
