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

    def __post_init__(self) -> None:
        if self.mode not in {"native", "diagnostic", "experimental_sparse"}:
            raise ValueError(f"unsupported H3 attention mode {self.mode!r}")
        if any(layer < 0 or layer >= 50 for layer in self.layers):
            raise ValueError("H3 diagnostic/sparse layers must be in [0, 49]")
        if self.diagnostic_head < 0 or self.diagnostic_queries < 1:
            raise ValueError("attention diagnostic selectors must be non-negative")
        if self.sparse_window < 1 or self.global_heads < 0 or self.query_chunk < 1:
            raise ValueError("sparse attention controls must be positive")
        if self.max_sequence < 1:
            raise ValueError("max_sequence must be positive")


def layout_summary(layout: Any) -> dict[str, Any]:
    segments = tuple((int(a), int(b), str(kind)) for a, b, kind in layout.segments)
    counts = {kind: b - a for a, b, kind in segments}
    signature = tuple(int(v) for v in layout.signature)
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
        local_mask = video_local_mask(layout, indices, radius=config.sparse_window, device=q.device)
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
            return backend(q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
        if config.mode == "diagnostic":
            _record_attention_diagnostic(q, k, layout, layer, config, metrics)
            return backend(q, k, v, heads, mask=mask, skip_reshape=True, **kwargs)
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

    return override


def _record_attention_diagnostic(q, k, layout, layer, config, metrics) -> None:
    head = min(config.diagnostic_head, q.shape[1] - 1)
    sequence = q.shape[-2]
    count = min(config.diagnostic_queries, sequence)
    indices = torch.linspace(0, sequence - 1, count, device=q.device).round().long()
    logits = torch.matmul(q[0, head, indices].float(), k[0, head].float().transpose(0, 1)) / math.sqrt(q.shape[-1])
    probabilities = logits.softmax(dim=-1)
    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1).mean()
    masses: dict[str, float] = {}
    for a, b, kind in layout.segments:
        masses[kind] = masses.get(kind, 0.0) + float(probabilities[:, a:b].sum(dim=-1).mean().item())
    metrics.event(
        "attention_diagnostic",
        layer=layer,
        head=head,
        queries=count,
        entropy=float(entropy.item()),
        modality_mass=masses,
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


def mark_layout_wrapper(wrapper, *, metrics: H3FlowMetrics):
    wrapper._h3_flow_layout_wrapper = True
    wrapper._h3_flow_metrics = metrics
    return wrapper
