"""Exact visual context for a single native reduced-grid video trajectory."""

from __future__ import annotations

import torch

POLICY = "uniform_video_exact_aligned_visual_prefix_v1"


def add_exact_prefix_visual_context(native, layout, payload, carrier, exact):
    """Add native reference rows without changing existing RoPE positions.

    Core accepts independent reference-video H/W. Its ordinary reference layout
    puts references before the target timeline; here the protected prefix is
    already part of that timeline. Restore the original positions and colocate
    these context rows with the corresponding prefix frames. They use native
    visual-condition augmentation/modulation and never become generated video.
    """
    if (
        exact.prefix_t != carrier.prefix_t
        or exact.temporal != carrier.temporal
        or (exact.source_h, exact.source_w) != (carrier.source_h, carrier.source_w)
        or carrier.target_hw != (carrier.source_h, carrier.source_w)
        or exact.target_hw == carrier.target_hw
    ):
        raise RuntimeError("exact visual prefix requires matching temporal ownership and a uniform reduced carrier")
    audio_start, _, _ = next(segment for segment in layout.segments if segment[2] == "audio")
    video_start, _, _ = layout.segments[-1]
    text_len = layout.segments[0][1]
    audio_t = (layout.segments[-2][1] - audio_start) // 2
    refs = [*(payload.get("refs") or ())]
    refs.append(
        {
            "kind": "video",
            "latent": exact.prefix,
            "latent_t": exact.prefix_t,
            "latent_h": exact.target_hw[0],
            "latent_w": exact.target_hw[1],
            "ref_audio_t": 0,
        }
    )
    extended = native.PackedLayout(
        text_len,
        carrier.temporal,
        carrier.source_h,
        carrier.source_w,
        audio_t,
        keyframes=payload.get("keyframes"),
        refs=refs,
    )
    rows = exact.prefix_rows
    expected_segments = [
        (first + (rows if first >= audio_start else 0), last + (rows if first >= audio_start else 0), kind)
        for first, last, kind in layout.segments
    ]
    expected_segments.insert(len(expected_segments) - 2, (audio_start, audio_start + rows, "ref_img"))
    if extended.seq_len != layout.seq_len + rows or extended.segments != expected_segments:
        raise RuntimeError("native exact-prefix visual layout changed existing conditioning ownership")
    frame, _ = native._frame_grid(*exact.target_hw)
    positions = native._video_grid(exact.prefix_t, frame, float(layout.position_ids[video_start, 0]))
    extended.position_ids = torch.cat((layout.position_ids[:audio_start], positions, layout.position_ids[audio_start:]))
    # Preserve Core/VDN's native geometry cache contract. Flow extends the block
    # layout signature separately with this policy and exact prefix geometry.
    if extended.signature != layout.signature:
        raise RuntimeError("exact visual prefix changed the native carrier signature")
    conditioned = dict(payload)
    conditioned["refs"] = refs
    conditioned["cond_video_latents"] = [*(payload.get("cond_video_latents") or ()), exact.prefix]
    conditioned["layout"] = extended
    return extended, conditioned, (audio_start, audio_start + rows)
