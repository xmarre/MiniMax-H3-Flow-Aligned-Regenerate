"""Exact visual reference context shared by native continuation stages."""

from __future__ import annotations

POLICY = "uniform_video_exact_native_reference_all_stages_v3"


def add_exact_prefix_visual_context(native, layout, payload, carrier, exact):
    """Append an exact prefix clip using Core's native reference timeline.

    Core accepts independent reference-video H/W. Its ordinary reference layout
    places references before the target timeline and advances target audio,
    video and keyframes together. Keep that complete layout: manually aligning
    a differently sized reference with the protected video prefix changes the
    native conditioning semantics. These rows use native visual-condition
    augmentation/modulation and never become generated video.
    """
    if (
        exact.prefix_t != carrier.prefix_t
        or exact.temporal != carrier.temporal
        or (carrier.source_h, carrier.source_w) not in ((exact.source_h, exact.source_w), exact.target_hw)
        or carrier.target_hw != (carrier.source_h, carrier.source_w)
    ):
        raise RuntimeError(
            "exact visual prefix requires matching temporal ownership and a uniform source or target carrier"
        )
    audio_start, _, _ = next(segment for segment in layout.segments if segment[2] == "audio")
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
    # Preserve Core/VDN's native geometry cache contract. Flow extends the block
    # layout signature separately with this policy and exact prefix geometry.
    if extended.signature != layout.signature:
        raise RuntimeError("exact visual prefix changed the native carrier signature")
    conditioned = dict(payload)
    conditioned["refs"] = refs
    conditioned["cond_video_latents"] = [*(payload.get("cond_video_latents") or ()), exact.prefix]
    conditioned["layout"] = extended
    return extended, conditioned, (audio_start, audio_start + rows)
