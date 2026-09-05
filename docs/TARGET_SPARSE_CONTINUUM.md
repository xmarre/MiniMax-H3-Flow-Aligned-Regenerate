# Target-Sparse Continuum continuation

**MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** is retained as a research/control path. It is **not the recommended quality path** in v0.3.0.

## What it does

Target-Sparse keeps the caller's sampler latent, mask and protected prefix on the final target grid. During the early stage it reduces only the generated-video hidden-token stream while retaining:

- every protected video row;
- all text/reference/audio rows;
- native target-grid RoPE coordinates for retained rows;
- target-grid sampler state and mask semantics.

The reduced hidden field is lifted back to the full target grid before H3's native final layer and before Spectrum's final-block observation. A fresh full-grid high-stage sampler lifetime follows.

This is a hidden-stream approximation, not a real lower-resolution sampler suffix.

## Why it is not promoted

Real decoded-media testing showed poor quality on this path. Because Target-Sparse does **not** perform the learned latent upscale used by the Mixed-Grid path, early approximation errors can survive the high stage and cascade into later continuation.

Observed failures included:

- skin imperfections;
- odd or unstable clothing details;
- spurious background additions;
- defects that propagated across continuation rather than remaining isolated to one frame.

That negative result is sufficient to keep Target-Sparse as an architectural experiment/control rather than requiring another acceptance sweep. For exact-prefix Continuum acceleration where output quality matters, use **Progressive Mixed-Grid Continuum**, which performs real low-grid suffix sampling followed by the learned 3D latent upscale and fresh target-grid refinement.

## Suffix DC bridge

Target-Sparse exposes the same narrow `suffix_dc_bridge` control on its dedicated Continuum node.

Unlike Mixed-Grid, it has no learned-upscaler prefix from which to derive the correction. The bridge therefore calibrates from the first **actual** H3 predicted-clean output of the fresh full-grid high stage before native mask restoration.

The contract remains:

- canonical whole-frame exact prefix only;
- one generated suffix latent token corrected;
- per-channel spatial-mean/DC offset;
- fixed weight `1.0`;
- protected prefix unchanged;
- later suffix tokens unchanged at bridge application;
- forecasts are not used for calibration;
- no extra H3 NFE.

This bridge placement is structurally tested, but it does not change the broader Target-Sparse quality conclusion above.

## VDN-H3

Target-Sparse retains the existing API-1-compatible external reduced-sequence contract. This is separate from the API-2 `mixed_grid_low_suffix` contract used by the recommended Mixed-Grid path.

## Status

Keep this node for controlled sparse-row experiments, profiling and comparison. Do not treat structural test success as evidence that it matches the decoded quality of the learned-upscale Mixed-Grid path.
