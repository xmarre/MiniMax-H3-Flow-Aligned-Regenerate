# Target-Sparse Exact-Prefix Continuum

This document defines the experimental **MiniMax H3 Progressive Target-Sparse Continuum** path. It exists to address a limitation of the conservative Target Input progressive node: Native Masked continuation chunks contain exact protected video rows, so resizing the whole early latent would alter the context seen by generated rows. The conservative implementation therefore falls back to one ordinary target-grid sampler lifetime for those chunks.

That fallback is semantically safe, but it means chunk 2 and later normally receive no progressive acceleration. The target-sparse path is an attempt to reduce continuation-chunk compute without weakening the exact target-grid prefix contract.

## Status

The implementation is **experimental**.

What is established structurally:

- the sampler latent, denoise mask, target geometry, and protected video values are never spatially resized on an exact-prefix chunk;
- exact protected H3 video patch rows are detected using the same 2x2 `amax` mask semantics as native H3;
- every non-video packed row is retained;
- every exact protected target-video row is retained;
- generated video rows are represented by a regular target-grid anchor lattice during the early transformer stage;
- retained rows keep their original target-grid RoPE rows;
- per-row H3 modulation metadata is reduced consistently with the selected video rows;
- the last H3 block lifts the generated-video hidden field back to the complete target grid before the native final layer;
- retained transformer outputs overwrite lifted/interpolated values exactly;
- Spectrum's final-block observation receives the restored full target hidden stream;
- the early and late sampler/Spectrum histories are separate lifetimes;
- the first late/full-grid H3 call is required to be actual;
- a late-stage failure invalidates the committed early trajectory instead of leaving stale guidance state reusable;
- when a compatible VDN-H3 object patch is present, target-sparse publishes an explicit reduced-sequence contract instead of letting VDN interpret the irregular sparse rows as its normal full frame grid;
- incompatible/older VDN-H3 object patches fail at node-patch time rather than crashing after expensive sampling work has begun.

What is **not** established yet:

- decoded-video quality parity with the ordinary target-grid continuation path;
- seam quality across 2+ Native Masked chunks;
- audio/video perceptual parity;
- net wall-time speedup after sparse-selection/lifting overhead;
- an optimal anchor density;
- quality when combined with all optional external patches;
- net speed/quality of the VDN-H3 reduced-sequence compatibility mode;
- behavior on long 4/8+ chunk Continuum sequences.

Passing CI is only structural evidence. Do not promote this mode based on CI alone.

## Why the old low-grid continuation approach is invalid

For an ordinary unprotected Target Input call, Flow can privately derive a lower spatial video grid, run the early schedule there, perform an exact handoff probe, transfer the predicted-clean video state to the target grid, and continue at target resolution.

A Native Masked continuation chunk is different. Its previous accepted tail is copied into the beginning of the new target latent and protected by a video denoise mask with zero-valued rows. Those values are the authoritative continuation context.

If Flow downsamples that prefix for an early low-grid stage, the generated suffix attends to an approximation of the previous chunk. Copying the pristine prefix back later cannot undo the fact that the suffix was denoised while seeing altered context.

Therefore the invariant is:

> An exact Native Masked video prefix must remain represented at its target-grid latent values and target-grid H3 coordinates throughout any stage that uses it as conditioning context.

The conservative Target Input node satisfies this by abandoning the low-grid handoff on exact-prefix chunks. Target-sparse instead keeps the target-grid sampler representation and reduces only internal transformer rows.

## Execution topology

### Chunk 1 / unprotected call

There is no exact protected continuation prefix, so the experimental node uses the existing Target Input path:

```text
private low grid
  -> exact handoff probe
  -> bicubic or learned_3d clean-video transfer
  -> target-grid conditional state
  -> fresh target-grid sampler lifetime
```

The learned latent-upscaler provider is relevant here when `handoff_transfer=learned_3d`.

### Exact-prefix chunk

The sampler itself remains target-sized:

```text
full target latent + full target noise + native mask
                       |
                       v
                early sampler lifetime
                       |
              H3 packed hidden stream
                       |
      +----------------+----------------+
      |                                 |
all text/ref/audio rows             target video rows
      |                                 |
    keep all                  keep every exact protected row
                              + coarse generated anchor lattice
                                        |
                                        v
                             reduced transformer sequence
                                        |
                             final H3 block output
                                        |
                           bilinear hidden-space lifter
                                        |
                          overwrite every retained row
                            with its exact block output
                                        |
                                        v
                             full target hidden stream
                                        |
                           native H3 final output layer
                                        |
                                        v
                             handoff sampler state
                                        |
                     fresh full-transformer sampler lifetime
```

There is no spatial sampler-state transfer and no exact x0 probe at this boundary because geometry does not change.

## Anchor density

On an exact-prefix target-sparse chunk, the ordinary Target Input source controls are reinterpreted as the density of generated-region transformer anchors:

- `source_mode=scale`: `source_scale` determines a coarse H/W lattice relative to the target H/W;
- `source_mode=pixels`: the explicit source dimensions are converted to safe latent H/W and determine the coarse anchor lattice.

These dimensions do **not** become sampler latent dimensions on exact-prefix chunks.

The anchor lattice is built independently for every target temporal slot. Spatial positions are selected from the actual target patch grid, so retained rows use native target-grid RoPE rather than synthetic low-grid coordinates.

Protected rows are unioned with the anchor lattice. If a protected row is not one of the regular anchors, it is still retained and its transformer output remains authoritative.

## Hidden-space lifting

Only the regular generated-region anchor lattice is used as the interpolation basis. Protected rows are not allowed to distort the coarse lattice geometry.

After the last H3 transformer block:

1. anchor hidden states are reshaped to `[T, H_anchor, W_anchor, hidden]`;
2. each temporal slot is bilinearly lifted to the full target patch grid;
3. every retained row, including every protected row, is copied back from the exact sparse-transformer output;
4. the complete packed hidden stream is reconstructed;
5. H3's native final layer consumes the full target audio/video ranges.

This does not make the early transformer semantically equivalent to full dense attention. Missing generated rows were absent from early attention and are approximated by the hidden lifter.

## Spectrum composition

Spectrum wraps native H3's first/last block replacements for actual feature capture. The target-sparse block wrapper composes with those replacements rather than replacing them destructively.

The important boundary is the final H3 block: target-sparse restores the complete target hidden stream before returning from the existing block replacement. Spectrum's last-block capture therefore observes the normal full target `[audio | video]` row range, not a reduced sequence.

At the target-sparse handoff, a new sampler/Spectrum lifetime begins and the existing `h3_refinement` contract requests an actual first full-grid call. Sparse-stage history is never reused as if it were full-grid history.

## DiffAid and Untwisting RoPE

All non-video rows are retained, so text and reference row indices before the target-video tail remain stable.

For H3 DiffAid block replacements, target-sparse passes reduced `mod_segments`: non-video segments are unchanged and per-row target-video modulation indices are selected with the same video-row index set as the hidden stream.

Untwisting RoPE reference rows are before the target-video tail and remain present. Retained target-video rows use exact subsets of the original target RoPE table.

These composition rules are source-contract checked against pinned sibling revisions, but the complete stack still needs a real decoded-media run.

## VDN-H3 composition

VDN-H3 normally publishes one regular packed-video geometry before the DiT blocks run: one `video_start:video_end` span with `num_frames * tokens_per_frame` rows. Its local softmax windows, retained activation scratch, spatial short-convolution, and bidirectional linear complement all rely on that uniform frame-major geometry.

Target-sparse deliberately violates that representation **inside the early transformer only**. Exact-protected Continuum frames can retain all target-grid spatial rows while generated regions retain only the coarse anchor lattice. That irregular row set cannot safely be relabeled as a smaller ordinary VDN frame grid.

A real runtime failure exposed this distinction: VDN published the full target video span, while target-sparse passed fewer selected video rows into the first block. The retained VDN scratch expected the full video-row count and failed while copying the reduced Q/K/V rows. Merely shrinking that scratch would not fix the algorithm, because the VDN window and linear branches would still interpret the irregular rows using the wrong frame geometry.

The interoperability contract therefore has deliberately narrow semantics:

```text
target-sparse early block
  -> reduced packed rows + exact target-grid RoPE subset
  -> VDN external-sequence contract v1
  -> dense softmax over the retained rows
  -> preserve the learned VDN softmax gate
  -> disable VDN's geometry-dependent local-window / linear-complement branch
  -> target-sparse hidden lifter
  -> full target packed rows restored

late full-grid stage
  -> ordinary VDN-H3 path again
```

The contract mode is `dense_gate_no_linear`. It is intentionally explicit and fail-closed:

- Flow publishes exact `full_sequence_rows` and `reduced_sequence_rows` for every reduced block call;
- VDN verifies those counts and the reduced RoPE length before running;
- VDN advertises the supported external-sequence API on its object-patched attention forward;
- Flow refuses to enable target-sparse with an older VDN object patch that does not advertise that capability;
- an unrelated object patch is not treated as VDN.

This compatibility mode is structurally safer than pretending the irregular rows form a normal VDN grid, but it is **not** a claim that VDN and target-sparse speedups multiply. Dense attention over the retained stream has quadratic cost, while normal VDN uses local windows plus its trained linear complement. Depending on anchor density and sequence size, the compatibility mode can be faster, neutral, or slower. Only matched GPU timing can answer that.

It is also not semantically identical to normal VDN: the trained linear far-field branch is absent during the reduced early stage. The mode follows VDN's own full-cover topology—dense softmax, learned softmax gate, no linear complement—but applies it to Flow's retained-row stream. Decoded-media validation is therefore mandatory.

## Flow Attention Lab interaction

The Attention Lab's H3 layout diagnostics use the authoritative full packed layout. During target-sparse early transformer calls the hidden sequence is shorter than that full layout. Existing guarded attention paths therefore must not pretend the reduced sequence is the full native layout.

Target-sparse does not claim to combine its row reduction with the Attention Lab's experimental sparse/VDN output-changing attention modes. Treat those as separate experiments unless a specific reduced-layout attention contract is implemented and validated.

## Shared suffix DC bridge

Target-Sparse exposes the same `suffix_dc_bridge` control as the other Continuum progressive nodes and defaults it to **on**. It does not alter the sparse early transformer stage. The correction is installed only on the fresh full-grid high-stage sampler lifetime, after the hidden-space lifter boundary.

For a canonical whole-frame exact prefix, Flow snapshots the authoritative model-domain prefix and waits for the first **actual** high-stage H3 predicted-clean output. It compares the per-channel spatial mean of that output's last predicted prefix token with the authoritative last prefix token and applies the resulting constant offset only to the first generated suffix token. Native masking subsequently restores the protected prefix as usual. Later suffix tokens are untouched.

This composes with the existing target-sparse invariant that the first high-stage call must be actual. If the bridge is requested on a canonical boundary but no actual high-stage H3 evaluation consumes it, runtime fails rather than silently claiming success. Noncanonical masks emit a skip metric and keep the existing sampling behavior.

The bridge itself adds no H3 NFE. It is a seam correction, not an acceleration mechanism, and does not change the requirement to validate target-sparse decoded-media quality and timing independently.

## Metrics to verify

A successful exact-prefix target-sparse chunk should show:

- `handoff_plan.input_mode = target_grid_sparse`;
- `source_latent_resize_performed = false` in `handoff_complete`;
- `learned_transfer_performed = false` for that exact-prefix chunk;
- `exact_probe_performed = false`;
- `sampler_invocation_count = 2`;
- `history_boundary_count = 1`;
- one or more `target_sparse_transformer` events during the early lifetime;
- `selected_video_rows < full_video_rows` when the configuration actually reduces work;
- `exact_protected_rows_retained = true`;
- `target_grid_rope_retained = true`;
- a `target_sparse_lift` event before full-grid output;
- the first late/full-grid model call marked actual.

With VDN-H3 enabled, also require the runtime to report its external reduced-sequence compatibility path for the sparse early stage, and verify that normal VDN resumes on the late/full-grid stage.

If `selected_video_rows == full_video_rows`, the sparse stage provides no video-row reduction and should not be counted as an acceleration result.

## Required runtime acceptance gate

Use ordinary SA-Solver-PECE without RefDelta for the first gate so RefDelta scheduling/jitter is not conflated with this mechanism.

### Gate A: two chunks

Run a matched pair with identical prompt, seed, target geometry, Continuum settings, sampler, Spectrum policy, references, VAE, and other patches:

1. conservative **Progressive Handoff (Target Input)** control;
2. experimental **Progressive Target-Sparse Continuum**.

Verify:

- chunk 1 follows the same normal progressive path in both runs;
- chunk 2 control records `progressive_target_fallback`;
- chunk 2 experimental records target-sparse events and a genuinely reduced early sequence;
- no duplicated/missing Continuum overlap after assembly;
- no new cut/seam, protected-prefix corruption, motion jump, audio shift, or reference-identity regression;
- per-chunk and whole-workflow wall times are recorded separately.

If VDN-H3 is part of the tested workflow, additionally require:

- the sparse early stage enters VDN `dense_gate_no_linear` rather than the normal full-grid VDN geometry path;
- no full-vs-reduced row-count mismatch occurs;
- the late/full-grid stage returns to normal VDN;
- continuation-chunk timing is compared against the same VDN-enabled conservative control. A target-sparse result without the same VDN baseline does not establish incremental speedup.

### Gate B: long sequence

If Gate A passes, run at least 4 chunks. Chunks 2, 3, and 4 must all execute target-sparse early stages rather than falling back. Compare seam quality and per-chunk timing against the conservative control.

A 2-chunk speedup is not enough evidence for long Continuum sequences. The purpose of this path is specifically to prevent the chunk-1-only acceleration from being amortized away.

### Promotion criteria

Do not make this the default until all of these are true:

- structural CI green on supported Python versions;
- pinned source-contract checks green;
- matched 2-chunk decoded-media gate passes;
- matched 4+ chunk decoded-media gate passes;
- target-sparse is actually used on every continuation chunk;
- measured continuation-chunk wall time improves enough to exceed sparse/lifter overhead;
- no material seam, motion, anatomy, audio, or reference-fidelity regression is observed;
- when VDN-H3 is enabled, its reduced-sequence compatibility mode passes the same decoded-media and wall-time gates instead of being inferred correct from shape tests alone.
