# Native source-prefix carry

Native Source Continuation uses the preceding chunk's already-computed low/probe
clean prediction as source context. For a single all-generated chunk, ordinary
progressive learned transfer computes
this prediction without requiring prior context. Continuum Native Masked uses
it for the next chunk only after exact target-suffix pairing. Slight continuation
darkening remains a known issue; native carry does not guarantee a tone fix.

## Setup

1. Use **MiniMax H3 Native Source Continuation** with VDN-H3-Plus and the learned
   3D upscaler. Add Continuum Native Masked for a sequence.
2. Generate the full sequence with the same patched MODEL, starting with the
   initial chunk. Native carry with target-band reconciliation and guarded
   detail transport are the new node's defaults. Saved workflows keep their
   stored projection value. A cached initial chunk from another model or input policy is insufficient.
3. Preserve the authoritative returned target suffix when supplying the next
   prefix. No saved audit bundle or video VAE is a runtime input.
4. Optional boundary capture saves source/stage operands for offline native-VAE
   replay. Judge tone and audiovisual continuity on rendered output.

The legacy node's existing default and stored widget values remain unchanged.
It can select native carry explicitly; its VAE roundtrip reconstruction control
is retained for compatibility. Live frozen A/B/A capture and kernel observers
have been retired. Saved bundles remain readable.

## State and numerical contract

The initial generation path records its already-computed source clean prediction
at the handoff sigma and pairs it with suffix hashes of the target video returned
by that sampling transaction. A continuation requires an exact float32 hash
match between its authoritative target prefix and the preceding returned target
suffix. Target and source grids must match the recorded grids, both clips must
use native `5k+2` temporal lengths, and the prefix must end on the same native
phase. When Continuum sequence metadata is present, chunk indices must advance
consecutively. Explicit session IDs, when supplied, must also match. The native
Continuum request does not supply a session ID: its receipt leaves that identity
unset, rather than claiming verified session provenance. Continuum's initial Flow
pass is unlabeled; the first labeled request (chunk 2) can bind that initial
pair by its exact target-suffix hash. An unlabeled continuation or skipped chunk
cannot use this exception. Later requests must advance the labeled chunk order.

The matching tail of that source clean prediction replaces only the protected
source prefix before low sampling. Low/probe and learned transfer reuse those
bytes. The generated source suffix retains its existing initialization and
sampling path. The authoritative target prefix, protected audio, masks, sigma
schedule, three sampler lifetimes and selected backend contracts are preserved.
Generated audio can change through joint AV conditioning and needs listening.

No additional H3, provider or VAE call is introduced. This carries a **clean
prediction at a nonzero handoff sigma**, not a fully sampled low-resolution
video. It can disagree with the high-refined target's pose, texture or tone;
rendered seam quality remains an empirical question.

Missing previous context, a changed prefix, skipped chunk or mismatched explicit
session, changed
geometry or invalid temporal phase stops before the first low sampler. There is
no silent bicubic fallback. Latent processing between generation and continuation
that changes the carried target bytes will fail the pairing check. This includes
post-sampling refinement that changes the authoritative prefix.

## Target-band reconciliation

`source_prefix_projection=native_source_carry_target_band` is the default for
new Native Source Continuation nodes; `native_source_carry` remains selectable.
Pairing, ownership and failure behavior match native carry. Only the latent bytes
supplied as the reduced-grid protected prefix change.

In user-reviewed rendered continuations, including a scene with a chain-link
fence and vegetation, it removed the visible boundary and the background redraw
a few frames after the join that plain carry showed. These renders were not
matched same-seed comparisons, and they do not establish the latent limits below
as harmless in every scene.

### What the reduced-grid stage sees

H3 receives protected prefix rows as clean conditioning. Core labels them with
the visual-condition timestep and re-injects them at every step. Generated rows
attend to these rows from the first sampler step onward. The reduced-grid stage
runs the highest-noise steps.

With native carry, those rows are the preceding chunk's low/probe clean
prediction at the handoff sigma, roughly 0.87 for the default handoff
coordinate. Target-high refinement of that chunk starts at this sigma and can
still change background content. The reduced-grid stage of the next chunk
therefore conditions on an earlier estimate of the scene, not on the returned
video. The target-high stage then receives the exact returned prefix. Its
direction guidance pulls generated tokens toward the reduced-grid reference,
below the low-frequency cutoff. When the two versions disagree, generated
content can follow each version in different parts of the suffix. This is a
code-path hypothesis for scene changes a few frames after a join. It has not
been isolated with trained weights.

### Construction

Let `C` be the carried prefix and `G` the returned target prefix resized to the
source grid. `G` uses the existing half-pixel latent resize, the same operation
`latent_bicubic` uses. Per frame and channel, with `g_s` a mirror-padded spatial
Gaussian of sigma `s` source latent cells:

```text
D = G - C
B = g_1(D)                                   # low band
mid(x) = g_1(x) - g_4(x)
r = sqrt(g_1(|mid(G) - mid(C)|^2) / (g_1(|mid(C)|^2 + |mid(G)|^2) / 2 + floor))
w = clamp((r - 0.35) / 0.35, 0, 1)            # channel-pooled, per location
prefix = C + B + w * (D - B)
```

`floor` is 5% of the frame's mean content energy. `g_1` has an amplitude
response of 0.92, 0.73 and 0.29 at 16-, 8- and 4-cell wavelengths. For
equal-amplitude content displaced by a fraction `p` of its wavelength,
`r = 2|sin(pi p)|`, so `w` begins at `p` of about 0.06 and is full at about 0.11.
Uncorrelated content gives `r` of about 1.41.

Exact latent properties:

- The band `g_1(D)` comes from `G`. A spatially constant difference transfers
  exactly. Mirror padding does not conserve the spatial mean of a varying
  difference near borders, so a frame's latent mean moves toward `G`'s but is
  not guaranteed to equal it.
- Where `w = 0`, the component of `C` above the band is kept.
- Where `w = 1`, the prefix equals `G` exactly.
- Displacement, disocclusion and replacement are handled only where they change
  mid-band structure enough to raise `w`. Differences confined to fine structure
  above the band keep the carry's component. This is deliberate: target-high
  regenerates the returned target's fine texture, which is generally uncorrelated
  with the carry's. A fine-band detector would therefore fire almost everywhere
  and reduce the prefix to the rejected latent resize.
- There is no temporal filtering, motion estimation or correspondence. Each prefix
  frame is reconciled with the target frame at the same time.

These are properties of H3 VAE latent channels. They do not establish decoded
object presence, geometry or luminance. Moving the latent band, and with it most of
the frame mean, toward `G` changes the reduced-grid stage's tone input relative
to native carry. A resize of the
target latent is assumed to represent its low band on the source grid, as the
`latent_bicubic` path already assumes.

Generated suffix initialization, noise, masks, sigma schedule, audio, learned
transfer, exact target-prefix restoration and target-high are unchanged. No
generated token is constrained directly. Any effect on the suffix comes through
H3's native attention to the protected rows. Motion, occlusion, camera changes
and prompted transitions remain under model and prompt control. When the returned
prefix ends on a new shot, the gate is mostly full and the reconciled prefix is
that shot.

The `partitioned_source_prefix_projection` receipt records:

- policy `native_source_carry_target_band_v1`;
- the underlying `native_source_carry_v1` pairing;
- RMS values of the mean, band and above-band differences;
- gate active and full fractions.

The operation runs on the protected prefix before the first sampler lifetime. It
costs five 24-channel and two single-channel separable Gaussian blurs, about
4×10^7 multiply-adds for a 12×24×28×38 prefix, plus a few prefix-sized float32 temporaries (about 1.2 MB
each at that size). Nothing is retained between chunks. The next chunk's carry
is still that chunk's own low/probe prediction.

### Relation to rejected alternatives

- **Bicubic or half-pixel prefix replacement** replaced the whole carry, and its
  source decodes were blurred and darker. This option keeps the carry's finer
  component wherever latent structure agrees. Elsewhere it uses the same resize.
- **Exact visual reference rows** added conditioning rows and changed target
  timeline positions. This option adds no rows and leaves RoPE positions
  unchanged.
- **Suffix DC/tone bridges, frame-gauge or geometric corrections** modified
  generated output. This option changes only the conditioning prefix.
- **Relaxed temporal correspondence** was not changed. The guidance matcher and
  its prefix-crossing exclusion remain as released.

### Limits

- **Fixed constants.** The band sigma, mid-band scales and gate thresholds are
  fixed choices, not fitted to rendered results.
- **Gate spread.** The gate spreads over the mid band's support, several cells
  around a change. There the softer resized target replaces carry detail.
- **Fine structure.** Content that differs only above the band, such as thin
  fence wires shifted by part of their spacing over a shared layout, keeps the
  carry's version and can disagree with the returned target.
- **Tone.** Latent channel means are not decoded tone. The low stage's decoded
  tone can still change.
- **Unprompted new shots.** Fresh shots that the model chooses on its own are
  not addressed. Flow cannot tell an unintended cut from a prompted one without
  prompt semantics. Suppressing cuts would contradict prompted transitions.
  Continuum owns prompt compilation and already emits a no-new-shot bridge for
  unprompted post-prefix gaps.

Synthetic tests cover:

- agreement;
- an appearing object;
- camera translation without mixed or doubled detail;
- sub-cell motion that keeps detail;
- disocclusion;
- a scene cut in the returned prefix;
- frame-wide offsets;
- the bounds: a fine-only difference keeps carry texture, and the mean of a
  varying difference is not conserved exactly.

Real-path CPU tests use small random-weight H3 blocks with Sol and native
attention. They cover:

- exact prefix bytes;
- unchanged suffix inputs and audio;
- bitwise protected output;
- a deterministic change in generated tokens;
- successive-chunk pairing;
- failure before sampling, with retry;
- capture replay.

None of these establishes trained-model scene continuity; the rendered evidence
above does that for the reviewed sequences only.

## Ownership and failure behavior

A model-local owner retains one detached CPU float32 source clip and a small
mapping of target-suffix hashes. It retains no source tensor on CUDA and no
full target clip between chunks. For example, `[1,24,62,44,44]` retains
11,523,072 source bytes (about 11 MiB). Copying and hashing use temporary CPU
buffers; while a continuation runs, the prior clip and newly staged clip can
both be present. This does not measure the full process peak or model-cache
lifetime.

Nested Flow wrappers for the same guider share one transaction. Another active
guider cannot sample using that owner. A new source/target pair is published
only when the enclosing Flow sampling transaction succeeds, including its
post-sampling audio checks. A failed continuation discards the staged pair and
leaves the preceding successful pair available for a retry. Beginning a new
initial chunk clears the previous pair. Selecting another projection mode
removes this owner from the newly configured MODEL; existing cached MODEL
clones may retain their own references until ComfyUI releases them.

## Evidence receipts

`partitioned_source_prefix_projection` records policy `native_source_carry_v1`,
source/target prefix fingerprints, previous source token span, generation and
sequence pairing, and zero extra H3/VAE calls. `source_prefix_carry_prepared`
records the next candidate pair, CPU bytes and its publication condition;
"prepared" is not proof that a later enclosing check succeeded.

The saved boundary manifest records this policy and projection receipt.
Extended Local Boundary Audit verifies the saved source prefix against that
receipt and uses the actual captured source state. Its geometry fits are
stage-specific image measurements, not additive causal transforms. Compare
source, provider, pre-high, first-high and final results with their fit support,
then inspect the rendered outputs.

### Compare paired prefix states without regenerating

Refresh the PR overlay and replay the saved carry bundle through **MiniMax H3
Local Boundary Audit**, using the production video VAE and
`audit_scope=transfer_and_decoder_context`. Keep the original `chunk_join_frame` and
select the applicable static ROI profile. Sampling is unnecessary.

For a uniform-source prefix of at least seven tokens, the report adds
`source_target_prefix_context`. It reuses the two standalone decoder windows
already evaluated for source and final stages, so the audit's VAE call count is
unchanged. The preceding window ends at the protected prefix; the following
window includes generated suffix tokens. Both describe the same last five
prefix pixel times. A two-token prefix cannot supply a complete preceding
seven-token window, so this comparison is omitted.

The four comparisons measure:

- `source_to_target_without_future`: preceding source-clean and returned-target
  suffix contexts, excluding the new generated suffix;
- `source_to_target_with_future`: those same pixel times in following contexts;
- `source_future_context_change`: source pixels changing between the two windows;
- `target_future_context_change`: target pixels changing between the two windows.

RGB residuals and ROI luminance bias, centered residuals, structure correlations,
gradient ratios and displacement estimates use a common image canvas of at most
704 pixels per side. Area reduction never upsamples either context. These are
absolute paired measurements: prefix normalization and registration do not hide
an existing mismatch. Coordinates are normalized image coordinates, not a
calibrated camera transform. ROI occlusion, texture changes and near-constant
patches can make correlations or displacement estimates unreliable.

This separates a measured state discrepancy from decoder sensitivity to future
context. It does not prove which caused the generated-suffix drift. Standalone
pixels are finalized and clamped; production blends before clamping. The audit
does not reassemble or correct production frames.

CPU tests cover initial capture, exact suffix pairing, native phase, sequence
lineage, bounded replacement, failure rollback and concurrent-owner rejection.
Native Core/Sol integration tests exercise the selected source bytes, unchanged
suffix initialization and protected AV, sampler masks/schedules, call budgets
and capture replay. Trained-weight CUDA quality and repeated-run residency
require the full rendered sequence; these CPU checks cannot establish them.

### Inspect source-to-target evolution with complete decoder context

[Full stage continuity](FULL_STAGE_CONTINUITY.md) replays the seven saved source,
transfer and high-stage timelines with their complete native decoder context.
It saves optional matched PNG frames and measures absolute same-time photometry
and structure, including the first retained continuation frame. Use it when a
source-prefix comparison is already qualified and the remaining question is
where target-grid continuation changes develop.

The [01907 full-stage audit](validation/CONTINUATION_01907_FULL_STAGE.md) completed
this replay on the captured native-source continuation. It measures source-stage
expansion and darkening, transfer detail loss, decoder effects of prefix
restoration, and substantial subsequent detail recovery. It does not qualify a
final seam correction or justify promoting geometric/tone extrapolation.
