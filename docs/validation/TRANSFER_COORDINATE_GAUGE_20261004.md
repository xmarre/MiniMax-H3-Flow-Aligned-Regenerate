# Transfer coordinate gauge: zero-user-LoRA evidence

## Evidence and limits

Run `metrics_01132_.json` / `Pasted text(20261004-191650).txt` reports zero user
model/CLIP hooks and still visibly shifts. External LoRA is therefore neither a
necessary condition nor an established sole root cause. VDN #37's
`boundary_suffix_local_group_dense_v1` executes at frames 12,13,14 and is not a
sufficient repair. Its scope is not extended by this candidate.

| Stage | Upper45 boundary-vs-pre norm | Full norm |
| --- | ---: | ---: |
| source_low_native | 0.162804 | 0.309795 |
| learned_native | 0.130946 | 0.451137 |
| exact_restored_pre_high | 0.824172 | 1.323388 |
| final_post_high | 0.972610 | 0.968162 |

Source measurements use source latent cells; later measurements use target
latent cells. The source-to-target factors are 66/46 vertically and 58/40
horizontally. The direct learned/restored comparison uses the same target grid.
Later suffix pairwise motion is essentially unchanged at restoration; the
frame 11->12 pair changes. The source prefix overwrite before the provider
changes the measured source trajectory by only rounding-scale amounts.

The new run exports an 18,741,888-byte decoder-window bundle, but those tensor
bytes were not among the available files. Its hashes are not enough to perform
a same-frame affine fit on 01132. This candidate does not report an invented
01132 fit or rendered success.

An older, separately identified 15:38 bundle was available and its tensor hashes
were verified. Same-frame full-affine fits between its learned and authoritative
prefixes at frames 10/11 gave center displacements about (-0.467,-0.410) and
(-0.477,-0.344) cells, with vertical displacement gradients 0.01348/0.01157.
The fit lowered normalized Huber loss by only about 3-4%; representation changes
remain. These are diagnostic results from seed 14766695179764173950, target
48x80, not measurements of 01132 or proof of a global affine cause.

## Proven operator mismatch

The transition is:

1. The authoritative target prefix is projected to a source carrier using
   `resize_spatial_5d_h3_patch_lattice`.
2. Generated low/probe suffix rows use the native source H3 spatial RoPE lattice.
3. The old learned resizer uses `F.interpolate(..., trilinear, align_corners=False)`
   between encoder and decoder, which samples a half-pixel image lattice.
4. The provider's prefix output is discarded and `[0:prefix_t]` is overwritten
   by the exact target prefix. The different sampling convention becomes exposed
   precisely at this splice, before high sampling.

H3's patch axis is `start=(1-dim/sqrt(area))*16`, `step=32/sqrt(area)`.
It is not the half-pixel image mapping `(i+0.5)*source/target-0.5`.
Both spatial axes and the four within-patch latent features matter.

For coordinate ramps in the actual 46x40 -> 66x58 geometry, the old round trip
fits the target-to-authoritative pullback below after averaging each 2x2 patch:

| Axis | Scale | Origin, latent cells |
| --- | ---: | ---: |
| x | 0.992134 | -0.218189 |
| y | 1.004168 | -0.577842 |

Affine fit residual is 0.196585 cells because phase aliasing is not fully affine.
Using H3 patch coordinates in both directions yields scales within 4e-8 of one,
origins below 1e-6 cells, and fit RMS 2.2e-6 cells. This is an operator experiment
without learned weights; it cannot prove the trained decoder's final geometry.

## Candidate and ownership

The upscaler fixes its selected encoder-to-decoder coordinate transform. Flow
requests its explicit `h3_patch_lattice_api=1` capability for partitioned learned
handoffs and rejects an old provider before a sampler executes. Ordinary
upscaling, same-grid identity and the named half-pixel bicubic diagnostic retain
their existing selection. No final image/latent warp or fitted affine correction
is applied. The authoritative prefix bytes, sampler masks, audio ownership,
NFE schedule and VDN boundary-dense policy remain governed by their existing
contracts. Real time/VRAM and rendered tone need measurement.

When `frame_gauge_residual_mode=measure`,
`partitioned_same_frame_prefix_affine` fits the provider's untouched prefix
against the same-time authoritative prefix for up to four tail frames. It
reports translation, both-axis scale/shear, nine local tile errors, input
digests and CPU wall time. It runs even when the old rigid registration is
rejected. It never selects a correction and adds no provider or H3 evaluation.

## Primary runtime validation

Replay 01132 with **zero user-LoRA hooks**, the same workflow/seed/reference,
46x40 -> 66x58, prefix_t=12 and the corrected upscaler/Flow candidates. Keep
`frame_gauge_residual_mode=measure` so the before-overwrite affine receipt is
available. Compare the same four trajectory stages and decoded boundary, then
repeat the matched user-LoRA on/off pair only after the zero-LoRA path is sound.

Execution evidence must include:

- the upscaler's `[H3 learned transfer]` log selecting
  `h3_physical_patch_lattice_v1` at `encoder_to_decoder`;
- `partitioned_transfer_lattice` with matching source/target geometry,
  `provider_calls=1`, unchanged authoritative-prefix ownership and zero extra NFE;
- `partitioned_same_frame_prefix_affine` with measured frames 8..11, digests,
  both-axis gradients and local tile errors;
- exact-prefix/audio checks, first-high input ownership, actual/forecast counts,
  wall time/peak VRAM, and decoded evidence without a new tone shock.

The existing 01132 tensor bundle is also useful without another baseline run:
its authoritative-prefix and provider-native-clean bytes enable the missing
same-frame affine comparison. The new metrics alone cannot recover those bytes.

## Secondary user adapter defect

Core `linear_input_act` can read an INT8 weight directly and skip replaced
`module.forward` and Module hooks. A real CPU INT8 reproduction shows zero
external-adapter contribution through that path even though the hook is
registered. The separate Core fix preserves module-call semantics while keeping
fusion for unmodified/ejected modules. This is a numerical correctness bug,
not an explanation of the zero-hook 01132 failure. Whether it amplifies the
transfer mismatch has not been established with a rendered A/B.
