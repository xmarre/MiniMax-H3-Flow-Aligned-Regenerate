# Run 01764: exact visual prefix context candidate

## Evidence and limits

The supplied adjacent crops show a change in the stars' shape and sharpness.
The watermark is outside these crops; its spelling change is reported by the
owner. All three Local Boundary Audit JSON reports identify manifest
`7df0e4a00e3176077a30efe26cbd2b3f4d501d1e157a4b1169bffb041b6f36af`.

The executed profile is `progressive_uniform_source`, with a 12-token protected
prefix, 62-token clip, 50x38 low/probe grid and 72x54 target grid. Frame-gauge
repair, suffix DC bridge, retired video overlap blending and post-high boundary
repair do not execute. There is one generated trajectory and no band splice.
The provider and pre-high decoded comparison operands are identical. At frame
175, the first actual high prediction changes RGB by RMS 0.040872 before Flow;
immediate Flow correction changes it by RMS 0.004862. Those are decoded changes,
not independent proof of their causes or semantic quality.

The continuation audit replaces protected-prefix bytes with the authoritative
prefix for comparison. Its decoded prefix context is not the original previous
chunk's assembled pixels. Scalar adjacent-frame differences cannot establish
watermark identity or conclusively exclude decoder amplification at the join.
No native operand bundle, complete video or trained GPU is available locally.

Historical 00684/00686 evidence implicated projected prefix context in static
background changes: exact target-grid low/probe context recovered continuity.
00689 separately implicated the learned transfer in boundary motion. Restoring
the historical heterogeneous path alone therefore does not establish a fix for
all symptoms. The suspected double video normalization was ruled out: pinned
Core's MiniMaxH3Video conversion is identity; additional H3 scaling is audio-only.

## Candidate architecture

Select `spatial_stage_control=progressive_uniform_source_exact_context` in
**MiniMax H3 Partitioned Exact-Prefix Handoff**. Preserve all other released
profile values for a matched comparison. New and saved node defaults do not
change in this candidate.

Low/probe retain the complete native reduced-grid video trajectory, including
its projected protected prefix. Additionally, the authoritative unresampled
target-grid prefix enters H3's native visual-conditioning embedding through a
reference-video segment. Its physical times match the prefix's existing times.
All original conditioning, audio and video RoPE positions stay unchanged.

Core's native reference API handles the independent H/W, visual noise
augmentation and condition timestep. These hidden rows evolve through every
block and contribute to the shared attention context; they have no generated
output ownership. VDN's generated video recurrence remains entirely on the
source grid, with no cross-grid video short-convolution taps. There is no second
generated stream, short-band author or band/tail splice. The learned transfer,
noise transport, exact prefix restoration and high stage remain unchanged.

Flow extends the block layout's numerical signature with the policy and exact
prefix geometry; the native Core/VDN carrier cache signature remains valid.
Boundary capture/replay accepts this uniform-source family member and records
its actual selected mode. `partitioned_exact_visual_prefix` receipts identify
the added conditioning rows and their grid/time ownership.

At 01764 geometry this adds 12 x 972 = 11,664 conditioning rows to 29,450 native
video rows. It adds no H3 evaluations, learned-provider calls, VAE calls, sampler
lifetimes or history boundaries. The extra row count has a real cost; no GPU
latency claim follows from arithmetic accounting.

## Validation and acceptance

CPU regressions demonstrate a concrete information-loss counterexample: exact
prefixes differing in the physical resize's nullspace have indistinguishable
projected carriers but distinct native conditioning rows in the candidate.
Native Core forwarding exercises low/probe/high, selected native and Sol
attention, protected output ownership and Sol history recognition. Native
Euler/Euler ancestral and capture/replay checks cover the new mode too. These
tests use a small random-weight H3 and a bicubic upscaler fixture; they qualify
contracts and execution, not trained image quality or GPU performance.

The required rendered discriminator is the same 01764 scene, inputs and seed
with only this spatial-stage selector changed. Inspect the actual assembled
join, stars, background and lettering, and compare source/provider/first-high
captures, actual/forecast counts, timing and residency. A matched
`same_grid_target_control` run remains the control if this candidate fails.
The candidate does not correct a provider-local temporal defect; that remains
an open contributor. Do not promote the candidate from CPU results alone.
