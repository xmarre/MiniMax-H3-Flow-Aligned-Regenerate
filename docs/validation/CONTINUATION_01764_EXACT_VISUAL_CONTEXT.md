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

## Follow-up: colocated candidate rejected by rendered evidence

Run 01767 used `progressive_uniform_source_exact_context` with residual transport,
DC bridge and frame-gauge repair disabled. The transfer/decode report includes
a native reduced-grid source stage; the separate stage-continuity report omits
that stage. All three reports identify manifest
`ef1e7b05575bdb067aceb092620d2749c3298d96f3c0a53975a05b15c6e2f54f`.

The 177→178 adjacent pair has full-frame diagnostic `zero_huber` 0.12622 in
source-grid decoding, 0.13512 after transfer and 0.15037 in final decoding.
It is already present before learned transfer and high refinement. Source
and target displacements use different native decoder pixel grids and cannot
be compared as equal pixel units. The earlier first-high redraw also persists;
it does not independently explain the new source-grid hitch. Adjacent report
row labels identify the later frame, so row 178 means 177→178.

This rejects the original time-colocated conditioning candidate's rendered
acceptance. It does not establish a unique trained-model cause. The original
candidate manually replaced Core's reference timeline with overlapping prefix
times; the revised candidate below removes that override. Native placement is
a structural correction and a new rendered hypothesis, not an accepted fix.

## Candidate architecture

### All-stage extension withdrawn after rendered failure

Run 01769 executed the all-stage native-reference extension: seven low/probe
and four high reference calls, all with the native 65-unit target timeline
shift. The reports identify manifest
`c47296c525bc7422991f9d6f653bddb2508cd67bb1281b4589d9625dab62a02d`
and authoritative prefix
`27f44dce13c0badc31edad859b1df388112f4c29a6dcb6530722c99285c26d16`.
The owner reports persistent multi-frame darkening and renewed visible shift.
This rejects the extension's rendered acceptance; it has been withdrawn.
The current production code restores the low/probe-only native reference
implementation, including its numerical policy and native high contract.

Within 01769, upper-45% diagnostic error for 175→176 rises from 0.0170 before
high to 0.0398 in the first actual prediction before Flow guidance, and 0.0402
finally. Estimated vertical displacement changes from -0.67 to -2.96 to -3.39
pixels on the same target decoder grid. At frame 181, mean luma is 0.33126
before high and 0.32500 finally. The first high prediction still redraws the
near-join region and final output remains darker, despite retaining the
reference. Immediate guidance is not the sole origin of those changes.

Run 01768 used a different authoritative prefix (`61826843…`). Its internally
decoded 175→176 pair also changed during high. These runs cannot prove that
adding the high reference caused a new shift or that low/probe-only native
placement robustly fixes it. Removing the failed extension restores the prior
implementation; it does not establish a complete tone or geometry fix.

Frames before the join in these audits are discarded, redecoded chunk context,
not the previous chunk's original assembled pixels. The reports establish
within-run stage changes, but cannot alone identify the cause of the actual
assembled seam. Decoder-window context, representation changes at transfer,
native high reconstruction and forecast/solver behavior remain possible
contributors. A unique handoff or decoder root is not established.

The production log supplies an additional assembled-pixel comparison:
`H3C-PT227` compares the last 22 decoded frames of the previous physical chunk
with the duplicate carried region in the next chunk. In both 01768 and 01769,
its sampled RGB RMS and mean differences are exactly zero, channel gains are
one, and the native overlap phase is aligned. Both the 17-frame interior and
five-frame right-context tail match on the diagnostic sample grid. No video
assembly patch executes. This rules out a measured decoder-phase or duplicate
overlap exposure mismatch as the seam's origin. It does not establish native
full-resolution pixel equality or exclude decoder amplification of genuinely
different generated latents. The first actual high prediction already changes
the generated region, so forecasts are not its sole origin either.

Select `spatial_stage_control=progressive_uniform_source_exact_context` in
**MiniMax H3 Partitioned Exact-Prefix Handoff**. Preserve all other released
profile values for a matched comparison. New and saved node defaults do not
change in this candidate.

Low/probe retain the complete native reduced-grid video trajectory, including
its projected protected prefix. Additionally, the authoritative unresampled
target-grid prefix enters H3's native visual-conditioning embedding through a
reference-video segment. Keep Core's native reference positions without manual
RoPE rewriting. The new reference follows existing references and precedes the
target timeline. Core advances target audio, video and keyframes by the same
reference duration: 65 time units for a 12-token prefix. Existing reference/text
positions stay fixed. Target A/V relative time is unchanged, but target↔text
and target↔existing-reference relative time changes. Prompt, motion, tone and
audio behavior therefore still need trained-model validation.

Core's native reference API handles the independent H/W, visual noise
augmentation and condition timestep. These hidden rows evolve through every
block and contribute to the shared attention context; they have no generated
output ownership. VDN's generated video recurrence remains entirely on the
source grid, with no cross-grid video short-convolution taps. There is no second
generated stream, short-band author or band/tail splice. The learned transfer,
noise transport, exact prefix restoration and high stage remain unchanged.

Flow extends the block layout's numerical signature with the policy and exact
prefix geometry; the native Core/VDN carrier cache signature remains valid.
The production node already puts Flow's diffusion wrapper first. Conditioning
therefore reaches VDN before it captures its native layout; reversing that
wrapper order is invalid for the candidate. Core and VDN must receive the same
prepared layout, rather than adjusting VDN's layout after attention starts.
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

A real VDN learned-branch test also holds generated input, audio and projected
prefix fixed, changes only exact prefix detail in the resize nullspace, and
observes a changed suffix prediction through the shared deep context. Repeating
the original context reproduces its output with retained VDN workspaces. This
test uses the production Flow-before-VDN wrapper order and native attention;
it does not substitute VDN's recurrence, short convolution or readout.

The required rendered discriminator is the failed 01767 scene, inputs and seed
with only this spatial-stage selector changed and
`uniform_source_detail_transport=false` (the default). Transport is a separate
experimental intervention whose prefix calibration does not establish suffix
motion safety; do not combine it with the initial exact-context comparison.
Inspect the actual assembled join, stars, background and lettering, and compare source/provider/first-high
captures, actual/forecast counts, timing and residency. A matched
`same_grid_target_control` run remains the control if this candidate fails.
The candidate does not correct a provider-local temporal defect; that remains
an open contributor. Do not promote the candidate from CPU results alone.
