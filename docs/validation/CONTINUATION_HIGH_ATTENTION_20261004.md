# Continuation: protected-prefix attention through high refinement

Status: source-validated candidate; no new GPU render qualifies the frame-shock
or tone outcome. PR #93 remains a draft stacked directly on Flow #89. The
accepted `same_grid_target_control` stays available and unchanged. Progressive
low-to-high continuation remains unqualified for the reported artifact.

The newer [zero-hook 01132 transfer review](TRANSFER_COORDINATE_GAUGE_20261004.md)
localizes a large discontinuity to exact-prefix restoration before high sampling.
The attention changes below are not a sufficient repair. The current candidate
also selects matched physical coordinates inside learned transfer.

Follow-up: [01119's native and regional review](CONTINUATION_01119.md) verifies
this policy on all four actual high calls. The supplied assembled video permits
separate forest-motion and foreground-tone checks; its large whole-frame motion
estimate does not establish a uniform frame translation. This is a different
seed and geometry from 01115, and rendered acceptance remains unqualified.

## Evidence and causal limits

[Run 01115's native tensor/decoder review](CONTINUATION_01115.md) reproduces a
within-token motion jump from captured clean tensors. Full-frame vertical motion
on raw decoder frames `9 -> 10` is `+0.059 px` before high refinement,
`+2.615 px` in the first high prediction before Flow, and `+4.502 px` finally.
The change begins in the model prediction, before Flow's prediction correction.
That capture includes all executing model patches; it does not isolate Sol,
VDN, noise transport, conditioning, or the checkpoint as a complete root cause.

Prefix replacement and the one-token DC bridge also change decoded dark-tail
and contrast statistics before high refinement. Restoring the prefix is required
for continuation; those tone observations do not establish a valid exposure
calibration or validate removing DC from the complete sampler trajectory.

Source inspection establishes two additional operator changes at the handoff:

| Policy | Low / endpoint probe | Previous native high | Candidate high |
| --- | --- | --- | --- |
| Protected-prefix local queries | Dense through the partitioned VDN/Sol contract | Native Sol selection allowed | Dense through the same contract |
| Generated local queries | Configured Sol policy | Configured Sol policy | Configured Sol policy |
| Default Sol startup, one dense evaluation | Probe starts a fresh request with warm-up | Flow continuation tag consumes warm-up | Fresh target refinement honors warm-up |
| Spatial representation | Exact target prefix, selected suffix grid | Native target grid | Equal-grid partition on the native target grid |
| High learned linear branch and target audio positions | Low/probe controls apply only to low/probe | Native | Native |

Sol's exemption is explicit: `dense_evaluations == 1`, stage `high`, and
`h3_refinement.source == "h3_flow_progressive_handoff"` cause the configured
startup evaluation to be consumed. Running a dense source endpoint probe does
not establish that the following target-grid noisy input has already received
a dense evaluation. This is especially clear when the spatial grid changes.

The candidate removes these avoidable attention-policy changes. This is a
source-level contributor and invariant repair, not proof that either change is
the sole cause of the rendered shock. A progressive handoff still changes the
suffix's spatial representation and noisy-input distribution. This patch does
not prove that transported clean/noise operands form the trained target-grid
distribution.

## Implementation and ownership

High refinement publishes `h3_flow_partitioned_refinement`, retaining the
existing first-actual-evaluation requirement. Sol no longer interprets the
endpoint probe as consumed target startup. Explicit warm-up settings, including
zero and two evaluations, keep their configured meaning. This does not impose
an extra warm-up setting when the user selected zero.

For exact-prefix continuation, the existing partitioned transformer and v4 VDN
attention interfaces remain installed during high refinement. A stage-local plan
borrows the authoritative prefix and its original noise, uses the native target
grid for both domains, and therefore uses unit key measure. Protected-prefix
local groups retain dense attention; generated local groups retain normal Sol
selection. VDN's window geometry, anchors, gates, learned linear complement and
equal-grid linear fast path remain in force.

Low/probe diagnostic selectors are temporarily normalized for this high owner:
normal softmax and learned linear policies, native temporal carrier, exact
prefix context, and native target-audio positions. Prior option presence and
object identity are restored on success or failure. A source-uniform execution
arm keeps its native high transformer, while still receiving the fresh startup
tag. Nested partition owners are rejected without replacing the original owner.

An unused high-stage boundary witness is omitted. A configured witness would
otherwise disable VDN's existing uniform-grid fast path and retain unnecessary
raw Q/K/V features. The low witness and its evidence remain unchanged.

There are still three sampler lifetimes and two history boundaries. No logical
model call is added. The first high call is still actual. Honoring warm-up and
the new provider identity can change subsequent actual/forecast counts; no
constant-actual-NFE claim is made. Dense prefix attention may increase high-stage
time or memory. CUDA cost is not measured here.

The clean/noise transfer, prefix mask, one-token DC bridge, subsequent Flow
guidance, decoder windows and assembly remain unchanged. The patch introduces
no suffix warp, prediction clamp, frozen successor guard, prefix repaint,
VAE round trip, or audio smoothing actuator. Audio outputs can respond to the
changed joint model evaluation; audio continuity still needs render validation.

## Research constraints

The six supplied papers support reviewing the operator and input distribution;
none supplies a validated inference repair for this H3 continuation:

- **DMD2** (`2405.14867v2`, sections 4.4-4.5) treats multi-step input-distribution
  mismatch through training and rollout design. It does not certify an arbitrary
  inference-only spatial/noise handoff.
- **Video DeltaNet** (`2609.20744v1`, sections 2.1-2.3) keeps nearby softmax
  interactions aligned with five-token decoder chunks, with neighboring chunks,
  boundary anchors, and a learned linear complement. The candidate preserves
  those components rather than replacing their geometry or gates.
- **Sol-Attn** (`2607.24027v1`, section 4.1) uses different startup policies for
  different models and stages. Its LTX second-stage result without warm-up does
  not establish that MiniMax-H3 continuation should consume its configured
  target-stage startup.
- **Spectrum** (`2603.01623v1`, section 3.3) analyzes forecasting a feature curve
  under analytic-function assumptions. It does not establish continuity across
  a changed attention operator. Independent stage histories remain required.
- **Sol video inference engine** (`2606.23743v2`) evaluates model-specific
  optimization recipes and quality/performance tradeoffs. It does not qualify
  this continuation configuration.
- **Video Sparse Attention** (`2505.13389v5`) uses trainable coarse/fine attention
  and a matched training/inference design. It does not justify grafting an
  unrelated sparsity policy onto this distilled model as a boundary repair.

## Source verification

Reviewed dependency pins:

| Source | Commit |
| --- | --- |
| Core | `6b4e05dc30d65740ce8931434607b9907996fb0e` |
| Sol | `f59bdd15c4ef1d1893052298a663da3850ac1a48` |
| VDN | `9af4cd1e8ff396fecf2237c8b08f588c53f7fa96` |
| Spectrum | `beb32dd210ef9e95520453107f158241d4f2ecf3` |

The paired source oracle exercises Sol's real warm-up predicate for zero, one
and two dense evaluations and VDN's real grouped-query policy at 47 tokens,
12 protected tokens, chunk size five, radius one, and both anchors. Protected
groups are dense, generated groups keep their configured selection, and the
high plan is equal-grid with unit key measure.

The native arithmetic regression uses two real Core H3 blocks and Sol dense
attention on a tiny frozen random model with matching same-grid input, sigma,
non-PDD head and native prefix augmentation. Probe and high video/audio outputs
are bitwise identical; generated video remains nonzero and protected-prefix
velocity is zero. The CPU harness bypasses Sol's GPU device-admission predicate.
It does not execute the trained checkpoint, sparse CUDA kernels, or an entire
VDN-patched model rollout. VDN's uniform-grid arithmetic and workspace oracles
are checked separately.

Lifecycle tests cover equal/progressive grids, failure cleanup, original prefix
and noise bytes, nested owner refusal, native source-uniform execution, and witness
omission. Runtime-gate tests reject changed geometry, key measure, startup
exemption, learned-linear policy, or missing actual high receipts.

Verification completed on Python 3.12 with CPU PyTorch:

- Full Flow suite: **898 passed, 28 skipped**. The skips require separately
  supplied Core sources; the new native Core/Sol test runs in the source job.
- Selected native Core/Sol/VDN dispatch, workspace, low-VRAM, forward-lifetime,
  grouping, factorization and runtime-buffer suite: **164 passed, 7 skipped**.
  Its skips cover CUDA-only checks.
- VDN partitioned linear suite, including uniform-grid reduction and dispatch:
  **62 passed**.
- Paired Flow/Sol/VDN source oracle, including bounded boundary-witness transport
  and the new high attention policy: **passed**.
- Ruff check/format, compile, wheel/sdist build, license metadata and isolated
  wheel import: **passed**.
- Reviewed checkpoint `283b57dbb39c8b6cea5356f0fe241d28c54fea2b`: all five CI jobs
  passed (Python 3.10-3.13 and the pinned source-contract job).
- Historical 01115 metrics still pass their ordinary runtime gate and are
  rejected by the new high-attention qualification flag.

No CUDA arithmetic, memory, latency or rendered-quality result is inferred from
CPU tests or CI.

## Qualification required on the new head

The new `partitioned_high_attention_plan` receipt identifies
`exact_prefix_query_continuity_v1`. Every actual high call must emit a target-grid
partitioned transformer receipt with the original prefix length, total temporal
length, equal source/target row counts, unit key measure, and native target-audio
positions. Logical bindings include high-stage forecasts; actual transformer
receipts count only actual executions.

The offline gate's `--expected-high-attention-policy
exact_prefix_query_continuity_v1` requires this evidence. Historical metrics
remain valid for their old source but fail this candidate qualification. Run
01115 has no high policy receipt and cannot qualify the repair. The gate reports
policy, verified status and high transformer-event count explicitly.

Use the same ComfyUI Patcher overlay stack with only Flow #93 updated. Retain the
accepted continuation selector, exact audio policy, seed, conditioning, model
and adapter strengths. Record the configured Sol warm-up and fresh source
receipts; do not reuse old rendered evidence for the changed operator.

Rendered acceptance must cover:

1. The complete assembled boundary, including the preceding decoder window, the
   within-token `9 -> 10` pair, the other frames in that token, and the following
   window. The jump must disappear without shifting it to a neighboring pair.
2. Black level and contrast across the boundary and subsequent window, assessed
   with matched composition and motion as well as luma quantiles. Mean alone is
   insufficient to qualify the tone outcome.
3. Exact protected-prefix bytes, continuing generated motion, no successor freeze
   or hitch, and unchanged exact-audio masking with no burst or new audio shock.
4. Actual/forecast receipts, mapped-query accounting, selected kernel path,
   sampler duration and peak VRAM under the complete existing adapter stack.

Until those checks pass on this head, the proper source correction is available
for evaluation but the reported rendered frame/tone problem remains open.
