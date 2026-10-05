# Continuation 01093

Run 01093 is the first completed hardware run after binding high guidance to the
actual learned-provider source/target pair. The rendered frame-shift/shock/tone
defect remains visible, so the provider-pair guidance correction is structurally
valid but not sufficient.

## What executed

The continuation selected the main exact-partitioned trajectory with seven exact
samples. The handoff guidance owner bound the actual learned provider pair. The
captured-versus-provider source suffix differs only at floating-point roundoff
(max about 4.77e-7), while the intentionally restored prefix differs more because
ownership is changed there. High guidance subsequently reports both
`handoff_reference_used=true` and `temporal_handoff_reference_used=true` with
`protected_prefix_t=12`.

The same run requested six video overlap tokens. The old exact-prefix overlap
implementation therefore wrote a nonzero denoise ramp into carried-prefix tokens
6..11 during early target-high sampling and closed it only before the final two
evaluations. Final exact-prefix restoration remained byte-exact.

## Localization

The learned provider's native first boundary pair is large, but exact-prefix
reconciliation reduces it substantially before target-high:

| stage | upper45 dx/dy (cell) | full dx/dy (cell) |
| --- | --- | --- |
| learned native | -0.8471 / -0.1874 | -1.5768 / -0.1229 |
| exact restored pre-high | -0.08379 / +0.00895 | -0.06836 / +0.01086 |
| first high prediction before Flow | -0.20175 / -0.07831 | -0.43309 / +0.11829 |
| same prediction after Flow | -0.20317 / -0.07834 | -0.43438 / +0.11425 |
| final post-high | -0.17971 / -0.08571 | -0.32576 / +0.00886 |

The first target-high H3 prediction therefore recreates a substantial boundary
displacement before Flow guidance. Flow changes the measured first-pair motion
only slightly. This excludes the corrected learned-reference guidance mismatch
as the dominant producer in this run.

The raw decoded Continuum boundary also remains discontinuous while assembly
reports no video patch. The plan-aligned duplicate overlap is exact, so replacing
or blending duplicate decoded overlap is not supported as the cause.

## 01093 correction

The prefix-release interpretation of `video_guided_overlap_tokens` is retired
for partitioned exact-prefix sampling. Positive values remain valid workflow
inputs but are observation-only: target-high keeps the authoritative video mask
and exact carried context unchanged for every evaluation. No overlap closure hook
is installed.

This is deliberately not another output warp, fixed clean anchor, prediction
gauge bridge, VAE-window translation or one-token high guard; those actuator
families were already falsified by hardware. It removes the newly demonstrated
discarded-context conditioning and leaves the next render as the acceptance
test. If the defect remains with exact target-high prefix context, the exported
native boundary-window tensors are the next causal evidence rather than a reason
to re-enable a rejected actuator.
