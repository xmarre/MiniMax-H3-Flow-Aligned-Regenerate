# Experimental Mixed-Grid seam repair

## Why the repair moved before the learned upscaler

The remaining Continuum defect is a small whole-frame shrink/zoom-out with top-edge reveal at the exact-prefix join.

Two downstream hypotheses have now been tested against matched real media:

1. An affine target-grid correction reduced the independently authorized vertical-scale residual to the estimator floor, but the decoded framing jump remained.
2. The exact-overlap representation bridge then repaired the learned-prefix replacement splice essentially exactly before high-grid refinement. In `metrics_00281`, the centered representation mismatch fell from about `0.400812` to `4.3e-08`, and corrected/native raw, low-pass, and spatial-mean seam ratios were all approximately `1.0`. The decoded framing jump still remained.

The second result is decisive: the exact-prefix replacement splice is a real measurable seam amplifier, but it is not sufficient to explain the visible framing defect. The same vertical-scale signature was already measurable in the genuine source-grid continuation before learned 3D transfer. The experimental repair therefore moves geometric correction to that earlier state instead of trying to compensate after upscaling or refinement.

`metrics_00294` then controlled the reference-image sizing confound by scaling all references to the same 1.1 MP budget. The preview framing shift remained. That run also used the current threshold-free source authorization policy and measured provisional source-scale residuals on both `sx` and `sy`, but the source repair was still a no-op because the temporal classifier treated directly observed same-direction cumulative drift as an unsafe divergence. The measured scale state actually moved farther from the natural-motion prediction over four consecutive source suffix transitions. That evidence changes the source repair from an offset-only recovering/persistent model to one that can also correct directly observed cumulative drift before learned upscaling.

## `metrics_00303`: measured drift was found, but the whole source repair was still discarded

The matched decoded video still contains the framing jump. Objective decoded-frame registration shows the largest local shrink at frame `52 -> 53`, about `0.953x` scale, matching the visible boundary defect.

The source stage did now classify `sx` and `sy` as `measured_drift`, but `source_trajectory_bridge_accepted=false`, `source_trajectory_bridge_reason=post_warp_residual_verification_failed`, and `learned_upscaler_input_modified=false`. The exact original source tensor therefore still reached the learned upscaler.

The boundary correction itself actually passed the post-warp test: the authorized residual magnitudes dropped to roughly `6.1%` (`sx`), `14.1%` (`sy`), and `50%` (`ty`) of their pre-warp values. The rejection came only from a later measured transition: the fourth `sx` follow-up crossed sign at about `-0.00839`, outside the `0.005` estimator floor, so the global fail-safe discarded every earlier verified correction as well.

`00303` also exposed a separate safety bug in cumulative measured-drift mode. The original candidate gate limits scale correction to `log(1.03)`, but the directly measured cumulative `sx` states grew to about `0.0496`, `0.0885`, `0.1384`, and `0.1844` after the boundary token. Those later transforms would exceed the same geometric safety contract even though the boundary candidate itself was safe.

The cumulative mode now inherits the candidate's existing per-axis safety bound for every directly applied token and keeps only the longest contiguous measured prefix that remains inside that bound. This is not a new tuned threshold or a handcrafted fade: it reuses the same safety bound already required for initial authorization. In the `00303` geometry, both measured scale axes leave the `log(1.03)` safety envelope immediately after suffix token 0, so only token 0 is eligible for the source correction; later source suffix tokens remain byte-identical. The existing boundary residual-reduction and sign-crossing checks still have to pass before that correction can reach the learned upscaler.

## `metrics_00311`: all-or-nothing post verification discarded three passing axes with one failing axis

The matched `00311` run still shifts and, critically, the source repair still did not reach the learned upscaler: `source_trajectory_bridge_accepted=false` and `learned_upscaler_input_modified=false`.

The safety-limited temporal classifier authorized all four affine axes. The boundary warp itself reduced every authorized residual (`sx` about `0.171x`, `sy` about `0.000205x`, `tx` about `0.298x`, `ty` to the estimator floor). At the first suffix transition, however, only `sx` failed: its residual grew to about `2.049x`, while `sy` improved to about `0.398x` and `tx` to about `0.156x`. The implementation nevertheless rolled back the entire source tensor because post verification was global rather than per-axis.

Post verification is now genuinely per-axis. An axis that would create a boundary or directly affected follow-up transition regression is removed, while independently passing axes are retried together from the original source tensor and fully re-verified. The retry is monotonic (axes can only be removed), has at most four rounds, performs no model/VAE/optical-flow evaluation, and preserves the exact-prefix/tail guarantees. One-token repairs also verify the trailing token-0 -> token-1 transition, preventing a boundary fix from merely moving the pulse one token later.

No estimator floor, objective-gain floor, geometric safety bound, sign-crossing rule, or temporal evidence rule was weakened.

## Source-trajectory repair

The legacy workflow input name `suffix_geometric_bridge` is retained for compatibility. On Mixed-Grid only, enabling it now activates two independent experimental stages. The first stage operates on the clean source-grid sequence immediately before the learned 3D upscaler.

The runtime order is:

```text
genuine source-grid low-stage continuation
-> exact handoff probe
-> recover clean source-grid x0
-> restore resized authoritative prefix only as private upscaler context
-> source-trajectory seam measurement/correction
-> learned 3D target-grid upscaler
-> exact-overlap representation reconciliation
-> existing one-token DC bridge
-> restore authoritative target-grid prefix exactly
-> fresh target-grid refinement
```

The source repair never edits the protected source-prefix tokens. It only operates on generated source-grid suffix tokens.

### Motion baseline

The repair estimates recent natural prefix motion from up to six source-grid prefix transitions. Each transition is fitted with a bounded diagonal affine transform `(sx, sy, tx, ty)`. A robust Theil-Sen trend predicts the expected next transition rather than assuming zero motion.

The first protected-prefix -> generated-suffix transition is measured against that prediction. A candidate axis must satisfy all of the existing estimator and safety requirements:

- scale estimator floor: `0.005` in log scale;
- translation estimator floor: `0.25` latent px;
- scale safety bound: `log(1.03)`;
- translation safety bound: `min(1.5 px, 2.5% of the corresponding source-grid axis)`;
- the fitted axis must materially improve the registration objective.

### Source-only authorization

Before the learned upscaler there is intentionally no independent target-grid observation. The earlier `2.5` quadrature gate combined independent source- and target-domain evidence, so it is not transplanted into this source-only stage. Doing so would add a second magnitude threshold on top of the estimator/objective gates without an independent domain to justify it.

A source axis is provisionally eligible only after it clears the estimator floor, objective-gain floor, and geometric safety bound above. Authorization then requires a safe state in the directly observed suffix transitions, and the applied warp must subsequently pass post-warp residual-reduction verification. `axis_evidence_score` remains diagnostic telemetry; it is not a standalone authorization threshold. Existing cross-grid evidence constants elsewhere are unchanged.

### Measured temporal extent

Up to four genuine source suffix transitions are then measured. The correction is accepted only when the observed temporal state is safe:

- **recovering** — the measured residual approaches the estimator floor or crosses zero without first making an unresolved excursion; the correction envelope is derived directly from that measured cumulative state;
- **persistent** — the measured cumulative residual remains within one estimator floor of the initial boundary state; a persistent correction is applied only when that state is directly observed through the end of the suffix, because truncating an unrecovered persistent correction would merely create a delayed seam at the first uncorrected token;
- **measured drift** — every directly observed follow-up residual stays on the same side/direction as the boundary residual, so each observed suffix token receives the corresponding measured cumulative signed correction instead of a scaled copy of the boundary correction.

Persistent and measured-drift states never extrapolate onto unmeasured tokens. A persistent state that extends beyond the measured window without an observed recovery now fails safe instead of ending a constant correction abruptly and moving the seam to the window boundary; it can be applied only when the directly observed persistent window reaches the suffix end. Measured-drift additionally reuses the original per-axis geometric safety bound on every cumulative token state and stops at the first token that would leave it. With four measured follow-up transitions, at most suffix token 0 plus those four observed tokens can be corrected, and often fewer. No correction is projected onto unmeasured or safety-rejected later suffix tokens.

There is no handcrafted `(1, 0.5, 0.25)`-style fade. Recovering weights come from measured residual evolution; persistent weights are constant only over the observed persistent window; measured-drift transforms come directly from the observed cumulative signed states that remain inside the pre-existing safety envelope.

### Post-warp verification

After applying the source-grid correction, the boundary is registered again. Every authorized axis must reduce its residual magnitude. For measured-drift axes, every corrected directly observed follow-up transition is also re-registered and must reduce the corresponding residual (or remain inside the estimator floor). A sign crossing is accepted only within the estimator floor. If a correction does not pass the required verification, the exact original source tensor is returned instead.

The implementation also asserts that:

- the source prefix is byte-identical;
- suffix tokens outside the measured active window are byte-identical;
- no non-finite values are introduced.

## Independent exact-overlap target reconciliation

The target-grid exact-overlap bridge remains a second, independent operation. It is not used as evidence that the source correction succeeded.

Let:

```text
L_p = learned-upscaler last prefix token
E_p = authoritative exact last prefix token
L_s = learned first suffix token
D   = E_p - L_p
```

The representation bridge applies only the zero-spatial-mean component of `D` to `L_s`. The validated `suffix_dc_bridge` independently owns the spatial-mean/DC component. With both active, the exact-prefix -> corrected-suffix transition reproduces the upscaler-native learned-prefix -> learned-suffix transition before high-grid refinement, modulo output-dtype rounding.

This target operation still modifies only suffix token 0 and never edits the authoritative prefix.

## Preserved contracts

The experimental repair:

- never modifies or warps the authoritative target-grid protected prefix;
- adds no H3 transformer evaluation/NFE;
- adds no VAE pass;
- performs no image-space or tensor crossfade;
- performs no decode-space repair;
- does not change audio, caller noise, masks, or conditioning;
- does not change VDN external-sequence API 2;
- does not change Spectrum history, forecast, or NFE accounting;
- leaves the released DC-only arithmetic unchanged when the experimental option is disabled or rejects its source candidate;
- is exposed only on the Mixed-Grid node;
- remains `false` by default.

## Diagnostics

`mixed_grid_source_trajectory_bridge` records the pre-upscale source decision, including:

- recent natural-motion model;
- measured source boundary transform;
- per-axis residual magnitude, safety bounds, and evidence;
- temporal classification and measured/safety-limited envelope;
- authorized axes;
- applied source transforms and corrected-token count;
- post-warp boundary and directly observed transition residual-reduction ratios;
- explicit accept/reject reason;
- whether the learned-upscaler input was modified.

`mixed_grid_representation_bridge` separately records target exact-overlap reconciliation, and `mixed_grid_transfer`/`mixed_grid_complete` retain the target-grid seam diagnostics.

## Validation gate

Unit/native-source tests can establish ordering, evidence gating, bounded temporal support, exact-prefix preservation, no-op behavior, target-splice algebra, and unchanged model-call accounting. They cannot establish perceptual success.

The next matched decoded-media run must keep the rest of the workflow unchanged and use:

```text
suffix_dc_bridge = true
suffix_geometric_bridge = true
```

The decisive evidence is whether the source event authorizes and reduces the real boundary residual, reports `learned_upscaler_input_modified=true`, and whether the decoded shrink/top-edge reveal is actually gone without a pulse, delayed wobble, detail loss, or motion regression. The option remains off by default and PR #24 remains draft until that gate passes.