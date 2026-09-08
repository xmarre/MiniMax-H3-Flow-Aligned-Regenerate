# Experimental Mixed-Grid seam repair

## Why the repair moved before the learned upscaler

The remaining Continuum defect is a small whole-frame shrink/zoom-out with top-edge reveal at the exact-prefix join.

Two downstream hypotheses have now been tested against matched real media:

1. An affine target-grid correction reduced the independently authorized vertical-scale residual to the estimator floor, but the decoded framing jump remained.
2. The exact-overlap representation bridge then repaired the learned-prefix replacement splice essentially exactly before high-grid refinement. In `metrics_00281`, the centered representation mismatch fell from about `0.400812` to `4.3e-08`, and corrected/native raw, low-pass, and spatial-mean seam ratios were all approximately `1.0`. The decoded framing jump still remained.

The second result is decisive: the exact-prefix replacement splice is a real measurable seam amplifier, but it is not sufficient to explain the visible framing defect. The same vertical-scale signature was already measurable in the genuine source-grid continuation before learned 3D transfer. The experimental repair therefore moves geometric correction to that earlier state instead of trying to compensate after upscaling or refinement.

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

### Source-only evidence gate

Before the learned upscaler there is intentionally no independent target-grid observation. The earlier two-domain gate required a quadrature evidence score of `2.5`. The source-only stage therefore uses the equal-contribution per-domain equivalent:

```text
minimum_source_evidence = 2.5 / sqrt(2) ~= 1.76777
```

This does not change the existing cross-grid constants elsewhere. It is a separate pre-upscale authorization gate, and evidence alone is not enough to apply a correction.

### Measured temporal extent

Up to four genuine source suffix transitions are then measured. The correction is accepted only when the observed temporal state is safe:

- **recovering** — the measured residual monotonically approaches the estimator floor or crosses zero; the correction envelope is derived directly from that measured cumulative state;
- **persistent** — the measured residual remains within one estimator floor of the initial boundary state for at least two follow-up transitions.

Persistent state is deliberately bounded to the directly observed window. With four measured follow-up transitions, at most suffix token 0 plus those four observed tokens can be corrected. The implementation never projects a constant correction over unmeasured later suffix tokens.

There is no handcrafted `(1, 0.5, 0.25)`-style fade. Recovering weights come from measured residual evolution; persistent weights are constant only over the observed persistent window.

### Post-warp verification

After applying the source-grid correction, the boundary is registered again. Every authorized axis must reduce its residual magnitude. A sign crossing is accepted only within the estimator floor. If verification fails, the exact original source tensor is returned instead.

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
- per-axis residual magnitude and evidence;
- temporal classification and measured envelope;
- authorized axes;
- applied source transforms and corrected-token count;
- post-warp residual reduction ratios;
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

The decisive evidence is whether the source event authorizes and reduces the real boundary residual and whether the decoded shrink/top-edge reveal is actually gone without a pulse, delayed wobble, detail loss, or motion regression. The option remains off by default and PR #24 remains draft until that gate passes.
