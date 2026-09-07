# Experimental geometric Continuum seam bridge

A progressive Mixed-Grid Continuum run exposed a second boundary failure after the released one-token DC bridge had already removed the obvious flash: a slight whole-frame framing change at the chunk join, described as a small shrink/zoom-out and top-edge reveal.

The DC bridge is still correct for its narrow purpose. It adjusts per-channel spatial means. It cannot correct global scale or translation.

This draft remains **experimental and off by default**. No corrected decoded run has yet validated the new policy.

## Evidence that changed the design

The first instrumented real run used source `40x54` and target `56x76` latent grids with `suffix_geometric_bridge=true` and `suffix_dc_bridge=true`. The initial bridge correctly rejected and changed zero geometric tokens.

The same-time learned-prefix fit was effectively identity:

- `sx=1.0125`, `sy=0.9950`, `tx=0`, `ty=0.25` latent px;
- identity error `0.171124`;
- aligned error `0.168418`;
- improvement only `1.58%`;
- `persistent_bias_candidate=false`.

That is evidence **against** a systematic learned-upscaler prefix warp being the main cause.

The actual protected-prefix -> generated-suffix boundary was different:

- before target-grid refinement: approximately `sx=1.0075`, `sy=0.9775`, `tx=-1.25`, `ty=0.5`;
- final accepted latent boundary: approximately `sx=0.9975`, `sy=0.9800`, `tx=-1.25`, `ty=0.25`;
- final fitted-vs-identity improvement was about `30%`.

The repeated `sy≈0.98` signal matches the observed slight vertical zoom-out/top-edge reveal. The leftward translation was already present in recent prefix motion and must not be blindly cancelled.

The old correction therefore had the wrong driver: **same-time learned-prefix registration must remain diagnostic, while the bridge must reason about the generated boundary relative to natural prefix motion.**

## Scope

Only `MiniMax H3 Progressive Mixed-Grid Continuum` exposes `suffix_geometric_bridge`.

- default: `false`;
- generic Progressive Target Input: unchanged;
- Target-Sparse: unchanged;
- authoritative target-grid prefix: never warped or blended;
- audio, masks, conditioning, sampler noise, VDN API 2, Spectrum histories and NFE count: unchanged;
- geometry is applied on the learned clean target-grid sequence before conditional re-noising;
- DC calibration remains after geometry;
- final exact-prefix restoration/checks remain mandatory.

## Registration coordinates

The estimator fits only `sx, sy, tx, ty`; no rotation or shear.

Coordinates are output-to-input sampling coordinates about the image centre:

```text
input_x = sx * (output_x - (W-1)/2) + (W-1)/2 + tx
input_y = sy * (output_y - (H-1)/2) + (H-1)/2 + ty
```

Positive `tx` samples to the right and therefore moves displayed structure left.

Warping uses float32 bilinear `grid_sample`, `padding_mode="border"`, `align_corners=false`, then restores the original dtype/device. Border extension avoids invented zero-valued edges but can stretch edge content, so accepted transforms are tightly bounded and out-of-bounds sampling is reported.

## DC-insensitive representation

Registration uses all latent channels, not one selected channel:

1. 5x5 spatial low-pass;
2. area reduction capped at 48 pixels on the long axis;
3. per-channel spatial centring and RMS normalization;
4. fixed interior scoring margin so padding cannot win the fit;
5. clipped squared residuals to limit local-detail outliers.

The deterministic coarse-to-fine search runs without autograd. NumPy is used only for the small CPU candidate-scoring arrays; global thread settings are not modified.

## Same-time transfer diagnostic

`P_exact[t] <-> U_prefix[t]` is still measured over the last up-to-three paired prefix tokens.

It can identify a genuine learned-transfer bias, but **it no longer authorizes suffix warping**. The first real run showed why: the learned prefix was essentially aligned while the generated boundary was not.

The existing conservative paired-prefix acceptance remains useful as a diagnostic field (`transfer_bias_candidate`).

## Natural-motion model

The bridge now measures up to the last six exact-prefix transitions:

```text
P[-6] -> P[-5]
...
P[-2] -> P[-1]
```

Each transition receives a diagnostic `sx, sy, tx, ty` fit. At least three usable transitions are required.

For each parameter, the motion model performs a robust Theil-Sen trend estimate and predicts the next transition. Scale is modelled in log space. The report includes:

- all measured prefix transitions;
- usable-transition count;
- expected next transform;
- local trend;
- robust dispersion.

This is important for slowly changing camera motion. A genuine ongoing zoom or pan should be extrapolated rather than treated as a seam defect.

## Boundary residual

Let:

- `T_expected` = predicted natural transform from the recent exact-prefix motion;
- `T_observed` = measured transform aligning the first learned generated suffix token to `P_exact[-1]`.

The correction is the residual transform `C` satisfying:

```text
C ∘ T_expected = T_observed
```

For the diagonal centred transform used here:

```text
C = T_observed ∘ inverse(T_expected)
```

This means ordinary translation/zoom already explained by recent motion is preserved. Only the unexpected residual is eligible for correction.

## Per-axis gating

A single ambiguous affine fit must not authorize a four-parameter warp. Each of `sx`, `sy`, `tx`, `ty` is gated independently.

An axis is corrected only when all relevant checks pass:

- the fitted boundary is materially better than the natural-motion prediction;
- the axis residual exceeds a fixed estimator-noise floor;
- the residual also exceeds `2.5x` recent-motion dispersion;
- restoring that axis to the expected-motion value measurably worsens the objective;
- the residual remains inside conservative safety bounds.

Current floors:

- scale: `0.5%` in log-scale magnitude;
- translation: `0.25` latent pixel.

Current maximum correction:

- scale: `3%` per axis;
- translation: `min(1.5 latent px, 2.5% of the spatial axis)`.

The whole boundary must improve by at least `5%` relative to the predicted natural transform and end with aligned error no worse than `0.25` before any axis can apply.

These are conservative engineering gates, not perceptual calibration constants. Do not loosen them merely because a real run rejects.

## Persistent suffix policy

The previous draft used a three-token `(1, 0.5, 0.25)` decay. That policy is removed.

The first real metrics indicate the generated suffix itself adopts the shifted framing while the learned prefix remains aligned. A short decay back to identity could therefore create a delayed zoom/wobble a few latent tokens after the seam.

When the motion-residual candidate is accepted, the same **residual-only** transform is applied to the complete learned generated suffix before re-noising. This is deliberately different from warping the raw observed boundary transform:

- natural camera/object motion remains in `T_expected` and is not cancelled;
- only statistically supported residual axes are non-identity;
- the correction is spatially coherent across the suffix rather than creating a second transition;
- the authoritative prefix remains untouched.

This persistent policy is still an empirical hypothesis and requires decoded-media validation before merge.

## Geometry then DC

If geometric correction is accepted:

1. estimate transfer-prefix geometry diagnostically;
2. estimate recent exact-prefix motion;
3. estimate the generated boundary relative to that motion;
4. apply only the accepted residual axes to the learned suffix;
5. use the learned prefix (or its accepted same-time diagnostic alignment) for DC calibration;
6. apply the existing one-token DC relation;
7. re-noise and restore the authoritative exact prefix.

If geometry is disabled or rejected, the runtime returns to the released v0.3.2 inverse-recovery/DC arithmetic. That path remains covered by bitwise regression tests.

## Metrics

`mixed_grid_geometry` now carries:

- `source_hw`, `target_hw`, independent grid scale and anisotropy;
- `prefix_registration`;
- `transfer_bias_candidate`;
- `natural_motion_model` with transition fits, trend and dispersion;
- `boundary_before` including error under the expected natural transform;
- `motion_residual` including raw residual transform, per-axis significance thresholds, objective gains, selected axes and applied transform;
- `requested`, `accepted`, rejection reason and policy;
- corrected-token count;
- out-of-bounds fraction;
- `boundary_after_geometry`;
- pre/post geometry DC residual;
- learned-prefix unchanged flag;
- final exact-prefix result supplied by the existing runtime.

`mixed_grid_transfer` and `mixed_grid_complete` continue to report the existing seam/DC measurements and final accepted-latent boundary registration.

## Interpretation

The metrics are intended to separate these cases:

### Transfer bias

`prefix_registration` shows a stable nonidentity same-time transform. This implicates learned transfer/context geometry. It is diagnostic only in this revision.

### Generated-trajectory framing residual

The learned prefix is aligned, but `boundary_before` differs materially from the robust recent-motion prediction. This is the condition the new bridge can correct.

### Natural motion

The observed boundary is consistent with the extrapolated recent prefix motion, or the residual is inside recent dispersion. The bridge must no-op.

### High-stage/decode candidate

If the clean learned boundary is normal but the final accepted boundary changes, inspect target-grid high-stage evolution. If latent boundaries remain normal while decoded media jumps, inspect temporal VAE/decode/assembly.

## Validation

Keep PR #24 draft.

Use the same workflow/seed/prompt/references/source scale/sampler/VDN/Spectrum/DC settings:

1. A: `suffix_geometric_bridge=false`, `suffix_dc_bridge=true`.
2. B: `suffix_geometric_bridge=true`, `suffix_dc_bridge=true`.

Return both decoded videos, full logs, H3 Flow metrics JSON, and Decode Context reports.

Inspect:

- top-edge reveal;
- whole-frame zoom/shift;
- edge stretching;
- whether ordinary camera/object translation is preserved;
- any persistent crop error;
- ghosting;
- any delayed wobble/second boundary;
- face/background detail;
- temporal motion;
- audio continuity.

Do not call the visible issue fixed until the corrected decoded-media run demonstrates it.
