# Experimental geometric Continuum seam bridge

Progressive Mixed-Grid Continuum has two distinct exact-boundary failure modes:

1. a per-channel DC mismatch, handled by the released one-token `suffix_dc_bridge`;
2. a whole-frame framing mismatch, visible as a small shrink/zoom-out and top-edge reveal.

The DC bridge cannot correct scale or translation. The geometric path remains **experimental and off by default**. Decoded media, not latent metrics alone, is the release gate.

## What the measurements show

The instrumented path uses a genuine low-grid generated suffix, an authoritative target-grid Continuum prefix, learned 3D latent transfer, exact-prefix restoration, and a fresh target-grid refinement stage.

The same-time authoritative-prefix / learned-upscaler-prefix registration remains close to identity, so the learned 3D upscaler is not supported as the dominant origin of the framing jump. The framing signature already exists at the genuine low-grid protected-prefix -> generated-suffix boundary and is independently visible on the target grid.

The earlier measured `sy` case was:

| domain | signed log-scale residual | residual scale | evidence |
| --- | ---: | ---: | ---: |
| source low grid | `-0.0253151371` | `0.9750026041` | `1.95311695` |
| target learned grid | `-0.0170032489` | `0.9831404905` | `1.83264785` |

Its engineering concordance score was `hypot(1.9531, 1.8326) ~= 2.6783`, above the configured `2.5` threshold, with both domains above the `1.25` minimum. This is an **engineering concordance score**, not a p-value or an independence claim.

The later accepted decoded-media B run (`metrics_00271`) strengthened the same conclusion:

```text
source sy signed residual     -0.0286183
source sy evidence             2.6971
target sy signed residual     -0.0204533
target sy evidence             2.7867
combined engineering score     3.8782
```

Only `sy` was cross-grid authorized. Target-only nuisance fits were not promoted.

## What the accepted B run disproved

The first transient redesign correctly authorized and applied the early `sy` correction. It used the measured source temporal state:

```text
boundary  -0.0286183
followup  -0.0001794
followup  +0.0300705
followup  +0.0380460
followup  -0.0008663
```

which produced an observed-recovery envelope:

```text
1.0, 1.0, 0.0, 0.0, ...
```

The pre-refinement target transform was `sy ~= 0.9797545`; two early suffix tokens were corrected; the authoritative prefix remained exact.

That was still not sufficient. The fresh target-grid refinement stage reintroduced a measurable framing residual at the **actual returned target latent**. The final boundary fit in that run was approximately:

```text
raw final transform     sx=1.0025  sy=0.9850  tx=-1.0  ty=+0.25
expected natural motion sx=0.9839  sy=0.9951  tx=-1.40625 ty=-0.1125
final sy residual       log(0.985 / 0.9951473) ~= -0.010249
final sy evidence       ~= 1.396
```

So a one-shot correction before re-noise/refinement is not a closed-loop solution. High-stage dynamics can partially recreate an already-authorized residual. This is the specific failure addressed by the final residual closure below.

The run also showed why a single scalar seam metric is insufficient. Pre-high geometry + DC improved the exact-restored splice relative to the uncorrected exact splice, but the final high stage raised full seam RMS again even while low-pass/DC components improved. The visible framing defect therefore has to be checked on the final returned latent and decoded media.

## Spatial authorization and temporal state

The initial bridge still separates **spatial authorization** from **temporal shape**:

1. Fit recent protected-prefix motion independently on source and target grids.
2. Measure source and target boundary residuals independently.
3. Cross-grid authorize only shared axes with compatible sign, magnitude and evidence.
4. Inspect genuine source suffix transitions only for those authorized axes.
5. Integrate the measured source residuals into a per-axis temporal state.
6. Classify each axis as `persistent`, `recovering`, `ambiguous`, or `inactive`.
7. Use target-grid residual magnitude for the geometric transform.
8. Use source-grid state only for the temporal envelope.

The source trajectory decides **when** a correction is allowed. It does not blindly provide target-grid transform magnitude.

## Evidence and safety gates

The existing conservative gates remain unchanged:

- scale estimator floor: `0.005` signed log-scale;
- translation estimator floor: `0.25` latent px;
- maximum scale residual: `log(1.03)`;
- maximum translation: `min(1.5 latent px, 2.5% of the spatial axis)`;
- minimum evidence per domain: `1.25`;
- initial combined source/target engineering evidence: `2.5`;
- source/target magnitude compatibility ratio: `0.35 .. 2.85`.

The final closure does not weaken these gates or authorize new axes.

## Measured temporal envelope

For an authorized source axis with boundary residual `b` and genuine follow-up residuals `r_k`:

```text
s0 = b
sk = s(k-1) + rk
raw_k = clamp(sk / b, 0, 1)  # while the original sign remains
w0 = 1
wk = min(w(k-1), raw_k)
```

A sign crossing or state at/below the existing estimator floor is terminal recovery. There is no handcrafted `(1, 0.5, 0.25)` fade and no extrapolated recovery.

Persistent axes require at least two usable follow-ups whose cumulative state remains within estimator resolution of the boundary state. Divergence, material reversal, unresolved partial drift, or non-contiguous evidence rejects the axis.

For scale `a` and weight `w_k`:

```text
effective_scale_k = exp(w_k * log(a))
```

For translation `d`:

```text
effective_translation_k = w_k * d
```

## Two-stage geometric closure

### Stage 1: clean handoff correction

The initial correction still occurs on the learned clean target-grid transfer before DC and re-noise:

```text
learned clean transfer
-> initial cross-grid geometric correction
-> one-token DC bridge
-> conditional re-noise / target-state mapping
-> fresh target-grid refinement
```

Only source+target authorized axes are changed. The authoritative prefix is never warped.

### Stage 2: final returned-latent residual closure

If and only if Stage 1 was accepted, the final high-stage sampler result is checked again **after the fresh target-grid refinement has returned**.

The final closure is deliberately narrower than a second estimator:

1. restore/canonicalize the authoritative exact prefix first;
2. reuse the original source/target cross-grid axis authorization;
3. reuse the original measured source temporal envelope;
4. re-measure only the current target-grid residual magnitude at the final boundary;
5. require the current target residual to remain provisional under the original estimator and to contribute at least the original `1.25` target-domain evidence floor;
6. require its sign to agree with the originally authorized source residual;
7. apply the current target magnitude only on the already-authorized early suffix tokens;
8. re-measure the boundary immediately;
9. retain the projection only when every applied axis has strictly smaller residual magnitude and no material sign overshoot beyond estimator resolution.

If any of those checks fails, Stage 2 returns the original final sampler tensor object. It does not guess or promote a new axis.

This closes the demonstrated loop:

```text
initial source/target authorization
          |
measured source temporal envelope
          |
pre-high target correction
          |
      high refinement
          |
measure actual returned target residual
          |
reapply only still-present authorized residual
          |
verify residual reduction or revert exactly
```

## Why the final closure is placed after high refinement

Applying a geometry projection only to pre-high `x0` assumed the subsequent target-grid solver would preserve that correction. `metrics_00271` disproved that assumption.

Applying geometry inside each high-stage H3 prediction would instead change the solver trajectory and could couple the experiment to Spectrum history/forecast behavior. The final returned-latent closure avoids that: it runs after the sampler lifetime, introduces no model call, and cannot alter Spectrum's actual/forecast decisions or history.

It is still a latent-space correction, not a decode-space patch.

## Vectorization and exactness

Both geometry stages use the same batched `affine_grid` / `grid_sample` suffix operator:

- only the active early suffix interval is resampled;
- recovered later suffix tokens remain bit-exact;
- persistent axes may keep the full suffix active;
- operation is float32 internally and returned to the original dtype;
- protected prefix tokens are never included in the warp.

The final closure runs only after exact-mask canonicalization and then verifies the protected mask remains exact.

## Unchanged runtime contracts

The feature does **not** change:

- H3 transformer NFE;
- Spectrum history or actual/forecast accounting;
- VDN external-sequence API v2;
- caller noise;
- masks;
- conditioning;
- audio;
- VAE/decode behavior;
- generic Progressive or Target-Sparse UI.

No optical-flow model, extra VAE pass, tensor/image crossfade, decode-space trick, or additional transformer evaluation is introduced.

`suffix_geometric_bridge` remains Mixed-Grid-only, optional, and `false` by default.

## No-op behavior

With geometry disabled, the final-closure code path is not entered and the released DC-only arithmetic is unchanged.

With geometry requested but rejected at Stage 1, the final closure is not authorized.

At Stage 2, missing initial authorization, weak current target evidence, sign disagreement, missing temporal evidence, non-finite state, or a projection that does not reduce every applied residual returns the original final tensor object.

## Diagnostics

`mixed_grid_geometry` continues to report Stage 1:

- source/target natural-motion models;
- provisional residuals and per-axis evidence;
- cross-grid sign/magnitude/evidence authorization;
- measured temporal state/envelope;
- target transform and corrected-token count;
- exact-prefix status and boundary diagnostics.

The final closure adds `mixed_grid_final_geometry`, including:

- `initial_axis_authorized`;
- `current_target_motion_residual`;
- current final target evidence and signed residual;
- per-axis final authorization reason;
- final target base transform;
- effective active temporal length and corrected-token count;
- boundary before and after final projection;
- residual magnitude ratio after projection;
- exact-prefix preservation;
- an explicit accept/reject reason.

The ordinary `mixed_grid_complete` event is emitted after this closure, so its final seam metrics describe the actual tensor returned downstream.

`suffix_persistence` remains a compatibility alias for `temporal_profile`; new consumers should use `temporal_profile`.

## Validation requirement

Structural tests must cover both stages, exact-prefix invariants, recovered-tail exactness, target-only nuisance rejection, sign/evidence rejection, projection verification/revert, disabled no-op, audio preservation, unchanged NFE accounting, and the native/sibling source contracts.

Those tests prove runtime structure, not visible quality.

After the exact final PR SHA is green, the next decoded B must keep the previous workflow fixed and run:

```text
suffix_dc_bridge = true
suffix_geometric_bridge = true
```

The decisive result is the final decoded join. It must remove the shrink/top-edge reveal without a delayed zoom, wobble, edge stretch, detail loss, or motion regression. The PR remains draft and the feature remains off by default until that is demonstrated.