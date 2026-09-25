# Exact-prefix residual geometry: measurement and gated refinement

Audit: 2026-09-25. Status: implementation specification with a mandatory measurement-first hardware gate. This document adds no runtime correction and does not establish that latent scaling improves decoded video.

## 1. Decision and scope

Preserve the successful `paired_prefix_rigid_v2` transaction. Add an opt-in, bounded regional diagnostic after rigid registration, then permit an independently gated horizontal residual scale/translation **only after** the diagnostic hardware stage supports that model. The first implementation milestone is measurement-only and must be numerically identical to rigid v2. No general affine, projective, local deformation, or automatic model escalation is authorized by the present evidence.

The smallest supported production model remains rigid translation. Horizontal scale plus translation is the first model to test, not a demonstrated remaining root cause. The architecture is hierarchical: existing rigid decision → residual measurement → model eligibility → independent video/guidance validation → optional composed suffix transform. A residual failure retains the successful rigid result. A failed original v2 transaction retains its existing production-baseline behavior.

Scope is the protected partitioned learned handoff. Ordinary/no-prefix paths, source-prefix physical projection, attention implementations, sampler schedules, and unrelated diagnostic PRs remain outside the change.

## 2. VERIFIED FACTS

### Source identities and ownership

| Item | Audited identity / result |
| --- | --- |
| Flow production `main` | `a6249b8343becc1458a4555d2bb25cd523983c90` |
| Flow draft PR #89 | `41b7da7d7303cb764bda1e8d41c6fa0c2c7f9dbc`, branch `candidate/frame-shift-exact-prefix-gauge-20260924`; open, unmerged, mergeable, exactly one commit above main |
| Candidate CI | Run `36126385474`, completed/success; combined status also reports CodeRabbit success |
| Preceding authoritative design | `docs/EXACT_PREFIX_SUFFIX_FRAME_GAUGE_REPAIR_DESIGN.md` at `4342a277051fdf72bcf2ebecf350195cdebae4f3`; this document supplements it and the implemented v2 policy |
| Instructions/compatibility | No `AGENTS.md` in the audited Flow tree; `docs/DEVELOPMENT.md` inspected; Python >=3.10, test Torch >=2.5 |
| Sole external source dependency inspected | Continuum `v3/trajectory_diagnostics.py`, its affine tests, and the measurement/assembly call site in `v3/assembly.py` at `5e1b18a78c1f91ca73cb98145dded5ae6259666d` (affine diagnostic branch). Needed to interpret PT224, not to audit Continuum broadly. The log does not prove this exact executed commit. |

Main and #89 match the task's supplied hashes. The preceding design resides on its documentation branch, not in the candidate tree. PR #89's current implementation and v2 split acceptance policy take precedence over superseded v1 thresholds in that design. No production or diagnostic PR was modified by this investigation.

Relevant Flow source owners at the candidate commit:

- `frame_gauge.py`: same-time prefix registration, bounded search, regional/frame/parity checks, translation and validity.
- `partitioned_scheduler.py`: `_frame_gauge_clean_postprocess`, `_frame_gauge_boundary_motion_check`, `_prepare_registered_guidance_reference`, pre-high restoration and stage observations.
- `handoff.py`: actual provider-clean postprocess hook before `conditional_renoise_target`; mutation guards and deterministic noise ownership.
- `guidance.py`: invocation-local `RegisteredGuidanceReference`, target-grid temporal correspondence, protected-prefix and validity masking.
- `tone_bridge.py`: one-token spatial-mean DC correction.
- `partitioned_runtime_gate.py` and `tools/check_partitioned_runtime_evidence.py`: receipt and hardware-control validation.
- `tests/test_frame_gauge.py`, `test_partitioned_frame_gauge.py`, `test_handoff.py`, `test_guidance.py`, `test_partitioned_runtime_gate.py`: relevant test entry points.

### Hardware witness 00663

The following was read from `metrics_00663_.json` and `Pasted text(20260925-112003).txt`. These are non-git hardware evidence. The visual verdict is the user's observation; no decoded media or raw calibration tensors were available for independent visual/registration reproduction.

| Receipt | Value |
| --- | --- |
| Transaction | accepted; spatial warp applied; `learned_suffix` and `derived_guidance_suffix` transformed |
| Video correction | dx=-0.4375, dy=-0.5625 target latent cells |
| Guidance correction | dx=-0.4375, dy=-0.4375; active `direction+temporal` |
| Actual clean operand | FP32, CUDA; target H/W=56/74; 12 prefix frames, selected 6–11, fit 6/8/10, holdout 7/9/11 |
| Video validation | NCC 0.93469682; RMS improvement 0.06907367; runner margin 0.07061402 |
| Guidance validation | NCC 0.98753242; RMS improvement 0.21993768 |
| Boundary error full | 0.78470067 → 0.06813602 cells, 91.3169% reduction |
| Boundary error upper45 | 1.02501120 → 0.07303868 cells, 92.8744% reduction |
| Prefix/noise | `final_prefix_exact=true`, `protected_video_noise_exact=true`; authoritative-prefix modification false |
| Work | 18 logical, 14 actual H3 NFE, 4 forecasts, 6 sampler invocations, 4 history boundaries; extra H3/provider/VAE/sampler/history work reported zero |
| Invalid translated area | 0.03112936 |
| Timing | frame-gauge transaction 18,706.677 ms; continuation sampler 223,601.976 ms; preceding chunk 260,073.222 ms |
| Auto-strength | unknown; formal matched-control acceptance remains open |

The transaction is about 8.37% of continuation sampler wall time, or 3.87% of both sampler walls combined. These ratios are not matched incremental overhead measurements, but they rule out treating the existing estimator as cost-free or assuming the preceding design's 1% target was demonstrated.

Decoded measurements at global frame 175 / trim frame 39:

| Metric | Full | Upper45 |
| --- | ---: | ---: |
| First PT212 dx/dy, pixels | +0.016412 / +0.028986 | +0.027960 / +0.047528 |
| PT224 scale x / y | 0.98956669 / 0.99992791 | 0.99245456 / 0.99968462 |
| PT224 translation x / y, pixels | +3.375420 / +0.071906 | +2.869116 / +0.115708 |
| x fit residual, pixels | 2.377846 | 3.062211 |
| x scale confidence | 0.329162 | 0.151337 |
| Pre-boundary median scale x | 0.99483263 | 0.99671954 |
| First-three median scale x (includes boundary) | 0.99504173 | 0.99618465 |

Both PT212 observations classify as continuous-shot candidates. PT224 is not a general affine registration: it fits dx against x and dy against y separately from overlapping 3×3 local phase-correlation patches, at a comparison long side <=512. It ignores cross-axis terms. Its “scale confidence” is fractional fit-residual reduction against translation-only, **not a probability or confidence interval**. Response weights are clipped to [1,50]; clipped patch counts are recorded by the helper but not printed in the inspected PT224 log. The inspected synthetic scale test allows ±0.015 x-scale error, wider than this run's signal. This does not disprove the estimate; it does not validate sub-percent accuracy either.

The full boundary/pre-median scale ratio is about 0.994707, upper45 0.995721. Thus approximately 0.53%/0.43% is an illustrative *excess* relative to those pre-medians; the whole 1.04%/0.75% cannot be assigned to the handoff. These ratios are descriptive, not counterfactual motion estimates or correction parameters.

### Stage localization actually available

First boundary translation (dx,dy), target latent cells except source rows:

| Stage | Upper45 | Full |
| --- | --- | --- |
| source low native | (+0.021654,+0.006229) | (-0.230162,-0.132799) |
| source low exact context | effectively identical to source native | effectively identical to source native |
| learned native | (+0.017414,-0.057427) | (-0.031995,-0.068870) |
| aligned learned witness | (+0.023899,-0.055431) | (-0.046975,-0.066924) |
| exact restored, before high | (+0.015377,+0.015583) | (-0.082239,-0.022847) |
| final after high | (+0.018310,-0.011331) | (-1.596041,+0.455170) |

The boundary gate's uncorrected exact-restored counterfactual was (+0.516512,+0.837865) upper45 and (+0.123105,+0.700350) full. It directly supports the rigid repair's effect before high. Source/native equality argues against an additional source-prefix substitution translation in this run, but says nothing decisive about scale.

The final full-frame latent estimate differs strongly from its upper45 and decoded counterparts. Moving foreground, representation sensitivity and stage effects remain alternatives; multiplying latent shifts by 16 is not a valid decoded-motion inference. Existing stage diagnostics measure translation, not regional scale. Furthermore, `suffix_aligned_before_dc` emits the **same aligned-witness trajectory** as `paired_prefix_aligned_witness`; it is not an independent exact-restored, pre-DC boundary observation. Future evidence must disambiguate these tensor owners without rewriting historical receipt meanings.

PT224 is measured from previous assembled frames and current raw decoded retained frames **before** the current segment is copied and any video patch applied. Therefore its signal exists before that current patch at the inspected call site. It does not exclude previous-segment patching, decode context, temporal trimming, or later assembly effects. The exact executed diagnostic/assembly revision and final pixels are not established by marker names alone.

## 3. ARCHITECTURAL CONCLUSIONS

The remaining error has not been localized uniquely to learned synthesis, restoration, high continuation, decoder or assembly. The rigid coordinate mismatch was present before high and was largely corrected there. Regional residual geometry needs a same-time latent witness and a cross-time, native-motion-preserving boundary witness; decoded adjacent-frame fits alone cannot choose a latent warp.

Asymmetry does not by itself require perspective. For decoded forward geometry `x_next=s*x_prev+t`, displacement is `(s-1)*x+t`, with effective stationary x coordinate `p=t/(1-s)`. The reported fits imply p≈323.5/380.2 pixels. A pivot toward the left can make right-edge motion much larger than left-edge motion. However horizontal-only geometry predicts the same dx at top and bottom for a given x. Only measured top/bottom differences at matched x can establish shear or locality; visual salience and unavailable texture can hide left-edge movement. A pure centered zoom is unproven. These noisy fitted pivots are explanatory examples, not physical camera pivots.

The regional learned correction estimates (left -0.25 uninformative; right -0.625 informative; upper dy=-0.6875; lower dy=-0.375) suggest spatial dependence but do not establish a horizontal-only field. A negative x slope in **latent correction** has a different meaning from negative x slope in **decoded forward motion**. Never copy or negate PT224 scale into the latent correction. Independently measure the correction field and validate its decoded consequence.

Keep clean target-grid state as the candidate correction domain. No evidence supports changing the existing noise, exact prefix, scheduler history or model work. Preserve v2 as a separately testable control; measured residual identity/rejection must not disable a valid rigid transaction.

## 4. Minimum diagnostic implementation

### Public mode and stage ownership

Append an optional serialized setting `frame_gauge_residual_mode` with enum `off|measure|horizontal`, default `off`; missing old-workflow input resolves to `off`. Require `frame_gauge_repair=true` for measurement/application; otherwise report residual not evaluated and take the exact existing OFF path. First milestone exposes only `off|measure`; add `horizontal` only after the hardware gate below. No hidden environment activation or auto-enabling existing workflows.

In measure mode, observe native L and exact E in the existing clean postprocess hook; use the accepted rigid parameters but never replace its selected tensors, DC computation or reference. If v2 rejects/is identity, report `not_evaluated: rigid_not_accepted`; do not widen eligibility. Observe guidance G only after its existing independent v2 registration succeeds. Do not repeat the provider or guidance-reference selection. Diagnostic arrays must be detached and invocation-local.

Record these explicit views, at most last six prefix and first four suffix tokens:

1. Same-time `L_prefix -> E`, both raw and analytically rigid-aligned; analogous `G_prefix -> E`.
2. Boundary pairs: L_last/L_first (native), E_last/L_first (unregistered), E_last/rigid-L_first (before DC), E_last/rigid-L_first-plus-DC (actual pre-high clean). Construct read-only pair views; no extra complete video clones.
3. Final clean E_last/final-suffix-first plus the next three available suffix pairs from the existing post-high output. Include pre-boundary pairs as motion controls.
4. For diagnosis, compare aligned learned-last/aligned learned-first separately from E_last/aligned learned-first. The former transports native motion; the latter tests prefix substitution.

Each receipt states stage, operand owners, same-time versus adjacent-time, temporal indices, dtype, H/W, normalization, downsampling, valid support and applied transform. Keep actual-clean and inverse-recovered states distinct. Do not reconstruct the new estimator's inputs from noisy Y.

### Regional observations

Use a disjoint 3×3 partition of the v2 interior support (margin five latent cells), giving TL/TM/TR, ML/center/MR, BL/BM/BR. Use identical coordinates across frames, preserve the four H3 parity phases, and cap the union at 96 samples per axis. Quadrants, upper/lower and left/right are summaries/validation views; overlapping summaries never count as additional independent observations. Require >=8 samples on each tile axis and >=8 textured channels, otherwise mark unavailable. No empty/ambiguous tile is a zero displacement.

Normalize/smooth as the existing estimator, sharing preparation within the new diagnostic without changing v2 arithmetic. In each tile fit residual pullback displacement relative to fixed d: compare E(r) with L(r-d-u). Evaluate directly from original normalized L, not a twice-resampled prefix. Use a bounded ±0.5-cell local search, spacing 0.125, then ±0.125 around the optimum at 0.03125 spacing (at most 162 scores per tile/frame; deduplicate). Reject a saturated optimum, nonfinite score, low texture, NCC<0.75, or an unresolved competing minimum separated by >=0.25 cell with loss margin <5% of rigid residual loss (denominator floor 1e-8). An almost-zero rigid residual is identity/insufficient evidence, not infinite confidence. Search bounds are proposed diagnostic policy, not thresholds validated on 00663.

Emit displacement, zero/best loss, NCC, runner separation/margin, valid pixels/channels, texture in both axes, saturation and uncertainty. For each axis, derive a deterministic near-optimal displacement envelope from all candidates within 5% of the zero-loss denominator above the minimum, plus the 0.03125-cell quantization floor. This is a sensitivity envelope, not a statistical CI. Require each half-width <=0.125 cell for model selection. Record geometric center and gradient-energy-weighted effective center; require adequate support on both sides of that center. Use equal accepted-tile weights for model fitting; do not let high-texture foreground dominate by response weight.

Use v2's disjoint fit/holdout temporal indices. Fit model parameters on fit frames only; freeze them before evaluating holdout frames, including the last prefix frame. Require >=2 fit and >=2 holdout frames with usable left/center/right support in both upper and lower bands. Do not count three tiles in one textured corner as global support. Failure to meet this stricter support is a residual rejection only.

### Fits and discriminating evidence

Fit inverse residual fields at output coordinates r=(x,y), centered at c=((W-1)/2,(H-1)/2):

| Model | Residual field | Eligibility |
| --- | --- | --- |
| Identity / residual rigid null | u=(0,0), and separately u=(bx,by) | mandatory null comparisons; residual constant does not authorize another translation refinement |
| Horizontal | ux=a(x-cx)+b; uy=0 | sole candidate application model after gate |
| Axis scales | ux=a(x-cx)+bx; uy=e(y-cy)+by | diagnostic only; coherent nonzero y dependence would reject horizontal |
| Similarity | common scale plus antisymmetric off-diagonal terms | diagnostic only; require reproducible common scale/rotation, not merely smaller SSE |
| Affine | ux=a(x-cx)+h(y-cy)+bx; uy=k(x-cx)+e(y-cy)+by | diagnostic only; cross-axis terms require matched-x/y support |
| Local/projective | no production optimizer | report unexplained residual; require additional spatial coverage and a separate design |

Use deterministic least squares in float64 on normalized coordinates and report rank/condition number. Reject rank-deficient fits or condition number >100. Fix tie order from simplest to most expressive. Calculate leave-one-tile-out predictions on fit frames and frozen predictions on temporal holdouts. Emit max tile error, RMS, worst-corner error and parameter envelopes under frame/tile deletion. Do not call this independent statistical replication: tiles within a frame and channels are correlated.

For horizontal eligibility require all of the following initial conservative gates:

- The zero-slope value is excluded by the full deletion/sensitivity envelope; edge-to-edge displacement signal >=0.125 cell and at least twice its uncertainty. Same slope sign in independently fitted upper and lower bands and every informative heldout frame.
- Horizontal versus residual-constant null reduces holdout displacement RMS >=25%, with absolute RMS <=0.125 cell and worst informative tile <=0.25 cell. No corner may degrade beyond its measured uncertainty. Direct heldout feature NCC and RMS must not degrade relative to rigid; fit improvement alone never suffices.
- Upper/lower slope predictions agree within 0.125 cell at both x edges. Observed uy and any cross-axis component must be consistent with zero within uncertainty and bounded by 0.125 cell. If not, horizontal rejects even if its x fit improves.
- Fitted coefficients must remain within section 5 bounds under every deletion fit, and all informative parity-phase checks must agree within 0.125-cell predicted displacement. Insufficient last-frame support rejects.
- Native-motion boundary validation must pass regionally, as specified below. The successful v2 full/upper45 gate remains an independent prerequisite.

These numbers are implementation starting assumptions requiring synthetic accuracy/false-positive calibration and the measurement hardware stage. Do not lower them solely to make 00663 pass. A changed threshold needs a versioned policy, independent fixture evidence and recorded reason.

## 5. Conditional horizontal correction mathematics

Only implement/activate this section after section 9's diagnostic gate. Parameters describe a **correction**, not observed inter-frame motion. Rigid content translation is `T_d(r)=r+d`. Let residual pullback `R^-1(r)=r-u(r)` with `ux=a(x-cx)+b`, uy=0. Then final content map is `F=R∘T_d`, so:

```
F^-1(x,y) = (x-a*(x-cx)-b-dx, y-dy)
W_F L(x,y) = L(F^-1(x,y))
sx = 1/(1-a)
tx_residual = (b-a*cx)/(1-a)
pivot_x = cx-b/a    # undefined/omit when a is indistinguishable from zero
```

The total forward x translation is `(dx+b-a*cx)/(1-a)`. Store both forward and pullback matrices and verify their product is identity. This convention avoids inferring a physical pivot or optimizing an ill-conditioned pivot near identity. Positive dx moves content right; residual positive a expands forward geometry. Fit tile observations as pullback corrections in output coordinates, then verify directly against original features so centroid approximations cannot determine acceptance.

Bounds: forward sx in [0.985,1.015], sy=1 exactly; residual b within ±0.125 cell; residual displacement <=0.5 cell everywhere; total displacement <=2 cells on each axis at all image corners; rotation/shear/projective coefficients exactly zero. Require <=8% invalid area for the total map. Bounds do not authorize scaling in their absence of evidence. Identity means max residual displacement <=0.03125 cell or unresolved nonzero slope; use the exact existing rigid result without resampling.

Sample original clean suffix once with the composed inverse grid, FP32 bilinear, `align_corners=False`, normalized source coordinates `2*(sx_source+0.5)/W-1` and analogous y; cast back to original dtype. Use border replication without wrap. Validity is true iff every nonzero-weight bilinear source contributor lies inside the original grid; no replicated border counts as valid evidence/guidance support. Derive it from composed source coordinates, not intersection of unrelated destination masks. Apply in batches of <=4 frames. No sequential rigid-then-residual interpolation in the committed candidate: the hierarchy is in decisions, while final application uses one composed sampling operation. Rigid fallback still calls the original v2 path for byte identity.

Transform the disposable learned prefix witness with the same F only to calculate/check registration and existing DC. Never transform E. Apply the one-token per-channel DC bridge once, using the final aligned witness, before conditional re-noise. Preserve the hook's restoration of the original learned prefix; scheduler then restores E under its existing ownership. Noise remains `Y=(1-sigma)*L_corrected+sigma*N`; neither N nor Y is spatially sampled. Raw audio, caller masks, original protected noise and returned protected latents remain exact. No temporal taper or later inverse camera move is introduced.

### Boundary acceptance and native motion

For nonrigid geometry, compare candidate boundary motion `M(E_last,W_F L_first)` to **transformed native motion** `M(W_F L_last,W_F L_first)`, on common valid regional supports. A scale changes displacement units; comparing only to untransformed native motion can penalize genuine motion or accept cancellation. Keep the original v2 boundary gate unchanged and additionally require candidate full/upper45 receipts to remain valid and within 0.125 cell of their rigid counterparts' native-motion error.

For each informative tile compare exact-restored versus transformed-native residual at rigid and candidate maps. Require all informative tiles to be non-degrading within their displacement uncertainty; require >=25% error reduction in each affected horizontal side whose rigid error >=0.125 cell. Require evidence on both horizontal sides and both upper/lower bands. Low-amplitude or textureless tiles cannot justify scaling. Measure E-restored pre-DC and actual post-DC boundaries separately to detect a measurement dependence on DC. Require final direct feature checks on the candidate, not just regional coefficient predictions.

## 6. Guidance and transaction ownership

Video L and guidance G use different mappings. Do **not** copy video d/a/b into G. Register G/E independently with its existing strict >=15% RMS gate, then independently measure its residual in the same target coordinates. “Corresponding transform” means mapping each source to the common authoritative E gauge; parameter equality is not required. A guidance residual identity can be valid even when video requires scale, provided its identity is positively supported by full spatial/temporal coverage. Ambiguity is not identity.

Retain a ready rigid video/guidance transaction until the residual pair is validated. If either residual registration or guidance support fails, commit the unchanged rigid pair. If original rigid guidance fails, retain v2's existing all-or-nothing baseline fallback. Guidance OFF needs no reference. Never publish speculative registered state before both decisions finish.

Build the accepted reference from original G with its own composed transform, restore protected E prefix, and exclude protected rows and invalid support from direction/acceleration/temporal corrections. Rebuild existing target-grid correspondence once from the selected final reference; do not independently warp cached flow vectors/confidence/innovations. This preserves their common geometry without introducing a new vector-field transformation contract. Keep sampler/reference-coordinate eligibility and temporal-radius cap (8) unchanged.

Extend `RegisteredGuidanceReference` with immutable transform/validity policy identity; retain legacy dx/dy meaning as rigid components. Include composed coefficients, geometry, selected reference identity, run/session/chunk and new policy in cache keys. Clear registered references, temporary measurements and caches in existing success/error/finally paths. No persistent cross-chunk calibration, shared trajectory mutation, retry sampler or new random draw.

## 7. Telemetry and offline validator

Keep existing v2 fields and meanings in off/measure/rigid fallback. Add a bounded `residual_geometry` block to the transaction with policy `paired_prefix_residual_geometry_v1`, independently of `paired_prefix_rigid_v2`. Required fields:

- requested mode; measured boolean; measurement status/reason; selected model; decision `not_evaluated|identity|rejected|accepted`; `applied` boolean; final path `baseline|rigid_v2|rigid_plus_horizontal`;
- video and guidance subreceipts independently, guidance status `off|not_evaluated|identity|rejected|accepted`; fit/holdout indices, tile bounds/centers/texture/uncertainties, counts and bounded model fit results;
- direct-feature checks, deletion envelopes, spatial/temporal holdout errors, boundary regional checks, coefficient bounds, forward/inverse matrices, sign convention and coordinate units;
- source/destination H/W, dtype, operand provenance, validity policy/fraction, DC count/order, transformed-state list, prefix/noise checks and cache identity;
- region-search count, support size, scoring/transfer/application time, incremental allocated CPU/GPU workspace, diagnostic bundle identity if exported.

Measure mode may report a model `eligible` in its measurement details but must say decision `not_evaluated` (application gate not enabled) and `applied=false`. Do not overload `accepted` to mean both fitted and applied. Application requires accepted video and accepted/positively supported identity guidance (or guidance off). Rigid-only counts as repair applied in existing spatial-warp fields, never as residual applied.

Bound per-transaction regional telemetry to <=128 KiB; no tensor serialization into metrics. Emit stage observations in a separately versioned `partitioned_residual_geometry_stage` event keyed by run/chunk. Preserve historical ambiguous labels; new labels carry explicit owners.

Extend `partitioned_runtime_gate.py` with separate legacy-v2 and residual-v1 validators; do not replace a single version constant and invalidate archived v2 evidence. Add CLI `--expected-residual-mode` and `--expected-residual-result`, distinguishing measured-only from applied. Recompute fits/envelopes/bounds and matrix composition from regional receipts; verify heldout membership, rank, required independent support, finiteness, no duplicate tile IDs, common validity and result/application consistency. Hashes/counts corroborate ownership, but offline telemetry alone does not prove tensors were unmodified: retain executable mutation tests.

Add pair comparison for identical input/probe identities, stack/provider/VAE/sampler identities, guidance settings, DoRA reports and low/probe/high work counters. Require measured-only tensor digests to equal rigid control. Keep formal auto-strength-OFF gate intact. A diagnostic-inspection mode may summarize 00663 with `controlled_acceptance=false`; it must not relax the acceptance command or invent a resolved DoRA receipt.

## 8. Tests, resource bounds and fallback

No production patch or new runtime tests were executed by this design investigation. Source and saved receipts were inspected; arithmetic summaries were computed locally. Implementation must add targeted tests, then run the repository's required CI on the consolidated candidate.

Required adversarial fixtures:

1. Analytic continuous textures evaluated at known inverse coordinates (not generated solely by the warp under test): rigid-only, ±0.25/0.5/1% horizontal scale, off-center pivots, natural pan and combined rigid+scale. Require corner-displacement error <=0.0625 cell in well-conditioned noiseless fixtures; otherwise improve diagnostic resolution before hardware.
2. Horizontal identity with DC/gain/detail synthesis differences; repeated stripes, low texture, moving foreground, occlusion, bad last holdout, informative top-right only, left support missing, phase-dependent distortion. Expect residual rejection/identity with byte-exact rigid fallback.
3. Vertical scale, true similarity rotation, shear, projective and localized corner deformation. Report diagnostic alternatives but reject horizontal whenever omitted components exceed bounds; smaller SSE cannot authorize a new model.
4. Fit-frame-only agreement, sign reversals across frames, deletion instability, anisotropic texture, clipped/ambiguous boundary receipts, genuine zoom already present before the boundary. Include modest-confidence PT224-like signals with preexisting contraction; no blind correction.
5. Single composed interpolation against analytic coordinates, identity bypass, border validity, no wrap, odd/aspect-changing support, dtype/device handling, batch equivalence, input alias/mutation guards, inference tensors and exceptions.
6. Independent G/L corrections, positively supported G identity, ambiguous G fallback, protected rows, invalid temporal match support, all guidance modes, unsupported reference schedule, stale-cache and interrupted/reentrant invocation tests.
7. Exact prefix/protected noise/raw audio unchanged, no RNG/provider/VAE/model/executor call increments; DC once with final witness; off and measure outputs byte-equal to v2 across handoff and final state.
8. Validator tampering: forged applied flags, missing last-frame/tile checks, NaN, rank deficiency, out-of-bounds matrix, mismatched reference policy, historical v2 records, missing DoRA receipt, duplicate/overlapping observations miscounted as independent.

Cost: prepare <=6 prefix frames, <=24 channels, <=96² total support points; 9 disjoint tiles, <=162 local candidates per frame, streaming candidates and tiles rather than materializing candidate×frame×channel×pixel tensors. Reuse v2-prepared features where doing so preserves identical existing outputs; do not silently turn bounded scoring support into a claim of bounded full-resolution feature storage. Account explicitly for any full-resolution prefix features and GPU-to-CPU copies. Measure peak memory; the existing reported workspace upper bound excludes some retained witnesses/references and is not a measured allocator peak.

New diagnostic scoring is capped at the work formula above and <=64 MiB incremental CPU scratch at 56×74; scale memory estimates explicitly with input dimensions and fail closed before allocation when configured headroom would be exceeded. Do not retain another complete suffix video for diagnostics. Candidate application uses the existing output buffer plus bounded batches; retain/release rigid fallback material deliberately to avoid simultaneous full G/L alternatives. A production eligibility target is incremental residual overhead <=1% of matched continuation sampler wall and no headroom/OOM regression; measure-only may exceed this but must report it and cannot promote. Existing 18.7 s registration needs separate cost accounting; do not weaken gates or launch unrelated optimization to claim budget compliance.

Data-dependent evidence rejection returns rigid v2 exactly. Invalid tensor shape, corruption or ownership invariant failure raises before high, clears state and never restarts sampling. Resource preflight rejection is explicit. Rollback uses `frame_gauge_residual_mode=off` through the same Patcher overlay; full repair OFF retains current production behavior.

## 9. Staged hardware and media gate

### A. Measurement first

Through the existing PR #89 Patcher overlay, run matched rigid-v2 control (`residual=off`) and `measure`, with identical workflow/input/prefix/probe hashes, seed, model/provider/VAE, schedules, guidance and stack overlays. Resolve every applicable DoRA auto-strength OFF receipt for formal comparisons. Confirm unchanged numerical tensors and call topology. Do not hardcode 00663's counters as universal scheduler targets.

Retain bounded clean evidence: E last six prefix tokens; L/G corresponding prefixes and first four suffix tokens; actual pre-high corrected slice; final corresponding slice; sigma, temporal indices, selected d, dtype/geometry, validity and provenance. These must be raw exact tensor bytes with checksums, not printed rounded values. Export only on explicit diagnostic mode, to the normal evidence output location; never model extra evaluations. The saved prefix is evidence, not transformed production state.

Use the regional stages to classify:

| Observation | Supported localization / action |
| --- | --- |
| Same-time L/E residual is coherent and exact substitution adds matching regional boundary error | latent gauge candidate; test horizontal eligibility |
| Native L boundary already carries the same deformation, while L/E calibration does not | native synthesis/motion; do not derive a gauge correction from adjacent motion |
| Pre-high regional result is clean but final same-region result changes | high continuation/guidance interaction; pre-noise scale not yet justified |
| Latent stages lack coherent geometry but decoded overlap differs at identical global frames | decoder context/representation hypothesis; do not warp latent on PT224 alone |
| Raw decoded retained boundary is clean but assembled output changes | assembly/patch/trim ownership; investigate exact assembly operation |

For decoder/assembly discrimination, retain already-produced raw decoded overlap and retained images, previous retained images and final assembled boundary (first six frames plus preceding context), with global frame/trim/decode-context IDs. Compare same-global-time overlap from the two existing decodes where available, then adjacent motion. Inspect top/bottom at matched x and left/right at matched y using the same regional estimator at two comparison resolutions (e.g. 512 and native crop support). No additional VAE calls are required or permitted in the production run. If existing outputs lack an observation, report that stage as unresolved; any later offline decode experiment is a separately authorized experiment, not hidden work in this candidate.

Do not edit Continuum automatically as part of Flow: use existing returned decode/assembly tensors in saved workflow evidence. If exact raw-versus-assembled export requires a source change, specify the minimum observation-only addition on its own correct diagnostic PR, without absorbing or repurposing existing diagnostics.

Proceed to B only when same-time latent evidence, regional boundary effects, synthetic calibration and raw/assembled observations support the horizontal model without contradictory y/shear/local behavior. If these fail, deliver the diagnostics and the rejected-model evidence; do not force a correction. This is a concrete gated implementation outcome, not permission to substitute a general optimizer.

### B. Bounded correction experiment

Add horizontal mode with the exact gates above. Compare rigid-only versus horizontal on matched inputs, preserving all controls and counting actual applied transformations. An identity/rejected run cannot demonstrate a correction effect. Compare unchanged prefix, no-extra-work topology, audio preservation, timings and peak memory. Repeat one subsequent chunk (lifetime regression) and one other aspect ratio/scene with different motion/texture. Preserve both accepted and rejected cases rather than tuning exclusively to 00663.

Human decoded-media acceptance is mandatory: boundary edge/corner motion must improve at normal playback and frame stepping; top-right improvement must not trade for bottom-left distortion. Inspect first six frames and the whole suffix for zoom breathing, perspective drift, stretching, border smear, ghosting, detail loss or a delayed compensating camera move; preserve natural subject/camera motion and acceptable audio/sync. Use raw retained and final assembled media. Numerical seam RMS or a better affine fit is never sufficient. No default-on promotion or merge until this gate passes.

## 10. Rejected paths and remaining uncertainty

### Preserved conclusions

- 00662 did not demonstrate an ineffective applied correction: v1 rejected on 15% RMS improvement, so no spatial repair executed.
- Lowering that threshold alone was inadequate. v2 added direct native-boundary-motion preservation while retaining strict independent guidance registration; 00663 validates execution and gross improvement.
- A remaining large uniform decoded boundary translation is unsupported by 00663. This does not mean every latent region is motionless.
- General affine/projective/local warps, pure centered zoom and a single decoder root cause remain unestablished, not falsified.
- Never restore the retired full-field additive residual, warp E/noise, or use lower latent seam RMS as visible acceptance.

### UNRESOLVED QUESTIONS

Does a stable same-time residual field exist in actual L/E and G/E after rigid correction? Does it remain horizontal across top/bottom and parity phases? Does it transfer from prefix to generated suffix? How much PT224 signal is natural motion, local synthesis, clipped-patch bias, decoder context or assembly history? Does the composed transform preserve decoded texture and temporal behavior? Can the additional diagnostic meet the overhead/headroom budget? Raw tensors and decoded media, absent from the present witness, are required to answer these.

### IMPLEMENTATION-TIME ASSUMPTIONS TO RE-CHECK

Re-fetch live main/#89, source instructions, candidate changes and CI once before editing. Re-check actual-clean hook domain, exact-prefix/noise restoration, DC ordering, guidance schedule/cache ownership and serialized widget ordering. Resolve exact diagnostic source versions and frame mapping in the saved hardware workflow. Verify the proposed search/uncertainty thresholds against independent fixtures and measurement hardware; they are not empirically established constants. If live evidence contradicts any assumption, document a narrow justified deviation and keep unsupported transforms disabled.

## 11. Repository and handoff discipline

This design belongs on a separate documentation branch/PR targeting production main; it is not an additional implementation commit on #89. Preserve #89 open/draft, unmerged and one clean implementation commit. Keep unrelated diagnostic PRs unchanged. Implementation uses a separate mirror from the then-live #89, with GitHub checkpoints before hostile review, risky work, long tests and after substantial progress. Consolidate only verified in-scope changes onto the existing candidate branch, preserving the one-commit topology and authorship. Deliver runnable changes through the correct ComfyUI Patcher PR overlay; no manual user Git procedure.

Required retained non-git evidence is 00663's metrics/full log, its matching saved workflow/source identities and boundary media if available, plus stage-A/B tensor bundles, controls, DoRA receipts and raw/final media. The two original files support the numerical witness above; they do not contain raw regional tensors or establish a formal controlled media pass. No paper supplies the missing coordinate evidence: this refinement is governed by source ownership and measured state.
