# Exact-prefix / suffix frame-gauge repair design

Status: implementation specification for a gated candidate; **not a verified media repair**. This change is documentation only. Production implementation belongs on existing draft PR #89, subject to the live-state checks below. A rigid translation is a testable model of the defect, not an established property of H3 latents or the learned provider.

## 1. Decision and scope

Implement an opt-in, deterministic **paired-prefix registration transaction** between learned transfer and high-stage continuation. Observe the actual learned clean output before noise mixing. Register its prefix against the same temporal indices of the authoritative target prefix. Apply one bounded sub-cell translation to the generated suffix only, and only when independent fit checks support a common rigid displacement. Retain the exact target prefix and the existing handoff noise realization.

When Flow guidance is active, independently register its existing target-grid reference against that same authoritative prefix. The learned upscaler and the bicubic guidance lift are different mappings; their translations must not be assumed equal. Commit the video and guidance changes together, or execute the unchanged physical-prefix baseline. No additional model evaluation, sampler lifetime, learned-upscaler call, VAE call, shadow sampling trajectory, or stochastic draw is allowed.

Do not deploy the additive full-field residual in current #89 as a proven coordinate repair. Its identities are algebraically valid, but neither those identities nor seam RMS establish spatial continuity. Historical hardware already rejected the same residual principle as a complete media fix.

Freeze DoRA **auto-strength OFF** in every frame-shift comparison. The user reports that disabling it restores coherent background structures previously rendered as black blobs. This is user-supplied hardware observation, not an independently reproduced causal experiment. No inspected historical run establishes the resolved OFF state. Why exceptional run 00625 behaved differently remains unresolved and is outside this repair.

## 2. Source and topology audit

The audit fetched the repository and all advertised branches/tags, then inspected these sources and PR discussions/reviews. No `AGENTS.md` was present in the Flow tree or the inspected historical snapshots. Flow supports Python >=3.10; its test extra specifies Torch >=2.5. Retain those constraints.

| Owner | Audited identity | Disposition |
| --- | --- | --- |
| Flow main / v0.3.8 | `a6249b8343becc1458a4555d2bb25cd523983c90` | Production baseline |
| Physical-both 00625 runtime | `e57f320b3ece93e3289ebcccd4104ea6a1265526`, tree `61bce93066b30c9923d8a99a0eec0ea80d0f54e5` | Required numerical starting point |
| #81 | Closed, unmerged; reported head `f1ac4c0825e7cb7674349c5e6cd03127be063dd1` | Historical investigation; reused branch makes PR body/head insufficient provenance |
| #84 | Closed, unmerged; `17364fa6362e50229c4159797e3f9b1badc95f1a` | Exact 00625 tree restored after multiple rejected candidates |
| #86 | Closed, unmerged; same `17364fa...` | Pristine replay, not a new implementation |
| #87 | Open draft; `mirror/00625-first-chunk-provenance-20260923`, `e57f320...` | Preserve as literal physical-both control |
| #88 | Open draft; `mirror/00625-first-low-model-provenance-20260924`, `f883adebd30925ef2f1f86bc0bd5c716703cdb59` | Diagnostic only; preserve |
| #89 | Open draft; `candidate/frame-shift-exact-prefix-gauge-20260924`, `ae1a7795254dec213a35bf09d83e4b6324a91c54` | Existing frame-shift implementation owner, one commit over main |
| #90 | Documentation only; `design/exact-prefix-suffix-frame-gauge-20260924` | This specification; no runtime files changed |
| Upscaler-Plus | `620165a311de9b28a36260219fb5cd370a304e3c` | Provider implementation inspected |
| DoRA Dynamic LoRA Loader | `51c44419bbe3bbdea2d7ea080e32bd61e9f0fa38` | Auto-strength owner inspected |

Relevant preserved Flow checkpoints:

- `checkpoint/pr81-h3-lattice-prefix-green-20260923`, `checkpoint/pr81-00625-physical-both-partial-20260923`, and `checkpoint/pr84-pre-exact-00625-tree-restore-20260923` all resolve to `e57f320...` in this audit.
- `checkpoint/pr81-rigid-gauge-unsquashed-20260923` resolves to `09e259ea0e82b7493e58a3768e122cd272892640`.
- Rigid consolidated candidate: `5dd44dc919cc2aa8401efc8209aa9ad68738c8b7`.
- Historical residual candidate: `67d3618c764b4a464c8d45dc6e21ac0637e275f1`, preserved by `checkpoint/pr84-00629-both-broken-20260923`.
- `checkpoint/pr89-exact-prefix-gauge-green-20260924` preserves current #89; its earlier unsquashed history remains evidence.
- Initial documentation checkpoint: `9462bcfe666a5c198f249f997cbd984ed5ac3168`.

Live #89 CI runs `36055243399` and `36055237117` completed successfully. Its reviews and inline-review lists were empty; the conversation includes a skipped automated review and a hardware-gate note. Green checks do not validate output quality. #81/#84 historical reviews raised retained-prefix memory lifetime and inconsistent flow/confidence/innovation coordinate mapping, respectively. #87 reviews concern diagnostic receipts and conditioning wrappers; they do not establish a spatial repair. #88 had no submitted reviews in the inspected list. Re-fetch current checks and review threads before implementation; never treat comments describing a superseded head as current source authority.

## 3. Verified runtime dataflow

The following is grounded in `e57f320...`, principally `partitioned_scheduler.py`, `handoff.py`, `runtime.py`, `guidance.py`, `partitioned_stage.py`, `partitioned_outer.py`, `geometry.py`, `tone_bridge.py`, and `seam_diagnostics.py`.

1. The outer wrapper separates initial/no-prefix chunks from protected continuation. The protected path requires batch-one Bx24xTxHxW video and a contiguous whole-frame binary mask: zero prefix, one suffix. Unsupported preflight may fall back before sampling. Once a split sampler has started, an exception does not restart under another numerical path.
2. `build_partitioned_stage_plan` copies the prefix from model-internal caller latents and separately saves the original prefix noise. `stage_plan.prefix` is the authoritative internal clean prefix E. The caller's external latents remain independently authoritative for returned protected values.
3. `run_partitioned_progressive` splits descending sigmas at index k: low uses `sigmas[:k+1]`, high uses `sigmas[k:]`. Source video noise is deterministic from the existing derived source seed. Audio noise is carried through. The generic resize of the caller latent is overwritten on prefix frames by `physical_prefix_source = resize_spatial_5d_h3_patch_lattice(E, Hs, Ws)`.
4. The effective production profile uses one `source_carrier_uniform_only` low/probe path. It does not execute the diagnostic duplicate audio/AV shadow paths or retired mixed-grid repairs. The resized mask remains a whole-frame prefix/suffix mask.
5. The low sampler stops at the split. `_raw_sampler_state` applies `process_latent_in` and multiplies its returned representation by `(1-sigma)` to recover the carried state. The exact probe evaluates that stopped state once. Its sampler compensates Comfy's terminal inverse-noise scaling. `_process_latent_in` recovers the model-internal clean `source_x0`.
6. Capture finishes before handoff. The final probe is an actual trajectory anchor; forecasts remain labeled and are not selected as exact anchors. The temporary physical prefix is explicitly written into `clean_video` again: `exact_prefix_source = physical_prefix_source.to(clean_video)`. This same physical tensor, rather than a fresh generic bicubic resize, enters the learned provider.
7. `build_handoff_state` generates target video noise N with its existing CPU generator/derived seed, calls the provider once on clean video, validates shape/dtype/finiteness, and constructs `Y=(1-sigma)L+sigma*N`. L denotes the provider output converted to the noise tensor's device/dtype, the actual operand used by `conditional_renoise_target`. Raw source audio is cloned without numerical change.
8. The historical scheduler regenerates the identical deterministic noise, computes `learned_clean=(Y-sigma*N)/(1-sigma)` in FP32, and derives a one-token DC bridge from it. Thus the function is named diagnostic recovery, but a DC correction calculated from its result does affect production. Recovery can magnify rounding by `1/(1-sigma)`; at the recorded `sigma=0.8780487775802612`, that factor is about 8.2. This is a numerical risk, not proof that recovery caused the jump. The actual observed Y dtype must be logged before attributing error to BF16.
9. `apply_suffix_dc_bridge` adds `mean(E_last)-mean(L_last)` per batch/channel to the first suffix token only. `map_clean_bridge_to_conditional_state` maps that clean delta with `(1-sigma)`, retaining N. Splice diagnostics compare learned native, exact-restored, and DC-corrected boundaries.
10. The scheduler overwrites the target prefix with E, packs target audio unchanged, computes `_noise_argument` relative to the caller's target latent, and merges the caller's original noise wherever the denoise mask protects values. This original noise matters to H3's masked visual-conditioning injection even when returned protected latents are restored exactly.
11. Guidance selects the captured run by session/chunk/conditioning identity. `apply_guidance` obtains a time-matched clean reference and lifts it with generic `resize_video(..., mode=bicubic)`. Temporal correspondence currently lives on the source grid; flow and confidence use bilinear target lifts, while innovations use the configured content lift. These are not the learned provider's output.
12. A fresh high sampler lifetime begins. Spectrum/solver numerical histories reset through existing boundaries; first high must be actual. The model wrapper may alter video x0 with Flow guidance while directly repacking audio x0. The native joint transformer can nevertheless produce different *future generated audio* in response to changed video conditioning.
13. Exact-prefix enforcement and final checks retain caller-owned protected values. Outer `finally` resets guidance and the active run. Any new registration/reference state must have the same cleanup lifetime, including failure paths.

For the supplied RES runs, the recorded high guidance coordinates are approximately 0.375, 0.25, 0.125. All use reference coordinate 0.375, first unclamped then clamped twice, with one temporal-cache construction. Source selection therefore needs only one derived probe reference for this schedule. This optimization requires verification for other samplers: a method that evaluates outside captured support cannot silently reuse it.

## 4. Coordinates and provider contract

### Physical patch-lattice map

For a latent grid HxW, let h=H/2, w=W/2, A=sqrt(h*w). For patch index i on axis a with length n (h or w), H3 uses:

`p_a(i) = o_a + q*i`, where `o_a = 16*(1-n/A)`, `q = 32/A`.

This is the endpoint-excluded, area-normalized patch lattice. A target patch samples a source patch at:

`i_s = (o_t + q_t*i_t - o_s)/q_s`.

Flow separates each 2x2 latent patch into four feature phases (a,b in {0,1}), samples the patch vectors bilinearly with border padding, and reconstructs the phases unchanged. Normalized sampler coordinates use `2*i_s/(n_s-1)-1` with `align_corners=True`; a singleton patch axis uses zero. This formula is not interpolation of an ordinary latent image on a half-pixel grid.

For source 40x52 and target 56x74, patch grids are 20x26 and 28x37. Let `r=sqrt(1036/520)=1.4114913008260777`. Projection of exact target prefix into source carrier samples target latent coordinates:

- `y_t = r*(y_s-a) - 0.2298260165215528 + a`;
- `x_t = r*(x_s-b) + 0.30122617852198275 + b`.

The reverse physical pullback samples source latent coordinates:

- `y_s = (y_t-a)/r + 0.16282496136323812 + a`;
- `x_s = (x_t-b)/r - 0.2134098724842935 + b`.

These hold for each retained phase separately. Bilinear resampling followed by its reverse is not an exact inverse of arbitrary data. The physical displacement scale is r on both patch axes; ordinary half-pixel resizing scales x by 74/52 and y by 56/40. Do not interchange these conversions. Estimate the selected correction directly in target latent-cell units, avoiding either source conversion in production.

### Learned provider

`H3LatentUpscalerProvider` API 1 delegates to `upscale_clean_video_exact` with B/C/T preserved and explicit target H/W. It normalizes input by fixed checkpoint statistics, applies `LatentResizer3D`, reverses normalization and returns in the input dtype/device. Scale embedding defaults to the average of target/source height and width ratios. The backbone uses learned spatial and temporal convolutions and `F.interpolate(..., mode=trilinear, align_corners=False)` at feature resolution. Time length is unchanged by interpolation, but temporal convolutions mix neighboring frames. It receives no audio, masks, H3 patch-lattice metadata, or exact target prefix.

Prefix and suffix share the same provider invocation and output lattice. Same-index learned/exact prefix frames are therefore a legitimate *candidate calibration pair*. They are not identical content transformed by a guaranteed rigid operator: learned synthesis, lost detail, boundary effects, phase-dependent physical projection, and slight aspect changes can contribute. This explains why a registration quality gate is essential. The provider's interpolation convention alone does not determine an output translation; learned convolutions can change it.

### Transform sign

Define content translation in target latent cells by:

`W_d Z(y,x) = Z(y-d_y, x-d_x)`.

Positive dx moves content right; positive dy moves it down. The estimator below directly minimizes mismatch between `W_d L_prefix` and E and returns the **correction** d. If a diagnostic instead reports learned-relative displacement q with `L ~= W_q E`, correction is `d=-q`. Store both the name and convention in telemetry; never invert twice.

One translation applies to every generated suffix frame. No temporal taper, ramp or per-frame motion estimate is permitted: tapering would create a deliberate camera move. Translation is applied on the ordinary learned target latent grid, not separately on the four H3 patch phases. The physical source-prefix projection remains unchanged.

## 5. Evidence and alternative hypotheses

`measure_translation_trajectory` subtracts channel means, Hann-windows the selected ROI, phase-correlates frames, restricts the peak search, and uses local parabolic refinement. Positive values describe later-frame content motion. `anchor_dx[0]` compares prefix frame P-1 with suffix P; `anchor_final_dx` compares P-1 with P+3 when four forward steps are available. Exact restoration changes that anchor and the first suffix's DC. Differences of these nonlinear estimators are not composable registration vectors.

| Run / full-frame stage | First anchor dx,dy | Final anchor dx,dy |
| --- | --- | --- |
| 00625 learned native | -0.140466, -0.010790 | +0.037421, -0.066870 |
| 00625 exact restored | +0.391845, +0.383626 | +0.699761, +0.758296 |
| 00642 learned native | -0.025545, +0.049979 | -0.138054, +0.001157 |
| 00642 exact restored | +0.983804, +0.295731 | +0.205937, +0.789488 |

The task's cited learned-native 00625 numbers were first-anchor values labeled as final-anchor values. The corrected table above preserves the localization conclusion without propagating that mismatch.

In 00642, source-native and physical-exact-context trajectory measurements differ only at floating-point noise scale. Learned-native seam RMS is 0.413672; exact replacement yields 0.550701; DC correction yields 0.517250. Spatial-mean seam ratio returns to about 1.0000003, while low-pass seam ratio remains 1.18615. This rules out a purely spatial-mean discrepancy as a sufficient explanation. It does not distinguish translation from affine distortion or learned content mismatch.

| Attempt | Source / hardware evidence | Classification |
| --- | --- | --- |
| One-token DC removal | #81 discussion records 00623 residual shift unchanged | Falsified as a sufficient spatial repair; tonal utility remains possible |
| Aspect-aware source quantization | 00624 metrics show 38x50 source, complete run, reported shift remains | Falsified as a sufficient repair; do not restore it in this candidate |
| Generic-bicubic learned context decoupling | `f196639...`; 00626 completes low/probe then fails a later CPU/CUDA diagnostic comparison | Superseded/incomplete continuation. Background verdict is confounded by auto-strength; do not call generic bicubic alone proven causal |
| Rigid correction | Historical rigid checkpoint and 00627 metrics | Implementation-specific failure of the applied estimate; not proof that every bounded rigid correction is impossible |
| Full-field additive residual | `67d3618...`; #84 records 00629 arithmetic errors <=4.77e-7 but visible shift remains | Failed media candidate. Algebra is valid; coordinate-equivalence interpretation is false in general |
| Physical target-high guidance lift | `d8303a57352690b8ccd9bb856ad686372555baf6`; #84 records 00630 shift remains | Insufficient as tested. Historical review also identified mixed flow/confidence/content mapping risk; do not copy selectively |
| First-low Sol/Spectrum divergence | #88 provenance / 00642 diagnostics | Separate unresolved numerical observation, deprioritized; no necessary causal link to splice repair demonstrated |
| Current #89 residual bridge | Source at `ae1a779...`, green structural CI, no new accepted media evidence | Unvalidated candidate repeating the residual principle, not the selected architecture |

The old rigid algorithm already compared E and L at the same indices. It used the last four prefix frames, a full-frame phase peak, lower componentwise median and lower MAD, response >=3, MAD <=0.75, and rejected only when most frames clipped. In 00627 its X estimates were `[0.335381,0.657972,0.359221,0.664755]`; lower median was 0.359221 and lower MAD 0.023841, whereas ordinary median/MAD are about 0.508597/0.152767. Its gate accepted a split set of estimates. It applied correction `(-0.359221,-0.496345)` to all 50 suffix frames, then independently shifted *every stored guidance sample* on the source grid using ordinary per-axis size ratios. There was no held-out registration-residual or regional-rigidity acceptance check. Finite-domain source translation followed by bicubic lift is not identical to target-grid translation, and G does not inherit L's displacement by contract.

These are concrete shortcomings and plausible contributing factors, not a reproduced proof of why that hardware run failed. The reported remaining jump survives the new background confound. Repeating the old estimator with slightly tuned constants is not a justified repair.

A simple counterexample rejects residual transport as general spatial registration: let L be an impulse, E its one-cell shift, and S a later two-cell translated impulse. `S+(E-L)` preserves `S'-E=S-L` exactly yet leaves an old negative impulse and a stationary positive impulse; it does not equal the desired translated S. A NumPy float64 audit produced max residual-versus-translation difference 1.0 while the boundary identity error was zero. It is unsafe to reinterpret a content residual as a coordinate change.

## 6. Registration estimator: deterministic and bounded

Create `frame_gauge.py` with pure estimation and application helpers. The following constants are an initial **versioned engineering policy**, not empirically validated H3 thresholds. Hardware promotion is gated; changing constants requires new policy version and receipts. No online threshold adaptation.

### Input and feature contract

- Input pairs: actual clean L prefix versus internal E, identical B/C/T/H/W, finite, same temporal origin. B=1, C=24. Use all of the last min(P,6) prefix frames; require P>=4. For fewer frames, return `insufficient_prefix_support` and keep baseline.
- Work in FP32 for feature construction, CPU float64 for deterministic score accumulation and selection. Detach; no autograd and no RNG. Do not modify process-global deterministic or thread settings.
- Feature image: 3x3 replicate-padded mean filter, then per-frame/per-channel spatial centering and RMS normalization. Ignore channels with RMS <= `max(1e-6, 0.01*median_positive_channel_RMS)` in either pair. Require >=8 valid channels and finite nonzero spatial gradient energy. Record count and gradient RMS. A constant/near-constant pair is ambiguous even if its raw error is small.
- Registration uses a fixed interior support with margin 5 target cells, excluding padded/interpolation border pixels from every candidate. For cost bounds, sample this support at a deterministic stride `ceil(max(H,W)/96)` with a fixed origin; coordinates and displacement still use full target latent-cell units. No resized-image unit ambiguity. Reject if fewer than 16 samples on either axis remain.
- Balance channels and frames equally; do not let high-energy channels or one frame dominate. Use mean Huber loss (delta=1 in normalized feature units) over spatial values, then equal averages. Hold preprocessing fixed across candidates. Also retain normalized RMS and channel-averaged NCC for acceptance, independently of Huber optimization.
- Alternate selected frames into fit and validation sets, with the last prefix frame in validation. Require at least two frames in each. Validation never chooses the displacement.

### Search and confidence

1. Evaluate integer correction candidates dx,dy in [-3,+3] on fit frames. This outer ring diagnoses saturation; accepted corrections satisfy `max(abs(dx),abs(dy))<=2`.
2. Around the winning integer candidate, search a +/-1-cell square at 0.25-cell spacing, then +/-0.25 at 0.0625-cell spacing. Use the same bilinear sampling definition as application. Reject minima outside the accepted bound. Never clamp an over-bound optimum into range.
3. Tie order: loss, squared displacement magnitude, dy, dx. A tie between separated solutions still fails ambiguity checks. Store candidate and runner-up losses; runner-up must be at least 0.5 cell from the optimum.
4. Accept only if validation NCC >=0.75, normalized RMS decreases at least 15% relative to zero displacement, and the separated-runner-up loss margin is >=5% of the zero-displacement loss. Denominators use `max(value,1e-8)`. Zero improvement and repetitive texture fail closed.
5. Estimate each held-out frame and informative spatial region independently as a *consistency check*, not a second correction. Use full interior, upper/lower halves and left/right halves; record all results. At least one informative region in both each vertical and each horizontal split must support the global displacement within 0.25 cell per axis. Any informative region that strongly prefers a conflicting displacement (>0.5 cell with the same margin test) vetoes the rigid model. Sparse/untextured regions cannot count as agreement.
6. All informative held-out frames must agree within 0.25 cell per axis. Compare ordinary midpoint median, maximum deviation and full range; lower-MAD alone is forbidden. Evaluate scores separately on the four target lattice parity subsets; a confident phase-dependent conflict vetoes a rigid latent-image translation.
7. Re-evaluate `W_d L` against E on the last held-out prefix frame. This residual check is mandatory. Exclude border pixels and do not use seam RMS or adjacent-frame motion as the fit objective.
8. An estimate with `max(abs(d))<=0.0625` is identity. Return exact no-op with `already_aligned` only if zero-displacement validation NCC passes and the solution is unique; otherwise `ambiguous`. Do not resample at zero. Exact tensor equality is an early identity path.

Return a structured result: accepted/identity/rejected, reason, signed correction, units, frame indices, support/stride, bounds, policy version, per-frame/per-region statistics, fit/validation/zero losses, NCC, peak separation, parity checks and maximum disagreement. Do not condense confidence to an unexplained scalar. These checks can reject the hypothesis; they cannot prove visual equivalence of a nonlinear VAE decoder.

### Application and borders

For accepted nonzero d, process only generated frames with one FP32 bilinear pullback, `align_corners=False`, explicit grid `g_x=2*(x-dx+0.5)/W-1`, similarly y, border replication. Cast once back to the input dtype. All channels use the same map. No wrap, reflection, crop, resizing, sharpness compensation or temporal interpolation. Use bounded temporal batches (e.g. <=4 frames); do not allocate an unbounded all-suffix FP32 clone.

Border replication is an explicit approximation where translated content is unavailable. Keep a validity map of coordinates with complete in-bounds interpolation support; never treat replicated pixels as registration evidence or confident temporal correspondence. Report invalid area and reject if >8% of the image. There is no guaranteed seam-free extrapolation beyond the observed field. Edge distortion remains a hardware acceptance question; do not conceal it with a spatial taper that would violate rigid translation.

Use the same d on disposable prefix copies for calibration/DC diagnostics, but never copy those shifted learned frames into protected state. The actual output prefix is E, copied from the owner, with exact external restoration retained.

## 7. Clean/noise contract and transaction ordering

The selected production placement is **before deterministic re-noise**, using actual provider output. Add an optional internal postprocess hook to `build_handoff_state`, used only by protected learned continuation:

`clean_video_postprocess: Callable[[Tensor], CleanVideoPostprocessResult] | None`

Result contains corrected clean video and bounded diagnostic/decision metadata, not arbitrary persistent model state. The hook input is exactly the clean operand `learned_x0.to(noise)`; the provider object and API version remain unchanged. Run after provider validation, before `conditional_renoise_target`. Validate returned geometry/device/dtype/finiteness and prefix ownership. No callback may call a sampler, model, provider or RNG. Nonpartitioned callers and hook=None retain the historical path.

Prepare registration and the optional guidance reference within this hook using already captured data. Do not publish binding state until every gate succeeds. Low-confidence estimation returns unchanged input and a reason, not an exception or a fallback sampler. Malformed/non-finite tensors and ownership violations are hard errors, following the existing post-start error policy.

For accepted d, form `L*` with suffix `W_d L_suffix`. For the retained one-token DC policy, calculate the channel-mean offset from **aligned disposable learned prefix** to E, add it only to the first generated token, and receipt it separately. This is the existing tonal mechanism evaluated after spatial registration; it is not the spatial estimator. Do not retain #89's full-field residual or add it on rejection. Keep one-token weight 1, unchanged in both controls; active ordering is spatial alignment, tonal bridge, re-noise, exact restore.

When repair is OFF, rejected, or identity, execute the original 00625 DC/recovery path unchanged to preserve baseline arithmetic. When accepted, mark DC as already applied and skip the scheduler's second DC application. Diagnostic code uses retained bounded slices of the actual L and L*; it must not invert Y and feed that approximation back into the spatial repair. Retain original telemetry names for comparison, and add an explicit clean-source field so actual-provider and inverse-recovered measurements cannot be silently mixed.

The active re-noised video is:

`Y* = (1-sigma)*L* + sigma*N`.

An alternative implementation may retain Y and use:

`Y* = Y + (1-sigma)*(L*-L)`.

These are exactly equivalent over real numbers, not necessarily bit-identical in BF16/FP16. Tests must use dtype-specific tolerances and observe actual production dtype. Prefer the pre-noise hook to avoid inverse recovery. Neither equation requires W(N). Warping the whole conditional state would instead produce `(1-sigma)W(L)+sigma*W(N)`, interpolating/correlating the noise and changing its realization. Do not do that.

The hook consumes no extra random state. N remains owned by `build_handoff_state` and can be released after mixing. Existing deterministic recovery replay on the unchanged baseline is not a new candidate RNG draw; do not introduce additional draws for registration. Seed, offsets and RNG state before/after must match repair OFF.

After the hook returns, restore E into actual target prefix, preserve audio, reconstruct high sampler noise using the existing `_noise_argument`, then `_merge_preserved_noise`. The numerical high-noise argument changes on generated video as needed to encode Y*, but protected caller noise and the random realization N do not change. A receipt must distinguish these three tensors.

## 8. Guidance: one derived reference with independent calibration

Let R be the selected exact low/probe reference. Existing guidance uses `G=B(R)`, where B is its bicubic content lift. Calibration of L against E yields dL; it says nothing exact about G. Determine dG independently by registering G's same-index prefix against E with the same estimator. If guidance is off, no G calibration is needed. If dG is identity, leave its suffix unshifted. If either required estimate rejects, reject the entire repair transaction; do not enable half a repair or silently disable guidance.

Create one ephemeral `RegisteredGuidanceReference` containing target-grid `G*`, prefix length, validity, source run/probe identity, reference coordinate and registration metadata. `G*` contains authoritative E on prefix and `W_dG G` on suffix. It is a derived reference, not a captured or sampled trajectory. Do not append it to `H3FlowTrajectory`, rewrite `TrajectorySample`, or shift every sample. Shared trajectory bytes and run identifiers remain unchanged.

For the audited descending RES schedule, `time_matched_reference_info` clamps all high calls to the actual probe endpoint. Validate this support and phase before activation. Pass the optional reference context through `FlowBinding` to `apply_guidance`; retain original reference-coordinate/clamping metadata. If a sampler can evaluate above that endpoint or change reference identity during high, v1 does not activate. Detect from the sampler contract before committing; a later unexpected mismatch raises rather than switching gauges mid-high. Other samplers can be enabled only with a validated coordinate-support contract.

Supported initial modes are off, direction, direction+acceleration and direction+temporal with verified reference support. For downsample_consistency, v1 rejects activation with `unsupported_guidance_operator` and runs the unchanged baseline. That operator compares on the original source grid; a correct conjugated restriction operator would be a separate design. This restriction preserves existing mode behavior.

### Spatial reference and temporal fields

- Direction: use G* directly at target resolution; apply low-frequency projection and existing weights/schedule/clamp unchanged. Protected prefix correction is explicitly zero.
- Acceleration: reset `GuidanceState` before the first high call, as the wrapper already does. Build high/reference velocities from current high state and G*. Do not translate old low/probe velocities or histories. Same-coordinate PECE semantics remain unchanged.
- Temporal: build correspondence **once on G*** at target resolution using the existing correspondence algorithm. Its flow, confidence, innovations and high-grid warp then inhabit the same grid; target-to-target resize helpers are identity. Scale search radius from the existing source radius by `ceil(radius*max(Wt/Ws,Ht/Hs))`; cap at 8, and reject activation if the required radius exceeds that cap. For the observed 40x52->56x74 and radius 4, target radius is 6. This modifies computation resolution, not a model evaluation or sampling trajectory.
- Build temporal correspondence only for generated-to-generated adjacent pairs. Set both directions of prefix-internal and exact-prefix/suffix-crossing pairs to zero confidence. The exact prefix is conditioning authority, and paired calibration does not certify temporal matching across a learned/exact representation substitution. The first suffix may still use the next generated frame. Report `cross_prefix_temporal_pairs_disabled=true`. Do not silently re-enable the pair or warp E to satisfy it. An independently tested boundary-pair matcher may be a later extension.
- Zero temporal confidence wherever either query or sampled support falls outside the translation validity region. Keep forward/backward cycle and uniqueness checks. If fewer than two generated frames exist, temporal correction is zero, with ordinary direction guidance retained.
- Calculate residuals/bounds on generated video only, then copy protected predictions unchanged. Never allow a large unused prefix residual to alter the suffix correction's RMS clamp. Audio remains excluded from Flow guidance.

The changed temporal-reference resolution has real runtime and empirical risk. It is selected to avoid the older partial-remapping defect where content, flow and confidence used incompatible coordinates. Reusing low-grid correspondence is permissible only with a complete tested transform of vector fields, confidence, innovations and boundary ownership; copying the old source-sample shift is not such a proof.

A target-grid temporal cache is scoped by run id, session, chunk, target shape, probe phase/coordinate, dG, prefix length, validity policy and estimator version. Build/reset it exactly once per accepted handoff and clear it on every exit. Do not store tensor caches on an immutable shared provider. A failed activation must leave the existing guidance path bit-identical.

## 9. State ownership and interoperability

| State | Action |
| --- | --- |
| Caller latent, protected prefix, original protected noise | Never transform/mutate; retain existing exact restoration and checks |
| `physical_prefix_source` / `exact_prefix_source` | Preserve physical-both semantics; release after transfer when no longer referenced |
| Actual learned output L | Read for registration; clone/produce new suffix in bounded batches; disposable shifted prefix for measurements only |
| Handoff N | Retain unchanged; no new stochastic draw |
| Target generated video raw state / high noise argument | Rebuild consistently from corrected clean state through existing affine initialization |
| Raw carried audio, caller audio mask/noise | Directly unchanged at handoff; same packing/bit values |
| Original guidance run/sample store | Read-only, unchanged |
| Derived G* / temporal fields / acceleration state | New per-handoff reference; reset or recompute in one coherent target gauge |
| Denoise/protection masks, reference images, keyframe conditioning | No shift or resize added by repair |
| Sampler/PECE history, Spectrum forecast history | Existing low/probe/high boundary reset; no history transport across gauges |
| VDN memory/buffers, Sol BSA pool/provider identity | Existing lifetime/release contracts; no new attention or kernel policy |
| Diagnostic contexts | Bounded, read-only, removed on success/failure; no hidden persistent transform |

H3 AV generation is joint. Zero direct audio mutation at the handoff is provable; bit-identical final *generated* audio across repair OFF/ON is not guaranteed because future audio attends to changed video. Protected audio remains exact under its existing mask contract. Acceptance must compare audio quality/synchronization and record final differences without claiming inevitable byte equality. Do not freeze generated audio or add a separate audio pass to manufacture equality.

Exact-prefix means exact caller-owned values, not merely equal after float conversion. Preserve the original target latent for return; verify bytes/dtype/shape and all protected mask values, including the existing H3 noise injection path. No decoded warp or camera correction is involved.

## 10. Auto-strength OFF provenance

The inspected owner is `DoraPowerLoraLoader.load_loras` in DoRA `nodes.py`. `_state_payload_get_loader_global` gives the selected State Manager stack's `loader_globals` precedence over node kwargs; legacy globals can also apply. A visible unchecked toggle alone does not prove OFF. `IS_CHANGED` uses the loader input cache key; record both effective input identity and whether a cached output is being reused.

When enabled and manual strength is nonzero, `_auto_strength_analyze_base_targets` estimates target/group norms after mapping and compatibility handling, computes cohort mean/norm, clamps to configured ratio floor/ceiling, and sets target strength to manual strength times that ratio. `_apply_base_strength_ratios` modifies LoRA tensors; `model.add_patches` still receives the row's manual strength. Norm estimation can use the effective current patched model, so adapter order matters. Do not model this as one global denoise-strength scalar.

Current `auto_strength_report_json` exposes resolved enabled/device/bounds and row model/CLIP strengths/status. Preserve this artifact and extend or accompany it with execution provenance, without redesigning the algorithm:

- loader node/slot, report schema, owner commit/effective source hash and patched model identity;
- configured and resolved enabled flags; resolution source (`node`, `state_manager_slot`, `legacy_state`, `default`), state revision;
- each active adapter identity/order, manual model/CLIP strengths and actually applied patch counts;
- effective per-target ratios/strengths, or all-ones sentinel when OFF, plus a deterministic digest and min/max; do not report a misleading aggregate scalar;
- value origin (`manual`, `auto`, `skipped_zero`, `unanalyzable_fallback`), separately from execution reuse (`fresh`, `cached`);
- ratio bounds, analysis device/fallback, logical grouping/broadcast scales, norms/cohort means when calculation executes; when OFF, mark calculation inputs unused, not fabricated;
- report/output cache identity linked to the model entering Flow, carried through cloning.

Prefer capturing the existing report and executed prompt/state snapshot. If resolved origin/model linkage cannot be recovered, implement a minimal immutable metadata receipt in the loader on its own correctly scoped PR, then have Flow log that receipt. Missing metadata is `unknown`, never implicitly OFF. Both repair controls must use the same provenance instrumentation. Do not inspect arbitrary model attributes or rerun strength analysis inside Flow. This is telemetry required to validate the control, not a request to repair auto-strength.

A valid hardware comparison requires resolved OFF for every applicable loader and no inherited auto-scaled adapter stack. Disabling a downstream loader does not undo earlier patched tensors. Reuse is valid only when the cached model/report identity proves the same OFF inputs. The inspected 00624/625/626/627 logs contain no auto-strength receipt; 00642 also does not establish resolved OFF. Preserve the user's background observation with that explicit provenance limit.

## 11. Proposed implementation surfaces

| File / surface | Required change |
| --- | --- |
| `h3_flow_regenerate/frame_gauge.py` (new) | Versioned config/result, paired estimator, validation, target-cell translation/validity helpers; no Comfy imports |
| `handoff.py::build_handoff_state` | Optional internal clean postprocess hook; retain provider/API/noise/audio behavior; no default-output change |
| `partitioned_scheduler.py::run_partitioned_progressive` | Prepare E and selected probe reference, execute all-or-nothing registration, route active versus historical DC path, retain bounded actual-clean diagnostics, publish context only after success |
| `runtime.py::FlowBinding` / model wrapper | Optional per-invocation reference context passed to guidance; no shared trajectory mutation |
| `guidance.py::apply_guidance` / temporal helpers | Optional target-grid registered reference, generated-only masking/bounds, cache identity and target-grid correspondence contract |
| `partitioned_outer.py` | Unconditional context/cache cleanup on every exit; verify no nested reuse |
| `partitioned_node.py` | Explicit advanced `frame_gauge_repair` option: default OFF until hardware approval; no hidden env toggle. OFF/ON available in the same PR overlay |
| `tone_bridge.py` | Reuse one-token DC formula on aligned disposable prefix; do not promote full-field residual; avoid duplicate application |
| `seam_diagnostics.py` | Keep existing diagnostics semantically stable; add paired-prefix fit, actual-clean provenance, separated stage labels |
| `tests/test_frame_gauge.py` and existing handoff/guidance/partitioned tests | Tests in section 13; no full-model requirement for CPU contracts |
| DoRA telemetry, only if current report is insufficient | Minimal receipt on a separately owned PR; no auto-strength arithmetic changes |
| Documentation / runtime evidence checker | Activation/fallback controls and receipt verification. No README claims of speed or quality until hardware results support them |

Do not append the new algorithm to the old rigid helper while retaining its acceptance gates. Do not silently enable both residual and registration. Remove the residual from the final #89 implementation diff after preserving its evidence checkpoint. This investigation changes only this design document.

## 12. Telemetry, performance and rollback

Emit one bounded `partitioned_frame_gauge` event per handoff: policy/version, mode, eligibility, accepted/identity/rejected, reason, dL/dG and signs/units, frames/ROIs, confidence components, search bounds/support, provider identity, actual clean dtype/device, exact-prefix checksum, mask classification, noise seed/offset identity, DC policy/order, transform domain, transformed-state list, guidance mode/support/cache policy, target radius, valid-area fraction, timings and peak workspace accounting. Link the auto-strength execution receipt rather than duplicating a large per-target report.

Stage names must distinguish `learned_native`, `paired_prefix_aligned_witness`, `suffix_aligned_before_dc`, `exact_restored_pre_high`, `final_post_high`. A shifted disposable witness is not the restored prefix. Report learned/exact paired residual before/after separately from cross-time trajectory values. Keep counters for extra H3 NFE/sampler/provider/VAE calls at zero and assert them against baseline.

Cost is bounded classical tensor work. For B=1,C=24,T=62,H=56,W=74, a complete target FP32 video is 24,665,088 bytes (~23.52 MiB). Six prefix frames in FP32 are ~2.28 MiB per pair member; float64 score workspace doubles the corresponding bytes. Four-frame FP32 application batches are ~1.52 MiB each. Retain one target reference if guidance is active, not every trajectory sample; release learned/calibration buffers before high. Temporal flow/confidence storage is O(T*H*W) with small channel counts.

Target temporal matching is more expensive than source matching. For radius 6 versus 4 and these areas, the raw search-work ratio is about `(169/81)*(4144/2080)=4.16` for that classical component; it is cached once. This is not a prediction of total run slowdown. Estimate search also performs fixed candidate sweeps on <=96 samples/axis and <=6 frames. Measure CPU transfer/synchronization time, registration time, temporal-cache build, peak allocated memory and total high-stage wall time. Initial promotion budget: added non-model time <=1% of paired total sampler wall time and no OOM/headroom regression; if exceeded, optimize the same tested operator or leave OFF, rather than silently weakening gates.

Rollback is a node toggle OFF in the same #89 overlay. Rejection preserves the physical-prefix 00625 path with its one-token DC, same sampler lifetimes and same noise construction. A low-confidence no-op is a safe outcome, not a repaired run. Invalid tensors/owner mismatches raise before high, clear transient state and do not restart low/probe. No global cache, transform reuse across chunks, or retry with a new seed.

## 13. CPU and structural tests required before hardware

This design task did not run Flow's Torch tests: Torch is absent in both available Python environments. It executed only NumPy arithmetic checks and source/metric audits. Do not report a tested production estimator. The implementation must run targeted tests with repository dependencies.

1. Physical coordinate maps: verify both formulas against `_h3_patch_resample_grid`, linear patch-phase ramps, same-size identity, singleton patch axes, 40x52->56x74, aspect changes, both dimensions even, odd dimensions rejected. Verify physical displacement conversion differs from half-pixel per-axis conversion.
2. Registration: independent analytical Gaussian/textured fields at known +/-x/y shifts, integer and fractional values, combined shifts, zero, exact equality, DC/gain perturbations, changed channel scales. Avoid generating all fixtures solely with the operator under test. Expected accepted error <=0.125 target cell for well-conditioned synthetic cases; identity is bitwise no-op.
3. Reject constant fields, stripes/repeated patterns with multiple peaks, non-finite values, over-bound shifts, too-short prefixes, missing valid channels, per-frame translations, local warp/scale mismatch and phase-dependent distortions. Include 00627's split-list confidence counterexample. A rejected candidate must produce exactly the old output and reason.
4. Hold-out tests: fitting frames can match while the last held-out frame/ROI disagrees; require rejection. Check isolated bad ROI cannot be averaged away. Tie ordering/repeated calls deterministic without RNG use.
5. Application: translation sign using a marker, border replication/no wrap, exact zero path, prefix bytes/dtype unchanged, no input alias mutation, bounded batches equivalent within dtype tolerance, finite output and validity accounting. Test FP32/FP16/BF16 where supported.
6. Noise: `Y+(1-sigma)*(L*-L)` equals `(1-sigma)*L*+sigma*N` in FP64 tolerance and production dtype error budget. Demonstrate W(Y) changes N. Check sigma near zero and near one; no estimator input may come from inverse re-noise. Spy RNG state/call counts and provider calls.
7. DC routing: active bridge applies exactly once using aligned witness; OFF/rejected/identity reproduce existing DC bytes and one-token scope; no full-field residual path activates. Tonal metrics cannot change registration acceptance.
8. Ownership: caller/internal exact prefix and original protected noise equal before/after; raw audio and masks identical. Fractional/spatial masks stay on existing unsupported/fallback behavior. No transform to image refs, keyframes or audio. Final generated audio byte equality is not a unit-test promise for a real joint model.
9. Guidance: construct L and G with deliberately different known shifts, prove independent dL/dG and all-or-nothing activation; shared run/sample tensors remain unchanged. Exercise direction, acceleration and temporal context; protected prediction/clamp support excluded. Target flow/confidence/innovation agree on one grid, invalid support has zero confidence, prefix-crossing pairs have zero confidence. One-suffix case is handled without out-of-range indexing.
10. Reference lifetime: exact probe selected over same-coordinate predictor; recorded RES high queries clamp as expected; unsupported/out-of-support sampler path rejected before activation. Cache built once, repeat predictor/corrector coordinate handling unchanged. Test two sequential chunks, interrupted high sampler, nested invocation rejection, and model clone isolation; no stale transform survives.
11. Call topology: spy executor/provider/model calls, low/probe/high slices and shapes, first-high actual requirement, zero shadow invocations. Assert old mixed-grid/deprecated repair keys inactive and physical prefix used for both low/probe and learned context.
12. Baseline identity: candidate toggle OFF on a fake deterministic provider and executor must match e57f320 behavior exactly, including DC recovery arithmetic. Rejection/identity match OFF. Ordinary no-prefix and nonpartitioned paths do not see the hook.
13. Provenance: State Manager precedence, cached report linkage, OFF effective ratios, missing receipt as unknown, and unknown not accepted by the hardware evidence checker. Test telemetry shape bounds; no unbounded sample/target dump in per-step metrics.

Targeted entry points after implementation: `tests/test_frame_gauge.py`, `tests/test_handoff.py`, `tests/test_guidance.py`, `tests/test_geometry.py`, relevant `test_partitioned_*` ownership/topology tests, plus existing source-contract checks. Run required repository CI once on the consolidated head; avoid unrelated broad benchmark campaigns.

## 14. Minimal hardware validation

### First authoritative pair

Use the saved 00625 workflow/control inputs as the template, auto-strength resolved OFF, fixed seed/reference/prompt/model/VAE/VDN/Sol/Spectrum/DiffAid configuration and schedule. Run both arms on the same new #89 overlay, toggling only `frame_gauge_repair`; the OFF branch must first have passed structural identity against the exact 00625 path. Do not compare a newly generated candidate against old 00625 as if auto-strength were controlled there.

Capture actual executed seed/inputs rather than assuming a historical prompt filename establishes identity. The saved exact workflow/prefix tensors were not available in this investigation. Before the pair, resolve them from the user's saved production workflow and record hashes. No request to recreate historical first-low Sol behavior is needed. Keep diagnostic overlays which intentionally perturb model execution out of the pair; record the full Patcher overlay list and effective source hashes.

Prefer saved identical raw prefix and low/probe handoff inputs for both arms if the workflow supports them without altering sampler topology. Otherwise require equal input/probe hashes across the paired full runs. If they differ, the comparison is not controlled; do not infer a registration effect from it.

For the audited two-chunk configuration, expected totals are 18 logical opportunities, 14 actual transformer NFE, 4 Spectrum forecasts, 6 sampler invocations and 4 history boundaries; continuation alone has three sampler lifetimes/two boundaries. These totals are case-specific, not hardcoded production targets. Auto-strength OFF or changed video may alter adaptive Spectrum decisions; if actual counts differ, report the failed identical-NFE control and resolve it before timing/quality attribution. Never force forecasts merely to satisfy a number.

### Acceptance questions, separately recorded

1. Is the background coherent in the OFF control under resolved auto-strength OFF? If not, the claimed background control is not reproduced; do not credit or blame the spatial candidate for it.
2. Is the visible exact-prefix/suffix camera/frame jump eliminated, with native suffix motion retained? Inspect full frame and the previously affected background region at normal speed and frame-by-frame. A lower seam RMS is insufficient.
3. Does target-high retain that alignment through the final suffix without a later pan, ghosting, border smearing, zoom or lost detail?
4. Are exact protected prefix and protected caller noise still byte-exact? Check before high and returned external latents.
5. Are call/sampler/NFE topology and stack identities matched? No hidden shadow/provider/VAE work.
6. Are carried/protected audio values exact and generated audio quality/timing preserved? Record whether final generated audio differs; do not imply direct audio mutation from that alone.
7. Did both required registrations have plausible bounded displacement, held-out improvement and regional/phase agreement? A no-op/rejection does not pass repair acceptance.

If ON rejects, preserve a bounded clean tensor bundle containing E, actual learned prefix, first four learned suffix frames, probe-guidance prefix and first four guidance suffix frames, shapes/dtype/sigma, decision metadata, and relevant source identities. This is the smallest evidence that can distinguish rigid translation from affine/content mismatch. Do not run a sequence of blind threshold changes. If translation fails held-out tests, do not auto-upgrade to affine/deformable warping or additive residual; record the failed model and design a separately evidenced next step.

Only after the matched pair passes, run one more same-settings chunk transition to check stale-state cleanup and one other geometry/aspect-ratio case. Use both displacement signs in CPU fixtures; hardware need not be forced to produce artificial sign cases. Promote default activation only after visible quality, audio, memory and overhead checks pass.

## 15. Non-git evidence inventory

Paths below identify preserved user artifacts, not files that must be committed into source. Keep logs/metrics and accepted/rejected boundary media; implementation should inspect the named artifacts and reproduce paired controls. Temporary checkouts, registry responses and helper scripts have no independent evidentiary value and may be discarded.

| Artifact | Claim supported | Retention / implementation use |
| --- | --- | --- |
| `/ComfyUI-Sol-H3/metrics_00624_.json` | 38x50 geometry, complete call topology, trajectories | Preserve; inspect comparison only |
| `/ComfyUI-Sol-H3/Pasted text(20260923-025006).txt` | 00624 full runtime context | Preserve; inspect stack/control claims |
| `/ComfyUI-Sol-H3/metrics_00625_.json` | Physical-both source/transfer localization and corrected first/final anchor values | Preserve; inspect, not an auto-strength-OFF control |
| `/ComfyUI-Sol-H3/Pasted text(20260923-041953).txt` | 00625 log, executed stack evidence, no resolved auto-strength receipt | Preserve; inspect |
| `/ComfyUI-Sol-H3/metrics_00626_.json` | Incomplete continuation: five sampler invocations, source-native measurements | Preserve; do not treat as completed splice test |
| `/ComfyUI-Sol-H3/Pasted text(20260923-045630).txt` | CPU/CUDA diagnostic failure after low/probe | Preserve; inspect scope of rejection |
| `/ComfyUI-Sol-H3/metrics_00627_.json` | Actual old rigid estimates, gates and applied correction | Preserve; inspect to avoid repeating rejected implementation |
| `/ComfyUI-Sol-H3/Pasted text(20260923-073334).txt` | Associated runtime context | Preserve; inspect |
| `/ComfyUI-Sol-H3/metrics_00642_.json` | Latest native/restored/DC measurements, guidance reference clamp and topology | Preserve; inspect |
| `/ComfyUI-Sol-H3/Pasted text(20260924-200514).txt` | Latest full diagnostic log and effective stack | Preserve; inspect; observer diagnostics are not the new controlled pair |
| `/ComfyUI-Sol-H3/Pasted text(20260924-210508).txt` | User task and auto-strength-off background observation | Preserve as provenance; observation has no machine-readable OFF receipt |
| Saved 00625 workflow, raw prefix/clean handoff tensors and boundary media | Needed for exact controlled inputs and visual acceptance | Not retrieved here; exact paths/hashes unresolved. Locate before hardware; do not invent |
| Future paired OFF/ON reports, latents and boundary media | Registration decision and actual repair outcome | Must preserve with exact overlay/head/input identities |

Primary project paper attachments were screened by title/abstract: DMD2 (`2405.14867v2`), Sol-Attn (`2607.24027v1`), Sol Engine (`2606.23743v2`), VSA (`2505.13389v5`), Spectrum (`2603.01623v1`), and VDN (`2609.20744v1`). They address distillation/attention/forecasting rather than this provider's exact-prefix coordinate contract. No claim of latent translation equivalence is derived from them; current source and actual runtime artifacts are the relevant authority here.

## 16. Verified conclusions, unresolved questions and implementation handoff

### Verified facts

Physical-both 00625 semantics, exact-prefix substitution, existing one-token DC scope, provider half-pixel feature interpolation, conditional-noise affine formula, independent bicubic Flow guidance lift, old rigid/residual implementation details, listed runtime metrics, and the DoRA auto-strength owner/dataflow were inspected directly. Historical visible verdicts come from user observations recorded in the task/PR discussions, not a new decode performed by this investigation.

### Architectural conclusions

The exact-prefix replacement is a well-supported localization of the spatial discontinuity. A conditional rigid registration candidate is justified, but a universal rigid-gauge root cause is unproven. Same-time prefix pairs are preferable to adjacent-frame motion. The correct correction domain is actual clean target video with unchanged noise. Guidance must be calibrated independently or the transaction must remain inactive. Full-field additive residual invariants do not establish spatial alignment. Existing solver/attention histories should reset, not be warped.

### Unresolved hardware-dependent questions

- Does actual L-to-E mismatch admit one rigid translation across frames, regions and patch phases with auto-strength OFF?
- Does the calibrated prefix displacement transfer to suffix content, and does the nonlinear VAE decode retain detail after sub-cell interpolation?
- Do independent G/E registration and target-grid temporal guidance prevent high-stage drift without degrading dynamics?
- Are the proposed confidence thresholds useful, or do they correctly reject the current case? No real paired prefix tensors were available for fitting them.
- Are border replication, generated audio changes and added classical-compute cost acceptable?
- What exact resolved auto-strength behavior produced 00625? Unresolved and not required to reproduce a new OFF/OFF pair.

### Assumptions to re-check before coding

Re-fetch main, #87/#88/#89/#90 heads/bases, review threads, CI, tags and instructions. Verify physical-both tree identity; the live provider normalization/output dtype; actual latent/noise domains and H3 mask behavior; selected sampler reference support; effective DoRA state/cache provenance; caller-owned prefix restoration; and Spectrum/Sol/VDN lifetime contracts. Initial implementation must not claim support for an unverified sampler or guidance operator.

### Branch and PR plan

Keep this design on documentation PR #90. Leave #87 and #88 unchanged. For runtime work, checkpoint the then-live #89 head and create a separate implementation mirror. Start from the physical-both baseline, replacing #89's residual candidate within its existing scope; do not create a replacement production PR or stack #87/#88 overlays. Add the design to the final runtime tree or pin its exact documentation commit in the PR. Preserve unsquashed checkpoints before hostile review, risky changes and long testing, and after substantial progress. Consolidate the validated final runtime diff onto `candidate/frame-shift-exact-prefix-gauge-20260924` as one clean commit over its live production base, preserving existing authorship requirements. Re-check branch ancestry/CI and update #89's description to reflect the actual algorithm and unresolved hardware gate. No merge until media acceptance.

The next implementation chat should treat this committed document as its specification, verify live assumptions before edits, implement the bounded transaction and tests, and deliver all ready changes through the correct Patcher PR overlay. Deviations require a documented source/runtime reason. Keep auto-strength OFF, caller-owned exact prefix, unchanged handoff noise/direct audio, zero extra H3 NFE/sampler lifetimes, compatibility contracts and GitHub checkpoint discipline.
