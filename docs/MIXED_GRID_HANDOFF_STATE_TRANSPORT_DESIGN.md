# Mixed-Grid handoff state transport design

Status: implementation specification for a controlled, Flow-local state-transport candidate. No production implementation or new CUDA/media acceptance is included. Audited 2026-09-11. The existing [Mixed-Grid attention-measure specification](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/3afc6c63063f410afec756c42811a8cf1ad2ed77/docs/MIXED_GRID_MEASURE_ARCHITECTURE.md) remains authoritative for attention ownership.

## 1. Executive conclusion

Implement **clean-latent transfer plus transport of the effective source flow displacement**, restricted initially to the actual Mixed-Grid continuation path. At the existing exact handoff probe, retain both the current internal sampler state `X` and its accepted clean prediction `C`. Apply the learned upscaler and existing representation/DC reconciliation only to clean video. Reconstruct each generated target suffix frame as:

```text
R_source = X_source - C_source
X_target = C_target_reconciled + U(R_source)
```

`U` is the existing framewise bicubic spatial lift, with explicit fixed interpolation semantics. This preserves `sigma * effective_velocity` under that lift, without another H3 call or division by sigma. Keep original caller noise for protected target rows and keep the carried audio state unchanged. Retain existing sampler/Spectrum lifetime boundaries.

This is a recommendation based on an explicit local flow invariant, **not a proof that the seam is solved**. Source confirms discarded video state; 00383 localizes the measured worsening to the first high-grid prediction. Neither proves that a bicubic velocity lift matches the target model's learned vector field. Noise-endpoint transport is a coherent alternative with a different invariant (§5); it is a controlled comparison, not an interchangeable spelling of the selected formula. Production promotion requires §9.

No attention, VDN, Spectrum, solver-history or decode redesign is required by the selected formula. In particular, do not transport multistep denoised history across the existing high-stage restart.

## 2. Verified facts

### Source and live state

Flow baseline is PR #30, `8dfb7a2e570458b5078734755b897c7e89ddac8c`, one commit over `970396db839ae7ab431b9718859f6d48a2e5019b`. Production files audited: `runtime.py`, `handoff.py`, `guidance.py`, `mixed_grid.py`, `comfy_compat.py`, `geometry.py`, `representation_bridge.py`, `tone_bridge.py`, `seam_diagnostics.py`, and handoff/mixed-grid/guidance tests. Paths in this document are relative to `h3_flow_regenerate/` unless qualified.

The diagnostic source is PR #32 at **`f0f0b1e5a54de6ff31a31a795a298ab9b1961dc7`**. The supplied snapshot's full SHA had an incorrect suffix. #32 is stacked on #30 and changes three files; it is not the clean baseline. Its raw/guided events are inside Flow's prediction wrapper before/after Flow guidance. “Raw” here means the incoming prediction at that wrapper boundary, not an unpatched, unconditioned transformer tensor.

Audited core source: `Comfy-Org/ComfyUI@8406da920f9df19cbc14b76aef1bcd3dc4cf1119`, particularly `comfy/{samplers,model_sampling,model_base,model_patcher,patcher_extension,latent_formats}.py`, `comfy/k_diffusion/sampling.py`, and `comfy/ldm/minimax/model.py`. Fetched current upstream `master` versions of the first five sampler/model files were byte-identical to this pinned companion revision. The production log reports `v0.35.0-15-g7803b74e`, branch `patcher/stack`; that local assembled revision is not available from upstream (404). Thus exact installed-stack byte equivalence is **not verified**. Re-check installed functions before production validation; do not substitute a version label for source evidence.

### Primary run 00383

Full files, identities and hashes are in §13. The metrics and matching full log were read locally, not reconstructed from the task prompt. Their paired upload times, six low/probe/high lifetimes, geometry, counters and weighted-route summaries are consistent. No workflow export, saved handoff tensors or decoded media for this run was retrieved in this investigation.

All entries below are video clean-prediction boundary measurements except the explicitly labeled transfer construction. RMS measures the difference across adjacent prefix/suffix latent frames; low-pass uses the diagnostic's spatial 5×5 replicated-edge average. These metrics measure a discontinuity, not a perceptual error against a ground-truth frame.

| Measurement | RMS | Low-pass RMS | Spatial-mean RMS |
| --- | ---: | ---: | ---: |
| Learned native clean boundary | 0.426179528 | 0.225515291 | 0.101971835 |
| Exact-prefix splice before repair | 0.558188617 | 0.367387146 | 0.243088365 |
| Repaired transfer clean boundary | 0.426179528 | 0.225515291 | 0.101971827 |
| First high raw prediction, actual | 0.533471823 | 0.278102726 | 0.174343079 |
| Same prediction after Flow guidance | 0.523939729 | 0.259582579 | 0.153376132 |
| First callback denoised | 0.523699582 | 0.259305000 | 0.153154626 |
| Next callback, forecast | 0.523714721 | 0.259287030 | 0.156907856 |
| Final callback, actual | 0.526700497 | 0.264019251 | 0.162515014 |

First raw versus repaired transfer: **+25.175375%, +23.318789%, +70.971810%**. Handoff sigma is `0.8780487775802612`, unshifted coordinate `0.3749999936359625`; do not use coordinate 0.375 as sigma in reconstruction. Mixed continuation has 12 protected + 50 generated latent frames, source H/W 40×52, target 56×74. Its high sequence is actual → forecast → actual. The whole two-chunk run totals 18 logical / 14 actual / 4 forecasts: low 10/8/2, probes 2/2/0, high 6/4/2.

Final canonicalization reports 266,641 changed protected elements, RMS `7.735740581438222e-8`, maximum `4.76837158203125e-7`, no nonfinite changes, and final exactness. That RMS is over changed finite elements, not all latent elements.

Log line 644 reports successful `cute_sm120`, 200 mixed weighted calls, 8,991,600 Q rows and the same number of KV rows, empty compatibility fallbacks and empty dense-provider failures. Probe line 655 reports 50 weighted calls / 2,247,900 rows each; its backend field is absent, so do not independently claim sparse-kernel execution for that one-call warmup. Ordinary high-stage line 700 reports native VDN global/anchor route counters. Therefore “no fallback” is established for the **mixed weighted path**, not globally for all native VDN subroutes. This does not establish every kitchen/CUDA/HIP acceptance gate.

## 3. Exact sampler/state model

### Coordinates and quantities

Use internal packed model coordinates. `P_in` and `P_out` denote the model's latent processing functions, with correct `latent_shapes` installed. Let `s` be **video sampler sigma**, `a=1-s`, `g=model_sampling.noise_scale>0`, `L=P_in(latent_image)`, `N` the sampler noise argument, `X` the internal sampler state, and `C` the accepted denoised prediction. Video latent format has scale factor 1; packed audio has additional scaling below. Core skips `P_in` on an entirely zero latent image; this is harmless for the audited zero-preserving H3 transforms but must be covered by compatibility tests.

Core combines `ModelSamplingAV` with `CONST` for FLOW_AV. `CONST` implements:

```text
initial X = a L + s g N
calculate_input(s, X) = X
C = X - s V_effective
inverse_noise_scaling(s, X) = X / a
```

The initial affine equation describes sampler entry, not a claim that an evolving state is still the original Gaussian plus a fixed clean endpoint. `C` can include CFG and wrapper effects. On generated suffix rows, `V_effective=(X-C)/s` is the vector implied by that accepted prediction. It is the quantity `to_d` uses in the first RES update. It is not necessarily the bare network velocity or the derivative of an exact continuous trajectory followed by a finite-step solver.

### Object trace

| Object / boundary | Actual meaning |
| --- | --- |
| Caller `noise`, `latent_image` | Target-sized noise argument and external clean/conditioning latent. They are independent inputs. |
| Low input | Video noise generated on source grid with `seed+source_noise_offset`; caller audio noise copied. Latent and mask are privately resized. Mixed transformer prefix independently uses original target prefix/noise. |
| `low_result` at nonzero `s` | `P_out(X_source/(1-s))`, after sampler return conversion; generally **not x0**. |
| `_raw_sampler_state` | Installs source shapes, computes `P_in(low_result)*(1-s)`, restores prior shapes. Recovers the internal sampled state, modulo conversion roundoff. |
| `_noise_argument` | `(X-(1-s)L)/(s*g)`, so normal sampler initialization reconstructs the requested state. This synthesized argument need not be Gaussian. |
| Exact probe | One existing model call at the same `X_source,s`; custom sampler returns `C*(1-s)`, undoing the subsequent inverse scaling. No solver advance. |
| Probe executor return | `P_out(C)`; `_process_latent_in` recovers internal clean `C`. |
| Upscaler input | A separate clean copy; for Mixed-Grid its prefix is replaced with bicubic original-prefix context. Generated suffix remains the accepted probe x0. |
| Learned target | Clean video returned by `upscale_clean_video`, not a sampled state. Audio never enters this API. |
| Current `target_raw` | For changed geometry, `(1-s)*learned_clean + s*fresh_noise`, then affine clean repair and prefix replacement. Source video state is discarded. |
| High input | Noise argument reconstructed from `target_raw` and caller `L`, protected noise merged from caller, then core `noise_scaling`. |
| Prediction/callback | Native inpaint modifies model input/output as below; raw Flow event precedes guidance; callback denoised follows wrappers and output masking. |
| RES update | Model → callback → Euler on first high interval; subsequent intervals use new high-stage multistep history. |

Never compute a residual from `low_result`, an earlier callback x0, a different sigma, or the upscaler-context prefix. Capture probe clean separately from the context-modified copy. Restrict transported residual to mask-one generated suffix frames.

### Velocity versus effective endpoint noise

For `0<s<1` define:

```text
V = (X-C)/s
R = X-C = s V
E = [X-(1-s)C]/s = C+V
N_effective = E/g
X = C+s V = (1-s)C+s E
```

No `L` term or extra `(1-s)` factor belongs in `V`. `L` belongs only to reconstructing the sampler **argument** `N`. `g` is already represented in `X`; dividing the velocity by `g` would change its units. `E` is an inferred endpoint field, not proof of a standard-normal sample or equality to the original low noise.

### Protected prefix

`KSamplerX0Inpaint` forms model input `mask*X+(1-mask)*scale_latent_inpaint(...)` and returns `mask*C+(1-mask)*L`. Native H3 uses visual conditioning augmentation `aug*L_video+(1-aug)*N_video` (audited default aug 0.999), not the ordinary `aL+sgN` rule, on exact protected video. It also reconciles token-grid masks. Hence the protected prefix's solver `X`, injected model input and clean returned prefix are different objects. Do not require raw sampler prefix to equal the injected conditioning prefix.

Flow's `_merge_preserved_noise` must continue restoring caller noise on mask-zero elements. Keep the original masks, augmentation semantics, original target latent, and final exact canonicalization; never transport low-carrier prefix residual into caller-owned rows. The first raw event includes the native augmented-prefix prediction before output canonicalization; callback clean measurements are the appropriate complementary check.

### Packed audio

Let video/audio shifts be `b_v,b_a`, `k=b_v/b_a`, and `s_a` the native time-shifted audio sigma. Core carries audio as `Y=(s/s_a)*x_audio`; `P_in` multiplies clean audio by `k`, so carried clean endpoint is `k*C_audio`. With `d=k+(1-k)s`, `s_a=s/d`, `Y=d*x_audio`, and `d(1-s_a)=k(1-s)`. Thus the packed sampler uses one video-sigma affine schedule while native audio still has its own effective schedule.

Native forward converts carried input back to audio coordinates and transforms its velocity back by the chain rule. Audio cannot be treated as an unscaled video stream or re-noised at audio sigma by this handoff. Copy the recovered carried audio `source_raw` slice unchanged; preserve its existing high initialization/mask path. This establishes handoff-state preservation. It does **not** guarantee identical future audio predictions when changed video influences a joint model; production audio fidelity must be checked.

## 4. Root architectural discontinuity

`build_handoff_state` accepts sampled state and clean prediction but, for changed spatial geometry, reads source video state only for shape/device/dtype. Both bicubic and learned routes replace its state content with clean transfer plus deterministic fresh target noise. Seeds are distinct: source offset `0x48334C4F574C52`, transfer offset `0x4833464C4F57`. No reduction relates source video noise to caller target noise. This behavior is also explicitly asserted by the existing learned-handoff unit test.

The confirmed discontinuity is **loss of sampled video displacement at handoff**. Clean target reconstruction, exact-prefix representation/DC reconciliation, audio carry, geometry and conditioning survive. This is compatible with first-high-call re-framing; causal attribution to this particular loss remains an inference. Changing resolution, native target-model dynamics and sparse/VDN topology at the high boundary may also affect that first prediction. The evidence excludes downstream operations as initiators, not every upstream input to the first call.

## 5. Solution-space comparison

Let `U` be a fixed linear spatial lift and `C_T` the reconciled learned clean output.

| Family | Semantics / invariant | Cost and limitations | Decision |
| --- | --- | --- | --- |
| Direct state lift | `X_T=U X_S` | One resize; preserves carried state under U but ignores nonlinear clean transfer/repair. Learned noisy-state upscale has no supported provider contract. | Use only as linear-transfer oracle. |
| Velocity/displacement lift | `X_T=C_T+U(X_S-C_S)`; implied `(X_T-C_T)/s=U V_S` | One suffix resize and additions; interpolation approximation, no target vector-field equality guarantee, altered effective noise law. | **Selected controlled candidate.** |
| Endpoint-residual lift | `X_T=(1-s)C_T+U[X_S-(1-s)C_S]`; inferred `E_T=U E_S` | Same cost; retains noise endpoint rather than tangent. Existing clean bridge coefficient remains `(1-s)`. | Coherent separate comparison if needed; not falsified. |
| Current independent re-noise | `(1-s)C_T+s Z_T` | Full-rank fresh noise; intentionally loses trajectory coupling. | Frozen baseline and rollback. |
| Reuse caller target noise only | `(1-s)C_T+s*g*N_caller` | Retains target noise identity, but still discards sampled source displacement and source noise is unrelated. | Insufficient state invariant by itself; no claim it cannot empirically help. |
| Couple initial low/high Gaussian fields | Derive low noise from target with a specified covariance-normalized restriction | Changes low trajectory and therefore the probe/upscaler result. Ordinary area/bicubic reduction is not iid-noise preserving. | Separate experiment, not bundled into first state repair. |
| Scale-consistent stochastic extension | Choose orthonormal-column U, restriction Uᵀ, add independent complement `(I-UUᵀ)z` to a transported Gaussian | Can preserve Gaussian covariance for independent Gaussian source; current bicubic U is not isometric and evolved residual is not known Gaussian. Requires new operator, normalization and tests. | Mathematically coherent future family; no ad hoc high-pass noise addition. |
| Later handoff / extra step / stronger guidance | Changes cost, schedule or denoiser influence | Does not enforce either transported-state invariant. | Diagnostic controls only; excluded from selected repair. |

The selected reconstruction is uniquely determined **after choosing** the clean endpoint `C_T` and tangent `U V_S`. This criterion is a design choice, not a consequence that follows uniquely from the sampler equations.

The two serious residual policies differ by:

```text
X_velocity - X_endpoint = s * (C_T - U C_S)
```

Both reduce to `U X_S` when `C_T=U C_S`; equal-grid and linear tests alone cannot distinguish them. The difference can be large at the observed sigma ≈0.878, especially with learned nonlinear transfer and one-token repair. A production comparison must label which policy ran.

Bicubic lifting of iid noise has covariance `U Uᵀ`, generally correlated and rank deficient. Per-pixel variance normalization does not restore independence. Accordingly this design transports an evolved displacement; it does not claim to sample the target prior. There is no scalar noise renormalization, random complement or fade in the selected implementation. If rank/spectral deficiency produces a concrete regression, investigate a covariance-defined extension separately and document changed invariants.

## 6. Recommended design

### Data flow and ownership

1. Keep low initialization, mask handling, mixed measure, seed and exact probe unchanged. Recover `source_raw` exactly as now. Retain immutable accepted internal probe clean before making the provider-context copy.
2. Unpack source raw and clean; slice only generated video suffix `[:, :, prefix_t:]`. Compute `R=(X-C)` in FP32 for FP16/BF16/FP32 inputs; preserve FP64 for algebraic fixtures. Reject invalid geometry, nonfinite sigma/state or unsupported partial-mask topology before reconstruction. Do not mutate caller tensors or retained trajectory samples.
3. Give the learned provider the same complete clean context sequence it receives today, exactly once. Validate its shape, dtype category and finiteness. Preserve full prefix context length; no unproved temporal truncation.
4. Retain learned clean output explicitly. Apply existing `apply_suffix_representation_bridge` followed by `apply_suffix_dc_bridge(weights=(1.0,))` in clean space. Preserve both bridge algorithms and their one-token support. Do not recover clean output by subtracting a guessed noise field.
5. Lift suffix `R` with framewise bicubic: latent H/W only, `align_corners=False`, `antialias=False` for these nonshrinking geometries, no temporal/channel mixing, no amplitude normalization. Reuse `resize_spatial_5d` for production dtypes; support a FP64-preserving test path or update its compute-dtype handling narrowly with existing behavior verified. A separate fixed state operator must not follow a user's guidance transfer-mode setting.
6. Set generated `target_video = corrected_clean + lifted_R`. Use original target prefix as staging values, then pack with cloned carried source audio. All generated suffix frames receive transported state; only clean bridge correction remains confined to the first suffix frame. This is intentionally different from promising that later suffix **state** is unchanged versus legacy re-noise.
7. Keep `_noise_argument(base_model,target_raw,s,target_latent_internal)` and `_merge_preserved_noise` intact. At suffix mask-one elements, core initialization must reconstruct the new `target_raw`; at exact protected elements, caller noise and native inpaint remain authoritative.
8. Start the same fresh high sampler lifetime. Keep first actual, sigma sequence, guidance initialization, conditioning rebuild and final canonicalization. No additional probe/model/optical-flow/decode call.

### Clean bridge mapping is policy-dependent

Under velocity transport, a clean correction `delta` maps to **`+delta` in state**, because `X=C+R`. Under legacy/endpoint transport it maps to `(1-s)*delta`. Do not call `map_clean_bridge_to_conditional_state` on a velocity-transport state. Apply clean repairs before constructing state; retain that old helper unchanged for old paths. Tests must distinguish these coefficients at nonzero sigma. Otherwise an implementation can pass generic shape tests while violating the chosen invariant.

The existing runtime uses `recover_conditional_clean_for_diagnostics` and then feeds recovered clean into repair despite that helper's “diagnostics only” docstring. For the new path, remove this inversion dependency, retain the actual provider result, and emit truthful direct-clean telemetry. Keep old arithmetic on the legacy path to preserve comparison semantics; do not claim bitwise equality from a mathematically equivalent rearrangement.

### Sigma, precision, lifetime

Production handoff remains strictly `0<s<1`; do not expand supported scheduler endpoints. `C+U(X-C)` itself is division-free. A pure helper may define the clean endpoint only when `s=0` and `X=C`, but the runtime must never call nonzero-return recovery at `s=1` or noise reconstruction at `s=0`. Test near both ends without clamping sigma or silently choosing another policy. `P_in(low_result)*(1-s)` can amplify prior rounding near one; retain current recovery for bounded scope and measure its roundtrip error. A future direct-state capture requires separate solver/callback-lifetime analysis.

Move operands to the current state device and compute dtype explicitly; never assume outer-wrapper caller tensors are already on CUDA. Preserve original dtype on packed output, and do not cast unchanged audio through a wider/narrower precision. Release provider clean/context/displacement tensors after transfer; do not store them in global metrics, persistent model options, trajectory archives or serialized workflows. Keep exception cleanup and active-run invalidation.

### Guidance, Spectrum and identity

Direction and temporal guidance still consume clean source trajectory x0 and target predictions. Acceleration already derives effective velocity as `(high_state-clean)/sigma`; no formula change is warranted. Reset guidance state at the same stage boundary and do not inject transported velocity as a synthetic previous acceleration sample. Source trajectory identity stays unchanged; the new transfer policy must participate in any existing configuration/run identity that actually includes handoff semantics, and telemetry must record it. Do not add it to low/probe attention-measure identity: that operator is unchanged.

Spectrum starts the same independent low/probe/high lifetimes; no fabricated history, receipts, calibration or forecast promotion. Do not transport `old_denoised`, SA-PECE history, RNG objects or BSA pools. The current RES first model call does not consult `old_denoised`; preserving it cannot explain or repair the pre-update raw seam. Solver-history transport is a separate problem.

## 7. Implementation map

| File/function | Planned change |
| --- | --- |
| `handoff.py::ProgressiveTargetInputConfig` | Append a versioned optional state policy, preserving positional compatibility; absent means legacy. Validate applicability independently of attention measure and bridge booleans. |
| `handoff.py::build_handoff_state` | Preserve existing legacy signature/return behavior. Factor clean video transfer/validation into a reusable internal helper; add a dedicated Mixed-Grid preparation result holding clean video, carried audio and suffix displacement or expose equivalent explicit keyword inputs. Do not overload `source_x0` with a noisy state. |
| New `handoff.py` helper or small `state_transport.py` | Pure suffix displacement lift/reconstruction with explicit policy, geometry, compute dtype and finiteness contract; no model/guider/global state access. |
| `runtime.py::_run_progressive` | Capture accepted probe clean separately, run clean transfer and bridges, construct new suffix state, retain protected-noise merge and audio. Branch before legacy inverse-noise clean recovery. Emit versioned transport policy and algebraic diagnostics. |
| Mixed-Grid node in `target_sparse_node.py` | Append optional `handoff_state_policy` with `legacy_renoise` and `velocity_bicubic_v1`; default legacy until media acceptance. Generic Source/Target Input and Target-Sparse keep prior behavior. A first chunk with no actual mixed plan remains legacy in this isolated seam comparison. |
| Tests | Extend `tests/test_handoff.py`, `tests/test_mixed_grid.py`, existing runtime/exact-mask suites and source-contract lane; add pure mathematical tests. Do not rewrite legacy assertions into new-policy assertions. |
| Workflow/docs | Document the explicit experimental selection and rollback; preserve existing saved graphs. Promote new-graph Mixed-Grid defaults/workflow examples only after matched media acceptance. Do not conflate measure profile with state policy. |
| Optional diagnostics | Reintroduce only necessary output-neutral PR #32 instrumentation from clean #30, using a shared transient holder (§9). Restore unrelated comments/docstrings from #30. |

Leave representation/DC algorithms, `mixed_grid.py`, `attention_measure.py`, geometry/RoPE contracts, guidance equations, `comfy_compat.py` canonicalization and all companion production code unchanged unless an executable test proves a specific integration requirement. A new endpoint-residual comparison can remain a diagnostic-only mode on the mirror; it need not become a permanent UI option.

No production implementation belongs on this design branch or PR #29/#32. Develop on a new mirror from the then-current #30 head; eventually consolidate onto existing #30 as one clean implementation commit, preserving established authorship and dependencies. Do not merge/release on CPU results alone.

## 8. Compatibility and invariants

* Equal-grid helper path returns the original state (clone where required) before learned transfer or random generation. This identity assumes no requested clean re-anchoring; a separate clean correction is not an identity operation.
* On generated suffix, exact arithmetic gives `X_T-C_T=U(X_S-C_S)`. For linear clean transfer `C_T=U C_S`, reconstruction equals `U X_S`. Floating implementation is checked with scale/dtype-aware tolerances.
* Clean repair maintains the existing learned-native boundary relation. Prefix values remain caller-owned; no resizing of protected H3 transformer conditioning or final output values.
* Transfer audio is byte-preserved in carried coordinates. Same caller-visible video/audio shapes, masks and session/chunk ownership. No downstream decode change; Flow #25 remains independent.
* Low/probe retain all Q/K/V, natural-log key measure, exact non-unit K blocks, independent Q/K exactness, unweighted calibration, native layouts/RoPE and VDN API 2. High remains ordinary target-grid attention.
* No H3 NFE or sampler interval added. Expected 00383 topology remains 18L/14A/4F, two probes, six sampler invocations and four history boundaries. This count is the matched fixture, not a universal promise for other samplers.
* Fixed input/policy/seed yields deterministic reconstruction subject to existing device operator determinism. New policy consumes no new RNG. Keep source noise offset and caller protected noise unchanged.
* Loader selection is unaffected. Do not add an allowlist for KJ/Sage/Sol; verify actual model-sampling equations and packed shape contract. Existing rejection of external mutable `noise_sampler` persists.
* Supported samplers retain their existing restart semantics. Stochastic samplers may inject subsequent noise; this proposal does not claim full stochastic-path continuity or inherited multistep order across grids.

## 9. Tests and production acceptance

### Structural gates

Run pure tests first, then existing targeted native/runtime tests; use normal repository CI once at the reviewable mirror head. No redundant full-suite campaign.

| Test | Required evidence |
| --- | --- |
| Source-return/probe conversion | Exercise actual pinned `KSAMPLER`, `CFGGuider`, FLOW_AV model transforms: recover `X` from nonterminal result; recover `C` from probe; include nonzero L, g≠1 and audio scaling. Stub-only duplicated equations are insufficient. |
| Displacement roundtrip | `C+(X-C)` matches X; target implied displacement matches U residual. No divide-by-sigma implementation needed. |
| Linear identity | Identical fixed linear U for clean and residual yields `U X`; equal-grid bypass is exact and does not call provider/RNG. |
| Nonlinear policy distinction | Synthetic `C_T != U C_S` proves velocity versus endpoint difference `s*(C_T-U C_S)` and clean repair coefficient 1 versus `1-s`. |
| Noise argument | Initialize core with reconstructed N and original L; match requested suffix X for g=0.5,1,2 and nonzero L. |
| Sigma | Near 0 and 1 finite valid cases; endpoints rejected by production API; no clamping or NaN/Inf accepted. |
| AV/geometry | Non-square/one-axis enlargement, B>1, packed video 24/audio 32 channels; unchanged carried audio, valid original target geometry. |
| Ownership | Inputs, source trajectory, caller prefix/noise/masks unchanged; bridges touch only first clean suffix frame; transport affects every suffix state frame only. |
| Precision/device | FP32 production, FP16/BF16 inputs, FP64 oracle; CPU caller/CUDA state, noncontiguous tensors and cancellation cleanup. |
| Compatibility | Legacy exact outputs remain; absent new option resolves legacy; generic and first unprotected chunk stay old path; custom noise sampler rejection remains. |
| Stack | Same actual/forecast topology and provider receipt ownership; real native wrapper order, no weighted high contract; guidance receives correct state and starts without false velocity history. |

### Diagnostic provenance

Core `create_model_options_clone` calls `copy_nested_dicts`: dictionaries recursively copy, lists shallow-copy, other objects retain identity. PR #32 mutates nested dict call index/history while callbacks read another dict, explaining null provenance. Later callback state labels can consequently be wrong; do not use those labels to claim later callbacks still precede all updates. Event order and native RES source establish actual → forecast → actual and per-iteration callback-before-update independently.

Use a per-high-invocation **non-dict holder object** containing only scalar call records. Place it in the contract and capture the same object in the callback closure; verify identity through actual cloning. Allocate afresh for every invocation, release in `finally`, reject nested cross-run ownership, and test exceptions and two simultaneous invocations. Do not store tensors or receipts as fabricated provenance. Require direct model-call actual/forecast evidence. If installed cloning becomes true deepcopy, use an explicit clone-sharing protocol scoped to that invocation and test it; do not assume current behavior forever.

Record clean boundary, reconstructed suffix algebraic error, actual first model-input boundary after native inpainting, raw/guided output and callback clean boundary, subsequent callbacks and final prefix exactness. Report scalar RMS/max and policy; do not persist full tensors by default. For new policy, measure actual retained clean result directly, not legacy inverse re-noise. Also record `C_T-U C_S` and low-frequency displacement statistics to interpret the two candidate policies without extra H3 evaluations.

### Matched CUDA/media sequence

1. Obtain the actual installed source manifest and original 00383 workflow/seed/reference/prompt/settings; freeze all of them, including model/LoRA, scheduler, guidance, upscaler, VAE/decode, attention provider and patch ordering. The available metrics/log do not substitute for missing workflow assets.
2. Rerun legacy with corrected output-neutral diagnostics to establish same-stack baseline/repeatability. Run `velocity_bicubic_v1` with only this policy changed. The no-prefix first chunk must remain legacy so its accepted tail is the same conditioning input to the mixed chunk.
3. Compare repaired clean boundary → first raw actual jump, absolute first raw/callback boundary, and full decoded seam/framing/motion/reference/audio behavior. Inspect the entire later suffix for delayed pulses. Require material relative improvement beyond baseline repeatability without trading it for composition, audio, detail or motion loss. Do not invent a numerical pass threshold from one baseline.
4. Verify 18L/14A/4F, real weighted mixed SM120 route/no mixed fallback, exact prefix, unchanged low/probe evidence, normal high ownership, peak VRAM and hot transfer/sampler/end-to-end cost. Expected added work is one spatial suffix resize plus elementwise operations; actual performance remains unmeasured.
5. If velocity transport fails the behavioral gate, do not promote it or add fades/extra actuals. Run one matched **endpoint-residual** comparison to distinguish tangent preservation from endpoint preservation. This changes state coefficient and bridge mapping deliberately; it is a different policy. If both fail, inspect spectral/noise-law or target vector-field mismatch with saved bounded diagnostics before designing another operator.
6. Before default promotion, check another seed/geometry and the supported sampler paths whose wrappers are exercised by the change. Preserve legacy rollback at new-run boundaries. Structural success authorizes a CUDA candidate, not a claim of solved media behavior.

No PyTorch or CUDA is installed in this investigation environment. Existing repository tests were inspected, not executed. A 24-case NumPy algebra check covered FP32/FP64, sigma near endpoints and g=0.5/1/2: maximum state roundtrip error 1.19e-7, noise-argument closure 5.96e-8, linear identities/policy-difference up to 1.53e-5 for the sampled arbitrary linear matrices. These are sanity checks, not production tolerance recommendations or native-core test results. A covariance counterexample also showed variance-normalized interpolation retaining off-diagonal correlation 0.7071 and rank 2 in a 3-dimensional target.

## 10. Rejected / constrained solution space

| Claim or approach | Classification and scope |
| --- | --- |
| Flow guidance initiates 00383 worsening | Falsified at observed wrapper boundary: raw worse first, guidance reduces all three metrics. Does not prove guidance is globally optimal. |
| First high Spectrum forecast initiates it | Falsified: first high is actual; forecast follows. Earlier low history is not thereby proved irrelevant. |
| Current first RES update initiates it | Falsified by raw/callback ordering and source. Does not prove every earlier state integration choice irrelevant. |
| Final exact-mask drift causes it | Falsified for observed defect: ULP-scale restoration is later and protected-only. |
| Purely VAE/assembly origin | Falsified for latent measurement; decode can still change visibility and has an independent context issue. |
| Representation/DC repair is useless | Falsified: clean splice restored to learned-native metrics. Real repair, insufficient by itself; retain it. |
| Source geometric/affine trajectory warp | Retired/failed under recorded finite-horizon contract: Flow #24 documents 00318 horizons moving the discontinuity; current runtime confirms disabled state. Original 00318 media was not re-reviewed here. This is not a falsification of state residual transport. |
| Representative K/V deletion | Superseded by #29/#30 weighted all-row architecture; retain only explicit legacy comparator. |
| Weighted mixed route inactive/falling back | Ruled out for 00383 mixed low stage by route counters; not a universal kernel-accuracy or kitchen claim. |
| Fresh re-noise is mathematically invalid | Not established. It is a legitimate restart policy that discards trajectory state. |
| Velocity transport necessarily solves framing | Inconclusive until controlled target-model/media run. Algebra does not establish resolution equivariance. |
| Caller-noise coupling alone or endpoint transport cannot work | Not established; coherent but distinct alternatives. |

## 11. Unresolved questions

| Uncertainty | Importance / re-check | Blocking scope |
| --- | --- | --- |
| Installed `patcher/stack` bytes | Export relevant functions and patch list for `7803b74e`; compare sampler conversions, native mask behavior and wrapper ordering with §3. | Blocks claiming exact production-source equivalence; must precede production acceptance. Does not block implementing pure tested candidate. |
| Target model responds favorably to lifted velocity | Run §9. H3 is not proved equivariant to bicubic latent lifting. | Blocks promotion/“fixed” claim. |
| Residual covariance / missing high-frequency degrees | Inspect transported displacement statistics and media; no iid assumption. Use endpoint control to isolate invariant choice before stochastic redesign. | Empirical gate, not justification for speculative correction. |
| Original workflow/media assets | Metrics/log alone do not encode all settings or perceptual evidence. Retrieve original run assets or have user execute the preserved production workflow. | Blocks a matched media claim. |
| Exact installed sibling source gates | Source-pinned Spectrum audits may reject a changed function even when attention semantics are unchanged. Inspect live allowlists and run native integration before CUDA. Update only a proven dependency with unchanged proof obligations. | Potential integration gate; no companion mutation currently justified. |

## 12. Implementation assumptions requiring live re-check

Re-fetch all PR heads/bases/reviews/CI, clean #30 source and any rebased companions. Re-check `build_handoff_state` signature, `_run_progressive` ordering, actual accepted probe position, model-options cloning, FLOW_AV/CONST algebra and noise_scale, H3 masking/packing/audio transform, RES callback placement, custom-noise restrictions, wrapper receipt ordering, VDN API 2, node optional-argument serialization, upscaler API 1 and workflow defaults. Preserve #29 design-only and #32 diagnostic-only. Adapt to contradictory source with documented evidence and a revised invariant; never blindly copy snapshot SHAs or function fingerprints.

## 13. Artifact registry

| Artifact | Exact reference / disposition |
| --- | --- |
| Primary metrics | Library `/ComfyUI-Sol-H3/metrics_00383_.json`, `libfile_a814372042488191be0a43b2a580ce5f`; local `/workspace/scratch/266da105e66e/evidence/ComfyUI-Sol-H3/metrics_00383_.json`. Preserve and inspect. SHA256 `6917146613c75c6e22914564379381abd316b03b8c3fc456f6fce935b6e87cb5`. |
| Matching full log | Library `/ComfyUI-Sol-H3/Pasted text(20260911-174109).txt`, `libfile_92d79c4c490c81918427d577b03ea782`; local `/workspace/scratch/266da105e66e/evidence/ComfyUI-Sol-H3/Pasted text(20260911-174109).txt`. Preserve and inspect. SHA256 `a2898654826a1e44032456c12dfc28c43494b7cddb9b39452597d2c57b3257cc`. |
| Source extracts | `/workspace/scratch/266da105e66e/sources/`, including `core/`, `diag/`, `tests/`. Read-only extracts, not a full checkout. Re-fetch pinned/current sources; disposable once reproduced. |
| Numerical sanity script | `/workspace/scratch/266da105e66e/analysis/state_algebra.py`. Reproduce §3/§5 algebra tests in actual PyTorch/native test suite; scratch script is disposable and not a runtime implementation. Essential equations/results are recorded here. |
| Design | Repository `docs/MIXED_GRID_HANDOFF_STATE_TRANSPORT_DESIGN.md`; local `/workspace/scratch/266da105e66e/Flow/docs/MIXED_GRID_HANDOFF_STATE_TRANSPORT_DESIGN.md`. Preserve committed version. |

No profiler traces, saved handoff tensors, test binaries, production patches or temporary datasets were produced. No new media was generated. Scratch sources are reproducible; primary uploaded evidence retains its persistent identities.

## 14. Repository / branch / checkpoint state

Planning branch: `design/mixed-grid-state-transport-20260911`, created directly from verified #30. Initial evidence/equation checkpoint commit: `9bbbaa912ccf063f40489cbb16eb64ca332871b4`. Baseline safety ref: `checkpoint/state-transport-baseline-20260911` at `8dfb7a2e570458b5078734755b897c7e89ddac8c`. The final document commit is the commit containing this version (reported with its immutable URL in the task handoff); a final safety ref is created after commit to avoid a self-referential SHA in the document.

| PR | Verified head | Status / current PR CI |
| --- | --- | --- |
| Flow #29 | `3afc6c63063f410afec756c42811a8cf1ad2ed77` | Open design; success 34520037381 |
| Flow #30 | `8dfb7a2e570458b5078734755b897c7e89ddac8c` | Open, 1 commit; success 34577142084 |
| Flow #32 | `f0f0b1e5a54de6ff31a31a795a298ab9b1961dc7` | Open diagnostic stacked on #30; success 34627393485 |
| Core #16239 | `8406da920f9df19cbc14b76aef1bcd3dc4cf1119` | Open; PR workflows action_required, not executed validation |
| Kitchen #171 | `663bfa3ab5c29e9cff6b2ab68d7f287ea54ae6bb` | Open; Build Wheels 34563734866 action_required |
| Sol-H3 #9 | `fdd52bbd07bd88dfe3d31c823de0d14c27c74aff` | Open; CPU contracts success 34582139977 |
| Spectrum #110 | `78a9a5b36c185a55af59a44c073bc7f8e534bc97` | Open; tests success 34581917615 |
| VDN-Plus #14 | `d5f158a1c1750d79b37cb6ef22b0f14e6a8cbad6` | Open; CI success 34555404823; #8 now stacked above it |

Flow #29/#30/#32 had no submitted reviews and no requested reviewers at audit. Current comments are automatic review-skipped notices. Companion PR descriptions contain historical/stale acceptance statements; source and 00383 evidence take precedence within their exact scope. Flow #25 was fetched as an independent decode-context PR, not modified. Existing `checkpoint/diag-seam-00383-localized-20260911` was located; this investigation does not move it. No production, companion, #29 or #32 branch was mutated.

Narrow history informing this design: [Flow #1](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/1) introduced progressive split sampling; [#11](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/11) explicitly preserved conditional re-noise while replacing clean transfer; [#20](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/20) established mixed carrier/prefix ownership and DC bridge; [#21](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/21) explains terminal ULP canonicalization; [#24](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/24) records retired source warp; [#26](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/26) fixes topology publication. These explain existing contracts, not independent proof that the proposed transport improves media.

## 15. Dead ends proved false / failed approaches

The scoped classifications in §10 are the authoritative negative findings. Do not restart guidance/forecast/current-update/final-canonicalization/decode-origin investigations without contradictory primary evidence. Do not reintroduce source geometric warps, representative deletion as the canonical measure, synthetic first-high x0 blending, fades, or extra H3 calls as a substitute for testing the explicit state invariant. Endpoint-residual and stochastic-field alternatives remain **unresolved**, not failed experiments.
