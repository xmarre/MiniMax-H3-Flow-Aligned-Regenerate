# First-high explicit-contract solution design

Status: authoritative **design-only**, 2026-09-15. Production implementation is gated. No source defect has been established that justifies changing production noise, sampling, attention policy, or weights. The next implementation is the bounded operator comparison in section 8; a favorable result requires an evidence-based amendment before production work.

This supersedes experiment selection in `FIRST_HIGH_STAGE_ARTIFACT_ARCHITECTURE.md` at `3b86aa212743e972cdd922bd9f235c0824346569`. Preserve that prior design as evidence. R has completed; do not repeat it as a lifecycle investigation.

## 1. Problem and causal boundary

Unprotected `H3ProgressiveTargetInputHandoff` produces a visually acceptable learned clean target followed by a defective first-high raw/pre-guidance prediction. The controlled video target is `[1,24,52,64,64]`, private source 44x44, audio `[1,32,2,292]`. Requested base coordinate 0.35 selects original schedule index 5, base coordinate approximately 0.375, video sigma 0.8780487775802612. These coordinates are distinct.

R reproduces the same first-high H3 input and output in a fresh high-only process. The explicit state plus active model/operator/conditioning contract is sufficient to reproduce the failure without the earlier low/probe/upscaler lifecycle. Retained lifecycle contamination is strongly weakened at this boundary.

This does **not** establish that an invalid implementation contract exists. A valid denoiser can change a supplied clean estimate after re-noising; an approximate/distilled model may perform poorly on that state. Only `1-sigma = 0.1219512224` of the clean target remains in initialized video. Clean decode and affine reconstruction do not establish that the joint AV state is well matched to the effective model's distribution.

Remaining alternatives include the explicit attention operator and limitations shared across operators in state distribution, conditioning, adapters, or the common H3 path. Source has not selected a production owner. The comparison below separates useful branches while holding the target state fixed.

## 2. Verified facts and evidence limits

Read and parsed all four R files: `replay_report_00001.json`, `executioncontractreport_00012.json`, `metrics_00463_.json`, and `Pasted text(20260915-032843).txt`.

- Dedicated report: `attribution_valid=true`; checkpoint, provenance, condition pre-core, guidance, Spectrum configuration, runtime policy and companion policy gates pass; active Untwist absent.
- Execution: high-only 3L/2A/1F, zero learned-upscaler calls. Log independently reports two transformer actuals, one forecast and no replayed transformer calls.
- First sampler video/audio, H3 video/audio inputs, H3 video/audio velocity, model raw video/audio, and pre-guidance video hashes match capture exactly.
- Last-high raw video differs. Last-high pre-guidance has no capture hash. These do not invalidate first-high attribution or establish a later-trajectory cause.
- Generic report captured 266412032 bytes under a 268435456-byte ceiling. Progressive-only incompleteness and the downstream PR #36 extractor failure do not invalidate R. No budget-only recapture is needed.

| Boundary | Capture/replay SHA-256 |
|---|---|
| First sampler video | `79dca62849b2a159061a1d6204828af7a4c5995c41beec256c1719a5932a41ee` |
| First sampler audio | `f5fd588101bf2ab68eb5aeeb49bfaf8e58f4c71aa99864de89cb1ddef8c859f3` |
| First H3 video input | `b49f17a317530795be43bc775486777b07d5b5dbb28996819033cdede64195c0` |
| First H3 audio input | `f54b22ff25a675c47a4c32aba54b5642033b048441ad6eec9bf5ef935cd29002` |
| First H3 video velocity | `5137e4a2accec6a28ed3012550ee7f56bcd06a7a495af41daae20f9f24df21d7` |
| First raw/pre-guidance video | `73587c900e2ac2bfefa4d9163874c105655e97b1cb0493cd1451d1fc04e6dd6b` |

Measurement precision: #41's sampler-input snapshot is a CONST forward reconstruction from real `SAMPLER_SAMPLE` arguments, not a hook after `KSAMPLER.noise_scaling`. Downstream model-input and exact H3 input hashes corroborate equivalence in this unmasked run. Do not generalize that reconstruction to arbitrary sampling subclasses/masks. The generic learned-clean snapshot is recovered from X and deterministic noise; direct provider comparison requires the capture bundle/#35 capture.

Conditioning evidence establishes pristine target versus high **pre-core** equality. Processed conditioning is observed, not compared to a separately executed oracle: `extra_conds` runs H3 text preprocessing. Some metadata tensor digests are sampled; require full tensor hashes for new exact claims.

### Recovered #37 route receipts

`Pasted text(20260914-030325).txt` accompanies `metrics_00442_.json`. First high reports `phase=dense`, sequence length 56349, routes `vdn_anchor_native,vdn_dense_warmup,vdn_global_native`, count 700. Later high calls transition to SOL. Untwist preprocessing is active in that run.

Thus #37 exercised native **local-window** softmax at first high. With the established artifact-persistence observation, this weakens missing first-high SOL warmup as a sufficient explanation. It does not clear VDN's restricted support, gate, linear complement, adapters, or the no-Untwist R counterfactual. Prefix and attention routing changed together.

R first-high receipts: 50 `vdn_global_native`, 100 `vdn_anchor_native`, 22 `vdn_dense_warmup`, 528 `vdn_local_sol`: 700 total. Fifty blocks each have eleven local groups; the first two blocks use native locals. Attention subcalls are not H3 NFEs.

### Media scope

Inspected one-second frames from `learned_transfer_clean_00001.mp4` and `firsthighpreguidance_00001.mp4`. Both show a coherent scene with facial-detail changes, not the global colored corruption of rejected transport. They are older files and are not hash-linked to R. R's known-broken classification and #37 artifact persistence remain supplied controlled media observations; this investigation does not claim independent full temporal validation or a hash-linked R video inspection. Papers cannot establish installed behavior; none supplies a root-cause claim here.

## 3. Reconstructed first-high execution contract

Source/ref/SHA comparisons are in section 12 and `FIRST_HIGH_EXPLICIT_CONTRACT_SOURCE_AUDIT.json`.

1. `runtime._run_progressive` retains caller target geometry, latent and noise; low uses a private grid. Protected-video `exact_prefix_mode=fallback` executes the untouched target-grid sampler without this learned boundary.
2. `_raw_sampler_state` applies `process_latent_in` to low's returned latent, then multiplies by `1-sigma` to undo terminal CONST scaling. This is a sampler state, not x0.
3. `_exact_probe_function` executes one exact low-grid prediction at handoff. Its return compensates terminal scaling; `_process_latent_in` restores internal clean AV. #35 captures and #36 guidance-anchor ownership remain intact.
4. `handoff.build_handoff_state` sends only clean video once through `H3LatentUpscalerProvider.upscale_clean_video` → `upscale_clean_video_exact`. The provider copies input, applies checkpoint normalization, runs its network, denormalizes, and returns exact geometry/dtype/device. Inspected core video latent scaling is unity; no double normalization is demonstrated.
5. `deterministic_video_noise` uses a private CPU generator, float32 Gaussian samples, seed plus `seed_offset`, then converts placement. `conditional_renoise_target` constructs `Xv=(1-sigma)Cv+sigma*Nt`. Audio is copied from the low sampler state, not re-noised.
6. `_noise_argument` constructs `(X-(1-sigma)*Linternal)/(sigma*noise_scale)`. Core initializes `sigma*(noise_scale*Nargument)+(1-sigma)*Linternal`. Coherent CONST algebra is not bit reversibility or distributional proof. R restores the original noise argument. Recorded high latent is all-zero float32; core's empty-latent conversion skip is harmless here.
7. `_reset_guider_conds` reinstates target conditions. `MiniMaxH3.extra_conds` preprocesses text and rebuilds payload/layout with seed, references, masks and audio scale. R has one independent image reference, no keyframes. Segments: text 1493, reference 1024, audio 584, video 53248, total 56349. The 64x64 image reference is not stale low-grid generated video.
8. `_high_stage_contract` publishes API 1, active, prefix 1, sigma reference 1.0, source `h3_flow_progressive_handoff`; `_flow_stage_contract` publishes high. Both restore prior values. Core `inner_sample` clones option containers and publishes the actual child suffix in `sample_sigmas`.
9. Spectrum begins an empty high history; its coordinate 1.0 is child-local policy position, not video sigma or Flow base coordinate. First high is actual. DiffAid uses sigma reference 1.0 and reports normalized sigma 0.8780487776. Untwist is inactive.
10. CONST `calculate_input` is identity; core casts inference tensors. H3 receives timestep `1000*sigma_v`, then derives per-stream times. At entry: base 0.3749999936, sigma_a 0.6428571366, t_v 0.1219512224, t_a 0.3571428634. FLOW_AV audio_scale=12/3=4. H3 converts carried audio by sigma_a/sigma_v≈0.7321428525 before DIFFUSION_MODEL wrappers and converts velocity back afterward. Flow must not apply another audio rescaling.
11. VDN constructs 32x32 token geometry, F=52, radius=1/chunk=5, anchors=both; full coverage is false. Non-anchor video uses window softmax plus learned linear complement. Global and anchor queries use global support. Native VDN `_sdpa` does not call KJ Sage. SOL receives VDN-selected local domains; exact block fusion is independent of sparse attention selection.
12. H3 returns stream velocities; core converts to denoised output. Flow's underlying predict result is #35 raw. #36 guidance acts afterward, too late to originate this raw defect.

`FinalLayer.forward` ignores schedule for a single head; PDD banks select heads from current sigma and the next endpoint in `sample_sigmas`. The original high suffix preserves those endpoints. A diagnostic must not use `[sigma,0]`, a one-point probe or a renamed sampler. Record the actual head count and sampling/format classes before implementation; checkpoint filename alone does not establish them.

## 4. Ownership and wrapper order

Recorded order below is outer-to-inner within each API boundary.

| Boundary | Relevant order / ownership |
|---|---|
| OUTER_SAMPLE | invocation recorder → R replay → Flow → stage recorder → SOL SamplingWrapper → Spectrum external patch wrapper → KJ preview |
| SAMPLER_SAMPLE | Spectrum → recorder → R entry gate; core owns prepared arguments and child sigmas |
| PREDICT_NOISE | Spectrum → post recorder → #35 Flow capture/guidance → raw recorder |
| APPLY_MODEL | VDN compiler guard → recorder → Continuum Join → core |
| DIFFUSION_MODEL | VDN layout/runtime lease → audio adapter scope → SOL → Spectrum external patch wrapper → recorders → H3 `_forward` |
| Block execution | DiffAid replacements and SOL block/exact routing; VDN owns attention forward/gather/gate/projection/complement; adapter providers own hooks |

DiffAid's model-function wrapper is an additional enclosing call path, not a DIFFUSION_MODEL entry. DoRA owns bypass injection/removal; VDN owns packed adapter scopes and fused residual integration. KJ supplies the inherited generic provider, unused by observed VDN native closures. R weakens lifecycle hook multiplicity as cause, not every common adapter arithmetic error.

Continuum `model_patch.py` matches fetched source. It directly forwards when neither Continuum payload nor mixed keyframe/reference payload is present, as in R's reference-only unprotected case. Preserve the wrapper; removing it is unnecessary.

Flow context is child/invocation-owned; Spectrum histories and SOL requests are per child. VDN layout is per forward, buffers/weights model-owned. Core option cloning recursively copies dictionaries and shallow-copies lists; callable/provider objects remain shared. New work must not mutate `state.cfg`, model branches, native forwards, shared adapters, or pools.

## 5. Causal conclusions

- Retained low/probe/reload state is unnecessary for the recorded first-high output. No reset is justified.
- Guidance and a later forecast cannot originate first-high raw corruption; keep #36.
- #37's native-window execution leaves VDN's support restriction and complement active. Full support remains untested.
- Source/receipts demonstrate coherent geometry and schedule domains. No stale 44x44 layout, second upscale, double noise scaling, audio reset or DiffAid child-clock reset is established.
- Fresh re-noising intentionally breaks the realized source stochastic trajectory. Correct Gaussian statistics and affine coefficients do not establish appropriate joint state for the effective VDN/Turbo/RefDelta model. Rejected transport does not prove all fresh target states valid.
- R is a reproducibility oracle, not a clean-output oracle. Same-seed native 64x64 generation has different draws/trajectory and is not a same-state control.

## 6. Rejection ledger

| Approach | Status | Evidence and scope |
|---|---|---|
| Learned output visibly contains original artifact | falsified at established boundary | clean transfer precedes failure; noisy-state validity remains unproved |
| Guidance originates raw defect | falsified | pre-guidance already broken; separate #36 fix retained |
| Middle forecast necessary | falsified | #37 A/A/A still fails; prefix/attention confound limits stronger claims |
| Missing dense-local first-high warmup sufficient | weakened | recovered #37 dense-local receipt; Untwist differs from R |
| Untwist clock trial | unsuccessful/superseded | supplied matched rejection; R fails without active Untwist |
| Retained lifecycle contamination | strongly weakened | exact first-high R; later divergence separate |
| #39 displacement re-anchor | unsuccessful/rejected | new global corruption despite structural checks; no coefficient tuning |
| #40 endpoint transport | unsuccessful/rejected | closure passed, media failed; no renamed transport |
| Final VAE stitching as broad origin | falsified for earlier latent-stage defect | prior tests/localization; not every decoder defect cleared |
| Reset/reuse low VDN layout | unsupported | high rebuild and R match; no owner-specific reset evidence |
| Full-support versus hybrid operator | unresolved | no existing same-state comparison |
| Common conditioning/adapter/state/model path | unresolved | pre-core equality narrows it; does not clear shared numerics |

## 7. Ranked remaining uncertainties

1. Does full key support remove the defect at identical state and common native softmax arithmetic? This selects/rejects VDN support-plus-complement as a sufficient correction family.
2. Does native-window execution itself remove the no-Untwist R defect? That implicates local SOL routing and redirects work to that owner. Old #37 is not this matched arm.
3. If neither helps, the common state/conditioning/adapter/model path remains unisolated. No production patch is authorized; do not automatically tune noise or sigma.

Before new H3 evaluation, close the concrete source gaps in section 12 and inspect/hash the existing bundle. These are read-only prerequisites, not another CUDA experiment.

## 8. Chosen architecture: paired operator experiment W

### Arms and invariants

Use the existing R bundle, one first-high H3 evaluation per arm, in separate fresh processes. Preserve Euler object/name, complete original high suffix, state, conditioning, model weights/adapters, gates, compiler configuration, Flow metadata and guidance configuration. Each stops after the first model result. Total diagnostic cost: **2L/2A/0F, zero upscales**, no added production NFE.

| Arm | Non-anchor video attention | Linear complement | Purpose |
|---|---|---|---|
| W-window | Existing restricted KV domains, native `_sdpa` for all locals | Existing branch | Same-state no-Untwist native-window control |
| W-full | Same Q groups, all packed K/V, same native `_sdpa` | Disabled | Coherent full-support operator comparison |

Full support leaves no excluded domain for the complement; disabling it is mandatory. This is one **composite operator** intervention, not separate causal attribution to window size versus linear math. VDN gate, projection, global/anchor paths and all adapter strengths remain fixed. Changed video can affect later audio activations; exact audio input does not imply equal output audio.

Do not change global radius/config, fake `external_reduced`, mutate shared `full_cover`, remove VDN or bypass its forward. The normal full-cover branch calls generic optimized attention and can invoke KJ/SOL, confounding the comparison. Explicitly retain native SDPA in both arms.

### Patch-level ownership

**Flow:** add sibling `h3_flow_regenerate/first_high_operator_comparison.py` and focused tests on a new mirror. Reuse #41 bundle integrity/checkpoint/conditioning/provenance utilities and snapshots. Add separate nodes/report type; do not edit PR #41's branch or weaken normal R semantics. Changed operator policy must not emit R `attribution_valid=true`.

**VDN:** add a strict diagnostic contract parser; consume it in `vdn_h3.hybrid.make_vdn_forward` and `vdn_h3.retained.window_softmax_grouped_runtime`. W-window preserves existing gather plans. W-full preserves Q groups and native global/anchor calls but gives locals canonical full K/V. Determine complement activation before raw-QKV scratch allocation. No changes to `_scope_softmax_gate`, `out_proj`, adapters, RoPE or exact epilogue; no shared config/pool mutation. Disabled path adds no tensor allocation or synchronization.

**SOL:** `sol_h3.runtime.BlockWrapper` local-provider closure recognizes the validated mode, returns the VDN native closure, and emits distinct diagnostic native-window/full receipts. Keep exact fusion and nonlocal routes intact. Include mode in `interop.HistoryPolicy` before Spectrum preflight and accept only the explicit new receipt kinds. Do not overwrite provider keys after preflight.

**Spectrum:** use the existing backend-history API. Do not force counts or alter scheduling. First high remains actual. Report opaque/actual-only handling if encountered and test the expected identity/receipt behavior. No middle call executes.

### Data contract and lifecycle

Publish an immutable namespaced transformer-option request, e.g. `h3_first_high_operator_diagnostic_v1`, distinct from `h3_refinement`:

```
api: 1
capture_id: exact saved capture id
mode: native_window | native_full_support
stage: high
logical_call_limit: 1
sigma: exact first-high sigma
target_shapes_digest: recorded geometry digest
source_contract_digest: reviewed diagnostic source manifest digest
```

Flow validates before child OUTER_SAMPLE/preflight; VDN owns support/complement, SOL provider/history publication. Restore prior option values in `finally`. Recorders are context-local and bounded; copied request data immutable. Preflight never advances actual counters. Leave `h3_refinement`, original schedule and sigma units unchanged.

Fail closed for malformed/unknown API or mode, missing owner support, changed sigma/shape, non-Euler or churn, protected mask, missing companion, active weighted/external Mixed-Grid, or second model call. Absence of the request preserves all ordinary behavior.

### Termination and output contract

Keep the high suffix `[0.8780487776,0.8000000119,0.6315789223,0]` and real Euler sampler. Its callback occurs after the first model result and before the state update. A private completion sentinel from that callback can terminate the diagnostic after raw/pre-guidance capture and accounting. Catch only that exact sentinel at the diagnostic outer boundary after nested cleanup. OOM, cancellation, provenance failure and arbitrary exceptions remain errors.

Test that the installed core/Flow/SOL/VDN/Spectrum stack unwinds the sentinel correctly. If that cannot be established, amend this harness before CUDA; do not silently execute the full suffix or add a call. Do not shorten sigmas: that can change final-head selection and companion semantics.

Report `completed_first_call_only=true`, `final_trajectory_available=false`, 1L/1A/0F and zero upscale per arm. Save native H3 velocity video/audio, model raw, pre-guidance video, full input hashes, ordered routes and adapter/gate fingerprints. Expose converted **diagnostic x0**, never a counterfeit completed trajectory. Do not evaluate #35/#36 progressive-only extractors.

### Provenance and transparency

Keep the R manifest immutable. A sibling diagnostic uses an exact reviewed source-delta manifest for changed Flow/VDN/SOL files: original recorded SHA, deployed candidate SHA, path/module and reason. Both arms use the same candidate files. Unchanged sources, checkpoint, configuration and conditions remain exact. Allowed differences are only the W request/history/receipts, mode-specific support/complement, and resulting outputs. Reject arbitrary exclusions, broad module exemptions, missing/extra manifest entries and edited R identities.

W-window is the same-deployment control, not an optional third baseline run. Existing R supplies the SOL-local reference. Verify exact sampler/model/H3 entry tensors, unchanged early-boundary metadata and first-block pre-attention inputs. Later QKV tensors need not match after intentional operator divergence. Structural transparency does not alone prove behavioral transparency; unexpected changes invalidate interpretation.

### Required receipts and interpretations

At every block record Q/K counts, generated-video span, support mode, complement execution, provider route and adapter/gate fingerprints. Target: 56349 packed rows, 53248 video rows, eleven local groups plus two anchors. Each arm expects 550 native local plus 150 global/anchor subcalls, zero SOL local sparse calls. Full support uses canonical K/V once per row, without duplicated global/anchor keys.

| Result after invariant gates | Interpretation / next action |
|---|---|
| Window fails, full removes defect without new corruption | Hybrid versus full support is sufficient for this state. Amend design for a narrow VDN-owned policy only after full-suffix/media/NFE validation. Does not prove a faulty linear kernel or a general need for full attention. |
| Window removes defect | Local SOL path contributes in the matched no-Untwist case. Do not change VDN support; localize approximation versus kernel/domain arithmetic before any Sol production correction. |
| Both fail | Full-support substitution is not sufficient for this state. Common state/model/conditioning/adapter path remains unresolved; no production patch or transport retry. |
| Outputs differ but improvement ambiguous, or new corruption | Inconclusive/unsuccessful. Latent difference is not quality proof; preserve evidence and amend design. |
| Invariant failure, OOM, missing receipt, extra H3 call or failed cleanup | Invalid arm; repair only the demonstrated diagnostic issue. No media attribution. |

W is the next decisive **branch-selection** experiment. It is not a promise that every outcome identifies a production fix; available evidence cannot justify that promise.

## 9. Implementation sequence

1. Re-fetch main/PR/review/CI state. Start a separate mirror from the effective #36+#41 diagnostic tree. Keep #33 independent, #37 disabled/separate, #39/#40 closed. Checkpoint before invasive review/testing.
2. Read/hash the bundle; verify installed source, effective overlays, model-sampling/format classes, head count and unresolved node construction. Compare source bytes, not commit labels.
3. Implement/test strict immutable request and exact source-delta validation; add separate W nodes/report without changing R acceptance.
4. Implement VDN support/complement selection and SOL native-provider/history publication. Test disabled behavior, canonical KV order, gates/projection/adapter identity and one preprocessing pass.
5. Implement/test bounded Euler callback termination, raw extraction, errors/cancellation, nesting and repeated invocation. Checkpoint complete diagnostic before hostile review/CUDA instructions.
6. Run the two fresh-process arms against the same explicit saved filename and deployment. No low/probe/upscale, no `[latest]` ambiguity. Obtain both reports and raw/pre-guidance diagnostic clips.
7. Apply the outcome table. A favorable result permits an evidence-based design amendment, not automatic production promotion. Keep any eventual correction separate from #41 and diagnostic implementation.

## 10. Validation and rollback matrix

| Gate | Expected result / limitation |
|---|---|
| Source | Exact manifests; sampling conversions, callback order, head selection, latent conversion and wrappers verified; no version-string inference |
| CPU contracts | Strict parser; disabled no-op; immutable state; nested/concurrent context isolation; finally restoration; source-delta tamper rejection; one executor call; no quality claim |
| Geometry | Current 52-frame case; rectangular grid; equal rows/different H/W; short full-coverage clips; anchors; no duplicate KV; each Q written once; complement off only in full arm |
| Provider/history | Mode before preflight; native receipts; exact fusion retained; no generic Sage substitution; no fabricated acceptance or forced forecast count |
| CUDA input | Captured noise argument and sampler/model/H3 video/audio hashes, dtype/device/stride, sigma and conditions exact; mismatch invalidates attribution |
| CUDA accounting | Each arm 1L/1A/0F, zero upscale, expected 700 subcalls, weighted activity zero; no speculative/replayed H3 |
| Cleanup | Sentinel distinct from errors; scopes restored; no accumulated hooks, config mutation or invalid retained state on subsequent execution |
| First-high media | Same decoder/frame times; original artifact, identity/detail, temporal coherence and new corruption assessed across clips; tensor distance alone insufficient |
| Eventual production | Full progressive workflow: learned clean, first raw/pre-guidance, last pre-guidance and final video/audio. Default acceptance stays 9L/7A/2F, one upscale |
| History transition | Native-first→SOL-later may invalidate Spectrum anchor and add an actual. Never hide that transition. Promotion blocked without validated compatible accounting or explicit approved constraint change |
| Audio | Entry exact; carried conversion/#33 policy preserved; speech/ambience/continuity tested empirically; changed video may affect output audio |
| Regressions | Protected-video diagnostic rejected; disabled behavior unchanged. Future fix: first chunk, protected continuation, normal non-progressive, source-input/bicubic, rectangular and second-seed cases |
| Performance | After causality, measure hot stage/sampler time and peak memory. Full support can erase progressive speed benefits; no broad speculative benchmark campaign |

Rollback removes W activation/nodes and restores legacy behavior. Preserve R bundle and reports despite older R documentation permitting deletion. Reject eventual production candidates on new corruption, audio/prefix regression, changed geometry, extra upscale, hidden NFE, inconsistent history, persistent mutation or unbounded cost. Require a narrow owner scope and explicit legacy behavior switch for any promoted policy.

## 11. Risks and invariants

Full support changes the effective operator; VDN gates/adapters were trained with hybrid attention. Improvement may be compensation rather than a coding defect. Failure does not prove state construction guilty. Preserve gate/adapters for attribution; removing VDN changes too many variables.

Use existing fused native SDPA with bounded Q groups, never materialize a full attention matrix. Increased KV work and OOM are performance/validity risks, not causal evidence. Video changes can alter audio later in the transformer.

Preserve caller prefix/noise, carried audio, fresh target initialization, target geometry, learned_3d, exact-prefix fallback, #36 guidance and current core model sampling. No second upscale, hidden production NFE, weighted Mixed-Grid, stochastic/path transport, blanket resets or broad disable sweep. Keep source-delta permissions narrow and R's comparator intact. Callback termination cannot counterfeit a full trajectory.

## 12. Live state and recheck requirements

Fetched 2026-09-15; existing PR branches unchanged.

| Flow line | Head | Base/state; observed CI |
|---|---|---|
| main | `970396db839ae7ab431b9718859f6d48a2e5019b` | design base |
| #33 | `9c400d46e990222cf723c426039b4c543192e185` | main; open, one commit; 34784515457 success |
| #35 | `a99ed7a7ca20ed3af5282c73cad4bb662e45f804` | #33; open, one commit; 34794442324 success |
| #36 | `cf260160df234f7a6c61787725b22eddd522c438` | #35; open, one commit; 34804117251 success |
| #37 | `174e0c2e884f268387897e06d4f11c0adebafacd` | #36; separate diagnostic; 34800106786 success |
| #41 | `4ae2e35f77ed961151ab5695bef9e1dbe277cc54` | #36 directly; open draft, one commit; 34923658902 success |
| #39/#40 | `205b5b1380593de0c9501d24f9b406a9f2f42cad` / `ff3fc7481f54ef26dde96a1f4b534ba088984028` | closed, not merged |

No inline threads returned for #33/#35/#36/#37. Both #41 inline threads (selector docs and provenance integrity) are resolved; current source contains the repairs. #41 description has stale head bookkeeping; API metadata above wins. Existing CI is structural evidence, not media approval.

| Owner | Installed HEAD / dirty | Audited matching source |
|---|---|---|
| Flow | `1e975a8c18211391e23544f7c0222e9ef73b450a`, clean | 28/28 recorded files match #41 `4ae2e35f77ed961151ab5695bef9e1dbe277cc54` |
| Core | `e3c077bd8a31eb6a9c0efa64a65b341337032dbd`, dirty | model.py, model_base.py, samplers.py, patcher_extension.py, ops.py, modules/attention.py match fetched `9a600f8118784eaa2e811ca5e4d2e59d2905b72f` |
| SOL | `b437d0898b245de981bd578b2a4f93d98567c0be`, clean | 10/10 match #9 `fdd52bbd07bd88dfe3d31c823de0d14c27c74aff` |
| VDN | `25abc53996edd63d6dd88bf20492c8dec7259839`, dirty | 23/23 match #8 `71549c02bc8e73c4c968a43b3955a3998715660d`, stacked on #14 `d5f158a1c1750d79b37cb6ef22b0f14e6a8cbad6` |
| Spectrum | `b1cf2870fd843530c7b7ef56150162c9a6125d21`, clean | 52 match #110 `78a9a5b36c185a55af59a44c073bc7f8e534bc97`; minimax_h3.py matches fetched main `120d72e2f48b781235b34149e39bbdf0f1317d82` |
| DiffAid | `ba9d9efbcf7e64c755e068cb76547d8cc85481eb`, clean | 3/3 same commit |
| KJ | `ef7156d3590dc6cdcc9a13799c9f2a025612f03c`, clean | 20/20 same commit |
| DoRA | `7b367d653ce300243f6340b90e8ddc9e7025875d`, clean | runtime_bypass.py and 3 others match `c91612008a4979b412b8bc49d5ffd7e6bef7be10`; nodes.py unmatched |
| Upscaler | `db76324d6bbf231bebcb9d794e133ef4d4d9ee87`, dirty | 8/8 same fetched commit |
| Continuum | `58306ecd43c4cc2ffa0dc41657d1dd8d8bd365e7`, clean | participating model_patch.py matches fetched `a5b8943844594545301b20d01af5d9e3fa38ae29` |

Several installed commits are not available as remote objects. Byte matches support the audit; unreachable labels do not. This is an effective mixture, not permission to replace workstation trees with PR heads. The JSON appendix lists exact path/SHA/ref comparisons: **156 of 157 match**; DoRA nodes.py is unresolved.

Missing transitive provenance includes explicit installed hashes/classes for `comfy/model_sampling.py`, `comfy/latent_formats.py`, `comfy/model_patcher.py`, and `comfy/k_diffusion/sampling.py`. Their fetched source was read; installed parity must be checked before implementing termination or stronger conversion claims. Confirm actual final-head count, model sampling overrides and adapter/checkpoint identities, plus unresolved DoRA construction. No live workstation connection was available; historical receipts are not current machine state.

Stable decisions: R scope, fixed state/audio/geometry in W, native-window/full-support distinction, VDN/SOL ownership and Spectrum history, empirical promotion gates. Re-fetch commit-specific topology, source paths, wrapper order, capabilities and CI before implementation. Document evidence and rationale for justified deviations; never silently reinterpret the design.

## 13. Artifact registry and completed checks

Inspection copies: `/workspace/scratch/c8ffab7070ed/evidence/ComfyUI-Sol-H3/`. This scratch path is disposable; original evidence is persistent. Resolve exact filenames and verify hashes when restoring.

| File | SHA-256 | Disposition |
|---|---|---|
| `replay_report_00001.json` | `6417bb19937d18d8bac0e546262392bac4360395530ba52a5eaecd9e3af4521c` | Preserve authoritative R attribution |
| `executioncontractreport_00012.json` | `708edef769750e50067d041f5fe5694c5bab4418b4806d9a2daa7a4e2e63dcb2` | Preserve runtime/source observations |
| `metrics_00463_.json` | `36bce71786d2173551d395b3cd88b2a4b9b58914ed39cc342bd5cf13aaa47dc7` | Preserve R accounting |
| `Pasted text(20260915-032843).txt` | `3a84dba2d2db418d618b66c6f728092bb77b7ea1959b6568b692627b8d3e7dc7` | Preserve runtime/failure context |
| `metrics_00442_.json` | `1b1ba08689b997f7ea98b1e6420e6213338222fbedd5a7a6a2853c78691932e3` | Preserve #37 accounting |
| `Pasted text(20260914-030325).txt` | `69f4d3ef9e81089c3fd20841133c77e92c91df7bb32d1ac95d449a6b9c7f2948` | Preserve recovered dense-local receipt |
| `learned_transfer_clean_00001.mp4` | `53b20942c0f2bbad4e01dddc1a9f4163f5d278224095a7d9b02d9190a0c2b85c` | Preserve older boundary media; not R-labeled |
| `firsthighpreguidance_00001.mp4` | `efcd0edb0d9db46dd127d4ba3eec8f5ec8aea247969cd54fa1247e5e37ca2ed5` | Preserve older boundary media; not R-labeled |

Critical workstation bundle:

- Report-confirmed manifest: `/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json`.
- Source-derived expected tensor payload: `/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.pt`. Confirm `tensors_file`; its on-disk filename was not independently observed.
- Both file hashes are **unknown here**. Hash manifest/payload and check `tensors_file_sha256` plus tensor contracts before W. Preserve untouched; further read-only inspection required, reproduction not currently needed. Discard only after a validated successor and explicit disposition decision.

No unique experimental tensors or uncommitted production code were created. Two one-second PNGs in `/tmp` are disposable extractions. Read-only checkouts are reproducible from refs; the source audit is committed as the JSON appendix.

Checks actually performed:

- Parsed four R files and #37 log/metrics; verified reported attribution/hash/topology/policy fields.
- Compared 157 recorded source entries: 156 exact matches, one unresolved. This is not complete transitive runtime proof.
- AST-executed audited `window_bounds`/`full_coverage`: 52 frames, eleven local groups; S=484 gives 24200 local+968 anchor queries; S=1024 gives 51200+2048; full coverage false. This validates partition counts, not CUDA gathers/kernels.
- AST-executed SOL warmup helpers: prefix 1 suppresses default high warmup; prefix 2 enables it; #37 runtime corroborates routing.
- Computed scalar AV mapping; read conversion/callback paths; inspected two media frames; fetched existing CI/reviews.
- No local production test suite, CUDA/model run, full temporal media validation or benchmark. Documentation/structural evidence cannot establish a production correction.

## 14. Dead ends and forbidden regressions

- Middle forecast is not necessary for the recorded defect.
- Guidance cannot originate a pre-guidance defect.
- Valid R does not require retained low/probe/reload state for the first-high result.
- Native local-window is not full-support H3 attention; do not relabel #37.
- Algebraic closure is not distributional or media validity; #39/#40 remain rejected.
- Clean decoded anchor does not prove appropriate joint noisy AV state.
- No new evidence supports global resets, VAE stitching changes, Untwist reinstatement, weighted Mixed-Grid, or a same-seed native run as an identical-state oracle.

Do not repeat R/U/#37/#39/#40 without new contradictory evidence. W targets the remaining explicit operator boundary with a different controlled question.
