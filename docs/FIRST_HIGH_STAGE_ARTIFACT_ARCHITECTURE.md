# First high-stage artifact: execution-contract architecture

Status: design-only, 2026-09-14. No production fix is authorized by a root-cause claim in this document. Source identifies concrete contract differences and a bounded way to distinguish them; the original artifact's causal owner remains unproved. Implement the observation and controlled-comparison stages first. Production changes below are conditional on their specified evidence gates.

## 1. Scope and immutable baseline

Investigate `H3ProgressiveTargetInputHandoff`, source scale 0.70, fixed requested coordinate 0.35, `learned_3d`, `direction+temporal`, conservative exact-prefix fallback. Preserve fresh target-grid stochastic initialization. No weighted Mixed-Grid activation, state interpolation, VAE experiments, extra production H3 evaluations, or second upscaler call.

The controlled Euler run has 9 logical calls: low 5L/4A/1F, probe 1L/1A/0F, high 3L/2A/1F. Handoff selects index 5, coordinate 0.3749999936, sigma 0.8780487776, source `[1,24,52,44,44]`, target `[1,24,52,64,64]`; audio `[1,32,2,292]`. Requested 0.35 is not the executed 0.375. Do not confuse either with sigma or a patch's schedule-index progress.

Exact protected-video continuation takes the existing target-grid fallback and does not traverse this learned handoff. It is a separate regression case, not a substitute reproduction.

## 2. Live source and topology ledger

All refs below were fetched during this investigation. Existing PRs were not edited, merged, or rebased. Design work is on `mirror/first-high-contract-design-20260914`, based on main; initial checkpoint is `8d6225b0400b6c7c7f397fb5a4a7997acb41457d`.

| Flow line | Exact commit | State / observed CI |
|---|---|---|
| main | `970396db839ae7ab431b9718859f6d48a2e5019b` | fetched HEAD |
| [#33](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/33) | `9c400d46e990222cf723c426039b4c543192e185` | open draft, one commit over main; CI 34784515457 success |
| [#35](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/35) | `a99ed7a7ca20ed3af5282c73cad4bb662e45f804` | open draft over #33; CI 34794442324 success |
| [#36](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/36) | `cf260160df234f7a6c61787725b22eddd522c438` | open draft over #35; CI 34804117251 success |
| [#37](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/37) | `174e0c2e884f268387897e06d4f11c0adebafacd` | open diagnostic draft over #36; CI 34800106786 success |
| [#39](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/39) | `205b5b1380593de0c9501d24f9b406a9f2f42cad` | closed, rejected |
| [#40](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/40) | `ff3fc7481f54ef26dde96a1f4b534ba088984028` | closed, rejected despite CI 34805448823 success |

Fetched PR discussions include the #40 rejection review. Automated CodeRabbit review was skipped on these PRs; green CI is not media approval. Preserve #33 independently. #35/#36 are diagnostic contracts, not permission to merge their monkey-patch machinery. Do not reopen #39/#40. Re-fetch reviews and CI before implementation.

Source snapshots inspected:

| Repository | Snapshot / relevant files |
|---|---|
| Comfy-Org/ComfyUI | v0.35.0 `40c4fcdf513a4523e39d54a9d391908af8df8171`; `comfy/samplers.py`, `model_sampling.py`, `model_base.py`, `model_patcher.py`, `patcher_extension.py`, `ldm/minimax/model.py`, `weight_adapter/bypass.py` |
| ComfyUI live HEAD | `eecbfb404637bcf5f29954ef12767c1d20ede924`; ref fetched, not substituted for installed code or fully audited |
| xmarre/ComfyUI-Sol-H3 | main `f82ff2693be37dbad3438a30eb389d77136c0276`; `sol_h3/interop.py`, `runtime.py`, `sparse.py`; also #9 `fdd52bbd07bd88dfe3d31c823de0d14c27c74aff` interop |
| xmarre/ComfyUI-VDN-H3-Plus | main `76b31323f9e09019b435237dcd8bad1e05476ce1`; `vdn_h3/hybrid.py`, `runtime.py`, `window.py`, `apply.py`; #8 `71549c02bc8e73c4c968a43b3955a3998715660d` layout/runtime differences |
| xmarre/ComfyUI-Spectrum-MiniMax-H3 | main `120d72e2f48b781235b34149e39bbdf0f1317d82`; `comfyui_spectrum_h3/sampling.py`, `runtime.py`, `refinement_compat.py` |
| xmarre/ComfyUI-DiffAid-Patches | `ba9d9efbcf7e64c755e068cb76547d8cc85481eb`; `nodes.py`, `spectrum_h3_compat.py` |
| xmarre/ComfyUI-Untwisting-RoPE | `63c8d8df5e08dfdad4b397480d966b90f9b5a6c6`; `nodes.py`, `flux_untwist/utils.py`, `patches.py` |
| kijai/ComfyUI-KJNodes | `d3cfe21625e5170126ce06fbfcfe1d88108688c3`; `nodes/model_optimization_nodes.py` |
| xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader | `c91612008a4979b412b8bc49d5ffd7e6bef7be10`; `runtime_bypass.py` |
| xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus | `620165a311de9b28a36260219fb5cd370a304e3c`; `nodes/minimax_h3_handoff_provider.py`, `minimax_h3_latent_upscaler_3d.py` |

Open companion state was also fetched: VDN #8 remains stacked over #14 (`d5f158a1c1750d79b37cb6ef22b0f14e6a8cbad6`), CI 34555399447 success; Sol #9 CI 34582139977 success; Spectrum #110 head `78a9a5b36c185a55af59a44c073bc7f8e534bc97`, CI 34581917615 success. Their weighted/research functionality must remain inactive. No claim that these heads equal the workstation's installed files is justified.

### Installed provenance: material unresolved gap

The actual log reports ComfyUI 0.35.0, Torch 2.10.0+cu130, FLOW_AV, video/audio shifts 12/3, and imports under `/home/toor/ComfyUI`. It identifies Untwist and KJ callable paths. Sol identifies Sana revision `2936c47637380842aaa4a4488fac5006cc542b70` and kernel contract `sana-sol-engine-sol-attn-64-rect-sm120-v3`.

The log is **not a commit manifest**. VDN emits audio-aware layout/control fields present in #8 but absent from inspected main; Sol emits weighted-route counters absent from inspected main's summary (all remain zero). Thus "released stack" does not establish main-only installed provenance. Inspect effective installed overlays, not just repository HEAD. VDN #8 also differs from main in compiler-guard placement and stream-lifetime handling. Do not apply main's older guard assumptions to that runtime.

Mandatory before intervention: capture imported `__file__`, resolved symlink path, git HEAD and dirty status, SHA-256 of participating source files, active overlay list/order, runtime callable module/qualname and code/source digest, model/checkpoint identity, provider config, compiler flags, and patcher injection/forward-hook identities **inside the running Comfy process**. A shell import can resolve a different module. No workstation access or installed source archive was available in this investigation. Version strings alone cannot close this gate.

## 3. Evidence and limits

Read the actual 00441, 00442 and 00444 JSON files and the full relevant 00444 model-call log. Inspected frames at 1 second from both 00444 MP4s: bright colored blotches are present in first and last pre-guidance output. This verifies that failure mode at those frames, not complete temporal quality or the original baseline artifact.

00441 confirms 9L/7A/2F; 00442 confirms 9L/8A/1F and high 3A/0F; 00444 confirms one endpoint intervention, no extra H3/upscaler evaluations, exact carried audio, target closure max error `1.1920929e-7`, direct-form max error `1.4305115e-6`, lifted/source endpoint RMS ratio `0.8618965`. Algebraic closure is not model-distribution validity.

The clean transfer, original artifact's first appearance, #36 media improvement, and #34 VAE result are reported controlled observations in the task/PR evidence. Their original paired media were not independently re-inspected here. Preserve their stated scope and require the original clean/broken checkpoints for final acceptance. The attached research PDFs cannot establish installed wrapper state or causality; no paper-based root-cause claim is made.

## 4. Reconstructed execution and ownership

Primary Flow source: [`runtime.py` at main](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/970396db839ae7ab431b9718859f6d48a2e5019b/h3_flow_regenerate/runtime.py), plus the explicit #33/#35/#36 overlays. #33 does not replace this handoff algorithm.

| Boundary | Representation / owner | Rebuilt versus retained |
|---|---|---|
| Core `CFGGuider.sample` | caller packed AV, masks, original conditions | post-hook condition entries copied; model-option containers cloned; same model/patcher retained |
| Flow `_run_progressive` | one enclosing invocation, low/probe/high children | saves shallow post-hook conditioning template; same guider and model objects |
| Low return / `_raw_sampler_state` | returned latent-format output is converted back, then multiplied by `1-sigma` | reverses terminal CONST scaling; not an x0 prediction |
| Exact probe | same low sampler state, independent sampler lifetime | `_noise_argument` reconstructs noise; one exact H3 result converted to internal clean AV |
| `build_handoff_state` | video clean goes once through learned provider | new target video noise; cloned carried sampler audio |
| High initialization | target internal packed state | original target latent/mask; protected caller noise merged; core initializes sampler |
| `_reset_guider_conds` | post-hook target conditioning template | outer entries copied; target-input high returns to original target keyframes; refs retain native independent resolution |
| Core `inner_sample` | sets `latent_shapes`, processes conditions | creates extra model-options clone and local `sample_sigmas`; `extra_conds` recreates packed layout/payload |
| Model function / APPLY_MODEL | current sigma, payload, conditional branch | Untwist/DiffAid wrappers and runtime injection ownership participate before transformer |
| H3 `forward` / DIFFUSION_MODEL | packed audio converted to its own schedule | VDN per-call layout, Sol request, Spectrum runtime; then actual `_forward` |
| Return | native velocity -> core denoised x0 -> guider result | Flow executor result, optional suffix bridge, guidance, sampler update |

### Correct initialization algebra

At the unprotected high boundary, video state is `Xv = (1-sigma)*Cv + sigma*Nt`, where Cv is the learned clean target and Nt is fresh deterministic target-grid noise using Flow's seed offset. Audio Xa is copied from the low **sampler state**.

`_noise_argument` computes `(X-(1-sigma)*Linternal)/(sigma*noise_scale)`. Core CONST `noise_scaling` computes `sigma*noise_scale*Nargument+(1-sigma)*Linternal`, recovering X. Core `calculate_input` is identity, `calculate_denoised` is `X-sigma*velocity`, and `inverse_noise_scaling` divides by `1-sigma`. FLOW_AV carries audio scaled onto the video schedule; MiniMaxH3 `process_latent_in/out` and `forward` own the conversion. Do not independently rescale audio a second time. This source algebra is coherent; installed override parity and actual entry tensors must still be measured.

### Conditioning

Core `encode_model_conds` copies `model_conds` before replacing entries. MiniMaxH3 `extra_conds` reconstructs text conditioning, payload, masks and `PackedLayout`. Flow low resizes target keyframes through `_resize_target_keyframes`; high target-input uses the pristine target template. Image references are not required to shrink with generated video. In 00444, the reference remains 64x64 and there are no keyframes. High segments are text 1493, ref-image 1024, audio 584, video 53248; total 56349. A retained 64x64 reference is not stale low geometry.

Shallow copies still retain nested tensors/objects, hooks and refs. Core copies dicts recursively and list containers shallowly; arbitrary provider objects are not cloned. Unknown condition keys could override generated `latent_shapes` or other parameters because `encode_model_conds` preserves existing params. Audit actual keys and aliases; do not blindly deepcopy weights, hook groups, provider state or all conditioning.

### Provider lifecycle audit

| Owner | Source finding | Remaining risk / appropriate observation |
|---|---|---|
| Core H3 | payload layout built per sampler; `_forward` checks current shape signature; RoPE generated from positions | same-dimension changes in reference contents are not fully described by shape signature; compare semantic segments/positions, not shape alone |
| VDN | `layout_from_payload` checks geometry; `make_layout_wrapper` sets/resets ContextVar per forward | runtime log confirms high S=1024, F=52; does not prove every gathered row/window is correct |
| VDN buffers | activation scratch keys include video/text rows, dtype/device; KV scratch is reusable capacity; execution lease falls back to transient pool on contention | retained storage is legitimate if fully overwritten; observe plans and write/read extents; no blanket pool reset |
| Sol | `SamplingWrapper` creates Request each outer invocation; per-layout arithmetic gates; actual QKV evaluated, not a cached low-grid output | first-high warmup suppression is policy; all-selected arithmetic gate does not validate sparse approximation quality |
| Spectrum | `start_run` resets forecasters/controllers; high log run_id=3, first actual history=0 and high topology | actual-first does not prove arbitrary wrapper/hook purity; backend and patch transitions remain forecast-history boundaries |
| DiffAid | `_normalized_sigma_with_refinement` honors `h3_refinement.sigma_reference=1.0` | verify publication before Spectrum preflight and actual patch application; target block config retained |
| Untwist | model function reconstructs reference ranges and per-call config; post-RoPE K preprocessing composes with inherited override | uses local `sample_sigmas` schedule progress, does not consume full-trajectory refinement clock |
| KJ | loader closure forwards to selected Sage function; no geometry cache in that closure | actual resolved Sage implementation/fallback and compiler state need runtime provenance |
| DoRA / VDN adapters | DoRA uses core BypassForwardHook with idempotent injection/reverse removal; VDN uses post-forward registrations | shared native modules persist; count hook identities after low unload, probe reload, upscale, high reload; exact fused Sol path must preserve native module hooks |
| Learned provider | copies input; normalizes/denormalizes once, returns exact target geometry and original dtype/device | provider offloads its own model; verify imported implementation and hook/device lifecycle after return |

## 5. Concrete contract differences and ranked hypotheses

### A. Untwist clock discontinuity — highest-priority observation

[`Untwist model wrapper`](https://github.com/xmarre/ComfyUI-Untwisting-RoPE/blob/63c8d8df5e08dfdad4b397480d966b90f9b5a6c6/nodes.py) calls `progress_from_schedule_index` using each child's `sample_sigmas`. For the eight nonzero Euler coordinates, full-run progress at original indices 5,6,7 is `5/7,6/7,1`; the three-call high child maps them to `0,1/2,1`. The exact one-point probe falls back to scalar sigma normalization. The low child also has its own denominator. These are analytically derived source results; an attempted direct utility execution was blocked by missing Torch, not passed as a test.

The log confirms Untwist active at first high and deactivated at the final high call. That matches local scheduling but does not prove which clock semantics the user intended or that this creates the broad artifact. DiffAid's refinement sigma clock is a different quantity; do not replace schedule-index progress with sigma by analogy.

### B. Sol warmup policy coupling — demonstrated, causality unresolved

[`dense_evaluation_warmup`](https://github.com/xmarre/ComfyUI-Sol-H3/blob/f82ff2693be37dbad3438a30eb389d77136c0276/sol_h3/interop.py) suppresses the default one dense evaluation only when Flow high has `min_actual_prefix_steps == 1`. The same predicate remains in fetched #9. Runtime confirms first low dense / first high sol. This is not proof of leaked Sol history.

**PR #37 confound:** changing prefix to 2 also makes that predicate false, restoring dense first-high warmup in these source versions. 00442 metrics establish prefix 2 and all-actual high, but the corresponding backend log is not available here. If that installed predicate matches and receipts confirm first-high dense, the remaining artifact is evidence against missing first-high dense warmup as a sufficient explanation. Do not repeat that experiment without checking this first. #37 still proves the middle forecast is not necessary in the tested configuration; it is not a perfectly isolated single-variable attribution across all companions.

Changing attention route can invalidate Spectrum history and add an actual at the following logical call. Never promise preserved 9L/7A/2F merely because the intervention itself adds no transformer invocation.

### C. Reload / adapter / retained-state ownership

Three sampler lifetimes and an intervening learned model call share base H3 modules. The actual loaded stack includes runtime adapter hooks and retained VDN resources. Source has scoped guards and shape-aware pools, weakening a generic stale-cache diagnosis. Hook multiplicity, forward identity, effective numerical path and buffer overwrite coverage are still unobserved. Rank this above speculative new latent constructions, below the observable clock/policy differences.

### D. Conditioning or wrapper transformation

Current layout rebuild and runtime topology weaken simple low-layout reuse. Nested mutations, reference content, per-call patch selection, and output conversion remain possible. `first_high_model_raw` in #35 means the result of Flow's underlying **predict-noise executor**, already processed through core/guider/companions. It is not the unconverted H3 transformer velocity. A clean/raw versus pre-guidance equality measurement is still needed before narrowing to H3 itself.

## 6. Implementation-ready observation architecture

Add optional `h3_flow_regenerate/execution_contract_diagnostics.py` and targeted tests. Integrate through existing wrapper callbacks in `runtime.py` / `comfy_compat.py`; retain #35 output names/order and #36 anchor invariant. Do not use process-global monkey-patch swaps for the final implementation.

A recorder is owned by one outer Flow invocation, with child stage id and call id. It must be context-local, reset in `finally`, bounded to low-last/probe/first-high/last-high. Disabled path allocates nothing. Keep scalar metadata on the hot path; collect GPU reductions together and transfer after completion. Byte snapshots must be detached clones, never views into reused scratch. Explicit byte ceiling and optional tensor capture; an over-budget optional capture is marked incomplete, not silently sampled as equality proof.

Capture these distinct boundaries:

1. exact probe internal clean, low sampler state, learned clean target, fresh target noise identity;
2. reconstructed high packed state before core and actual post-mask sampler input;
3. processed conditions/payload just before APPLY_MODEL; model-option/transformer-option alias graph;
4. H3 DIFFUSION_MODEL input after core audio conversion; current sigma and transformed timestep;
5. completed H3 velocity output, core denoised output, #35 executor result, pre-guidance and final guidance output.

For every tensor record shape/stride/dtype/device, finite count, stream role and coordinate domain. Store bounded per-channel RMS/mean and adjacent spatial/temporal correlation for X and Nt; aggregate RMS alone cannot test stochastic structure. Exact comparison uses full tensor data/digest, with explicit dtype conversions. Across processes, ids are alias labels only; use callable source digests and relative ownership graphs for matching.

Metadata schema must include: stage schedule and original schedule digest; actual sigma and patch progress; full ordered wrappers per API boundary; replacement chain per block; native module forward and post-hook identities/counts; VDN segments/window indices/anchors/gather lengths and pool generation; Sol request generation/evaluation index/route/receipts; Spectrum run/history/topology/decision reason; DiffAid normalized sigma/active blocks; Untwist progress/scales/reference ranges/preprocessor multiplicity; inherited KJ provider; latent format, shifts, masks, refs/keyframes/tags/augmentation seed.

Record wrapper order at runtime rather than reconstructing order from node placement. Observer wrappers must call their executor exactly once; do not insert unknown replacements into Spectrum's audited block chain. Prefer callbacks inside existing audited wrappers. If observability changes history eligibility, reject the recording as nontransparent.

Provenance collection belongs here, inside the active process. Do not import a second custom-node copy to inspect it. Persist the manifest and diagnostic report next to metrics; never include full prompts/reference contents in public reports without need. Use hashes for equality.

## 7. Minimum controlled experiments

Run on #36's effective baseline, with #37/#39/#40 disabled and #33 preserved. First reproduce the original artifact and verify the recorder is transparent. Same checkpoint/model, adapters, prompt, reference, seed, sigma schedule, masks, transfer, decoder settings and output ordering throughout.

| Test | Hypothesis and exact intervention | H3 NFE / unchanged contract | Outcomes and limits | Removal |
|---|---|---|---|---|
| O: observation | contract differs across stages; enable recorder only | exactly 9L/7A/2F, one upscale | same outputs/accounting required; locates discrepancy but does not prove causality | disable recorder |
| C: condition rebuild oracle | processed high conditions differ from pristine target reconstruction; rebuild only condition metadata through core, no model call | zero added H3/upscale; preserve post-hook semantics and RNG | mismatch identifies exact field/owner; equality does not clear hooks/numerics; text preprocessing must not be rerun against a differently patched model | discard oracle objects |
| U: clock trial | Untwist restarts schedule progress; supply full original schedule to Untwist only, leave sampler `sample_sigmas` unchanged | no direct extra H3; Spectrum may change actual/forecast count on changed patch regime, which must be reported | if first-high raw/pre improve, clock implicated; final-only changes do not localize first-high cause; no improvement rejects this correction for this case | remove marker, restore local clock |
| R: same-state high replay | high result depends on low/probe/reload side effects; cold reconstruction executes exactly the captured high suffix with identical X/L/mask/conditions and equivalent explicit stage policy | separate diagnostic job, 3L/2A/1F expected; 2 H3 actuals, zero upscale; never inserted into production run | same bad output weakens hidden lifecycle cause; clean cold output implicates retained state if provenance/policies match; any contract mismatch invalidates attribution | delete replay bundle, no production hooks |
| B: fresh VDN scratch | only if R/telemetry implicates retained buffers: swap to an owner-local transient pool for high, preserving adapters/layout/backend | no direct extra H3/upscale; normal scheduling must match | improvement implicates storage lifetime/overwrite; does not justify resetting weights or all caches | restore pool in finally |
| A: attention-route test | only if #37 receipts do not already cover it: first high actual local SOL calls use existing native VDN provider; preserve other routes and exact fusion | no duplicate call; later Spectrum may become actual, so total can be 9L/8A/1F | first-high output comparison remains valid before schedule divergence; later media is non-parity; route improvement alone does not identify kernel versus approximation | remove diagnostic route selector |

Do O and C first. Choose U or R according to observed discrepancy; do not run the table as a blind subsystem-disabling sweep. No repeated #37 test or VAE tiling test.

### Valid native control

A native high-resolution run with the same integer seed follows a different trajectory and different RNG shapes. Its state at sigma 0.878 is not the progressive X. It is useful for outcome quality, not a causal model-contract diff.

The primary control is an **identical-state high replay**: capture X, original target Linternal, mask, caller noise, full original post-hook conditions, high suffix, seed and all patch configuration. In a fresh process rebuild the same model stack, reconstruct noise through core's exact initialization inverse, and run the suffix without low/probe/upscale. Explicitly retain high continuation policy and the selected patch clock; removing the Flow marker would change Sol warmup and DiffAid normalization. Flow guidance consumes the saved anchor-relative trajectory when comparing final media. For first-high pre-guidance only, save the output before guidance. Tensor equality of entry state/conditioning and receipt equivalence is required before comparing outputs.

Do not pickle a live guider, monkey-patched module graph or arbitrary provider object. Serialize tensors plus a versioned declarative manifest and rebuild through installed node constructors. Reject mismatched source/checkpoint/config or missing artifacts. A separate replay job has an explicit diagnostic NFE cost; there is no hidden extra call in the production run.

## 8. Conditional production changes

### If U validates full-trajectory patch timing

Add a versioned Flow-owned `h3_flow_sampling_context` transformer option, distinct from `h3_refinement`: original nonzero sigma tuple in core sigma units, stage, original stage-start index, schedule digest and invocation generation. Publish in `_run_progressive` before children and restore the previous value in `finally`. Do not overwrite `sample_sigmas`, which the actual sampler and Spectrum own. Low/probe/high share trajectory timing but keep independent solver/forecast histories.

Untwist `nodes.py` should resolve its progress from this validated original schedule through a helper in `flux_untwist/utils.py`, retaining legacy local behavior when the marker is absent. Duplicate/same-coordinate solver evaluations share progress; the probe uses its original coordinate, not a new trajectory endpoint. Validate descending finite coordinates, units, stage bounds and source. Unknown API: legacy behavior with explicit unsupported telemetry; recognized malformed marker in a requested diagnostic: fail before execution. Do not silently guess sigma units. Update Untwist's Spectrum runtime publication to use the **same** progress/active values as execution. Do not change DiffAid's existing sigma policy or Spectrum's feature coordinate.

Full-trajectory timing is a behavior change and requires low-stage as well as high-stage media validation. A high-only diagnostic correction is not sufficient to promote a full-path change. If correct clock transitions alter NFE, preserve truthful history invalidation and keep promotion blocked under the fixed-count requirement until a compatible design is validated. Do not hide route changes to retain a forecast.

### If R/B validates lifecycle defect

Fix the proven owner only: Flow owns child context/conditioning, Core owns model loading, VDN owns scratch/gather plans and stream storage, Sol owns request/calibration, Spectrum owns prediction history, adapter providers own injection handles. Correct the specific key, write-before-read, stream lifetime or missing restoration. Add failing regression at that boundary. Do not deep-copy H3 weights, globally clear CUDA caches, regenerate noise, or reset all providers at the handoff.

### If A validates attention policy defect

Keep routing in Sol; do not make Flow impersonate a different `h3_refinement` prefix. Separate trajectory-start and geometry-change policy explicitly and make `HistoryPolicy` describe the actual route. Preserve VDN's native KV domain, anchors, gate/projection and preprocessor exactly once. A changed route invalidates incompatible Spectrum anchors. Production promotion still requires unchanged accepted NFE or an explicitly justified, user-reviewed relaxation; this design does not authorize that relaxation.

### Learned-anchor guidance extraction

Preserve #36's `target_ref(c)=target_anchor+resize(source_ref(c)-source_anchor)` and equality at h. A future production extraction belongs in `handoff.py`/`runtime.py`/`guidance.py` with explicit request-owned anchor data, not duplicated upscaling or permanent diagnostic monkey patches. Release anchors after the high stage; never mutate committed low trajectory entries. Keep it independently reviewable from the pre-guidance fix.

## 9. Rejected approaches

| Hypothesis | Evidence | Rejection scope |
|---|---|---|
| Final VAE spatial stitching creates original broad artifact | reported #34 changed overlap and introduced a new tile seam while original persisted; latent checkpoint localization precedes final assembly | reject original stitching explanation; not every possible decoder defect |
| Learned clean transfer creates it | reported clean `learned_transfer_clean`, broken first pre-guidance | reject transfer output as observed origin; does not prove every noisy target state is valid |
| Flow guidance creates original pre-guidance artifact | reported first pre-guidance already broken; #36 fixes separate reference mismatch | retain anchor guidance correction; do not blame it for earlier artifact |
| Single middle high forecast is necessary | 00442 verified high 3A/0F; reported artifact persists | forecast not necessary; Sol warmup confound limits narrower attribution |
| #39 full displacement re-anchor | reported new global corruption despite structural invariants | reject `Ctarget+U(Xsource-Csource)`; no coefficient tuning |
| #40 endpoint transport | actual 00444 metrics and sampled media, review rejection | reject bicubic source stochastic/path endpoint transfer; covariance explanation remains a hypothesis, not measured proof |
| Simply reusing low VDN layout | source rebuild plus high runtime geometry | weakened, not globally disproved for all internal plans/installed versions |

## 10. Validation and promotion

| Gate | Required evidence / expected result |
|---|---|
| Source contract | pin actual imported source plus effective overlays; verify wrapper order, core conversions and provider ABI; fail if unresolved |
| No-CUDA unit | recorder disabled no-op, single executor call, bounded capture, nested-call isolation, cancellation restoration, immutable template, clock indices including probe/duplicate phases; current test runtime requires Torch CPU |
| Core integration | use real pinned `extra_conds`, `encode_model_conds`, `noise_scaling`, latent AV conversions; target state roundtrip within declared dtype tolerance, caller protected values exact |
| Geometry | 44x44 -> 64x64 -> repeated 64x64; rectangular low/high grids; same total rows with different H/W; changed refs at same generated shape; missing/changed dtype/device |
| Companion source | VDN current layout/window/gather domain; Sol route preflight equals receipts; Spectrum new high history empty, no cross-route forecast; Untwist preprocessing once; DiffAid applied regime equals telemetry |
| Hook lifetime | original forward/hook ownership preserved across low/probe/upscale/high, exception, cancellation and second workflow execution; no accumulated hooks |
| CUDA | O exactly 9L/7A/2F and one upscale, no hidden H3 calls; finite tensors, exact carried audio and protected outputs; inactive weighted counters zero |
| Media | paired original baseline and candidate: learned clean, first model result/pre-guidance, last pre-guidance, final; same frame times/decoder settings; compare broad artifact, identity, texture, action, temporal continuity, speech/ambience and protected prefix |
| Regression media | original failing seed plus a second seed; first chunk and protected continuation; relevant patches individually absent only after cause localization; same-size native run unchanged |
| Performance | hot sampler/stage timings and memory; distinguish recorder overhead from production; no optional broad benchmark campaign |

Do not promote from CPU CI, algebra, an all-selected attention gate or a single hidden-space distance. Require original artifact removal without new colored blotches/temporal corruption, #36 guidance fidelity, exact audio/prefix preservation, correct topology and installed-source manifest. If no test isolates a cause, stop at the diagnostic result and revise this design with evidence; do not implement a guessed production fix.

## 11. Recovery, artifacts and handoff

All changes in this investigation are documentation. Rollback of future diagnostics must be one feature disable / removal of diagnostic module registration. Context/options/pool ownership restoration must execute on exception; do not mutate the user's WSL branches or reset installed repositories. Preserve #33 guided audio defaults/emergency disable, Continuum First Frame/Qwen/keyframe/cache behavior, VDN ownership/API, Sol backend ownership and Spectrum truthful accounting.

Relevant evidence is not committed to git. Local root for this investigation is `/workspace/scratch/91e78cbb9cc8/evidence/ComfyUI-Sol-H3/`. Resolve original files by these exact names if scratch expires:

| File | SHA-256 | Required action |
|---|---|---|
| `metrics_00441_.json` | `84a2b8372daef81c46792d1c6e15439d0db6b875c17332d899889eadafbd5c3e` | inspect baseline accounting |
| `metrics_00442_.json` | `1b1ba08689b997f7ea98b1e6420e6213338222fbedd5a7a6a2853c78691932e3` | inspect prefix-2 result; obtain matching route log before revisiting warmup |
| `metrics_00444_.json` | `beede291390dcf2ac9472a64b2211e45692226941a61fe500db05b81af1047c2` | preserve negative state-transport evidence |
| `Pasted text(20260914-043151).txt` | `aa3cac2376eb8077d6595cbe786b2faa8f35af9d1cb350eba8aab2ef7087e7e8` | inspect effective stack, high receipts and shapes |
| `firsthighpreguidance_00004.mp4` | `ab3fbd2caddb11f4b47100ff86406615699b6eaeb1f1fa48bb0db6faf2ec4f56` | preserve failed #40 media |
| `last_high_pre_guidance_00004.mp4` | `fec5b3d15d7ace3c69ce7fae0a2de2d338c13c27bc7acad051b699419a3431c8` | preserve failed #40 media |

`/tmp/first.png` and `/tmp/last.png` are disposable 1-second frame extractions from the listed videos, reproducible with FFmpeg. No unique diagnostic tensors or code exist outside git. Read-only source checkouts are reproducible from the ledger. Still required: installed source manifest, original baseline paired media, 00442 backend log, and a newly captured identical-state replay bundle. Their absence must remain explicit.

Next implementer: use the exact final commit of this document as the design specification; re-fetch live main/PR/review/CI state; preserve #33 and #35/#36 independently, keep #37 diagnostic and #39/#40 closed. Work on a new implementation mirror and checkpoint to GitHub before risky/destructive work, hostile review, long tests and after substantial progress. Verify actual installed provenance first. Implement O/C and the evidence-selected bounded experiment, then only the validated owner fix. Keep final PR topology clean and intentional; do not replace existing PRs or silently activate weighted research. Record justified deviations when new source/runtime evidence invalidates assumptions. Media and truthful NFE accounting are mandatory promotion gates.
