# Exact-prefix, semantics-preserving MiniMax-H3 continuation

Status: authoritative **gated implementation design**, audited 2026-09-20. No production fix is implemented or empirically established by this document. Flow owns the sampler/transformer representation boundary. Existing diagnostic PRs remain intact. Production defaults must remain unchanged until the matched-runtime acceptance gate passes.

## 1. Decision and evidence correction

Retain the heterogeneous target-prefix/source-suffix representation, the complete learned-linear complement, and H3's layerwise context evolution as the semantic baseline. Do not implement a frozen-prefix or suffix-only-query transformer as an assumed repair. Introduce an explicit, call-owned distinction between sampler write protection and transformer hidden-state evolution. The first bounded candidate changes **only low/probe target-audio spatial position ownership to the source carrier**; it retains exact target-prefix embeddings and all heterogeneous VDN/Sol semantics. This is a source-grounded candidate, not a demonstrated cause or guaranteed repair. Sections 7–9 define its implementation and a decision gate; a failed gate does not authorize promotion or another arbitrary toggle.

Two premises in the preceding interpretation are contradicted or weakened by primary evidence:

1. **00546 does not demonstrate a continuous lizard shot without the boundary shift.** Its output cuts from the lizard at frame 174 to the iguana at frame 175, precisely the physical chunk boundary. The near-zero PT212 post-boundary median largely measures the new iguana shot. It cannot certify preservation of lizard camera motion. Same-shot continuity remains unproven, even though the previously observed multi-frame displacement pattern is absent.
2. **The changed fish motion cannot be directly explained by the continuation-only intervention.** The fish is in the initial chunk. `partitioned_outer_wrapper` delegates an unmasked initial chunk to the released progressive path before the continuation diagnostic runs. Both videos visibly differ there despite matching sampling seed and first-chunk geometry. The source does not support attributing this difference to removal of continuation prefix queries.

Consequently, no architecture can honestly be called a verified solution preserving both the claimed seam benefit and continuation semantics from these artifacts alone. This design supplies a precise conservative candidate, ownership rules, source findings, and the smallest new evidence required to accept or reject it. It does not substitute a new model architecture for missing causal evidence.

## 2. Live repository baseline and topology

All four default branches are `main`. GitHub metadata, relevant PR heads/bases, changed-file lists, and merged discussion/review timelines were fetched. The following are observed refs, not inferred from PR prose.

| Repository / PR | Base | Head SHA | Head branch | PR-local commits |
|---|---|---|---|---:|
| Flow #49 | main | d1b32f530b4cc2298eba68f08c2da44a094c7cfe | mirror/exact-prefix-progressive-v2-20260917 | 71 |
| Flow #54 | #49 | b4abdc9bbf5139e328788fa698c18377f7ae54d0 | mirror/partitioned-diagnostic-controls-20260920 | 1 |
| Flow #56 | #54 | 1250959f1b9231335bf6f669fba8553e9366113f | mirror/flow54-cross-grid-temporal-ab-20260920 | 1 |
| Flow #57 | #56 | 6caf039e78513c2d78ee6623590468ac21da8adb | mirror/flow56-raw-token-measure-ab-20260920 | 1 |
| Flow #58 | #57 | 8118a293d9e98fc43fcb2f46e3609c8fbddbde87 | diagnostic/source-carrier-audio-stage-00545 | 1 |
| VDN #19 | main | 675cd6e7b43c9d8d63b7a91834fb6d775388a2ee | mirror/exact-prefix-partitioned-vdn-20260917 | 26 |
| VDN #22 | #19 | 25bca63ab7ede45b3dc8af461f589475c2f9aa20 | mirror/arithmetic-validation-vdn-20260918 | 1 |
| VDN #23 | #22 | b584070b37b3ec5ea2be97e96c420b982b5aab5f | mirror/retained-scratch-pre-evict-20260919 | 1 |
| VDN #26 | #23 | b853289ae5c4dd35430acc4a13daa8e383e2f26a | mirror/partitioned-linear-bypass-diagnostic-20260920 | 1 |
| VDN #27 | #26 | b686f96d554609b5e00ad94bb22c44665c217688 | mirror/vdn26-cross-grid-temporal-ab-20260920 | 1 |
| VDN #28 | #27 | 1c01d53504c7585b72843e0bd3d3c3b6f72d6628 | mirror/vdn27-raw-token-measure-ab-20260920 | 1 |
| Sol #15 | main | 93b3e03f2b7b579aaf55fa0f87f55083b259e25c | mirror/exact-prefix-partitioned-attention-20260917 | 19 |
| Continuum #24 | main | 6167dcbae8ab239ea660fed52d03ddb63aa515fb | mirror/production-phase-aware-audio-20260913 | 1 |
| Continuum #28 | #24 | 688c754b19664b3812d462af6d5a3414bfc1b06b | mirror/pr24-decoded-trajectory-diag-20260919 | 1 |
| Continuum #29 | #28 | 922002f3debaa424bd8fe3847560276a156e8568 | mirror/continuum28-audio-overlap-context-diagnostic-20260920 | 1 |

Default SHA: Flow `b659b311fa548c7e3275db0e6f0a05c037dac730`; VDN `d244ae4cc635123826e243aac335b4719c865670`; Sol `0208ddbaaa8a94d70cbf29003680a24d6404fc66`; Continuum `fb435c526643ca71afcac827df82ee6b2e811c3a`.

Heads #58/#28/#15/#29 include their listed ancestors: inspect those complete trees to reconstruct effective behavior, not isolated PR diffs. Patcher may apply additional Core or extension overlays; these refs alone do not certify installed bytes. Do not enable unrelated Flow #52/#55 VAE work or Continuum #25 in this stack.

Fetched check-runs report successful Flow Python 3.10–3.13/source-contract jobs; successful VDN pinned-oracle/current-main/migration jobs; successful Sol test/native-interop/Windows-provenance jobs; successful Continuum #28/#29 Python and publisher-toolchain jobs. VDN #23 also has cancelled historical check-runs. Continuum #24 returned no check-runs; its success status was CodeRabbit's review-skipped status, not a test result. No substantive review submissions were present in the fetched merged review timelines. None of these checks establishes motion quality.

Only Continuum's inspected head tree contained `AGENTS.md`. Its constraints prohibit global Core class replacement, additional prompt restrictions, silent State resizing, and private Spectrum calls; preserve native `position_ids` identity where existing wrappers rely on it. Diagnostics must not turn prompt content into execution blockers. Re-read all repository instructions before implementation, including newly added files.

Design branch: `design/exact-prefix-semantics-00546-20260920`, originally based on Flow #58. Checkpoint `4428d6b95367df9749405a88f6110b563d705b7f` preserves the initial evidence correction. No diagnostic head was changed.

## 3. Runtime artifacts, pairing, and observed behavior

### 3.1 Artifact identity

| Run | Artifact | SHA-256 |
|---|---|---|
| 00545 | Pasted text(20260920-160548).txt | 2a28806856485b6aafcfb493261ce51a6c59e56b3712176675aac5948abb2fba |
| 00545 | metrics_00545_.json | 95fc1ca5e9246fb8a0856dc6a3b381aa0f1716dd1bdcd501798293db0686fedb |
| 00545 | MiniMax_H3_00012-audio(1).mp4 | b11b41caeae7d45e71746c72d67e2ae6e4480b9e42bee689de2c9bac5abd95b4 |
| 00546 | Pasted text(20260920-174204).txt | c0f9fdd0ab79f39001ef2d3ad95418172665e05f29d9c4fdc26c79ff50e6fac3 |
| 00546 | metrics_00546_.json | 19fbf9af2a3ac421a4fa4e84104a652b28aca6e48306edd7a451657e03455dcd |
| 00546 | MiniMax_H3_00013-audio.mp4 | cf562082f2b46c0dbe12c03657ae5ffb8924ca34cd16d2a1d96d0e7747a4595e |

Metric event windows are 15:44:16–15:59:32 UTC for 00545 and 17:28:41–17:37:54 UTC for 00546 on 2026-09-20. Embedded MP4 execution graphs carry the corresponding diagnostic settings: raw-token measure versus normal/source-carrier-uniform. They contain no independent authoritative export timestamp. UI `videopreview` filenames refer to the preceding output and must not be mistaken for current-file provenance.

Pairing is corroborated by decoded AAC audio: the MP4 500-ms windows around frame 175 give approximately +11.695 dB and +3.163 dB, respectively, close to log PT213 values +11.6698 and +3.1274 dB. Small differences are expected after AAC encoding. Video geometry, prompt graph, and boundary behavior also agree. This is strong artifact association, not proof of full byte-identical runtime conditions.

Both videos are 1216×896, 336 frames, 24 fps, 14 seconds, with stereo 32-kHz AAC. Natural retained output is 345 frames. `finalize_assembled_timeline` calls `enforce_total_frames`, which trims/pads the end; it does not globally retime the sequence. Therefore physical boundary frame 175 remains output frame 175, at **7.291667 s**. It is not frame 168 at the authored 7-s routing signal.

### 3.2 Frame inspection

Frames use zero-based indexing. Exact transition endpoints below are scene-identification observations, not a generic perceptual benchmark.

| Observation | 00545 control video | 00546 source-carrier video |
|---|---|---|
| Fish, frames 24/48 (1/2 s) | Predominantly lateral presentation | Turns toward/away from camera; markedly different body orientation |
| Lizard begins | Frame 143, 5.958333 s | Frame 143, 5.958333 s |
| Last carried output frame | Frame 174: lizard | Frame 174: lizard |
| First newly retained frame | Frame 175: lizard | Frame 175: iguana |
| Continuation action | Head raises around 176–178; alert posture through 186; darts out 187–190 | No retained lizard continuation action after boundary |
| Iguana cut | Frame 191, 7.958333 s | Frame 175, 7.291667 s |
| Lizard-shot duration | 48 frames, 2.000 s | 32 frames, 1.333 s |

The diagnostic loses **16 frames / 0.667 s** of lizard display relative to this control, including all its newly retained continuation. The control's one-frame-early cut relative to authored 8 s is distinct from the diagnostic's immediate physical-boundary cut. Later turtle/bird shots remain present; presence alone does not establish full reference fidelity.

PT212 upper45: control pre-median dy −0.1106 and first-three post-median +4.0505 px/frame. Diagnostic pre-median −0.1172 and post-median −0.1194. However, diagnostic first pair is **dx +16.5651 / dy +2.4918**, with response 4.5205, followed by responses roughly 130–215 within the iguana shot. The full-frame first match is also weak and cross-scene anchor estimates are unstable. A same-shot displacement inference from this median is invalid. Preserve the receipts; change their interpretation, not their values.

PT213 pre/post-seam values match within each run; Audio Seam is Off. PT214 v2 gives exact context-safe carried-audio interior (5 ticks / 4,000 samples): correlation 1, gain 1, zero residual. Its 30-tick edge exclusions matter. The user's judgment that 00546 music loudness is acceptable is an acceptance constraint; zero dB is not required. This audit did not independently establish perceptual audio quality by listening.

### 3.3 Prompt/reference control comparison

PT209 matches exactly:

- source digest `f43646219f617d3f03cfc316523db63efc01e4c885179add5b4b25db9af18299`;
- raw/expanded hash `9f1959742269265fba5d82fcffb633205993eda028d76d62a5ebbafa3ec73fe5`;
- verified legacy sequence, matching geometry and skeleton.

Compiled text hashes match: initial `c49f2f66b3545ccf4458955bd148cdf409c892c2a7f00bf9f2a07c09ed0c1c0c`; continuation `74ee492782515477fc40096f4e1babe38bb6d06918e9c99cd10c3b2dd87e4bb8`. Compiler versions, interval counts, fallback statuses and physical windows match: v2/[0,175)/4 intervals/gap-hold; v3/[136,345)/5 intervals/physical-overrun-hold.

PT210 matches all seven ordered reference hashes:
`9630f436a98726f8, 1900c1d71bd85ab2, 7813bedb33f89ce5, 311d16f89788863d, ebe112fd0b9b156b, a632dfdf05fcb4be, fc20dfaafd8687b1`.
Picture offset is zero, public-to-Qwen mapping is 1→1 through 7→7, first/last-frame presentation is absent, and exact prefix is [136,175). PT211 selects the prior latent tail, not its head.

The source compiler scopes the `[7-14s]` logical body, replaces [136,175) with immutable-context text, and retains the physical clock. Derived continuation intervals are [136,175) context; [175,192) remaining lizard; [192,240) iguana; [240,288) turtle; [288,345) jay with end overrun held. These boundaries are derived from inspected compiler rules and embedded authored text; the Basic logs do not contain the complete serialized interval metadata. Capture that metadata in the validation fixture before claiming its exact object identity.

Packed-layout receipts match text rows 8286/8353, reference rows 7309, audio rows 584/696 for initial/continuation calls. Full conditioning embedding bytes, dtype and hidden dimension are not independently hashed by these receipts. Matching text is not proof of identical Qwen embedding output.

Embedded execution graphs differ at exactly these input leaves: node 333's prefix selector and VDN selector; node 309's wildcard seed (237862536758780 → 469441669431324). Wildcard source/populated text matches. Actual sampling seed node 125 is **384415445495679** in both, connected to Continuum and RandomNoise. Do not read the UI's displayed `base_seed=0` as the effective seed. The wildcard-seed change is not evidence of a text change and has not been established as the fish cause. It can affect execution/cache invalidation or other process state; conditioning/noise/source provenance is needed to distinguish these possibilities.

## 4. Effective low/probe/high dataflow

### 4.1 Flow and Core

`partitioned_outer_wrapper` detects whole-frame exact masks. Initial/no-prefix chunks use the released progressive path. For supported continuation, `run_partitioned_progressive` captures the authoritative target prefix and its target noise, constructs source-grid noise and resized sampler carrier/mask, and executes low, exact probe, then high as separate sampler invocations. Low/probe use one stage owner per invocation. High starts with an actual H3 evaluation; Spectrum histories are not carried across geometry boundaries.

Continuation geometry: video latent `[1,24,62,40,54]` on the sampler source carrier; target `[1,24,62,56,76]`. Twelve prefix latent frames carry target patches 28×38=1064/frame; fifty generated frames use source patches 20×27=540/frame. Nonvideo rows N=16,358 comprise 8,353 text, 7,309 references and 696 audio. Heterogeneous hidden sequence has N+12×1064+50×540=56,126 rows. Uniform source sequence has 49,838. High target sequence has 82,326. Attention uses 56 heads × 128 channels; model hidden width and actual checkpoint projection behavior must be confirmed locally.

`partitioned_diffusion_wrapper` builds a carrier layout from target dimensions, preserves conditioning rows, replaces only video positions with source coordinates, then creates the heterogeneous layout. At block zero it replaces carrier-prefix hidden rows with `video_patch_proj(patchify(0.999*prefix + 0.001*prefix_noise))`. Prefix gets target spatial RoPE; suffix retains source spatial RoPE and unrestarted temporal positions. `partitioned_mod_segments` expands native prefix timestep labels to target row count.

Every block consumes and updates the entire heterogeneous sequence. Prefix queries are active. No blockwise prefix freeze occurs. After the last block, Flow discards target-prefix hidden outputs and inserts zero source-prefix placeholders. Core FinalLayer then receives native carrier row geometry and native per-row modulation labels; Flow also zeros prefix video velocity. Final exact sampler restoration remains authoritative. Zero hidden placeholders alone would not prove zero velocity because FinalLayer has affine modulation/output bias.

Learned 3D transfer receives genuine source-grid clean predictions with a resized exact prefix as transient convolution context. Its prefix output is discarded. Target prefix is restored; audio sampler state is copied unchanged. High runs native target geometry and exact-mask final canonicalization. Preserve the existing distinction between internal floating-point clean reconstruction and caller-visible final exactness.

### 4.2 VDN

API 4 / `partitioned_attention_variable_grid_linear` validates Flow geometry against its native source layout. `partitioned_grouped.py` builds per-frame ranges, released temporal windows, global rows and anchor rows/columns. Runtime gathers Q/K/V from the same full hidden sequence. Global and anchor queries use dense attention; prefix local queries are forced dense because their evolved outputs feed deeper K/V. Suffix local groups use mapped Sol. All hidden rows must be covered once by global/local/anchor query ownership.

Raw pre-QK-normalization, pre-RoPE video Q/K/V are copied into managed scratch before Core's in-place RMSNorm/RoPE. Text raw K/V are preserved when text state is enabled. Inherited DiffAid/Untwist preprocessing runs once on full physical softmax Q/K/V before gathering, not once per gathered group and not on the learned-linear raw features.

`partitioned_linear_readout` computes frame-local spatial short-conv and physically mapped temporal taps, Q/K activation, beta-weighted frame statistics, frame means/alpha, bidirectional scans with text state, bridge/gather, query readout, norm/gates and output projection. Prefix participates in K/V statistics and hidden evolution, not merely attention lookup. With `anchor_frames=both`, released `skip_ends` excludes first/last frame from the learned-linear interior; retain that exact exception.

The #19 temporal mapping uses H3 area-normalized coordinates with FP32 resampling. #23 preserves managed scratch pre-eviction and release at probe→high and terminal-high. Continuation reserved memory fell 76,256→56,384 MiB / 91,136→60,256 MiB in 00545 and 73,920→54,912 / 89,504→55,360 in 00546. These are release receipts, not proof that first-run overflow is solved.

**Additional verified defect in diagnostic completeness:** #28 selects zero `prefix_log_key_measure` for global/local calls and unit learned-linear frame measure in `raw_token_measure`, but its anchor call passes `plan.prefix_log_key_measure` directly. At the supplied `anchor_frames=both`, anchors retain the density correction. Counters 250/3000 prove execution, not total coverage of the intended measure switch. Keep the 00545 negative result for the executed arm; do not claim every softmax route was raw-token. This does not authorize reopening the density campaign or imply it is the seam cause.

### 4.3 Sol, Spectrum, DiffAid and Untwist

Sol #15 `partitioned_request_attention` already accepts rectangular THD Q and K/V with equal heads/head dimension, request ownership, one K/V union, additive prefix key measure, and mapped query-position runs. The mapped descriptor represents each Q row's index in the gathered K/V union; it is not a substitute for H3 physical RoPE coordinates. Source suffix rows remain present in K/V, so suffix-only local Q can be represented without geometric approximation. No square-Q expansion or independent per-partition sparse thresholds are justified.

Flow's front door still requires square BHTD input before VDN gathering. Flow's canonical wire asserts `exact_prefix_queries_preserved=True`; VDN requires complete query coverage. Therefore existing kernel capability does not make a suffix-only transformer an ABI-compatible change.

Provider identity must remain stable under equivalent per-call preprocessing closures while preserving strict terminal-provider object identity. Rebind transforms before execution. Sol numerical-route history and Spectrum actual/forecast entitlement must not be bypassed. The two runs each report 14 actual model evaluations and 4 forecasts overall; 00546 additionally counts six source-carrier transformer-wrapper entries whereas 00545 counts five heterogeneous block-zero executions. Those counters instrument different boundaries and must not be equated.

### 4.4 Continuum and installed Core provenance

Continuum owns physical prompt compilation, seven-reference presentation, carried AV tail, decode grouping and retained timeline. It must not repair transformer behavior by rewriting prompt timing, changing references, shortening overlap, or smoothing output. Its Video Seam setting in both embedded workflows is **Auto**, not Off. PT212 is measured before optional video patches. Exported video alone therefore cannot certify absence of assembly correction; capture pre-patch media and patch-action receipts. Existing scene-cut guards reject generic corrections across detected scene cuts.

Logs report Core `v0.36.0-26-g673afefe`, `patcher/stack`, PyTorch 2.10.0+cu130, comfy-kitchen 0.2.35, HIGH_VRAM/cudaMallocAsync, SM120. The abbreviated Core SHA could not be resolved in either public Comfy-Org/ComfyUI or xmarre/ComfyUI. The compared public model file is pinned to Core `c194dd00cd42aa18d9dbf27d977bf6b85d9ea565`, blob `c4b7a68aa60388b6a8721ecce9909a2fbc1b5921`; it is not certified as the installed file. Export actual model, sampler-mask, model-base, compiler/quantization and wrapper sources with hashes through the Patcher/runtime tooling before implementation acceptance. Preserve installed RMSNorm/RoPE mutation ordering, FinalLayer/PDD behavior, audio-mask velocity multiplication and compiler allocation contracts.

## 5. Why suffix-only Q is not the selected repair

Let hidden state at block l be `(G_l,P_l,S_l)`: nonvideo, protected-prefix context and generated suffix. H3 computes rowwise norm/AdaLN, joint attention plus VDN complement, gated residual, then gated MLP. Current prefix recurrence is `P_(l+1) = F_l(G_l,P_l,S_l)_P`.

For a fixed K/V snapshot, computing attention only for suffix Q preserves those suffix attention rows. Removing prefix Q does not directly change that same block's suffix output. It changes the next block only if it changes `P_(l+1)`. Three choices follow:

- Evolve P exactly using separate context queries: mathematically coherent and preserves existing semantics, but retains the same recurrence and is not a causal seam repair.
- Reuse P_0 at every depth: changes layerwise feature distributions, prefix K/V, beta/alpha statistics and text/reference/audio feedback. Sampler protection does not justify this freeze.
- Construct a separate prefix encoder or cache previous-generation K/V: requires a new recurrence, timestep/text/reference/noise conditioning contract, cache invalidation and memory budget. Prior chunk K/V generally belongs to different conditioning and sigma. No evidence or training result here validates this replacement.

Nonvideo rows also evolve and include generated audio; removing all non-suffix queries would break that path. “Read-only exact prefix” must mean caller-owned latent storage is immutable, not that model hidden context is constant. The production ownership table therefore keeps P queries/evolution and discards only prefix output writeback.

## 6. Independent architectural axes

| Axis | Retained production-candidate rule |
|---|---|
| A sampler/output carrier | Source low/probe carrier; target high/output; exact external restoration |
| B prefix hidden representation | Target exact latent + native conditioning augmentation, embedded once per call; evolve through blocks |
| C prefix grid | Actual target grid; no replacement with resized source hidden rows |
| D suffix grid | Actual source grid in low/probe, target after learned transfer |
| E queries | Keep global, prefix, suffix and anchor ownership; no silent query removal |
| F K/V | Full declared union, with VDN windows/anchors; prefix available at every block |
| G VDN routing | API 4 grouped transport retained |
| H learned-linear | Full variable-grid complement, retained physical mapping and released anchor exclusions |
| I Sol positions | Q-in-K/V index runs plus independently correct physical RoPE; no square inflation |
| J output/AdaLN | Native modulation for every row; discard prefix after final block, mask velocity, restore exact output |
| K conditioning/positions | Identical physical text/reference presentation; explicitly separate audio spatial-domain policy |

The omitted axis in the broad 00546 ablation is audio position geometry. `partitioned_carrier_layout` constructs `PackedLayout` with target H/W. Core `_audio_grid` pins stereo width coordinates to the target grid extrema, and Flow retains those nonvideo positions. Uniform source execution builds native source `PackedLayout` instead. For this geometry, stereo width endpoints are approximately source `[-2.5903201,33.2132593]`, target `[-2.6394359,33.6584130]`. Thus #58 changes AV positional geometry even when audio samples, masks, shapes and timestep controls match.

This difference is a verified source consequence, **not a proven coordinate bug**: a heterogeneous video has no unique native single grid from which audio endpoints must follow. Source-domain audio is selected as the first candidate because it isolates a concrete difference associated with the acceptable-audio arm while retaining the high-resolution continuation information. Its effect on visual continuity is a hypothesis through joint AV attention, not a geometric prediction. If it fails, do not claim that read-only-prefix Q is therefore the answer.

## 7. Proposed candidate contract and exact changes

### 7.1 Flow: separate diagnostic branch after #58

Files: `partitioned_stage.py` (`PartitionedStageRuntime`, `partitioned_carrier_layout`), `partitioned_transformer.py` (`partitioned_diffusion_wrapper`), `partitioned_diagnostics.py`, `partitioned_scheduler.py`, `partitioned_node.py`, `partitioned_runtime_gate.py`; targeted tests alongside existing partitioned-stage/context/diagnostic/provider tests.

Add a diagnostic-only selector `audio_position_domain = legacy_target | source_carrier`. Missing field is `legacy_target`. Require `prefix_transformer_context=exact_target_partitioned` and `vdn_linear_diagnostic=normal` for the new candidate. Do not alter old diagnostic values or their behavior.

Old: target-dimension `PackedLayout` supplies all nonvideo positions, including target-audio stereo spatial endpoints. New candidate: retain its exact row order and all existing text/reference/keyframe geometry; replace **only the target `audio` segment's spatial position columns** with the source-native audio grid values. Preserve audio temporal coordinates and stereo channel-major order. Do not change `ref_audio`, `cond_audio`, any reference-video audio coordinates, token tags, timesteps, audio samples, conditioning embeddings or masks. If those extra audio domains exist, keep their existing policy and report them; the new candidate does not silently migrate them. Current evidence has seven image references and no such extra domains.

Build the replacement from Core `_frame_grid(source_h,source_w)` and `_audio_grid` or its verified native equivalent. Do not use half-pixel resize, recentering, token offsets as physical coordinates, or substitute a new aspect-ratio formula. Work on a call-owned layout/position tensor before it is exposed to wrappers; preserve the identity of the published tensor thereafter. Never mutate shared incoming layout objects. For legacy mode, take the existing arithmetic path without an additional copy/recomputation.

No Q/K/V row counts or query subsets change. Prefix target representation, prefix noise, source suffix, VDN statistics/measure, Sol single-union routing and final writeback are unchanged. Expected continuation counts remain N=16,358 and total=56,126. Generated audio and video remain coupled through existing attention; there is no gain normalization or seam smoothing.

The geometry wire/API remains Flow v1 / VDN API 4 / Sol single-union v1. Do not insert new fields into the existing canonical digest without updating all validators. Instead add a separate versioned model-local position-policy record containing mode, source/target geometry, exact audio segment/range, position digest and stage owner. Append a versioned immutable policy tuple to `partitioned_layout.signature` in candidate mode, leaving its required leading Flow identity marker intact. Keep the native carrier `layout.signature` exactly `(text_len,T,source_h,source_w,audio_t)` so Core does not discard the supplied layout and rebuild it. Sol #15 `_partitioned_flow_replacement_identity` already includes `repr(partitioned_layout.signature)` in its numerical identity; this is the concrete existing transport to Spectrum backend history. Add a regression test proving the candidate signature is recognized and distinguishes the policy. The key-bias tensor geometry remains unchanged. If the installed history implementation contradicts this inspected source, add a narrowly versioned capability and matching test before permitting selection. Never silently change numerical policy inside a live stage.

`PartitionedStageRuntime` owns the immutable selected policy and its validated positions for one stage lifetime. Low/probe transitions rebuild call data; high removes the policy and uses native target geometry. Reuse existing teardown/finally logic. No retained GPU context/K/V bank is introduced.

Receipts: selected policy; native-source and target audio endpoint values; audio temporal-coordinate equality; exact hashes of unchanged non-audio positions; prefix/suffix RoPE hashes; sampler masks unchanged; model-timestep override counts; final exact AV restoration; actual-versus-wrapper execution counts separately. Count successful candidate block execution, not entry to a wrapper that Spectrum can bypass. Fail closed for malformed selected internal contracts or a candidate selector that never executes. Do not add prompt restrictions or reject unknown model names solely by name. Legacy preflight fallback behavior stays unchanged; selected candidate execution cannot silently fall back after starting.

The four-tick `model_timestep_only` audio setting remains diagnostic. Its inner audio-mask override must not replace the outer exact sampler mask or outer Core velocity mask. Preserve the exact source→target audio state copy. A source-coordinate candidate passing audio acceptance does not by itself promote model-timestep-only guidance for all workflows.

### 7.2 VDN: explicit diagnostic coverage correction, separate from candidate arithmetic

Files: `vdn_h3/partitioned_runtime.py::_partitioned_vdn_forward` and its targeted tests. Base: #28, in a new stacked PR if this correction is implemented.

Use the already-resolved `prefix_log_key_measure` consistently for anchor/global/local calls in raw-token diagnostic mode. Normal mode is unchanged. Add route-specific requested/applied measure receipts and a test with `anchor_frames=both` that inspects all three route classes. The old 00545 artifact remains the old mixed-coverage arm; do not relabel it or rerun it as a prerequisite for the source-audio candidate.

No change to variable-grid recurrence, spatial/temporal features, raw scratch, projection lifetime, index cache, release hooks, audio ownership, or native VDN. No ABI bump is needed for this diagnostic-only branch correction. Its evidence must remain separate from any claim about the production seam.

### 7.3 Sol: history identity only if required

No Sol production change is expected at the inspected #15 head: its replacement identity already includes the partitioned layout signature. Extend source-contract/history tests to prove the Flow signature transport. Only if installed source differs, use a child of #15 affecting `partitioned_history.py` and the existing history tests; `partitioned_request.py` changes require a demonstrated receipt gap. Keep kernel/mapped-neighbor ABI, Q/K/V layouts and routing arithmetic unchanged. Include a stable policy identity in numerical-history comparison so a policy change forces the existing conservative actual-evaluation boundary; equivalent per-call closure rebuilding must not trigger one. Do not weaken provider identity or force arbitrary forecasts. No Sol kernel change is justified by this design.

### 7.4 Continuum: validation receipts only

Base #29 if source changes are needed. Files `v2/physical_runtime.py` and `v3/trajectory_diagnostics.py`/assembly receipt integration. Preserve existing compiler versions, intervals, text and reference bytes, AV ownership and decode behavior.

Expose the already-constructed interval metadata and full conditioning shapes/dtypes in a bounded validation manifest. Add a non-mutating PT212 interpretation field indicating whether the compared window is a continuous shot or scene-change/unknown, using existing scene analysis where possible. Do not suppress raw displacement values, invent a universal correlation threshold, or make this classifier a production gate. Human frame review remains authoritative. Record whether optional Video Seam patched exported frames and retain a pre-patch boundary window. No changes to State schema, prompt UI, seven-reference routing, overlap length, decoder scaling, phase correction or seam algorithms.

### 7.5 Promotion and rollback

All additions are opt-in and missing serialized fields preserve current behavior. Unknown versioned internal contracts fail before sampling; failure after sampling starts restores scoped state and reports failure without switching algorithms mid-chunk. Ordinary no-prefix Flow, native VDN, native Sol, noncandidate workflows and high-stage geometry retain their existing behavior.

After matched evidence passes, create a distinct promotion commit/PR with the accepted policy and compatibility rationale. Until then production defaults remain unchanged. Rollback is disabling the new Patcher overlay/selector; never restore protected latents from model output, leave candidate history cached, or rewrite old diagnostic heads. All handoff-ready changes must reside on the correct PR stack.

## 8. Minimal implementation and validation sequence

1. Re-fetch live refs/reviews/CI/instructions and retain GitHub checkpoints. Establish exact installed source hashes through Patcher, including Core, quantization/compiler, Qwen encoder and the other model wrappers. No manual checkout operations are required from the user.
2. Implement only the position-policy candidate and bounded receipts on a new Flow implementation mirror based on #58. Implement a Sol history extension only if required by the existing public contract. Keep any VDN diagnostic-coverage correction separate.
3. Run focused CPU/source tests: legacy layout equality; only target-audio spatial columns differ; unchanged temporal/reference/video coordinates; no shared-layout mutation; invalid policy rejection; per-row AdaLN and final carrier geometry; exact AV restoration; exception cleanup; stable provider identity with changing closures; position-policy history transition. Extend the existing cross-repository contract checker rather than inventing a new benchmark. Existing backend tests must still prove mapped rectangular row accounting and unchanged VDN linear arithmetic.
4. Prepare one matched SM120 fixture with exact first-chunk output, carry/noise, compiled conditioning and references captured once. Use existing Continuum State/Session or a bounded test-owned capture at the continuation boundary; do not change production cache semantics. Persist it on CPU, hash it, and release GPU copies at existing boundaries. This is needed because the two supplied first chunks differ. Record resolved sampler seed, RNG states where relevant, complete runtime overlay inventory and stage schedules.
5. Run **two new continuation evaluations** against that identical fixture: retained heterogeneous `normal/legacy_target` control, then `normal/source_carrier` candidate. Keep four-tick model-timestep guidance, seed, references, prompt, schedule, handoff and all other settings equal. This is a new one-axis comparison, not a replay of a closed ablation. Compare with archived 00545 semantic and 00546 audio evidence. If an existing saved control can be proven to match the fixture exactly, reuse it and run only the candidate.
6. Inspect raw and exported boundary frames and audio. If the candidate passes, run one full two-chunk workflow to establish no-prefix invariance and integration. If it fails, preserve evidence and stop promotion. Do not jump to frozen context or expand to a grid of toggles. The remaining decision is whether source-audio coordinates explain any part of the result; this test answers that specific question.

A same-grid component oracle is useful only if the new source tests expose a routing discrepancy: compare explicit Q/K/V/feature inputs against native grouped/linear execution. Current public partition contracts reject equal grids; do not weaken production validation just to run such an oracle. It must be an isolated test construction using the existing low-level helpers.

### Acceptance gates

| Gate | Required evidence |
|---|---|
| Caller-owned prefix | Bit-exact original video/audio prefix at final sampler output; separate receipt for internal reconstruction roundoff |
| Same-shot continuity | Lizard remains in the newly retained boundary frames; inspect 168–192 and pre-patch frames; no premature cut used to pass PT212 |
| Motion shift | Compare all first six displacement pairs, pre-motion and anchor reliability on the same scene; no renewed visible upward impulse; median alone is insufficient |
| Lizard action/timing | Retain the roughly 16-frame continuation seen in control, head raise/alert pause/exit; cut near the accepted control's frame 191/authoring 192, not frame 175; human review required |
| Fish | Full-workflow initial chunk preserves accepted turn behavior; hashes/conditioning/runtime must explain any difference before blaming continuation |
| References | Seven identities/order/conditioning dimensions unchanged; synchronized visual review of all animals |
| Learned transfer | Learned 3D provider executes; prefix output discarded/restored; first high evaluation actual |
| Route ownership | VDN linear active; full query coverage; mapped Sol requested Q=kernel Q, no square expansion; DiffAid/Untwist preprocessing once; Spectrum schedule receipts |
| Audio | User/listener acceptance relative to 00546; PT213 supporting measurement, no arbitrary 0-dB requirement; PT214 safe interior stays exact; audio state copy exact |
| Lifetime | Probe→high/terminal-high release receipts, no retained activation/index growth across repeats, no new full target-prefix KV cache |
| Controls | Identical fixture hashes; full conditional tensor provenance; pre/post-assembly patch actions recorded |

A quality gate failure blocks production promotion even if all tests and arithmetic gates pass. No CUDA quality evidence was generated during this design task.

## 9. Rejected/closed paths and their scope

| Approach | Status and scope |
|---|---|
| Full learned-linear bypass | Rejected production approach: lost intended continuation and early iguana. Useful semantics are carried by this branch; individual arithmetic details are not thereby proven correct. |
| Flow #58 source-carrier-uniform | Rejected production approach: immediate boundary cut, semantic regression, acceptable reported audio. Broad intervention changes representation, routing, measure and audio coordinates. It is not proof of same-shot seam repair or isolation of target-prefix injection. |
| Raw-token measure / 00545 | Executed global/local/linear arm failed to repair shift (250 calls/3000 prefix frames). Anchor route remained corrected; do not overstate coverage or reopen it without new causal evidence. |
| Cross-grid temporal-tap suppression | Executed 250 calls/3000 taps/2,406,000 rows; first impulse unchanged. Closed as dominant repair; retain #19 physical mapping. |
| Half-pixel mapping/recentering | Superseded by retained H3 physical-coordinate mapping, which had positive prior runtime evidence. |
| Physical short-conv rescaling | Rejected prior experiment; no revival. |
| Forced dense Flow/Sol | Failed/worsened prior displacement; sparse Sol alone is not isolated as cause. |
| Post-low source-native vs exact-context substitution | Earlier numerical no-op after transformer; did not test internal prefix injection. |
| VAE tiled decode | Closed as first origin of framing defect; separate tile artifacts and Flow #52/#55 remain separate. |
| Zero audio overlap ticks | Rejected: improved audio with dark-frame/trajectory regression. |
| Missing phase correction / seam assembly as loudness cause | Prior phase correction executed; pre/post seam levels agree. Closed as dominant explanation. |
| Independent Core carried-audio decode gain | PT214 safe interior exact; edge divergence consistent with decoder context. Does not prove all possible audio behavior globally. |
| Generic AV seam smoothing | Prohibited as a repair. Existing Auto video processing must be disclosed in evidence. |
| First-run-only overflow | Unresolved/separate; successful later release receipts do not explain it. No speculative scratch/lifetime rewrite. |
| Freeze prefix hidden state or omit all nonsuffix Q | New architectural change unsupported by H3 recurrence; not selected for implementation. |

Historical negative experiments not supplied as raw artifacts here are retained as task-provided results corroborated where available by fetched PR history; they were not independently rerun. Scope them accordingly.

## 10. Remaining uncertainties and implementer rechecks

- Same seed does not establish identical conditioning, augmentation noise, quantized kernels or persistent state. Initial fish divergence is unresolved. Leading distinguishable causes are conditioning/noise variation, runtime overlay/compiler differences, or nondeterministic/stateful execution; capture their hashes at the first actual model entry rather than speculate.
- Matching compiled text/routing rules rules out an observed compiler transport difference, not learned prompt-timing drift. H3 timestamps are conditioning, not hard per-frame attention masks.
- Source-audio spatial policy is a candidate convention for a heterogeneous grid, not a theorem. The matched test must establish whether it preserves semantics and improves audio/continuity. A negative result leaves the production fix unresolved.
- Recheck whether the actual installed Core honors supplied carrier layouts, does in-place RMSNorm/RoPE, and applies outer audio velocity masks in the inspected order. Public master is insufficient.
- Confirm actual wrapper ordering and history capabilities in Spectrum/DiffAid/Untwist; this audit traces their transport through Flow/VDN/Sol, not byte-identical installed implementations of those other repositories.
- MP4/log pairing is strongly corroborated; there is no independent output receipt binding the media SHA to the metrics. Add that manifest in future validation.
- Current log shapes establish token/row dimensions, not exact full conditioning tensor equality. Existing Flow bounded tensor signatures sample values; use full byte hashes for the one captured validation fixture.
- Any deviation justified by then-current source/evidence must be documented against these decisions. Do not silently rewrite the ownership model or call an unvalidated approximation equivalent.

## 11. PR sequence and non-git artifact handling

Keep this design on its separate Flow design/mirror branch/PR based on #58. Future Flow candidate gets a separate implementation mirror and new draft PR directly on #58; VDN coverage correction, if made, goes on a new child of #28; a necessary Sol history change goes on a new child of #15; Continuum receipts, if made, go on a new child of #29. Preserve existing chains and one-clean-commit topology on new implementation PRs. Keep development/checkpoint history separately before consolidation. Do not squash the 71/26/19-commit foundational PRs merely because newer diagnostics are single-commit PRs. Re-fetch before every consolidation and use the Patcher overlays for user validation.

The six named original files above remain external primary evidence. Preserve their bytes, embedded workflow/prompt metadata, hashes and chronological association. Derived contact sheets are review aids, not replacement evidence. `audio00545.f32` is unnecessary for this design: PT214 and MP4 decoded audio already address the carried-prefix/level question; do not retrieve or reinterpret it unless decoder analysis is reopened for a concrete reason. A future matched fixture additionally needs exact first-chunk latents/carry/noise, actual conditioning and seven media assets or their reproducible sources, Patcher overlay manifest, model/VDN/upscaler hashes, and raw pre-patch boundary media. Embedded filenames alone do not supply those tensors or model files.

Supplied papers were inspected for scope: Sol-Attn (2607.24027) supports treating sparse routing as part of numerical semantics; Sol Engine (2606.23743) emphasizes instance-specific implementation; Spectrum (2603.01623) concerns feature forecasting; VSA (2505.13389) concerns trainable sparse attention; DMD2 (2405.14867) concerns distribution matching distillation. None proves frozen-prefix equivalence, heterogeneous H3 positional policy, or the claimed same-shot seam repair. No paper is used to override observed source/runtime evidence.

## 12. Pinned source entry points

- [Flow transformer and hidden-row ownership](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/8118a293d9e98fc43fcb2f46e3609c8fbddbde87/h3_flow_regenerate/partitioned_transformer.py)
- [Flow stage geometry and positions](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/8118a293d9e98fc43fcb2f46e3609c8fbddbde87/h3_flow_regenerate/partitioned_stage.py)
- [Flow scheduler, learned transfer and restoration](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/8118a293d9e98fc43fcb2f46e3609c8fbddbde87/h3_flow_regenerate/partitioned_scheduler.py)
- [VDN grouped execution and anchor measure discrepancy](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/blob/1c01d53504c7585b72843e0bd3d3c3b6f72d6628/vdn_h3/partitioned_runtime.py)
- [VDN variable-grid linear recurrence](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/blob/1c01d53504c7585b72843e0bd3d3c3b6f72d6628/vdn_h3/partitioned_linear.py)
- [Sol rectangular single-union requests](https://github.com/xmarre/ComfyUI-Sol-H3/blob/93b3e03f2b7b579aaf55fa0f87f55083b259e25c/sol_h3/partitioned_request.py)
- [Sol replacement history identity](https://github.com/xmarre/ComfyUI-Sol-H3/blob/93b3e03f2b7b579aaf55fa0f87f55083b259e25c/sol_h3/partitioned_history.py)
- [Continuum physical compiler runtime](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/blob/922002f3debaa424bd8fe3847560276a156e8568/v2/physical_runtime.py)
- [Continuum assembly and finalization](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/blob/922002f3debaa424bd8fe3847560276a156e8568/v3/assembly.py)
- [Compared public Core H3 model](https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/comfy/ldm/minimax/model.py)
