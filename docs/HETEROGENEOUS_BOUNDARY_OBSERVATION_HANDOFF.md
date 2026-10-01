# Heterogeneous boundary observation: qualification handoff

Status: matched SM120 A/B/C plus complete partitioned learned-linear bypass
discrimination is complete. Candidate C changed the internal pre-high trajectory
but **00722 still rendered the frame shift**. 00724 then bypassed the complete
VDN learned-linear complement and **the visible frame shift still remained**.
No VDN geometry fix is promoted.

00726 successfully executed the direct same-grid target control. The user still
observed a slight frame shift, but reported that it was **substantially less
prominent**. This is rendered evidence that the low-resolution -> high-resolution
stage transition is materially involved, not evidence that it is the sole cause.
The run also exposed a separate Flow clean-state ownership bug: a source-residual
handoff was later inverted as though it had used the deterministic Gaussian-noise
contract.

00727 repeated the same-grid arm after that ownership correction. The correction
validated exactly: the actual clean postprocess tensor was used, inverse recovery
was absent, identity transfer/frame gauge remained identity, and the one-token DC
delta collapsed to zero. **A very slight rendered frame shift still remains.**
The residual defect is therefore not explained by the learned transfer, the
invalid clean inverse, the DC bridge, target-high recreation, or the complete
partitioned VDN learned-linear complement. Its measurable boundary motion is
already present in the same-grid partitioned low/probe output and largely
survives target-high.

00728 completed the same-domain dense-suffix intervention. The residual rendered
shift and audio burst remained. The next qualification changes audio ownership
as described below. No rendered production fix is promoted.

## 00737/00738: slow equal-grid control; installed admission fix not evidenced

The new user report is latency only. The attached appended log contains two
prompts: lines 321–810 correspond to 00737, and lines 811–1252 to 00738. Their
completion receipts are 14:33 (873 seconds) and 10:08 (608 seconds). A fresh
startup precedes the first prompt; the second reuses that process. Both prompts
and physical conditioning hashes differ, so this pair is not a matched
cold-versus-hot benchmark.

Evidence SHA256:

- 00737 metrics: `f5263a72ff4df5af2dfc7b8d3cf4d31236ccd4d96967e69b7c3fc6bb282520d3`
- 00738 metrics: `7cf2db3c0ef42f23ede5f9d88bea1d10ea18a5540ac5181de393166dc0a08460`
- appended log: `963e776ffb10fbbb299e6e64eaf49ff7c3492cb85986c7054b75602752b75a76`

### Effective preparation code

All twelve VDN admission receipts have the old format, for example log line
1044: `sampling admission eviction stage=low unloaded=2 allocator_before=...`.
None includes `kept_resident_h3`, `eviction_elapsed_ms` or `sampling preparation`.
VDN #36's retained CUDA path always adds those fields at this same log site.
The observed admission wrapper therefore predates #36; these runs cannot
validate its clone-preservation fix. The files do not distinguish a missing
Patcher overlay, overwritten overlay order, stale checkout or loaded module.
Do not infer the installation cause or claim that #36 saved time.

The hot first low stage again evicts four loaded models and then requests H3;
continuation low evicts two and requests H3. The old log still does not name
which models were evicted or time eviction versus Core preparation.
Both physical encodes are cache misses and load the text encoder. Continuum's
conditioning cache remains invocation-local. 00738's first Spectrum profile is
also a miss (build 11.007314 seconds, lookup 5.513360 seconds); its continuation
lookup is a hit (0.029433 seconds). These profile fields overlap existing
intervals and must not be added together as independent preparation costs.

### Workload and measured cost

Both runs use `same_grid_target_control`, normal softmax and normal VDN linear
attention. The configured progressive source is 38x50; the initial chunk uses
that source, but continuation low/probe executes directly at the **54x72 target
latent grid** (864x1152 decoded geometry). Its transformer contains 11,664
protected-prefix plus 48,600 generated-suffix video rows: 60,264 total. The
learned continuation transfer is the control's identity operation. This control
still pays target-grid low/probe cost.

| Stage wall | 00737 first | 00738 first | 00737 continuation | 00738 continuation |
|---|---:|---:|---:|---:|
| Low | 230.838 s | 92.521 s | 216.190 s | 198.940 s |
| Probe | 10.229 s | 10.904 s | 42.943 s | 39.336 s |
| High | 63.607 s | 51.616 s | 65.397 s | 55.898 s |
| Whole sampler | 306.182 s | 155.720 s | 328.109 s | 297.495 s |

00738 has **453.215 seconds** in its two samplers and **356.760 seconds** in
all timed model calls, leaving 96.455 seconds outside those call timers.
Prompt completion exceeds the sampler sum by 154.785 seconds; that remainder
includes work before and after sampling, including conditioning and decode.
The metrics do not independently isolate text-encoder time.

The first low stage spends 31.134 seconds before its first timed model call;
continuation low spends 16.982 seconds there. Their model-call sums are
53.956 / 159.309 seconds and their remaining low-stage costs are 38.565 / 39.630
seconds. These are nested attribution intervals, not additional costs to sum.
A faster model-admission path alone cannot eliminate 356.760 seconds of model
work or the full-grid control workload.

### Canonical equal-grid history correction

The source-backed history limitation recorded under 00733 is reproduced:
`backend-history diagnostic opaque` names the actual Flow partitioned block
wrapper at 00738 log line 1049. Spectrum proposes forecasts at low steps 2 and
3, but both calls execute actual transformer work. Continuation low has **five
actual calls and zero forecasts**. Decision labels are not execution receipts.

Sol #37 head `012ae39aa2137699ee8bbcef977b7965a9cb1446` extends the two classifiers only for the canonical API-1 equal-grid
control. It verifies matching spatial axes, zero key measures, current ranges,
exact query ownership and the semantic digest; its replacement classifier also
binds the contract to actual Flow latent-grid closure fields and the inherited
replacement. Full signatures, VDN binding, provider/warmup transitions and
request-owned completion receipts remain required. Equal row products on
unequal grids, stale state and malformed metadata remain opaque. This addresses
the supported control's missing classification rather than bypassing history
validation. Heterogeneous attention, transformer arithmetic, sampler schedules
and masks are unchanged.

Two regressions fail on released Sol main. The reviewed candidate passes 46
local targeted history/native-layout tests, including actual rebuilt Flow
closures with exact block fusion on/off, preserved dense-to-Sol transitions,
rectangular equal-grid controls and rejected ownership/digest/layout changes.
Flow's existing cross-repository source oracle now also consumes a canonical
control from actual Flow/VDN modules and requires both Sol classifiers to accept
it. Hosted CI validates the pinned integration separately from the scratch
runtime's optional dependencies. GPU forecast counts, timing and decoded
video/audio remain unqualified. Forecast use can change the generated trajectory.

### Patcher handoff

Retain Flow #89 -> #93, VDN #33 -> #34 -> #35 -> **#36**, and Continuum
#36 -> #37. Add **Sol-H3 #37** as the Sol overlay for this bounded history
correction. The Sol, VDN and Continuum PR numbers belong to different
repositories. Refresh the overlays and restart ComfyUI before measuring the
reviewed pair. Preserve these runs' equal-grid control, normal attention,
native temporal carrier and other inputs; leave residual measurement off.

Require the new VDN admission/preparation fields to establish which wrapper
executed. If a resident clone exists, `kept_resident_h3` should count it;
zero can be valid when no clone remains loaded. Core still owns patch/device/
memory-dependent reloads. For Sol, require an established identity followed by
accepted receipts and inspect actual/forecast execution counts. Forecast
eligibility is not a guaranteed forecast count or elapsed-time gain.

No new rendered quality verdict was supplied. Exact audio has nine verified
entries in each run and no extra model work, but those structural receipts do
not establish perceptual quality. 00738's measured motion is already present
in low/probe, while the previously rejected 00734 cross-grid/high-stage defect
remains a separate unresolved path. Do not combine the separate residual
measurement arm with this performance check or promote rejected geometry
actuators.

## 00734: cross-grid video rejected; hot preparation delay and clone lifetime fix

The user reports that the frame shift returns after restoring progressive
low-to-high sampling, and that substantial delay occurs before visible sampling
even on a hot run. The restored cross-grid video is rejected. Preserve 00733's
visually accepted same-grid arm; the reduced-grid speed result is not an accepted
production-quality replacement.

Evidence SHA256:

- metrics: `4b981f2a312b6d17a7be619123f50d9dad37ebba5f1d7501e9a9ccb28ca683a7`
- appended runtime log: `f9455e9eb5b6d1fb9c1ea72dda0014800293b787523c3841c30a67e98a94d323`

The appended log contains 00733 and the subsequent hot prompt. Only the final
`got prompt` segment belongs to 00734; its completion is **599.99 seconds**.

### Actual restoration and frame-shift localization

The 44x44 source -> 64x64 target plan executes: 12 exact target-prefix tokens,
50 fresh source tokens, 12,288 prefix rows and 24,200 fresh rows, or 36,488 video
rows. Normal suffix attention, native temporal carrier and normal learned-linear
remain active. Learned continuation transfer executes in 0.739 seconds.

Sol now establishes the low-stage history identity. Actual low/probe/high call
counts are 8/2/4 across both chunks, with four forecasts (two low and two high),
18 logical calls, six invocations and four history boundaries. Continuation low
has four actual evaluations and one forecast. This independently confirms that
the normal cross-grid identity condition permits an actual skipped model call.

The carried-video prefix SHA256 is exactly the same as 00733:
`32aed92ec74945a42b3ae6f7ccd59aa05f8824f159babed9f18801af04dd8fdf`.
Logged physical-conditioning and compiled-text hashes also match. The model
controls differ only by restored `progressive_low_to_high`; the executed low
geometry and forecast sequence necessarily change with it. This is stronger
carried-input evidence than the 00732/00733 comparison, but does not isolate the
individual transfer, prediction, guidance and solver mechanisms.

| Estimated first upper45 successor pair | X, target latent cells | Y, target latent cells |
|---|---:|---:|
| Source low/probe, target-equivalent | +0.08294 | +0.23328 |
| Learned provider clean | -0.04984 | +0.79176 |
| Exact-restored pre-high | -0.01734 | +0.01770 |
| Final post-high | +0.22366 | +0.86176 |

The exact-overlap bridge produces a nearly aligned first pair before high;
target-high recreates the large vertical displacement. The authoritative prefix
remains exact. Pre-high alignment and structural tests therefore do not establish
final quality. The learned provider's first pair is already displaced before
the bridge, but its low wall time excludes it as the large preparation delay.
No rejected rigid, prediction-gauge, destination-stencil or decoder-window
candidate is re-enabled.

The high guidance path uses its ordinary source trajectory, with no registered
target reference when the frame-gauge transaction is rejected. Its three
correction RMS ratios are 0.07590, 0.04978 and 0.02378; protected-prefix and
cross-prefix temporal exclusions remain active. That is a possible contributor,
not proof of causality. The saved run has no per-prediction before/after-Flow
trace, so native high prediction, forecasting, Flow direction guidance and the
solver update cannot be separated from this final-stage comparison.

### Hot delay before model execution

| Sampler timing | 00733 first (cold) | 00734 first (hot) | 00733 continuation | 00734 continuation |
|---|---:|---:|---:|---:|
| Low | 197.340 s | 114.433 s | 250.356 s | 140.226 s |
| Exact probe | 12.173 s | 11.568 s | 49.397 s | 26.312 s |
| High | 69.975 s | 48.998 s | 73.550 s | 60.898 s |
| Whole sampler | 280.506 s | 175.689 s | 376.697 s | 243.406 s |

Continuation sampler time decreases 133.291 seconds (35.38%). The first hot low
stage still spends **66.848 seconds before its first timed model call**. Its
five logical call timers total 40.510 seconds, leaving 73.923 seconds outside
those timers. Continuation spends 20.488 seconds before its first timed call,
with 109.002 seconds in call timers and 31.224 seconds outside them. Core
preparation is included in these stage wall times even though the progress bar
has not started. The first trajectory begins 98.050 seconds after the metrics
installation event; that interval is not a pure text-encoder timer.

The first hot Spectrum profile lookup is a cache hit taking 0.018892 seconds;
Sol's first low arithmetic-gate totals are about 0.075 seconds. These exclude
those cold-start explanations for the large hot preparation gap. Both physical
text groups still report `cache_hit=False`, with text-encoder load messages.
Continuum's physical-conditioning cache is intentionally local to each sequence
invocation, so a hot process does not imply reused encoded text. Text encoding
precedes sampler admission and is not individually timed in this log. Do not
replace it with an unqualified persistent cache that can reuse stale encoder,
reference or prompt state.

VDN admission reports four evicted models before first low and two before
continuation low, followed by H3 load messages. The original receipt does not
identify each evicted model or its duration, so it cannot allocate the entire
66.848-second gap. A separate source-backed defect does exist in this path.

### VDN #36: preserve resident H3 clones at admission

Continuum's `clone_model_for_chunk` creates a fresh `ModelPatcher` for each chunk.
VDN protected only `LoadedModel(incoming)` before asking Core to free all other
GPU models. Core's `LoadedModel.__eq__` compares patcher object identity, while
`ModelPatcher.is_clone` identifies a shared underlying model. A resident sibling
clone could therefore be fully unloaded before Core's native clone switch,
which normally detaches it with `unpatch_all=False` and reconciles weight patches.

[VDN #36](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/36), head
`2ff221258a0fe878bb3b0cf391ccc142f68fdda5`, keeps native Core-identified resident
clones on the sampling device at this boundary. Unrelated models remain evictable and Core receives
unchanged preparation arguments, including forced load/offload flags. Different
patch UUIDs still reach Core's normal weight reconciliation. The keep list is
call-local; CPU/non-retained paths, attention, sampler histories and buffer
release policies are unchanged.

Two regression cases using actual Core patcher/loaded-model classes fail on the
original code, for equal and differing patch UUIDs. The full local Core/OpenVDN
suite passes **252 tests**, with compileall, CI-selected Ruff and diff checks.
These prove structural lifetime/ownership behavior, not saved GPU seconds or
rendered equivalence.

New admission receipts separately report `kept_resident_h3`,
`eviction_elapsed_ms`, and delegated `prepare_elapsed_ms` with `success`.
The host intervals do not overlap and add no CUDA synchronization; original
delegate errors propagate. They can distinguish admission eviction from Core
preparation on the next hot run, while the earlier conditioning interval remains
outside their scope.

### Current handoff and remaining frame diagnosis

Apply **VDN #36 after #35** through Patcher and restart ComfyUI. Existing #33 ->
#34 -> #35 remain untouched. Flow #93's paired source-contract pin advances to
the new head; Flow runtime arithmetic is unchanged. Keep Continuum #37 after
its existing #36 overlay, normal suffix attention, normal learned-linear,
native temporal carrier, coherent exact audio, Audio Seam Auto and Video Seam
Analyze Only. The new VDN #36 and the existing Continuum #36 are distinct repos.

For usable output and preparation timing, restore the accepted
`spatial_stage_control=same_grid_target_control` with final MP 1.1 and measure
a hot run with residual measurement off. Require the native audio receipt,
both PCM endpoint checks and the new VDN preparation receipts. A resident clone
should be preserved when one is present; Core may still reload when its native
patch/device/memory policy requires it. No measured speedup for #36 exists yet.

For a separate frame-causality run, retain 00734's failing
`progressive_low_to_high` configuration and change only
**`frame_gauge_residual_mode=measure`**. The existing
`partitioned_high_boundary_prediction` diagnostics measure before and after
Flow guidance on each high call, including actual/forecast classification and
an authoritative-prefix disposable witness. They add no model/provider/VAE
calls and make no production correction. Their FFT/scalar synchronization cost
means this run is not a performance benchmark. Inspect the first native
prediction, each paired guidance result, the forecast call and the final solver
output before choosing a new video arithmetic change. Details are in
`HIGH_STAGE_BOUNDARY_DIAGNOSTICS.md`.

00734's coherent exact-audio validator still passes nine model entries, zero
effective overlap, no timestep override, exact final prefix and zero added work.
The applied PCM patch again preserves both endpoints; its jump falls from
0.003133159 to 0.000250964. Production audio boundary levels are -0.2766 dB
before and -0.0555 dB after the patch. These retain audio execution evidence;
no new rendered audio verdict was supplied with 00734.

Pre-review checkpoints preserve the preceding Flow and Continuum heads. The
VDN fix is preserved at
`checkpoint/00734-clone-admission-before-review-and-full-suite-20261001` and
`checkpoint/00734-clone-admission-252-passed-20261001`. #89 and Continuum #37
retain their clean one-commit topology. Production Sol and all rejected geometry
candidates remain unchanged. Cross-grid video remains unqualified.

## 00733: normal attention and PCM correction, cold-run speed still unresolved

The user identifies this as a **cold run**, reports that the video remains fine,
and says the audio is now okay "I think". This supplies rendered acceptance of
the video and tentative acceptance of the audio for the 1.1 MP same-grid arm
with normal suffix attention and Continuum #37. It does not qualify progressive
cross-grid sampling or establish general soundtrack acceptance.

Evidence SHA256:

- metrics: `1f5602306fb055abec04652177553ebfca6376b7901c78947ec9f81ac02d3ef4`
- runtime log: `1cfcfca93ea2e9b9e369c7f14217bfdc638c215a30c41972a35ef57d589626f0`

### Executed controls and audio result

The diagnostic controls differ from 00732 only by removal of
`softmax_diagnostic=dense_suffix_same_domain`. Source in
`apply_partitioned_diagnostic_controls` deliberately omits this field and
removes its transformer option for `normal`; the scheduler defaults to normal.
Positive Sol teardown evidence independently confirms restored sparse execution:
continuation low records **2,200 sparse calls / 800 dense warmup calls / five
actual evaluations**. The exact probe separately records zero sparse calls and
800 dense warmup calls. Its snapshot must not be mistaken for the complete low
stage. No dense-suffix intervention counter or verification event remains.

The stage plan is identical to 00732: configured source 44x44, effective source
and target 64x64, 12 prefix and 50 fresh temporal tokens, 63,488 partitioned video
rows, identity continuation transfer, normal learned-linear execution and native
temporal carrier. The stage/actual/sigma call sequence also matches: 18 logical /
15 actual / three forecasts, six sampler invocations and four history boundaries.
The coherent exact-audio evidence validator passes nine verified model entries,
zero effective overlap, no timestep override, exact final prefix and zero added
work. Low/high audio-mask digests remain equal and match 00732.

Continuum's positive `H3C-PT226` receipt records an applied
`convex_native_endpoints_v1` patch at frame 175, 1,920 fade samples (60 ms at
32 kHz), and **both native endpoint checks true**. The measured splice jump
falls from **0.004413930 to 0.000127817**, a 97.10% reduction. This establishes
execution of #37's endpoint correction on the user's hardware.

| Audio boundary measurement | 00732 | 00733 |
|---|---:|---:|
| Matched low/probe decode | +0.7471 dB | -1.1841 dB |
| Matched final-high decode | +3.1113 dB | -0.9117 dB |
| High suffix versus low/probe suffix | +2.3676 dB | +0.2700 dB |
| Production before Audio Auto | +2.7291 dB | -1.4227 dB |
| Production after Audio Auto | +2.6833 dB | -1.2003 dB |

Extra left decode context again changes the generated suffix by exactly zero;
production normalization is provably inactive. These windows do not reproduce
00732's generated level increase. PCM assembly cannot explain the earlier
matched-decode change because that audit precedes the seam patch. Cross-run
attribution to normal attention alone is also unsupported: the authoritative
carried-video prefix SHA256 changes from
`181188b45c0e73e7f8387d788109919d703b47e511500d99584fb272016fe0a6`
to `32aed92ec74945a42b3ae6f7ccd59aa05f8824f159babed9f18801af04dd8fdf`,
although both physical conditioning and compiled-text hashes match. The cause
of the different carried input is not established by these files. This is a
combined restoration/assembly run, not a tensor-matched causal audio A/B.

### Cold overhead and continuing full-grid cost

The prompt finishes in **00:14:41 (881 seconds)**, versus 00:11:54 for 00732.

| Sampler timing | 00732 first | 00733 first | 00732 continuation | 00733 continuation |
|---|---:|---:|---:|---:|
| Low | 111.596 s | 197.340 s | 260.276 s | 250.356 s |
| Exact probe | 12.174 s | 12.173 s | 50.768 s | 49.397 s |
| High | 67.592 s | 69.975 s | 76.007 s | 73.550 s |
| Whole sampler | 192.070 s | 280.506 s | 390.660 s | 376.697 s |

The first sampler increases by 88.436 seconds. Within its low stage, measured
model-call time totals 75.560 seconds, leaving 121.780 seconds outside those
call timers; 116.108 seconds precede the first timed call. The corresponding
00732 values are 56.776 / 54.820 / 49.117 seconds. The log confirms model loading,
admission eviction and a cold Spectrum profile lookup, but does not time these
components separately enough to assign the whole gap. The VDN-specific AIMDO
model compiler is explicitly disabled. Do not label all overhead as compilation
or add overlapping profile timers.

Continuation remains **376.697 seconds**, an observed 13.963-second / 3.57%
decrease. Low plus probe still costs **299.753 seconds (79.57%)**. The five actual
low calls alone total 205.067 seconds; another 45.288 seconds lie outside their
timers, including 32.900 seconds before the first call. Normal sparse attention
therefore executed, but neither the full-grid diagnostic cost nor all setup cost
is removed. These runs are not a controlled warm performance benchmark. The
first-chunk learned upscaler takes only 0.925 seconds; continuation uses identity
transfer.

There is a second, source-backed limitation of this diagnostic arm. Sol logs
`backend-history diagnostic opaque` for Flow's partitioned block wrapper.
Released Sol v0.1.6 `sol_h3/partitioned_history.py` requires
`target_rows > source_rows` in both its replacement classifier and layout
validator, so the equal-grid control cannot supply an accepted history identity.
Spectrum's low-stage summary confirms five actual calls and zero forecasts.
Two intermediate step-decision lines say "forecast", but actual execution and
the summary prove that no model call was skipped. Do not count those decisions
as speedup or weaken the history gate to force forecasting. Sol production and
Spectrum arithmetic remain unchanged.

### 00733 next arm (historical; executed and rejected in 00734)

Preserve 00733's accepted same-grid baseline. Through the existing Patcher stack,
change only the model diagnostic
**`spatial_stage_control=progressive_low_to_high`**. Retain 1.1 MP final target,
`softmax_diagnostic=normal`, `vdn_linear_diagnostic=normal`, native temporal
carrier, coherent exact audio, source-carrier audio positions, learned handoff,
witness off, Continuum #37, Audio Seam Auto and Video Seam Analyze Only. Do not
combine this arm with an attention/cache change or a decoded geometry actuator.

With the existing 44x44 configured source and the same temporal lengths, the
expected low/probe partition has 12,288 exact target-prefix rows plus 24,200
fresh source rows: **36,488 video rows rather than 63,488** (42.53% fewer). With
the same nonvideo conditioning, the sequence has 43,229 rather than 70,229 rows.
These row counts are not a FLOP or elapsed-time forecast.

Require the actual 44x44 -> 64x64 stage plan, positive learned transfer instead
of diagnostic identity, positive coherent exact-audio receipt, unchanged native
VDN/mask ownership and raw/assembled media review. Inspect Sol history and actual
forecast receipts again; genuine cross-grid geometry satisfies the strict size
condition, but complete identity acceptance is not guaranteed. Compare stage
times while recording cold/warm state. The current same-grid visual/audio verdict
cannot be extrapolated to this restored arm. The broad production runtime gate
is not applicable to 00733's identity transfer.

Flow #93 and Continuum #37 were checkpointed before this investigation at their
respective `checkpoint/00733-before-runtime-review-20261001` and
`checkpoint/00733-before-audio-receipt-review-20261001` refs. This observation
changes documentation only; it preserves #89's clean one-commit topology,
Continuum #37's one-commit fix, the rejected video candidates and production Sol.

## 00732: accepted visual boundary, residual audio and diagnostic cost

The user reports no visible boundary issue after increasing final MP to 1.1.
This accepts the rendered video for this same-grid/dense arm. It does not
qualify restored sparse suffix attention or progressive cross-grid sampling.
The user still suspects an audio defect and reports very slow execution.

Evidence SHA256:

- metrics: `cda4dcefad83d2370c76efbf8e09ebbadb6e46b3bf795431a0cc39133c080d1e`
- appended runtime log: `d1c7088424cc80f3dfd1dcf28ea4fe28d5d19122235d862257501eff6ea61441`

The log contains two prompts. Only the final `got prompt` segment belongs to
00732; its completion is **00:11:54**, not the earlier 576.06-second completion.
Actual target geometry is 64x64 latent / 1024x1024 decoded (1.048576 MP), versus
48x48 / 768x768 in 00731. Configured reduced source geometry is 44x44, but
`same_grid_target_control` runs continuation low/probe on the full 64x64 grid.
Its source tensor is `[1,24,62,64,64]`, with 12 exact prefix and 50 fresh tokens.
Continuation transfer is identity; the first chunk still uses learned transfer.

`vdn_linear_diagnostic=normal` and `native_grid_then_map_v1` remain active.
**`softmax_diagnostic=dense_suffix_same_domain` is still enabled.** Its positive
receipt reports 3,300 calls / 15,052,800 Q rows / 75,698,100 KV rows, with gathered
domain, prefix measure and grouped ownership unchanged. Compared with 00731,
aggregate Q and KV rows increase 77.8% and 67.4%. Aggregate row totals do not
determine attention FLOPs: that requires the sum of each group's Q-by-KV work,
or measured CUDA component timings. Reference/conditioning shapes also change.
Calls remain 18 logical / 15 actual /
three forecasts, with six sampler invocations and four history boundaries.

| Sampler timing | First chunk | Continuation |
|---|---:|---:|
| Low | 111.596 s | 260.276 s |
| Exact probe | 12.174 s | 50.768 s |
| High | 67.592 s | 76.007 s |
| Whole sampler | 192.070 s | 390.660 s |

Continuation low plus probe consumes 311.044 seconds, about 80% of its sampler
time. Its total rises from 208.609 seconds in 00731. First-chunk learned
upscaling takes only 0.633 seconds; continuation makes no learned-upscaler call.
Neither increased NFE nor the learned upscaler explains this slow arm. The
same-grid control intentionally forfeits low-resolution continuation savings,
and dense suffix attention intentionally forfeits sparse selection.

The existing coherent exact-audio evidence validator passes. Nine model entries
verify the authoritative native masks, effective overlap and timestep override
remain zero, and final AV prefixes are exact. Protected low/probe-to-high audio
error is max 4.768e-7 / RMS 2.853e-8. The original fractional-mask defect has
not returned.

Matched audio decode reports low/probe +0.7471 dB and final-high +3.1113 dB;
high suffix RMS increases +2.3676 dB relative to low/probe, while prefix level
changes only +0.0034 dB. Extra left decode context changes the suffix by zero,
and both production normalizers are provably inactive. Production boundary
level is +2.7291 dB before Audio Auto and +2.6833 dB afterward. These are level
measurements, not proof that the unspecified audible defect is a generation
burst. The 267-sample physical phase correction is already applied correctly.

Tracing assembly separately exposed a concrete PCM seam defect in Continuum's
released `v2/seam_guard.py`: equal-power blending amplifies correlated overlap,
global peak scaling changes both native patch endpoints, and gain/DC correction
does not return to identity before the unmodified suffix. A matching 0.01-amplitude
2 Hz waveform at 1 kHz reproduces a native step of 7.90e-7 becoming 0.002110.

[Continuum #37](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/37), head
`560e0c4e460a8a141832692281e7789852de5fdd`, fixes that local endpoint defect with
a convex overlap blend and tapered bounded gain/DC corrections. Seventeen new
regressions failed before the change and pass afterward, including actual
assembly preserving the whole continuous waveform and fresh suffix. The head
also covers all four transient fade lengths and Auto/Off assembly, with 23
waveform/assembly cases and 415 passing full CPU-suite tests. Exact-head CI run
36796253302 is green across Python 3.10–3.13 and the publisher-toolchain job.
This introduces no model/provider/VAE
calls. It does not establish a fix for the separate high-stage audio level
change or qualify the complete rendered soundtrack.

After a V3 audio patch is actually copied, `H3C-PT226 decoded-audio-seam receipt`
records `policy=convex_native_endpoints_v1 applied=True`, its boundary/fade
length, `left_endpoint_exact`, `right_endpoint_exact`, and before/after jump.
Both endpoint equality fields must be true for an applied patch. The existing
correlation gate and Audio Seam Off may produce no patch; absence of PT226 does
not establish successful correction. This observation adds no decode or model
calls.

### 00732 restoration instructions (historical; executed in 00733)

Apply Continuum #37 through Patcher after existing #36
`9ea1e75b2ddf8a3cb2a30e26c369aa63c0a32aea`, and restart ComfyUI. #37 targets
released main independently of rejected geometry candidate #36. The audio helper
was identical in both trees; the new assembly receipt touches a separate hunk in
their shared `v3/assembly.py`. The exact #36 -> #37 composition applies cleanly
and passes 59 seam, phase, video-seam and assembly-memory regression checks. Keep
Audio Seam Auto and Video Seam
Analyze Only. With retained decoded chunks, compare Audio Auto before/after the
PCM correction without resampling H3 to isolate its audible effect.

For the next sampling run, keep 00732's 1.1 MP workflow and change only the
**model diagnostic** `softmax_diagnostic=normal`. Retain same-grid control,
normal VDN linear, native temporal carrier, coherent exact audio, source-carrier
audio positions, learned handoff selector, witness off and all other inputs.
This restores the existing production attention path; no new speed algorithm is
introduced. PCM #37 is a separately recorded assembly change, so a combined run
is not a one-change audio-quality A/B. Require the positive exact-audio receipt,
no dense-suffix intervention, and raw/assembled media review.

Restore `spatial_stage_control=progressive_low_to_high` in a later separate arm
after normal suffix attention is accepted. The clean 00732 visual result cannot
be extrapolated across that restoration. Do not promote any diagnostic or
decoded geometry actuator based on this run.

GitHub pre-investigation checkpoints exist on Flow and VDN as
`checkpoint/00732-audio-and-speed-pre-investigation-20261001`. Continuum has
separate released-base, loaded-overlay and pre-full-validation checkpoints.

## 00731: normal learned-linear execution restored

The same-grid restoration run completed in 576.06 seconds with normal VDN
learned-linear execution. This establishes that the equal-grid validity guard
correction permits the intended hardware path. No rendered verdict was supplied
with this evidence; output-quality qualification remains pending.

Evidence SHA256:
- metrics: `400dffcdf05aabbcfef7907bb29b35a015342a18cfd0c2b72460a0393d1b95b8`
- runtime log: `0f622b74a4b608ddd699b42345f9e62f4184ada8ac097863c3fe913c586b7dbd`

The diagnostic controls and stage plan differ from 00729 only in
`vdn_linear_diagnostic`: `bypass_partitioned_linear` becomes `normal`. The log
contains 50 block-local active-linear markers and no diagnostic-bypass markers.
The six partitioned transformer calls publish 300 block contracts. Dense suffix
again verifies 3,300 calls / 8,467,200 Q rows / 45,230,700 KV rows with the same
gathered-domain, prefix-measure and grouped-ownership guarantees. The per-event
stage/actual/sigma model-call sequence matches 00729; aggregate counters are
18 logical calls, 15 actual calls and three forecasts.

The completed JSON now contains `partitioned_exact_audio_mask_verified`.
The existing coherent exact-audio evidence validator passes: nine model entries,
binary authoritative sampler/timestep/velocity masks, zero effective overlap,
no timestep override and no regenerated-prefix restoration. Low/high mask
digests match; final audio and video prefixes are exact. Protected audio
low/probe-to-high difference remains max 4.768e-7 / RMS 2.620e-8.

The measured loud boundary burst does not return: matched low/probe decode is
-15.5700 dB and final-high is -13.9562 dB; production pre-seam is -14.2845 dB.
Extra left decode context changes the suffix by exactly zero, and both
production normalizers remain provably inactive. These values describe a quieter
generated boundary window, not constant loudness or acceptance of the complete
soundtrack.

Upper45 latent successor-pair estimates remain small in the low/probe result and
survive high refinement. Source-low X is (0.175719, 0.070158, -0.084077, -0.017367)
and Y is (0.169830, 0.107721, 0.070430, 0.001224); final-high X is
(0.170640, 0.071526, -0.133896, -0.007813) and Y is
(0.194475, 0.120680, 0.052957, 0.011766) target latent cells. The decoded first
upper45 pair still measures approximately (-3.9904, +3.6290) pixels. These
estimates do not establish elimination of the visible chunk-to-chunk change.

This is not a tensor-matched causal A/B despite the matching recorded controls.
The authoritative carried-video prefix SHA256 changes from
`9f76b8337b500a0eaa26b4e19dbfaf42e4199c2a5b55207b50264a71a2552f18` to
`0aa7e745f9869047ee236d948fa4fe798308e12d17ee964837b80a454eac1178`.
Production preceding-audio RMS changes from 0.004993902 to 0.008718249, while
generated RMS changes from 0.001798426 to 0.001683459. Prompt-routing digests,
logged Core/Torch/Kitchen/Aimdo versions and the continuation noise seed match;
the reason the carried input changed is not established by these files. Neither
the apparent motion change nor the cross-run dB difference can be attributed
solely to restoring learned-linear execution.

The broad production runtime gate remains inapplicable to this same-grid arm:
it requires a learned cross-grid transfer that intentionally does not execute
under the identity control. Its frame-gauge qualification also requires resolved
DoRA auto-strength reports absent from this evidence. The positive exact-audio
validator and explicit diagnostic receipts pass independently; do not label this
run a complete production-gate pass.

First obtain the rendered verdict for raw/assembled video and audio. If this
normal-linear arm is acceptable, retain 00731's inputs and change only
`softmax_diagnostic=normal` for the next sparse-suffix restoration arm. Keep
same-grid target control, `vdn_linear_diagnostic=normal`, native temporal carrier,
coherent exact audio, source-carrier audio positions and witness off. Restore
cross-grid learned transfer separately after sparse attention is accepted.
No runtime arithmetic change or spatial correction is justified by these files.

## Learned-linear restoration: equal-grid guard correction

The subsequent restoration attempt aborted after 406.87 seconds when the first
partitioned learned-linear call reached `partitioned_frame_contract()`. Runtime
log SHA256: `aacc3095f851532319844b0b7f5ab323cefa55d4a4bf5bab9dc21ee130f64c5d`.
This failed run provides no rendered result for restored learned-linear execution.

Flow's same-grid control and VDN's sequence parser already permit equal source
and target grids, but the linear frame helper still required strictly fewer
source rows. The prior bypass arm did not execute this helper. VDN #35 now
accepts `0 < source_rows <= target_rows`, retaining the generated-suffix and
source-exceeds-target rejection. Equal grids produce unit physical measure;
the existing readout uses native arithmetic without interpolation. This fixes
an inconsistent validity guard and introduces no new numerical policy or calls.

All twelve square/rectangular, prefix-length and anchor-trimming regression
cases failed at the original guard and pass after its correction, matching
released VDN readout within the existing FP32 oracle tolerance. Invalid geometry
remains rejected. The complete local pinned-Comfy/official-oracle suite passes
248 tests. Flow's paired source-contract CI pin uses corrected VDN head
`e253432b9f79db91a0d99461bbd85fb1574a39b3`.

Refresh VDN #35 and Flow #93 through Patcher and restart ComfyUI. Repeat the
failed restoration arm: `vdn_linear_diagnostic=normal`, retaining all other
00729 settings. Require successful learned-linear execution, the persisted
positive exact-audio receipt, unchanged dense-suffix ownership receipts and
complete raw/assembled media acceptance. The slight chunk-to-chunk appearance
change remains unresolved; this guard correction supplies no new media evidence.

## 00729: audio improvement in the frozen diagnostic arm

User inspection reports that the audio issue appears gone and the frame shift is
pretty much gone, with a slight change between chunks still visible. This is
qualified evidence for this same-grid, learned-linear-bypass, dense-suffix arm;
it does not qualify restored production attention or cross-grid learned transfer.

Evidence SHA256:
- metrics: `496c1c2df344e4aee3276580381e31e1e065429bf8a55a7f473a820e7821165b`
- runtime log: `53de2985bdac0c090a2bba917c873b999a410e51be6369a0d426d1d24c27d3aa`

The diagnostic control fields, same-grid geometry and per-event logical/actual
model-call sequence match 00728. The complete learned-linear bypass again records
300 calls / 10,713,600 video rows; dense suffix again records 3,300 calls /
8,467,200 Q rows / 45,230,700 KV rows. The authoritative low/high audio masks now
match, nine native-mask verifications execute, and inner timestep overrides are
zero. The stored legacy selector remains unchanged with 16 requested ticks; its
sampler ramp is inactive. The low/probe-versus-high protected-audio difference
falls from max 11.1940 / RMS 0.874204 to max 4.768e-7 / RMS 2.416e-8.

The matched low/probe decode boundary changes from +6.5377 dB to -9.6175 dB;
final-high changes from +7.1323 dB to -8.6536 dB. Production PT213 changes from
+7.0079 dB to -8.8709 dB before seam processing. The preceding production RMS is
identical at 0.004993902; generated RMS falls from 0.011190105 to 0.001798426.
This supports removal of the loud burst, rather than a downstream normalizer or
seam repair. It does not establish constant audio loudness across the boundary.

Video phase estimates do not show a corresponding disappearance of displacement.
Upper45 final-high X pairs change from (-0.108105, -0.390088, -0.128388, -0.018460)
to (-0.145154, -0.429790, -0.093538, -0.017967) cells; decoded first-pair motion
remains approximately (-6.31, +4.16) pixels. These estimates are not a reliable
proxy for the user's rendered assessment. No further spatial actuator is
authorized by these measurements. The remaining chunk-to-chunk appearance change
and general video causality are unresolved.

An export-order defect was also found: `sampler_wall` triggers autosave before
the outer audio verification events are emitted, so the saved 00729 JSON omits
the final exact-audio receipt even though its nine verification calls and the
corrected runtime logs are present. Flow now persists the completed invocation
after those events. This changes evidence export only and adds no sampling,
provider or VAE work. The saved-file regression covers the actual autosave order.
The original 00729 evidence remains intact; do not invent its missing receipt.

The next production-restoration discriminator should keep 00729 frozen and
change only `vdn_linear_diagnostic=normal`, retaining same-grid geometry, dense
suffix and coherent exact audio. Qualify complete raw/assembled media and the
positive final mask receipt before separately restoring sparse suffix selection
or cross-grid learned transfer. No unchanged diagnostic rerun is needed merely
to reconfirm the already observed audio improvement.

## 00728: coherent exact-audio correction

Evidence SHA256:
- metrics: `36b43d3bdaa418aacec9a1c75b5e443fd6c1e1594eb51d26bbe42f33a9a2220d`
- runtime log: `3a59694fdd7a75cc4ad6ff4e2fd8b6766296703ebf9b15acd416721f8bbbf08d`

Dense suffix executed 3,300 calls / 8,467,200 Q rows / 45,230,700 KV rows with
unchanged domain, prefix measure and grouped ownership. Learned-linear bypass
executed 300 calls / 10,713,600 video rows. Upper45 source-low X successor pairs
were (-0.124866, -0.344425, -0.166585, -0.030154) cells; final-high was
(-0.108105, -0.390088, -0.128388, -0.018460). Identity transfer again added zero
DC correction. Raw decoded PT212 still reports an initial upper45 displacement
of (-6.3550, +4.4346) pixels. Sparse suffix selection is not a sufficient cause.

With shared authoritative prefix and matched decode context, low/probe audio
already has +6.5377 dB boundary RMS; final-high has +7.1323 dB. Extra left context
changes the decoded suffix by exactly zero. The production/common decode
difference is approximately 0.0771%, and both production normalizers are inactive.
PT213 measures +7.0079 dB before seam processing and +7.0059 dB afterward.
The burst originates before high refinement; left decoder context, normalization
and seam processing are not sufficient explanations.

Source tracing found a definite conditioning mismatch in the selected
`sampler_mask_exact_timestep` mode: 16 protected audio ticks were released into
fractional sampler ownership, while inner H3 labels described them as exact.
Core's input is `m*x + (1-m)*clean`, and its outer audio velocity conversion uses
m. Labeling that partially regenerated context clean breaks the native input /
timestep / velocity contract. Final restoration then substitutes the original
prefix for the regenerated context. This source defect is proven; its rendered
contribution remains a hardware hypothesis.

The correction keeps the authoritative mask at every low/probe/high sampler
entry. Core uses matching native input, timestep and velocity masks. `exact_mask`
is the explicit selector; stored `sampler_mask_exact_timestep` values alias it.
Requested overlap width remains provenance; effective width is zero. Each inner
model entry verifies equality with the native mask and passes kwargs unchanged.
A mismatch fails before the inner forward. Invocation context is cleaned up on
success, execution failure and preflight rejection, and nested contexts fail
without replacing the existing owner. Exact masks must be binary and uniform
across channels, stereo lanes and batch items. Core may repeat the native mask
for CFG batching or omit it for fully generated audio. The Core #15988 outer
velocity-mask compatibility requirement remains enforced. Released Target Input and the explicit `sampler_mask` /
`model_timestep_only` comparison modes retain their behavior.

Refresh Flow #93 through the existing Patcher overlay stack and restart ComfyUI.
Keep the entire 00728 workflow frozen, including its stored legacy selector and
16 requested ticks. Keep same-grid control, linear bypass, dense suffix, native
temporal-carrier policy, source-carrier audio positions, prompts, seeds, LoRAs,
the audio audit and all other overlays unchanged. Selecting `exact_mask` is
optional because the legacy selector invokes the same correction.

Require `partitioned_exact_audio_mask_verified` with positive `verified_model_entries`,
`policy=coherent_exact_audio_mask_v1`, `effective_overlap_ticks=0`, exact input /
timestep / velocity flags true, `timestep_override_applied=false`,
`regenerated_prefix_restored=false`, `final_prefix_exact=true`, `fail_closed=true`,
and every extra-work count zero. Require dense-suffix and linear-bypass receipts
again. `audio_guided_overlap.applied=false` is expected; the new exact-mask receipt
replaces an applied-ramp requirement. The old mapped-sparse production gate does
not qualify this all-dense diagnostic arm.

Compare the matched low/probe audio decode, final PT213 RMS/spectrum, complete raw
decoded video boundary window and background continuity. Joint audio context may
couple into video, but the frame-shift root is unresolved. Exact-main semantics,
attention domains, VDN/Sol arithmetic and the three continuation sampler lifetimes
/ two history boundaries remain unchanged. No model/provider/VAE call is added.
Flow #89 and VDN #33/#34/#35 are not consolidated or promoted from source tests.

The authoritative specification is
[the design at b97ed34d0db253f26e5c7a391c3fb1358c3c3684](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/b97ed34d0db253f26e5c7a391c3fb1358c3c3684/docs/HETEROGENEOUS_EXACT_PREFIX_BOUNDARY_IMPLEMENTATION_DESIGN.md).
Its hardware conditional remains in force: a mathematical ordering difference
does not establish that cross-grid taps cause the rendered artifact.

## Refreshed source baseline

| Component | Ref |
|---|---|
| Flow main | `a6249b8343becc1458a4555d2bb25cd523983c90` |
| Flow #89 | `fa8d65b84e0666ce467b838544d92c801394a002` |
| VDN Plus main | `b78e94d0365ffc5059924a048af707e565d0380e` |
| VDN Plus #33 | `da3627f85d494bdf4213251deb3ba2a94b8a2f36` |
| VDN Plus #34 observer | `cae13fb5e8d71b93ee3134b629d23fec7c819c5b` |
| VDN Plus #35 diagnostic head (candidate-C retained; equal-grid + dense-suffix discriminator) | `e253432b9f79db91a0d99461bbd85fb1574a39b3` |
| Sol main / v0.1.6 | `bef9b300275a89290ca2d53eafff79af06f5e0ef` |
| Sol source-contract pin | `93b3e03f2b7b579aaf55fa0f87f55083b259e25c` |
| Continuum Plus main | `e870875b1a29968d39d72ba304e9f2b05544c015` |
| Rejected Continuum #35 | `28624d2577a62b6b48a04f3897e7b03a12697eb7` |
| Rejected Continuum #36, actuator disabled | `9ea1e75b2ddf8a3cb2a30e26c369aa63c0a32aea` |
| Learned upscaler | `620165a311de9b28a36260219fb5cd370a304e3c` |
| Core current main | `9d80841aa1990305cc7a280c5ea505f317efcc2d` |
| Flow Core oracle / native-mask oracle | `1af040bf022569d7a890241c8dd79b296cda483f` / `421a1c245c682c04d4325ba365f40c834c66f5b0` |
| VDN Core oracle | `6c53f8c9a06d95f3d847009ceaae55c624169247` |
| OpenVDN oracle | `b8cb28fbfca0266d1c7742a9f25ab8b58191de97` |

All requested branches/pins were fetched. Flow/VDN review threads were resolved
at retrieval. Sol's source-contract pin differs from main only in release
metadata. These are fetched source refs, not proof of installed Patcher trees.
Earlier installed evidence abbreviated Core `34b50ec9` on `patcher/stack`.
The 00726 hardware log instead reports ComfyUI `v0.38.0-12-g0d48b6032` on
`patcher/stack` with Torch 2.10.0+cu130. This runtime change is a comparison
confound relative to 00724 and must not be silently attributed to the same-grid
selector.

Both repositories have remote `checkpoint/heterogeneous-observation-start-20260930`
refs at their protected PR heads. Development uses separate
`mirror/heterogeneous-observation-20260930` branches. Pre-review checkpoints
preserve the first observation implementation. Sol and Continuum source branches
remain untouched.

## Original evidence recovered

The original bytes were recovered using the design's materialized paths and
verified against its hashes:

| Original file | SHA256 |
|---|---|
| `metrics_00717_.json` | `8abe3148210934d0765b36064108cccfd73e1af08bd313628d30c602ec575add` |
| `Pasted text(20260929-203040).txt` | `11ac6cfedde9c97ddda976f16fdd8fb46b8cd7c8edc7c59fea125679961c6e52` |

00717 uses prefix12, temporal62, target latent48×48 and source34×34. Its
upper45 pre-high pair0 is approximately (−0.021,+0.010) target cell, while
pair1 is (−3.898,−3.809). This window is not clean. Whole-workflow counters for
two physical groups are 14 actual H3 evaluations (low8/probe2/high4),
18 logical calls, four forecasts and six sampler/four history boundaries.
One continuation remains three sampler lifetimes/two history boundaries.
The original MP4, original complete workflow JSON, full overlay graph and
checkpoint hashes were not available to this implementation environment.

## Implementation and explicit deviations

1. Milestone one remains the bounded observation ABI. Flow owns a default-off
   CPU sink; VDN #34 exposes actual raw/filtered/mapped/activated features and
   A/B/alpha norms. Observation itself is output-neutral.
2. 00719 established the actual-checkpoint operator mismatch. 00721 then ran the
   prescribed `suppress_cross_grid_temporal_taps` B arm on SM120 and materially
   changed the inherited pre-high boundary. That satisfies the design's gate to
   implement candidate C, but not its production/media acceptance gate.
3. VDN candidate C is isolated in **VDN #35**, stacked on observer #34, which is
   stacked on #33. #33 and #34 remain intact. Flow #93 carries only the paired
   selector/contract/capability/receipt logic required to invoke and verify #35.
   Flow #89 remains untouched at its one-clean-commit production-candidate
   baseline.
4. Candidate C adds independently versioned numerical leaf
   `h3_flow_partitioned_vdn_temporal_carrier_v1` API 1. Absence remains
   `native_grid_then_map_v1`; the opt-in arm is
   `destination_grid_stencil_v1`. The numerical digest binds the Flow semantic
   plan digest, diagnostic mode, actual checkpoint short-conv specification and
   physical mapping/precision policy.
5. For cross-grid temporal taps only, VDN #35 maps the **raw projected** feature
   to the receiving H3 lattice with the existing FP32 bilinear/border mapper,
   restores feature dtype, applies the trained depthwise 5x5 spatial stencil on
   that destination grid, then preserves the existing temporal weight, SiLU,
   Q/K L2, A/B/alpha, recurrence and readout path. Same-grid arithmetic remains
   the existing native batched path. No authoritative residual rows, RoPE,
   sampler latents, output geometry or decoded pixels are resampled.
6. The candidate is fail-closed: it requires paired VDN carrier API1,
   `vdn_linear_diagnostic=normal`, exact partitioned transformer context and a
   compatible checkpoint short-conv declaration/weight geometry. Native/default
   execution does not depend on candidate capability and retains the previous
   provider-cache identity.
7. The current-Core native mixed-grid test fixture accepts Core's `attention`
   keyword only for compatibility testing; production model code is unchanged.
8. Observation remains bounded to two complete heads in the first actual
   low-stage convolution block, the first mixed boundary and temporal radius ≤4.
   These bounds affect evidence completeness, not default arithmetic.

Tests cover observation neutrality, dtype/branch/anchor coverage, CPU lifetime,
budget enforcement, capability rejection, stage owner cleanup, option cloning
and the paired source-contract ABI. A missing selected low-stage observation
cannot silently become a successful diagnostic sample.
Final review corrected receipt metadata so bounds and physical measure scales
use the same optionally anchor-trimmed frame domain as captured features. The
untrimmed runtime scales are recorded separately; production weighting is
unchanged. The four suppression/anchor combinations verify this metadata.

## Supplied paper check

The supplied DMD2, Sol-Attn, Sol-engine, VSA, Spectrum and Video DeltaNet PDFs
were inspected. Video DeltaNet section 2.3 confirms the released K/V depthwise
5×5 spatial and five-tap temporal filters, SiLU and Q/K L2, with separately
calibrated branches. Appendix A.5 makes checkpoint architecture metadata part
of the release contract. Sol-Attn routing and Spectrum feature history remain
separate mechanisms. These papers do not by themselves establish rendered causality or overturn the
rejected output repairs. The 00721 hardware intervention, rather than the paper
review, is what authorized implementation of candidate C. The design's
whole-window SM120/media acceptance gate remains unchanged.

## Patcher dependency order

The implementation PR descriptions record exact final diagnostic SHAs. Use
those SHAs and the following dependency order, refreshing all overlays before
starting ComfyUI:

1. VDN Plus main, #33, [VDN #34](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/34)
   at `cae13fb5e8d71b93ee3134b629d23fec7c819c5b`, then
   [VDN #35](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/35) at
   `96b4e49e507d223f7900e3cb35a9ae0a207ec213`.
2. Flow main, existing #89, then [Flow #93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93), its stacked qualification/candidate-control draft.
3. Sol v0.1.6/main at the ref above; no diagnostic Sol overlay.
4. Continuum Plus main, then #36 solely for its retained diagnostics and disabled
   rigid actuator. Do not add #35. Verify loaded #36 source at the ref above.
5. Learned upscaler at the ref above, and the remaining already-selected
   Spectrum/DiffAid/RefDelta/Untwist/LoRA/DoRA providers at frozen installed refs.

The diagnostic draft bases are the corresponding existing PR branches, not
main. Applying only a stacked draft's delta without its parent is invalid.
Record Patcher's effective tree and ordered overlay graph after refresh; the
public PR sequence alone does not prove which code is loaded. Do not update
Core or any unrelated provider between comparison arms.

## Matched workflow and observation

Recover the original workflow and reference inputs before claiming a matched
00717 comparison. The metrics/log do not reconstruct every prompt, seed,
adapter strength and input. Freeze that workflow and its effective checkpoint
and source manifest. Disable decoded geometry mutation; with #36 use
`Video Seam = Analyze Only` to preserve raw video while keeping diagnostics.
Keep the original audio policy in both arms and record any indirect audio
output differences independently.

Use node type `H3PartitionedExactPrefixDiagnosticHandoff`, displayed as
**MiniMax H3 Partitioned Exact-Prefix Handoff**, with its diagnostic selectors. Verified
00717 selectors are:

| Selector | Both arms |
|---|---|
| `prefix_transformer_context` | `exact_target_partitioned` |
| `audio_position_domain` | `source_carrier` |
| `audio_guided_overlap_mode` | `sampler_mask_exact_timestep` |
| `audio_guided_overlap_ticks` | `4` |
| `provider_boundary_stabilization` | `soft_support_v1` |
| target/source latent grid | `48×48 / 34×34` |
| exact prefix / T | `12 / 62` |

Keep all remaining controls, the learned full-sequence provider, existing
one-token bridge, Sol tau/routing, Spectrum, seeds/noise and LoRA/DoRA settings
unchanged. Do not select a source-uniform shadow path. Run:

* A: `vdn_linear_diagnostic=normal`.
* B: `vdn_linear_diagnostic=suppress_cross_grid_temporal_taps`.

For each observed arm, set the node's `capture_boundary_witness=true`.
This is a per-run control and does **not** require an environment variable or a
ComfyUI restart. Artifacts use unique filenames under
`ComfyUI/output/h3-flow-boundary-witness`; the receipt records the active
`vdn_linear_diagnostic` mode, so A and B remain attributable without separate
process launches. Set it independently for:

* A: `vdn_linear_diagnostic=normal`, `capture_boundary_witness=true`.
* B: `vdn_linear_diagnostic=suppress_cross_grid_temporal_taps`,
  `capture_boundary_witness=true`.

The historical `H3_FLOW_BOUNDARY_WITNESS_DIR` process variable remains only as
a headless/backward-compatible fallback when no explicit node selection is
present. An explicit node OFF overrides a stale environment value.

Witness capture is a default-off observer, not a stencil selection or a
dense-attention switch. It creates one paired `.pt`/`.json` artifact for each
observed continuation low lifetime and publishes `partitioned_boundary_witness`
in Flow metrics. The `.json` contains loaded module paths/hashes, actual
precision settings, frame/head indices and tensor SHA256. Requested observation
requires the paired VDN draft before sampling.

Verify actual suppression counters and
`partitioned_vdn_linear_diagnostic_verified`; a widget label is insufficient.

### Completed A/B hardware evidence

00719 is the witness-enabled A/normal arm. Its witness is actual low-stage block
0, heads 0/55, boundary inner frame 11, and contains the cross-grid K/V temporal
taps. Offline replay on that captured input shows a material
native-filter-then-map versus map-then-destination-filter difference, while
00718 and 00719 trajectory receipts are identical, establishing witness
neutrality.

00721 is the B arm. It verified 250 suppression calls, 3000 cross-grid taps and
1,297,500 destination rows with the same 14 actual H3 evaluations and existing
sampler/history topology. The upper45 exact-restored pre-high successor pair1
changed from A `(-3.89833,-3.80939)` target cells to B
`(-0.94051,-0.32042)`, about an 81.8% reduction in vector magnitude. The
cross-grid short-conv path is therefore causally material to the inherited
pre-high discontinuity.

B is not a fix. Its final upper45 window remains displaced and its aggregate
upper45 net displacement is worse than A. No rendered A/B movies were supplied
with 00721, so background-retention/media acceptance cannot be inferred from
metrics alone.

### Candidate C result: 00722 rejected by rendered-media gate

00722 executed candidate C exactly as intended on SM120:

* `vdn_linear_diagnostic=normal`
* `vdn_temporal_carrier_policy=destination_grid_stencil_v1`
* 250 verified destination-stencil calls
* 3,000 cross-grid taps
* 2,000 mapped carriers / 865,000 mapped carrier rows
* stable numerical digest
  `82fd2c133290aece625eb432cee2764be70d73a7f0e68b17bc5b3306f7025547`
  across low/probe and final verification
* unchanged 14 actual H3 NFE / 18 logical calls / six sampler invocations /
  four history boundaries.

The actual low-stage witness records block 0, heads 0/55, K/V short conv,
destination-grid policy and the same numerical digest with zero extra
H3/provider/VAE calls.

C materially improves the internal inherited pre-high transition:
upper45 successor pair1 changes from A/native
`(-3.89833,-3.80939)` to B/suppression
`(-0.94051,-0.32042)` and C
`(-0.70003,-0.07865)`. C therefore reproduces the causal B-arm effect without
deleting the learned cross-grid contribution.

That internal improvement is **not sufficient**. User inspection of the actual
00722 rendered output reports that the same frame shift is still visible. This
fails the design's explicit whole-window rendered-media criterion. The
destination-stencil path must not be tuned further or promoted from these
metrics. Preserve VDN #35 and the 00722 artifacts as rejected diagnostic
evidence only.

### 00724 result: complete learned-linear bypass also fails rendered acceptance

00724 is a valid bypass arm:

* `vdn_linear_diagnostic=bypass_partitioned_linear`
* `vdn_temporal_carrier_policy=native_grid_then_map_v1`
* 250 fail-closed verified bypass calls / 5,340,500 video rows
* unchanged 14 actual H3 NFE / 18 logical calls / six sampler invocations /
  four history boundaries
* source-carrier audio-position execution verified and the run completed.

The bypass strongly changes internal geometry but does not remove the rendered
defect. Upper45 `exact_restored_pre_high` successor pair1 becomes
`(+2.97523,-0.24417)`, i.e. it reverses direction relative to A/native rather
than simply shrinking. After target-high the upper45 net is
`(-0.37610,+0.28953)`. Despite that numerical change, user inspection reports
that the same visible frame shift remains. This rejects the complete partitioned
VDN learned-linear complement as a sufficient cause of the rendered artifact.

The more important localization is earlier in the handoff. In 00724 the
source-low upper45 successor pair1 is only about
`(-0.12913,+0.01193)` target-equivalent cells. Immediately after the learned
34x34 -> 48x48 clean-video transfer it is about
`(+3.08880,-0.35257)`, roughly 24x larger in vector magnitude. Exact-prefix
restoration then changes pair0 as intended but leaves the successor jump at
`(+2.97523,-0.24417)`. Similar source->learned amplification occurs in the
other matched arms despite their different VDN interventions. This makes the
low->high transfer the leading current suspect.

This is not yet proof that the learned checkpoint itself is the sole cause:
the remaining alternatives are (a) the learned 3D upscaler introduces the
spatial/temporal gauge change, or (b) the broader low->high resolution
handoff plus target-high response does so even with a simple spatial transfer.

### 00725 first same-grid attempt: aborted by stale strict-smaller contract guards

The first hardware attempt at the new same-grid control did **not** test the
hypothesis. The control reached continuation low-stage setup, but Flow's
`PartitionedExactPrefixPlan` still rejected equal source/target grids before
the first continuation transformer evaluation with:

`partitioned exact-prefix plan requires a strictly smaller source grid`.

That exposed an incomplete implementation: the scheduler/stage layer allowed
the explicit same-grid diagnostic, but the published partition contract and the
paired VDN parser still encoded the old heterogeneous-only invariant.

The fix is now coordinated across both repositories:

* Flow #93 authorizes equal source/target partition geometry only when the
  explicit same-grid control is active, publishes zero prefix measure bias and
  `heterogeneous_spatial_domains=false`, and routes the same-grid case through
  the full handoff plumbing instead of returning early on equal geometry.
* VDN #35 accepts equal-grid partition contracts, preserves the ordinary
  strict/non-expanding geometry checks, and canonicalizes the equal-grid
  contract with zero measure bias and
  `heterogeneous_spatial_domains=false`.

The failed attempt produced no rendered discriminator result and must not be
counted as evidence for or against the resolution-transition hypothesis.

### 00726 same-grid hardware result and clean-state ownership defect

00726 is the first successful execution of the direct same-grid control. Runtime
receipts prove that configured progressive source intent remained 34x34 while
continuation low/probe actually ran at 48x48, target-high remained 48x48, the
handoff operator was `diagnostic:same_grid_target_identity`, it executed once,
and the actual learned-checkpoint provider was not called. The complete
partitioned VDN learned-linear bypass also remained active.

Rendered-media inspection remains authoritative: the user reports that a
**slight frame shift still happens, but it is much less prominent**. Removing
the low->high spatial transition therefore materially changes the visible
defect, but does not eliminate it.

The pair is not perfectly controlled. 00726 recorded 15 actual H3 NFE
(low9/probe2/high4) and three Spectrum forecasts, whereas 00724 recorded 14
actual H3 NFE (low8/probe2/high4) and four forecasts. 00726 also ran a newer
ComfyUI/Patcher Core. Those differences prevent assigning the entire visible
improvement to resolution alone from this single comparison.

00726 additionally exposed a concrete Flow ownership bug. The handoff noise
policy was `source_residual_patch_refinement_v1`; at equal 48x48 geometry that
residual transport itself was identity, and frame-gauge registration returned
`identity/already_aligned`. Despite that, downstream splice logic reported
`splice_clean_source=inverse_recovered` and
`splice_recovery=inverse_conditional_renoise`: it reconstructed clean video
with deterministic Gaussian noise even though the conditional state had been
built from the carried source residual.

At the 00726 split sigma (~0.8780488), that incorrect inverse strongly amplifies
the residual-vs-Gaussian difference. The synthetic clean tensor then drove a
production one-token DC bridge: one suffix token was changed with delta RMS
~0.231223 and max absolute delta ~0.455920 even though the same-grid provider
prefix was already aligned with the authoritative prefix. Consequently the
00726 `learned_native` / `exact_restored_pre_high` diagnostics are
contaminated by this reconstruction error and are not reliable evidence of a
real identity-handoff spatial jump.

Flow #93 now fixes ownership at the source:

* the clean postprocess hook retains the exact clean tensor that actually feeds
  conditional re-noising;
* downstream splice/bridge logic uses that tensor directly;
* source-residual mode fails closed if that tensor is missing instead of
  falling back to Gaussian inverse recovery;
* deterministic inverse recovery remains available only for the independent
  deterministic-noise contract;
* regression tests prove that an identity same-grid handoff cannot manufacture
  a suffix DC correction from this mismatch.

The clean-ownership correction first reached CI-green head
`53e99780c3379e2d7606c2cf0de2b0b6eb1252dc`; the subsequent documentation/
provenance head `7366e5acf654490b16e2a443584700c8f06668fb` also passed exact-head CI
run `36741231859` across source-contracts and Python 3.10/3.11/3.12/3.13.
Checkpoints preserve both the pre-fix 00726 state and the validated correction.

### 00727 same-grid rerun: ownership fix validated, residual shift remains

00727 repeated the same-grid configuration on the same reported Core runtime as
00726. Its handoff receipts validate the ownership correction exactly:

* `same_grid_identity_transfer_applied=true`;
* `splice_clean_source=actual_clean_postprocess`;
* `splice_recovery=actual_clean_postprocess_no_inverse`;
* `suffix_dc_bridge_delta_rms=0.0` and max/mean absolute delta both zero;
* seam amplification ratios are exactly 1.0 at the identity handoff;
* VDN learned-linear remained fail-closed bypassed for 300 calls /
  10,713,600 video rows.

The user nevertheless reports a **very slight frame shift** in the rendered
00727 movie. The upper45 source-low pairwise trajectory is unchanged from 00726:
dx `[-0.12936,-0.28895,-0.14254,-0.01740]`, dy
`[-0.02179,-0.03148,-0.00128,-0.00273]`. After target-high it remains close:
dx `[-0.09521,-0.27566,-0.11301,-0.00237]`, dy
`[-0.04498,-0.00125,-0.04743,-0.01593]`.

This is the key localization. Under equal 48x48 low/high grids, identity
handoff, zero learned-upscaler calls, correct clean ownership, zero DC mutation,
identity frame gauge and complete VDN learned-linear bypass, the small boundary
translation already exists in low/probe and target-high mostly preserves it.
The large historical defect is strongly associated with the spatial transition;
the remaining defect is upstream inside the exact-partitioned low/probe
transformer path.

### Next discriminator: same-domain weighted-dense suffix attention

The authoritative design's deferred sparse-selection discriminator is now
authorized. Flow #93 and VDN #35 add an opt-in leaf
`h3_flow_partitioned_softmax_diagnostic_v1` API 1 with
`dense_suffix_same_domain`.

This arm does **not** change grouped ownership. VDN constructs the same
partitioned grouped plan, gathers the same Q/K/V rows, transports the same
query-position wire, sink rows, target-prefix K range and prefix log-measure,
then changes only the existing Sol request's `force_dense` flag for
generated-suffix local-query groups. Prefix-query groups remain on their
existing dense path; global/anchor behavior is unchanged. The VDN learned-linear
selector remains independent.

The diagnostic publishes positive counters for calls/Q rows/KV rows and Flow
fails closed unless nonzero work is observed. Flow also adds the diagnostic leaf
to provider-cache and partitioned layout/history identity only while selected,
so a dense diagnostic sample cannot reuse sparse numerical history. Normal/
absence semantics remain unchanged.

For the next hardware arm, keep the complete 00727 workflow frozen and change
only:

* `softmax_diagnostic=dense_suffix_same_domain`

Keep in particular:

* `spatial_stage_control=same_grid_target_control`
* `handoff_transfer_control=learned_3d`
* `vdn_linear_diagnostic=bypass_partitioned_linear`
* `vdn_temporal_carrier_policy=native_grid_then_map_v1`
* `capture_boundary_witness=false`
* `audio_position_domain=source_carrier`

The run must emit `partitioned_softmax_diagnostic_verified` with positive
`dense_suffix_calls`, `dense_suffix_q_rows` and `dense_suffix_kv_rows`,
plus `same_gathered_domain=true`, `prefix_measure_unchanged=true`,
`grouped_ownership_unchanged=true` and `fail_closed=true`. The VDN bypass
receipt and 00727 identity-clean receipts must remain positive/zero respectively.

If the residual rendered shift materially changes, sparse Sol selection is
causally material and routing/history becomes the next investigation target.
If it remains with the same character, sparse selection is not a sufficient
cause and the design advances to the provider-transfer discriminator. Internal
motion metrics remain diagnostic; rendered whole-window inspection is the gate.

### Revised next discriminator: remove the spatial grid transition itself

The previous learned-vs-bicubic proposal is **not** the required test for the
current hypothesis. The user is specifically testing whether Flow Regenerate's
low-resolution -> high-resolution stage transition is what creates the frame
shift. Replacing one 34x34 -> 48x48 resize operator with another still retains
that transition and therefore cannot answer the question directly.

Flow #93 now exposes an appended diagnostic selector
`spatial_stage_control`:

* `progressive_low_to_high` — historical/default behavior. The configured
  reduced source geometry remains active.
* `same_grid_target_control` — ignores the reduced source geometry for actual
  low/probe execution and runs low/probe directly on the caller target grid.
  The original configured reduced source geometry is retained only for handoff
  split selection, so fixed and auto-computed handoff timing remain matched.
  At the handoff, an identity provider is used through the same handoff plumbing:
  no spatial resize and no learned-checkpoint invocation occur. Residual/noise
  transport, clean postprocess, exact-prefix restoration, audio state, guidance,
  Spectrum/history ownership and target-high sampling remain in place.

The control is fail-closed. It requires
`handoff_transfer_control=learned_3d` and
`vdn_temporal_carrier_policy=native_grid_then_map_v1`; combining it with the
bicubic transfer diagnostic or destination-grid stencil is rejected.

For the next matched hardware run keep the complete **00724** workflow and
every selector frozen and change only:

* `spatial_stage_control=same_grid_target_control`

In particular keep:

* `handoff_transfer_control=learned_3d`
* `vdn_linear_diagnostic=bypass_partitioned_linear`
* `vdn_temporal_carrier_policy=native_grid_then_map_v1`
* `capture_boundary_witness=false`
* the existing source scale/width/height values unchanged.

Do **not** manually set source scale to 1.0; the ordinary configuration rejects
that by design. The diagnostic selector overrides only the effective low/probe
spatial stage internally while preserving the configured source geometry as
provenance and for handoff split selection.

The run must emit `partitioned_stage_plan` with
`same_grid_control_active=true`, equal effective source/target H/W, and the
original `configured_progressive_source_hw`. It must also emit
`partitioned_handoff_transfer_control` with operator
`diagnostic:same_grid_target_identity`, exactly one spatial-control call and
zero actual learned-checkpoint provider calls.

Interpretation is direct:

* if the visible frame shift disappears/materially collapses, the
  low-resolution -> high-resolution stage transition is causally implicated;
* if it remains with comparable character, the artifact is not caused simply by
  changing spatial resolution between low/probe and high, and the next
  discriminator must move elsewhere.

The earlier `bicubic_same_source_control` remains available as archived
diagnostic machinery but is **not** requested for this hypothesis.

## Source and checkpoint provenance

The read-only helper hashes installed tracked source bytes, full HEAD/tree,
status, workflow and explicitly named checkpoints; safetensors headers retain
every tensor shape/dtype and metadata. Run it separately from timed inference:

```bash
python /home/toor/ComfyUI/custom_nodes/minimax-h3-flow-aligned-regenerate/tools/capture_boundary_provenance.py \
  --comfy-root /home/toor/ComfyUI \
  --workflow /absolute/path/frozen-workflow-A.json \
  --patcher-state /absolute/path/patcher-overlay-export.json \
  --checkpoint /absolute/path/actual-model.safetensors \
  --checkpoint /absolute/path/actual-vdn.safetensors \
  --checkpoint /absolute/path/actual-upscaler.safetensors \
  --witness-receipt /absolute/path/boundary-witness-ID.json \
  --output /absolute/path/provenance-A.json
```

The Flow directory above is the path reported by the original log; verify the
current loaded path. Replace input/output placeholders with actual paths. Repeat
`--checkpoint` for every effective model, BF16 affine source, encoder, VAE,
LoRA/DoRA and adapter checkpoint. Include their effective strengths and VDN
configuration in the frozen workflow/Patcher export. The separate helper's
`collector_process_precision` is not the running model's precision; use the
actual witness's `precision_at_actual_call` and runtime log for that.

The helper deliberately leaves `installed_provenance_complete=false` until
the overlay graph, all loaded modules/checkpoints/settings and reference/noise
identities have been checked. It does not reconstruct unavailable Patcher
state. Missing runtime hashes must remain missing evidence.

Analyze each tensor witness with VDN's `tools/analyze_boundary_witness.py` as
documented in VDN's `docs/BOUNDARY_FEATURE_WITNESS.md`. Preserve the original
observed receipt and report the CPU replay as additional diagnostic work.

## Runtime, media and costs

Save each existing Core decoder IMAGE output before Continuum assembly and the
assembled movie after it. Fan out the already-decoded frames to saving nodes;
do not add another VAE decode. Preserve raw physical-group frames including the
discarded prefix context so frame indices and trim39 can be independently
checked. Store the native fps and embedded frozen workflow with each movie.

First run the square 17→24 patch-grid comparison. Acceptance requires two
continuation boundaries, then a non-square case and Spectrum enabled/disabled
history validation. If the original graph has only one continuation, its
original two-group reproduction remains the first comparison; additional
three-group validation is separately recorded experimental work. Do not
pretend a new third-group prompt was part of the original run.

Inspect the first retained frame at local39, at least 24–40 subsequent frames
and the complete retained segment at normal speed. Assemble annotated A/B
side-by-side media using fixed top-edge/cabinet/curtain landmarks. Assess upper,
central and lower correspondences/flow/scale/shear over multiple pairs and 4×4
regions against ordinary pre-boundary camera motion. Reject a displaced jump,
freeze, border exposure, expansion or compensation drift even when first-pair
phase error improves. Exact-main background continuity must remain preserved.
Final candidate acceptance requires user inspection of these rendered movies.

Per run, the implementation adds zero H3 NFE, provider calls, VAE calls,
sampler lifetimes or history boundaries. It adds default-off bounded GPU
indexing/norm reductions, blocking CPU copies, hashes and file writes. The
32 MiB bound covers retained CPU tensor payload, not total process/serialization
RSS or transient device workspace. Diagnostic timing spans include intervening
actual block computation and synchronization; they are not a pure production
overhead measurement. Compare cold/warm, observer-disabled primed timing and
actual allocated/reserved VRAM separately. Offline witness replay and full-file
checkpoint hashing are counted diagnostic costs. No SM120 cost or speedup is
measured in this environment.

## Executed validation and open gates

CPU environment: Torch2.14.0+cpu, no CUDA. Results:

| Check | Result |
|---|---|
| Flow full tests with pinned Flow Core | 612 passed; later added stage-copy regression passes in targeted checks |
| VDN full tests with pinned Core/OpenVDN | 224 passed, including official linear/hybrid oracle; final receipt correction: 34 affected tests passed |
| Sol partitioned/mapped/history source tests | 38 passed after installing kernel import dependencies; no kernel execution |
| Flow native mixed-grid/decode with current Core | 39 passed after fixture compatibility change |
| Flow native mixed-grid/decode with pinned Core | 39 passed |
| Native/sibling source-contract scripts | passed at their documented source pins |
| Partitioned Flow/Sol/VDN contracts | passed, including bounded witness transport and unchanged provider-v4 geometry |
| VDN current-Core node/compiler smoke | passed |
| Flow Ruff/format; VDN CI-selected Ruff; compileall | passed |

Completed gates now include the actual SM120 feature witness, matched A/B
suppression discriminator, 00722 candidate-C execution, 00724 complete
partitioned learned-linear bypass, 00726 direct same-grid control and 00727
same-grid clean-ownership validation. The same-grid arms materially reduce the
rendered defect, proving the low->high spatial transition is a major contributor,
but a very slight residual shift remains even after identity transfer and zero
DC mutation. The next open gate is the same-domain weighted-dense suffix
attention discriminator described above. Audio remains independently unresolved.
No first-pair metric, source test or green CI can substitute for rendered
whole-window acceptance.

Candidate C is retained only as rejected diagnostic evidence in VDN #35. For
subsequent discriminators keep #35 installed but return the Flow selector to
`native_grid_then_map_v1` so the candidate arithmetic is inert and source
topology remains frozen. Full rollback can remove #35 and the #34/#93 diagnostic
overlays to the frozen #89/#33 baseline. Sol/Continuum stay unchanged.
Production consolidation into #89 and any release remain unqualified.
