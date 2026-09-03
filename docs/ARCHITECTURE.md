# Architecture and derivations

## Verified native H3 contract

Current ComfyUI implements MiniMax H3 as a joint audio/video flow model. Video is
`B x 24 x T x H x W`; audio is `B x 32 x 2 x Ta`. The sampler flattens both streams into
one state and scales audio by `12/3 = 4` so the packed state follows video sigma. Inside
the DiT, audio sigma is reconstructed through the common unshifted coordinate.

The DiT patch is `(1,2,2)`, so video latent H/W must be even. The VAE spatial factor is 16.
A pixel dimension aligned to 32 maps cleanly through VAE and DiT patching. The transformer packs
text, keyframe/direct reference rows, audio, then video, and derives area-normalized MM-RoPE
coordinates from the current layout.

## Flow coordinates and joint AV law

Define

\[
f_s(t)=\frac{s t}{1+(s-1)t},\qquad
f_s^{-1}(\sigma)=\frac{\sigma}{s+(1-s)\sigma}.
\]

H3 video uses `sigma_v = f_12(t)` and derives

\[
\sigma_a=f_3(f_{12}^{-1}(\sigma_v)).
\]

Trajectory matching and handoff selection use `t = f_12^-1(sigma_v)`. This retains one
full-trajectory coordinate for both streams and avoids matching different samplers by step index.

### Resolution-aware schedule probe

SD3 derives `alpha = sqrt(m/n)` for target/source observations under a constant-image assumption
and uses `f_alpha(t)`. That assumption does not establish an optimal H3 video mapping, so the node
is off by default. Since `f_a(f_b(t)) = f_ab(t)`, an already H3-shifted schedule maps as

\[
\sigma'_v=f_{12}(f_\alpha(f_{12}^{-1}(\sigma_v))).
\]

H3 still inverts video to the new base coordinate and applies `f_3` for audio. Therefore this
SIGMAS-only experiment moves the shared AV coordinate and changes the derived audio sigma schedule
as well; it is not a video-only schedule modifier. Analytic tests cover endpoints, monotonicity,
inverse/composition behavior, and AV invariants, while decoded audio remains part of the media gate.

## Transactional trajectory schema

`H3_FLOW_TRAJECTORY` owns immutable committed runs and at most one pending transaction. A run has
API version, run/session/chunk IDs, sampler and schedule signatures, latent/pixel/padded geometry,
audio shape, AV latent-layout and conditioning signatures, storage policy, timestamps, and samples. Each
sample records unshifted coordinate, both stream sigmas, outer/call index, phase, actual/forecast
provenance, and video denoised estimate.

Capture wraps ComfyUI `PREDICT_NOISE`; current ComfyUI's sampling function returns the CFG-combined
denoised estimate at that wrapper point, not raw H3 velocity. The flow PREDICT wrapper is deliberately
inside Spectrum's PREDICT wrapper, so Spectrum's call-local copied model options carrying
`spectrum_h3_actual`, solver phase, and outer-step identity reach capture before Spectrum finalizes the
step. Failed outer sampling records a bounded incomplete diagnostic run. Only complete runs are
selectable. CPU storage detaches, copies as fp32, and pins when CUDA is available; VRAM storage clones.

Trajectory selection is newest-attempt strict within the requested session/chunk/conditioning identity:
an aborted or invalidated run for that identity blocks fallback to an older successful run. Conditioning
is part of selection rather than only a post-selection assertion. Current Continuum interop metadata
does not expose a unique sequence/session identifier, so independent Continuum sequences should use
separate trajectory handles. The conditioning filter prevents ordinary same-index cross-talk when their
fingerprints differ, but it cannot disambiguate two independent sequences with identical relevant
conditioning fingerprints. Newest-attempt strictness also prevents a failed regeneration attempt from
silently reactivating stale low-resolution guidance state.

A sampler may finish while a diagnostic trajectory contains only Spectrum forecasts. Such a run remains
recorded as complete sampling telemetry, but it is not selectable as a guidance source: selection requires
at least one provenance=`actual` anchor and fails before the target sampler begins otherwise.

ComfyUI's `ModelPatcher.clone()` recursively copies dict/list containers but leaves arbitrary objects
shared by reference. The flow binding therefore installs an `ON_CLONE` callback: clones share only the
intentional trajectory handle, immutable guidance config, and metrics sink, while receiving fresh
capture transaction, selected-guidance-run, and guidance-state fields. This prevents downstream model
branching (including Continuum's per-chunk MODEL clones) from sharing mutable execution state.

Capture is an explicit binding capability rather than a side effect of holding a trajectory handle.
The capture node and progressive handoff enable writes; the two-pass regenerate node is read-only so
a guided target pass cannot become the next source trajectory by accident. Forecast calls advance
topology but are stored only when explicitly requested. Exact anchors drive guidance. At duplicate
SA-Solver-PECE coordinates, corrected exact endpoints take precedence over predicted endpoints; a
dedicated handoff probe takes precedence at the split coordinate.

Standalone two-pass workflows may legitimately rebuild target-grid conditioning between sampler 1
and sampler 2 (the learned H3 upscaler resizes target keyframes). The trajectory fingerprint is not
weakened to accommodate that geometry change. Instead, **Flow-Aligned Regenerate** accepts an optional
`source_conditioning` input and optional `source_negative` input, then stores the exact guider-key
shape (`positive`, and `negative` when supplied) in the bounded signature expected from sampler 1.
A negative without source positive is rejected. When source conditioning is absent, selection uses
sampler 2's live guider conditioning and therefore requires true conditioning parity. Continuum's
refine-state adapter uses a positive-only BasicGuider and obtains that source identity directly from
the exact sampler-1 positive conditioning captured in `H3_CONTINUUM_REFINE_STATE`.

## Two-pass flow guidance

HiFlow's published initialization alignment changes the high-resolution sampler state before the
guided trajectory begins. This package does not label a PREDICT_NOISE correction as initialization
alignment. The two-pass node therefore exposes direction, H3-adapted HiFlow velocity acceleration,
and downsample-consistency guidance; sampler initialization remains an explicit upstream workflow
operation. Progressive handoff uses the rectified-flow conditional-state law described below.

Exact low-grid clean estimates bracket and interpolate the high call's coordinate. A cheap spatial
map `U` transfers the reference. The conservative direction update is

\[
\hat x^{HR}_0 \leftarrow \hat x^{HR}_0 +
\lambda(t)L\left(U(\hat x^{LR}_0(t))-\hat x^{HR}_0\right),
\]

where `L` is an explicit average-pool low-frequency projection. The schedule weight is normalized
to the first coordinate of the current high-resolution invocation, i.e. `(t / tau)^p` clamped to
`[0,1]`, rather than to the full 1-to-0 trajectory. This preserves the intended weakening schedule
when a refine pass begins at low sigma. A per-sample RMS bound limits the combined correction.

For acceleration alignment, let \(x_t^{HR}\) be the current high-grid sampler state, let
\(\hat x_{0,i}^{HR}\) be the direction-corrected high-grid clean estimate, and let
\(\hat x_{0,i}^{R}=U(\hat x_{0,i}^{LR})\) be the time-matched transferred reference. H3's
`CONST.calculate_denoised` contract is \(\hat x_0=x_t-\sigma v\), so the two velocity fields are
reconstructed exactly in the sampler's shifted-sigma domain:

\[
v_i^{HR}=(x_t^{HR}-\hat x_{0,i}^{HR})/\sigma_i,\qquad
v_i^{R}=(x_t^{HR}-\hat x_{0,i}^{R})/\sigma_i.
\]

At a new coordinate the HiFlow acceleration update is then adapted directly as

\[
v_i^{HR}\leftarrow v_i^{HR}+\beta_i
\left(v_i^{R}-v_{i-1}^{R}-v_i^{HR}+v_{i-1}^{HR}\right),
\]

followed by \(\hat x_{0,i}^{HR}=x_t^{HR}-\sigma_i v_i^{HR}\). Unlike direction guidance, this
acceleration term is deliberately **not** low-pass filtered: HiFlow uses velocity acceleration to
recover detail fidelity. The schedule uses the same normalized \((t/\tau)^p\) decay. SA-Solver-PECE evaluates a predictor
and corrector at the same coordinate. Both are estimates of the current-time velocity, so both are
aligned against the **previous distinct-coordinate** velocity pair; the corrector then replaces the
current endpoint used as history at the next coordinate. Treating the predictor as the previous
timestep would incorrectly collapse the corrector's acceleration interval to zero. The final returned
correction, including direction plus acceleration, remains subject to the RMS guard. Guidance telemetry
separates direction and acceleration RMS ratios and records whether a call is exact or a Spectrum
forecast so this interaction remains auditable.

### Video-time correspondence guidance

The `direction+temporal` experiment is intentionally separate from denoising-time acceleration.
For the resolved exact low-grid clean estimate \(R\), adjacent video-latent frames are lightly
spatially smoothed and matched inside a bounded local search window using channel-normalized cosine
similarity. Both \(i\rightarrow i-1\) and \(i-1\rightarrow i\) maps are computed.

The first decoded-media smoke exposed an important ambiguity failure in repetitive texture. High
absolute cosine similarity alone was not discriminative: grass matches had ~0.94-0.96 similarity but
mean best-vs-second-best margins only ~0.0027-0.0038. The original implementation treated
`temporal_min_margin` as a soft scaling term, so those ambiguous matches still transported detail.
The result was visible repeated/patterned grass and worse motion artifacts.

The current matcher therefore treats its confidence conditions as **admission gates**, not suggestions.
A candidate must meet the configured minimum similarity and minimum uniqueness margin, and the
forward/backward integer correspondence must satisfy the configured cycle tolerance. The default
cycle tolerance is now zero latent cells. Rejected candidates contribute exactly zero temporal
correction.

For a trusted mapping from target frame \(i\) to neighbor \(j\), let \(W\) warp along that low-grid
correspondence and let \(U\) transfer a low-grid residual to the target grid. The high-grid temporal
target is

\[
P_i = W(\hat x^{HR}_{0,j}) + U\left(R_i - W(R_j)\right).
\]

Thus the neighbor contributes high-resolution detail while the low-resolution H3 prior supplies the
same-reference structural innovation. The correction \(P_i-\hat x^{HR}_{0,i}\) is confidence weighted.
Failed/ambiguous/cycle-inconsistent matches contribute zero instead of copying stale or repetitive
content into regions treated as disoccluded.

Previous- and next-frame estimates are accumulated symmetrically. When their summed confidence is
below one, the correction remains attenuated; above one, the two estimates are normalized rather than
double-counted. The result is non-recursive within a model call: warped neighbors come from the
original current clean-state estimate, not from an already temporally corrected frame.

Progressive handoff imposes one additional constraint. The low-grid capture stops at the exact handoff
probe. Once the high-stage coordinate advances below the final low-grid anchor, there is no later
low-grid trajectory sample to interpolate. Reference lookup therefore clamps to that final exact anchor
(~0.4 in the D10 smoke). This is explicit rather than hidden: temporal cache identity is keyed by the
**resolved reference coordinate**, so predictor/corrector calls at later requested coordinates reuse the
same correspondence map instead of recomputing an identical one. Telemetry reports both
`temporal_reference_coordinate` and `temporal_reference_clamped`.

The ordinary direction term keeps its existing decaying high-stage schedule. Temporal transport is
therefore still strongest at handoff and weakens toward the end even when its low-grid correspondence
map is handoff-anchored.

Correspondence is computed only from the captured H3 video clean-state latent. No external optical-flow
network is loaded, audio is untouched, and no packed non-video tokens participate. The combined
direction+temporal correction still passes through the same per-sample RMS guard.

Telemetry exposes temporal RMS ratio, mean confidence, strict valid/disocclusion fractions, similarity
and uniqueness-margin statistics, low-grid flow magnitude, cache-hit state, resolved reference
coordinate, and clamp state. Structural telemetry can show that the mechanism is active and conservative;
only decoded-media comparison establishes quality.

Downsample consistency forms the low-grid residual before lifting it.

### Continuum integrated-refine adapter

The public trajectory node returns an always-changed Comfy cache fingerprint because its output is mutable execution state. Each queued prompt therefore receives a fresh handle, while all nodes inside that prompt still share the same object. This prevents old completed/aborted runs and accumulated metrics from leaking into later prompt executions through Comfy's intermediate-node cache.

The trajectory history is a delayed-consumer buffer in this topology: Continuum samples its base chunk list before downstream list-mapped refinement consumes the matching runs. `max_runs` therefore must be at least the number of physical refine items; the public node default is 16, matching Continuum's current maximum configured chunk count. Eviction remains bounded and deterministic. Initial validation keeps Continuum Run Storage off because cache reuse can return a previously sampled chunk without executing the current capture wrapper, leaving no fresh trajectory transaction corresponding to that refine item.

Current Continuum V3.4 internally captures MODEL, positive conditioning, and the sampler-boundary
`noise_mask` for every freshly sampled chunk. Its public `H3_CONTINUUM_REFINE_STATE` deliberately
emits only API + a fresh per-chunk MODEL + exact positive conditioning; the captured mask is split and
reattached to the parallel video/audio LATENT outputs instead. The learned H3 upscaler/refine consumes
MODEL + positive from the refine state and derives its refinement `denoise_mask` from the LATENT
input. There is therefore no external high-resolution MODEL wire to patch, and the LATENT mask path
must remain intact. The **Flow-Aligned Refine State** node shallow-copies the public state, patches only
its MODEL with read-only trajectory guidance, and preserves positive plus any future/opaque fields.

The learned refine path legitimately resizes target-grid `minimax_keyframes` before sampler 2.
The adapter computes the same content signature as CFGGuider conversion from the exact sampler-1
`positive` object and stores it on the read-only guidance binding. Sampler 2 is checked against that
source signature rather than against its resized conditioning. There is no keyframe-specific
geometry/content exemption: keyframe tensors/shape, Qwen/context tensors, and independent
`minimax_refs` all pass through the same bounded deterministic fingerprint. That fingerprint samples
tensor positions to avoid full accelerator readback; it is intended to reject ordinary conditioning
drift, not to prove byte-for-byte tensor identity.

## Progressive handoff law

The flow wrapper is first in ComfyUI's outer-sample chain. Progressive mode accepts only a complete
1-to-0 H3 sigma schedule and invokes the downstream chain three times, creating fresh Spectrum and
multistep lifetimes:

1. low-grid native sampler over the early schedule;
2. one exact denoised probe at the nonterminal handoff sigma;
3. high-grid native sampler over the remaining schedule.

ComfyUI's flow sampler returns an inverse-scaled state at nonterminal sigma. The wrapper reverses
that output transform to recover packed `x_sigma`. The probe feeds that exact state through the normal
guider/model path and compensates terminal inverse scaling, yielding `x0_hat` with one visible NFE.

The target video state uses the rectified-flow conditional law

\[
x^{HR}_\sigma=(1-\sigma)U(\hat x^{LR}_0)+\sigma\epsilon^{HR},
\qquad \epsilon^{HR}\sim\mathcal N(0,I).
\]

This supplies high-frequency stochastic degrees of freedom without an arbitrary texture coefficient.
It is a training-free conditional state, not RALU's nearest-neighbor correlated-noise formula, whose
specific assumptions do not transfer unchanged to arbitrary H3 geometry and packed AV state.

Audio's packed sampler slice is copied exactly. Source-input mode mutates the caller's final shape
metadata at the handoff. Target-input mode instead starts with final-grid metadata, creates a private
low-grid AV shape for the early invocation, and returns to the caller's original final-grid shape for
the high invocation. This second topology is required for Continuum: its session and native-masked
continuation contracts validate spatial geometry across chunks and therefore must never observe a
54x40 output chunk when the sequence is configured for 64x48.

Caller callback geometry is also fenced at the target-input boundary. ComfyUI creates its packed AV
callback closure against the caller-visible target shapes before OUTER_SAMPLE wrappers execute, so
private low-stage callback tensors are spatially lifted back to the target video grid before that
closure is invoked. This keeps previews and any callback consumer from trying to unpack a private
source-grid tensor with target-grid metadata.

Before each independent low/probe/high inner sampling invocation, conditioning is reconstructed from
a pristine snapshot of `guider.conds` taken at progressive-wrapper entry. Current ComfyUI has already
run `preprocess_conds_hooks` and `filter_registered_hooks_on_conds` before entering OUTER_SAMPLE, so
rebuilding directly from `guider.original_conds` here would silently discard ControlNet/registered-hook
preprocessing. The preserved snapshot is still pre-`process_conds`, allowing each geometry lifetime to
resolve percentage areas, masks, and other
shape-dependent conditions against that stage's live geometry. Source-input mode resizes target
keyframes on the high stage. Target-input mode does the inverse: it temporarily downsizes
`minimax_keyframes` for low/probe and restores the original target-grid conditioning for high.
Independent `minimax_refs` retain their own geometry. H3 rebuilds layout and MM-RoPE when the
signature changes.

Target-input mode also cannot reuse the caller's target-grid video noise directly as a lower-grid
Gaussian sample without changing its distribution. It therefore derives a separate deterministic
standard-Gaussian low-grid video tensor from the graph seed while preserving the caller's audio noise.
The handoff high-frequency video noise uses a separate domain offset. Both generators are isolated
from the sampler RNG.

`auto_compute` is a deterministic, bounded handoff heuristic that moves the base coordinate later
as the target/source area ratio grows. It uses no model calls and is reported in metrics. It is a
compute-allocation probe, not a learned quality criterion; fixed `0.35` remains the reproducible
benchmark setting until decoded runs establish a better policy.

Progressive model-call telemetry is stage-scoped. The live transformer options carry a temporary
`h3_flow_stage` marker for `low`, `probe`, and `high`; because the flow PREDICT wrapper is
inside Spectrum, Spectrum's call-local model-options copy preserves that marker alongside actual/
forecast provenance. Counters therefore expose low/probe/high logical calls and actual/forecast
counts separately. The wrapper also counts the three downstream sampler invocations and the two
fresh-lifetime boundaries instead of reporting an unverified solver-reset boolean.

### Reset and exact-anchor contracts

- **SA/PECE/SEEDS:** separate sampler invocations prevent old-grid history crossing.
- **Inpaint sampler options:** the exact probe preserves the selected sampler's KSampler-level
  `inpaint_options` (including deterministic `random` inpaint noise), so masked-state evaluation
  uses the same native inpaint policy as the low stage rather than silently reverting to defaults.
- **Spectrum:** separate outer executions end the low runtime and create the high runtime.
  Explicit Spectrum provenance is authoritative: an exact handoff probe that is reported as a forecast
  fails closed rather than being relabeled as an actual anchor.
  `h3_refinement` API v1 supplies the full-trajectory sigma reference; the high stage requests
  an actual prefix.
- **External patches:** exact calls traverse the normal guider/model patch chain. A pre-existing
  `h3_refinement` dictionary is preserved when compatible, including opaque provider fields; a
  conflicting API/prefix/sigma/source contract fails closed rather than being overwritten. The probe also
  publishes the full-trajectory refinement reference so sigma-sensitive patches such as DiffAid
  do not renormalize the split coordinate to the probe invocation's first sigma.
- **Continuum:** selection includes the available session/chunk namespace and a bounded content
  fingerprint of raw conditioning. Current V3.4 interop exposes chunk index but no unique sequence
  session ID; independent sequences should therefore use separate trajectory handles. Conditioning
  identity additionally separates same-index runs when their fingerprints differ. The fingerprint
  samples deterministic tensor positions rather than reading entire
  Qwen/reference tensors back from the GPU. Multi-chunk progressive operation uses the
  target-input topology so Continuum's output/session/native-mask geometry stays final-sized.
- **Failure:** low/probe failure aborts the active trajectory transaction. The low run must be committed
  before same-invocation high guidance can select it; if transfer or the high stage then fails, that
  committed run is immediately invalidated and becomes ineligible for future trajectory selection.
  Guidance state clears in `finally`.
- **Denoise masks:** current ComfyUI prepares each user mask to full latent-channel AV geometry before
  OUTER_SAMPLE wrappers run. Progressive mode resizes only the prepared video portion, preserves the
  audio portion, spatially transfers the external latent image, converts it through H3's normal latent-in
  transform, and solves `noise = (x_sigma - (1-sigma) * latent_image) / (sigma * noise_scale)` so
  the next sampler invocation starts from the exact carried state. In target-input mode, mask-protected
  regions retain the caller's original target-grid sampler noise because MiniMax H3's
  `scale_latent_inpaint` explicitly mixes that noise into preserved video context at timestep 0.999.

## Reference budget

Qwen3-VL context and direct H3 reference rows are separate. Native mode returns the exact input.
Experimental direct decoupling shallow-copies conditioning and caps each direct video latent by patch
rows while preserving aspect ratio and even H/W. Already encoded Qwen tokens remain unchanged.

Direct-reference budgeting updates the resized reference latent and its native `latent_h`/`latent_w`
(and existing `latent_t`) metadata atomically. MiniMax H3's `PackedLayout` consumes those metadata
fields to allocate reference rows and positions, so leaving source-grid metadata attached to a resized
latent would make the packed reference contract inconsistent.

## Attention lab

Diagnostics sample selected heads/queries and report entropy and modality mass without retaining full
matrices. Experimental sparse layers keep text/reference/audio global. Video queries retain non-video
keys and all temporal positions inside a spatial patch window; configured heads remain fully global.
Query-local restrictions are passed to ComfyUI attention backends as numeric QxK additive masks,
because Core's boolean-mask path is a batch/key mask and cannot represent per-query sparsity.
Unsupported mask backends fall back to native attention. Query chunking avoids a sequence-square mask.
RoPE is untouched and block replacements are chained. If another optimized attention override exists,
the experiment records a fallback and delegates to it.
