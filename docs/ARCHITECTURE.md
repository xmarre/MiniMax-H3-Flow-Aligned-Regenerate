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
audio shape, layout and conditioning signatures, storage policy, timestamps, and samples. Each
sample records unshifted coordinate, both stream sigmas, outer/call index, phase, actual/forecast
provenance, and video denoised estimate.

Capture wraps ComfyUI `PREDICT_NOISE`; current ComfyUI's sampling function returns the CFG-combined
denoised estimate at that wrapper point, not raw H3 velocity. The flow PREDICT wrapper is deliberately
inside Spectrum's PREDICT wrapper, so Spectrum's call-local copied model options carrying
`spectrum_h3_actual`, solver phase, and outer-step identity reach capture before Spectrum finalizes the
step. Failed outer sampling records a bounded incomplete diagnostic run. Only complete runs are
selectable. CPU storage detaches, copies as fp32, and pins when CUDA is available; VRAM storage clones.

Capture is an explicit binding capability rather than a side effect of holding a trajectory handle.
The capture node and progressive handoff enable writes; the two-pass regenerate node is read-only so
a guided target pass cannot become the next source trajectory by accident. Forecast calls advance
topology but are stored only when explicitly requested. Exact anchors drive guidance. At duplicate
SA-Solver-PECE coordinates, corrected exact endpoints take precedence over predicted endpoints; a
dedicated handoff probe takes precedence at the split coordinate.

## Two-pass flow guidance

HiFlow's published initialization alignment changes the high-resolution sampler state before the
guided trajectory begins. This package does not label a PREDICT_NOISE correction as initialization
alignment. The two-pass node therefore exposes only direction, the experimental acceleration proxy,
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
when a refine pass begins at low sigma. A per-sample RMS bound limits the correction. The optional
acceleration term compares adjacent low/high denoised-estimate changes and
stays inactive until prior exact state exists. This is an experimental first-difference proxy; it does
not reproduce HiFlow's acceleration formulation, which is derived from changes in the reference-flow
velocity field. Downsample consistency forms the low-grid residual before lifting it.

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

Audio's packed sampler slice is copied exactly. Target shape metadata is mutated in place so ComfyUI's
final unpack/callback uses new geometry. Before each independent low/probe/high outer invocation,
raw conditioning is reconstructed from `guider.original_conds`; current ComfyUI can then resolve
percentage areas, masks, and other shape-dependent conditions against that stage's live geometry.
H3 rebuilds layout and MM-RoPE when the signature changes.
An isolated CPU generator, seeded by graph seed plus a fixed domain offset, makes retries reproducible
without perturbing sampler RNG.

`auto_compute` is a deterministic, bounded handoff heuristic that moves the base coordinate later
as the target/source area ratio grows. It uses no model calls and is reported in metrics. It is a
compute-allocation probe, not a learned quality criterion; fixed `0.35` remains the reproducible
benchmark setting until decoded runs establish a better policy.

### Reset and exact-anchor contracts

- **SA/PECE/SEEDS:** separate sampler invocations prevent old-grid history crossing.
- **Spectrum:** separate outer executions end the low runtime and create the high runtime.
  `h3_refinement` API v1 supplies the full-trajectory sigma reference; the high stage requests
  an actual prefix.
- **External patches:** exact calls traverse the normal guider/model patch chain. The probe also
  publishes the full-trajectory refinement reference so sigma-sensitive patches such as DiffAid
  do not renormalize the split coordinate to the probe invocation's first sigma.
- **Continuum:** selection includes session/chunk identity and a bounded content fingerprint of
  raw conditioning. The fingerprint samples deterministic tensor positions rather than reading
  entire Qwen/reference tensors back from the GPU.
- **Failure:** low capture aborts; guidance state clears in `finally`.
- **Denoise masks:** progressive mode fails closed. Current ComfyUI inpainting semantics combine the
  mask with a preserved sampler `latent_image`; transforming the mask without carrying that latent
  state to the new spatial grid would inject the wrong preserved-region values.

## Reference budget

Qwen3-VL context and direct H3 reference rows are separate. Native mode returns the exact input.
Experimental direct decoupling shallow-copies conditioning and caps each direct video latent by patch
rows while preserving aspect ratio and even H/W. Already encoded Qwen tokens remain unchanged.

## Attention lab

Diagnostics sample selected heads/queries and report entropy and modality mass without retaining full
matrices. Experimental sparse layers keep text/reference/audio global. Video queries retain non-video
keys and all temporal positions inside a spatial patch window; configured heads remain fully global.
Query chunking avoids a sequence-square mask. RoPE is untouched and block replacements are chained.
If another optimized attention override exists, the experiment records a fallback and delegates to it.
