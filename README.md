# MiniMax H3 Flow-Aligned Regenerate

Research-grade ComfyUI nodes for H3-native coarse-to-fine generation: transactional
low-resolution trajectory capture, time-aligned high-resolution guidance, and a true
in-flight spatial-resolution handoff.

> [!IMPORTANT]
> This project is an independent, training-free approximation informed by public
> research. It does **not** reproduce MiniMax's closed H3-Regenerate-2K model or its
> unreleased sparse-attention topology. Every semantics-changing mode is experimental
> until matched decoded-media validation is complete.

## Why this exists

In the motivating workflow, native H3 generation near 0.8 MP was visibly weaker than
generation on a 54x40 video-latent grid (864x640 pixels), followed by a learned latent
upscale and low-sigma H3 refinement on a 64x48 grid (1024x768 pixels). The latter repeats
part of the denoising trajectory and costs extra H3 evaluations.

This package tests three narrower hypotheses:

1. an additional resolution-dependent flow coordinate may improve native high-resolution scheduling;
2. intermediate low-resolution clean-state estimates may guide a high-resolution pass;
3. early low-resolution steps can transition grids without replaying the late trajectory.

H3 is treated as its actual joint AV model: 24-channel video, 32-channel two-track audio,
16x spatial VAE compression, a `(1, 2, 2)` DiT patch, video/audio shifts 12/3, and a packed
`text | conditioning/reference | audio | video` transformer sequence.

## Status

The architecture, guards, instrumentation, and synthetic test suite are implemented.
Decoded-media comparison is still required before recommending non-native settings. The
safe default for the schedule, reference budget, and attention lab is `off`/`native`.

## Install

From `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate.git
```

Restart ComfyUI. PyTorch is supplied by ComfyUI; no sibling custom node is required to
import this package. The learned LBH upscaler remains an optional, separately installed
component and is used upstream when desired.

## Nodes

| Node | Purpose | Default posture |
|---|---|---|
| MiniMax H3 Flow Trajectory | Explicit `H3_FLOW_TRAJECTORY` handle with RAM/VRAM policy | RAM |
| MiniMax H3 Trajectory Capture | Transactionally records exact denoised estimates | Stable infrastructure |
| MiniMax H3 Flow-Aligned Regenerate | Time-matched low-frequency guidance for an explicit second-pass MODEL | Conservative direction term |
| MiniMax H3 Flow-Aligned Refine State | Patches Continuum V3.4 per-chunk refine_state for the integrated learned-upscale/refine node | Conservative direction term |
| MiniMax H3 Progressive Handoff | Low-grid input grows to the target grid during one full schedule | Experimental |
| MiniMax H3 Progressive Handoff (Target Input) | Keeps the workflow/session on the final grid but runs the early H3 stage internally on a smaller grid | Experimental; Continuum-compatible topology |
| MiniMax H3 Resolution-Aware Sigmas | Composed resolution/H3 flow shift | Off |
| MiniMax H3 Reference Budget | Token diagnostics and direct-reference-only cap | Native |
| MiniMax H3 Attention Lab | Sampled diagnostics and guarded H3-local attention | Native |
| MiniMax H3 Metrics JSON | Structured counters/events for benchmark capture | On demand |

### Two-pass flow-aligned use

1. Create one **Flow Trajectory** handle.
2. Patch the low-resolution H3 model with **Trajectory Capture** and run the base sample.
3. Perform the existing learned latent upscale, or another explicit initialization step.
4. Patch the high-resolution H3 model with **Flow-Aligned Regenerate**, using the same trajectory.
5. Keep pass-one audio locked in the surrounding workflow.

The high-resolution model call is guided only in low spatial frequencies. Its control schedule is
normalized to the first coordinate of that high-resolution invocation, so low-sigma refinement starts
at the configured guidance strength and then weakens toward zero. Exact H3 model evaluations are
preferred as trajectory anchors; Spectrum forecasts retain their provenance
and are excluded from trustworthy anchors by default. The regenerate patch reads the supplied
trajectory but does not append the high-resolution pass back into it, so the low-resolution
source run cannot be silently replaced by its own guided output. HiFlow-style initialization
alignment is not exposed as a two-pass guidance mode: its published operation changes the
high-resolution sampler's starting state, whereas this node modifies denoised predictions.
Use the explicit upstream initialization/refine step or the progressive handoff path instead.

### Continuum V3.4 integrated-refine use

Keep **Flow Trajectory → max_runs** at least as large as the number of physical chunks that will be refined; the node default is 16 to cover Continuum's current maximum sequence length. Downstream refinement happens after the base sampler has emitted its chunk list, so a smaller history can evict early chunk trajectories before their matching refine item executes.

For the current integrated **MiniMax H3 Latent Upscaler + Refine (3D)** path, do not look for a
MODEL output from the learned upscaler. Enable Continuum's `emit_refine_conditioning`, pass each
chunk's `refine_state` through **Flow-Aligned Refine State**, and connect that patched state to the
integrated upscaler/refine node. Patch the MODEL entering Continuum with **Trajectory Capture**
after the normal DiffAid/Untwist/Spectrum patches. The refine-state adapter preserves the exact
chunk positive conditioning and any opaque refine-state fields while changing only its fresh
per-chunk MODEL clone. Continuum V3.4 carries native denoise masks on the parallel video/audio
LATENT outputs, not inside `refine_state`; those LATENT wires remain connected directly to the
integrated upscaler/refine node.

Target-grid `minimax_keyframes` are expected to be spatially resized by the learned-refine node.
The refine-state adapter therefore records the sampler-1 conditioning signature before that resize
and binds sampler 2 to it explicitly. There is no keyframe-specific geometry exemption in the
trajectory identity check: keyframe latents/shape, Qwen/context tensors, and independent
`minimax_refs` all pass through the same bounded deterministic content fingerprint. The fingerprint
samples tensor positions instead of reading every tensor byte back from the accelerator, so it is a
practical drift fence rather than a cryptographic proof of full tensor equality.

### Progressive handoff use

Both progressive nodes require a complete H3 1-to-0 sigma schedule, split it at the closest valid
unshifted coordinate, invoke the selected native sampler for the low stage, perform one explicit H3
clean-state probe, create the target-grid conditional state, and start a fresh sampler invocation on
the high grid. Partial low-sigma refinement schedules are rejected because their absolute flow origin
is ambiguous for a progressive split.

Use **Progressive Handoff** when the sampler input itself is the low grid and changing the workflow's
output latent geometry is acceptable. Use **Progressive Handoff (Target Input)** for Continuum V3.4:
configure Continuum at the final 1024x768 target, set the progressive source to 864x640, and patch the
MODEL entering Continuum. The wrapper derives a fresh low-grid video noise tensor, downsizes the
target-grid latent/mask/keyframes for the low and probe stages, then returns to the untouched target
latent/mask/keyframe contract for the high stage. In masked target-input mode, protected regions also
keep Continuum's original target-grid noise because native H3 uses that noise in its 0.999 visual
conditioning injection. Continuum therefore always receives and stores
64x48 output chunks, so native masked continuation and session geometry remain target-sized across
multiple chunks.

The split invocation is the reset contract: SA/PECE Adams history and Spectrum hidden-feature
history cannot cross the geometry boundary. The exact probe and high stage advertise
`h3_refinement` API v1 with a full-schedule sigma reference; the high stage additionally requests
a mandatory actual prefix. Audio is never spatially transformed. In target-input mode the original
audio noise/latent/mask are preserved while only the early video grid is reduced. Handoff and
low-grid source noise use isolated documented CPU generators derived from the graph seed.

All source/target latent H/W are normalized to even values. H3's native circular padding is never
used as an alignment mechanism; this prevents the known odd-grid flashing-border failure.

## Compatibility

- **ComfyUI:** targets current native H3 and fails closed if the 24/32-channel, `(1,2,2)`,
  shift-12/3 contract is absent.
- **SA-Solver/PECE, SEEDS, ER-SDE, Euler/RES:** downstream sampler objects are preserved. The
  progressive split creates independent solver lifetimes. Unclassified calls remain explicit.
- **Spectrum:** no import dependency. Actual/forecast, solver-phase, and outer-step metadata are
  consumed when present. A new downstream outer-sample execution is used per geometry.
- **Continuum:** trajectory selection uses the available chunk identity plus the bounded conditioning
  signature (and a session namespace when one is present). Current Continuum V3.4 interop does not
  expose a unique sequence/session ID, so use a separate trajectory handle for each independent
  Continuum sequence. Conditioning identity adds a strong drift/cross-talk fence but cannot uniquely
  distinguish two independent sequences whose relevant conditioning fingerprints are identical.
  Multi-chunk progressive use must use the Target Input node so Continuum's latent/session geometry
  remains on the final grid.
- **DiffAid / Untwisting RoPE:** patches stay in the downstream path. The `h3_refinement` exact-anchor
  contract is published during the high stage.
- **Other attention overrides:** experimental sparse attention delegates to the existing provider.

## Experimental controls

`direction+acceleration` is an experimental first-difference trajectory proxy. It needs adjacent exact
trajectory support and is bounded by an RMS guard; it is not an implementation of HiFlow's published
acceleration-alignment equations.
`downsample_consistency` is an independent alternative. The direct reference cap changes only H3's
direct latent-reference rows; already encoded Qwen3-VL tokens are measured and left unchanged.
Sparse attention retains global text/reference/audio paths and global temporal video reach, but is
an investigative implementation rather than a reconstruction of MiniMax internals. Resolution-aware
sigmas move H3's shared AV flow coordinate, so they also change the derived audio sigma schedule;
the probe is not a video-only schedule modifier and requires decoded-audio validation.

## Metrics and benchmarking

Runtime events distinguish H3 transformer calls from Spectrum forecasts and record sigma, the
unshifted coordinate, sampler topology, packed layout rows, trajectory transactions, guidance,
handoff decisions, exact probes, reset/re-anchor evidence, and attention fallbacks. Use the
[benchmark protocol](docs/BENCHMARKS.md) and [machine-readable matrix](workflows/benchmark-matrix.json)
for matched runs. The `workflows/` directory also includes two-pass and progressive graph overlays.
Decoded video and audio—not preview frames or logs—are the quality gate.

## Development

```bash
python -m pip install -e '.[test]'
ruff check .
ruff format --check .
pytest
python -m compileall -q .
python -m build
```

See [architecture and derivations](docs/ARCHITECTURE.md), [research transfer notes](docs/RESEARCH.md),
and [benchmark instructions](docs/BENCHMARKS.md).

## Limitations

- No decoded-media claim is made yet.
- The progressive probe intentionally costs one visible exact H3 NFE at the transition.
- Target-input progressive mode intentionally derives its private low-grid **video** noise from a
  documented standard-Gaussian CPU generator keyed by the graph seed; it cannot preserve the semantics
  of an arbitrary custom/non-Gaussian video-noise node at that private grid. The caller's audio noise
  and target-grid protected-region noise remain preserved.
- Progressive handoff carries ComfyUI denoise-mask semantics across the grid transition by resizing
  ComfyUI's prepared packed video mask, preserving the audio mask, spatially transferring the sampler
  `latent_image`, and reconstructing the exact noise argument against that preserved latent state.
  This path is synthetic-tested but still requires real masked Continuum media validation.
- Reference decoupling cannot retroactively change Qwen3-VL tokens.
- The sparse path uses bounded query chunks and may not improve wall time on every backend.
- Model assets, the optional learned upscaler, and sibling custom nodes are not bundled.
