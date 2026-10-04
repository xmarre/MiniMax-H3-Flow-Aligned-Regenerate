# Generated boundary motion, temporal guidance and throughput: 01041

01041 fails rendered boundary acceptance. The two supplied crops contain a
new detached hair strand and local shading/content changes. The decoded motion
receipt also contains a sharp third-successor jump. Identical decoded carried
overlap does not establish continuity of the newly generated suffix. This
capture predates the temporal-guidance and dynamic-compilation changes below.

## Frame mapping and motion

The initial chunk has 52 video latent tokens and decodes to 175 frames. The
continuation has 62 tokens, including a twelve-token/39-frame carried prefix.
Its raw 209 decoded frames retain indices `[39,209)` at global `[175,345)`.
Continuum assembly uses contiguous slices. Exact duration removes nine frames
from the final output tail to reach 336 frames; it does not remove a frame at
the join. The screenshots' 7 ms timestamp separation is insufficient to assign
them distinct 24 fps frame indices, so crop observations are not used as a
frame-number witness.

Flow decode context supplies five real future latents (17 decode-only frames)
to the preceding decode. All 22 sampled duplicate overlap frames agree exactly,
including the five-frame tail. Video Seam Auto applies no video patch. These
observations rule out assembly replacement and a retained-overlap mismatch in
this capture, but leave generated content and its decoding to be assessed.

| Measured vertical transition, decoded pixels | First successor | Second | Third | Pre-boundary median |
|---|---:|---:|---:|---:|
| Full frame | 2.061 | 4.817 | 14.238 | 2.363 |
| Upper 45% | 1.858 | 2.491 | 10.984 | 2.318 |

The receipt rescales its comparison-grid estimates into decoded pixel units.
The third transition is global frame 176 to 177. Its estimate is not clipped.
These phase-correlation values describe a translation fit, not a calibrated
confidence, optical-flow ground truth or proof of a literal skipped frame.
Immediate affine scale estimates are near one and have low confidence. They
do not certify that the garment or local image geometry remains unchanged.

The full-frame first latent transition has estimated dy 0.08567 before high
refinement, 0.87866 in the first actual high prediction before Flow guidance,
and 0.87866 after that guidance. Final high dy is 0.86934. The upper-region
first high prediction likewise has dy 0.85261. The larger displacement therefore
already exists before the direction/temporal correction in this realization.
Closing video overlap reaches exact conditioning for the final two high calls
and restores the prefix, but does not recover the earlier generated transition.
No retired prediction bridge, registration warp or VAE repair is re-enabled.

## Tone and temporal comparison consistency

| First retained transition, whole-frame luma | Previous | Generated | Change |
|---|---:|---:|---:|
| Mean | 0.519843 | 0.522468 | +0.505% |
| Standard deviation | 0.249185 | 0.249912 | +0.292% |
| Fifth percentile | 0.110393 | 0.100362 | -9.086% |
| Ninety-fifth percentile | 0.873060 | 0.877530 | +0.512% |

The supplied crop mean falls 2.733%; its standard deviation falls 1.069%.
The darker lower tail and local crop support a remaining shading change.
They do not establish uniform contrast amplification. Global latent gradient
and centered low-pass boundary RMS decrease to about 0.920 of pre-high values;
those global measurements do not explain the new detached strand.

Registration rejects Euler's `unsupported_sampler_contract`; the coupled
four-token handoff `(1,0.75,0.5,0.25)` and all six rebased direction comparisons
execute. Temporal guidance still transports its high operand using native
source innovations. Comparing the rebased operand in that native representation
creates a correction even if the handoff already agrees with the reference.

A stationary textured counterexample with real temporal correspondence and
different source/target grids produces a 0.037315 maximum suffix error and a
-0.022378 first-suffix mean correction. Direction-only error is below 6e-8.
The prefix stays exactly equal in both cases. Flow now removes the existing
bounded representation residual from the temporary temporal operand before
native transport. The resulting correction is added to the reconciled operand.
Source motion correspondence, confidence, cache, reference tensors, protected
prefix and audio ownership retain their existing behavior. Coupled and DC-only
tests cover stationary and moving scenes plus cached later coordinates.

This corrects an operator invariant; it does not demonstrate that the first
high prediction's independent shift or the rendered hair/garment issue is fixed.
`guidance.temporal_reference_gauge_used` identifies an executed pullback on the
next capture. It is false for no usable correspondence or inactive temporal
guidance. No extra H3, provider or VAE evaluation is added.

## Throughput and executed source validation

| Sampling stage, seconds | Initial | Continuation | Increase |
|---|---:|---:|---:|
| Low | 74.593 | 137.994 | 63.401 |
| Exact probe | 10.270 | 19.631 | 9.361 |
| Transfer | 0.546 | 7.933 | 7.387 |
| High | 97.521 | 115.554 | 18.032 |
| Whole sampler | 182.950 | 282.973 | 100.023 |

The prompt takes 499.95 s, including about 34.027 s outside the two samplers.
Low/probe account for 72.762 s (72.7%) of the continuation increase. Source and
target grids are 36x54/50x76: continuation low video rows increase from 25,272
to 35,700 (+41.3%); high rows from 49,400 to 58,900 (+19.2%). There are still
22 actual H3 calls, twelve forecasts, six sampler scopes and four progressive
history boundaries. Memory admission contributes only 0.161 s at continuation
high, and at most 0.001 ms elsewhere.

The previous Sol request-owned source fix demonstrably executes. Continuation
low/probe each perform one full source check, taking 3.768/3.858 ms; other scopes
perform zero. Thus repeated source scanning no longer explains the penalty.
This capture's different grids and conditioning prevent a matched speedup
claim against earlier runs.

Sol's partitioned helper still keys compiled functions by concrete row counts
and Q/K/V strides even though its packaged converter marks these layout values
dynamic. Sol #37 now keys the machine function by the actual static tensor ABI,
device/architecture and bias/map specialization. All physical-layout and
semantic arithmetic gate keys remain unchanged. The converter and vendor
kernel bytes remain unchanged. See
[dynamic compilation ownership](https://github.com/xmarre/ComfyUI-Sol-H3/blob/fix/canonical-same-grid-history-20261001/docs/PARTITIONED_DYNAMIC_COMPILATION.md).

The nine-launch host matrix reduces compilation requests from nine to three
while rebinding current Q/K/V, output, bias, descriptor and scalar arguments.
The new `partitioned_kernel_compile_calls`, `partitioned_kernel_compile_cache_hits`
and `partitioned_kernel_compile_wall_s` receipts measure real compilation work.
They add no per-launch CUDA synchronization. 01041 has no compile timing receipt,
so its gate wall time cannot be called compilation time or removable overhead.
GPU numerical reuse tests are included but require SM120 hardware. No matched
GPU speed gain or rendered acceptance has been established here.

## Evidence identity

- `Pasted text(20261003-214250).txt`, SHA-256 `f0028e283ce64fbd803b935c378969c16ac0338ae14c0f58d9bb7b6f1cfb174c`.
- `metrics_01041_.json`, SHA-256 `d9806a45c6acb6979cd923811a89bfaac4f6639f03de228340f93db16a5ec5e4`.
- Crops `MiniMax_H3_00119-audio.mp4_snapshot_00.07.324_cr.jpg` and `...00.07.331_cr.jpg`.

Install changes through the existing ComfyUI Patcher overlays: Flow #93 and
Sol #37. Sol preserves the updated 0.1.7/SM121 base and remains independent of
optional diagnostic overlay #35. The next GPU capture must establish actual
compile timing and rendered behavior with the same seed, conditioning, carried
prefix, geometry and schedule for a performance comparison.
