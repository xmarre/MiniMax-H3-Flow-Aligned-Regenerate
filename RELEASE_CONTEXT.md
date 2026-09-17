## Coordinated production release

This Flow v0.3.5 release is coordinated with [ComfyUI-Sol-H3 v0.1.5](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.5) and [ComfyUI-VDN-H3-Plus v1.5.5](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.5).

Implementation PRs: Flow [#33](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/33) + [#48](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/48), [Sol-H3 #14](https://github.com/xmarre/ComfyUI-Sol-H3/pull/14), and [VDN-H3-Plus #18](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/18).

## Motivation: one production continuation path and the first-high artifact

The coordinated release resolves two production issues that had become coupled during validation.

First, Continuum had accumulated multiple progressive continuation experiments. The validated ordinary path is now **Progressive Handoff (Target Input)**: learned-3D low-to-high handoff on unprotected/fractional calls, conservative single-target-grid sampling for exact protected continuation, and a bounded four-tick guided-audio overlap during the sampler with caller-owned exact audio/video restored at output. Mixed-Grid remains registered for one compatibility release but is no longer the recommended production topology or a release gate.

Second, the production stack could reproduce a first-high visual artifact when VDN retained grouped local attention was executed through Sol sparse attention after the low-to-high handoff. Controlled R/W/E/M investigation showed that VDN's restricted local K/V window and learned complement did not need to be removed. The actual mismatch was positional: VDN gathers requested Q rows separately from its restricted K/V domain `[global_rows, permitted_window_rows]`, so a query's physical K/V position is not generally its local Q ordinal. Sol's ordinary exact-neighbor selector could therefore protect the wrong K blocks.

## Resolution: Target Input plus mapped physical neighbors

Flow v0.3.5 standardizes Target Input and retires Mixed-Grid from the production acceptance matrix. In the paired attention fix, VDN-H3-Plus v1.5.5 provider v4 transports the exact owner-bound physical query position inside the already-gathered restricted K/V domain. Sol-H3 v0.1.5 validates that map, compiles bounded K64 descriptor intervals, and adds them to the real SM120 sparse kernel as:

```text
new_exact = old_exact OR mapped_neighbor
```

The mapped-neighbor fix does not broaden VDN's K/V domain, remove its learned complement, reconstruct full K/V, construct square Q, or add an H3 transformer NFE. Unsupported or stale mapping metadata fails closed to VDN's supplied native restricted-domain callback.

The validated coordinated stack is:

```text
MiniMax H3 Flow-Aligned Regenerate v0.3.5
ComfyUI-Sol-H3                    v0.1.5
ComfyUI-VDN-H3-Plus              v1.5.5
```

The separate VDN PR #8 audio-fidelity/training experiment and the retired weighted Mixed-Grid companion PR cluster are not part of this release.

## Coordinated production validation

Run `00494` completed the representative two-chunk release gate:

```text
17 logical calls
13 actual H3 NFE
4 Spectrum forecasts

low:    4 actual / 1 forecast
probe:  1 actual / 0 forecast
high:   2 actual / 1 forecast
later:  6 actual / 2 forecast
```

The first chunk used one learned-3D progressive handoff. The later exact-prefix chunk used the conservative Target Input target-grid fallback with the four-tick guided-audio overlap, no private low-grid sampler/probe/upscaler/history boundary, and exact caller-visible AV restore at output. Mapped VDN local routing stayed direct with requested Q rows equal to kernel Q rows and zero square-Q expansion, with no mapped-local fallback or kernel-unavailable failure.

The decoded 14-second output showed no captured first-high corruption, frame shift, zoom-out, top-edge reveal, flash, grid artifact or physical AV seam; the chunk boundary was perceptually seamless.

The paired Sol mapped-kernel timing gate measured `1.109472036 ms` versus `1.105535984 ms` for historical diagnostic M, a `+0.356031%` median delta inside the `+5%` acceptance budget.

00494 evidence hashes:

- runtime log: `699c17b6d85d186039b9179c4b99edb596e88fdda37d3f8814917c50b6905ca8`;
- metrics JSON: `c25a90af61d83c1430c3f29b9e99de7b0adb22f60d714e2d1392a269e6f0802e`;
- final MP4: `1ffb5ebff47417f2f9354d3ae7cdfa32b6b6e292cd661efb9ddc2fd818d66ab2`.
