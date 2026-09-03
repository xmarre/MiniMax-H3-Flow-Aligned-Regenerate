# Decoded-media benchmark protocol

## Evidence policy

Decoded video and audio are the quality gate. Latent telemetry establishes that a feature is active and structurally correct; it does not establish a perceptual win.

For matched comparisons keep seed, prompt, Qwen/direct references, LoRAs, DiffAid, Untwist, sampler, scheduler, VAE, Continuum chunking/session settings, masks, audio behavior, crop, and decode identical unless the experiment explicitly changes one of them.

Attach **MiniMax H3 Metrics JSON** to experimental paths. The autosaved JSON is the authoritative final artifact because later sampler completions atomically refresh the same sink. For two-pass capture/guidance tests, feed the capture metrics object into the downstream regenerate/refine adapter's optional `metrics` input so both passes share one artifact.

## Completed difficult-motion smoke ledger

| ID | Path | Result |
|---|---|---|
| C7 | 7 base + 7 learned-refine outer steps with flow guidance | Accepted |
| C6 | 7 + 6 | Accepted |
| C5 | 7 + 5 | Accepted |
| C4 | 7 + 4 exploratory lower-step point | Accepted smoke; not promoted to formal matrix row |
| D10-fixed-direction | Progressive Target Input, 10 outer steps, fixed 0.35, direction 0.25 | Valid; roughly baseline-level on the original rectangular difficult-motion case |
| D10-acceleration | Corrected `direction+acceleration` | Structurally valid; neutral/inconclusive media |
| D10-temporal-v2 | Standard-VAE `direction+temporal`, temporal 0.20 | Weak/tentative positive only |
| D10-consistency | Square matched run, `downsample_consistency=0.25` | Active measurable corrections; no visible improvement |
| D10-auto | Square matched run, `auto_compute` handoff | Functional; no clear advantage over fixed 0.35 |
| D12-direction | Square matched run, fixed 0.35, direction 0.25 | Perceptually better than D10 |
| D14-direction | Square matched run, fixed 0.35, direction 0.25 | Again slightly better; current tested quality point |
| D14-temporal | Same D14 topology plus temporal 0.20 | No perceptual difference from D14 direction-only |
| E0/E1 | Learned-refine SIGMAS identity control vs resolution-aware remap | Structurally valid; no relevant E1 quality improvement |

Earlier temporal media generated with the accidental TensorRT `w4a16_awq` VAE is excluded from media conclusions.

## Current progressive quality reference

The final matched quality sweep used:

```text
private source pixels = 736 x 736
private source latent = 46 x 46
target pixels = 896 x 896
target latent = 56 x 56
source_mode = scale
source_scale = 0.83
handoff_selection = fixed
handoff_coordinate = 0.35
guidance_mode = direction
direction_weight = 0.25
acceleration_weight = 0
temporal_weight = 0
consistency_weight = 0
low_frequency_cutoff = 0.25
sampler = SA-Solver-PECE
outer_steps = 14
```

D14 fixed handoff selected schedule index 9 at unshifted coordinate ~0.35799987 / sigma ~0.86998779. Across the two physical chunks:

- 54 logical H3 calls;
- 36 actual transformer NFEs;
- 18 Spectrum forecasts;
- low stage: 34 logical / 22 actual / 12 forecast;
- exact probes: 2 logical / 2 actual;
- high stage: 18 logical / 12 actual / 6 forecast;
- 6 progressive sampler invocations;
- 4 history boundaries;
- both high stages begin with an actual H3 call;
- sampler failures: 0.

This is effectively 9 low-grid + 5 high-grid outer steps per chunk. It is not equivalent to a separate 7+7 learned-refine workflow, but it is a useful quality/compute comparison because the majority of the trajectory runs on the cheaper low grid.

The D12 predecessor used fixed 0.35 -> ~0.334/index 8 and recorded 46 logical / 32 actual / 14 forecast calls. The user judged D12 better than D10 and D14 again a bit better than D12. Stop the current step sweep at 14 unless a future scenario gives a specific reason to test higher budgets.

## Handoff policy smoke

On the matched 46x46 -> 56x56 latent setup, `auto_compute` uses

```text
0.35 / sqrt((56*56)/(46*46)) = 0.2875
```

and snapped to ~0.30/index 7. It kept 38 logical / 28 actual / 10 forecast calls but reallocated them:

- low: 26 logical / 18 actual / 8 forecast;
- probes: 2 actual;
- high: 10 logical / 8 actual / 2 forecast.

Media was valid/different but not clearly better. Fixed 0.35 remains the tested quality recommendation; `auto_compute` remains an optional functional policy.

## Downsample consistency smoke

The corrected positive-weight smoke used `consistency_weight=0.25` with all other matched D10 settings fixed. The mode was definitely active and unclamped, with correction RMS ratios roughly 1.5-6.8% across high-stage calls. Decoded media looked the same and the difficult fast-motion clothing smear remained. Do not weight-sweep from this neutral result.

## Temporal correspondence smoke

The final promotion check compares D14 direction-only directly against D14 `direction+temporal` with `temporal_weight=0.20`. Both runs have the same 54 logical / 36 actual / 18 forecast topology and the same handoff.

Temporal telemetry:

- chunk 1 valid fraction ~0.00294674 (0.2947%), confidence mean ~0.00156669, similarity mean ~0.921323;
- chunk 2 valid fraction ~0.000844464 (0.08445%), confidence mean ~0.000314343, similarity mean ~0.743523;
- mean temporal RMS ratio ~0.00103940;
- maximum temporal RMS ratio ~0.00348035;
- mean direction RMS ratio ~0.02188087.

The user saw no perceptual difference. Temporal is therefore functioning but neutral at the final quality budget. Do not increase temporal weight merely to compensate for sparse matcher support.

## Resolution-aware refine SIGMAS smoke

E is a **downstream learned-refine SIGMAS** experiment. Correct graph:

```text
MP sizing -------------------------------> Continuum width/height
   \
    -> Refine Target Geometry -----------> Resolution-Aware Sigmas target width/height

existing refine scheduler SIGMAS
    -> Resolution-Aware Sigmas
    -> MiniMax H3 Latent Upscaler + Refine (3D).sigmas
```

Forbidden:

```text
Refine Target Geometry -X-> Continuum
Resolution-Aware Sigmas -X-> Continuum.sigmas
```

The actual learned refiner stays enabled in both E0 and E1. E0 is exact identity; E1 changes only the refine sigma map.

The completed square E pair used 928x928 target geometry and a 768x768 H3-native reference. For a square target, manual 768x768 is numerically identical to `source_width=source_height=0` auto-reference, so no extra media rerun is needed just to exercise auto geometry.

E1 diagnostics:

- target/reference area ratio ~1.46006944;
- relative factor ~1.20833333;
- native video shift 12;
- effective video shift 14.5;
- `shared_av_coordinate=true`;
- nonzero sigma delta (max ~0.04666 in the recorded run).

E0/E1 both completed at 52 logical / 34 actual / 18 Spectrum forecast calls. E1 showed no relevant decoded-media improvement. Keep resolution-aware mode off and do not start a strength/calibration sweep.

## Broader optional formal matrix

The smoke program above is sufficient for functional PR closure. A broader research-quality claim would require the full matrix below.

| ID | Path | First-pass outer steps | Refine outer steps | Denoise |
|---|---|---:|---:|---:|
| A | Native direct at final grid | base schedule | n/a | n/a |
| B | Existing learned upscale baseline | 7 | 7 | 0.25 / 0.30 |
| C7 | Flow-aligned learned-upscale refine | 7 | 7 | 0.25 / 0.30 |
| C6 | Flow-aligned learned-upscale refine | 7 | 6 | 0.25 / 0.30 |
| C5 | Flow-aligned learned-upscale refine | 7 | 5 | 0.25 / 0.30 |
| D | Progressive Target Input | split schedule | split schedule | n/a |
| E | Resolution-aware refine SIGMAS | unchanged | unchanged | unchanged |

Machine-readable axes remain:

- seeds `0, 1, 2`;
- base area regimes ~0.55 MP and ~0.70 MP toward ~0.79 MP target;
- scenarios: difficult motion/scene change, small people/faces, fine text, reference-heavy conditioning;
- refine denoise 0.25 primary / 0.30 secondary for B/C rows.

Do not imply that the full formal matrix has been spent; it is an optional publication-grade expansion beyond the completed PR smoke evidence.

## Required metrics

Record at minimum:

- actual pixels and video latent T/H/W;
- Qwen, direct-reference, audio, video, and total packed rows;
- logical calls, actual H3 NFEs, Spectrum forecasts;
- sampler outer/phase topology;
- handoff selected/requested coordinate and sigma;
- low/probe/high wall times;
- guidance component RMS ratios and clamp state;
- trajectory storage, reset boundaries, exact anchors, fallbacks;
- peak VRAM and host RAM when comparing performance.

## Human review

Decode final media identically and review synchronized playback plus audio. Inspect composition, prompt/reference fidelity, motion, temporal stability, detail, faces, small objects, scene changes, borders, audio, and Continuum seams. Full clips are required; preview frames are insufficient.

The machine-readable ledger is [`workflows/benchmark-matrix.json`](../workflows/benchmark-matrix.json). The configuration overlays are:

- [`flow-aligned-two-pass.overlay.json`](../workflows/flow-aligned-two-pass.overlay.json)
- [`progressive-handoff.overlay.json`](../workflows/progressive-handoff.overlay.json)
- [`resolution-shift-only.overlay.json`](../workflows/resolution-shift-only.overlay.json)
