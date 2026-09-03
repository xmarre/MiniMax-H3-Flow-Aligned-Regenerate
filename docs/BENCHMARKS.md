# Decoded-media benchmark protocol

## Fixed inventory

Duplicate the current two-chunk Continuum graph and keep seed, prompt, Qwen/direct references,
LoRAs, DiffAid, Untwist, sampler, scheduler, base frames, audio behavior, chunk boundaries, crop,
and decode settings identical. Record exact git revisions for ComfyUI and every custom node.

Attach **MiniMax H3 Metrics JSON** to each experimental path. The output node allocates a unique
`.json` file under ComfyUI's normal `output/` directory using its optional `filename_prefix`
(default: `h3_flow_regenerate/metrics`), writes the current snapshot immediately, and keeps that
same file registered as a live sink. Each later sampler completion refreshes it atomically, so a
node that executes before the learned-refine sampler still ends with the post-refine guidance/counter
state. The file is therefore the authoritative final artifact; the node's STRING output is the
registration-time snapshot retained for compatibility. In the two-pass path, route the
Trajectory Capture metrics object into the Flow-Aligned Regenerate/Refine State optional `metrics`
input so low-pass capture and high-pass guidance share one sink. Capture sampler-only wall time around
KSampler and node time around upscaler/handoff; total prompt time is secondary.

## Primary matrix

| ID | Path | First-pass outer steps | Refine outer steps | Denoise |
|---|---|---:|---:|---:|
| A | Native direct at final 64x48 (~1024x768) | base schedule | n/a | n/a |
| B | Existing learned upscale baseline, 54x40 -> 64x48 | 7 | 7 | 0.25 |
| C7 | Flow-aligned learned-upscale refine, 54x40 -> 64x48 | 7 | 7 | 0.25 |
| C6 | Flow-aligned learned-upscale refine, 54x40 -> 64x48 | 7 | 6 | 0.25 |
| C5 | Flow-aligned learned-upscale refine, 54x40 -> 64x48 | 7 | 5 | 0.25 |
| D | Progressive target-input handoff; workflow stays 64x48, early stage internally 54x40 | split schedule | split schedule | n/a |
| E | Resolution-shift-only native; shift reference 54x40 -> generation 64x48 | base schedule | n/a | n/a |

The machine-readable matrix fixes the seed set to `0, 1, 2` and expands every applicable row across
four required content scenarios. The actual working workflow uses **7 first-pass steps and 7 refine
steps**. Keep the first pass fixed at 7 throughout the learned-refine comparison. The reduced-refine
smoke ladder is complete: C7=7+7, C6=7+6, C5=7+5, and exploratory C4=7+4 all completed with acceptable
decoded media in the difficult-motion smoke case, and corrected C4 telemetry matched the expected
Spectrum provenance exactly (9 actual / 4 forecast and 9 exact anchors per base chunk). C4 remains an
exploratory lower-step operating point rather than a formal primary-matrix row. The next feature-path
gate is D, progressive target-input. The primary learned-refine comparison remains denoise 0.25; the
same B/C rows are also repeated at 0.30. Two
patch-safe base regimes are explicit: 54x40 latent
(864x640, 0.553 MP) and 62x44 latent (992x704, 0.698 MP), both evaluated against the 64x48
(1024x768) target. The scenarios cover difficult motion/scene changes, small people/faces, fine text,
and reference-heavy conditioning. Prompts and reference assets may differ between scenarios but must
remain identical across rows within one scenario/area/denoise/seed slice.

Row D must use **Progressive Handoff (Target Input)** in the two-chunk Continuum graph: Continuum is configured at 64x48 while the wrapper's internal source is 54x40. This preserves target-sized session and native-mask geometry between chunks. The source-input progressive node is a standalone control and is not the Continuum benchmark topology.

The difficult-motion D smoke is now anchored to the user's aspect-correct 10-outer-step reference.
Keep the target video latent at 58x56 HxW (928x896 pixel HxW; 896x928 W×H), use Target Input
`source_mode=scale` with `source_scale=0.83`, which resolves to 48x46 latent HxW
(768x736 pixel HxW; 736x768 W×H), and keep the same seed, prompt, references, DiffAid, Untwist,
Spectrum and Continuum chunking. The separate learned-upscale/refine branch remains removed. Use
fixed handoff coordinate 0.35, direction guidance 0.25, acceleration/consistency 0, low-frequency
cutoff 0.25, and keep Continuum Run Storage off.

The accepted direction-only D10 smoke selected schedule index 6 (actual unshifted coordinate about
0.400000016, sigma about 0.888888896). Across two chunks it recorded 38 logical model calls,
28 actual NFEs and 10 Spectrum forecasts: low 22/16/6 logical/actual/forecast, two exact probes,
and high 14/10/4. Both high stages began with an actual H3 call, all sampler walls completed without
fallback, and decoded media was accepted as roughly baseline-level for the difficult motion case.
It did not eliminate the characteristic clothing deformation and newly revealed grass/background
artifacts. Direction+acceleration experiments must keep every other D10 setting fixed so their decoded
media can be compared against this reference.

The first D10 run with the full-spectrum velocity-acceleration implementation preserved the same
38/28/10 logical/actual/forecast topology and completed both chunks without a guidance clamp or
sampler failure. The decoded result changed shot structure but remained acceptable; the fast-motion
clothing/grass artifacts were judged approximately unchanged, with at most a possible slight
improvement. Post-run audit found that the initial SA-Solver-PECE adaptation advanced acceleration
history on the predictor, so the same-coordinate corrector no longer compared against the previous
distinct denoising time.

The corrected PECE acceleration rerun is now complete. It again preserved the D10 topology
(38 logical / 28 actual / 10 forecast, 22 low + 2 probe + 14 high), used the same 48x46 -> 58x56
geometry and schedule-index-6 handoff, and completed both progressive sampler walls without failure.
Telemetry confirms the intended PECE semantics: predictor and same-coordinate corrector calls at
coordinate ~0.3 both use ~0.4 as their acceleration anchor, calls at 0.2 both use ~0.3, and calls at
~0.1 both use 0.2. No guidance correction hit the RMS clamp.

Decoded media remained good and returned to one continuous shot, but the difficult fast-motion
clothing/grass/disocclusion artifact class did **not** show a clear perceptual improvement. The
acceleration result is therefore recorded as neutral/inconclusive rather than promoted. Do not continue
global acceleration-weight tuning from this smoke result. If that artifact class is pursued further,
the next research target should be video-time correspondence/occlusion-aware guidance rather than
additional denoising-time acceleration strength.

### D10 video-time correspondence experiment — VAE-confounded, clean rerun pending

Two matched `direction+temporal` runs were previously decoded against the accepted D10 direction-only
control. Structurally they remained comparable: 38 logical / 28 actual / 10 Spectrum forecast calls,
22 low + 2 exact probes + 14 high, 48x46 -> 58x56 geometry, handoff index 6 at coordinate
~0.400000016 / sigma ~0.888888896, no sampler failures, and no RMS clamp activation.

Those **media verdicts are not valid temporal evidence**. The workflow had accidentally switched to
`ComfyUI-H3VAE_TRT` and compiled the `w4a16_awq` decoder. The user later identified that decoder as
the source of the repeated/patterned output. The console/runtime evidence confirms that the affected
run used `MiniMaxH3TRTVAE`. Therefore the patterned grass cannot be attributed to
`direction+temporal` from those runs.

The structural findings remain useful:

- v1 exposed a real ambiguity bug: high cosine similarity but weak best-vs-second-best uniqueness
  allowed repetitive regions to receive temporal transport;
- v2 fixed that independently by making similarity/margin hard admission rules, requiring exact
  reverse-cycle consistency, and caching by the resolved low-grid reference coordinate;
- on the latest v2 run, strict valid support fell to ~0.2202% / ~0.2235%, mean confidence to
  ~0.00118 / ~0.00120, and 12/14 guidance calls correctly reused the resolved/clamped ~0.4 anchor.

Because the VAE changed at the same time as the temporal experiment, neither v1 nor v2 can answer the
quality question. The public `direction+temporal` mode and conservative v2 implementation are restored.

The next gate is exactly one clean **D10-temporal-v2-rerun** after the corrected VAE is installed:

- 10 SA-Solver-PECE outer steps;
- Target Input `source_mode=scale`, `source_scale=0.83`;
- fixed handoff request 0.35;
- `guidance_mode=direction+temporal`;
- direction 0.25, temporal 0.20, acceleration 0, consistency 0;
- low-frequency cutoff 0.25;
- same seed, prompt, references, LoRAs, DiffAid, Untwist, Spectrum, Continuum chunking, audio behavior,
  and decode settings;
- Continuum Run Storage off;
- use the corrected VAE decoder, explicitly record the exact decoder/engine identifier, and do **not**
  use `w4a16_awq`.

Do not tune temporal weight, search radius, margin, or cycle tolerance before this clean rerun.
The media gate returns to the original question: fast-motion clothing and newly revealed
grass/background must improve without freezing, stale copying, repetition, ghosting, or smearing.

Resolution-shift-only path E remains pending after this clean temporal rerun.


Row E does not generate a low-resolution first pass. Its `shift_reference_grid` is the active area
regime's source grid (54x40 or 62x44), while H3 itself samples directly on the 64x48 target grid.
The source and target observations therefore differ in both regimes, keeping the resolution mapping
non-identity while isolating it from trajectory guidance and progressive handoff.

### E10 resolution-shift smoke

Before spending the formal E matrix, use one matched difficult-motion pair on the user's current
working geometry. This is a **direct target-grid** experiment, not another progressive handoff:

- source reference geometry for the analytic mapping: 736x768 pixels (46x48 latent W/H);
- actual generation geometry: 896x928 pixels (56x58 latent W/H);
- target/source area ratio: ~1.47101449;
- strength-1 SD3-derived factor: ~1.21285386;
- 10 SA-Solver-PECE outer steps;
- same seed, prompt, references, LoRAs, DiffAid, Untwist, Spectrum, Continuum chunking, audio behavior,
  and decode;
- no progressive handoff, trajectory guidance, or separate learned upscale/refine;
- Continuum Run Storage off.

Run **E0-direct-control** first with Resolution-Aware Sigmas set to `off`. Then run
**E1-resolution-aware** with mode `resolution_aware`, strength 1.0, source 736x768 and target
896x928. Do not tune strength before decoding this pair.

A two-chunk 10-outer-step PECE run should still produce 38 sampler logical calls, but there must be
zero progressive sampler invocations/history boundaries and zero handoff probes. Do **not** require
the same actual-NFE/forecast split between E0 and E1: changing the sigma coordinates can legitimately
change Spectrum's policy decisions, so record that difference rather than forcing parity.

The media gate includes motion/disocclusion, clothing, grass/background texture, anatomy, detail,
shot dynamics, prompt/reference fidelity, and decoded audio. The resolution remap moves H3's shared AV
coordinate; a video improvement that damages audio is not a pass.

## Required metrics

- actual pixels and video latent T/H/W; padded H/W and whether padding occurred;
- Qwen, direct-reference, audio, video, and total packed rows;
- actual H3 transformer NFEs and Spectrum forecasts;
- sampler logical calls and SA/PECE outer/phase topology;
- effective shifts and unshifted coordinates;
- low/high sampler, upscaler, handoff/probe, and guidance time;
- trajectory storage; resets, exact anchors, and fallbacks;
- peak VRAM and host RAM from the same profiler.

## Human review

Decode final media identically. Review synchronized playback plus audio. Score composition,
prompt/reference fidelity, motion, temporal stability, detail, faces, small objects, scene changes,
borders, audio, and Continuum seams. Full clips are required; preview frames are insufficient.

The machine-readable plan is [`workflows/benchmark-matrix.json`](../workflows/benchmark-matrix.json).
Store each workflow, metrics JSON, decoded checksum, timing log, and review sheet together.

The two configuration overlays are
[`flow-aligned-two-pass.overlay.json`](../workflows/flow-aligned-two-pass.overlay.json) and
[`progressive-handoff.overlay.json`](../workflows/progressive-handoff.overlay.json). They deliberately
describe the new nodes and graph insertion points instead of embedding model filenames, prompts, or
third-party node IDs that would silently diverge from the user's canonical Continuum workflow.
