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
steps**. Keep the first pass fixed at 7 throughout the learned-refine comparison. C7 is the
flow-aligned parity smoke baseline; test C6 next and test C5 only if C6 decoded video/audio remains
acceptable. Further refine-step reductions are exploratory rather than a required gate. The primary
learned-refine comparison remains denoise 0.25; the same B/C rows are also repeated at 0.30. Two
patch-safe base regimes are explicit: 54x40 latent
(864x640, 0.553 MP) and 62x44 latent (992x704, 0.698 MP), both evaluated against the 64x48
(1024x768) target. The scenarios cover difficult motion/scene changes, small people/faces, fine text,
and reference-heavy conditioning. Prompts and reference assets may differ between scenarios but must
remain identical across rows within one scenario/area/denoise/seed slice.

Row D must use **Progressive Handoff (Target Input)** in the two-chunk Continuum graph: Continuum is configured at 64x48 while the wrapper's internal source is 54x40. This preserves target-sized session and native-mask geometry between chunks. The source-input progressive node is a standalone control and is not the Continuum benchmark topology.

Row E does not generate a low-resolution first pass. Its `shift_reference_grid` is the active area
regime's source grid (54x40 or 62x44), while H3 itself samples directly on the 64x48 target grid.
The source and target observations therefore differ in both regimes, keeping the resolution mapping
non-identity while isolating it from trajectory guidance and progressive handoff.

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
