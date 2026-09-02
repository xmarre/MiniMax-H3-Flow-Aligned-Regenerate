# Decoded-media benchmark protocol

## Fixed inventory

Duplicate the current two-chunk Continuum graph and keep seed, prompt, Qwen/direct references,
LoRAs, DiffAid, Untwist, sampler, scheduler, base frames, audio behavior, chunk boundaries, crop,
and decode settings identical. Record exact git revisions for ComfyUI and every custom node.

Attach **MiniMax H3 Metrics JSON** to each experimental path. Capture sampler-only wall time around
KSampler and node time around upscaler/handoff; total prompt time is secondary.

## Primary matrix

| ID | Path | Base/final grid | Refine outer steps | Denoise |
|---|---|---:|---:|---:|
| A | Native direct | final 64x48 (~1024x768) | base schedule | n/a |
| B | Existing learned upscale baseline | 54x40 -> 64x48 | 6 | 0.25 |
| C6 | Flow-aligned learned-upscale refine | 54x40 -> 64x48 | 6 | 0.25 |
| C5 | Flow-aligned learned-upscale refine | 54x40 -> 64x48 | 5 | 0.25 |
| C4 | Flow-aligned learned-upscale refine | 54x40 -> 64x48 | 4 | 0.25 |
| D | Progressive handoff | 54x40 -> 64x48 | one split schedule | n/a |
| E | Resolution-shift-only native | shift reference 54x40 -> generation 64x48 | base schedule | n/a |

Run the same set of at least three fixed seeds for every paired row. The primary learned-refine
comparison remains denoise 0.25. Repeat the relevant learned-refine rows at denoise 0.30 only after
the 0.25 behavior is understood. Repeat the matrix at a ~0.5 MP base and nominal ~0.7 MP base, and
include difficult motion/scene changes, small people/faces, fine text, and reference-heavy conditioning.

Row E does not generate a low-resolution first pass. Its 54x40 value is the spatial reference used to
derive the experimental resolution-dependent coordinate shift while H3 itself samples directly on the
64x48 target grid. This makes the shift ratio non-identity and keeps the row isolated from trajectory
guidance or progressive handoff.

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
