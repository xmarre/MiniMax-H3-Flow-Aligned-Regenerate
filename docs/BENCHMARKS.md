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
| E | Resolution-shift-only direct target; MP base -> geometry-only mirror of refine scale 1.20 -> direct generation target | base schedule | n/a | n/a |

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

### D10 video-time correspondence experiment — clean standard-VAE result

Two earlier `direction+temporal` media runs remain **invalid for temporal attribution** because the
workflow had accidentally switched to `ComfyUI-H3VAE_TRT` with the `w4a16_awq` decoder. The user
identified that decoder as the source of the repeated/patterned output. Their structural evidence is
retained, including the v1 ambiguity bug and the v2 hard uniqueness/cycle fix, but their decoded-media
verdicts are excluded.

The clean **D10-temporal-v2-rerun** used the standard MiniMax H3 video VAE and completed successfully.
The console shows the native `VAELoader` / `MiniMaxH3VideoVAE` path rather than
`MiniMaxH3TRTVAE`.

It preserved the accepted D10 structure exactly:

- 10 SA-Solver-PECE outer steps;
- Target Input `source_mode=scale`, `source_scale=0.83`;
- 48x46 private source -> 58x56 target;
- fixed handoff request 0.35 -> index 6 / coordinate ~0.400000016 / sigma ~0.888888896;
- `direction+temporal`, direction 0.25, temporal 0.20, acceleration 0, consistency 0;
- 38 logical calls = 22 low + 2 exact probes + 14 high;
- 28 actual H3 NFEs / 10 Spectrum forecasts;
- high stage 10 actual / 4 forecast;
- 6 sampler invocations, 4 history boundaries, 2 exact probes;
- both high stages begin actual, both sampler walls complete with `failed=false`;
- no guidance correction hits the RMS clamp.

V2's conservative temporal gate remained extremely sparse, which is expected after the earlier
ambiguity fix:

- chunk 1 valid fraction ~0.0015985 (0.1598%), confidence mean ~0.0008162,
  similarity ~0.8924, margin ~0.03559;
- chunk 2 valid fraction ~0.0005420 (0.0542%), confidence mean ~0.0002955,
  similarity ~0.9495, margin ~0.03056;
- mean temporal RMS ratio across all 14 guidance calls ~0.0008996, maximum ~0.002206;
- mean direction RMS ratio ~0.02426;
- 12/14 temporal calls reuse the resolved ~0.4 cache and 12/14 report the expected post-handoff
  reference clamp.

The decoded result is **good** and no longer contains the VAE-induced pattern. The user's matched
perceptual verdict is **maybe slightly better than the non-temporal D10 run**. That is the first clean
positive evidence for the temporal hypothesis, but it is deliberately recorded as **weak/tentative**
rather than a validated quality improvement: the difference is small and only one clean difficult-motion
case has been judged.

Decision:

- retain conservative `direction+temporal` v2 as an experimental candidate;
- keep temporal weight 0.20 as the current reference operating point;
- do not claim a general quality improvement and do not start a parameter sweep from this single weak
  positive;
- proceed to the orthogonal **E: resolution-shift-only** gate next;
- a later second-seed matched direction-only/temporal pair can be used if stronger promotion evidence
  is needed.


Row E does not generate a low-resolution first pass. Its resolution reference is the documented
**H3-Base 768p regime**, not the arbitrary private source grid used by progressive D10. MiniMax's
public H3 documentation states that H3-Base produces 768p and that the shorter output side is 768
pixels by default. Pinned ComfyUI expresses the same native canvas rule as `BASE_SHORT_EDGE=768`,
`CANVAS_MULTIPLE=32`, and `adapt_canvas()`. For the current 896x928 target, that exact rule yields a
768x800 analytic reference grid.

This distinction matters. SD3 Eq. 23 maps between two resolution *observation regimes*; it does not
require that a low-resolution sampling pass actually run. Using D10's 736x768 private source here
would silently couple E to an implementation detail of the progressive experiment and would overstate
the relative shift.

### E10 resolution-shift smoke

Use one matched difficult-motion pair on the current standard-VAE workflow. E is a **direct
target-grid** experiment.

The real graph topology matters here: the 1.20 scale shown in the workflow belongs to the downstream
**MiniMax H3 Latent Upscaler + Refine (3D)** node. That node executes only after Continuum sampler 1
has already produced its base-resolution chunks. Its scale widget therefore cannot be wired backward
to Continuum and cannot directly define E's first-pass canvas.

For E, keep the existing pre-Continuum MP sizing node as the base-resolution authority and insert
**MiniMax H3 Refine Target Geometry [Experimental]** between the MP dimensions and Continuum:

- MP width -> Refine Target Geometry `base_width`
- MP height -> Refine Target Geometry `base_height`
- Refine Target Geometry: `scale=1.20`, `align=32`, `keep_proportion=true`
- Refine Target Geometry `target_width` -> Continuum `width`
- Refine Target Geometry `target_height` -> Continuum `height`
- the same target width/height -> Resolution-Aware Sigmas `target_width/target_height`
- Resolution-Aware Sigmas `source_width=0`, `source_height=0`

The helper is geometry-only. It mirrors the pinned LBH refine node's exact scale-by-multiplier target
calculation and 32px/H3 alignment, but performs no learned upscale and no refinement sampling. This
preserves the user's automatic MP-driven aspect/size selection while making the **post-refine-size**
canvas available before sampler 1 without manual per-image arithmetic.

For the current example:

- MP/base Continuum canvas: 736x768 pixels
- downstream refine scale to mirror: 1.20
- geometry-only resolved direct E target: 896x928 pixels
- analytic H3-Base reference regime: 768x800 pixels
- target/reference area ratio: ~1.35333333
- SD3 relative factor `alpha=sqrt(m/n)`: ~1.16332856
- H3 native video shift remains 12; composition gives effective video shift ~13.95994269 at strength 1
- 10 SA-Solver-PECE outer steps with `MiniMax H3 SA-Solver Scheduler / simple_control`
- same seed, prompt, references, LoRAs, DiffAid, Untwist, Spectrum, Continuum chunking, audio behavior,
  standard MiniMax H3 VAE, and decode
- no progressive handoff, trajectory capture/guidance, temporal guidance, learned upscale, or sampler-2 refine
- Continuum Run Storage off

Wire **MiniMax H3 Resolution-Aware Sigmas** between the existing SA-Solver scheduler SIGMAS output and
Continuum's SIGMAS input. On the MODEL path keep DiffAid -> Untwist -> Spectrum, then insert
**MiniMax H3 Runtime Metrics Probe** immediately before Continuum. Leave the probe's optional
`metrics` input disconnected. Fan its `metrics` output to both Resolution-Aware Sigmas `metrics`
and **MiniMax H3 Metrics JSON**.

Run **E0-direct-control** first:

- Resolution-Aware Sigmas `mode=off`
- `source_width=0`, `source_height=0`
- target width/height from Refine Target Geometry
- strength 0

The output SIGMAS must be bit-exact parity with the input schedule. Diagnostics should report
`extra_shift_factor=1`, `base_video_shift=12`, and `effective_video_shift=12`.

Then run **E1-resolution-aware** with only:

- `mode=resolution_aware`
- strength 1.0

changed. For the current 896x928 target, diagnostics should resolve source 768x800 and report area
ratio ~1.35333333, relative factor ~1.16332856, effective video shift ~13.95994269,
`reference_mode=h3_native_auto`, and `shared_av_coordinate=true`.

The implementation first inverts H3's native video shift, applies only the **relative** SD3 map, and
then restores H3's native video shift:

[
sigma_v' = f_{12}!left(f_{alpha}!left(f_{12}^{-1}(sigma_v)ight)ight),
qquad
f_s(t)=rac{s t}{1+(s-1)t}.
]

Because H3 derives audio sigma from the same shared base coordinate, E1 also changes the audio
schedule. Decoded audio remains part of the pass/fail criterion.

A two-chunk 10-outer-step PECE run should still produce **38 sampler logical calls**, with:

- 0 progressive sampler invocations
- 0 progressive history boundaries
- 0 exact handoff probes
- 0 flow/temporal guidance events

Do **not** require the same actual-NFE/forecast split between E0 and E1. Spectrum sees different sigma
coordinates in E1 and may legitimately change forecast decisions. SA-Solver's default stochastic
interval also remains tied to effective supplied sigma.

The media gate is E1 versus E0: fast-motion clothing, newly revealed grass/background,
limb/disocclusion boundaries, anatomy, detail, shot continuity/dynamics, prompt/reference fidelity,
Continuum seams, and decoded audio.

Do not tune strength or use `calibrated` mode before this pair is decoded. If E1 is clearly better,
the next experiment is a combination test on the accepted D10 progressive+temporal path. If E1 is
neutral/worse, leave resolution-aware sigmas experimental/off rather than beginning a strength sweep.

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

The configuration overlays are
[`flow-aligned-two-pass.overlay.json`](../workflows/flow-aligned-two-pass.overlay.json),
[`progressive-handoff.overlay.json`](../workflows/progressive-handoff.overlay.json), and
[`resolution-shift-only.overlay.json`](../workflows/resolution-shift-only.overlay.json). They
deliberately describe the new nodes and graph insertion points instead of embedding model filenames,
prompts, or third-party node IDs that would silently diverge from the user's canonical Continuum
workflow.
