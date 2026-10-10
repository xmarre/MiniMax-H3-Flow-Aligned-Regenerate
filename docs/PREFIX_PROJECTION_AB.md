# Native H3 source-prefix projection A/B

**MiniMax H3 Prefix Projection A/B** tests whether an authoritative target-grid
prefix survives source-grid projection more accurately through the native video
VAE than through latent bicubic interpolation. It operates on an existing
boundary witness. It performs no diffusion sampling and changes no generation
settings or outputs. The RGB roundtrip is an experimental reconstruction, not an
accepted continuation correction.

## Run the existing witness

1. Refresh Flow PR #99 through ComfyUI Patcher and restart ComfyUI.
2. Open `workflows/examples/prefix-projection-ab.workflow.json`, or add
   **MiniMax H3 Prefix Projection A/B** under **MiniMax H3 / diagnostics** to a
   separate small workflow.
3. Connect the **same native H3 video VAE** used for the captured generation to
   `video_vae`. The example selects `minimax_h3_video_vae_int8_convrot.safetensors`
   from the reported run; select your actual checkpoint if different.
   A preview VAE, audio VAE or decoder-only replacement cannot run
   this encode/decode experiment.
4. Set `bundle_path` to the existing uniform-source boundary bundle directory
   containing `manifest.json` and its original `.bin` files, or the manifest
   itself. A previous numerical audit JSON alone is insufficient. Linux paths
   and WSL share paths for the running distribution are supported.
5. Set `chunk_join_frame` to the first generated assembled frame. For a join at
   175, use **175**. This changes report labels only; source/target dimensions and
   prefix length come from the verified witness.
6. Select `static_roi_profile=01784_room` for the bookshelf/picture/curtain/wall
   scene; use `custom` and `static_roi_json` for a different scene. Custom ROIs
   are fractional `[x0,y0,x1,y1]` rectangles. Use fixed background on both sides
   of the frame; exclude actors and intended moving objects.
7. Leave `feature_tracking_enabled=true`. OpenCV is optional: its absence is
   recorded as `opencv_unavailable`; photometry/detail measurements still run.
   `static_roi_profile=off` requires tracking to be disabled.
8. Queue **only this diagnostic workflow**. The example has no H3 model,
   sampler, Continuum assembly, upscaler or audio nodes. It needs no H3 rerun.

For a 62×62 target grid, 44×44 source grid and 12-token protected prefix, each
IMAGE output contains **39 frames at 704×704**. With a join at 175, these are
frames **136–174**, not generated frames after the join. The example saves and
previews all three PNG frame sequences using ComfyUI **Save Image** nodes:

| Output | Folder beneath your configured ComfyUI output directory |
|---|---|
| `rgb_reference` | `h3_flow_regenerate/prefix_projection_audits/rgb_reference/` |
| `latent_bicubic` | `h3_flow_regenerate/prefix_projection_audits/latent_bicubic/` |
| `vae_rgb_roundtrip` | `h3_flow_regenerate/prefix_projection_audits/vae_rgb_roundtrip/` |

Each folder receives 39 numbered PNGs per room-example run, in frame order
136–174. Counters in filenames are save counters, not assembled frame labels.
These are 8-bit RGB visual exports; report metrics use the original float32
pixels before export. No video files are created automatically.

The earlier workflow used **Preview Image** nodes, which wrote only temporary
PNGs to your configured ComfyUI temp directory (normally `ComfyUI/temp/`, with
names like `ComfyUI_temp_abcde_00001_.png`). Those runs did not save permanent
image outputs. Refresh through Patcher and load the corrected example to save
all three sequences; rerun only the VAE diagnostic, using the existing witness.

To inspect animation, connect each IMAGE output separately to
VHS Video Combine at **24 fps**, with no audio, selecting `video/ffv1-mkv`.
Keep identical save settings for all three. Do not judge geometry or local
contrast from an independently resized or differently encoded preview.

| Output | Construction |
|---|---|
| `rgb_reference` | Exact saved target prefix → native target VAE decode → spatial antialiased bicubic RGB resize → clamp to [0,1] |
| `latent_bicubic` | Exact saved target prefix → production latent bicubic resize → native source VAE decode |
| `vae_rgb_roundtrip` | The **same** `rgb_reference` → native H3 encoder → native source VAE decode |
| `numerical_report` | Paired metrics, witness identity, context controls, call timing and CUDA memory samples |

The node writes a fresh `prefix-projection-*.json` under
`output/h3_flow_regenerate/prefix_projection_audits/`. It verifies operand hashes,
native timing, the saved source-prefix projection, exact final target-prefix
bytes and protected/editable high-mask ownership before any VAE evaluation.
The saved bundle is never rewritten. Native encoder output is already
normalized; latent mean/std or model scaling is not applied a second time.

## Read the paired result

`comparisons.<method>.regions.<region>.frames` contains absolute RGB means,
RGB RMSE, luma mean/bias/RMSE, local luma contrast, Sobel amplitude/error,
3px/9px box-highpass amplitude/error and 11px luma SSIM. It also reports errors
in adjacent-frame RGB increments, retaining real reference motion.

`roundtrip_minus_bicubic_summary` reports **B minus A** for each region. Negative
RMSE, absolute-bias and detail-error differences favor the roundtrip; positive
SSIM differences favor it. Signed mean luma bias alone is insufficient because
positive and negative errors can cancel. Raw sharpness alone is insufficient
because both oversharpening and missing detail can change its amplitude.

`same_time_geometry` matches each reference/candidate frame pair with
forward/backward LK and RANSAC, using background ROIs. Its displacement units
are source-canvas pixels. Inspect supported measurements and inlier counts;
indeterminate fits do not establish zero displacement. A global similarity fit
does not establish local shape or actor-pose preservation: inspect the IMAGE
outputs at native scale as well.

## Resolve terminal decoder context

A protected latent prefix does not imply its final decoded pixels are
independent of future tokens. The native H3 decoder uses overlapping windows.
The candidate uses an isolated prefix because future generated source tokens
are not available when constructing the low-stage input.

Two extra decodes compare isolated target/source prefixes with the **same
saved prefix plus five following saved tokens**. See
`decoder_context_controls.target_prefix_isolated_vs_saved_future` and
`source_prefix_isolated_vs_saved_future`. These controls preserve prefix bytes
and native temporal phase. They use saved generated future context, not unknown
future ground truth.

`comparisons_against_target_with_saved_future` repeats A/B against target RGB
with that saved future context. Review the last five frames in both comparison
tables. An apparent gain only against terminally padded RGB, which disappears
or reverses against the contextual target, is not evidence of improved
production prefix consistency.

## Acceptance and next stage

The representation-consistency hypothesis is supported when the roundtrip
reduces paired brightness/RGB/detail errors in the problematic background,
including the final protected frames, while preserving geometric correspondence
and temporal behavior relative to A. Improvements must survive the saved-future
reference control and visual inspection. A smoother result with worse Sobel or
highpass correspondence, changed structures/pose, new motion, increased end-frame
error or temporal instability rejects the roundtrip as a continuation input.
Report tradeoffs rather than averaging a damaged region into full-frame gains.

There is deliberately no automatic numerical threshold that enables production
sampling. **No `source_prefix_projection` sampler setting is added by this
stage.** Only a supported GPU result justifies integrating a once-per-chunk
roundtrip candidate; the authoritative target prefix, audio, masks, schedule,
backend and generated suffix must retain their existing contracts. A successful
VAE-only result still does not prove that continuation expansion or speech is
fixed. A later continuation A/B must retain the first chunk, seed, prefix hash,
checkpoint set, geometry and schedule, changing only source-prefix projection.

Return the report JSON, the selected video-VAE checkpoint name and the three
saved PNG sequences (or native-scale corresponding
frames, especially the last five). The report records each of the **five
decode calls and one encode call**, synchronized CUDA timing where applicable,
allocated/reserved memory before/after calls, and the cumulative process peak.
It does not reset global CUDA peak counters. About **0.65 GiB** of CPU memory is
required for the three 39-frame float32 outputs at 704×704; transient decode and
metric buffers require additional RAM. GPU workspace depends on the VAE and its
precision. Existing resident models are preserved; insufficient memory stops
rather than entering the managed model-unloading retry path.
