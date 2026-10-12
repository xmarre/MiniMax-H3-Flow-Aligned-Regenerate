# Boundary audit replay

Saved-bundle analysis does not change inference. Fitted motion and sharpness are observations, not automatic defect classifications. Camera and focus behavior can follow conditioning; slight darkening remains unresolved. No Spectrum cause is established.

Adjacent-frame geometry and RGB/luminance changes are measured for every stage.
Stage comparisons also report how much the temporal pixel increment changes,
which helps distinguish a stable refinement difference from one that develops
between adjacent frames. These measurements do not classify motion or cuts as
defects. The extra comparisons use already decoded pixels and add CPU fitting
work without additional VAE calls.

Set the optional `detail_region=upper_left` to also measure the left third of
the upper 45% of each decoded canvas. Its per-stage luminance, RGB changes,
adjacent-frame geometry and matching-time stage comparisons help inspect a
localized feature separately from broader motion. Extended scope also compares
that region across the two decoder contexts. Each result includes the region's
pixel bounds and canvas size; reduced-grid measurements retain their native
pixel units. The region can still contain motion or insufficient texture, so
these measurements are not automatic defect classifications. The default is
`off`. Region analysis adds CPU work without extra decoding or image export.

**Static-room zoom and background detail localization (01784/00040):**
The Local Boundary Audit also supports `static_roi_profile=01784_room`
for four named fractional-image regions: left bookshelf, framed picture,
right curtain and white wall. For a different scene, select
`static_roi_profile=custom` and provide `static_roi_json` such as
`{"books":[0.0,0.42,0.13,0.60],"curtain":[0.83,0.04,0.99,0.36]}`.
Names are arbitrary; the values are `[x0,y0,x1,y1]` fractions in
`[0,1]`, not pixel coordinates. The default is `off`, adding no work.

Set `chunk_join_frame=175` and
`audit_scope=transfer_and_decoder_context` for a complete 01784
boundary bundle. Within **each decoded stage independently**
(`source_grid`, `provider`, `pre_high`,
`first_high_before_flow`, `first_high_after_flow`, `final`),
`stages.<stage>.static_background_rois` reports stationary-region Sobel
gradient-energy ratios between the last prefix frame (174) and frame 180,
pre-join trend, local luma difference, phase-correlated position changes,
and an approximate horizontal background-scale slope from spatially
separated room landmarks. Texture-unstable or implausibly large
correspondences are marked unreliable rather than allowed to distort the
scale estimate. White-wall texture is useful for shading but can be
unsuitable for phase correlation. The full report records which landmarks
were supported.

**Feature-tracked background motion (new, optional):** Phase correlation
can return an apparent zero displacement when books, curtains and framed
images are progressively defocused. That is **not evidence of zero image
motion**. Set `feature_tracking_enabled=true` with a static ROI profile in
the Local Boundary Audit to add `stages.final.tracked_background_features`.
The **MiniMax H3 Image Geometry Probe (diagnostic)** also has
`feature_tracking_enabled` (default true), applied to the exact in-graph
IMAGE frames fed to Save Video. Both use sequential pyramidal Lucas–Kanade
optical flow, forward/backward consistency and RANSAC similarity estimation
from spatially separated background regions, with an f174 anchor for join
175. Inspect `frame_trajectories.180` / `.190`, including
`cumulative_apparent_scale_percent`, `ransac_inliers`,
`support_left/right` and `median_inlier_residual_px`. Ambiguous
correspondences report `low_confidence_indeterminate`, **not** a
fictitious 0% movement. Requires OpenCV (`cv2`) for the optional
diagnostic; when unavailable the report explicitly says
`opencv_unavailable` without preventing ordinary generation. The
tracker does not change a single production pixel, frame, latent, mask,
model call or decode. Its scale is an *apparent background-image*
similarity estimate and does not prove the physical camera was zooming.

For 01792 and the lossless 00008 case, do **not** infer encoder-only motion
from an in-graph phase-correlation reading near zero. Run the **same
feature tracker** on the in-graph frames and the lossless video, then
compare the decoded final stage, Finalize Duration output and file at
identical frame indices. Any downstream stage implicated must first
reproduce the nonzero tracked displacement; no spatial-warp correction
should be based on phase matching alone.

**Compare each stage's pre/post ratios**; raw 704px reduced-grid and
992px target-grid gradients have different native pixel bandwidths.
Each region therefore reports **both native-grid Sobel energy and a
common-grid (maximum 704px, antialiased-downsampled) energy ratio**, along
with prefix-baseline stability. Region registration uses diagnostic-only
subpixel phase correlation; only previously stable, plausible static
landmarks contribute to separate horizontal and vertical apparent-scale
fits. Ambiguous backgrounds (including a moving actor or texture-changing
wall) can still bias the estimate, so the reported scale is **not physical
camera ground truth**.

The report additionally contains
`static_background_same_frame_stage_pairs`: co-located ROI sharpness at the
same decoded times for `source_grid → provider → pre_high →
first_high_before_flow → first_high_after_flow → final`, resampled to a
**shared target canvas of no more than 704 pixels on its longest side**.
A stage's candidate/reference sharpness ratio is normalized to the same
stage pair on frame 174 before evaluating frames 176–181, which prevents
stable resolution-induced baseline differences from masquerading as a new
join defect. These numerical comparisons identify a candidate first
failure stage for subsequent manual review; they are not a production
acceptance check.
These static-ROI measurements only use RGB frames already decoded by
the audit; no additional VAE calls, sampling steps or production
adjustments occur, and the analysis does not modify source operands or
rendered frames. Combine the stage receipts with the saved capture
provenance before proposing a correction to the Flow path. The tone
correction remains owned by Continuum PR #42 and is separate.

Set `audit_scope=transfer_and_decoder_context` for the extended replay. It adds
the saved uniform reduced-grid handoff view and compares identical pixel times
from two independently decoded seven-token contexts near the band/tail edge for
each stage. The reduced view contains the saved source prefix and native
reduced-grid tail. Bicubic and VAE roundtrip prefixes reconstruct target context;
`native_source_carry` uses the preceding chunk's actual source clean prediction.
Its geometry
uses its own decoded pixel units, reported alongside the canvas dimensions.
The loader verifies its hash, exact tail ownership and the head projection
within float32 interpolation tolerance (`atol=1e-4`, `rtol=1e-5`).

Each window-context comparison uses the five overlapping pixel times. It can
show how the decoder's temporal context changes a frame without comparing two
different scene times. Standalone pixels are finalized and clamped separately;
production blends before clamping. The audit keeps the original native blended
replay and never assembles a replacement from standalone pixels. Extended mode
uses eighteen VAE calls, additional CPU fitting and one additional saved operand;
the default `stage_continuity` mode uses five calls. Both modes save numerical
JSON only and perform no diffusion generation or provider inference. Geometry
estimates and context disagreement still require comparison with visually
accepted output; they are not automatic defect classifications.

Uniform-source prefixes of at least seven tokens also receive an absolute
source/target prefix comparison, separating the preceding standalone window
from the following window with generated future context. This reuses existing
decodes and adds no VAE calls. See [paired prefix replay and interpretation](NATIVE_SOURCE_PREFIX_CARRY.md#compare-paired-prefix-states-without-regenerating).

**Decoder-context equivalence check (optional, expensive):**
Set `validate_full_video_decoder=true` on the **MiniMax H3 Local Boundary Audit**
to decode the *entire* saved final target-grid clean latent through the same
connected native H3 VAE, in addition to the existing cropped-window stage
replay. It compares identical global frames (e.g. 170–195 for join 175)
by per-frame RGB RMS/luma, full-frame diagnostic geometry and—when a static
ROI profile is enabled—same-time ROI sharpness and static-landmark
motion. Receipts appear at
`full_video_decoder_context_validation`, including the full-decoder global
origin, cropped origin, pixel-time labels and extra VAE call count.
The option defaults to **false**. It adds **one full-timeline VAE decode**
with potentially significant GPU/CPU memory cost, and may stop with OOM
rather than evicting other resident models. It does not change the
production video, sampling state, prompt or audio; it does **not**
automatically prove that the two assembled chunk decodes match the final
MP4. Differences would isolate context sensitivity of the native
decoder (crop vs full); matching pixels would redirect investigation
toward the external assembly/stitching path or rendered-image registration.

Only numerical JSON is saved in `output/h3_flow_regenerate/boundary_audits` and
returned by the node. Images, prompts and binary operands are not included in
the report. Original media and saved operands remain local and unmodified.
The replay needs complete full-video target-band snapshots and rejects
incomplete context or altered prefix/mask/band ownership. It uses
the connected VAE, including native quantized variants, rather than loading a
separate decoder checkpoint and does not modify generation behavior. It does not
compare against the prior chunk's retained pixels. Frames labeled before the
join are decoded chunk context discarded by assembly. Setting the
join frame to zero changes only labels. Crop length grows with band length;
VAE call count alone does not determine replay time.
Reports use `local_target_band_native_window_audit_v2`: the plan can contain
multiple shared-token pairs and blend intervals. Older v1 reports describe the
earlier two-window crop and remain historical evidence for their measured range.

| Node | Purpose |
|---|---|
| **MiniMax H3 Runtime Metrics Probe** | Passive sampler/model-call instrumentation. |
