# Native source-prefix decoder counterfactual (saved 01795 capture)

This is a **read-only diagnostic** for MiniMax-H3 continuation. It tests whether
replacing the decoder's source-grid protected prefix changes the *decoded*
geometry, tone, or texture of the same previously generated source latent suffix.
It **does not** modify the source sampler, perform denoising, change the
production continuation workflow, or establish a permanent defect correction.

## Experiment and operands

Load the original **01795 `capture_boundary_witness` bundle** with all its
unmodified `manifest.json` and `.bin` operands, and connect the exact
`minimax_h3_video_vae_int8_convrot.safetensors` native production VAE.

* Saved manifest SHA-256:
  `80a082e53f5a0c5ab433cc39701da21f7da4429b3c8f4754361e7a509d47bd56`
* Authoritative prefix SHA-256 (float32):
  `de374402b384b34645af023e9978c6666ef0b8d82cd621bc284bc6438dd91977`
* Source latent: `source_probe_clean_full` [1,24,62,44,44]; source VAE
  canvas 704×704.
* Authoritative target prefix: `authoritative_prefix_full`
  [1,24,12,62,62]; target VAE canvas 992×992.
* Previous production policy: `native_source_carry_v1`.
  That previous carried prefix is an actual source clean prediction at a
  nonzero sigma, not a target-grid latent.

The loader validates the bundle's original manifest, all supported stage
ownership/masks, the native source-carry projection receipt, individual operand
byte hashes, source dimensions, native temporal phase and model-internal clean
domain **before the first VAE call**. Only original bundle bytes are read.

**A** independently clones `source_probe_clean_full` without changing
any value. **B** clones the same full source tensor and replaces only
`[:, :, 0:12]` with the source-grid bicubic projection of
`authoritative_prefix_full`, using the existing
`resize_spatial_5d(..., mode="bicubic")` half-pixel image-grid contract.
Generated source tokens `[12:62]` remain bitwise identical. A and B share
source shape [1,24,62,44,44], output normalization and native VAE weights.

Two complete, sequential native VAE decodes retain production five-token /
two-overlap temporal stitching and pre-clamp mixing. The decoder's full outputs
are reduced to matching assembled pixel times f170–195 and released before the
next call. A third decode of the original Local Boundary Audit short source
window `[5:22]` independently measures whether that earlier **cropped**
replay differs from full-context A. The cropped replay is **not** substituted
for either primary counterfactual arm. Three VAE decodes, zero encodes, zero
sampler calls, zero H3 NFE are expected.

## Running in ComfyUI Patcher

Refresh **Flow PR #99** (not `main`) in ComfyUI Patcher and restart ComfyUI.
Add **MiniMax H3 Native Prefix Counterfactual** under
`MiniMax H3/diagnostics`. Supply only the native video VAE and the existing
saved 01795 bundle path, as a Linux/WSL-accessible directory or
`manifest.json`; do **not** connect a sampler or generate a new clip.

| Node input | Value |
|---|---|
| `video_vae` | Native H3 production VAE for the saved run |
| `bundle_path` | Existing complete 01795 capture directory or manifest |
| `chunk_join_frame` | `175` |
| `static_roi_profile` | `01784_room` |
| `static_roi_json` | Empty |
| `feature_tracking_enabled` | `true` |
| `expected_manifest_sha256` | Leave default 01795 SHA-256 |

The report is written to
`ComfyUI/output/h3_flow_regenerate/boundary_audits/native-prefix-counterfactual-*.json`.
Only the JSON numerical report is saved; there are no intermediate PNGs,
new videos, rewritten source operands or workflow-output modifications.

## Reading the JSON

* `identity`, `plan`, `generated_suffix_bitwise_identical`,
  `replacement_tokens`, and `changed_prefix_float32_elements`: exact
  operands, source/target domains, spatial projection and mutation check.
* `full_vs_cropped_baseline.same_frame`: same-frame full-native A
  versus existing cropped native source replay, including absolute RGB/luma,
  Sobel and structure comparisons. The two decoder contexts may disagree.
* `baseline` and `counterfactual`: individual ROI luma/Sobel and phase
  trajectories, adjacent-frame geometry, and OpenCV background tracking.
  The default profile includes bookshelf, framed picture, curtain and wall;
  `upper45_full` adds an upper-region diagnostic.
* `same_frame_A_to_B`, `same_frame_geometry`,
  `same_frame_geometry_upper45` and `temporal_increment_delta`:
  same pixel times in the two arms, including per-frame RGB RMSE and
  differences in frame-to-frame motion/appearance increments.
* `feature_tracking_from_f174` **and**
  `feature_tracking_from_f178`: separate apparent-scale trajectories.
  A smaller f190/f174 scale need not mean corrected continuation if B
  moved the decoded f174 anchor. The f178 anchor lies in the generated
  suffix, although the *decoded* f178 pixels can still depend on the prefix.
  Confidence failures, scene motion or texture changes remain indeterminate.
* `vae`: total VAE call counts and individual elapsed times. Inspect
  `limitations` before attributing image-space changes to a root cause.

### Causal interpretation

A substantial A/B post-join pixel difference, with identical suffix latent
bits and valid geometry support, establishes **decoder sensitivity** to the
source-prefix representation for the captured sequence. If later geometry
does not change, a dominant decoder-prefix contribution becomes less likely.
A difference in the f174 anchor alone must **never** be called a zoom fix.
A measured difference between full and cropped source replays is a temporal
context limitation of the prior measurements, not evidence of a changed source
trajectory. Different feature correspondences and static backgrounds can
change tracked scale independently of physical camera zoom.

A result in this diagnostic cannot change the existing source sampler's latent
drift, show restored rendered acceptance, establish causation for high-stage
darkening, or validate production audio. The source model is not invoked.

### Validation

Synthetic tests cover exact suffix/prefix ownership, bitwise immutability,
wrong shape/phase and mismatched provenance rejection before decoding,
equal-prefix identity, independent/dependent temporal fake decoders, no H3
sampling, and three VAE calls. These do **not** validate quality from trained
H3 VAE weights. The trained, captured 01795 replay must be separately
measured with this node before drawing an empirical conclusion.
