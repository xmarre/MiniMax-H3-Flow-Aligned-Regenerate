# Continuation 01119: high policy executes; regional motion changes the diagnosis

Run 01119 exercises the protected-prefix attention and startup policy from Flow
#93 `6fc1aeedc498ae2079f19b9585580464d1464341`. Every actual high call has the
required target-grid receipt. This is hardware policy verification, not a
matched comparison or rendered acceptance of the original frame/tone complaint.

The supplied assembled video permits a distinction that the earlier latent
window alone could not establish: its whole-frame phase estimate follows the
running dinosaur, while a selected forest region has little vertical movement.
The large phase estimate must not be interpreted as a calibrated, uniform
translation of the frame. Local deformation, motion cadence, other regions,
zoom and tone remain separate questions.

## Executed contracts

- Continuation uses `34x56 -> 48x80`, 47 video tokens and 12 exact prefix tokens.
  The seed is `14766695179764173950`; the assembled video is `1280x768`, 24 fps.
  Both the seed and geometry differ from 01115.
- `same_grid_target_control` is inactive. This is the actual progressive arm,
  not a render of the accepted equal-grid control.
- All four actual high evaluations report equal source/target patch rows
  (`960` per frame), unit key measure, protected-prefix attention, native
  generated-query policy, and native target-audio positions. The refinement
  source is `h3_flow_partitioned_refinement`; the startup exemption is false.
  Sol is configured for one dense evaluation and two dense layers.
- The continuation has 17 logical calls: 11 actual and six forecast. High has
  six logical calls, four actual. The existing three sampler lifetimes remain.
  VDN's high uniform-grid linear and pre-RoPE paths each execute 200 times.
- Exact audio mode is `sampler_mask_exact_timestep`, requested width 16,
  effective width zero. Low/probe use the selected `source_carrier` audio
  position domain; high uses its native target domain.
- All eight tensor hashes, shapes and finite-value checks pass. Pre-high and
  final prefix bytes are authoritative; the initial window mask is exactly
  `[0,0,1,1,1,1,1]`. Provider-to-pre-high mutation changes only the first
  generated token's channel means. Recovered generated high residual RMS is
  `1.000632`; this verifies the saved input algebra, not its trained joint
  distribution.

The ordinary runtime gate passes with the high-attention and exact-audio
qualification flags. This does not qualify the rejected rigid registration,
unverified Auto Strength receipt, or rendered quality. Registration rejects
`boundary_upper45_degraded_over_bound`; the current one-token DC fallback runs.

## Native stage replay

The replay uses the unmodified Core VAE at
`6b4e05dc30d65740ce8931434607b9907996fb0e`, official unquantized fp16 weights,
CPU BF16 arithmetic, native PyTorch SDPA and all 28 spatial tiles. Four clean
stages are decoded: pre-high, first high before/after Flow, and final. Each uses
the captured seven-token window `[10,17)` with authoritative prediction-prefix
replacement. Sampler input and masks are never decoded as clean video.

The table measures the third retained transition, raw decoder frames `9 -> 10`
(assembled frames `125 -> 126`). It is inside the current window, beyond the
preceding temporal blend, and inside the first generated token.

| Clean stage | Upper45 dx / dy, px | Full dx / dy, px |
| --- | ---: | ---: |
| Pre-high DC | -2.550 / -2.751 | -2.941 / -0.537 |
| First high, before Flow | -3.196 / -3.024 | -3.052 / -2.929 |
| First high, after Flow | -3.134 / -2.897 | -3.045 / -2.922 |
| Final | -2.809 / -4.048 | -2.100 / -3.954 |
| Hardware final receipt | -2.809 / -4.048 | -2.094 / -3.949 |

Final replay dy differs from the hardware receipt by `-0.005000 px` in full
and `+0.000236 px` in upper45. Production loads
`minimax_h3_video_vae_int8_convrot.safetensors`; the independent replay uses
unquantized weights. Agreement here does not establish pixel equality or equal
photometry between the two decoder precisions.

The first high full-frame sequence starts `+0.433, -2.332, -2.929 px`; after
Flow it is `+0.455, -2.314, -2.922 px`. The first correction leaves that
sequence close to the native prediction. Final becomes
`-0.090, -0.729, -3.954 px`: subsequent refinement concentrates more of this
estimator's motion into the third transition. This differs from 01115's
near-zero pre-high motion and cannot be promoted to a matched treatment effect.
It also does not identify a particular model patch or checkpoint operation.

## Regional checks on the actual assembled video

The motion ROI is the top-right forest, pixels `[1024,0,1280,84]`, selected by
inspection. Lucas-Kanade tracks use a forward/backward error below 0.5 px;
reported displacement is the median of accepted tracks. The foreground tone
ROI is a dinosaur flank patch, `[240,150,520,260]`. Its current pixels are
aligned to the preceding patch using a partial affine fit with a 1 px RANSAC
reprojection bound. These are selected-region observations, not scene-wide
motion or exposure classification.

| Encoded-video pair | Forest dx / dy, px | Accepted forest tracks |
| --- | ---: | ---: |
| 122 -> 123 | -7.723 / -0.211 | 134 |
| 123 -> 124, first retained | -7.138 / -0.017 | 127 |
| 124 -> 125 | -7.951 / +0.152 | 125 |
| 125 -> 126, third retained | -8.207 / +0.154 | 124 |

The final native replay independently gives forest dy `+0.172 px` at
`125 -> 126`. The forest measurements contradict treating the approximately
`-4 px` whole-frame estimate as a uniform vertical shift. They do not certify
absent foreground hitch, local shock, scale change, or artifact outside the ROI.

The same whole-frame estimator applied to the encoded clip gives `-4.164 px`
at this transition; larger vertical displacements occur before the boundary.
Its magnitude is below the preceding chunk's 90th percentile of absolute
vertical displacement (`7.088 px`). Encoding changes the phase estimate from
the raw hardware receipt. A percentile comparison is context, not an acceptance
threshold or proof of natural motion.

For the tracked flank, `123 -> 124` changes mean luma `-0.584%`, population
deviation `+2.860%`, and p05 `-2.061%`. At `125 -> 126` these are `-1.958%`,
`+3.077%`, and `-6.148%`. Alignment residual RMS is `0.0182` and `0.0234`
respectively. Motion, nonrigid skin deformation, lighting and encoding affect
these values; they do not establish a uniform contrast/gain correction.

The production full-frame first-retained receipt changes mean by approximately
`-0.90%`, deviation by `+1.43%`, and p05 by `-7.83%`. All 22 sampled duplicate
overlap frames are equal, and Video Seam Auto replaces zero frames. These
ownership checks do not qualify newly generated tone. The existing PCM endpoint
correction runs with both endpoints exact; audio acceptance remains perceptual.

## Disposition and reproducibility

The attention invariant repair remains on #93. This review introduces no new
sampling correction: the supplied run verifies execution, while regional
analysis invalidates a uniform-shift interpretation of the global metric.
The original rendered complaint remains unqualified. A visible remaining
artifact would need localization in that region and the complete executing H3
call; seven decoder-window tokens cannot reconstruct 47-token video, audio,
conditioning and patch state or isolate the internal predictor's cause.

[Machine-readable evidence](CONTINUATION_01119_NATIVE_DECODE.json) retains source
hashes, runtime receipts, four native stage reports, regional measurements and
whole-clip motion context. The offline
[`audit_decoded_boundary_regions.py`](../../tools/audit_decoded_boundary_regions.py)
accepts the supplied video or replay `.pt` pixels and explicit pixel ROIs. It
records failed measurements as such and never turns them into zero displacement
or rendered acceptance. It requires OpenCV and NumPy; native pixels also require
PyTorch. For this video, use frame range `[114,140)` and the ROIs above. Native
replay pixel labels start at 122; its first boundary pair remains raw/unblended
because the preceding decoder window is absent.

The replay artifact writer now clones each stage before saving. A saved view
previously serialized the complete three-stage backing storage: 353,894,400
bytes per artifact rather than 117,964,800 owned pixel bytes. The change preserves
pixel values and prevents other stages from being copied into each artifact.
Existing replay admission and high-attention tests pass: **20 passed**. Both
offline tools pass Ruff and format checks. The original source-validated
sampling tree's **898-test** result remains applicable; no new GPU quality or
matched runtime improvement is inferred from these checks.

Evidence identities:

- Metrics: `56248b3d65f0b3c86dfdfc00bbe5e95c2ae071925cd931804a3fcb60ea49c5df`.
- Log: `64fe6b9dc5478125ee2ee01576abb2e752bd9f527a59f2dd0a5f8660d1e451c1`.
- Manifest: `815fdfaf4e64fd4de80dbd721c6c8298718b6d545b2a14891c0694bbec31b811`.
- Video: `bcd6e71dbf77d27148515af3d3ae51ea99b30c3e52a90c8a1e5a3c2de7119439`.
- VAE: `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`.

Before review, source is preserved at
`checkpoint/pr93-before-01119-review-20261004`. The replay storage correction is
preserved at `checkpoint/01119-native-replay-storage-20261004`. PR #93 retains
one xmarre-authored commit directly over Flow #89; prior heads remain recoverable.
