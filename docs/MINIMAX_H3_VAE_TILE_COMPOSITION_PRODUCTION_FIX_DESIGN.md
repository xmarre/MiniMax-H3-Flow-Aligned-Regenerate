# MiniMax-H3 video VAE: normalized spatial tile composition

Status: implementation-ready composition correction; learned-decoder visual acceptance remains pending. Design-only work, 2026-09-20. No production implementation is included.

## 1. Decision and scope

Implement the correction in **ComfyUI Core's MiniMaxH3VideoVAE spatial decode compositor**. Keep the released 256 px decoder windows, native minimum overlap 64 px, tile-local positions, neural decoder, latent normalization and temporal logic. Replace the raw-neighbor blend/crop sequence with separable, normalized overlap-add over every contributing decoded tile. Preserve the existing spatial tile plan; no new node, workflow option, model weights or global monkey patch is required.

The source has two demonstrable composition defects: a missing diagonal contribution where X and Y blends intersect, and loss of older contributions where overlap redistribution creates triple coverage. These defects create seams even with synthetic constant tile outputs, without any decoder context starvation. They exist in the official MiniMax implementation as well as Core. Reference agreement therefore does not establish composition correctness.

This design does **not** claim that correcting those defects is already proven to eliminate every learned-decoder artifact. The runtime gate distinguishes the demonstrated compositor defect from any residual tile-context disagreement. No evidence currently establishes a safe halo width or trusted interior. Do not introduce cropping, larger windows, global RoPE, or post-decode smoothing to fill that evidentiary gap.

Excluded: continuation frame shift, audio discontinuities, sampling, Sol/VDN arithmetic, exact-prefix contracts, VAE encoding and neural decoder changes. Flow #52 remains historical diagnostic evidence.

Evidence labels used below: **Fact** means source inspection, recomputed measurement, or executed arithmetic; **Inference** means supported causal interpretation; **Decision** is the specified architecture; **Open** identifies missing evidence; **Recheck** requires implementation-time verification.

## 2. Source and repository provenance

All refs below were fetched during this investigation. Branch names and review status can change; implementations must refetch them.

| Repository / role | Exact inspected revision | Topology / finding |
| --- | --- | --- |
| Flow main | `b659b311fa548c7e3275db0e6f0a05c037dac730` | Design branch starts here |
| Flow #49 | `d1b32f530b4cc2298eba68f08c2da44a094c7cfe` | `mirror/exact-prefix-progressive-v2-20260917`, base main |
| Flow #52 | `c7322b9f1c983ff12566337842b97d8390993344` | `feature/h3-vae-large-tile-decode-20260920`, one commit on #49 |
| Continuum Plus main | `fb435c526643ca71afcac827df82ee6b2e811c3a` | Canonical repo is `xmarre/ComfyUI-H3-Continuum-Plus` |
| Continuum Plus #24 | `6167dcbae8ab239ea660fed52d03ddb63aa515fb` | `mirror/production-phase-aware-audio-20260913`, base main |
| Continuum Plus #28 | `688c754b19664b3812d462af6d5a3414bfc1b06b` | `mirror/pr24-decoded-trajectory-diag-20260919`, base #24 |
| ComfyUI master | `c8ed2c8ce957475459731135c4ca31c6856a4542` | Production owner |
| ComfyUI v0.36.0 | `ee71d5c4993f29086b27fde1629a945ae48425bf` | VAE file identical to inspected master |
| Official MiniMax-AI/MiniMax-H3 | `d21241f0a4b3acbb34c97dae47fa417b7065e438` | Released window, local positions, same compositor defect |
| Independent ostris/ai-toolkit port | `8fa15e356939a922b8fd3307610bf03100c803f6` | Corroborating implementation; not the model authority |

The handoff shortened Continuum #28's SHA by one trailing character; the full fetched SHA above is authoritative. Its older repository URL redirects, and some API operations do not follow that redirect.

Reviews: no submitted reviews on the four scoped PRs at audit time. Conversations were read; automated review-skipped notices do not constitute review. Exact-head successful Actions include Flow #49 `35485173805` / `35485172930`, Flow #52 `35482907312` / `35482905053`, Continuum #28 `35485716274`. The exact #24 SHA query returned no workflow runs; #28's successful stack includes #24. These are existing stack checks, not validation of this correction.

### Runtime provenance limit

00536 reports `v0.36.0-16-gacbf3eb1`, local `patcher/stack`, PyTorch `2.10.0+cu130`, video VAE FP16. `acbf3eb1` is absent from the fetched upstream history and GitHub's upstream commit endpoint returns no such commit. Do not relabel the installed stack as upstream master.

Core's VAE Git blob is `36d9abb576228424cbec785f0e51f2be7f5fcee9` at both v0.36.0 and inspected master; file SHA-256 is `aae581a7da3a1fba6ed35c8d11912af139246830764c525c14900f1bfd07225f`. This establishes the upstream baseline, not the installed file identity. Before CUDA A/B, a local diagnostic must record the imported VAE module path, file hash, class/method ownership, full Patcher stack and model file hash. No manual Git steps are required from the user.

Workflow node `ver` metadata is stale and inconsistent across nodes (including Flow `82805de...` / `ebdff905...` and Continuum `4e0966e...`); it is not proof of effective runtime source. The diagnostic's printed profile and restoration receipts do prove its executed 256/128 behavior. PT212 proves decoded trajectory instrumentation ran. Actual imported-source hashes remain a runtime prerequisite.

## 3. Verified 00536 evidence

Primary files, preserved outside Git:

| File | SHA-256 |
| --- | --- |
| `Pasted text(20260920-034539).txt` | `cce8ae33bc2921eab8d1cfaae9c60c1f754e8644eaf8572d04e6bd848773217c` |
| `metrics_00536_.json` | `088d88ec6187d6e1aafd88644e457368a072d648d74561ad2d64f1857d9cbfe4` |
| `MiniMax_H3_00003-audio(2).mp4` | `2e64e2593f3932ac8b190223dabbe83c5a21bd598dee65ec821fea4fee685776` |
| `MiniMax_H3_00004-audio(5).mp4` | `1328b4cbbf262539284d8828c7b9c8194fbfdb7aa930902ae01ea5fb3f25f05e` |

Both MP4s contain 336 frames, 1216×896, 24 fps, H.264 CRF 19, plus AAC audio. Both contain the same complete workflow/prompt graph. The graph routes the same `304:0` latent and `208:0` FP16 VAE into native node 211 and diagnostic node 329. Diagnostic `329 -> 330 -> 331 -> 332` and native `211 -> 227 -> 245 -> 308` use matching assembly/finalizer settings. Node 304 obtains its latent from Continuum node 226. The graph is a same-sampled-latent design; no latent bytes or tensor hash were supplied for independent identity verification.

The log executes diagnostic writer 332 first, then native writer 308 with the same filename prefix. Together with the distinct measured boundary patterns this supports **00003 = diagnostic 256/128; 00004 = native 256/64**. This mapping is not inferred from filename order alone. Embedded `videopreview` filenames are stale (`00002` in both branches) and cannot identify the current output. There is no per-file output-node execution receipt, so retain that limit on provenance.

Both decoded groups print 256/128, 9×6 tiles, and restoration to 256/64. First-group active seam ratios include X=448:1.278, 576:1.356, 704:1.333 and Y=128:1.451, 256:1.388, 384:1.414. These are raw-decoder diagnostics; they are not numerically interchangeable with lossy MP4 measurements.

Recomputed all-frame measurement: at each boundary p, take mean absolute RGB difference between lines p and p−1, divide by the median of the other 24 adjacent-line differences within ±12 pixels, then average ratios over frames. The analysis uses decoded RGB float32 from FFmpeg, not optical-flow alignment.

| Boundary category | Diagnostic minus native seam ratio, all 336 frames |
| --- | --- |
| New-only X: 112,224,336,448,704,832 | +0.02403,+0.04175,+0.02981,+0.01294,+0.05027,+0.03117 |
| Old-only X: 192,384,768 | −0.04154,−0.02518,−0.04350 |
| New-only Y: 128,256,384,512 | +0.12411,+0.13297,+0.12765,+0.11256 |
| Old-only Y: 160,320,480 | −0.00892,−0.02075,−0.03281 |

These reproduce the earlier directional claim, not every earlier number; estimator/aggregation and lossy decoding matter. Early and late frame subsets show the same broad pattern. Frame indices in the MP4 are finalizer output indices; they must not be substituted for PT212's raw decoded group indices. All-frame difference heatmaps visibly contain the rectangular lattice. Scene edges and H.264 block artifacts prevent interpreting an individual seam ratio as ground truth.

PT212 upper45 values read from the log: diagnostic pre dy −0.1086 / first-three post median +2.6880 px; native −0.1088 / +2.6726 px. The overlap change does not resolve the separate continuation impulse. No claim is made that spatial composition can repair that defect.

## 4. Complete decode path and history

Core `comfy/sd.py:VAE.decode` loads/offloads through the normal model patcher, casts input to VAE dtype/device and uses `decode_output_shape` plus `output_buffer` for chunked I/O. `handles_tiling=True` delegates spatial tiling to this model. Public IMAGE conversion remains downstream: Core emits channels-last video, and VAEDecode/Continuum flatten `[B,T,H,W,C]` to `[B*T,H,W,C]` when needed. OOM fallback must continue to use model-owned tiling.

`MiniMaxH3VideoVAE.decode` denormalizes 24-channel latent samples once. Temporal decode slices the existing chunks, calls `_adaptive_decode`, applies its existing temporal overlap/cropping and finalizes ImageNet-normalized pixels to float32 [0,1]. Spatial tiling operates on raw decoder values **before** temporal blending and final clamp.

For a spatial tile, `_decode_pixels = decoder(post_quant_conv(z))`. The post-quant convolution is 1×1×1. Decoder is a 36-layer full-attention ViT with 32 heads, 64 head channels, four learned register tokens and a zero suffix token; patch projection emits 4×16×16 pixels per latent token. Coordinates are `2*(arange(n)+0.5)/n-1`, separately for local T,H,W, with zero suffix positions. The decoder is not a finite-radius spatial CNN. Encoder reflection padding and causal temporal convolution do not establish a decoder context halo. Spatial tiles are latent slices, not independently zero-padded CNN patches. Small external dimensions are passed at their native size.

Official sources: [configuration](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/FL2VA/video_vae/config.json), [split/blend/stitch](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/FL2VA/video_vae/klvae.py), [coordinates](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/FL2VA/video_vae/func.py), [decoder](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/FL2VA/video_vae/vae_vit.py). Ref2VA contains the same tiling logic. Official code supports separate encoder/decoder tile settings but released configuration is 256/64. That parameterization is not evidence that enlarged windows preserve quality.

[Core source](https://github.com/Comfy-Org/ComfyUI/blob/c8ed2c8ce957475459731135c4ca31c6856a4542/comfy/ldm/minimax/vae.py) history:

- `57500fc5bc92566a63f2046824f522cd55c335ca`, #15224: H3 support, including raw-neighbor composition defect.
- `16e3f3034f2bba1fff6c70cbd759339778555cd6`, #15268: raw parameter device casts.
- `bbda83647da6957e6c0ce52dd86f80b7a6501662`, #15334: INT8 ConvRot VAE support.
- `2a68ce33b4c9ea6ee4283e618a74560cefb32694`, #15446: chunked I/O/finalization and memory optimizations, not a spatial semantic repair.
- `b2e31e89412a01a67be599571cc57ff74b242a82`, #16187: kitchen operations and up-to-four-tile decode batching; raw tails unchanged.
- `f14bbe28697778b7c2427d4b71c7fac24b78f8f4`, #16332: generator/lifetime improvement, not blend semantics.

[Issue #15416](https://github.com/Comfy-Org/ComfyUI/issues/15416) explicitly narrows its report to single-latent-frame failure and reports clean legal video in the author's test. It neither establishes a halo nor disproves a scene-dependent compositor defect. The five supplied papers concern DMD2, Sol/Sol-Attn, VSA and Spectrum; inspected titles/abstracts provide no model-specific VAE stitching contract and are not evidence for changing this decoder.

## 5. Root-cause proof

Let A,B be top-left/top-right raw tile values at one shared image coordinate, C,D bottom-left/bottom-right, and a,b the incoming Y,X ramp fractions.

The current compositor computes:

`F_old = (1-b)*C + b*((1-a)*B + a*D)`.

Separable two-axis composition requires:

`F = (1-b)*((1-a)*A + a*C) + b*((1-a)*B + a*D)`.

Therefore `F_old-F = (1-a)*(1-b)*(C-A)`.

**Fact:** Core saves `new_tails` and `next_left_tail` before applying either blend. The horizontal operation consumes raw C, overwriting part of the vertical blend. The diagonal A has zero weight. At the next X write start, b=0 and output resets to C even when the preceding region contains a vertical mixture. The error occupies overlap rectangles and has jumps at write boundaries. All weights still sum to one; this is not a simple missing brightness denominator. A constant-all-tiles test alone will miss it.

Executed unchanged Core `split_tiles`, `blend` and `tiled_decode` via AST extraction with NumPy float64 tensor primitives and a synthetic decode iterator. For 448×448, top-row tiles=0, bottom-row tiles=1, no horizontal variation: at y=224, x=191 outputs 0.5 while x=192 outputs 1.0. At y=192 the maximum X boundary jump is 1.0. The mathematical compositor must be X-invariant. This is a geometry/arithmetic counterexample, not PyTorch/CUDA execution or a learned-output test.

**Second fact:** `split_tiles` can create triple coverage. At 1216 px, overlap 128 yields starts `[0,112,224,336,448,576,704,832,960]`, overlaps `[144,144,144,144,128,128,128,128]`. At x=224 the first tile still covers the pixel. Raw-neighbor blending drops its surviving contribution. A 1D sequence of constant tiles 0,1,2 gives x223=0.7708333 and x224=1, a 0.2291667 jump. Triple coverage is also possible with native overlap 64: length 464 gives starts `[0,96,208]`, overlaps `[160,144]`. Fixing only the missing diagonal with a relocated tail assignment is insufficient for arbitrary sizes.

**Inference:** these proved mechanisms explain why moving the tile/write lattice moves discontinuity energy and why more overlap does not cure it. Context-dependent disagreement C−A supplies amplitude, but the compositor turns disagreement into avoidable discontinuities. Actual learned-tile attribution and residual quality need raw-tile CUDA measurements.

## 6. Exact production algorithm

### Geometry and planning

Keep the existing `split_tiles` algorithm and its 16 px redistribution. For input `[B,24,T_l,H_l,W_l]`, pixel dimensions are `H=16*H_l`, `W=16*W_l`. Spatial planning is in integer **pixels**; slice bounds are divided by 16 exactly. Multi-tile axes use length 256 (16 latent cells); an axis ≤256 has one native-sized tile and no invented padding. Decode full input tiles; emitted values have `[B,3,4*T_clip,L_y,L_x]` before the existing temporal trim.

Default 1216×896: X starts `[0,192,384,576,768,960]`, Y starts `[0,160,320,480,640]`, 30 tiles per temporal decode call. Preserve exactly those input windows and their order. The diagnostic has 54 windows; production does not adopt that cost.

A final tile remains full length and ends at the image boundary for legal latent-derived dimensions; irregular strides absorb the slack. There is no partial off-image tile and no new external padding. Image sizes presented to this decode interface are already multiples of 16. Do not change unrelated encode-side rounding or reject working workflows.

### Weight definition

For one axis, tile i has start s_i, length L_i, incoming overlap o_left and outgoing overlap o_right from the existing plan. Define local integer u=0…L_i−1:

```
in_i(u)  = 1                                  if i is first
           min(1, u/o_left)                    otherwise
out_i(u) = 1                                  if i is last
           min(1, (L_i-u)/o_right)             otherwise
q_i(u)   = in_i(u) * out_i(u)
Z(x)     = sum_i q_i(x-s_i) over covering tiles
p_i(x)   = q_i(x-s_i) / Z(x) within support; 0 outside
```

No division for an absent/zero overlap; its factor is one. Do not change the ramp to inclusive linspace: the existing incoming ramp is k/o and the outgoing complement is 1−k/o. External-facing edges are not tapered. For two-tile support this preserves the native one-dimensional ramp exactly in real arithmetic. Product ramps provide continuous membership at both ends when incoming and outgoing overlap intersect; explicit normalization handles three or more contributors. This is a conservative extension of the released ramp, not a decoder-confidence estimate.

Using independently normalized axis weights, output is:

`F[...,y,x] = sum_(i,j) p_y_i(y) * p_x_j(x) * D_ij[...,y-s_y_i,x-s_x_j]`.

The weights are nonnegative, sum to one, and are independent of image content, batch and time. All covering tiles contribute, including the diagonal. One-axis denominators suffice because the tile grid is Cartesian: the two-dimensional denominator factorizes. No full-frame weight/count tensor is needed. Never clamp a zero denominator to epsilon to hide a coverage bug; geometry tests must prove strictly positive coverage, and an implementation invariant may fail clearly if violated.

### Streaming and precision

Keep `_decode_tile_row` and its memory-aware grouping; do not retain all raw decoded tiles. Accumulate in float32 in fixed row-major order. Precompute axis-local float32 weight vectors and normalization, scoped to this call. Return the raw spatial canvas in the decoder output dtype so temporal blend/finalization contracts remain unchanged. Never clamp/normalize pixels per tile.

Maintain a float32 active Y band of at most `min(256,H)` rows and W columns, with leading dimensions `[B,3,4*T_clip]`. Each tile adds its weighted value into its absolute coordinate support. After all X tiles of row i have contributed, coordinates strictly below the next row start can receive no future contributions: cast/write that prefix to the raw output canvas and release/zero its band slots. Last row flushes through H. A circular band can use absolute y modulo band height; split writes at wrap, and zero only finalized slots before reuse. This avoids repeatedly copying large tails. A simpler bounded band implementation is acceptable only if measured memory/time satisfy the same contract.

Output cursor advances to the next actual Y start, not an assumed stride. Triple/higher coverage stays in the active sum until every contributor has arrived. Avoid scatter atomics and parallel accumulation. Do not let a view pin a whole decoded batch after its last tile; retain Core's explicit tile lifetime discipline. Write the canvas once per finalized strip. On exception, no shared configuration, accumulator or cached partial output survives.

For one spatial tile, use the existing direct tile decode route without weighting, accumulation, recasting or an extra decoder invocation. Preserve its batching and temporal routing so bit-identical native behavior is testable on the same backend. `tiling=False` remains unchanged. This single-tile requirement does not assert equality of large untiled attention and tiled decoding.

## 7. Ownership, API, serialization and concurrency

Primary changes should be confined to `comfy/ldm/minimax/vae.py`, targeted tests, and the existing H3 decode memory estimate in `comfy/sd.py` if required by the measured scratch change. Reuse a small private axis-weight helper; do not introduce a general tiling framework. Core owns the model's spatial semantics; Flow and Continuum must not implement competing compositors.

No public signature or serialized input needs to change for the production correction. Keep `decode(z, output_buffer=None)`, `decode_tiled`, `_adaptive_decode`, `VAE.decode` result contracts, state-dict keys and all node identifiers. Existing ordinary workflows automatically receive the corrected spatial composition after the Core PR overlay. No checkpoint conversion/re-encode is required. The encoder also has raw-neighbor stitching; changing it would alter conditioning latents and is explicitly outside this decode correction.

The production path needs no temporary profile mutation: read the existing settings once at entry into a call-local plan, then use only that plan and local weights/state. No global cache or decode lock is necessary for the new compositor. This removes new composition-state races; it does not claim Core's GPU loading/offloading is universally thread-safe.

Flow #52's lock serializes that diagnostic module only; ordinary VAEDecode does not acquire it. Its finally restoration does not make concurrent ordinary decoding safe. Preserve #52 unchanged, and run historical diagnostic A/B sequentially in an isolated process. The normal Core correction must not depend on that module. If a later bounded harness needs multiple profiles within one process, use a private model-owned function receiving explicit immutable plan/composition arguments, without swapping instance methods or attributes. No unused model-specific option should be added to the shared Core API merely for future use. Baseline/candidate Patcher overlays in separate fresh processes remain a valid A/B route.

Flow #49/#52 and Continuum #24/#28 stay untouched. Do not combine this work with Flow #54 / VDN #26. A future production implementation belongs on a separate Core mirror and its own Core PR; all user handoff changes must be on that PR. The design PR in Flow is documentation/analysis only and should not be enabled as a runtime overlay.

## 8. Alternatives and context questions

| Approach | Disposition and reason |
| --- | --- |
| Correct normalized overlap-add, native plan | Chosen; directly repairs proved weight/topology defects with no extra neural tiles |
| Move left-tail capture after Y blending | Insufficient alone; leaves triple-coverage ancestor loss and needs careful Y recursion |
| Halo + trusted-center crop | Deferred; full tile attention provides no finite receptive-field cutoff or measured safe margin |
| Hard crop-first stitch | Rejected as first fix; converts disagreement into a new hard boundary, with no trusted-region evidence |
| Cropped interiors plus normalized blend | Conditional future option only after residual context-distance measurements |
| Increase overlap to 128 | Diagnostic, not correction; moves the lattice, increases tile count and introduces additional triple coverage |
| Change split_tiles to a regular grid | Unnecessary for demonstrated defect; changes decoder contexts and external-edge policy, confounding attribution |
| Global-position RoPE or 320 px windows | Outside released coordinate/window contract; 320 also has negative runtime evidence |
| Final image blur/filter/flow smoothing | Prohibited; does not correct decode composition |

Open: no measurement identifies the distance from a tile edge at which output becomes reliable. Global attention and register tokens allow context differences throughout the tile; a fixed halo cannot be proved from kernel sizes. If residuals survive composition repair, capture unblended outputs from matching 256 px windows and group same-coordinate disagreements by distance to every edge, corner, local token phase and content type. Use distances 0,16,32,48,64,80,96,112,128 pixels, valid shifted windows only, fixed temporal context. Agreement between tiles is not truth; compare controlled reconstruction fixtures and central-window consensus where supported. A margin is eligible only if stable across held-out scenes/dtypes/resolutions and gives complete coverage with acceptable cost. If no stable interior exists, do not advertise cropping as a correct decoder fix.

Untiled decoding beyond 256 changes the decoder's domain and is not an acceptance oracle. Where the image fits one native tile, tiled/direct paths must agree exactly. Large-domain experiments remain excluded unless new authoritative model evidence changes that contract.

## 9. Memory and runtime implications

At native geometry, neural decode work remains 30 tiles per temporal call at 1216×896. Diagnostic 128 requires 54, a 1.8× tile count, not a measured wall-time ratio. The chosen correction does not incur that multiplier.

For B=1, C=3, raw T_clip=7 -> decoded T=28, W=1216, H=896:

- Existing FP16 raw canvas: 174.5625 MiB; preserved.
- Float32 active band, 256 rows: 99.75 MiB.
- One float32 tile temporary: at most 21 MiB if the chosen tensor expression requires it.
- Axis weights: small 1D vectors, not a full video-sized denominator.

Old raw row tails are removed, so these are component bounds, not a measured net peak. Decoder batch activations, generator views, temporal overlap and output transfer lifetimes still matter. Scratch scales linearly with B, raw temporal frames and W. No full-video GPU accumulator is allowed. Audit/update H3's memory reservation formula to include simultaneously live band, cast workspace and canvas; preserve dynamic VRAM/offload ownership. Do not rely only on free-memory batching after an underestimated reservation.

Additional weighting/addition bandwidth and float32 accumulation have an unmeasured time cost. Measure synchronized decode-only wall and allocated/reserved peaks; do not infer speed from operation counts. Record cold load separately from warm decode and keep tile grouping fixed for quality comparisons. No claimed VRAM or speed improvement is supported yet.

## 10. Concrete implementation sequence

1. Refetch Core and scoped stack, instructions, histories, CI and reviews. Verify installed VAE source and model provenance. Create independent Core implementation mirror/checkpoint; preserve all existing PR heads.
2. Add targeted CPU regression tests around Core's actual spatial methods with a fake decoder. Prove the 2×2 axis-invariance and triple-overlap failures before editing. Port the independent formula oracle to real PyTorch tests; the NumPy adapter is evidence, not a substitute for backend tests.
3. Add minimal private axis-weight planning and normalized composition. Preserve split_tiles and neural `_decode_tile_row`. Implement bounded active-band streaming, a direct single-tile path, deterministic accumulation and raw output dtype.
4. Audit live allocations and adjust only H3 memory estimation if necessary. Check current kitchen/shared helpers for an appropriate weighted accumulation primitive; reuse it only if semantics, dtype and deterministic order match. Do not write a new CUDA kernel for this task.
5. Test integration through VAE.decode/output_buffer and model-owned tiled fallback. Verify IMAGE/Continuum output contracts and temporal frame counts without modifying consumers.
6. Publish the implementation and tests on a separate Core PR through the established fork/mirror topology. Checkpoint before long CUDA validation. Provide Patcher overlay instructions with exact PR head; no manual Git workflow for the user.
7. Execute bounded same-latent runtime validation below. If composition tests pass but visual gates fail, retain the proved correction as a candidate and investigate the measured residual. Do not silently promote, widen scope, or choose a halo from appearance alone.

## 11. Validation and acceptance gates

### CPU and CI

Use actual Core methods plus a fake decoder whose tiles are constant, global-coordinate affine, and deliberately different per tile. Tests must cover:

- 2×2 rows 0/1 and columns identical: X-invariance; transposed analogous case.
- Four distinct tile values against the bilinear formula, including diagonal weight.
- Triple and higher overlap, including 464×464 at native 64 and 1216×896 at 128.
- Constant reproduction and nonnegative partition of unity; coordinate placement without gaps, repeated pixels or offsets.
- Small/one-axis cases: 16×16, 256×256, 256×448, 448×256, 272×272, 464×464, 512×512, 1216×896 and transposed; latent batch B=1 and B=2, noncontiguous input where supported.
- Deterministic repeat, raw inputs unchanged, interleaved independent composition calls, injected failure then clean call.
- One-tile native bit identity, decoder call/window identity, output_buffer identity and full overwrite, correct dtype/device/channel order, temporal shape for T_l=1,2,7,12 and the real decode groups.
- Streamed output versus independent full-canvas float64 oracle; float64 synthetic partition error ≤1e−12. For float32, derive tolerance from summed absolute contributions and floating-point roundoff; include the final cast. Do not use a loose image-quality threshold to waive coordinate/weight failures.

Suggested new test target: `tests-unit/minimax/test_vae_tile_composition.py`, adapt to current Core test layout. Commands for the implementation agent: `python -m pytest -q tests-unit/minimax/test_vae_tile_composition.py` and the existing targeted VAE/Core node tests, plus the repository's required lint/CI gates. These tests do not exist yet and have not been run. Avoid full-suite repetition without a relevant failure.

Executed design evidence: extracted unchanged Core arithmetic counterexamples; 387 axis-weight oracle cases (lengths 16..2064 by 16, overlaps 64/128/240), positive denominators, maximum float64 partition error 4.440892098500626e−16. No PyTorch package, model execution or CUDA test was performed in this architecture session.

### Bounded runtime capture

Preserve the exact sampled latent, both decode groups, workflow settings, VAE checkpoint hash, effective imported source, backend/dtype, tile plan and raw output shape. Existing MP4s cannot reconstruct the latent. Capture the latent once with the established Patcher/workflow infrastructure, then reuse it; rerunning the sampler separately for each decoder is insufficient.

First attribution pass: native 256/64 and corrected 256/64 using identical raw tile outputs for one full temporal clip in each group. Capture tile hashes or a bounded tile bank to show the neural decoder was unchanged, and recompose offline. This isolates composition from decode batching/precision. The temporary capture is diagnostic code, not a permanent model output API. Include four adjacent tiles around an intersection and a triple-overlap geometry. Validate the predicted `(1-a)*(1-b)*(C-A)` residual on pairwise-support intersections before temporal blend/clamp.

Then decode the complete 00536 groups with native and candidate in isolated matched processes, output lossless RGB tensors/frames before Continuum assembly. Use the existing MP4s as visual/provenance context, not as numerical ground truth. Record optional assembled outputs under identical seam/finalizer settings. Check chunk 1 and chunk 2 independently; do not retune sampling. Confirm PT212 is not being used as the spatial acceptance gate.

Additional bounded cases: one native-window reconstruction clip, 464×464 (triple support), 512×512, 896×1216 portrait, and one larger nonuniform-stride legal size such as 1008×752. Use two scene classes: low-frequency/background gradients and fine texture/diagonal edges. Source real images/clips through unchanged encode for reconstruction tests; synthetic RGB fixtures alone do not represent generated-latent quality. Run FP16 first; separately smoke-test supported FP32 and quantized decoder paths without conflating their quality baselines. Preserve existing temporal input semantics, including T=1; do not fold issue #15416 into this fix.

Measure full-frame and edge/corner results, not selected crops only:

- Native and candidate boundaries, their union, overlap ends, and a full scan of all X/Y line positions.
- RGB adjacent-line gradients and residual gradients on matched same-latent outputs; phase-matched nearby control lines to account for the 16 px output patch grid.
- Whole-frame absolute/signed residual heatmaps and row/column projections; tile-phase and period-16 spectral summaries with controls for texture/compression.
- Unblended same-coordinate tile disagreement versus distance to borders and corners if residual defects remain.
- Spatial defect persistence over frames, consecutive-frame difference maps and tile-phase temporal modulation; protect moving texture/scene edges from being mistaken for seams.
- Peak allocated/reserved GPU memory, CPU output memory, decoder invocation/grouping counts and synchronized cold/warm wall time. Three warm repeats suffice initially; report dispersion and separate model load.

### Promotion criteria

Structural gates are exact: correct coverage/shape, no configuration mutation, no extra neural tiles, unchanged input windows/temporal contract, same-source one-tile identity, and agreement with the mathematical oracle within justified arithmetic error. A native compositor failing those tests is expected; the correction must pass them.

Visual gate: remove the measured artificial boundary response without introducing new boundary peaks, period-16 checkerboard, temporal flicker, detail loss or corner defects. Native-versus-candidate difference alone is not a quality score. Calibrate residual boundary excess against same-scene phase-matched off-boundary controls and repeat-decode numerical floor; estimate uncertainty with temporal blocks rather than treating correlated frames as independent. Freeze that baseline-derived limit before evaluating held-out candidate clips. Require candidate boundary excess to fall within the calibrated control envelope across cases, with no unexplained full-frame residual lattice and visual review at 1:1. No arbitrary PSNR/seam ratio threshold is justified by the present lossy MP4s.

Performance gate: same tile count is mandatory. Measure and disclose net memory and latency; no universal percentage budget has been established. Any regression beyond repeat variability needs explicit review of its size and required precision, and must fit supported memory reservations without OOM. A resource regression cannot be hidden by smaller batches or reduced output quality. CUDA visual/performance gates remain required even if CI is green.

## 12. Dead Ends Proved False / Rejected Directions

- **Empirically rejected, supplied prior runtime:** 320 px decoder extent produced an approximately 16 px checkerboard in 00534. Its original media was not reanalyzed here; preserve the historical report and checkpoint `checkpoint/pr52-global-position-dead-end-20260920` (`1c6992b49635c6a049fc9a7730127187a1f445ea`). Rejection covers this implementation/model/domain, not every possible retrained decoder.
- **Source-contract rejection:** global-position rewrite contradicts official local normalized coordinates and changes decoder semantics. It is not a universal impossibility proof.
- **Falsified sufficiency:** more overlap alone does not eliminate the rectangular defect; 00536 moves the measured lattice. It is useful as a discriminator only.
- **Falsified simple explanation:** missing scalar normalization alone cannot explain the source counterexample, since old local coefficients already sum to one. The wrong contributors/weights are the defect.
- **Falsified sole-cause hypothesis:** context starvation is unnecessary to create the demonstrated discontinuity; synthetic constant tile outputs reproduce it. Learned context disagreement may still contribute residuals.
- **Falsified recent-regression hypothesis:** raw-neighbor errors predate current batching/kitchen changes and appear in official and original Core code.
- **Rejected for a separate defect:** the overlap A/B leaves continuation motion impulse effectively unchanged. Do not restart frame-shift repair in this task.
- **Superseded incomplete repair:** moving one tail capture is not sufficient when triple overlaps occur.
- **Unresolved:** safe interior width, residual learned context sensitivity, exact installed patch-stack file identity, candidate quality/latency/VRAM. These are not proved-false directions.

## 13. Remaining questions and re-verification

There is no missing algorithmic decision blocking implementation of the proved composition correction. Runtime access is required to establish that it fully resolves the user's visual artifact. Installed source/model hashes and raw same-latent evidence block acceptance, not writing CPU tests or the candidate implementation. A halo design remains blocked on measured residual context behavior and is not authorized by this specification as an automatic fallback.

Recheck live Core file locations, API kwargs/output-buffer behavior, model-owned OOM routing, attention/quantized batching, memory estimate, AGENTS instructions and relevant PR topology before editing. Follow symbols/contracts rather than stale line numbers. If upstream fixes composition or changes the model contract, document the discrepancy and adapt the smallest correct patch; do not blindly recreate obsolete code.

## 14. Artifact registry and reproducibility

Git-tracked analysis under `docs/analysis/vae_tile_composition/`:

- `composition_counterexample.py`: source-extracted geometry reproduction plus independent weight oracle; preserve and inspect. Expects the inspected Core checkout at `core/` relative to its working directory. NumPy primitive adapter is explicitly not a torch backend test.
- `composition_counterexample.json`: executed arithmetic results; preserve.
- `compare_media.py`: FFmpeg/NumPy/Pillow 336-frame comparison; preserve/reproduce. Expects primary files in `evidence/ComfyUI-Sol-H3/` and writes `analysis/results/`.
- `media_metrics.json`: exact all-frame and early/late aggregate data, positions and media hashes; preserve. The early/late grouping at 175 is an exported-frame subset, not an asserted raw decode-group mapping.

Non-git primary evidence remains in the user's files, exact identities:

- log: `libfile_a1814e16bca08191bb056d81e708bf61`;
- metrics: `libfile_e6249c02fe2c8191b03cdd648d089300`;
- 00003 video: `libfile_91c9dbd8d45881918adb456d8eef870b`;
- 00004 video: `libfile_b6e5120cd6a88191a1b5efce64e1f02b`.

Materialized originals: `/workspace/scratch/9776e5c01f58/evidence/ComfyUI-Sol-H3/`, with filenames and hashes above. Preserve originals; retrieve by exact filename/identity if this transient path disappears. Do not publish prompts, entire workflow metadata or source media to GitHub merely to support the design.

Derived visual evidence: `/workspace/scratch/9776e5c01f58/analysis/results/comparison_sheet.jpg`, `mean_abs_diff_x10.png`, and selected `v3_fNNN.png` / `v4_fNNN.png` at frames 0,48,120,174,175,192,240,300. These are reproducible from the committed script and originals; implementer should inspect the sheet/heatmap. A separate saved evidence archive preserves these visuals. Raw ffprobe JSON in `/tmp/` is disposable because it is reproducible and includes private workflow content. The failed `/tmp/installed_vae.py` extraction is not evidence and must not be used.

Design checkpoint: `3a037cb24dabbf8b28f7fe06c43e69ba995b56e6` on `design/h3-vae-tile-composition-20260920`, before arithmetic/media work. Subsequent commits on that branch preserve completed analysis and the final specification. Historical Flow #52 is unmodified; its existing green/native-overlap and rejected-global-position checkpoint branches remain separate.
