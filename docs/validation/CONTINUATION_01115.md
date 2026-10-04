# Continuation 01115: native tensor and decoder review

The supplied run contains all eight native boundary-window tensor operands. Their
bytes match the manifest and the runtime receipt. This is a continuation of the
failed DC-only rendered qualification, not an accepted boundary repair.

The reviewed Flow source starts at `71de5dd27299e2b86ee76ac8da267ff258fbe202`,
preserved at `checkpoint/01115-before-tensor-review-20261004`. The log does not
identify installed Flow source bytes; the executed policies are verified from
receipts and captured tensor operands.

## Captured arithmetic

The run uses `32x58 -> 46x82`, 47 temporal tokens, and 12 protected prefix tokens.
The native decoder boundary window is `[10,17)`, with two carried tokens followed
by five generated tokens. Its decoded chunk starts at frame 34 and the retained
suffix starts at local frame five, after the 39-frame trim. The raw decoder's
three-frame pre-padding makes that raw decoder frame eight, inside the first
generated token. The third retained frame is raw frame ten of that same token.

- Pre-high and final carried-prefix bytes exactly match the authoritative prefix.
- The initial video mask is exactly `[0,0,1,1,1,1,1]` throughout the window.
- Recovered high suffix residual RMS is `1.001194784`, with mean `-0.002145056`.
- The first predicted-prefix discrepancy is expected native inpaint augmentation:
  `0.999*exact + 0.001*noise`. Recovered injected prefix noise RMS is
  `1.000074185`; it is not a released/repainted prefix.
- Provider-to-pre-high mutation changes only the first generated token's channel
  means, with RMS `0.180843134`. Its centered residual is below `4e-8`; all later
  generated tokens are byte exact to provider output.
- First actual high prediction changes the generated tokens by approximately
  `0.35-0.43` RMS before Flow. Flow changes that prediction by `0.04-0.06` RMS.

The ordinary metrics/log replay passes: 17 logical calls, 11 actual and six
forecast, including six high calls with four actual. Requested/kernel Sol Q rows
both equal 3,944,000. This verifies runtime ownership and the selected exact-audio
alias with requested width 16 and applied width zero. It is not rendered
acceptance, frame-gauge approval, or a DoRA loader approval.

## Full native decoder replay

The offline replay uses the unmodified MiniMax-H3 VAE implementation from Core
`6b4e05dc30d65740ce8931434607b9907996fb0e` and the official
`Comfy-Org/MiniMax-H3` fp16 video VAE weights. Model input/output video conversion
is identity in Core's MiniMax-H3 video latent format; VAE channel normalization is
owned by the native decoder.

The completed replay executes all seven temporal tokens and the native spatial
blend across 28 tiles, producing `736x1312` frames. Latent tile origins are
`y=[0,10,20,30]`, `x=[0,11,22,33,44,55,66]`. Protected prediction tokens are
restored from the supplied authoritative bytes before decode. The six clean
stage comparisons do not rerun H3, the provider or the sampler.

The full-frame run uses CPU BF16 decoder arithmetic with native PyTorch SDPA.
Five stages were decoded together; the after-Flow stage was replayed separately
with the same checkpoint, decoder and spatial blend. Initial FP32 tile runs at
`(0,33)` and `(0,11)` showed region-dependent motion, so tile measurements alone
were not promoted to global evidence. The FP32 native CPU attention/SDPA tile
comparison has RGB RMS below `1.2e-7`; sampled BF16/FP32 tile differences are
approximately `0.0023-0.0026` RGB RMS. Production decoder precision and checkpoint
identity are not reported, so the full replay is not a pixel-exact hardware run.

The same pinned Continuum trajectory implementation measures all stages. The
table uses the transition into the third retained frame, raw decoder frames
`9 -> 10`. This pair is inside the current native window, beyond its temporal
blend, and inside the first generated latent token.

| Clean stage | Upper45 dx / dy, px | Full dx / dy, px |
| --- | ---: | ---: |
| Native provider, including its own prefix | -0.201 / -0.292 | +0.590 / -0.139 |
| Provider with authoritative prefix | +0.659 / +0.204 | +1.838 / +0.595 |
| Pre-high, exact prefix and one-token DC | +0.411 / +0.105 | +0.884 / +0.059 |
| First actual high prediction, before Flow | +0.811 / +3.280 | +2.384 / +2.615 |
| First actual high prediction, after Flow | +0.561 / +3.114 | +2.329 / +2.518 |
| Final clean output | +2.213 / +4.395 | +2.911 / +4.502 |
| Supplied final decoded receipt | +2.215 / +4.386 | +2.893 / +4.504 |

Final replay dy differs from the supplied receipt by `+0.008531 px` in upper45
and `-0.002738 px` in full. This reproduces the surviving within-token jump from
the captured final clean window with an independent native decode. It is not an
assembly-only shock: sampled RGB in the supplied duplicate overlap matches
exactly in all 22 frames, and seam assembly replaces zero frames.

The large third-frame jump is already present in the first high prediction
before Flow's correction. The first correction leaves it present; subsequent
refinement ends with the larger final jump. The pre-Flow capture includes the
executing model and other patches, so this localizes the onset to the
pre-high-to-high-prediction transition without identifying a particular model,
conditioning, noise-transport or external-patch operation as its cause. It also
does not reconstruct the counterfactual later sampler trajectory with Flow off.

## Tone in the same decoded frames

These are matched native decoded stage observations at the first retained frame:

| Clean stage | Luma mean | Luma p05 | Luma p95 | Luma deviation |
| --- | ---: | ---: | ---: | ---: |
| Native provider | 0.359700 | 0.065626 | 0.793879 | 0.244470 |
| Provider with authoritative prefix | 0.365869 | 0.060575 | 0.821315 | 0.256172 |
| Pre-high DC | 0.363924 | 0.055521 | 0.823289 | 0.258219 |
| First high, before Flow | 0.370195 | 0.060787 | 0.827796 | 0.256346 |
| First high, after Flow | 0.369207 | 0.059802 | 0.828215 | 0.256912 |
| Final | 0.371492 | 0.059841 | 0.831532 | 0.258274 |

Replacing only the provider prefix increases decoded contrast and lowers its
dark-tail value before high sampling. The DC bridge lowers the dark tail further
in this pre-high counterfactual. High refinement changes that pattern again;
there is no demonstrated single exposure/gain correction. Mean, quantiles and
deviation respond to motion and composition as well as tone. These observations
do not validate removing DC from a full sampled run, or calibrating exposure
against an independently composed prior frame.

## Admission and remaining limits

[The offline replay tool](../../tools/decode_native_boundary_evidence.py) checks
all eight hashes, shapes, finite values, native window phase, actual-provider
provenance, protected pre-high/final bytes and the exact initial mask before
loading decoder weights. It accepts only the six clean comparison stages;
sampler input and masks are inspected but never decoded as clean video. Stage
batching bounds full-frame canvas memory. Reports identify the VAE and source
hashes, precision and device, and explicitly mark rendered acceptance false.

The preceding seven-token decoder window is not supplied. Its temporal blend
cannot be reconstructed, so the first pair against the preceding decoded frame
and all pre-boundary/anchor comparisons use raw unblended prefix frames. They are
not assembled-boundary comparisons. The subsequent retained pairs and post tone
in the table use the native current-window spatial blend. No VAE encode/decode
round trip, guessed padding, latent-motion surrogate or additional H3 evaluation
is substituted for the captured operands.

All measurements and supplied receipt comparisons are retained in
[the machine-readable native replay report](CONTINUATION_01115_NATIVE_DECODE.json).
This evidence narrows the investigation but does not validate a replacement
production actuator. PR #93 remains a rendered-failed draft. The existing
`same_grid_target_control` and companion stack are preserved; prefix release,
spatial transplants, prediction/VAE warps and fixed successor guards remain
retired. No additional generation is requested by this review.

## Verification

- Native full spatial replay completed for all six clean stages with the real
  decoder checkpoint and unmodified Core spatial blending.
- Evidence admission regressions: 14 passed, including corrupted bytes,
  rehashed non-finite operands, changed protected bytes/masks (including signed
  zero), escaped paths,
  altered phase/trim, missing operands and forecast/wrong-provider provenance.
- Full local suite: 882 passed, 27 skipped; native source oracles run separately.
- Focused decode-context, native temporal-source and replay-admission suite with
  the reviewed Core checkout: 40 passed.
- Ruff check/format and compile checks pass.
- Sampling code is unchanged from `71de5dd2`; no rendered repair is claimed.

## Evidence identity

- Metrics SHA256: `4312d16e3b5886bdb54e1f927fc80a35b5e220d4290f07f507111f386e6791e1`.
- Log SHA256: `e46bea4d9acee52ec10dc766bf34dedfe59e8622af31a419edc6b8adca16e853`.
- Manifest SHA256: `c576e836ee055cce6ba4b6dfd752b28c6a1b0da6d4e2103a6efd537330881844`.
- Tensor bundle: `session-continuum_chunk-2_seed-16356530102724618645_sigma-0.87804878_1791094863345029183`.
- Eight finite tensors, 18,467,712 bytes, all SHA256 hashes verified.
- Official VAE file SHA256: `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`.
- Core VAE source SHA256: `8a889ea28fc24b13edfbdea9e85b8a0762b2e7e079bbe34db026c6e6d4f5a556`.
