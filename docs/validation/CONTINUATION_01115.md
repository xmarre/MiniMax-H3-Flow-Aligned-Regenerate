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

## Decoder replay checkpoint

The offline replay uses the unmodified MiniMax-H3 VAE implementation from Core
`6b4e05dc30d65740ce8931434607b9907996fb0e` and the official
`Comfy-Org/MiniMax-H3` fp16 video VAE weights. Model input/output video conversion
is identity in Core's MiniMax-H3 video latent format; VAE channel normalization is
owned by the native decoder.

Initial paired runs use native 16x16 spatial tiles at latent origins `(0,33)` and
`(0,11)`, all seven temporal tokens, and FP32 decoder arithmetic. Protected
prediction tokens are restored from the supplied authoritative bytes before
decode. They compare provider native, provider with exact prefix, pre-high DC,
first actual high predictions, and final output without rerunning H3 or the
learned provider.

The `(0,33)` tile has a pronounced final third-frame motion change absent in the
provider-native tile. Exact-prefix restoration also changes decoded tone before
high sampling. The other tile has different motion and ambiguous large later
estimates. These are local counterfactuals: they do not reproduce the full-frame
spatial blend or the preceding temporal blend. Full-frame replay must decide the
global localization before another production actuator is selected.

No new production sampler mutation is selected at this checkpoint. In particular,
tile evidence cannot justify another spatial-residual transplant, prefix release,
or prediction/VAE warp.

## Evidence identity

- Metrics SHA256: `4312d16e3b5886bdb54e1f927fc80a35b5e220d4290f07f507111f386e6791e1`.
- Log SHA256: `e46bea4d9acee52ec10dc766bf34dedfe59e8622af31a419edc6b8adca16e853`.
- Manifest SHA256: `c576e836ee055cce6ba4b6dfd752b28c6a1b0da6d4e2103a6efd537330881844`.
- Tensor bundle: `session-continuum_chunk-2_seed-16356530102724618645_sigma-0.87804878_1791094863345029183`.
- Eight finite tensors, 18,467,712 bytes, all SHA256 hashes verified.
- Official VAE file SHA256: `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`.
- Core VAE source SHA256: `8a889ea28fc24b13edfbdea9e85b8a0762b2e7e079bbe34db026c6e6d4f5a556`.
