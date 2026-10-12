# Uniform continuation: image-grid contract correction

The continuation used a different spatial representation from the initial
progressive chunk even after its generated video became a single native clip.
This is a concrete, independently reproduced contract mismatch. It is not
proof that it alone accounts for every rendered tone or motion defect.

## Source audit

The initial Target Input path in `runtime.py` resizes its latent image with
`_resize_packed_latent_image`: bicubic, `align_corners=False`, antialiased when
shrinking. Its learned handoff invokes `provider.upscale_clean_video`.

The companion provider's ordinary method defaults to `half_pixel_latent_v1`.
In `LatentUpscaler3D.forward`, the encoder-to-decoder transport then uses
`F.interpolate(..., mode="trilinear", align_corners=False)` with unchanged T.
Source inspected: Upscaler-Plus `nodes/minimax_h3_handoff_provider.py` and
`nodes/minimax_h3_latent_upscaler_3d.py` at
`1ddcc33e9b674aaabbccc3808fe36e1a47bdd36d`. This establishes actual interpolation
routing; it is not a claim about an unseen training dataset.

Before this correction, `partitioned_scheduler.py` always replaced that normally
resized protected prefix with `resize_spatial_5d_h3_patch_lattice`. It also
wrapped the learned provider in `H3PatchLatticeTransferProvider`, forcing
`h3_dense_patch_center_lattice_v2` between encoder and decoder. This happened in
both uniform-source modes, despite their ordinary native single-grid clip.
The README's assertion that this used the same learned transfer as chunk 1 was
incorrect.

Those operators align attention patch coordinates across explicitly mixed
grids. Attention coordinates do not establish the pixel-center convention of
the VAE image or replace the ordinary learned provider's interpolation contract.
Applying them to a complete native uniform image is a separate numerical
intervention, not a necessary part of equal-grid attention transport.

## Independent coordinate reproduction

For equal aspect ratios, let r be source length / target length and j a target
dense-cell index. The ordinary half-pixel map samples source index
`(j + 0.5)*r - 0.5`; the attention-coordinate extension samples
`(j - 0.5)*r + 0.5`. Their difference is `1 - r`.

CPU coordinate ramps through the actual physical operator and an independent
`F.interpolate(..., mode="bilinear", align_corners=False)` reference give:

| Source HxW | Target HxW | Mean interior difference, source cells (y,x) |
| --- | --- | --- |
| 48x36 | 72x54 | (0.333332, 0.333332) |
| 36x48 | 54x72 | (0.333332, 0.333332) |
| 50x38 | 72x54 | (0.300943, 0.300941) |

The last geometry is the one recorded in 01768/01769. The difference is a
sampling-convention displacement, not a measured rendered translation. The
aspect-ratio rounding also makes it spatially varying. Reciprocal resampling
can cancel a coordinate-ramp displacement while the individual operator remains
different; round-trip tests alone do not establish the appropriate image grid.

Claude independently confirmed the call routing and ramp calculation. His
counter-evidence is retained: the existing 01764/01767/01768/01769 same-frame
learned-prefix affine fits have center displacement no larger than roughly
0.09 target latent cells, far below the interpolation-convention difference.
Source-grid adjacent motion also lacks a corresponding large join outlier.
Thus a direct uncompensated global coordinate offset is not established as the
rendered cause. The remaining testable mechanisms are altered source-prefix
conditioning/aliasing and the learned network's response to a different feature
interpolator, especially when high reconciles that output with the exact prefix.

An independent aliasing reproduction at 72x54 -> 50x38 uses a +/-1 one-cell
checkerboard. Away from the border, the ordinary antialiased resize leaves RMS
0.017799; the physical bilinear projection leaves RMS 0.328483. This demonstrates
folded high-frequency energy entering the old source condition, not a measured
darkening mechanism in the trained model. Both the coordinate convention and
downsampling filter change when the uniform prefix is corrected.

## Implementation

Both `progressive_uniform_source` and
`progressive_uniform_source_exact_context` now retain the ordinary antialiased
prefix already produced by `_resize_packed_latent_image` and use the ordinary
provider method. Their exact target prefix remains authoritative at high and
return. Heterogeneous modes retain their explicit physical-lattice operator.

The prefix and transfer receipts report `half_pixel_latent_v1` for uniform
continuation. No new widget, noise policy, sigma schedule, brightness adjustment,
output warp or extra model evaluation is added. Existing native low/probe visual
reference conditioning remains selected by the existing exact-context mode.

Capture also stores `source_prefix_projection_policy`. Local replay validates
the source head with the saved operator. Missing policy in older bundles means
the historical physical projection; unknown or incoherent policies are rejected.
The byte/hash/mask checks and audit model-lifetime ownership stay intact.

## Verification and practical limit

Before the production change, the native Core/Sol scheduler regression failed:
the actual clean prefix consumed by the provider differed from the initial
chunk's resized image. The provider fixture previously aliased its ordinary and
physical methods, hiding which contract executed. It now records them separately.

The regression covers both uniform modes, portrait and landscape non-integer
scales and rounded aspect ratios. It requires the actual provider input prefix
to equal the ordinary initial-chunk resize, exactly one ordinary provider call,
and low/probe/high ownership. A separate test requires heterogeneous continuation
to retain its physical provider. Existing native sampler and protected-prefix
tests also run against the change.

Completed local validation: standalone suite 1,163 passed / 203 optional source
skips; native Core/Sol/VDN scheduler, capture/replay, prefix-context, modulation
and high-attention group 97 passed; explicit physical-transfer companion source
checks 6 passed; native Core audit/model-lifetime checks 99 passed. Lint,
172-file formatting, diff and wheel/sdist build pass.
Native Core is `c75777af`, Sol `ef41676f`, VDN `1113ea65`. These are CPU,
random-weight/fixture checks, not trained rendering evidence.

Rendered acceptance requires the user's trained H3/upscaler/VAE and failing
scene. This workspace has CPU fixtures and no trained checkpoints or GPU, so
contract tests cannot establish removal of darkening, lettering changes or
boundary movement. Keep detail transport off for that comparison. The same
PR #99 overlay supplies this correction to both existing uniform modes.
For matched rendered comparison, preserve chunk 1 and inspect source-grid join,
learned-prefix affine fit, transfer seam ratio and the first native high
prediction before Flow, then the final motion and tone across the entire join
window. Improvement only in a latent fit is insufficient for acceptance.
