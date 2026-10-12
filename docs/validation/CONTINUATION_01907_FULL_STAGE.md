# Continuation 01907: trained-VAE full-stage audit

`boundary-t5k9rxg8.json` successfully runs the full-context protocol with the
production native video VAE: seven full clips, 62 tokens / 209 decoded frames,
native origin 136, assembled join 175. The source canvas is 704×704; all target
canvases are 992×992. Its boundary manifest hash and every operand hash match
the original 01907 metrics. All 28 supplied PNGs have the declared dimensions;
their global luminance agrees with the floating-point report to within
`1.314e-5`, consistent with the recorded eight-bit quantization.

[Selected measurements and all PNG SHA256 hashes](CONTINUATION_01907_FULL_STAGE_SUMMARY.json)
provide a compact evidence receipt. The audit reruns no H3 sampling and changes
no production output. It does not contain target-high B or an audio comparison.

## Geometry develops before transfer

Independent LK/RANSAC background fits relative to each stage's frame 174:

| Stage | Scale at f175 | Scale at f180 | Scale at f190 |
| --- | ---: | ---: | ---: |
| Native source | +0.03955% | +0.14154% | +0.36798% |
| Native learned provider | +0.02431% | +0.10151% | +0.28730% |
| Provider with exact prefix restored | +0.07227% | +0.38158% | +0.52662% |
| Corrected pre-high | +0.04166% | +0.12592% | +0.35591% |
| First high before Flow | +0.04402% | +0.17738% | +0.39624% |
| First high after Flow | +0.04317% | +0.17558% | +0.45876% |
| Final | +0.03452% | +0.19904% | +0.41346% |

The source contains nonzero expansion before learned transfer or target-high.
Final f180 has 498/499 fitted inliers, median residual 0.4421 px; f190 has
396/459, median residual 0.5459 px. The provider-with-restored-prefix fits have
weaker support (f180 199/382, residual 1.0219 px). Their large apparent scale
is not a calibrated splice-induced warp. Each stage selects different features;
these scale estimates must not be subtracted as causal geometric transforms.
The first Flow application changes same-frame ROI structure very little
(Sobel structure cosine above 0.998 at f180), despite different later independent
tracking estimates. No rigid geometric correction is justified by those fit
differences alone.

## Transfer and prefix restoration change the detail baseline

Absolute source→native-provider comparisons use an area-reduced common 704px
canvas, with no source upsampling:

| ROI | Provider/source edge energy f174 | f180 | Provider−source luminance f174 |
| --- | ---: | ---: | ---: |
| Bookshelf | 0.8190 | 0.8262 | −0.004648 |
| Framed picture | 0.8249 | 0.8022 | −0.002752 |
| Right curtain | 0.7261 | 0.7142 | −0.006286 |

The learned transfer already loses some absolute detail and starts darker. This
measurement includes the different latent/VAE representations; no isolated
checkpoint or interpolation defect has been established.

Replacing only the provider prefix with the authoritative prefix increases
f174 edge energy by 1.554× / 1.485× / 2.152× for bookshelf/picture/curtain.
The same ratios at f180 are 1.023× / 1.001× / 1.093×. This is a controlled
prefix-only decoder intervention with an identical generated latent suffix.
It changes the pre-join baseline much more than the later suffix's absolute
edge energy. Consequently the within-stage f180/f174 ratios fall from
0.906/0.958/1.051 in the native provider to 0.599/0.669/0.568 after restoration.
Those lower ratios do not demonstrate that restoration deleted 33–43% of the
suffix's absolute detail. The discarded-prefix reference became sharper.

Opt-in detail transport increases f180 edge energy relative to this restored
provider by 1.136× / 1.131× / 1.273×. The first high prediction adds another
1.357× / 1.362× / 1.614× relative to pre-high; the first Flow application reduces
it by factors 0.939/0.976/0.942. Later refinement restores more energy. Multiplying
the common-grid energy ratios along the paired stages gives final/source f180
ratios 1.380/1.283/1.632. This demonstrates recovered edge energy, not exact
texture recovery or motion-safe transport. Fitted/model texture and shifted
content can increase this metric; rendered detail remains a visual judgment.

## Tone changes have multiple stages and signs

Native-stage luminance change f174→f180, normalized RGB:

| Stage | Bookshelf | Framed picture | Right curtain |
| --- | ---: | ---: | ---: |
| Source | −0.003250 | −0.006481 | −0.004017 |
| Native provider | −0.002577 | −0.004287 | −0.001474 |
| Provider with restored prefix | −0.006266 | −0.005097 | −0.002498 |
| Corrected pre-high | −0.005705 | −0.004996 | −0.002618 |
| First high before Flow | −0.004151 | −0.007561 | −0.002978 |
| First high after Flow | −0.003948 | −0.006541 | −0.002630 |
| Final | −0.000378 | −0.005068 | −0.002408 |

The first high prediction further darkens the picture: co-located pre-high→first
high bias is −0.002771 at f180 and −0.005009 at f190. The first Flow application
brightens that region by +0.000935/+0.001284; later refinement brightens it again
by +0.002509/+0.004528. The source itself already darkens. A single global tone
offset cannot be inferred from this pattern. Final f174→f190 changes are
−0.000372 / −0.007352 / −0.003322 for bookshelf/picture/curtain.

The white wall gains texture and cannot serve as a static registration witness.
All f174 replay measurements describe discarded context within the new chunk,
not the previous chunk's assembled pixels. Comparing the PNGs to the encoded
01907 video finds roughly −0.0043 to −0.0049 global luminance bias across both
prefix and suffix times; encoding and decoder/output differences intervene.
That comparison cannot attribute the bias solely to a boundary decoder effect.

## Code-path disposition

The selected path copies the hash-paired previous source tail into protected low
context, generates one uniform source trajectory, performs one learned transfer,
applies the requested clean-space detail correction, restores the authoritative
target prefix, and then samples high. The source/probe capture occurs before the
provider and correction. The native-provider versus restored-provider audit arm
changes only the prefix on CPU copies; the saved generated suffix is identical.
Earlier independent reconstruction also verified the pre-high clean correction
within float32 roundoff. No mask, timeline, stale carry, double correction or
current native H3 latent-scale error was established by this evidence.

Retain native source carry as the tested working candidate. This audit establishes
that source-stage expansion and darkening remain, while transfer changes detail
and prefix restoration changes decoder context. Their ultimate model-level cause
remains unresolved: source conditioning/trajectory effects and approximate
sampling/attention effects are not distinguished by this fixed-setting capture.
Do not promote a geometric warp or stronger global tone correction based on
these stage fits. Final audiovisual acceptance and a target-high causal quality
comparison remain separate from this successful stage-decode qualification.


## Release cleanup interpretation — 2026-10-12

The user reports good boundaries across many prompts and reference images, with slight darkening remaining. The fitted motion and sharpness observations above do not establish an unwanted zoom or bokeh defect: camera and focus can follow conditioning and subject motion. No Spectrum cause is established. Retain native source carry and the measured evidence, retire live replay/kernel probes, and carry the darkening as a known issue rather than a claimed tone fix.
