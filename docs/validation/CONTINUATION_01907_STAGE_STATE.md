# Continuation 01907: saved state and rendered boundary evidence

The supplied continuation capture pairs native source carry with opt-in
`prefix_calibrated_static_detail_transport_v1` before target-high refinement.
This is a combined production-A result. It cannot isolate the visible effect
of source carry, detail transport or the numerical attention correction.

## Integrity and clean correction

Both archive manifests and all 35 tensor files match their recorded SHA256
values. Boundary manifest SHA256 is
`1ca54dc4e9b0656100e15dc470a2d13c906d9a5da6c9d075bda758e04dac374f`;
frozen A/B/A manifest SHA256 is
`661b34f6703cccc7b151e228e8fa879ac0d0d41b00f3692b6f68d3ecda9508d9`.
The boundary source-clean hash equals the frozen production-A source hash
`bcc4783c9e26865b7e45909b2d0da31cf4c241b3126f75228cbc167f18b13ced`.

Recomputing detail transport on CPU from the saved provider output and exact
prefix reproduces the saved pre-high generated suffix with RMS error
`5.0140e-9` and maximum error `2.3842e-7`. The protected prefix is exact.
The measured provider→pre-high generated-suffix change is RMS `0.02875123`,
matching the runtime correction receipt. The first suffix mean weight is
`0.252221`, decreasing to `0.00434863` at the final token. Prefix holdout error
ratio is `0.830055`. This verifies the saved clean correction within float32
roundoff. It does not qualify suffix motion safety, rendered quality, or every
intermediate noised sampler state. No extra unintended clean correction is
needed to explain the saved pre-high bytes.

## Encoded production-A video

MP4 SHA256:
`97e2385d0bb464c8232caa0385eb94df50a315dcb2f8ed3407feb4d50759ce2e`.
The table measures the encoded 992×992 video at 24 fps, assembled join 175,
with frame 174 as the reference. Values use normalized RGB luminance.

| Background ROI | Luminance change f174→175 | Luminance change f174→180 | Native Sobel energy f180/f174 |
| --- | ---: | ---: | ---: |
| Bookshelf | +0.000958 | +0.000759 | 0.9266 |
| Framed picture | −0.000783 | −0.005057 | 0.9823 |
| Right curtain | +0.000285 | −0.001905 | 1.0239 |

Static-landmark LK/RANSAC fits using the bookshelf, picture and curtain give:

| Frame | Apparent scale change | Fitted center displacement (px) | Inliers | Median inlier residual (px) |
| --- | ---: | --- | ---: | ---: |
| 175 | +0.04723% | (+0.0549, −0.0107) | 525 | 0.1482 |
| 180 | +0.20419% | (+0.0791, +0.1097) | 486 | 0.4444 |
| 190 | +0.39661% | (+0.0729, +0.3055) | 438 | 0.7341 |

The fits support gradual background expansion, rather than a large immediate
translation at the join. The picture and curtain develop modest darkening;
there is no uniform luminance step shared by all three ROIs. These are encoded
image measurements, not calibrated camera motion or final continuity acceptance.
The white wall is excluded because changing texture invalidates static tracking.
A scene change later in the clip also limits this tracking interval.

The qualified native low A/B report rejects switching to half-pixel B because
of its absolute blur and darker baseline. Native A already has nonzero source
trajectory change. Differences between native-A and encoded-final fits cannot
be subtracted to attribute a causal high-stage transform: different grids,
landmark selections, prefix rendering and encoding intervene.

## Remaining qualification

Replay the existing boundary bundle with `full_stage_continuity`, static ROIs
and saved previews as described in [the protocol](../FULL_STAGE_CONTINUITY.md).
This measures the source→provider→corrected handoff→first high→final transitions
through complete native decoder contexts. It requires six VAE calls and zero
H3 evaluations. No repeated low A/B or CUDA token-admission run is needed.
No trained VAE or CUDA device was available for this stage decode in the
review environment; its final pixel result remains unmeasured here. Flow #99
remains experimental until decoded stage evidence and audiovisual acceptance
support the continuation behavior.
