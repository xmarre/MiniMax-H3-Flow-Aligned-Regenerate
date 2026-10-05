# Local boundary tone and DC-reference consistency

## Decoded crop evidence

Two supplied 289x239 JPEG crops show a knitted garment and skin at a video
boundary. The later crop has darker knit valleys and a darker skin shadow.
The following measurements use the weighted RGB luma proxy
`0.2126 R + 0.7152 G + 0.0722 B`, with JPEG channel values divided by 255.

| Crop timestamp | Mean | Population deviation | 5th percentile | 95th percentile |
|---|---:|---:|---:|---:|
| 00.07.257 | 0.570532 | 0.112645 | 0.349271 | 0.685198 |
| 00.07.289 | 0.537786 | 0.137369 | 0.257778 | 0.678456 |

Mean falls 5.740%, population deviation rises 21.949%, and the 5th percentile
falls 26.195%. The crop contains a local tone/contrast discontinuity. Motion,
crop alignment, JPEG encoding and subject shading also affect these statistics;
they are not calibrated contrast gain or an instruction to apply image gain.
The same-pixel affine fit has a residual RMS of 0.05725, so it does not establish
a spatially uniform photometric transformation.

The 01000 assembly receipt samples the full frame at 126x192. Its first retained
luma deviation rises only 0.547% from the preceding frame. This global average
does not qualify local contrast in the crop. All 22 sampled duplicate-overlap
frames are equal, and Video Seam Auto replaces zero frames. Those checks cannot
establish tone continuity in newly generated frames.

## Active execution and violated invariant

01000 executes the original one-token DC handoff. Its measured per-channel
spatial-mean offset has RMS 0.269202, mean absolute value 0.180826 and maximum
absolute value 0.733313 latent units. The actual clean postprocess tensor is
used; there is no deterministic-noise inverse recovery on this path.

The structural registration rejects for regional disagreement. The coupled
exact-overlap bridge consequently does not execute, and all twelve high-stage
guidance receipts report `reference_gauge_used=false`. The prior reference
rebase only covered an applied coupled bridge. It omitted the independently
active one-token DC handoff.

For a native last-prefix frame `L`, first suffix `S` and authoritative prefix
`E`, the DC handoff adds `d = mean(E) - mean(L)` to `S`. Direction guidance then
compares the corrected suffix with an uncorrected reference. In a fixture whose
high prediction already equals the DC-corrected reference, direction weight
0.35 undoes a 0.2-unit DC offset by 0.070 units. The prefix remains exactly equal,
so prefix ownership checks do not detect the error. Forcing the full structural
rebase in that fixture introduces a spatially varying error up to 0.035 units.

The invariant is component consistency: a comparison must use the same accepted
representation component and support as its handoff. An accepted DC correction
does not authorize a rejected spatial correction.

## Correction and limits

An applied, nonzero one-token DC handoff now compares an unregistered direction
reference using `mean(E) - mean(reference_last_prefix)`. That scalar per
batch/channel is added only to the first generated reference comparison.
Acceleration reference velocity uses the same component. Coupled handoffs keep
their existing full residual and four-token support. Source trajectory tensors,
source temporal correspondence, exact prefix/audio ownership and sampler
overlap closure keep their existing contracts.

The DC owner retains only per-channel FP32 means plus geometry metadata. Recursive
option copies share that owner. Success, sampler failure and receipt failure
release its activation. The runtime policy is
`exact_prefix_guidance_reference_dc_v1`; the completion receipt reports
`spatial_mean_only=true` and one support token. It adds no model/provider/VAE
evaluation, sampler lifetime or per-reference host scalar read.

Tests cover FP32/FP64, filtered and unfiltered direction, actual/forecast
production wrapper calls, acceleration velocity, activation eligibility,
unchanged spatial detail, source immutability, bounded ownership, nested scopes
and failure cleanup. These establish comparison consistency. They do not
establish that the 01000 crop change is caused by this defect or that the
updated render has acceptable contrast. Learned transfer, high-stage generation
and decoder temporal behavior remain possible contributors to local detail.

Hardware qualification should preserve the same input, geometry, seed, prompt,
video overlap six, strengths 0.5/0.5 and sixteen steps. Confirm the DC policy in
executed guidance receipts and compare the same boundary region across retained
frames, including subsequent frames beyond the first token's decode footprint.
Evaluate timing with the paired weighted dense candidate separately from
rendered tone acceptance.

## Evidence identity

- `MiniMax_H3_00082-audio.mp4_snapshot_00.07.257_cr.jpg`: SHA256
  `328ccf4e974747def200d1ea1618e89570ca24775d88cfbbfa34fb53e8edac6b`.
- `MiniMax_H3_00082-audio.mp4_snapshot_00.07.289_cr.jpg`: SHA256
  `c5ee2e1fa04beb244a00b1a8d55de22e537796ab9522d07e6aec9f798249e07a`.
- `metrics_01000_.json`: SHA256
  `b27cab732a625b04332fa400d086effdb798a402de9f0ea564636038aada5865`.
- `Pasted text(20261003-165928).txt`: SHA256
  `f42e236309b68c9e3eff8afc2fd34e4868536b525538cfcdf457ec9e24486c62`.
