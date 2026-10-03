# Continuation execution, source validation and boundary qualification: 01023

The continuation sampler takes 99.215 s longer than the initial sampler.
Low generation and the exact probe contribute 74.165 s, or 74.8%, of that
increase. The executed partitioned attention path also repeats a complete
packaged-source validation on every kernel invocation. Request-owned source
validation removes that redundant CPU work; its end-to-end GPU benefit has
not been measured. The decoded contrast discontinuity remains unresolved.

## Workload and sampling time

The source/target latent grids are 44x44/62x62. The initial chunk contains 52
video latent tokens. Continuation retains twelve target-grid prefix latent tokens and
generates fifty source-grid suffix latent tokens. Initial low/high video patch rows
are 25,168/49,972; continuation low/high rows are 35,732/59,582. Low rows
increase 42.0% and high rows 19.2% within the capture. Initial conditioning
contains 1,304 text, 961 reference and 584 audio rows, for 28,017 packed rows.
These grids and conditioning differ from both previous captures, so their
absolute durations do not form a matched before/after speed comparison.

| Sampling stage | Initial chunk, s | Continuation, s | Difference, s |
|---|---:|---:|---:|
| Low | 64.231 | 129.425 | +65.195 |
| Exact probe | 10.172 | 19.142 | +8.970 |
| Transfer | 0.544 | 4.033 | +3.489 |
| High | 86.279 | 106.120 | +19.841 |
| Whole sampler | 161.247 | 260.462 | +99.215 |

The total prompt receipt is 457.06 s. Approximately 35.351 s lies outside the
two sampler scopes; those scopes do not separately identify every conditioning,
VAE, assembly and other node cost. Stage and nested arithmetic-gate durations
are already included in their sampler times.

The capture retains 22 actual H3 evaluations and twelve forecasts: per chunk,
six low actual calls plus four forecasts, one exact probe, and four high actual
calls plus two forecasts. There are six sampler scopes and four progressive
history boundaries. Increased evaluation count does not explain the penalty.
The initial-low Spectrum model-profile lookup is already a cache hit (0.014 s).

All six admissions preserve the required resident H3 clone. The continuation
high admission takes 0.111 s; the adjacent Core receipt reports a 288.02 MiB
partial unload. Other admissions are at most 0.001 ms. The 15.266 s admission
seen in 01020 therefore cannot account for this capture's continuation penalty.
The partial-unload receipt alone does not identify the affected model.

## Repeated source validation

Continuation low executes 2,050 weighted all-selected production calls,
2,750 partitioned sparse calls and eleven sparse arithmetic-reference calls
through `_sm120_union`. The probe executes another 800 weighted production
calls. The old helper invokes `verify_source()` on all 5,611 entries. Each
invocation validates the full 51-file, 616,846-byte packaged Sana source tree.
These counts are derived from executed dispatcher/gate receipts and the
corresponding source path; the old capture has no per-scan timing field.

Sol #37 moves successful source trust into the existing native `OUTER_SAMPLE`
request. For this execution, that reduces full scans to one in continuation
low and one in its independent probe. Nested and subsequent requests validate
separately; changed source/verifier identity invalidates trust; failure clears
trust and propagates before the kernel. Concurrent first entries share one
check. Standalone calls remain uncached. The request retains no tensors or CUDA
resources. Installed packaged source must remain fixed within a request and
source updates require restarting the application.

The new sampling summary fields are `partitioned_source_tree_verified`,
`partitioned_source_verification_calls` and
`partitioned_source_verification_wall_s`. Low and probe should each report one
successful full check for this workload. Other scopes that do not enter the
partitioned union report zero. Device/layout arithmetic gates, vendor/kernel
bytes, attention math, native bias rounding and forecasting identities remain
unchanged. See [source ownership](https://github.com/xmarre/ComfyUI-Sol-H3/blob/d37e34499117e980cff471e31361bced3abe147e/docs/REQUEST_SOURCE_VERIFICATION.md).

A local CPU measurement of the existing verifier takes 0.153099 s for fifty
warmed scans (3.062 ms each). Repeating that local average 5,611 times gives
17.181 s; this is a CPU-local extrapolation, not a prediction for the capture's
machine. The patched helper takes 0.008758 s for 5,611 entries in one request,
including one 0.003577 s full scan. These measurements establish removal of
repeated scanning, not total prompt speedup or GPU kernel latency.

The existing weighted route and independent arithmetic gates execute in the
capture. Low/probe retain 21/10 Core reference calls and ten weighted gate
identities each. Weighted-gate wall totals are 13.445/0.203 s; sparse low gates
add 5.630 s. Cold compilation, preparation and queued work are included, so
these are not isolated GPU timings and cannot all be counted as removable.

## Boundary and decoded contrast

Registration rejects `unsupported_sampler_contract` for `sample_euler` after
video-boundary acceptance. The fallback applies the coupled exact-overlap
bridge over four suffix tokens with weights `(1, 0.75, 0.5, 0.25)`. All six
continuation guidance calls, including forecasts, use
`exact_prefix_guidance_reference_coupled_v1`. Completion reports four-token
support, `spatial_mean_only=false`, unchanged source/temporal reference and
zero extra H3/provider/VAE calls. The one-token DC-only guidance-reference
policy is not exercised in this realization.

Video overlap closes over the final two exact high calls with a final mask
maximum of zero. All 22 sampled duplicate-overlap decoded frames are equal,
including the five-frame right-context tail. Video Seam Auto replaces zero
frames. This establishes retained-overlap agreement; the first generated frame
still requires independent rendered acceptance.

| First retained transition, whole-frame luma | Previous frame | Generated frame | Relative change |
|---|---:|---:|---:|
| Mean | 0.546357 | 0.549621 | +0.597% |
| Standard deviation | 0.227274 | 0.237668 | +4.573% |
| Fifth percentile | 0.189292 | 0.162188 | -14.319% |
| Ninety-fifth percentile | 0.926173 | 0.928418 | +0.242% |

The increased deviation and darker lower tail support the reported contrast
discontinuity. Motion, shading and composition also affect global statistics;
they do not locate its cause. No local video or boundary crops accompany 01023.
The native-to-restored latent boundary preserves gradient RMS 0.085748 and
centered low-pass RMS 0.281348. After high refinement, they fall to
0.082921/0.272065, both about 0.967 times their pre-high values. This witness
does not demonstrate high-stage latent gradient amplification and does not
identify the decoded photometric or VAE mechanism. Source-validation caching
does not alter numerical output or provide a tone correction.

Immediate affine estimates differ by region: full-frame scale
`(1.009005, 1.006992)` versus upper-region `(1.000058, 1.000043)`.
Confidence is low (full X/Y 0.101/0.050; upper X/Y 0.0022/0.0006).
These observations cannot certify absent zoom or establish a regression across
different scenes and workloads.

## Qualification and evidence identity

A matched performance comparison must hold geometry, conditioning, seed,
carried prefix and schedule constant, and distinguish cold from warm execution.
The new request source receipts isolate the scan count and CPU check duration.
GPU speed acceptance and local rendered contrast acceptance remain open. The
policy signatures identify executed paths, not an exact installed Git SHA.

- `metrics_01023_.json`: SHA256
  `e33dbf28ad9db653ae1a677d6d1c1096d581d960d30349fd92411e98f512ff30`.
- `Pasted text(20261003-194541).txt`: SHA256
  `217c48ee4b27074900f116e64f5a4fa8cb357200a6f6ad94623352d3f9066771`.
- Previous captures: [01020 qualification](CONTINUATION_01020.md),
  [01000 throughput](CONTINUATION_THROUGHPUT_01000.md) and
  [local crop/DC evidence](BOUNDARY_TONE_01000.md).
