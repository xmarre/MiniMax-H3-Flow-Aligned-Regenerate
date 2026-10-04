# Continuation throughput qualification: 00878

The native arithmetic-reference correction is active in this capture, but the
continuation throughput goal remains unmet. The prompt completes in 420.62 s.
The preceding two-chunk capture completed in 407.12 s; these are separate
realizations, so their difference is not an isolated patch regression measurement.

| Sampling stage | Initial chunk, s | Continuation, s | Increase, s |
|---|---:|---:|---:|
| Low | 58.026 | 133.041 | 75.016 |
| Exact probe | 9.706 | 28.251 | 18.545 |
| Transfer | 0.558 | 1.443 | 0.885 |
| High | 69.337 | 81.364 | 12.027 |
| Whole sampler | 137.646 | 246.004 | 108.358 |

Low/probe accounts for 93.561 s (86.3%) of the 108.358 s continuation sampler
increase. Time outside both sampler invocations is 36.970 s. Each chunk has nine
actual model evaluations and two forecasts; the continuation does not introduce
additional evaluations relative to the initial chunk.

The geometry is the material workload difference. Initial low has 52 frames on
the 38x50 latent grid, or 24,700 video patch rows. Continuation low has 62 frames
on the 54x72 grid, or 60,264 video patch rows (2.440x), including the twelve
protected prefix frames. `same_grid_target_control` explicitly overrides the
configured 38x50 progressive source grid. It is not a silent admission fallback.
The control also replaces the spatial transfer with an identity provider.

Execution receipts confirm the existing optimizations:

- Native gates report fifteen Core references and zero raw-Torch references
  across initial low/high and continuation high.
- Continuation low/probe report 1,811/800 Core dense calls and zero raw-Torch
  partitioned dense calls.
- All six admissions report zero evictions and preserve a resident H3 clone.
- Six modulation validations cover the 300 partitioned blocks. The uniform
  pre-RoPE linear path executes; the existing factorization checks remain bounded.

Dispatcher counters establish call ownership, not the selected GPU kernel.
Gate host time includes queued computation and compilation as well as references
and reductions. It cannot be treated as an entirely removable cost. This capture
does not establish a VRAM leak or repeated-run stability.

Source tracing confirms the full-grid policy in
`h3_flow_regenerate/partitioned_scheduler.py` and the corresponding explicit
partitioned geometry in `partitioned_stage.py` and `partitioned_prefix.py`.
Removing the control would change the executed numerical path. The earlier
rendered progressive failure remains a rejection of that replacement; a reduced
grid is not qualified merely by its lower token count or passing CPU tests.

The remaining boundary cause is not established by this capture: it executes no
cross-grid transition and has residual measurement disabled. The existing
`frame_gauge_residual_mode=measure` discriminator on the failing progressive path
can locate the current first native high prediction, paired Flow correction and
subsequent evolution without adding model/provider/VAE evaluations. Measurement
timing must not be used as production throughput timing. A corrected progressive
candidate requires a rendered boundary verdict before replacing the accepted
same-grid control. No new runtime arithmetic is promoted by this qualification.

Evidence: metrics SHA256
`3b34235a1922617ae7fb3c24ad0acbeadf1c29fd8ebafb336f4d8a7d97eb5465`;
runtime log SHA256
`bfbd55e1f859861e8db513ac3248a11af539a0cc90715fea0e585469bbdf4c50`.
