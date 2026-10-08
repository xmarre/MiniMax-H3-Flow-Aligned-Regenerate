# Run 01734: structural correction and qualification limits

This review uses the supplied `metrics_01734_.json`, log and three numerical
Local Boundary Audit reports. No decoded frames, saved latent operand bundles
or trained GPU execution were available. CPU tests cannot establish that the
reported visible darkening or frame shift has been repaired.

## Measurements and interpretation

| Stage | First chunk | Continuation |
| --- | ---: | ---: |
| Low | 71.081 s | 201.466 s |
| Probe | 10.819 s | 21.771 s |
| High | 75.036 s | 96.686 s |

The continuation low stage recorded 10 actual calls and zero forecasts; high
recorded four actual calls and two forecasts. The existing owning-stage history
receipt correction is retained. Forecast counts and wall time after that
correction have not been measured here.

The two-stream layout contains 22,580 target-stream hidden rows and 36,016
source-stream hidden rows per evaluation, totaling 58,596. A full uniform
source clip has 36,016 rows: 38.5% fewer hidden rows than the two-stream layout
at this geometry. This is arithmetic accounting, not a GPU speed estimate.

The stage-continuity report already shows the prefix/band/tail luminance
separation in the low-derived provider operand. Prefix frames 171–174 are
approximately 0.511–0.513, band frames 175–191 approximately 0.493–0.510,
and tail frames 192–195 approximately 0.595–0.604. The final operand retains
that separation. This rules out a cause exclusive to high refinement. The
five-token band gives the short target stream a native 17-token length, so
changing to that length did not remove the reported symptom. Numerical
luminance/RGB differences alone do not prove an unintended content cut or
identify a unique cause.

## One generated trajectory

`progressive_uniform_source` constructs an equal-grid low/probe stage plan from
the physical-lattice projection of the authoritative prefix and its source-grid
prefix noise. Every generated frame and all conditioning rows belong to one
native-duration source clip. The learned provider transfers the complete
generated trajectory. There is no independent short target-band author, second
conditioning stream or band/tail splice. High refinement restores the original
target-grid stage plan and the caller's exact prefix; returned prefix bytes are
also canonicalized exactly.

The projected prefix changes low/probe attention context. Reduced prefix detail,
learned transfer and the replacement with exact target-grid prefix can still
affect rendered geometry and tone. This mode is an explicit candidate, not a
qualified production default. Existing modes and defaults remain available.

Use normal VDN/softmax diagnostics and native temporal-carrier policy. Reset
inapplicable target-band context/state selectors to their defaults. Captures
save the actual native source operand, provider, pre-high, prediction and final
operands; replay records zero band tokens and does not require a padded
target-band carrier. Capture is output-neutral in the CPU regression.

## Non-evicting audit decode

`load_models_gpu(memory_required=0)` still invokes Core's minimum inference
memory admission and can evict unrelated models. A managed `VAE.decode` retry
after OOM can do the same. The audit now calls neither path. It verifies that
the connected VAE patcher is fully resident on its decode device and invokes
the native decoder directly, preserving production latent and pixel processing.
Missing/partially offloaded weights or decode OOM stop the audit. The audit
never loads the VAE or unloads another model to make space.

## Validation scope

Regression coverage exercises real Core low/probe/high forwards with Sol and
native attention, equal-grid Spectrum history ownership, exact prefix return,
single-stream receipts, output-neutral source capture and native source replay.
Managed-VAE fixtures check resident reuse and prohibit model admission on
success, missing/partial/off-device weights and OOM. Native temporal VAE source
oracles validate replay timing. Paired source contracts and repository lint/
format checks pass. Exact-head CI is recorded in PR #97.

A matched trained-GPU run must still compare visible joins, tone, audio,
actual/forecast counts, stage wall time and peak memory with the accepted
same-grid control. There is no new photometric correction, image crossfade or
automatic quality claim.
