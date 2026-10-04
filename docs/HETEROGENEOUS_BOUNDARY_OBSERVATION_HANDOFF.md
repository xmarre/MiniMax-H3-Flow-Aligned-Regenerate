# Heterogeneous boundary observation: qualification handoff

Current status: the user reports the frame shift resolved after restoring
`same_grid_target_control`. Retain that accepted configuration. The active
qualification concerns continuation speed, VRAM pressure and reported overcooking; older rendered
observations below remain specific to their earlier configurations.

## 2026-10-02: new captures confirm the full-grid continuation cost

The new two-chunk and single-chunk captures run the published dense-dispatch and
factorization fixes. The two-chunk continuation reports 1,811 Core dense calls
in low and 800 in the probe, with zero raw-Torch partitioned dense calls. Every
actual evaluation reports 100 deferred factorizations and 50 status reads,
consistent with block-boundary checking for streamed branch weights. H3 admission keeps the resident model,
with no evictions. Both workloads still execute nine actual evaluations per
chunk: five low, one probe and three high.

| Interval (seconds) | Single chunk, 00867 | First of two, 00866 | Continuation, 00866 |
| --- | ---: | ---: | ---: |
| Low sampler | 56.756 | 53.294 | 127.277 |
| Endpoint probe | 9.754 | 9.541 | 26.836 |
| Transfer | 0.560 | 0.571 | 1.389 |
| High sampler | 69.317 | 71.180 | 78.266 |
| Complete sampler | 136.413 | 134.607 | 235.806 |

The prompt totals are 154.96 s and 407.12 s, respectively. Outside-sampler work
accounts for 18.547 s and 36.707 s; the capture does not attribute those totals
to individual decoder or other nodes. The two-chunk total exceeds twice the
single-chunk total by 97.20 s. These are not identical inputs: the single run
has a different reference image, 1,411 text rows and a 72x54 target; the first
of two has 1,243 text rows and a 70x54 target. The within-two-chunk comparison
isolates the continuation policy more closely.

Continuation sampling costs 101.199 s more than the first chunk. Low/probe
alone accounts for 91.278 s, or 90.2% of that increase. The configured 48x38
source grid actually runs at 70x54 under `same_grid_target_control`, with
62 latent frames including 12 protected prefix frames. Its low/probe processes
58,590 video rows versus 23,712 initially (2.471x). Initial high processes
49,140 video rows, so the continuation high's increase is much smaller.
The label "low" describes its sigma interval; it does not mean a low spatial
resolution under this control. The larger workload remains the main reason
the second seven seconds cannot currently match the first's sampler cost.

### Remaining native arithmetic-reference dispatch corrected

Native Sol's all-selected arithmetic references still bypassed Core's SDPA
dispatcher. This candidate corrects that remaining path and batches the same
finite/error reductions into one five-scalar host transfer, replacing seven
separate reads on successful gates. Each new sampling request still verifies
its layouts; checks, thresholds and sparse routing remain active. Real Core
execution errors propagate without retry; standalone raw-Torch and explicitly
supplied dense providers retain their contracts.

The recorded native gate host intervals sum to 31.254 s in 00866 and 20.070 s
in 00867. They include the all-selected kernel, dense reference, reductions,
possible compilation and pending device work; they are not a measurement of
removable overhead or the dispatch change's benefit. Partitioned gate host
timing was absent in those captures and is now recorded too. Teardown publishes
`native_core_reference_calls`, `native_torch_reference_calls` and the aggregate
`arithmetic_gate_wall_s`. These counters identify calls, not selected GPU kernels.

The source-gated mapped-neighbor probe records the changed bridge blob. Earlier
GPU captures continue to qualify their original source, not this candidate's
speed. CPU regressions establish dispatcher ownership, layout/bias preservation,
unchanged statistics and failure decisions, and request-local checks. Actual
CUDA statistics regressions require a GPU. The full-grid control and its accepted
boundary behavior remain in place; this correction does not restore progressive
low-grid continuation or establish a near-2x two-chunk runtime.

The paired Sol source is `0240ccb24193cd84fdeb8081a774368bea1aa427`, with unchanged VDN
`9af4cd1e8ff396fecf2237c8b08f588c53f7fa96` and Core
`6b4e05dc30d65740ce8931434607b9907996fb0e`. Sol's local suite passes
276 tests, with 67 hardware/optional-source skips. Fourteen new regressions fail
on the preceding source; the new CUDA statistics checks are included among the
skips. Flow pins this source and adds the new regressions to its paired checks.

## 2026-10-02: previous continuation execution fixes, before the new captures

The captured continuation slowdown has a workload component and two avoidable
execution costs. Under the accepted same-grid boundary control, continuation
low/probe processes 60,264 video rows, versus 24,700 initially. The source
50x38 grid is overridden to 72x54 for continuation, including 12 protected
frames. The 00832 sampler intervals are 152.575 s initially and 259.271 s for
continuation; its continuation low/probe accounts for 169.564 s. Those figures
predate the fixes below and do not establish their latency benefit.

### Dense attention uses Core's native dispatcher

Sol's partitioned dense helper previously called raw PyTorch SDPA, while native
VDN calls `comfy.ops.scaled_dot_product_attention`. Core explicitly prefers
FlashAttention, cuDNN, efficient attention and math in that order when supported.
The partitioned helper now uses the same Core dispatcher. This affects the dense
protected-prefix/global/anchor queries, the all-dense endpoint probe (800 calls
in 00832), and sparse arithmetic-gate references. Query/key row ownership,
scaling, real measure biases, unit-measure `None` masks and sparse selection
are preserved. Core execution errors propagate; they are not retried elsewhere.

The Sol teardown publishes `partitioned_core_dense_calls` and
`partitioned_torch_dense_calls`. They count successful calls to the dispatcher,
including gate references, and do not identify the GPU kernel ultimately chosen.
ComfyUI continuation should report positive Core calls and zero raw-Torch calls.
Standalone CPU oracles still work without Core.

### Factorization error checks move to the prediction boundary

The normal VDN path factors both text and video statistics in each of 50 blocks.
`torch.linalg.cholesky` introduces a CUDA-to-host synchronization per call:
normally 100 checks per actual transformer evaluation. In a VDN diffusion
execution, inference now uses `cholesky_ex(check_errors=False)` and collects only
its integer statuses. The execution validates them together before returning a
prediction to the sampler. Cholesky, triangular solve, inverse construction,
statistics and recurrence arithmetic are unchanged.

The status batch is execution-local, flushes at 128 calls or 4 MiB, and is checked
on its producer stream. A device/stream switch flushes the prior batch. A single
larger status tensor is checked immediately. Resident 50-block execution needs one
host status read. Streamed branch weights check at each attention-block boundary
(usually 50 reads for 100 factors), preserving bounded transfer lifetimes.
Failure still rejects the prediction. Cancellation, nested
execution and a changed next call cannot retain or reuse statuses. CPU, autograd,
compilation, CUDA capture and calls outside the execution owner keep immediate
checking. No matrix, factor or activation is retained for this optimization.

VDN logs `[vdn] factorization checks deferred_calls=100 status_reads=1` for the
resident configuration, or `status_reads=50` with streamed weights.
Additional configured work or a stream switch can increase
the read count; the receipt must be interpreted with the actual adapter setup.

PyTorch documents these synchronization semantics in
[cholesky](https://docs.pytorch.org/docs/main/generated/torch.linalg.cholesky.html)
and [cholesky_ex](https://docs.pytorch.org/docs/main/generated/torch.linalg.cholesky_ex.html).

### Validation and application

Seven dense-dispatch regressions fail on the preceding Sol source. Four
50-block factor/status-read regressions fail on the preceding VDN source.
The candidates preserve CPU output parity, measure bias, native layouts, gradients,
compiled fallback, error propagation, bounded status storage and execution isolation.
Local validation passes 370 VDN tests (including all 12 official-source oracles),
256 Sol tests and 58 paired Core/Flow/VDN/Sol checks. Two CUDA-only factorization
tests are skipped; 64 Sol hardware/optional-source tests are skipped locally.
CUDA parity and producer-stream tests are present but require a GPU; CPU tests
do not qualify GPU latency, rendered output or repeated-run memory stability.

The paired source pins are Sol #37
`e26dced8eeabc5c8a19e34add56e92e4436411fb` and VDN #36
`9af4cd1e8ff396fecf2237c8b08f588c53f7fa96`, alongside unchanged Core #16720
`6b4e05dc30d65740ce8931434607b9907996fb0e`. Flow #93 adds their integration
checks without changing sampling arithmetic. Refresh these existing overlays
through Patcher and restart ComfyUI. Preserve the accepted spatial, softmax,
VDN-linear and audio-carrier controls for a matched timing run.

These fixes remove a backend-policy mismatch and per-layer host barriers. The
full target-grid continuation workload remains. A measured end-to-end speedup,
and any further reduction of that workload, are still unqualified.

## 00832: active forward-cost paths, changed sampling and conditioning

The 00832 capture records Euler, VDN bypass mode, default-adapter strength 1.0
and Turbo strength 0.75 in both `trajectory_begin` events. It does not qualify
the previously reported strength-1 failure. No rendered clip or new visual
verdict accompanies this capture.

The runtime counters confirm six partitioned transformer forwards, six
modulation validations and 300 uniform linear reads before RoPE. The
probe-to-high release reports zero retained activation-scratch entries. These
receipts establish that the new paths executed; the capture does not include
loaded-file hashes proving every installed blob matches a particular PR head.
The audited source heads are Flow `5558c698a8a3e1a3dfcb6b8c48dd45a85d34988c`,
VDN `efff8ecaada268da1f9ccaf7bb00844c54762b55`,
Sol `a812b7064b422d109078f8b1ff288478a7086762` and
Core `6b4e05dc30d65740ce8931434607b9907996fb0e`.

### Workload comparison

The video grids and prefix length match 00827, but the sampling coordinates,
text and references do not. Physical-conditioning and compiled-text hashes
change in both chunks. The manifest now contains two reference latents.

| Receipt | 00827 | 00832 |
|---|---:|---:|
| Low/high logical steps per chunk | 8 / 4 | 6 / 4 |
| Actual transformer evaluations, both chunks including probes | 18 | 18 |
| Forecasts, both chunks | 8 | 4 |
| Initial text / reference rows | 1,253 / 972 | 2,232 / 1,944 |
| Continuation text / reference rows | 1,335 / 972 | 2,314 / 1,944 |
| Initial low packed rows | 27,509 | 29,460 |
| Continuation packed rows | 63,267 | 65,218 |
| Handoff coordinate / sigma | 0.334000 / 0.857510 | 0.400000 / 0.888889 |
| Applied Turbo strength | Unknown | 0.75 |

Each low stage still makes five actual evaluations; reducing its logical steps
removed forecasts. The original actual evaluation count is preserved. DiffAid
and visual-reference preprocessing report the same strengths and block scopes
in both logs. Their input conditioning and sampled coordinates still change.

### Measured costs

| Interval | 00827 initial | 00832 initial | 00827 continuation | 00832 continuation |
|---|---:|---:|---:|---:|
| Low | 59.656 s | 65.232 s | 137.392 s | 140.743 s |
| Probe | 9.782 s | 10.825 s | 33.384 s | 28.821 s |
| High | 71.035 s | 75.930 s | 79.365 s | 86.407 s |
| Complete sampler lifetime | 141.054 s | 152.575 s | 253.660 s | 259.271 s |

00832 takes 452.25 s: 152.575 s initial sampling, 259.271 s continuation
sampling and 40.404 s outside those two lifetimes. It is 21.32 s longer than
00827, comprising 11.521 s initial, 5.611 s continuation and 4.188 s outside
sampling. Probe, transfer and other intra-sampler work are already included in
the complete sampler lifetimes; do not add their timings again.

Continuation low/probe takes 169.564 s versus 76.057 s initially. Under the
accepted `same_grid_target_control`, continuation low/probe uses the complete
72x54 target grid with 62 temporal rows per spatial patch, including 12 protected
frames. Its 60,264 video rows are 2.44x the initial low's 24,700 rows on the
50x38 grid. That cost remains present after the scalar-read and activation-lifetime
fixes. Packed conditioning adds work on top of the video geometry. This is not a
matched latency comparison that isolates the effect of those fixes.

### Memory and quality limits

00832 completes without a captured OOM. H3 remains resident throughout all six
sampling admissions; eviction is bounded to at most one pass in this capture.
The continuation probe and high admissions request headroom without unloading
a model. In 00827 the probe admission unloads the video VAE. The post-high live
allocation is 64,619.157 MiB versus 63,624.104 MiB in 00827. Residency and
conditioning differ, so these endpoint readings do not isolate a leak or a
total-peak reduction. Reserved memory reported before the final high-stage
release reaches 88,960 MiB; it is not a measured peak of live tensor allocation.
One completed prompt does not qualify repeated-run stability.

Protected video and audio prefixes remain exact. The first high-stage Flow
correction is 9.382% of baseline RMS initially and 1.378% in continuation; its
relative amplitude is lower in continuation. This is not a perceptual saturation
measurement and does not establish the cause of overcooking. VDN's own adapter
path applies each named strength once. The separate runtime DoRA loader still
does not identify its checkpoint or strengths in the supplied log.

The remaining source differences are unchanged: partitioned low/probe forces
protected-prefix local queries dense, while native high uses its native Sol
selection; the low/probe endpoint and first high evaluation therefore do not
have interchangeable operator ownership. Eliminating the exact probe or
changing prefix sparsity without an equivalence result would change sampling.
The ten-step Euler schedule also differs from the released eight-step
distillation recipe described in Video DeltaNet section 4.2. Neither difference
is established as the quality cause. Further discrimination needs a controlled
rendered comparison and GPU component timings. Preserve the accepted spatial,
softmax, VDN-linear and audio-carrier controls while obtaining that evidence.

Evidence SHA256: `metrics_00832_.json`
`1029d953a530221c7555cda404a5a0380e8ef40d2f5d353a312baee9b87b0f06`;
`Pasted text(20261002-134934).txt`
`11a94681c95961573ffef2c9f52630255bcddbbf632a45083ffc79e67351a6eb`.

## 00827 and full-strength feedback: quality remains unqualified

The user reports less overcooking with the settings used for 00827, then reports
that `turbo_strength=1` still overcooks, especially subsequent chunks. The 00827
log identifies Euler; 00815 identifies RES multistep. Prompt/reference receipts
and spatial dimensions also differ. Neither capture records the applied Turbo
strength. No capture or matched rendered pair for the subsequent strength-1
trial is supplied here. The visual improvement cannot be attributed to the K/V
workspace patch: it affects partitioned continuation low/probe, while the first
chunk still uses native VDN. Treat the latest report as an unresolved quality
qualification, not proof of a completed fix.

00827 takes 430.93 s versus 424.82 s for 00815. Both have 26 logical calls,
18 actual transformer evaluations and 8 forecasts. Continuation video rows are
60,264 versus 24,700 in initial low, a 2.44x ratio. Both continuation low and
its exact probe use the full target grid under the accepted spatial control.

| 00827 measured interval | First chunk | Continuation |
|---|---:|---:|
| Low | 59.656 s | 137.392 s |
| Probe plus transfer | 10.363 s | 36.903 s |
| High | 71.035 s | 79.365 s |
| Sampler lifetime | 141.054 s | 253.660 s |

The remaining 36.216 s is outside those sampler lifetimes. It is not another
transformer evaluation and this ledger alone does not identify its individual
preprocessing/decode consumers. The full-grid low/probe work is the main measured
continuation cost; there is no basis for expecting twice the first-chunk duration
with these unequal geometries. Arithmetic-gate wall times include queued GPU
work and qualification; do not classify them as compilation alone.

### Forward-cost candidate, preserving the accepted geometry

Flow previously checked uniform protected-prefix timestep labels at every one
of the 50 block replacements. The check performs a CUDA scalar read. Core
constructs the native label table once per forward, so Flow now validates and
expands it once in that forward's local cache. Each downstream replacement
receives its own video index tensor, preserving write isolation. The cache is
not retained across forwards or stages and new native table ownership triggers
new validation. With the six actual partitioned low/probe calls in 00827, this
removes 294 redundant prefix-validation scalar reads without weakening the
prefix invariant.

VDN's linear epilogue also previously created its constant epsilon on CUDA and
read it back at every block. A CPU scalar cached by dtype preserves the exact
FP16/BF16/FP32/FP64 rounding of that constant and removes those device reads.

For normal uniform-grid inference, VDN additionally computes the validated
fixed-grid complement before RoPE, consuming the native raw projection views.
The projected result is retained through softmax and added after its existing
output projection. This replaces three retained raw video tensors and two text
copies, while preserving the native branch arithmetic, prefix/suffix query
groups, key measure, dense decisions, Sol gates, sampler schedule and forecasts.
Streamed branch weights are loaded once and lookahead remains after attention.
Autograd, mixed grids, diagnostics and feature witnesses retain the late path.
At the 00827 video shape, three BF16 raw video copies total 2,471.766 MiB;
the retained hidden-width projection is 617.941 MiB. This 1,853.824 MiB lifetime
difference is not a measured reduction in total GPU peak allocation: the earlier
linear workspace overlaps the native QKV projection instead.

The actual Core 50-block loop verifies one validation per forward, fresh ownership
on the next forward and write isolation for downstream patches on both equal and
mixed grids. Eighteen VDN forward cases exercise strengths 0, 0.5 and 1 across
FP32/FP16/BF16, with raw-input preservation before destructive fake RoPE,
no retained activation copies, one weight load and late-path output parity.
Independent strided-readout cases retain text-state and anchor coverage. These
tests establish operation/lifetime behavior; they do not establish hardware
speed, allocator stability or a full-strength rendered-quality fix.

VDN now publishes its applied adapter mode and named strengths from the actual
application report in `vdn_h3_adapter_config_v1`. Flow records that configuration
alongside the sampler in `trajectory_begin`, including when the Apply node is
cached. Older captures remain strength-unknown. The source audit found one scale
application per named adapter and clone replacement/ejection ownership; no
double-strength cause is established. The partitioned path still forces protected
prefix local queries dense and uses a different operator from native high/Sol
sampling. The eight-low/four-high schedule also differs from the released
eight-step training recipe. Those differences warrant matched discrimination;
neither observation establishes the cause of overcooking. This candidate changes
no Turbo strength, sparse policy or geometry to conceal that open question.

Evidence SHA256: `metrics_00827_.json`
`e9f7918c725c854dfd8528524c9ac4a8e40249a54e7b73dc188cbe4a7336b1d9`;
`Pasted text(20261002-044811).txt`
`e49ea66a4b787b84f587a5596e41a1b470afe73138ec780a6866e8c8fb80135e`.

00815 confirms recovery of the missing continuation forecasts and a 49.20 s
reduction from 00813, but total latency remains 424.82 s. The user reports the
improvement insufficient. Earlier settings-change OOM evidence identified VDN's
separate adapter post-forward residual addition. Repeated-run memory stability
and the reported overcooked second chunk remain open qualifications.

## 00815: forecast recovery measured; full-grid continuation still dominates

The accepted same-grid continuation, normal softmax, normal VDN linear path,
source-carrier audio positions and RES 8-low/4-high schedule remain active.
The geometry and handoff split match 00813. These are successive user runs,
not a controlled GPU benchmark proving identical random inputs.

| Sampling stage | 00813 first chunk | 00815 first chunk | 00813 continuation | 00815 continuation |
|---|---:|---:|---:|---:|
| Low | 60.861 s | 58.984 s | 178.665 s | 135.129 s |
| Exact probe | 9.626 s | 9.516 s | 29.022 s | 29.048 s |
| Transfer | 0.556 s | 0.562 s | 1.577 s | 1.395 s |
| High | 74.359 s | 72.722 s | 77.454 s | 78.423 s |
| Sampler lifetime | 145.423 s | 141.809 s | 288.910 s | 245.839 s |
| Actual low / probe / high calls | 5 / 1 / 3 | 5 / 1 / 3 | 7 / 1 / 3 | 5 / 1 / 3 |
| Low / high forecasts | 3 / 1 | 3 / 1 | 1 / 1 | 3 / 1 |

The latest prompt total is 424.82 s versus 474.02 s, a 49.20 s (10.38%)
improvement. Continuation low is 43.536 s (24.37%) faster, matching the restored
forecasts at steps 4 and 6. The remaining continuation low/probe costs 164.176 s,
66.78% of its sampler lifetime. Its 58,590 video rows compare with 23,712 in
initial low. The 104.030 s difference between the latest two sampler lifetimes
is chiefly low-stage work (+76.144 s) and the exact probe (+19.532 s).
Time outside both sampler lifetimes is 37.172 s.

### Remaining allocation work

VDN's partitioned softmax loop still allocated fresh gathered K/V tensors for
each temporal group. The released native grouped loop already reuses a pair.
VDN #36 now uses a block-local pair sized for the largest local domain: global
sink rows are copied once and each local group overwrites its video rows. Shape,
row order, strides, query maps, key measure, dense/sparse decisions, receipts,
arithmetic gates and the sigma schedule remain unchanged. The pair is released
before anchor attention and learned-linear work, with no retained-pool entry.

Eighteen FP32/FP16/BF16 cases compare with an independent per-frame dense
attention oracle across equal/mixed grids and all four anchor modes. They also
verify one K/V storage pair and release before branch-weight retrieval. The
preceding source passes the numerical comparison but fails all eighteen storage
checks with distinct per-group allocations. Local VDN validation passes 311
tests with 12 official-oracle skips; candidate CI additionally checks the pinned
official source and passes 323 tests. This establishes removal of redundant
allocations, not a GPU latency or peak-memory improvement. The number of
full-grid evaluations and exact probes is unchanged.

### Memory and quality qualification

00815 completes without OOM. Continuation low retains 2,431.7 MiB of Spectrum
history versus 607.9 MiB in 00813, confirming the expected additional anchors.
Stage teardown reports release of those history buffers. One completed run
does not establish repeated-run allocator or retention stability.

The first continuation high-stage Flow correction is 1.398% of baseline RMS,
versus 7.744% in the first chunk. The transfer remains identity, audio copy is
exact, the spatial-mean bridge delta is zero and no acceleration is applied.
The measured continuation seam gradient ratio after/before high is 1.0115;
these local latent observations do not settle visible overcooking elsewhere
in the generated suffix. No new rendered clip is supplied with 00815. The
K/V workspace change preserves numerical inputs and is not a quality fix.

Evidence SHA256: `metrics_00815_.json`
`499d20ce484592d916a78732ee1cd7b9659f776bbf504a11a9170417561cc251`;
`Pasted text(20261002-033346).txt`
`190eef4537df3135eec82f40d0a1dc9fc4d42fe4ae306545bd3eb94aacb86de1`.

## 00812–00813: repeated numerical-receipt resets and continuation cost

Both runs use RES multistep with eight low and four high logical calls, plus an
exact handoff probe. The split is index 8, sigma 0.8575096726417542, video
coordinate 0.3340000817667999. The accepted same-grid continuation control,
normal softmax and normal VDN linear path remain active.

| Sampling stage | 00812 single 7 s | 00813 first chunk | 00813 continuation |
|---|---:|---:|---:|
| Low | 47.351 s | 60.861 s | 178.665 s |
| Exact probe | 9.642 s | 9.626 s | 29.022 s |
| Transfer | 0.566 s | 0.556 s | 1.577 s |
| High | 60.168 s | 74.359 s | 77.454 s |
| Sampler lifetime | 117.743 s | 145.423 s | 288.910 s |
| Actual low / probe / high calls | 5 / 1 / 3 | 5 / 1 / 3 | 7 / 1 / 3 |
| Low / high forecasts | 3 / 1 | 3 / 1 | 1 / 1 |

The logged prompt totals are 138.36 s and 474.02 s. Twice the single-run total
is 276.72 s, leaving 197.30 s excess. Combined sampling accounts for 198.847 s
of that difference; time outside sampling is 1.547 s lower than twice the
single-run overhead. Extra encoding, decoding or saving is not the net cause.

First-chunk low has 23,712 video rows on a 48 x 38 latent grid. Continuation low
has 58,590 video rows on the 70 x 54 target grid, including 12 protected prefix
frames and 50 generated frames. This is 2.4717 times the initial-low video work
before accounting for the extra actual calls. High-stage video rows increase
from 49,140 to 58,590 because of continuation context and temporal padding.

Continuation repeatedly reports `history=0` before an actual call and retains
one anchor afterwards. Steps 4 and 6 execute actuals for `insufficient actual
history`, where initial low forecasts. Source execution reproduces the cause:
Sol's partitioned v1 receipt includes a changing evaluation number, while
Spectrum compares complete receipts as numerical backend identities. Every
otherwise unchanged actual receipt therefore resets the preceding history.
The two observed low calls cost 22.962 s and 23.071 s. Replacing them with
forecasts could avoid most of that model time, but the candidate's actual GPU
schedule, policy decisions and wall time have not been measured.

Sol #37 now publishes v2 partitioned receipts with stable numerical fields.
Completion proof is retained separately on the request and checked against
the active forward. Old completion proof cannot authorize an unexecuted current
call; geometry, measure, mapping, kernel and execution-mode changes still reset
history. Regression tests execute the actual CPU attention request through the
real Spectrum history consumer. The old source resets seven times and executes
all eight calls in the fixed-route fixture; the candidate retains anchors and
executes five actuals and three forecasts. This fixture disables bootstrap and
model-aware forecasting to isolate receipt behavior; it is not a GPU timing or
quality result.

The initial 00813 sampling lifetime is 27.680 s slower than 00812. Sol's five
low arithmetic gates total 14.363 s and its five high gates total 14.441 s,
compared with 0.030 s and 0.090 s in 00812. The 28.684 s extra gate lifetime
accounts for the two model-call outliers within ordinary timing variation.
The gate timer combines kernel execution and independent reference/error
measurement; these receipts do not distinguish compilation, reference
execution or synchronization within that timer. The checks remain enabled.

### Quality and memory limits

The user reports second-chunk overcooking. The first high-stage Flow correction
is 7.841% of baseline RMS in initial 00813 and 1.369% in continuation; the
records do not show doubled Flow correction, clamping or acceleration. The
same-grid handoff has identity transfer, exact audio copy and zero spatial-mean
bridge delta. Prefix restoration remains exact. These checks do not establish
the cause of the visible degradation or demonstrate that forecasts fix it.
Initial low is coarse-grid while continuation low is full-grid, so their
trajectories are not matched quality controls. No guidance, adapter strength,
spatial-control or denoising-schedule change is promoted from these metrics.

Neither latest run OOMs. The repeated initial-low allocation remains
55,342.969 MiB; probe admission evicts VideoVAE once and high admission meets
its finite target. Completion of these two runs does not qualify longer-run
retention or allocator behavior.

Evidence hashes: `metrics_00812_.json`
`64d187052dc5592d774750a962a0e31cdc2252b29540f7785126ce8a39667538`;
`metrics_00813_.json`
`ca0d39df68804e7ec1c715426ebca409e6fd69dfa414987424d8988c88445ba2`;
`Pasted text(20261002-021921).txt`
`b47d89c6c99240c2ad45e03f2ba100812f93d2ae22ad93fe66b975071bc46791`.

## 00788–00789 and third run: VDN adapter residual allocation OOM

00788 and 00789 complete. A separately supplied third-run log fails in the first
actual target-high evaluation of the second chunk; no third-run metrics file or
successful high completion receipt is supplied.

| Run | Low / high schedule | Continuation packed rows | Initial sampler | Continuation sampler | Prompt result |
|---|---:|---:|---:|---:|---|
| 00788, cold | 5 / 3 | 72,676 | 215.483 s | 253.353 s | completed, 10m 16s whole-second log |
| 00789, warm | 5 / 3 | 76,836 | 159.178 s | 273.222 s | completed, 488.48 s |
| Third run, warm | 8 / 4 | 76,836 | low / probe / high: 77.908 / 15.342 / 88.283 s | low / probe: 253.415 / 42.195 s; high failed after 29.522 s | OOM; 10m 15s includes failure handling |

The larger context first appears in successful 00789, with six references
totalling 5,730 rows and 6,178 continuation text rows, compared with four
references totalling 3,658 rows and 4,090 text rows in 00788. The third run has
the same larger packed layout and more sampler steps. Its continuation low
records seven completed actual calls and one forecast; high records zero
completed actual calls. The probe performs evaluation even though its Spectrum
summary has zero scheduled steps. This is not a controlled comparison that
isolates prompt/reference changes, schedule changes or a leak.

The accepted `same_grid_target_control` and `softmax_diagnostic=normal` remain
active. Continuation still has 64,232 video rows, 696 audio rows and 12 protected
video frames on the 56 x 74 target grid. No geometry or guidance adjustment is
promoted by this memory correction.

### Failure and memory evidence

The traceback reaches native MiniMax `MLP.fc1`, PyTorch's forward-hook dispatch,
then VDN `_PostForwardLoRA.__call__` at `return output + delta`. A BF16 tensor of
76,836 x 28,672 elements occupies 4.103485 GiB, matching the requested 4.10 GiB.
CUDA reports 1.74 GiB free, 69.87 GiB currently allocated and a 95.59 GiB device
limit. Both the base output and projected adapter delta are live when the old
addition asks for a third full-sized result.

High admission reports `target_met=True`, requiring 31,054.700 MiB and observing
34,126.382 MiB free through Core's API. The preceding probe evicts VideoVAE in
one pass and meets its finite target. These observations still do not make
admission a worst-case bound on adapter activations or allocator availability.
The first low-entry live allocation is exactly 55,342.969 MiB in both warm
00789 and the third run. Those samples do not show a growing baseline, and do
not establish absence of retained storage, fragmentation or a longer-run leak.

The startup log reports ComfyUI 0.38.0, Patcher stack `3fc64ec0f`, Torch
2.10.0+cu130, Kitchen 0.2.36, Aimdo 0.5.5 and HIGH_VRAM/cudaMallocAsync. It does
not supply exact loaded Core module hashes or an overlay manifest. Therefore
this evidence alone does not establish whether Core #16720 executed. The VDN
post-hook allocation is independently present and needs its own correction.

### VDN correction and qualification

[VDN #36](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/36), head
`68aebc6c5192bbfbd9f12e45803eae92616d1f50`, adds the base output into the newly allocated projection delta
in inference when shape, dtype and strides match for ordinary tensors. The
base output, cached factors and cached bias remain untouched. Bias-only hooks,
gradients, tensor subclasses, promotion, broadcasting and differing layouts
retain the original out-of-place addition. Projection math, factor scaling,
strengths, hook order and native quantized projections are unchanged.

28 targeted checks pass with CPU Torch 2.10.0 and Core #16720's exact head:
16 new adapter workspace cases plus reinjection, curve and low-VRAM contracts.
They cover exact FP32/FP16/BF16 results, gradients, caller/earlier-hook aliases,
cache immutability and both Core/VDN installation orders. The new storage
regression fails on preceding VDN source with three full-sized storage pointers
and passes with two on the correction. Lint, compile and diff checks pass.
Flow's source-contract job pins this VDN head and executes the paired checks
alongside the existing Core tests and historical native fixtures. These CPU
checks do not qualify the user's GPU peak or repeated-generation completion.

In ComfyUI Patcher, update the **VDN #36** and **Flow #93** repository cards and
restart ComfyUI. Retain VDN #33 -> #34 -> #35 -> #36, Flow #89 -> #93,
Sol-H3 #37 at `3f2f244f277fc0d8fafc15fcac724c2cffb7eacf`, Core #16720 at
`6b4e05dc30d65740ce8931434607b9907996fb0e` after existing Core overlays, and the
existing Continuum overlays. Keep the accepted same-grid/current normal suffix
configuration. The next normal run qualifies the corrected VDN hook under the
workload; reproducing the captured failure is not required to establish it.

Evidence SHA256:

- 00788–00789 log: `9a8839ecb2975937e406ad2319c24b4748e50606bceb83ecfc1fdba87c10c009`
- 00788 metrics: `1e00de3f2e6e7e9f8910ea00fde456899deb9c68da54d1a5cec4ba4ecae27315`
- 00789 metrics: `2cddcd9b1b236f81ad2070a7db0f273607f3145a766aa8e8905eca9954da2c27`
- Third-run OOM log: `bc5efbebbc39afd6ed33849731dd17abd66857913ec593e83dab380cc711ae49`

## 00786–00787: captured high-stage LoRA allocation OOM

The user reports six generations preceding these two runs. The supplied log
begins with warm execution. 00786 completes, while 00787 captures an OOM in the
first actual target-high evaluation of the second chunk. The earlier successful
admission and completion observations do not qualify the longer-run VRAM issue.

| Run | Initial sampler | Continuation sampler | Prompt result |
|---|---:|---:|---|
| 00786 | 134.575 s | 264.811 s | completed, 446.53 s |
| 00787 | 117.453 s | 203.393 s, failed | OOM; failure/cleanup log interval 375.31 s |

Both continuation low/probe paths still record 250 successful uniform linear
readouts and 250 fast requests. 00787's completed low/probe stages take
135.867 / 35.791 seconds. Its continuation high does not complete and publishes
no successful high-stage wall receipt. The final counters contain 15 logical
calls and 12 completed actual evaluations; the interrupted high attempt is not
an extra successful evaluation. The accepted same-grid/current normal suffix
configuration is preserved.

### Failure localization

The traceback reaches native MiniMax `MLP.forward`, `fc1`, Core's
`BypassForwardHook._bypass_forward`, then `LoRAAdapter.h` at `return out * scale`.
The first high packed layout has 72,676 rows; the native feed-forward first
projection has 28,672 output columns. Its BF16 output is 3.881317 GiB, matching
the reported 3.88 GiB allocation request. CUDA reports only 2.36 GiB free at the
failure. This is a captured feed-forward adapter peak, not an attention traceback.

High admission had reported `target_met=True`, requesting 30,455.639 MiB and
observing 33,327.572 MiB available by Core's API. That finite target includes a
fixed scratch allowance and is not a worst-case bound for runtime adapter
activations or allocator availability during evaluation.

Source tracing confirms two avoidable output-sized allocations. Core LoRA
projects the adapter delta, then allocates another full output to scale it.
The default bypass subsequently allocates a full `base_out + h_out` result.
These two sites overlap the original base/projection storage at separate peaks.
Changing only the scaling site would leave the same-sized addition pending.

The first low-entry allocated checkpoints are 55,911.719 MiB in 00786 and
55,342.969 MiB in 00787. The latter matches earlier warm entries. This does not
establish the absence of a longer-run ownership leak, retained allocator storage
or fragmentation. The log does not expose ownership over all six preceding
generations. The concrete correction below targets the captured temporary peak;
it is not presented as proof that every source of VRAM pressure is removed.

### Core correction and qualification

[Core #16720](https://github.com/Comfy-Org/ComfyUI/pull/16720), head
`6b4e05dc30d65740ce8931434607b9907996fb0e`, reuses only the native LoRA adapter's
fresh projection output. In inference, scalar scaling writes into that output;
the adapter's custom bypass implementation then adds the base output into the
same owned buffer. It releases its base reference before `g()` and never mutates
the base output, which may alias caller state. Native quantized projections,
full projection shapes, adapter strengths and `g()` invocation remain unchanged.
There is no row chunking, weight folding, new cache, allocator purge or added
synchronization.

Autograd, an overridden `h()` implementation, tensor subclasses, broadcast/dtype
promotion and differing output layouts retain the relevant out-of-place
arithmetic. The implementation uses Core's existing custom-bypass interface;
other adapter implementations are unchanged. Scalar multiplication and addition
retain their original operation order and dtype rounding.

23 targeted CPU adapter tests pass, including exact FP32/FP16/BF16 arithmetic,
gradients, strengths, nested hooks, convolutions, middle weights, base alias
safety, dtype/broadcast/layout behavior, subclasses and storage reuse. Five
MiniMax mask/embedding tests pass on the current upstream base. Two new
storage/lifetime regressions fail on unpatched source at their expected
assertions. Required lint, compile and diff checks pass. The declared-base Core
delta also applies cleanly to the earlier `651ca296a73cd21c12a57eb8741d52e40dc6528f`
runtime source, where 25 adapter and mask tests pass against that overlay.
These CPU checks do not establish GPU peak reduction, repeated-generation
completion, allocator fragmentation behavior or rendered equivalence.

Core #16720 is one xmarre-authored commit above upstream
`77c0f39e343aa83597d67cd95811df9e4fbfef2e`. That upstream base already includes the
separate embedding-temporary lifetime change `2d6b73283af2447bdd065ece4090b8c6b1784544`;
Core #16720's overlay changes only LoRA arithmetic and its tests. Applying that
overlay to an older Core does not install unrelated upstream commits. Flow #93
pins and executes the exact Core adapter/MiniMax tests alongside the preserved
historical source fixtures. Core's upstream workflow runs currently report
`action_required`; no executed upstream CI result is claimed. The owned Flow
workflow supplies the independent pinned CPU checks. VDN #36 and Sol-H3 #37 remain at
`28792f27427e44d312cc4948a15deb78039e3510` and
`3f2f244f277fc0d8fafc15fcac724c2cffb7eacf`, respectively.

In ComfyUI Patcher, enable **ComfyUI/Core #16720** after existing Core overlays,
update **Flow #93**, then restart ComfyUI. Preserve VDN #33 -> #34 -> #35 -> #36,
Flow #89 -> #93, Sol-H3 #37 and the existing Continuum overlays. Retain the
accepted same-grid/current normal suffix configuration. The next normal run
qualifies the new adapter path under the user's workload; the prior six-run
report and the captured OOM do not need to be reproduced to establish the bug.
The reported second-chunk appearance remains a separate open qualification.

Evidence SHA256:

- 00786–00787 log: `e68ebdc98b71e35b25c80a53c1a8620a1186fff0c5419414b0a2d1dc36ae5bed`
- 00786 metrics: `c5663272470c31fab9879d084d7a453f551fba9c978ac2bcd97c4d18c9dc632d`
- 00787 metrics: `abd67ebb4935eb07d15e8484761c9b483e897f828e4d4c1c1b7eb065d398aba0`

## 00779–00780: uniform readout executes; full-grid continuation remains expensive

The supplied log contains a cold two-chunk run followed by a warm two-chunk run
in the same process. Both complete without a captured OOM or cancellation. Each
records 250 `partitioned_vdn_uniform_linear_calls` and 250
`partitioned_vdn_uniform_fast_requested_calls`: the published VDN #36 uniform
readout dispatch executes in all 50 blocks across four actual continuation low
calls and one probe. The requested-fast counter does not prove successful fused
compilation, and these are not matched eager/fused benchmarks.

| Run | Initial low / probe / high | Continuation low / probe / high | Initial / continuation sampler | Prompt completion |
|---|---:|---:|---:|---:|
| 00779, cold | 139.498 / 12.512 / 65.471 s | 132.128 / 43.238 / 72.788 s | 218.469 / 251.878 s | 10m 01s, whole-second log |
| 00780, warm | 48.177 / 12.683 / 49.449 s | 123.568 / 34.492 / 66.386 s | 110.961 / 227.842 s | 385.67 s |

The warm prompt is 5.9% shorter than 00776's 409.73 seconds, but the inputs are
not a controlled A/B comparison. Continuation low falls from 133.379 to 123.568
seconds while high rises from 60.414 to 66.386 seconds. Do not attribute the whole
prompt difference to the shortcut or claim the remaining speed issue resolved.

Warm H3 Core preparation takes 0.200–0.411 seconds per stage. The continuation's
model-call intervals total 210.929 of its 227.842 sampler seconds, including the
forecast, so 92.6% of that sampler interval is inside model calls. Total sampler
time is 338.804 seconds; another 46.866 seconds of prompt time includes
conditioning, decoding, assembly and output. The H3 preparation receipts do not
measure all work before sampling. The text encoder is requested again before
00780's first chunk, after a partial eviction during 00779; Core's
"25883.83 MB loaded" is total resident model size, not bytes transferred or a
timed full reload. Continuum's conditioning cache is invocation-scoped, so an
unchanged compiled first-chunk text hash still reports `cache_hit=False` on a new
prompt. No per-node timing separates those costs in this log.

The accepted `same_grid_target_control` remains active. Initial low has 27,040
video rows; continuation low/probe has 64,232, including 12,432 protected-prefix
rows. This is 2.38 times the video rows; warm continuation low takes 2.56 times
the initial low interval. Source tracing confirms that global, anchor and
protected-prefix query groups remain dense, and the exact probe performs dense
warmup. Normal suffix low attention still executes 1,650 sparse calls; low/probe
record 3,200/800 validated unit-measure calls. These are concrete workload and
attention-domain differences, not a measured breakdown of GPU kernel time.
Both chunks retain nine logical calls, seven actual calls and two forecasts;
there are no added continuation evaluations. Do not change the accepted grid or
prefix semantics merely to make the timing ratios match.

### Memory observations and ownership

Every `bounded_headroom_v2` admission target is met. The first low-entry live
allocation in 00780 again returns to 55,342.969 MiB, matching earlier warm
entries. Only one subsequent run is supplied, so this does not establish absence
of an accumulating leak over a longer sequence or qualify the user's reported
transient near-overflow peak.

Before 00780's continuation probe, the admission requests 30,208.095 MiB and
observes 26,212.918 MiB free. One 1.905-second eviction pass raises free memory to
30,797.509 MiB by partially unloading the text encoder and unloading VideoVAE.
H3 remains protected. The partial text-encoder receipt frees 1,456.74 MiB and
leaves 24,427.09 MiB loaded. In 00779 it frees 5,158.95 MiB and leaves 20,724.88
MiB loaded. Both sums match the 25,883.83 MiB text encoder, not the 19,996.14 MiB
H3 model. A partial unload can leave the model in Core's registry and therefore
need not increment the receipt's fully unloaded-model count.

00780's probe-to-high release reduces allocated/reserved storage from
62,014.686 / 81,696 MiB to 58,825.164 / 61,056 MiB. Terminal-high release reduces
it from 63,145.405 / 86,400 MiB to 62,013.876 / 64,064 MiB. The warm probe-admission
reserved checkpoint is 84,512 MiB, versus 91,232 MiB in 00776. These are sampled
allocator checkpoints, not whole-run GPU peak measurements. Reserved storage is
distinct from live storage and does not prove spill. End-of-run allocation also
depends on which text-encoder/VAE weights remain resident. There is no new
allocator purge or source change justified solely by these readings.

### Appearance qualification

No rendered clip accompanies these metrics. Both runs preserve identity frame
registration and same-grid handoff, with source-state reconstruction maximum
error 4.768e-7. The continuation-only DC bridge reports an exactly zero offset
(RMS and absolute maximum both zero); its source adds only a channel-wise spatial
mean offset to the first suffix token. It cannot account for a new direct tone
adjustment in these particular receipts. Boundary stabilization and post-high
video repair remain unapplied. In 00780 the first actual high-call Flow correction
ratio is 0.078587 for initial sampling and 0.014698 for continuation, so these
receipts do not show an increased Flow correction ratio.

Initial sampling still uses reduced-grid low followed by learned upscale;
continuation uses target-grid low plus exact-prefix context. Compiled first-chunk
text is identical across these two runs, but continuation text differs (3,212
versus 3,221 packed text rows). The records therefore do not isolate either the
performance effect of fusion or the cause of the reported overcooked appearance.
That rendered quality report remains open. No guidance, geometry, noise or tone
change is promoted from the latent boundary metrics alone.

VDN #36 remains `28792f27427e44d312cc4948a15deb78039e3510`; Sol-H3 #37 remains
`3f2f244f277fc0d8fafc15fcac724c2cffb7eacf`. This qualification adds evidence only;
the exact dependency pins and Flow runtime are unchanged. Keep the current
Patcher overlay stack and accepted same-grid/current normal suffix configuration.
A rendered clip is still needed to locate the appearance change. Remaining
component costs and full transient peaks are not captured by these receipts.

Evidence SHA256:

- 00779–00780 log: `ba49982ca553bacc88624d91efc4d35d27f430e37470cc4052bb51ba0e60671d`
- 00779 metrics: `6339a01ad6bbe6d7721a892a875b8ed86ca5421e8662b3f5b467a02a2d4b93f0`
- 00780 metrics: `582e23cd999df3c23bdbe0f9e7395dccf7cbb63487905db809a6c4b72beac6f0`

## 00772–00776: full-grid continuation cost and remaining workspace pressure

The supplied runs execute `bounded_headroom_v2`, and every admission target is
met. Warm Core preparation takes less than half a second per stage. Initial
low-entry live allocation returns to approximately 55,342.969 MiB. These
checkpoints do not establish accumulating live storage between prompts, and
they do not capture the user's observed transient near-overflow peak.

| Run | Chunks | Initial sampler | Continuation sampler | Prompt completion |
|---|---|---:|---:|---:|
| 00772 | 1, cold | 226.365 s | — | 333.77 s |
| 00773 | 1, warm | 131.779 s | — | 155.39 s |
| 00774 | 2, warm | 144.833 s | 259.220 s | 449.62 s |
| 00776 | 2, warm | 128.362 s | 233.988 s | 409.73 s |

00774 takes 2.89 times the warm single-chunk prompt interval. Its initial
low/probe/high stages take 63.419 / 13.204 / 67.549 seconds; continuation takes
133.799 / 49.450 / 72.242 seconds. The 00773 single-chunk baseline takes
56.947 / 12.576 / 61.614 seconds. The continuation probe includes a 10.954-second
VideoVAE eviction; Core preparation itself takes 0.238 seconds.

The accepted `same_grid_target_control` carries 62 latent frames, including
12 protected-prefix frames, on the full target grid during low/probe. In these
samples, initial low has 27,040 video rows and continuation has 64,232: 2.38 times
as many rows. Initial high has 53,872 video rows. 00776 transposes the spatial
axes relative to 00774 while retaining those row counts, and its references/text
differ; it is not a matched benchmark. Both two-chunk runs have 18 logical calls,
14 actual calls and four forecasts, with no extra continuation evaluations.

The current low receipts contain 3,200 unit-measure calls and 1,650 sparse calls;
probe contains 800 unit-measure dense-warmup calls. Dense-suffix discriminator
counts are absent. Retain the actual normal suffix configuration used by these
runs; the earlier handoff's instruction to retain a dense-suffix control does not
describe this evidence.

00776 continuation low/probe/high take 133.379 / 36.438 / 60.414 seconds. At
probe admission the allocator reports 66,356.254 MiB allocated and 91,232 MiB
reserved. One 2.576-second VideoVAE eviction meets the target. At terminal high
release, allocated/reserved memory falls from 59,703.858 / 88,672 MiB to
58,572.576 / 62,592 MiB. Reserved pool size is distinct from live storage and
does not identify spill. The reported transient pressure remains unqualified.

### Confirmed source changes

VDN #36 `28792f27427e44d312cc4948a15deb78039e3510` preserves the finite admission
and attention-lifetime corrections, and adds two bounded changes:

- Identical-grid, exact-unit-measure partitioned readout reuses the released
  fixed-grid implementation on an execution-local branch copy. The former
  general path always selected unfused gather/epilogue and did not request the
  fast-kernel query layout. With fast kernels disabled, reuse remains eager.
  Mixed grids, nonunit measures, Q convolution, witnesses, diagnostics and
  alternate carrier policies retain the general path. Bounds are validated
  before dispatch, including the anchor-trimmed domain.
- Native runtime and general partitioned readouts release normalized K/V and
  beta after statistics, then statistics after scans. Native runtime also
  releases query features and gathered state after matmul. These tensors no
  longer overlap later workspaces. Retained scan-bank ownership is preserved;
  there is no allocator purge or added synchronization.

`partitioned_vdn_uniform_linear_calls` positively records successful shortcut
calls. `partitioned_vdn_uniform_fast_requested_calls` records the flag and does
not prove compilation succeeded. Existing compile-failure eager fallbacks remain.
These receipts count readout work, not completed sampler success.

The reviewed candidate is checkpointed on GitHub. 102 focused CPU tests pass,
including released/general numerical oracles, full-forward receipts, text and
anchor semantics, nested execution, shared selector stability, malformed inputs,
diagnostics and witnesses. Six new lifetime/forward-route regressions fail at
their expected assertions on the preceding source. Required lint, compile and
diff checks pass. GPU speed, peak memory and rendered equivalence remain
unqualified. Fused kernels can change floating-point rounding.

### Second-chunk appearance remains open

The user reports a visibly overcooked second chunk. 00776 has the same sigma
coordinates and logical/actual call counts in both chunks. The same-grid transfer
is identity, invokes no learned upscaler for continuation, and reconstructs the
source state with maximum error 4.768e-7. Frame registration is identity; boundary
stabilization and post-high video repair are not applied. The external patch
profile retains the same patch count and declared strengths through both chunks.
At high entry, the Flow guidance correction/baseline RMS ratio is 0.094286 for the
initial chunk and 0.012661 for continuation. These observations do not establish
a guidance-strength increase or an extra re-noising operation at handoff.

The initial chunk uses reduced-grid low sampling followed by a learned upscale;
the accepted continuation samples low directly on the target grid and has
protected-prefix context. That is a concrete path difference, not proof of the
rendered degradation's cause. Boundary latent metrics cannot judge texture,
color or saturation over the full suffix. No quality correction is promoted from
these logs. Qualification needs the rendered clip along with the next metrics.

Use ComfyUI Patcher repository-card **Update** for VDN #36 and Flow #93, preserving
VDN #33 -> #34 -> #35 -> #36 and Flow #89 -> #93, then restart ComfyUI. Sol-H3 #37
remains pinned at `3f2f244f277fc0d8fafc15fcac724c2cffb7eacf`; existing Continuum
overlays remain part of the stack. Retain the accepted same-grid configuration.
The next repeated two-chunk run qualifies shortcut counts, timing, completion,
memory and rendered appearance; no changed guidance or geometry is requested.

Evidence SHA256:

- 00772–00774 log: `13d59d3cfd82286095aae7c5314263832f8053d756746fcc84c9973646cbd6f3`
- 00772 metrics: `213f0205ca0b81c067b278261d2d65d7305c84971e2d101cab8f493f700edb2c`
- 00773 metrics: `cf06d613abdfa17fe9a6242891d5175fde2297236bad3b5538a9daadd98c32dd`
- 00774 metrics: `e119b9fba49d316a4a136a41380fd0c62e4f10450de4d1ee0e58e1aa03294ee6`
- 00776 log: `3a37c37651b81385066578c1ba6617af71309a3204aa86b2fd7ebfc9b12fe5cf`
- 00776 metrics: `4daf6727b987796efd5735ef2dcae2b1bb5afc0a1ad3593106dff64a4ac0c295`

## 00759–00762: warm admission succeeds; continuation memory needs qualification

The supplied startup and all stage receipts execute VDN's finite
`bounded_headroom_v1` policy. Its installed receipt reports retained buffers and
actual fast kernels enabled. The old unconditional VDN purge is absent.
00760's warm single 7-second chunk completes in **137.40 seconds**, with low-stage
Core preparation **0.257 seconds**. Warm stage preparation stays below half a
second in the subsequent runs. The new delay is inside continuation evaluation.

| Run | Chunks | Completed sampler wall | Prompt completion |
|---|---|---:|---:|
| 00759 | 1, cold | 211.780 s | 312.89 s |
| 00760 | 1, warm | 113.102 s | 137.40 s |
| 00761 | 2, warm | 136.796 + 365.739 s | 550.31 s |
| 00762 | 2, warm, canceled | first chunk 107.014 s; continuation low 798.120 s | absent |

The 00762 metrics file was saved after the first chunk. It does not contain the
canceled continuation's final counters or complete sampler interval. Its log
ends during probe admission, without an OOM traceback. The user reports VRAM
overflow and cancellation followed by a ComfyUI restart; distinguish that report
from a captured exception.

The accepted same-grid continuation performs low/probe work on the full 66x58
target grid, with 62 latent frames including 12 protected-prefix frames. Packed
video rows increase from 23,920 in the initial low stage to 59,334 in the
continuation. 00761 continuation low/probe/high take **225.744 / 59.904 / 75.559
seconds**. Both continuation low stages execute four actual calls plus one
forecast: canonical history eligibility is now observed at runtime.

00762 changes only the prompt according to the user. Its continuation has fewer
text rows (3,916 versus 4,132), the same video geometry, and more available
memory at low entry (35,217.834 versus 33,669.371 MiB by Core's API). Nevertheless,
later actual calls deteriorate sharply. After low, allocated/reserved receipts
reach **66,320.374 / 95,552.000 MiB**. The first-stage allocation returns to
approximately 55,342.969 MiB between prompts; this evidence does not establish
an accumulating live-tensor leak. Reserved pool size alone does not identify
live ownership or prove physical spill, especially under `cudaMallocAsync`.

Source tracing identifies three concrete issues:

- Validated unit prefix measure still allocates an all-zero key bias and passes
  it as an SDPA mask. PyTorch 2.10's CUDA FlashAttention selector rejects any
  non-null mask. The supplied logs do not identify the selected dense backend,
  so its share of the measured delay remains unproven.
- Partitioned VDN retains the QKV projection through raw/RoPE views, the cloned
  V tensor, and softmax/gated output until the linear readout returns. Their last
  attention use precedes that large workspace allocation.
- 00761 probe admission requests 29,028.840 MiB but observes only 28,462.990 MiB
  after Core eviction. It continues despite the remaining shortfall. The fixed
  10 GiB allowance is a heuristic, not an attention peak-memory bound.

Sol-H3 #37 candidate `3f2f244f277fc0d8fafc15fcac724c2cffb7eacf` omits only the
validated exact-zero bias, retaining prefix validation and completion metadata.
Nonzero key measures remain biased. `partitioned_unit_measure_calls` counts the
validated identity route. VDN #36 candidate
`e4684c45c157f07ccb83c3ce9cb4d35c58d9ac58` releases attention temporaries after
their last use and uses `bounded_headroom_v2`: one further finite eviction pass
is permitted after partial recovery; an unmet target fails before evaluation
and cleans prepared additional models. Receipts include `eviction_passes` and
`target_met`. Prepared models remain protected; sufficient headroom still skips
eviction.

These commits are checkpointed on GitHub before review. Focused CPU regressions
cover unmasked dense-oracle equivalence and request-owned receipts, malformed
zero-measure rejection, raw-copy preservation with and without retained buffers,
gated/ungated attention lifetimes, bounded recovery and cleanup on failure. The
canonical Flow/VDN/Sol source oracle passes. They do not qualify GPU peak usage,
speed, selected backend or rendered equivalence. Omitting the zero mask may
change floating-point rounding when PyTorch chooses a different fused kernel.

Use Patcher repository-card **Update** actions for Sol-H3 #37, VDN
#33 -> #34 -> #35 -> #36, and Flow #89 -> #93, then restart ComfyUI. Retain the
accepted same-grid configuration. The runtime qualification needs
one repeated two-chunk prompt followed by a prompt-only change, checking the
new admission policy, identity-measure counts, completion, peak memory and
boundary output; the single-chunk baseline already establishes the warm loading
improvement.

Evidence SHA256:

- Log: `77f7459aa4d505e0425bfb2f9189c3814820e6bba6eaa1054fc913d0309b6581`
- 00759 metrics: `b8f0a8fbb30a8cb85e9df7f6d7b06462b9b932178047b5fa665a94c622a6ed3d`
- 00760 metrics: `429fa2e59d67c0ffd92a6b9a9fbedb88ac539f945987fb97428c0cf7bd3045bd`
- 00761 metrics: `97e53a94916aebe5348d5df32e555ac0ba56073e9849a68308d8b5998897dc79`
- 00762 metrics: `54e27fff450f3876e6e987c8ef151762ba76434f489359803631d7969b28cb08`

## Earlier boundary qualification

Matched SM120 A/B/C plus complete partitioned learned-linear bypass
discrimination is complete. Candidate C changed the internal pre-high trajectory
but **00722 still rendered the frame shift**. 00724 then bypassed the complete
VDN learned-linear complement and **the visible frame shift still remained**.
No VDN geometry fix is promoted.

00726 successfully executed the direct same-grid target control. The user still
observed a slight frame shift, but reported that it was **substantially less
prominent**. This is rendered evidence that the low-resolution -> high-resolution
stage transition is materially involved, not evidence that it is the sole cause.
The run also exposed a separate Flow clean-state ownership bug: a source-residual
handoff was later inverted as though it had used the deterministic Gaussian-noise
contract.

00727 repeated the same-grid arm after that ownership correction. The correction
validated exactly: the actual clean postprocess tensor was used, inverse recovery
was absent, identity transfer/frame gauge remained identity, and the one-token DC
delta collapsed to zero. **A very slight rendered frame shift still remains.**
The residual defect is therefore not explained by the learned transfer, the
invalid clean inverse, the DC bridge, target-high recreation, or the complete
partitioned VDN learned-linear complement. Its measurable boundary motion is
already present in the same-grid partitioned low/probe output and largely
survives target-high.

00728 completed the same-domain dense-suffix intervention. The residual rendered
shift and audio burst remained. The next qualification changes audio ownership
as described below. No rendered production fix is promoted.

## 00755–00758: verify loaded source before another generation

The user reports two generations with VDN fast kernels enabled, then two with
them disabled, changing prompts, reference pictures or LoRA strength between
generations. A fresh startup precedes 00755; the other three runs share that
process. The flag mapping is the user's report: supplied VDN receipts do not
expose the applied flag. Complete prompt ranges are log lines 302–616,
617–854, 855–1113 and 1114–1351.

Evidence SHA256:

- Log: `5eb9df11293ff8fca41b5e62d9c9522e4c163ddc200ad3c5c617fb53b767256c`
- 00755 metrics: `b3cb9ddd4d20bde392f04e9f82f6fc75881c62eaac77f1ecc8f04b7919beb896`
- 00756 metrics: `bbc30372f46b7dba27e0e53c8b332634d4175f78d019f0e0446af831e9c818e4`
- 00757 metrics: `5a0d9cb5ad6759226e8c7eab847f81fe8991548181aa4071286c0dd7ec2864c8`
- 00758 metrics: `22cec1345070ddf2bac4185a45575b745d2c7a2090b9b87f71d33ad8de450b20`

Each run is one standalone chunk: three sampler invocations, nine logical
calls, seven actual transformer calls and two forecasts. There is no
continuation prefix. These runs cannot qualify the canonical equal-grid
history fix or the previous frame/audio boundary defect.

| Host wall interval | 00755 (on) | 00756 (on) | 00757 (off) | 00758 (off) |
|---|---:|---:|---:|---:|
| Prompt completion | 393.420 s | 261.360 s | 201.190 s | 233.880 s |
| Whole sampler | 279.356 s | 172.168 s | 160.613 s | 151.820 s |
| Timed model-call sum | 141.112 s | 112.571 s | 125.988 s | 107.525 s |
| Sampler outside timed calls | 138.245 s | 59.597 s | 34.625 s | 44.295 s |
| Outside whole sampler | 114.064 s | 89.192 s | 40.577 s | 82.060 s |
| Low stage | 201.032 s | 101.442 s | 77.598 s | 84.759 s |
| Exact probe | 12.067 s | 12.581 s | 12.296 s | 11.901 s |
| High stage | 65.114 s | 57.538 s | 70.157 s | 54.589 s |

Model calls, preparation, profile work and kernel arithmetic gates overlap the
stage totals. Outside-sampler time includes conditioning and final decode; it
is not a transfer measurement. Logged host-memory trims sum to 1.795 / 7.036 /
6.477 / 5.204 seconds and are nested in these totals. The first Spectrum profile
reports build 19.004467 seconds and lookup 9.506764 seconds; those fields
overlap, so do not add them as independent preparation costs. Unix event
timestamps use `time.time_ns`, while durations use `perf_counter`; inferred
post-low windows disagree by 3.862–6.567 seconds. Avoid a precise pre-first-call
transfer estimate made by mixing those clocks.

Source grids are 50x38 for 00755/00756 and 46x40 for 00757/00758. Targets change
from 72x54 to 66x58; text/reference row counts differ, and the first run is cold.
These observations cannot isolate a causal fast-kernel speedup. Changing LoRA
strength may legitimately require Core patch reconciliation; keep that separate
from VDN's unconditional eviction.

All **twelve** admission receipts still use the pre-#36 form
`sampling admission eviction stage=...`. None has `bounded_headroom_v1`,
`prepare_elapsed_ms` or the earlier clone-preservation fields. Cold low reports
`unloaded=2`; every warm low reports `unloaded=4`, then requests H3. Warm
allocation falls from 53,486.034 to 20,682.938 MiB. Each prompt also requests the
25,883.83 MiB text encoder and final-decode VAEs. The log cannot identify the
four evicted entries or measure transfer time separately.

The startup `patcher/stack` label describes **ComfyUI Core**, not the VDN
checkout; it is not a VDN overlay receipt. Current Patcher source at `65cc7d6`
fetches each enabled PR head during **Update**, validates its captured
test-merge parents and applies the declared-base delta with
`git apply --3way --index`. An isolated reproduction on VDN main
`b78e94d0365ffc5059924a048af707e565d0380e` in #33 -> #34 -> #35 -> #36 order
produces the exact published `64a33b6` tree and hybrid file. Hybrid SHA256 is
`e04fb51624f66588e836c80765c55a585be7b76dcc8ad3a3a8aa4d89e23bef5d`;
the old receipt text is absent. This does not establish the user's Patcher
version, overlay list, Update result or a Patcher failure.

The prior "refresh" handoff was imprecise. Use the **VDN repository card's
Update action** to rebuild the enabled stack, then restart ComfyUI. Refreshing
installation details or checkpoint history and previewing an update do not
apply the stack. Retain #33 -> #34 -> #35 -> #36 on the existing tracked base;
Flow remains #89 -> #93, Sol-H3 #37 and Continuum #36 -> #37 remain in place.

VDN #36 adds source receipts at `01ae7bbcab305a46a735f2468c71937d1b6fc35c`,
still **one clean commit** above unchanged #35. At startup,
`sampling admission source policy=bounded_headroom_v1` reports the loaded hybrid
and importing package paths; `policy=unversioned` exposes an older shared import
without assuming its cause. Applying VDN reports `sampling admission installed`,
retained buffers and the branch's actual fast-kernel flag. These are loaded
module paths, not Git revisions or disk hashes. The memory policy and numerical
behavior are unchanged from reviewed `64a33b6`.

Check the startup receipt **before queuing a generation**. If absent,
unversioned or pointed at another package, keep the VDN Patcher Update operation
log and startup log; another full run is not needed for import provenance.
Startup alone does not prove the current MODEL uses that wrapper: successful
retained CUDA sampling must emit preparation timings and
`sampling admission policy=bounded_headroom_v1`. Only then can the finite
admission correction be evaluated. GPU speed and generated quality remain
unqualified. Final source/CI validation is recorded on the paired PRs.

## 00746–00748: standalone model-loading attribution and finite admission

The user reports some improvement after disabling VDN fast kernels, with model
loading still slow when changing prompts or reference images. The appended log
contains the tail of an earlier run, followed by complete runs 00746 (lines
124–370), 00747 (371–618) and 00748 (619–866). The receipts map to the three
metrics files by their stage geometry, work counts and wall intervals.

Evidence SHA256:

- Log: `ae7d6f718d6c06fed27cdf6d3f7f1994c13199bfb7ac3ecfc724fb67f0aabcc3`
- 00746 metrics: `4873586a368276993a2a08a72be719e5915f0aca8d698db51dd3df45c7b0d394`
- 00747 metrics: `d992771d45599272bddead759e4a2c302e4264e41db02f8ce06013cdeef7f39c`
- 00748 metrics: `0bad889521dd7367d01e60085a80f7b8ae8bb836987270b4bd5d59ec23bac2d9`

These are **single-chunk** runs: three sampler invocations, two history
boundaries, nine logical calls, seven actual transformer calls and two forecasts
(one low, one high). There is no protected continuation prefix or seam audit.
The low grids are 50x38 / 44x44 / 50x38, and the high grids are 72x54 / 62x62 /
72x54. The learned transfer actually executes and takes 0.532 / 0.596 / 0.548
seconds. The same-grid continuation control is configured but has no carried
prefix to act on. These runs do not exercise the Sol equal-grid continuation
history correction and cannot establish a boundary-quality result.

| Host wall interval | 00746 | 00747 | 00748 |
|---|---:|---:|---:|
| Prompt completion | 196.170 s | 223.350 s | 246.000 s |
| Whole sampler | 153.828 s | 177.530 s | 186.032 s |
| Timed model-call sum | 111.703 s | 130.963 s | 127.565 s |
| Sampler outside model-call timers | 42.125 s | 46.567 s | 58.468 s |
| Low stage before first timed call | 28.096 s | 29.980 s | 42.549 s |
| Outside whole sampler | 42.342 s | 45.820 s | 59.968 s |

The low-stage pre-call interval is nested inside sampler wall time and includes
preparation beyond weight transfers. The outside-sampler interval includes both
conditioning and decode. Neither interval isolates physical transfer latency.
Logged host-memory trim hooks contribute 6.451 / 8.345 / 9.101 seconds across
44 / 52 / 52 receipts, including skipped checks. These costs are also nested;
do not add them again to prompt completion.

All nine low/probe/high VDN admission lines use the format preceding VDN #36;
none has its protected-clone or preparation-time fields. Each low stage reports
`unloaded=4`, allocator allocation falls from 53,486.034 to 20,682.938 MiB,
and H3 is requested immediately afterward. Each prompt also requests the
25,883.83 MiB text encoder, and both VAEs are requested for final decode.
The old log does not identify the four evicted models or time physical transfers.
The receipts establish the older admission path; overlay order, checkout and
loaded-module causes cannot be distinguished from this log.

Source tracing identifies an unconditional `free_memory(1e30, ...)` request in
retained VDN preparation. The earlier clone-preservation correction still made
that request for unrelated models. `fast_kernels` selects fused branch
arithmetic and does not gate memory admission. Compilation can affect model-call
cost, but cannot explain this explicit pre-call purge. Prompts, references and
geometry change between these runs, and the logs do not identify the selected
fast-kernel flag, so they are not a controlled kernel on/off benchmark.
Conditioning changes legitimately require new encoding; reusing resident
weights must not reuse stale conditioning.

VDN #36 is updated to `64a33b69c2e92014e2adfe43c3e572f7799ce6a6`, one clean
commit above unchanged #35 `e253432b9f79db91a0d99461bbd85fb1574a39b3`.
Core now prepares required models and reconciles clones/patches first. VDN then
requests Core's sampling/conditioning estimate and reserved-memory policy plus
the existing shared 10 GiB retained allowance. Sufficient headroom skips
eviction; pressure preserves the prepared H3, additional models and their
patch backings while using Core's eviction policy. Admission failure cleans
prepared additional-model execution state and propagates the original error.
The allowance is a heuristic, not a strict peak-memory guarantee.

Two sufficient-headroom regressions fail on the previous #36 head. The initial
complete pinned suite passes 255 tests; the final failure-cleanup revision
passes 24 targeted pinned checks. Actual Core partial eviction, required model
ownership, changed patch UUIDs, device/dead references, force flags, identity and
failure propagation are covered. Final-head CI and current-Core checks are
recorded on VDN #36; source correctness does not establish GPU speed or quality.
Flow runtime, attention arithmetic, schedules and all geometry policies are
unchanged by this qualification update.

Through Patcher, refresh **VDN #36 after #35** while retaining the existing
#33 -> #34 -> #35 -> #36 stack. Refresh Flow #93 after #89; retain Sol-H3 #37
and the existing Continuum overlays. Restart ComfyUI. Keep fast kernels off
and other settings fixed for the next loading measurement. Require
`sampling preparation ... prepare_elapsed_ms=...` followed by
`sampling admission policy=bounded_headroom_v1`. Sufficient available headroom
should report `eviction_requested=False`; under pressure inspect the finite
request, free-memory fields and eviction time. A zero full-unload count can
represent partial unloading. No particular protected-clone count is universal.
GPU latency and rendered output remain unqualified, and the prior continuation
frame-shift defect remains separate.

## 00737/00738: slow equal-grid control; installed admission fix not evidenced

The new user report is latency only. The attached appended log contains two
prompts: lines 321–810 correspond to 00737, and lines 811–1252 to 00738. Their
completion receipts are 14:33 (873 seconds) and 10:08 (608 seconds). A fresh
startup precedes the first prompt; the second reuses that process. Both prompts
and physical conditioning hashes differ, so this pair is not a matched
cold-versus-hot benchmark.

Evidence SHA256:

- 00737 metrics: `f5263a72ff4df5af2dfc7b8d3cf4d31236ccd4d96967e69b7c3fc6bb282520d3`
- 00738 metrics: `7cf2db3c0ef42f23ede5f9d88bea1d10ea18a5540ac5181de393166dc0a08460`
- appended log: `963e776ffb10fbbb299e6e64eaf49ff7c3492cb85986c7054b75602752b75a76`

### Effective preparation code

All twelve VDN admission receipts have the old format, for example log line
1044: `sampling admission eviction stage=low unloaded=2 allocator_before=...`.
None includes `kept_resident_h3`, `eviction_elapsed_ms` or `sampling preparation`.
VDN #36's retained CUDA path always adds those fields at this same log site.
The observed admission wrapper therefore predates #36; these runs cannot
validate its clone-preservation fix. The files do not distinguish a missing
Patcher overlay, overwritten overlay order, stale checkout or loaded module.
Do not infer the installation cause or claim that #36 saved time.

The hot first low stage again evicts four loaded models and then requests H3;
continuation low evicts two and requests H3. The old log still does not name
which models were evicted or time eviction versus Core preparation.
Both physical encodes are cache misses and load the text encoder. Continuum's
conditioning cache remains invocation-local. 00738's first Spectrum profile is
also a miss (build 11.007314 seconds, lookup 5.513360 seconds); its continuation
lookup is a hit (0.029433 seconds). These profile fields overlap existing
intervals and must not be added together as independent preparation costs.

### Workload and measured cost

Both runs use `same_grid_target_control`, normal softmax and normal VDN linear
attention. The configured progressive source is 38x50; the initial chunk uses
that source, but continuation low/probe executes directly at the **54x72 target
latent grid** (864x1152 decoded geometry). Its transformer contains 11,664
protected-prefix plus 48,600 generated-suffix video rows: 60,264 total. The
learned continuation transfer is the control's identity operation. This control
still pays target-grid low/probe cost.

| Stage wall | 00737 first | 00738 first | 00737 continuation | 00738 continuation |
|---|---:|---:|---:|---:|
| Low | 230.838 s | 92.521 s | 216.190 s | 198.940 s |
| Probe | 10.229 s | 10.904 s | 42.943 s | 39.336 s |
| High | 63.607 s | 51.616 s | 65.397 s | 55.898 s |
| Whole sampler | 306.182 s | 155.720 s | 328.109 s | 297.495 s |

00738 has **453.215 seconds** in its two samplers and **356.760 seconds** in
all timed model calls, leaving 96.455 seconds outside those call timers.
Prompt completion exceeds the sampler sum by 154.785 seconds; that remainder
includes work before and after sampling, including conditioning and decode.
The metrics do not independently isolate text-encoder time.

The first low stage spends 31.134 seconds before its first timed model call;
continuation low spends 16.982 seconds there. Their model-call sums are
53.956 / 159.309 seconds and their remaining low-stage costs are 38.565 / 39.630
seconds. These are nested attribution intervals, not additional costs to sum.
A faster model-admission path alone cannot eliminate 356.760 seconds of model
work or the full-grid control workload.

### Canonical equal-grid history correction

The source-backed history limitation recorded under 00733 is reproduced:
`backend-history diagnostic opaque` names the actual Flow partitioned block
wrapper at 00738 log line 1049. Spectrum proposes forecasts at low steps 2 and
3, but both calls execute actual transformer work. Continuation low has **five
actual calls and zero forecasts**. Decision labels are not execution receipts.

Sol #37 head `012ae39aa2137699ee8bbcef977b7965a9cb1446` extends the two classifiers only for the canonical API-1 equal-grid
control. It verifies matching spatial axes, zero key measures, current ranges,
exact query ownership and the semantic digest; its replacement classifier also
binds the contract to actual Flow latent-grid closure fields and the inherited
replacement. Full signatures, VDN binding, provider/warmup transitions and
request-owned completion receipts remain required. Equal row products on
unequal grids, stale state and malformed metadata remain opaque. This addresses
the supported control's missing classification rather than bypassing history
validation. Heterogeneous attention, transformer arithmetic, sampler schedules
and masks are unchanged.

Two regressions fail on released Sol main. The reviewed candidate passes 46
local targeted history/native-layout tests, including actual rebuilt Flow
closures with exact block fusion on/off, preserved dense-to-Sol transitions,
rectangular equal-grid controls and rejected ownership/digest/layout changes.
Flow's existing cross-repository source oracle now also consumes a canonical
control from actual Flow/VDN modules and requires both Sol classifiers to accept
it. Hosted CI validates the pinned integration separately from the scratch
runtime's optional dependencies. GPU forecast counts, timing and decoded
video/audio remain unqualified. Forecast use can change the generated trajectory.

### Patcher handoff

Retain Flow #89 -> #93, VDN #33 -> #34 -> #35 -> **#36**, and Continuum
#36 -> #37. Add **Sol-H3 #37** as the Sol overlay for this bounded history
correction. The Sol, VDN and Continuum PR numbers belong to different
repositories. Refresh the overlays and restart ComfyUI before measuring the
reviewed pair. Preserve these runs' equal-grid control, normal attention,
native temporal carrier and other inputs; leave residual measurement off.

Require the new VDN admission/preparation fields to establish which wrapper
executed. If a resident clone exists, `kept_resident_h3` should count it;
zero can be valid when no clone remains loaded. Core still owns patch/device/
memory-dependent reloads. For Sol, require an established identity followed by
accepted receipts and inspect actual/forecast execution counts. Forecast
eligibility is not a guaranteed forecast count or elapsed-time gain.

No new rendered quality verdict was supplied. Exact audio has nine verified
entries in each run and no extra model work, but those structural receipts do
not establish perceptual quality. 00738's measured motion is already present
in low/probe, while the previously rejected 00734 cross-grid/high-stage defect
remains a separate unresolved path. Do not combine the separate residual
measurement arm with this performance check or promote rejected geometry
actuators.

## 00734: cross-grid video rejected; hot preparation delay and clone lifetime fix

The user reports that the frame shift returns after restoring progressive
low-to-high sampling, and that substantial delay occurs before visible sampling
even on a hot run. The restored cross-grid video is rejected. Preserve 00733's
visually accepted same-grid arm; the reduced-grid speed result is not an accepted
production-quality replacement.

Evidence SHA256:

- metrics: `4b981f2a312b6d17a7be619123f50d9dad37ebba5f1d7501e9a9ccb28ca683a7`
- appended runtime log: `f9455e9eb5b6d1fb9c1ea72dda0014800293b787523c3841c30a67e98a94d323`

The appended log contains 00733 and the subsequent hot prompt. Only the final
`got prompt` segment belongs to 00734; its completion is **599.99 seconds**.

### Actual restoration and frame-shift localization

The 44x44 source -> 64x64 target plan executes: 12 exact target-prefix tokens,
50 fresh source tokens, 12,288 prefix rows and 24,200 fresh rows, or 36,488 video
rows. Normal suffix attention, native temporal carrier and normal learned-linear
remain active. Learned continuation transfer executes in 0.739 seconds.

Sol now establishes the low-stage history identity. Actual low/probe/high call
counts are 8/2/4 across both chunks, with four forecasts (two low and two high),
18 logical calls, six invocations and four history boundaries. Continuation low
has four actual evaluations and one forecast. This independently confirms that
the normal cross-grid identity condition permits an actual skipped model call.

The carried-video prefix SHA256 is exactly the same as 00733:
`32aed92ec74945a42b3ae6f7ccd59aa05f8824f159babed9f18801af04dd8fdf`.
Logged physical-conditioning and compiled-text hashes also match. The model
controls differ only by restored `progressive_low_to_high`; the executed low
geometry and forecast sequence necessarily change with it. This is stronger
carried-input evidence than the 00732/00733 comparison, but does not isolate the
individual transfer, prediction, guidance and solver mechanisms.

| Estimated first upper45 successor pair | X, target latent cells | Y, target latent cells |
|---|---:|---:|
| Source low/probe, target-equivalent | +0.08294 | +0.23328 |
| Learned provider clean | -0.04984 | +0.79176 |
| Exact-restored pre-high | -0.01734 | +0.01770 |
| Final post-high | +0.22366 | +0.86176 |

The exact-overlap bridge produces a nearly aligned first pair before high;
target-high recreates the large vertical displacement. The authoritative prefix
remains exact. Pre-high alignment and structural tests therefore do not establish
final quality. The learned provider's first pair is already displaced before
the bridge, but its low wall time excludes it as the large preparation delay.
No rejected rigid, prediction-gauge, destination-stencil or decoder-window
candidate is re-enabled.

The high guidance path uses its ordinary source trajectory, with no registered
target reference when the frame-gauge transaction is rejected. Its three
correction RMS ratios are 0.07590, 0.04978 and 0.02378; protected-prefix and
cross-prefix temporal exclusions remain active. That is a possible contributor,
not proof of causality. The saved run has no per-prediction before/after-Flow
trace, so native high prediction, forecasting, Flow direction guidance and the
solver update cannot be separated from this final-stage comparison.

### Hot delay before model execution

| Sampler timing | 00733 first (cold) | 00734 first (hot) | 00733 continuation | 00734 continuation |
|---|---:|---:|---:|---:|
| Low | 197.340 s | 114.433 s | 250.356 s | 140.226 s |
| Exact probe | 12.173 s | 11.568 s | 49.397 s | 26.312 s |
| High | 69.975 s | 48.998 s | 73.550 s | 60.898 s |
| Whole sampler | 280.506 s | 175.689 s | 376.697 s | 243.406 s |

Continuation sampler time decreases 133.291 seconds (35.38%). The first hot low
stage still spends **66.848 seconds before its first timed model call**. Its
five logical call timers total 40.510 seconds, leaving 73.923 seconds outside
those timers. Continuation spends 20.488 seconds before its first timed call,
with 109.002 seconds in call timers and 31.224 seconds outside them. Core
preparation is included in these stage wall times even though the progress bar
has not started. The first trajectory begins 98.050 seconds after the metrics
installation event; that interval is not a pure text-encoder timer.

The first hot Spectrum profile lookup is a cache hit taking 0.018892 seconds;
Sol's first low arithmetic-gate totals are about 0.075 seconds. These exclude
those cold-start explanations for the large hot preparation gap. Both physical
text groups still report `cache_hit=False`, with text-encoder load messages.
Continuum's physical-conditioning cache is intentionally local to each sequence
invocation, so a hot process does not imply reused encoded text. Text encoding
precedes sampler admission and is not individually timed in this log. Do not
replace it with an unqualified persistent cache that can reuse stale encoder,
reference or prompt state.

VDN admission reports four evicted models before first low and two before
continuation low, followed by H3 load messages. The original receipt does not
identify each evicted model or its duration, so it cannot allocate the entire
66.848-second gap. A separate source-backed defect does exist in this path.

### VDN #36: preserve resident H3 clones at admission

Continuum's `clone_model_for_chunk` creates a fresh `ModelPatcher` for each chunk.
VDN protected only `LoadedModel(incoming)` before asking Core to free all other
GPU models. Core's `LoadedModel.__eq__` compares patcher object identity, while
`ModelPatcher.is_clone` identifies a shared underlying model. A resident sibling
clone could therefore be fully unloaded before Core's native clone switch,
which normally detaches it with `unpatch_all=False` and reconciles weight patches.

[VDN #36](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/36), head
`2ff221258a0fe878bb3b0cf391ccc142f68fdda5`, keeps native Core-identified resident
clones on the sampling device at this boundary. Unrelated models remain evictable and Core receives
unchanged preparation arguments, including forced load/offload flags. Different
patch UUIDs still reach Core's normal weight reconciliation. The keep list is
call-local; CPU/non-retained paths, attention, sampler histories and buffer
release policies are unchanged.

Two regression cases using actual Core patcher/loaded-model classes fail on the
original code, for equal and differing patch UUIDs. The full local Core/OpenVDN
suite passes **252 tests**, with compileall, CI-selected Ruff and diff checks.
These prove structural lifetime/ownership behavior, not saved GPU seconds or
rendered equivalence.

New admission receipts separately report `kept_resident_h3`,
`eviction_elapsed_ms`, and delegated `prepare_elapsed_ms` with `success`.
The host intervals do not overlap and add no CUDA synchronization; original
delegate errors propagate. They can distinguish admission eviction from Core
preparation on the next hot run, while the earlier conditioning interval remains
outside their scope.

### Current handoff and remaining frame diagnosis

Apply **VDN #36 after #35** through Patcher and restart ComfyUI. Existing #33 ->
#34 -> #35 remain untouched. Flow #93's paired source-contract pin advances to
the new head; Flow runtime arithmetic is unchanged. Keep Continuum #37 after
its existing #36 overlay, normal suffix attention, normal learned-linear,
native temporal carrier, coherent exact audio, Audio Seam Auto and Video Seam
Analyze Only. The new VDN #36 and the existing Continuum #36 are distinct repos.

For usable output and preparation timing, restore the accepted
`spatial_stage_control=same_grid_target_control` with final MP 1.1 and measure
a hot run with residual measurement off. Require the native audio receipt,
both PCM endpoint checks and the new VDN preparation receipts. A resident clone
should be preserved when one is present; Core may still reload when its native
patch/device/memory policy requires it. No measured speedup for #36 exists yet.

For a separate frame-causality run, retain 00734's failing
`progressive_low_to_high` configuration and change only
**`frame_gauge_residual_mode=measure`**. The existing
`partitioned_high_boundary_prediction` diagnostics measure before and after
Flow guidance on each high call, including actual/forecast classification and
an authoritative-prefix disposable witness. They add no model/provider/VAE
calls and make no production correction. Their FFT/scalar synchronization cost
means this run is not a performance benchmark. Inspect the first native
prediction, each paired guidance result, the forecast call and the final solver
output before choosing a new video arithmetic change. Details are in
`HIGH_STAGE_BOUNDARY_DIAGNOSTICS.md`.

00734's coherent exact-audio validator still passes nine model entries, zero
effective overlap, no timestep override, exact final prefix and zero added work.
The applied PCM patch again preserves both endpoints; its jump falls from
0.003133159 to 0.000250964. Production audio boundary levels are -0.2766 dB
before and -0.0555 dB after the patch. These retain audio execution evidence;
no new rendered audio verdict was supplied with 00734.

Pre-review checkpoints preserve the preceding Flow and Continuum heads. The
VDN fix is preserved at
`checkpoint/00734-clone-admission-before-review-and-full-suite-20261001` and
`checkpoint/00734-clone-admission-252-passed-20261001`. #89 and Continuum #37
retain their clean one-commit topology. Production Sol and all rejected geometry
candidates remain unchanged. Cross-grid video remains unqualified.

## 00733: normal attention and PCM correction, cold-run speed still unresolved

The user identifies this as a **cold run**, reports that the video remains fine,
and says the audio is now okay "I think". This supplies rendered acceptance of
the video and tentative acceptance of the audio for the 1.1 MP same-grid arm
with normal suffix attention and Continuum #37. It does not qualify progressive
cross-grid sampling or establish general soundtrack acceptance.

Evidence SHA256:

- metrics: `1f5602306fb055abec04652177553ebfca6376b7901c78947ec9f81ac02d3ef4`
- runtime log: `1cfcfca93ea2e9b9e369c7f14217bfdc638c215a30c41972a35ef57d589626f0`

### Executed controls and audio result

The diagnostic controls differ from 00732 only by removal of
`softmax_diagnostic=dense_suffix_same_domain`. Source in
`apply_partitioned_diagnostic_controls` deliberately omits this field and
removes its transformer option for `normal`; the scheduler defaults to normal.
Positive Sol teardown evidence independently confirms restored sparse execution:
continuation low records **2,200 sparse calls / 800 dense warmup calls / five
actual evaluations**. The exact probe separately records zero sparse calls and
800 dense warmup calls. Its snapshot must not be mistaken for the complete low
stage. No dense-suffix intervention counter or verification event remains.

The stage plan is identical to 00732: configured source 44x44, effective source
and target 64x64, 12 prefix and 50 fresh temporal tokens, 63,488 partitioned video
rows, identity continuation transfer, normal learned-linear execution and native
temporal carrier. The stage/actual/sigma call sequence also matches: 18 logical /
15 actual / three forecasts, six sampler invocations and four history boundaries.
The coherent exact-audio evidence validator passes nine verified model entries,
zero effective overlap, no timestep override, exact final prefix and zero added
work. Low/high audio-mask digests remain equal and match 00732.

Continuum's positive `H3C-PT226` receipt records an applied
`convex_native_endpoints_v1` patch at frame 175, 1,920 fade samples (60 ms at
32 kHz), and **both native endpoint checks true**. The measured splice jump
falls from **0.004413930 to 0.000127817**, a 97.10% reduction. This establishes
execution of #37's endpoint correction on the user's hardware.

| Audio boundary measurement | 00732 | 00733 |
|---|---:|---:|
| Matched low/probe decode | +0.7471 dB | -1.1841 dB |
| Matched final-high decode | +3.1113 dB | -0.9117 dB |
| High suffix versus low/probe suffix | +2.3676 dB | +0.2700 dB |
| Production before Audio Auto | +2.7291 dB | -1.4227 dB |
| Production after Audio Auto | +2.6833 dB | -1.2003 dB |

Extra left decode context again changes the generated suffix by exactly zero;
production normalization is provably inactive. These windows do not reproduce
00732's generated level increase. PCM assembly cannot explain the earlier
matched-decode change because that audit precedes the seam patch. Cross-run
attribution to normal attention alone is also unsupported: the authoritative
carried-video prefix SHA256 changes from
`181188b45c0e73e7f8387d788109919d703b47e511500d99584fb272016fe0a6`
to `32aed92ec74945a42b3ae6f7ccd59aa05f8824f159babed9f18801af04dd8fdf`,
although both physical conditioning and compiled-text hashes match. The cause
of the different carried input is not established by these files. This is a
combined restoration/assembly run, not a tensor-matched causal audio A/B.

### Cold overhead and continuing full-grid cost

The prompt finishes in **00:14:41 (881 seconds)**, versus 00:11:54 for 00732.

| Sampler timing | 00732 first | 00733 first | 00732 continuation | 00733 continuation |
|---|---:|---:|---:|---:|
| Low | 111.596 s | 197.340 s | 260.276 s | 250.356 s |
| Exact probe | 12.174 s | 12.173 s | 50.768 s | 49.397 s |
| High | 67.592 s | 69.975 s | 76.007 s | 73.550 s |
| Whole sampler | 192.070 s | 280.506 s | 390.660 s | 376.697 s |

The first sampler increases by 88.436 seconds. Within its low stage, measured
model-call time totals 75.560 seconds, leaving 121.780 seconds outside those
call timers; 116.108 seconds precede the first timed call. The corresponding
00732 values are 56.776 / 54.820 / 49.117 seconds. The log confirms model loading,
admission eviction and a cold Spectrum profile lookup, but does not time these
components separately enough to assign the whole gap. The VDN-specific AIMDO
model compiler is explicitly disabled. Do not label all overhead as compilation
or add overlapping profile timers.

Continuation remains **376.697 seconds**, an observed 13.963-second / 3.57%
decrease. Low plus probe still costs **299.753 seconds (79.57%)**. The five actual
low calls alone total 205.067 seconds; another 45.288 seconds lie outside their
timers, including 32.900 seconds before the first call. Normal sparse attention
therefore executed, but neither the full-grid diagnostic cost nor all setup cost
is removed. These runs are not a controlled warm performance benchmark. The
first-chunk learned upscaler takes only 0.925 seconds; continuation uses identity
transfer.

There is a second, source-backed limitation of this diagnostic arm. Sol logs
`backend-history diagnostic opaque` for Flow's partitioned block wrapper.
Released Sol v0.1.6 `sol_h3/partitioned_history.py` requires
`target_rows > source_rows` in both its replacement classifier and layout
validator, so the equal-grid control cannot supply an accepted history identity.
Spectrum's low-stage summary confirms five actual calls and zero forecasts.
Two intermediate step-decision lines say "forecast", but actual execution and
the summary prove that no model call was skipped. Do not count those decisions
as speedup or weaken the history gate to force forecasting. Sol production and
Spectrum arithmetic remain unchanged.

### 00733 next arm (historical; executed and rejected in 00734)

Preserve 00733's accepted same-grid baseline. Through the existing Patcher stack,
change only the model diagnostic
**`spatial_stage_control=progressive_low_to_high`**. Retain 1.1 MP final target,
`softmax_diagnostic=normal`, `vdn_linear_diagnostic=normal`, native temporal
carrier, coherent exact audio, source-carrier audio positions, learned handoff,
witness off, Continuum #37, Audio Seam Auto and Video Seam Analyze Only. Do not
combine this arm with an attention/cache change or a decoded geometry actuator.

With the existing 44x44 configured source and the same temporal lengths, the
expected low/probe partition has 12,288 exact target-prefix rows plus 24,200
fresh source rows: **36,488 video rows rather than 63,488** (42.53% fewer). With
the same nonvideo conditioning, the sequence has 43,229 rather than 70,229 rows.
These row counts are not a FLOP or elapsed-time forecast.

Require the actual 44x44 -> 64x64 stage plan, positive learned transfer instead
of diagnostic identity, positive coherent exact-audio receipt, unchanged native
VDN/mask ownership and raw/assembled media review. Inspect Sol history and actual
forecast receipts again; genuine cross-grid geometry satisfies the strict size
condition, but complete identity acceptance is not guaranteed. Compare stage
times while recording cold/warm state. The current same-grid visual/audio verdict
cannot be extrapolated to this restored arm. The broad production runtime gate
is not applicable to 00733's identity transfer.

Flow #93 and Continuum #37 were checkpointed before this investigation at their
respective `checkpoint/00733-before-runtime-review-20261001` and
`checkpoint/00733-before-audio-receipt-review-20261001` refs. This observation
changes documentation only; it preserves #89's clean one-commit topology,
Continuum #37's one-commit fix, the rejected video candidates and production Sol.

## 00732: accepted visual boundary, residual audio and diagnostic cost

The user reports no visible boundary issue after increasing final MP to 1.1.
This accepts the rendered video for this same-grid/dense arm. It does not
qualify restored sparse suffix attention or progressive cross-grid sampling.
The user still suspects an audio defect and reports very slow execution.

Evidence SHA256:

- metrics: `cda4dcefad83d2370c76efbf8e09ebbadb6e46b3bf795431a0cc39133c080d1e`
- appended runtime log: `d1c7088424cc80f3dfd1dcf28ea4fe28d5d19122235d862257501eff6ea61441`

The log contains two prompts. Only the final `got prompt` segment belongs to
00732; its completion is **00:11:54**, not the earlier 576.06-second completion.
Actual target geometry is 64x64 latent / 1024x1024 decoded (1.048576 MP), versus
48x48 / 768x768 in 00731. Configured reduced source geometry is 44x44, but
`same_grid_target_control` runs continuation low/probe on the full 64x64 grid.
Its source tensor is `[1,24,62,64,64]`, with 12 exact prefix and 50 fresh tokens.
Continuation transfer is identity; the first chunk still uses learned transfer.

`vdn_linear_diagnostic=normal` and `native_grid_then_map_v1` remain active.
**`softmax_diagnostic=dense_suffix_same_domain` is still enabled.** Its positive
receipt reports 3,300 calls / 15,052,800 Q rows / 75,698,100 KV rows, with gathered
domain, prefix measure and grouped ownership unchanged. Compared with 00731,
aggregate Q and KV rows increase 77.8% and 67.4%. Aggregate row totals do not
determine attention FLOPs: that requires the sum of each group's Q-by-KV work,
or measured CUDA component timings. Reference/conditioning shapes also change.
Calls remain 18 logical / 15 actual /
three forecasts, with six sampler invocations and four history boundaries.

| Sampler timing | First chunk | Continuation |
|---|---:|---:|
| Low | 111.596 s | 260.276 s |
| Exact probe | 12.174 s | 50.768 s |
| High | 67.592 s | 76.007 s |
| Whole sampler | 192.070 s | 390.660 s |

Continuation low plus probe consumes 311.044 seconds, about 80% of its sampler
time. Its total rises from 208.609 seconds in 00731. First-chunk learned
upscaling takes only 0.633 seconds; continuation makes no learned-upscaler call.
Neither increased NFE nor the learned upscaler explains this slow arm. The
same-grid control intentionally forfeits low-resolution continuation savings,
and dense suffix attention intentionally forfeits sparse selection.

The existing coherent exact-audio evidence validator passes. Nine model entries
verify the authoritative native masks, effective overlap and timestep override
remain zero, and final AV prefixes are exact. Protected low/probe-to-high audio
error is max 4.768e-7 / RMS 2.853e-8. The original fractional-mask defect has
not returned.

Matched audio decode reports low/probe +0.7471 dB and final-high +3.1113 dB;
high suffix RMS increases +2.3676 dB relative to low/probe, while prefix level
changes only +0.0034 dB. Extra left decode context changes the suffix by zero,
and both production normalizers are provably inactive. Production boundary
level is +2.7291 dB before Audio Auto and +2.6833 dB afterward. These are level
measurements, not proof that the unspecified audible defect is a generation
burst. The 267-sample physical phase correction is already applied correctly.

Tracing assembly separately exposed a concrete PCM seam defect in Continuum's
released `v2/seam_guard.py`: equal-power blending amplifies correlated overlap,
global peak scaling changes both native patch endpoints, and gain/DC correction
does not return to identity before the unmodified suffix. A matching 0.01-amplitude
2 Hz waveform at 1 kHz reproduces a native step of 7.90e-7 becoming 0.002110.

[Continuum #37](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/37), head
`560e0c4e460a8a141832692281e7789852de5fdd`, fixes that local endpoint defect with
a convex overlap blend and tapered bounded gain/DC corrections. Seventeen new
regressions failed before the change and pass afterward, including actual
assembly preserving the whole continuous waveform and fresh suffix. The head
also covers all four transient fade lengths and Auto/Off assembly, with 23
waveform/assembly cases and 415 passing full CPU-suite tests. Exact-head CI run
36796253302 is green across Python 3.10–3.13 and the publisher-toolchain job.
This introduces no model/provider/VAE
calls. It does not establish a fix for the separate high-stage audio level
change or qualify the complete rendered soundtrack.

After a V3 audio patch is actually copied, `H3C-PT226 decoded-audio-seam receipt`
records `policy=convex_native_endpoints_v1 applied=True`, its boundary/fade
length, `left_endpoint_exact`, `right_endpoint_exact`, and before/after jump.
Both endpoint equality fields must be true for an applied patch. The existing
correlation gate and Audio Seam Off may produce no patch; absence of PT226 does
not establish successful correction. This observation adds no decode or model
calls.

### 00732 restoration instructions (historical; executed in 00733)

Apply Continuum #37 through Patcher after existing #36
`9ea1e75b2ddf8a3cb2a30e26c369aa63c0a32aea`, and restart ComfyUI. #37 targets
released main independently of rejected geometry candidate #36. The audio helper
was identical in both trees; the new assembly receipt touches a separate hunk in
their shared `v3/assembly.py`. The exact #36 -> #37 composition applies cleanly
and passes 59 seam, phase, video-seam and assembly-memory regression checks. Keep
Audio Seam Auto and Video Seam
Analyze Only. With retained decoded chunks, compare Audio Auto before/after the
PCM correction without resampling H3 to isolate its audible effect.

For the next sampling run, keep 00732's 1.1 MP workflow and change only the
**model diagnostic** `softmax_diagnostic=normal`. Retain same-grid control,
normal VDN linear, native temporal carrier, coherent exact audio, source-carrier
audio positions, learned handoff selector, witness off and all other inputs.
This restores the existing production attention path; no new speed algorithm is
introduced. PCM #37 is a separately recorded assembly change, so a combined run
is not a one-change audio-quality A/B. Require the positive exact-audio receipt,
no dense-suffix intervention, and raw/assembled media review.

Restore `spatial_stage_control=progressive_low_to_high` in a later separate arm
after normal suffix attention is accepted. The clean 00732 visual result cannot
be extrapolated across that restoration. Do not promote any diagnostic or
decoded geometry actuator based on this run.

GitHub pre-investigation checkpoints exist on Flow and VDN as
`checkpoint/00732-audio-and-speed-pre-investigation-20261001`. Continuum has
separate released-base, loaded-overlay and pre-full-validation checkpoints.

## 00731: normal learned-linear execution restored

The same-grid restoration run completed in 576.06 seconds with normal VDN
learned-linear execution. This establishes that the equal-grid validity guard
correction permits the intended hardware path. No rendered verdict was supplied
with this evidence; output-quality qualification remains pending.

Evidence SHA256:
- metrics: `400dffcdf05aabbcfef7907bb29b35a015342a18cfd0c2b72460a0393d1b95b8`
- runtime log: `0f622b74a4b608ddd699b42345f9e62f4184ada8ac097863c3fe913c586b7dbd`

The diagnostic controls and stage plan differ from 00729 only in
`vdn_linear_diagnostic`: `bypass_partitioned_linear` becomes `normal`. The log
contains 50 block-local active-linear markers and no diagnostic-bypass markers.
The six partitioned transformer calls publish 300 block contracts. Dense suffix
again verifies 3,300 calls / 8,467,200 Q rows / 45,230,700 KV rows with the same
gathered-domain, prefix-measure and grouped-ownership guarantees. The per-event
stage/actual/sigma model-call sequence matches 00729; aggregate counters are
18 logical calls, 15 actual calls and three forecasts.

The completed JSON now contains `partitioned_exact_audio_mask_verified`.
The existing coherent exact-audio evidence validator passes: nine model entries,
binary authoritative sampler/timestep/velocity masks, zero effective overlap,
no timestep override and no regenerated-prefix restoration. Low/high mask
digests match; final audio and video prefixes are exact. Protected audio
low/probe-to-high difference remains max 4.768e-7 / RMS 2.620e-8.

The measured loud boundary burst does not return: matched low/probe decode is
-15.5700 dB and final-high is -13.9562 dB; production pre-seam is -14.2845 dB.
Extra left decode context changes the suffix by exactly zero, and both
production normalizers remain provably inactive. These values describe a quieter
generated boundary window, not constant loudness or acceptance of the complete
soundtrack.

Upper45 latent successor-pair estimates remain small in the low/probe result and
survive high refinement. Source-low X is (0.175719, 0.070158, -0.084077, -0.017367)
and Y is (0.169830, 0.107721, 0.070430, 0.001224); final-high X is
(0.170640, 0.071526, -0.133896, -0.007813) and Y is
(0.194475, 0.120680, 0.052957, 0.011766) target latent cells. The decoded first
upper45 pair still measures approximately (-3.9904, +3.6290) pixels. These
estimates do not establish elimination of the visible chunk-to-chunk change.

This is not a tensor-matched causal A/B despite the matching recorded controls.
The authoritative carried-video prefix SHA256 changes from
`9f76b8337b500a0eaa26b4e19dbfaf42e4199c2a5b55207b50264a71a2552f18` to
`0aa7e745f9869047ee236d948fa4fe798308e12d17ee964837b80a454eac1178`.
Production preceding-audio RMS changes from 0.004993902 to 0.008718249, while
generated RMS changes from 0.001798426 to 0.001683459. Prompt-routing digests,
logged Core/Torch/Kitchen/Aimdo versions and the continuation noise seed match;
the reason the carried input changed is not established by these files. Neither
the apparent motion change nor the cross-run dB difference can be attributed
solely to restoring learned-linear execution.

The broad production runtime gate remains inapplicable to this same-grid arm:
it requires a learned cross-grid transfer that intentionally does not execute
under the identity control. Its frame-gauge qualification also requires resolved
DoRA auto-strength reports absent from this evidence. The positive exact-audio
validator and explicit diagnostic receipts pass independently; do not label this
run a complete production-gate pass.

First obtain the rendered verdict for raw/assembled video and audio. If this
normal-linear arm is acceptable, retain 00731's inputs and change only
`softmax_diagnostic=normal` for the next sparse-suffix restoration arm. Keep
same-grid target control, `vdn_linear_diagnostic=normal`, native temporal carrier,
coherent exact audio, source-carrier audio positions and witness off. Restore
cross-grid learned transfer separately after sparse attention is accepted.
No runtime arithmetic change or spatial correction is justified by these files.

## Learned-linear restoration: equal-grid guard correction

The subsequent restoration attempt aborted after 406.87 seconds when the first
partitioned learned-linear call reached `partitioned_frame_contract()`. Runtime
log SHA256: `aacc3095f851532319844b0b7f5ab323cefa55d4a4bf5bab9dc21ee130f64c5d`.
This failed run provides no rendered result for restored learned-linear execution.

Flow's same-grid control and VDN's sequence parser already permit equal source
and target grids, but the linear frame helper still required strictly fewer
source rows. The prior bypass arm did not execute this helper. VDN #35 now
accepts `0 < source_rows <= target_rows`, retaining the generated-suffix and
source-exceeds-target rejection. Equal grids produce unit physical measure;
the existing readout uses native arithmetic without interpolation. This fixes
an inconsistent validity guard and introduces no new numerical policy or calls.

All twelve square/rectangular, prefix-length and anchor-trimming regression
cases failed at the original guard and pass after its correction, matching
released VDN readout within the existing FP32 oracle tolerance. Invalid geometry
remains rejected. The complete local pinned-Comfy/official-oracle suite passes
248 tests. Flow's paired source-contract CI pin uses corrected VDN head
`e253432b9f79db91a0d99461bbd85fb1574a39b3`.

Refresh VDN #35 and Flow #93 through Patcher and restart ComfyUI. Repeat the
failed restoration arm: `vdn_linear_diagnostic=normal`, retaining all other
00729 settings. Require successful learned-linear execution, the persisted
positive exact-audio receipt, unchanged dense-suffix ownership receipts and
complete raw/assembled media acceptance. The slight chunk-to-chunk appearance
change remains unresolved; this guard correction supplies no new media evidence.

## 00729: audio improvement in the frozen diagnostic arm

User inspection reports that the audio issue appears gone and the frame shift is
pretty much gone, with a slight change between chunks still visible. This is
qualified evidence for this same-grid, learned-linear-bypass, dense-suffix arm;
it does not qualify restored production attention or cross-grid learned transfer.

Evidence SHA256:
- metrics: `496c1c2df344e4aee3276580381e31e1e065429bf8a55a7f473a820e7821165b`
- runtime log: `53de2985bdac0c090a2bba917c873b999a410e51be6369a0d426d1d24c27d3aa`

The diagnostic control fields, same-grid geometry and per-event logical/actual
model-call sequence match 00728. The complete learned-linear bypass again records
300 calls / 10,713,600 video rows; dense suffix again records 3,300 calls /
8,467,200 Q rows / 45,230,700 KV rows. The authoritative low/high audio masks now
match, nine native-mask verifications execute, and inner timestep overrides are
zero. The stored legacy selector remains unchanged with 16 requested ticks; its
sampler ramp is inactive. The low/probe-versus-high protected-audio difference
falls from max 11.1940 / RMS 0.874204 to max 4.768e-7 / RMS 2.416e-8.

The matched low/probe decode boundary changes from +6.5377 dB to -9.6175 dB;
final-high changes from +7.1323 dB to -8.6536 dB. Production PT213 changes from
+7.0079 dB to -8.8709 dB before seam processing. The preceding production RMS is
identical at 0.004993902; generated RMS falls from 0.011190105 to 0.001798426.
This supports removal of the loud burst, rather than a downstream normalizer or
seam repair. It does not establish constant audio loudness across the boundary.

Video phase estimates do not show a corresponding disappearance of displacement.
Upper45 final-high X pairs change from (-0.108105, -0.390088, -0.128388, -0.018460)
to (-0.145154, -0.429790, -0.093538, -0.017967) cells; decoded first-pair motion
remains approximately (-6.31, +4.16) pixels. These estimates are not a reliable
proxy for the user's rendered assessment. No further spatial actuator is
authorized by these measurements. The remaining chunk-to-chunk appearance change
and general video causality are unresolved.

An export-order defect was also found: `sampler_wall` triggers autosave before
the outer audio verification events are emitted, so the saved 00729 JSON omits
the final exact-audio receipt even though its nine verification calls and the
corrected runtime logs are present. Flow now persists the completed invocation
after those events. This changes evidence export only and adds no sampling,
provider or VAE work. The saved-file regression covers the actual autosave order.
The original 00729 evidence remains intact; do not invent its missing receipt.

The next production-restoration discriminator should keep 00729 frozen and
change only `vdn_linear_diagnostic=normal`, retaining same-grid geometry, dense
suffix and coherent exact audio. Qualify complete raw/assembled media and the
positive final mask receipt before separately restoring sparse suffix selection
or cross-grid learned transfer. No unchanged diagnostic rerun is needed merely
to reconfirm the already observed audio improvement.

## 00728: coherent exact-audio correction

Evidence SHA256:
- metrics: `36b43d3bdaa418aacec9a1c75b5e443fd6c1e1594eb51d26bbe42f33a9a2220d`
- runtime log: `3a59694fdd7a75cc4ad6ff4e2fd8b6766296703ebf9b15acd416721f8bbbf08d`

Dense suffix executed 3,300 calls / 8,467,200 Q rows / 45,230,700 KV rows with
unchanged domain, prefix measure and grouped ownership. Learned-linear bypass
executed 300 calls / 10,713,600 video rows. Upper45 source-low X successor pairs
were (-0.124866, -0.344425, -0.166585, -0.030154) cells; final-high was
(-0.108105, -0.390088, -0.128388, -0.018460). Identity transfer again added zero
DC correction. Raw decoded PT212 still reports an initial upper45 displacement
of (-6.3550, +4.4346) pixels. Sparse suffix selection is not a sufficient cause.

With shared authoritative prefix and matched decode context, low/probe audio
already has +6.5377 dB boundary RMS; final-high has +7.1323 dB. Extra left context
changes the decoded suffix by exactly zero. The production/common decode
difference is approximately 0.0771%, and both production normalizers are inactive.
PT213 measures +7.0079 dB before seam processing and +7.0059 dB afterward.
The burst originates before high refinement; left decoder context, normalization
and seam processing are not sufficient explanations.

Source tracing found a definite conditioning mismatch in the selected
`sampler_mask_exact_timestep` mode: 16 protected audio ticks were released into
fractional sampler ownership, while inner H3 labels described them as exact.
Core's input is `m*x + (1-m)*clean`, and its outer audio velocity conversion uses
m. Labeling that partially regenerated context clean breaks the native input /
timestep / velocity contract. Final restoration then substitutes the original
prefix for the regenerated context. This source defect is proven; its rendered
contribution remains a hardware hypothesis.

The correction keeps the authoritative mask at every low/probe/high sampler
entry. Core uses matching native input, timestep and velocity masks. `exact_mask`
is the explicit selector; stored `sampler_mask_exact_timestep` values alias it.
Requested overlap width remains provenance; effective width is zero. Each inner
model entry verifies equality with the native mask and passes kwargs unchanged.
A mismatch fails before the inner forward. Invocation context is cleaned up on
success, execution failure and preflight rejection, and nested contexts fail
without replacing the existing owner. Exact masks must be binary and uniform
across channels, stereo lanes and batch items. Core may repeat the native mask
for CFG batching or omit it for fully generated audio. The Core #15988 outer
velocity-mask compatibility requirement remains enforced. Released Target Input and the explicit `sampler_mask` /
`model_timestep_only` comparison modes retain their behavior.

Refresh Flow #93 through the existing Patcher overlay stack and restart ComfyUI.
Keep the entire 00728 workflow frozen, including its stored legacy selector and
16 requested ticks. Keep same-grid control, linear bypass, dense suffix, native
temporal-carrier policy, source-carrier audio positions, prompts, seeds, LoRAs,
the audio audit and all other overlays unchanged. Selecting `exact_mask` is
optional because the legacy selector invokes the same correction.

Require `partitioned_exact_audio_mask_verified` with positive `verified_model_entries`,
`policy=coherent_exact_audio_mask_v1`, `effective_overlap_ticks=0`, exact input /
timestep / velocity flags true, `timestep_override_applied=false`,
`regenerated_prefix_restored=false`, `final_prefix_exact=true`, `fail_closed=true`,
and every extra-work count zero. Require dense-suffix and linear-bypass receipts
again. `audio_guided_overlap.applied=false` is expected; the new exact-mask receipt
replaces an applied-ramp requirement. The old mapped-sparse production gate does
not qualify this all-dense diagnostic arm.

Compare the matched low/probe audio decode, final PT213 RMS/spectrum, complete raw
decoded video boundary window and background continuity. Joint audio context may
couple into video, but the frame-shift root is unresolved. Exact-main semantics,
attention domains, VDN/Sol arithmetic and the three continuation sampler lifetimes
/ two history boundaries remain unchanged. No model/provider/VAE call is added.
Flow #89 and VDN #33/#34/#35 are not consolidated or promoted from source tests.

The authoritative specification is
[the design at b97ed34d0db253f26e5c7a391c3fb1358c3c3684](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/b97ed34d0db253f26e5c7a391c3fb1358c3c3684/docs/HETEROGENEOUS_EXACT_PREFIX_BOUNDARY_IMPLEMENTATION_DESIGN.md).
Its hardware conditional remains in force: a mathematical ordering difference
does not establish that cross-grid taps cause the rendered artifact.

## Refreshed source baseline

| Component | Ref |
|---|---|
| Flow main | `a6249b8343becc1458a4555d2bb25cd523983c90` |
| Flow #89 | `fa8d65b84e0666ce467b838544d92c801394a002` |
| VDN Plus main | `b78e94d0365ffc5059924a048af707e565d0380e` |
| VDN Plus #33 | `da3627f85d494bdf4213251deb3ba2a94b8a2f36` |
| VDN Plus #34 observer | `cae13fb5e8d71b93ee3134b629d23fec7c819c5b` |
| VDN Plus #35 diagnostic head (candidate-C retained; equal-grid + dense-suffix discriminator) | `e253432b9f79db91a0d99461bbd85fb1574a39b3` |
| Sol main / v0.1.6 | `bef9b300275a89290ca2d53eafff79af06f5e0ef` |
| Sol source-contract pin | `93b3e03f2b7b579aaf55fa0f87f55083b259e25c` |
| Continuum Plus main | `e870875b1a29968d39d72ba304e9f2b05544c015` |
| Rejected Continuum #35 | `28624d2577a62b6b48a04f3897e7b03a12697eb7` |
| Rejected Continuum #36, actuator disabled | `9ea1e75b2ddf8a3cb2a30e26c369aa63c0a32aea` |
| Learned upscaler | `620165a311de9b28a36260219fb5cd370a304e3c` |
| Core current main | `9d80841aa1990305cc7a280c5ea505f317efcc2d` |
| Flow Core oracle / native-mask oracle | `1af040bf022569d7a890241c8dd79b296cda483f` / `421a1c245c682c04d4325ba365f40c834c66f5b0` |
| VDN Core oracle | `6c53f8c9a06d95f3d847009ceaae55c624169247` |
| OpenVDN oracle | `b8cb28fbfca0266d1c7742a9f25ab8b58191de97` |

All requested branches/pins were fetched. Flow/VDN review threads were resolved
at retrieval. Sol's source-contract pin differs from main only in release
metadata. These are fetched source refs, not proof of installed Patcher trees.
Earlier installed evidence abbreviated Core `34b50ec9` on `patcher/stack`.
The 00726 hardware log instead reports ComfyUI `v0.38.0-12-g0d48b6032` on
`patcher/stack` with Torch 2.10.0+cu130. This runtime change is a comparison
confound relative to 00724 and must not be silently attributed to the same-grid
selector.

Both repositories have remote `checkpoint/heterogeneous-observation-start-20260930`
refs at their protected PR heads. Development uses separate
`mirror/heterogeneous-observation-20260930` branches. Pre-review checkpoints
preserve the first observation implementation. Sol and Continuum source branches
remain untouched.

## Original evidence recovered

The original bytes were recovered using the design's materialized paths and
verified against its hashes:

| Original file | SHA256 |
|---|---|
| `metrics_00717_.json` | `8abe3148210934d0765b36064108cccfd73e1af08bd313628d30c602ec575add` |
| `Pasted text(20260929-203040).txt` | `11ac6cfedde9c97ddda976f16fdd8fb46b8cd7c8edc7c59fea125679961c6e52` |

00717 uses prefix12, temporal62, target latent48×48 and source34×34. Its
upper45 pre-high pair0 is approximately (−0.021,+0.010) target cell, while
pair1 is (−3.898,−3.809). This window is not clean. Whole-workflow counters for
two physical groups are 14 actual H3 evaluations (low8/probe2/high4),
18 logical calls, four forecasts and six sampler/four history boundaries.
One continuation remains three sampler lifetimes/two history boundaries.
The original MP4, original complete workflow JSON, full overlay graph and
checkpoint hashes were not available to this implementation environment.

## Implementation and explicit deviations

1. Milestone one remains the bounded observation ABI. Flow owns a default-off
   CPU sink; VDN #34 exposes actual raw/filtered/mapped/activated features and
   A/B/alpha norms. Observation itself is output-neutral.
2. 00719 established the actual-checkpoint operator mismatch. 00721 then ran the
   prescribed `suppress_cross_grid_temporal_taps` B arm on SM120 and materially
   changed the inherited pre-high boundary. That satisfies the design's gate to
   implement candidate C, but not its production/media acceptance gate.
3. VDN candidate C is isolated in **VDN #35**, stacked on observer #34, which is
   stacked on #33. #33 and #34 remain intact. Flow #93 carries only the paired
   selector/contract/capability/receipt logic required to invoke and verify #35.
   Flow #89 remains untouched at its one-clean-commit production-candidate
   baseline.
4. Candidate C adds independently versioned numerical leaf
   `h3_flow_partitioned_vdn_temporal_carrier_v1` API 1. Absence remains
   `native_grid_then_map_v1`; the opt-in arm is
   `destination_grid_stencil_v1`. The numerical digest binds the Flow semantic
   plan digest, diagnostic mode, actual checkpoint short-conv specification and
   physical mapping/precision policy.
5. For cross-grid temporal taps only, VDN #35 maps the **raw projected** feature
   to the receiving H3 lattice with the existing FP32 bilinear/border mapper,
   restores feature dtype, applies the trained depthwise 5x5 spatial stencil on
   that destination grid, then preserves the existing temporal weight, SiLU,
   Q/K L2, A/B/alpha, recurrence and readout path. Same-grid arithmetic remains
   the existing native batched path. No authoritative residual rows, RoPE,
   sampler latents, output geometry or decoded pixels are resampled.
6. The candidate is fail-closed: it requires paired VDN carrier API1,
   `vdn_linear_diagnostic=normal`, exact partitioned transformer context and a
   compatible checkpoint short-conv declaration/weight geometry. Native/default
   execution does not depend on candidate capability and retains the previous
   provider-cache identity.
7. The current-Core native mixed-grid test fixture accepts Core's `attention`
   keyword only for compatibility testing; production model code is unchanged.
8. Observation remains bounded to two complete heads in the first actual
   low-stage convolution block, the first mixed boundary and temporal radius ≤4.
   These bounds affect evidence completeness, not default arithmetic.

Tests cover observation neutrality, dtype/branch/anchor coverage, CPU lifetime,
budget enforcement, capability rejection, stage owner cleanup, option cloning
and the paired source-contract ABI. A missing selected low-stage observation
cannot silently become a successful diagnostic sample.
Final review corrected receipt metadata so bounds and physical measure scales
use the same optionally anchor-trimmed frame domain as captured features. The
untrimmed runtime scales are recorded separately; production weighting is
unchanged. The four suppression/anchor combinations verify this metadata.

## Supplied paper check

The supplied DMD2, Sol-Attn, Sol-engine, VSA, Spectrum and Video DeltaNet PDFs
were inspected. Video DeltaNet section 2.3 confirms the released K/V depthwise
5×5 spatial and five-tap temporal filters, SiLU and Q/K L2, with separately
calibrated branches. Appendix A.5 makes checkpoint architecture metadata part
of the release contract. Sol-Attn routing and Spectrum feature history remain
separate mechanisms. These papers do not by themselves establish rendered causality or overturn the
rejected output repairs. The 00721 hardware intervention, rather than the paper
review, is what authorized implementation of candidate C. The design's
whole-window SM120/media acceptance gate remains unchanged.

## Patcher dependency order

The implementation PR descriptions record exact final diagnostic SHAs. Use
those SHAs and the following dependency order, refreshing all overlays before
starting ComfyUI:

1. VDN Plus main, #33, [VDN #34](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/34)
   at `cae13fb5e8d71b93ee3134b629d23fec7c819c5b`, then
   [VDN #35](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/35) at
   `96b4e49e507d223f7900e3cb35a9ae0a207ec213`.
2. Flow main, existing #89, then [Flow #93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93), its stacked qualification/candidate-control draft.
3. Sol v0.1.6/main at the ref above; no diagnostic Sol overlay.
4. Continuum Plus main, then #36 solely for its retained diagnostics and disabled
   rigid actuator. Do not add #35. Verify loaded #36 source at the ref above.
5. Learned upscaler at the ref above, and the remaining already-selected
   Spectrum/DiffAid/RefDelta/Untwist/LoRA/DoRA providers at frozen installed refs.

The diagnostic draft bases are the corresponding existing PR branches, not
main. Applying only a stacked draft's delta without its parent is invalid.
Record Patcher's effective tree and ordered overlay graph after refresh; the
public PR sequence alone does not prove which code is loaded. Do not update
Core or any unrelated provider between comparison arms.

## Matched workflow and observation

Recover the original workflow and reference inputs before claiming a matched
00717 comparison. The metrics/log do not reconstruct every prompt, seed,
adapter strength and input. Freeze that workflow and its effective checkpoint
and source manifest. Disable decoded geometry mutation; with #36 use
`Video Seam = Analyze Only` to preserve raw video while keeping diagnostics.
Keep the original audio policy in both arms and record any indirect audio
output differences independently.

Use node type `H3PartitionedExactPrefixDiagnosticHandoff`, displayed as
**MiniMax H3 Partitioned Exact-Prefix Handoff**, with its diagnostic selectors. Verified
00717 selectors are:

| Selector | Both arms |
|---|---|
| `prefix_transformer_context` | `exact_target_partitioned` |
| `audio_position_domain` | `source_carrier` |
| `audio_guided_overlap_mode` | `sampler_mask_exact_timestep` |
| `audio_guided_overlap_ticks` | `4` |
| `provider_boundary_stabilization` | `soft_support_v1` |
| target/source latent grid | `48×48 / 34×34` |
| exact prefix / T | `12 / 62` |

Keep all remaining controls, the learned full-sequence provider, existing
one-token bridge, Sol tau/routing, Spectrum, seeds/noise and LoRA/DoRA settings
unchanged. Do not select a source-uniform shadow path. Run:

* A: `vdn_linear_diagnostic=normal`.
* B: `vdn_linear_diagnostic=suppress_cross_grid_temporal_taps`.

For each observed arm, set the node's `capture_boundary_witness=true`.
This is a per-run control and does **not** require an environment variable or a
ComfyUI restart. Artifacts use unique filenames under
`ComfyUI/output/h3-flow-boundary-witness`; the receipt records the active
`vdn_linear_diagnostic` mode, so A and B remain attributable without separate
process launches. Set it independently for:

* A: `vdn_linear_diagnostic=normal`, `capture_boundary_witness=true`.
* B: `vdn_linear_diagnostic=suppress_cross_grid_temporal_taps`,
  `capture_boundary_witness=true`.

The historical `H3_FLOW_BOUNDARY_WITNESS_DIR` process variable remains only as
a headless/backward-compatible fallback when no explicit node selection is
present. An explicit node OFF overrides a stale environment value.

Witness capture is a default-off observer, not a stencil selection or a
dense-attention switch. It creates one paired `.pt`/`.json` artifact for each
observed continuation low lifetime and publishes `partitioned_boundary_witness`
in Flow metrics. The `.json` contains loaded module paths/hashes, actual
precision settings, frame/head indices and tensor SHA256. Requested observation
requires the paired VDN draft before sampling.

Verify actual suppression counters and
`partitioned_vdn_linear_diagnostic_verified`; a widget label is insufficient.

### Completed A/B hardware evidence

00719 is the witness-enabled A/normal arm. Its witness is actual low-stage block
0, heads 0/55, boundary inner frame 11, and contains the cross-grid K/V temporal
taps. Offline replay on that captured input shows a material
native-filter-then-map versus map-then-destination-filter difference, while
00718 and 00719 trajectory receipts are identical, establishing witness
neutrality.

00721 is the B arm. It verified 250 suppression calls, 3000 cross-grid taps and
1,297,500 destination rows with the same 14 actual H3 evaluations and existing
sampler/history topology. The upper45 exact-restored pre-high successor pair1
changed from A `(-3.89833,-3.80939)` target cells to B
`(-0.94051,-0.32042)`, about an 81.8% reduction in vector magnitude. The
cross-grid short-conv path is therefore causally material to the inherited
pre-high discontinuity.

B is not a fix. Its final upper45 window remains displaced and its aggregate
upper45 net displacement is worse than A. No rendered A/B movies were supplied
with 00721, so background-retention/media acceptance cannot be inferred from
metrics alone.

### Candidate C result: 00722 rejected by rendered-media gate

00722 executed candidate C exactly as intended on SM120:

* `vdn_linear_diagnostic=normal`
* `vdn_temporal_carrier_policy=destination_grid_stencil_v1`
* 250 verified destination-stencil calls
* 3,000 cross-grid taps
* 2,000 mapped carriers / 865,000 mapped carrier rows
* stable numerical digest
  `82fd2c133290aece625eb432cee2764be70d73a7f0e68b17bc5b3306f7025547`
  across low/probe and final verification
* unchanged 14 actual H3 NFE / 18 logical calls / six sampler invocations /
  four history boundaries.

The actual low-stage witness records block 0, heads 0/55, K/V short conv,
destination-grid policy and the same numerical digest with zero extra
H3/provider/VAE calls.

C materially improves the internal inherited pre-high transition:
upper45 successor pair1 changes from A/native
`(-3.89833,-3.80939)` to B/suppression
`(-0.94051,-0.32042)` and C
`(-0.70003,-0.07865)`. C therefore reproduces the causal B-arm effect without
deleting the learned cross-grid contribution.

That internal improvement is **not sufficient**. User inspection of the actual
00722 rendered output reports that the same frame shift is still visible. This
fails the design's explicit whole-window rendered-media criterion. The
destination-stencil path must not be tuned further or promoted from these
metrics. Preserve VDN #35 and the 00722 artifacts as rejected diagnostic
evidence only.

### 00724 result: complete learned-linear bypass also fails rendered acceptance

00724 is a valid bypass arm:

* `vdn_linear_diagnostic=bypass_partitioned_linear`
* `vdn_temporal_carrier_policy=native_grid_then_map_v1`
* 250 fail-closed verified bypass calls / 5,340,500 video rows
* unchanged 14 actual H3 NFE / 18 logical calls / six sampler invocations /
  four history boundaries
* source-carrier audio-position execution verified and the run completed.

The bypass strongly changes internal geometry but does not remove the rendered
defect. Upper45 `exact_restored_pre_high` successor pair1 becomes
`(+2.97523,-0.24417)`, i.e. it reverses direction relative to A/native rather
than simply shrinking. After target-high the upper45 net is
`(-0.37610,+0.28953)`. Despite that numerical change, user inspection reports
that the same visible frame shift remains. This rejects the complete partitioned
VDN learned-linear complement as a sufficient cause of the rendered artifact.

The more important localization is earlier in the handoff. In 00724 the
source-low upper45 successor pair1 is only about
`(-0.12913,+0.01193)` target-equivalent cells. Immediately after the learned
34x34 -> 48x48 clean-video transfer it is about
`(+3.08880,-0.35257)`, roughly 24x larger in vector magnitude. Exact-prefix
restoration then changes pair0 as intended but leaves the successor jump at
`(+2.97523,-0.24417)`. Similar source->learned amplification occurs in the
other matched arms despite their different VDN interventions. This makes the
low->high transfer the leading current suspect.

This is not yet proof that the learned checkpoint itself is the sole cause:
the remaining alternatives are (a) the learned 3D upscaler introduces the
spatial/temporal gauge change, or (b) the broader low->high resolution
handoff plus target-high response does so even with a simple spatial transfer.

### 00725 first same-grid attempt: aborted by stale strict-smaller contract guards

The first hardware attempt at the new same-grid control did **not** test the
hypothesis. The control reached continuation low-stage setup, but Flow's
`PartitionedExactPrefixPlan` still rejected equal source/target grids before
the first continuation transformer evaluation with:

`partitioned exact-prefix plan requires a strictly smaller source grid`.

That exposed an incomplete implementation: the scheduler/stage layer allowed
the explicit same-grid diagnostic, but the published partition contract and the
paired VDN parser still encoded the old heterogeneous-only invariant.

The fix is now coordinated across both repositories:

* Flow #93 authorizes equal source/target partition geometry only when the
  explicit same-grid control is active, publishes zero prefix measure bias and
  `heterogeneous_spatial_domains=false`, and routes the same-grid case through
  the full handoff plumbing instead of returning early on equal geometry.
* VDN #35 accepts equal-grid partition contracts, preserves the ordinary
  strict/non-expanding geometry checks, and canonicalizes the equal-grid
  contract with zero measure bias and
  `heterogeneous_spatial_domains=false`.

The failed attempt produced no rendered discriminator result and must not be
counted as evidence for or against the resolution-transition hypothesis.

### 00726 same-grid hardware result and clean-state ownership defect

00726 is the first successful execution of the direct same-grid control. Runtime
receipts prove that configured progressive source intent remained 34x34 while
continuation low/probe actually ran at 48x48, target-high remained 48x48, the
handoff operator was `diagnostic:same_grid_target_identity`, it executed once,
and the actual learned-checkpoint provider was not called. The complete
partitioned VDN learned-linear bypass also remained active.

Rendered-media inspection remains authoritative: the user reports that a
**slight frame shift still happens, but it is much less prominent**. Removing
the low->high spatial transition therefore materially changes the visible
defect, but does not eliminate it.

The pair is not perfectly controlled. 00726 recorded 15 actual H3 NFE
(low9/probe2/high4) and three Spectrum forecasts, whereas 00724 recorded 14
actual H3 NFE (low8/probe2/high4) and four forecasts. 00726 also ran a newer
ComfyUI/Patcher Core. Those differences prevent assigning the entire visible
improvement to resolution alone from this single comparison.

00726 additionally exposed a concrete Flow ownership bug. The handoff noise
policy was `source_residual_patch_refinement_v1`; at equal 48x48 geometry that
residual transport itself was identity, and frame-gauge registration returned
`identity/already_aligned`. Despite that, downstream splice logic reported
`splice_clean_source=inverse_recovered` and
`splice_recovery=inverse_conditional_renoise`: it reconstructed clean video
with deterministic Gaussian noise even though the conditional state had been
built from the carried source residual.

At the 00726 split sigma (~0.8780488), that incorrect inverse strongly amplifies
the residual-vs-Gaussian difference. The synthetic clean tensor then drove a
production one-token DC bridge: one suffix token was changed with delta RMS
~0.231223 and max absolute delta ~0.455920 even though the same-grid provider
prefix was already aligned with the authoritative prefix. Consequently the
00726 `learned_native` / `exact_restored_pre_high` diagnostics are
contaminated by this reconstruction error and are not reliable evidence of a
real identity-handoff spatial jump.

Flow #93 now fixes ownership at the source:

* the clean postprocess hook retains the exact clean tensor that actually feeds
  conditional re-noising;
* downstream splice/bridge logic uses that tensor directly;
* source-residual mode fails closed if that tensor is missing instead of
  falling back to Gaussian inverse recovery;
* deterministic inverse recovery remains available only for the independent
  deterministic-noise contract;
* regression tests prove that an identity same-grid handoff cannot manufacture
  a suffix DC correction from this mismatch.

The clean-ownership correction first reached CI-green head
`53e99780c3379e2d7606c2cf0de2b0b6eb1252dc`; the subsequent documentation/
provenance head `7366e5acf654490b16e2a443584700c8f06668fb` also passed exact-head CI
run `36741231859` across source-contracts and Python 3.10/3.11/3.12/3.13.
Checkpoints preserve both the pre-fix 00726 state and the validated correction.

### 00727 same-grid rerun: ownership fix validated, residual shift remains

00727 repeated the same-grid configuration on the same reported Core runtime as
00726. Its handoff receipts validate the ownership correction exactly:

* `same_grid_identity_transfer_applied=true`;
* `splice_clean_source=actual_clean_postprocess`;
* `splice_recovery=actual_clean_postprocess_no_inverse`;
* `suffix_dc_bridge_delta_rms=0.0` and max/mean absolute delta both zero;
* seam amplification ratios are exactly 1.0 at the identity handoff;
* VDN learned-linear remained fail-closed bypassed for 300 calls /
  10,713,600 video rows.

The user nevertheless reports a **very slight frame shift** in the rendered
00727 movie. The upper45 source-low pairwise trajectory is unchanged from 00726:
dx `[-0.12936,-0.28895,-0.14254,-0.01740]`, dy
`[-0.02179,-0.03148,-0.00128,-0.00273]`. After target-high it remains close:
dx `[-0.09521,-0.27566,-0.11301,-0.00237]`, dy
`[-0.04498,-0.00125,-0.04743,-0.01593]`.

This is the key localization. Under equal 48x48 low/high grids, identity
handoff, zero learned-upscaler calls, correct clean ownership, zero DC mutation,
identity frame gauge and complete VDN learned-linear bypass, the small boundary
translation already exists in low/probe and target-high mostly preserves it.
The large historical defect is strongly associated with the spatial transition;
the remaining defect is upstream inside the exact-partitioned low/probe
transformer path.

### Next discriminator: same-domain weighted-dense suffix attention

The authoritative design's deferred sparse-selection discriminator is now
authorized. Flow #93 and VDN #35 add an opt-in leaf
`h3_flow_partitioned_softmax_diagnostic_v1` API 1 with
`dense_suffix_same_domain`.

This arm does **not** change grouped ownership. VDN constructs the same
partitioned grouped plan, gathers the same Q/K/V rows, transports the same
query-position wire, sink rows, target-prefix K range and prefix log-measure,
then changes only the existing Sol request's `force_dense` flag for
generated-suffix local-query groups. Prefix-query groups remain on their
existing dense path; global/anchor behavior is unchanged. The VDN learned-linear
selector remains independent.

The diagnostic publishes positive counters for calls/Q rows/KV rows and Flow
fails closed unless nonzero work is observed. Flow also adds the diagnostic leaf
to provider-cache and partitioned layout/history identity only while selected,
so a dense diagnostic sample cannot reuse sparse numerical history. Normal/
absence semantics remain unchanged.

For the next hardware arm, keep the complete 00727 workflow frozen and change
only:

* `softmax_diagnostic=dense_suffix_same_domain`

Keep in particular:

* `spatial_stage_control=same_grid_target_control`
* `handoff_transfer_control=learned_3d`
* `vdn_linear_diagnostic=bypass_partitioned_linear`
* `vdn_temporal_carrier_policy=native_grid_then_map_v1`
* `capture_boundary_witness=false`
* `audio_position_domain=source_carrier`

The run must emit `partitioned_softmax_diagnostic_verified` with positive
`dense_suffix_calls`, `dense_suffix_q_rows` and `dense_suffix_kv_rows`,
plus `same_gathered_domain=true`, `prefix_measure_unchanged=true`,
`grouped_ownership_unchanged=true` and `fail_closed=true`. The VDN bypass
receipt and 00727 identity-clean receipts must remain positive/zero respectively.

If the residual rendered shift materially changes, sparse Sol selection is
causally material and routing/history becomes the next investigation target.
If it remains with the same character, sparse selection is not a sufficient
cause and the design advances to the provider-transfer discriminator. Internal
motion metrics remain diagnostic; rendered whole-window inspection is the gate.

### Revised next discriminator: remove the spatial grid transition itself

The previous learned-vs-bicubic proposal is **not** the required test for the
current hypothesis. The user is specifically testing whether Flow Regenerate's
low-resolution -> high-resolution stage transition is what creates the frame
shift. Replacing one 34x34 -> 48x48 resize operator with another still retains
that transition and therefore cannot answer the question directly.

Flow #93 now exposes an appended diagnostic selector
`spatial_stage_control`:

* `progressive_low_to_high` — historical/default behavior. The configured
  reduced source geometry remains active.
* `same_grid_target_control` — ignores the reduced source geometry for actual
  low/probe execution and runs low/probe directly on the caller target grid.
  The original configured reduced source geometry is retained only for handoff
  split selection, so fixed and auto-computed handoff timing remain matched.
  At the handoff, an identity provider is used through the same handoff plumbing:
  no spatial resize and no learned-checkpoint invocation occur. Residual/noise
  transport, clean postprocess, exact-prefix restoration, audio state, guidance,
  Spectrum/history ownership and target-high sampling remain in place.

The control is fail-closed. It requires
`handoff_transfer_control=learned_3d` and
`vdn_temporal_carrier_policy=native_grid_then_map_v1`; combining it with the
bicubic transfer diagnostic or destination-grid stencil is rejected.

For the next matched hardware run keep the complete **00724** workflow and
every selector frozen and change only:

* `spatial_stage_control=same_grid_target_control`

In particular keep:

* `handoff_transfer_control=learned_3d`
* `vdn_linear_diagnostic=bypass_partitioned_linear`
* `vdn_temporal_carrier_policy=native_grid_then_map_v1`
* `capture_boundary_witness=false`
* the existing source scale/width/height values unchanged.

Do **not** manually set source scale to 1.0; the ordinary configuration rejects
that by design. The diagnostic selector overrides only the effective low/probe
spatial stage internally while preserving the configured source geometry as
provenance and for handoff split selection.

The run must emit `partitioned_stage_plan` with
`same_grid_control_active=true`, equal effective source/target H/W, and the
original `configured_progressive_source_hw`. It must also emit
`partitioned_handoff_transfer_control` with operator
`diagnostic:same_grid_target_identity`, exactly one spatial-control call and
zero actual learned-checkpoint provider calls.

Interpretation is direct:

* if the visible frame shift disappears/materially collapses, the
  low-resolution -> high-resolution stage transition is causally implicated;
* if it remains with comparable character, the artifact is not caused simply by
  changing spatial resolution between low/probe and high, and the next
  discriminator must move elsewhere.

The earlier `bicubic_same_source_control` remains available as archived
diagnostic machinery but is **not** requested for this hypothesis.

## Source and checkpoint provenance

The read-only helper hashes installed tracked source bytes, full HEAD/tree,
status, workflow and explicitly named checkpoints; safetensors headers retain
every tensor shape/dtype and metadata. Run it separately from timed inference:

```bash
python /home/toor/ComfyUI/custom_nodes/minimax-h3-flow-aligned-regenerate/tools/capture_boundary_provenance.py \
  --comfy-root /home/toor/ComfyUI \
  --workflow /absolute/path/frozen-workflow-A.json \
  --patcher-state /absolute/path/patcher-overlay-export.json \
  --checkpoint /absolute/path/actual-model.safetensors \
  --checkpoint /absolute/path/actual-vdn.safetensors \
  --checkpoint /absolute/path/actual-upscaler.safetensors \
  --witness-receipt /absolute/path/boundary-witness-ID.json \
  --output /absolute/path/provenance-A.json
```

The Flow directory above is the path reported by the original log; verify the
current loaded path. Replace input/output placeholders with actual paths. Repeat
`--checkpoint` for every effective model, BF16 affine source, encoder, VAE,
LoRA/DoRA and adapter checkpoint. Include their effective strengths and VDN
configuration in the frozen workflow/Patcher export. The separate helper's
`collector_process_precision` is not the running model's precision; use the
actual witness's `precision_at_actual_call` and runtime log for that.

The helper deliberately leaves `installed_provenance_complete=false` until
the overlay graph, all loaded modules/checkpoints/settings and reference/noise
identities have been checked. It does not reconstruct unavailable Patcher
state. Missing runtime hashes must remain missing evidence.

Analyze each tensor witness with VDN's `tools/analyze_boundary_witness.py` as
documented in VDN's `docs/BOUNDARY_FEATURE_WITNESS.md`. Preserve the original
observed receipt and report the CPU replay as additional diagnostic work.

## Runtime, media and costs

Save each existing Core decoder IMAGE output before Continuum assembly and the
assembled movie after it. Fan out the already-decoded frames to saving nodes;
do not add another VAE decode. Preserve raw physical-group frames including the
discarded prefix context so frame indices and trim39 can be independently
checked. Store the native fps and embedded frozen workflow with each movie.

First run the square 17→24 patch-grid comparison. Acceptance requires two
continuation boundaries, then a non-square case and Spectrum enabled/disabled
history validation. If the original graph has only one continuation, its
original two-group reproduction remains the first comparison; additional
three-group validation is separately recorded experimental work. Do not
pretend a new third-group prompt was part of the original run.

Inspect the first retained frame at local39, at least 24–40 subsequent frames
and the complete retained segment at normal speed. Assemble annotated A/B
side-by-side media using fixed top-edge/cabinet/curtain landmarks. Assess upper,
central and lower correspondences/flow/scale/shear over multiple pairs and 4×4
regions against ordinary pre-boundary camera motion. Reject a displaced jump,
freeze, border exposure, expansion or compensation drift even when first-pair
phase error improves. Exact-main background continuity must remain preserved.
Final candidate acceptance requires user inspection of these rendered movies.

Per run, the implementation adds zero H3 NFE, provider calls, VAE calls,
sampler lifetimes or history boundaries. It adds default-off bounded GPU
indexing/norm reductions, blocking CPU copies, hashes and file writes. The
32 MiB bound covers retained CPU tensor payload, not total process/serialization
RSS or transient device workspace. Diagnostic timing spans include intervening
actual block computation and synchronization; they are not a pure production
overhead measurement. Compare cold/warm, observer-disabled primed timing and
actual allocated/reserved VRAM separately. Offline witness replay and full-file
checkpoint hashing are counted diagnostic costs. No SM120 cost or speedup is
measured in this environment.

## Executed validation and open gates

CPU environment: Torch2.14.0+cpu, no CUDA. Results:

| Check | Result |
|---|---|
| Flow full tests with pinned Flow Core | 612 passed; later added stage-copy regression passes in targeted checks |
| VDN full tests with pinned Core/OpenVDN | 224 passed, including official linear/hybrid oracle; final receipt correction: 34 affected tests passed |
| Sol partitioned/mapped/history source tests | 38 passed after installing kernel import dependencies; no kernel execution |
| Flow native mixed-grid/decode with current Core | 39 passed after fixture compatibility change |
| Flow native mixed-grid/decode with pinned Core | 39 passed |
| Native/sibling source-contract scripts | passed at their documented source pins |
| Partitioned Flow/Sol/VDN contracts | passed, including bounded witness transport and unchanged provider-v4 geometry |
| VDN current-Core node/compiler smoke | passed |
| Flow Ruff/format; VDN CI-selected Ruff; compileall | passed |

Completed gates now include the actual SM120 feature witness, matched A/B
suppression discriminator, 00722 candidate-C execution, 00724 complete
partitioned learned-linear bypass, 00726 direct same-grid control and 00727
same-grid clean-ownership validation. The same-grid arms materially reduce the
rendered defect, proving the low->high spatial transition is a major contributor,
but a very slight residual shift remains even after identity transfer and zero
DC mutation. The next open gate is the same-domain weighted-dense suffix
attention discriminator described above. Audio remains independently unresolved.
No first-pair metric, source test or green CI can substitute for rendered
whole-window acceptance.

Candidate C is retained only as rejected diagnostic evidence in VDN #35. For
subsequent discriminators keep #35 installed but return the Flow selector to
`native_grid_then_map_v1` so the candidate arithmetic is inert and source
topology remains frozen. Full rollback can remove #35 and the #34/#93 diagnostic
overlays to the frozen #89/#33 baseline. Sol/Continuum stay unchanged.
Production consolidation into #89 and any release remain unqualified.
