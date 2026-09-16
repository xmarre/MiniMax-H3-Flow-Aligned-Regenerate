# Production mapped-neighbor candidate replay

This validation layer tests the production mapped-neighbor query-position contract from Sol-H3 design commit `941b8571b099d18d1fe4e4deef9f04fd343b1098`. It is a bounded release gate, not a sampling-policy change and not promotion approval.

The candidate is intentionally separate from the W/E/M diagnostics. Flow PRs #43, #44 and #45 remain frozen diagnostic evidence and are neither imported nor executed by this path. The candidate executes the ordinary upgraded VDN retained grouped-attention path and the ordinary packaged Sol-H3 ABI.

## Required paired implementation

The source-delta manifest pins the reviewed production pair:

- Sol-H3 PR #14 at `1459b34853a39d1054fd5c8643de28a7b0e240c7`;
- VDN-H3-Plus PR #18 at `333d63f81d33fe29dc1f1f637c5f4a4396880f99`;
- this Flow validation layer on top of the experiment-R implementation at `4ae2e35f77ed961151ab5695bef9e1dbe277cc54`.

The runtime source gate verifies exact candidate Git blobs for the changed Flow, Sol, packaged SM120, and VDN files. Unchanged R-capture sources remain byte-checked by the existing replay provenance machinery. The candidate fails closed if an unreviewed source difference appears.

## Workflow

Start from the same production model, references, prompt, seed, schedule, Flow/Spectrum/Sol/VDN/DiffAid settings and companion stack used by the preserved experiment-R capture. The saved bundle for capture `234ed062128e43ed8d5ec63e27517b22` is reused; R is not regenerated.

Run the candidate from a fresh ComfyUI process and queue this validation alone. Do not overlap another queue that shares the same Spectrum runtime instance while the bounded receipt observer is installed.

The queued validation path is:

1. **MiniMax H3 Execution Contract Diagnostics** with `strict_provenance=true`;
2. **MiniMax H3 Production Mapped-Neighbor Candidate**, selecting the preserved R bundle;
3. the unchanged normal sampler, using the original full schedule and seed;
4. **MiniMax H3 Production Mapped-Neighbor Candidate Report**, connected to the candidate handle and sampled candidate LATENT;
5. decode/save all three report LATENT outputs:
   - `first_high_model_raw`;
   - `first_high_pre_guidance`;
   - `first_high_final`.

The candidate replaces the normal progressive execution only inside its validation wrapper. It restores the Flow binding, progressive model option, trajectory and validation instrumentation before returning. Core outer-sample cleanup must complete normally.

Do not place any W/E/M first-high operator node in the queued graph. The candidate also rejects W/E/M transformer instrumentation, selector replacements, replay capture/replay wrappers, and the rejected Untwist trial.

## Backend-receipt ownership

Spectrum owns `attention_backend_receipts_v1` for each actual model call. Its backend-history preflight creates the per-call receipt list and Sol appends ordinary production receipts to that list. The candidate must not replace or pre-own that key because doing so would inspect a different object from the one used by the executed production call.

For the bounded replay, validation temporarily mirrors the receipt tuple passed to the active Spectrum runtime's completed `observe_backend_history(...)` boundary. The original Spectrum observer executes first. The validation mirror therefore sees the same provider-qualified numerical receipt tuple that Spectrum consumes, without changing Sol dispatch, VDN dispatch, receipt construction, backend-history policy, selector logic, or the generic receipt-list owner. The temporary instance-level observer is removed before the candidate returns.

A valid result requires exactly one completed Spectrum backend-history observation for the one actual H3 call and requires Spectrum to have accepted the mapped provider route as forecast-safe. A stale pre-owned generic receipt key, a pre-existing instance-owned observer, duplicate observations, malformed receipts, or failed provider qualification invalidates the candidate.

## Structural acceptance

A valid bounded run must report all of the following:

- mode `production_mapped_neighbor_v4_candidate` and the preserved R capture ID;
- exactly `1 logical / 1 actual / 0 forecast` first-high model calls;
- zero learned-upscaler calls;
- exact captured first-high sampler video/audio and H3 video/audio entry hashes;
- exact runtime provenance except for the reviewed production source delta;
- packaged Sol contract `sana-sol-engine-sol-attn-64-rect-sm120-mapped-neighbor-v4`;
- VDN provider API v4;
- no loaded or executed W/E/M operator substitution;
- exactly one Spectrum backend-history completion with `spectrum_forecast_safe=true` and candidate ownership of the generic receipt list reported as false;
- exactly 700 Sol backend receipts from that completed Spectrum observation:
  - 528 `vdn_local_sol_mapped_v1`;
  - 22 `vdn_dense_warmup` on blocks 0 and 1 only;
  - 50 `vdn_global_native`;
  - 100 `vdn_anchor_native`;
- exactly 14 VDN attention subcalls per H3 block;
- mapped calls only on blocks 2–49, with all 11 local groups and the preserved Q/KV restricted-domain geometry;
- all mapped receipt fields bound to one VDN owner generation, one plan digest, the mapped-neighbor-v1 policy, the v4 kernel contract, and `executed=true`;
- no local native fallback or kernel-unavailable fallback;
- exact Core callback-x0 externalization;
- complete Core cleanup: no retained `inner_model`, `loaded_models`, or multigpu thread-pool ownership.

A structural failure is not a performance or media result. Fix the failed invariant before any full-trajectory run.

## Media acceptance

The report deliberately emits `media_clean=null`. Structural success does not establish visual quality.

Decode and inspect all three candidate clips: raw model output, pre-guidance output, and final first-high output. The candidate is not accepted for full-trajectory validation until all three are manually clean. Preserve the report and decoded media as the production-candidate evidence set.

## Relation to the SM120 same-input gate

The Flow replay is downstream of the separate Sol same-input production gate. That gate compares the real packaged SM120 candidate against the preserved E witness and historical diagnostic-M implementation without rerunning H3. The controlled Flow replay must not substitute for that operator/performance evidence, and the operator probe must not substitute for this end-to-end first-high replay.

Only after both gates pass should the representative full trajectory be run and reviewed for timing, NFE topology, video and audio.
