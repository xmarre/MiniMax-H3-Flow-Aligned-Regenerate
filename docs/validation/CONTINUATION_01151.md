# 01151: target-grid continuation passes the reported boundary check

The user reports that `spatial_stage_control=same_grid_target_control` removes
the visible frame shift and subsequent-chunk tone shift in this case. The user
also reports that the audio defects, including unprompted dialogue at the chunk
end, disappear. The supplied runtime receipts corroborate a clean raw video
boundary. This qualifies the selected configuration for this reported case;
the heterogeneous path that failed in 01150 remains unresolved.

The initial evidence qualification changed documentation only. The subsequent coordinated v0.3.9 release adopts the selected profile as the new-node default without changing its sampler arithmetic.
The control already exists on **MiniMax H3 Partitioned Exact-Prefix Handoff**.
Keep `handoff_transfer_control=learned_3d` and
`vdn_temporal_carrier_policy=native_grid_then_map_v1`, as required by this control.
Refresh PR overlays through ComfyUI Patcher when updating dependencies.

## Matched evidence and actual execution

[Comparison receipts](CONTINUATION_01151_RECEIPTS.json) retain input-file hashes,
selected events, prompt-manifest identities, raw decode measurements and stage
timings. Both complete PT215 manifests are equal between 01150 and 01151. Their
prompt text, physical timeline, conditioning descriptors and reported hashes
match. This does not independently establish equality of all reference and text
embedding tensor bytes.

The runtime hash of the full 12-token authoritative video prefix is identical:
`34716a72391e5e7b5a2ee44d6a3cf35f5dde501f1b9293156dd3ecd4ef89a7ca`.
`partitioned_stage.tensor_sha256` hashes the contiguous CPU byte view, rather
than a statistical summary. The exported two-token prefix and initial high mask
hashes also match. The 01151 tensor bundle itself was not supplied for independent
verification or a first-high decoder replay; exported hashes remain runtime
receipts. The 01150 bundle was independently verified in its preceding review.

| Continuation property | 01150 | 01151 |
| --- | --- | --- |
| Executed source / target grid | 44x44 / 62x62 | 62x62 / 62x62 |
| Video tokens / protected prefix | 47 / 12 | 47 / 12 |
| Handoff sigma | 0.8780487775802612 | 0.8780487775802612 |
| Learned checkpoint transfer invoked | Yes | No |
| Clean / residual transfer | Cross-grid | Identity |
| Prefix overwrite RMS | 0.38858736 | 0 |
| First-token DC delta RMS | 0.17525901 | 0 |
| Low-stage prefix log key measure | -0.68588950 | 0 |
| Native uniform VDN linear calls, whole run | 200 | 550 |

The configured progressive source is still 44x44, but the continuation control
overrides the actual low/probe grid to 62x62 and preserves the configured handoff
split. The identity provider clones the clean video and does not invoke the
learned checkpoint. The one-token DC bridge still has declared support, but its
delta is exactly zero. Metadata retaining a learned-provider kind does not mean
that the checkpoint executed: the explicit invocation receipt is false.

The first all-generated chunk remains progressive, 44x44 to 62x62, in both
runs. The selector is not a global disable of progressive generation. The seed
is unchanged; differently shaped low-grid noise and denoising trajectories are
not identical realizations. High sampler masks match, and both final exact
audio/video prefixes pass the runtime integrity check.

## Rendered boundary and tone

| Raw decoded first-transition estimate | 01150 | 01151 |
| --- | ---: | ---: |
| Upper45 horizontal displacement, px | -2.379722 | -0.109698 |
| Upper45 vertical displacement, px | +4.107829 | +0.140327 |
| Full-frame horizontal displacement, px | -1.518189 | -0.051131 |
| Full-frame vertical displacement, px | +2.867863 | +0.105009 |
| Upper45 fitted X scale | 0.98995782 | 0.99991046 |
| Upper45 fitted Y scale | 0.99075236 | 0.99991372 |

These PT212/PT224 measurements precede assembly correction. They are motion
estimates across successive frames, not comparisons with a stationary ground
truth. Near-zero affine scale confidence prevents treating the fitted scales as
a calibrated proof of perfect alignment. Their small residuals, the raw
translation estimates and the user's inspection support the successful result.

PT228 classifies 01151 as `clean_boundary`, with no micro-flash or exposure ramp.
PT216 reports `applied=false`, `patch_frames=0`: the result is not explained by
the one-frame photometric patch. The sampled mean luma changes from 0.32903495
on the last retained pre-boundary frame to 0.33050627 on the first generated
frame. The raw tone diagnostic does not modify images. Absence of a tone change
across the subsequent chunk is the user's visual report; this short diagnostic
does not independently certify an entire clip.

The coarse latent phase-correlation metric is not a rendered acceptance gate.
Its final upper45 boundary-versus-pre norm is **larger** in the successful
control: 0.09056434 cells versus 0.06144467 in failed 01150. Do not introduce a
repair threshold from that scalar or interpret it as evidence contradicting
the raw decode and reported visual result.

## Audio and causal limits

Audio is not spatially resized, but H3 predicts it jointly with video. Moving
the low/probe video to the target grid also makes source-carrier and target
audio spatial RoPE endpoints coincide, sets the prefix measure adjustment to
zero and selects native uniform VDN math for those stages. It therefore changes
audio conditioning without changing audio mask ownership.

The requested 16-tick audio overlap remains the legacy
`sampler_mask_exact_timestep` alias with **zero effective/applied ticks** in both
runs. Masks remain exact, and the carried audio interior correlation is 0.999999
with a -0.0008 dB difference in both runs. Those unchanged overlap receipts do
not explain why unwanted dialogue disappears. In 01151 the raw half-second
boundary window has +0.9735 dB level difference and spectral centroids
333.419 / 321.750 Hz. These signal measurements cannot prove the absence of
speech; that semantic acceptance comes from the user's listening report.

Zero user DoRA hooks still does not mean zero adapters: the built-in VDN default
and turbo adapters remain at 0.5 each. The successful control supports keeping
the current same-grid configuration. It does not justify changing adapter
strengths or blaming an external LoRA for the underlying failure.

This comparison narrows the failing configuration to heterogeneous execution
as a whole. It changes the low trajectory, spatial noise layout, cross-grid
attention/VDN behavior, audio position geometry, learned clean transfer and
residual transport together. It cannot distinguish those contributions. The
same target-high/decoder stack can produce an accepted result with the same-grid
inputs; interactions with heterogeneous inputs remain possible. Do not promote
this result to a learned-upscaler-only diagnosis or closure of the original
heterogeneous defect.

## Speed

| Measured wall time | 01150, s | 01151, s | Change |
| --- | ---: | ---: | ---: |
| Whole prompt execution | 539.630 | 353.230 | -34.54% |
| First-chunk sampler | 190.802 | 103.210 | -45.91% |
| Continuation sampler | 209.128 | 221.201 | +5.77% |
| Continuation low stage | 100.731 | 118.402 | +17.54% |
| Continuation probe | 15.202 | 20.744 | +36.46% |
| Continuation transfer | 8.038 | 1.391 | -82.69% |
| Continuation high stage | 82.981 | 78.793 | -5.05% |
| Whole execution outside both samplers | 139.699 | 28.818 | -79.37% |

Stage timings come from Flow's wall receipts; Spectrum's narrower model-loop
timings are not substituted. Stage sums omit wrapper work, so they need not
equal the sampler wall time. Logical calls (34), actual transformer evaluations
(22), forecasts (12), sampler lifetimes (6) and history boundaries (4) are equal
across the whole runs. No additional evaluation explains this control.

The successful run is 3m06s faster overall, while its continuation costs about
12s more. Much of the overall saving occurs outside the modified continuation
path: even the unchanged first-chunk path is substantially faster. These are
sequential executions, not a controlled cold/warm benchmark; cache, loading and
other runtime-state differences can contribute. Do not advertise same-grid as
intrinsically faster. The observed continuation cost is modest in this case,
and the larger target-grid low stage remains a resource tradeoff.

## Qualification

Keep `same_grid_target_control` for the currently accepted workflow. Broader
scene, seed and hardware acceptance is not established by one run. The coordinated release promotes the selected target-grid profile; the heterogeneous path remains unresolved. The original evidence update preserved runtime arithmetic. The later default promotion is checked separately for widget/runtime agreement and explicit saved-value compatibility. Existing native arithmetic tests establish their limited contracts;
they are not the source of the visual or audio acceptance claim.
