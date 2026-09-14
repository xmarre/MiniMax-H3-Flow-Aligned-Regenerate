# Untwist full-trajectory clock trial

Status: diagnostic U experiment only. This is not a production fix.

## Why this experiment exists

The controlled 00446 O/C run reproduced the original broad banding without the rejected PR #40 endpoint transport. The colored blotch failure introduced by #40 disappeared, while banding remained in both first-high and last-high pre-guidance media.

The execution-contract recorder established the controlled `9L/7A/2F` topology, one learned upscaler call, equality of pristine target versus Flow high pre-core conditioning, and exact equality of first-high raw model output versus first-high pre-guidance output. This clears the conditioning-rebuild hypothesis for this run and selects the architecture document's bounded U experiment before a same-state replay.

Current Untwist derives progress independently from each child sampler's `sample_sigmas`. For the controlled eight-denoiser-coordinate Euler trajectory, the executed handoff is original index 5. Therefore the first high call belongs at global progress `5/7`, while its three-call child schedule currently restarts at progress `0`.

## Intervention

Apply `MiniMax H3 Untwist Full-Trajectory Clock Trial` **after** `MiniMax H3 Execution Contract Diagnostics` and before the production sampler.

The node publishes a temporary `h3_flow_sampling_context` only while each low/probe/high child sampler is executing. It contains the original nonzero sigma trajectory, original schedule digest, stage, original stage-start index and an invocation generation token. The marker is restored in `finally` and is rejected if another owner already published one.

A compatible diagnostic build of `ComfyUI-Untwisting-RoPE` consumes that marker only for MiniMax H3 Untwist progress. When the marker is absent, Untwist keeps its legacy child-local `sample_sigmas` behavior.

The trial does **not**:

- replace or mutate sampler-owned `sample_sigmas`;
- change Flow state/noise or learned handoff state;
- change guidance, VDN, Sol or Spectrum ownership;
- add an H3 evaluation;
- add an upscaler invocation;
- enable weighted Mixed-Grid or rejected state transport.

Untwist publishes the same corrected progress/active state to its Spectrum runtime metadata that it uses for execution.

## Controlled expected clock

For 00446's original nonzero coordinates, the intended global Untwist progress is:

- low calls: original indices 0..4 => `0/7, 1/7, 2/7, 3/7, 4/7`;
- exact probe at handoff: original index 5 => `5/7`;
- high calls: original indices 5..7 => `5/7, 6/7, 7/7`.

The probe and first high call intentionally share the same trajectory coordinate and therefore the same Untwist progress. The sampler histories remain independent.

## Required run

Keep the 00446 workflow inputs and generation settings unchanged. Keep PR #40 absent. Keep the existing #35 checkpoint diagnostic, #36 learned-anchor validation and #41 execution-contract recorder.

Model order:

1. all ordinary MODEL patchers, including Untwist;
2. `MiniMax H3 Execution Contract Diagnostics` (`capture_mib=256`, `strict_provenance=true`);
3. `MiniMax H3 Untwist Full-Trajectory Clock Trial`;
4. production sampler.

Save the execution-contract report and Flow metrics. Because this intervention deliberately corrects the trajectory clock across **low, probe and high**, decode the same checkpoint set needed to see where any difference first appears: `low_last_model_clean`, `exact_probe_clean`, `learned_transfer_clean`, `first_high_model_raw`, `first_high_pre_guidance`, `last_high_pre_guidance`, and the ordinary final output. Use the same decoder settings and comparison frame times as 00446. Do not judge U only from the final video.

The run is valid only if accounting remains truthful and any changed Spectrum actual/forecast decisions are reported rather than hidden. The expected controlled accounting remains `9L/7A/2F` with one learned-upscaler call; a different route/NFE outcome must be recorded and treated as an attribution limitation rather than forced back to the expected count.

Decision rule:

- first-high raw/pre banding materially improves: Untwist child-clock restart is implicated; proceed to production-contract design and broader low/high media validation;
- banding remains materially unchanged: reject U for this case and proceed to the architecture document's same-state high replay R;
- topology/provenance/route contract changes in an uncontrolled way: the run is attribution-invalid and must not be interpreted as U evidence.
