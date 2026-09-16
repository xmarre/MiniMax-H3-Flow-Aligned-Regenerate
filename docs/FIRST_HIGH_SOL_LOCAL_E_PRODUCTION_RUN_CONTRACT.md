# First-High Sol-Local Experiment E — Production Run Contract

This document defines the workstation procedure for the single allowed next CUDA experiment in the MiniMax-H3 first-high artifact investigation: **Experiment E** from the authoritative Sol-local design at ComfyUI-Sol-H3 commit `0b8715faa0a82c730f5aaf0e44b1185e64291e49`.

Experiment E is diagnostic-only. It replays the existing experiment-R first-high state, executes exactly one real high-stage H3 call, keeps VDN restricted local support plus its complement, and changes only the final local Sol-Attn selection so every valid local K block is exact. It does not regenerate R, rerun W, run a full diffusion trajectory, or authorize a production fix.

## 1. Exact PR-overlay stack

Run E through the existing **ComfyUI Patcher PR-overlay stack**. Do not replace custom-node checkouts manually.

| Repository | E PR overlay | Reviewed source | Purpose |
| --- | --- | --- | --- |
| `xmarre/ComfyUI-Sol-H3` | PR #12 on PR #11 | current #12 head; corrected E diagnostic source blob `1c02baa4bae1c17c2ee2ba2e5081012737e83e01` | all-selected local path and numerical witnesses |
| `xmarre/ComfyUI-VDN-H3-Plus` | PR #16 on PR #15 | `10bf368bee38b9f942961e6d8c374b5af6c4bc16` | restricted VDN geometry/complement and witness bridge |
| `xmarre/MiniMax-H3-Flow-Aligned-Regenerate` | PR #44 on PR #43 | current #44 head; E core `7ea52470ab5ac82b12d58bcb87db82d715b8005c` plus bounded instrumentation corrections | preserved-R one-call replay, provenance, report and durable evidence |

PRs #11, #15 and #43 remain preserved W evidence bases. E is an overlay above them and does not rewrite or repurpose them.

## 2. Workstation instrumentation evidence already established

Four successive workstation attempts identified diagnostic instrumentation defects. None is operator/media evidence for the underlying Sol-local hypothesis.

1. **Derived replacement-chain closure depth.** E's extra reviewed VDN wrapper exposed one current-only `closure` level in Core's derived `replacement_chain_dit` provenance. #44 now normalizes only that E-owned closure-depth expansion after proving all fifty live chains are exactly `E -> reviewed W -> captured production VDN`. All non-closure changes and capture-owned differences remain fatal.
2. **Pre-H3 Sol sparse import.** Installing E imports unchanged production `sol_h3/sparse.py` before Flow's pre-H3 source inventory, whereas R captured before that bridge was necessarily imported. #44 suppresses only that replay-only entry after proving the already source-gated Sol package, exact reviewed sparse blob `9f462591e3492d0b7af476c16024c79d1237505e`, and recorded SHA-256 against disk bytes. Later lazy-loaded SM120 kernel/compiler provenance is still mandatory.
3. **Replay storage device versus H3 execution device.** R may store the replay latent on CPU. #44 no longer treats that storage device as the Sol execution device: explicit CUDA remains authoritative; a non-CUDA replay resolves to CUDA only with exactly one CUDA device; multi-GPU non-CUDA replay is ambiguous and fails closed; the resolved device must be compute capability `(12, 0)` before memory preflight.
4. **E rich-metric arithmetic-gate schema.** The next run passed the previous gates, entered the real H3 call, loaded the verified packaged `cute_sm120` backend, completed the 22 native dense local calls and reached the first block-2 local witness. The ordinary Sol arithmetic gate itself produced a valid payload including `reference_peak_abs=18.125` and `catastrophic_max_abs_limit=72.5`, but E's richer `all_selected_vs_native` metric payload omitted those two standard gate fields. Calling the unchanged production `arithmetic_gate_passes()` therefore raised `KeyError: 'catastrophic_max_abs_limit'` during the first witness. Sol PR #12 now computes the reference peak incrementally across the existing Q64 pass and derives the catastrophic limit from the same production constants; non-finite metrics remain fail-closed. No attention route, threshold, kernel, VDN geometry, model state, receipt, or H3-call count is changed by this correction.

The fourth attempt **did enter** a real H3/SM120 call, unlike the first three, but it aborted inside the first block-2 witness before the E call completed. It therefore still does not establish a valid E arithmetic, operator, or media result.

## 3. Preserved R evidence

E consumes the existing R bundle and must not overwrite or regenerate it:

```text
/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json
/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.pt
```

Before changing the active overlay stack, record its hashes:

```bash
cd /home/toor/ComfyUI
sha256sum \
  output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json \
  output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.pt
```

The capture ID must remain exactly `234ed062128e43ed8d5ec63e27517b22`.

## 4. Patcher preparation and byte checks

Stop ComfyUI completely. Keep the existing stack and refresh these overlays in order:

- **Sol-H3:** through PR #11, then PR #12;
- **VDN-H3-Plus:** through PR #15, then PR #16;
- **Flow-Aligned-Regenerate:** through PR #43, then PR #44.

Do not manually switch, reset, rebase, clean, or replace the custom-node repositories. Let ComfyUI Patcher construct the working trees.

After Patcher applies the overlays, verify the critical bytes:

```bash
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3
test "$(git hash-object sol_h3/first_high_sol_local_diagnostic.py)" = \
  1c02baa4bae1c17c2ee2ba2e5081012737e83e01
test "$(git hash-object sol_h3/first_high_sol_local_witness_bridge.py)" = \
  3ce204248286a6664d2b0e02fff7a20ddc233185
test "$(git hash-object sol_h3/__init__.py)" = \
  c4184d48f53a1b0703816c371f616e538acde5a7
test "$(git hash-object sol_h3/sparse.py)" = \
  9f462591e3492d0b7af476c16024c79d1237505e

cd /home/toor/ComfyUI/custom_nodes/ComfyUI-VDN-H3-Plus
test "$(git hash-object vdn_h3/first_high_sol_local_diagnostic.py)" = \
  8d0b0c9af1a20bc621635d9931376cca43a71fea
test "$(git hash-object vdn_h3/first_high_sol_local_bridge.py)" = \
  2277f7071118d53eb56fe9cd86110cf7ca799efb
test "$(git hash-object __init__.py)" = \
  c61e2b763d9fba35361729aa0ac1ef9259dcb298

cd /home/toor/ComfyUI/custom_nodes/MiniMax-H3-Flow-Aligned-Regenerate
test "$(git hash-object h3_flow_regenerate/first_high_sol_local_diagnostic.py)" = \
  138a27c8fbba0ddc9140b6301187a2b01ce0f1fb
test "$(git hash-object __init__.py)" = \
  f15b9ab236a7a566354d22b0b7430843e68fd5b6
test "$(git hash-object h3_flow_regenerate/first_high_sol_local_e_source_delta.json)" = \
  b6b1cbd6b06a8a564515b5ed9aca849e74fa821b

cd /home/toor/ComfyUI
test "$(git hash-object comfy/model_sampling.py)" = \
  4a55655207115a150d5254c5b364b03a6c256627
test "$(git hash-object comfy/latent_formats.py)" = \
  8d8ff6c5f179da8dcdeba854f220fb657de2af1c
test "$(git hash-object comfy/model_patcher.py)" = \
  08e69fae149692fed60d886890cc0a8eb80bdf26
test "$(git hash-object comfy/k_diffusion/sampling.py)" = \
  4a638008a3df3a2f167e9bf343a258a0f5655ede
```

Successful `test` commands are silent. Any failure means the active Patcher-built runtime does not contain the reviewed E bytes; do not queue E.

## 5. Process and graph contract

Start a **fresh ComfyUI process**. Keep the same production checkpoint/model, prompt/references, seed, target geometry, complete sampler schedule, Flow/Spectrum/Sol/VDN/DiffAid settings, model loader and attention backend represented by R.

The model path is:

1. ordinary production model and companion nodes;
2. ordinary progressive Target Input with `transfer_mode=learned_3d`, `exact_prefix_mode=fallback`;
3. `MiniMax H3 Execution Contract Diagnostics`, `capture_mib=512`, `strict_provenance=true`;
4. its MODEL output into `MiniMax H3 First-High Sol-Local Diagnostic E`;
5. E selects `h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json`;
6. E-patched MODEL goes to the same sampler path used by R.

Do **not** queue R capture/save/replay nodes, W comparison/report nodes, Untwist, Weighted/Mixed-Grid attention-measure research, external/reduced VDN sequence mode, or progressive-only old diagnostic extractor branches.

The high suffix remains exactly:

```text
[0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]
```

E stops after the first real high-stage Euler model call while allowing Core callback-x0 externalization and outer-sample cleanup to complete.

## 6. Report and media wiring

Add `MiniMax H3 First-High Sol-Local E Report`:

- `diagnostic` <- `H3_FLOW_FIRST_HIGH_SOL_LOCAL_E` from E;
- `trigger` <- sampler LATENT from the E-patched model path.

The report returns `report`, `first_high_model_raw`, and `first_high_pre_guidance`. Decode/save both complete clips with the same VAE used for the controlled baseline.

Durable machine evidence is written under:

```text
/home/toor/ComfyUI/output/h3_first_high_sol_local_e/
```

Preserve the generated `.json` and `.pt` files.

## 7. Memory and SM120 contract

E defaults to a 2 GiB additional diagnostic budget for CUDA and 2 GiB for host memory. Optional overrides are `H3_FIRST_HIGH_E_CUDA_BUDGET_GIB` and `H3_FIRST_HIGH_E_CPU_BUDGET_GIB`; do not lower them simply to bypass a failure.

Replay storage and H3 execution device are distinct. A CUDA replay uses that exact device. A non-CUDA replay resolves to CUDA only when exactly one CUDA device exists. Multi-GPU non-CUDA replay fails closed. The resolved device must be SM120 before memory-headroom checks run.

## 8. Queue exactly once

From the fresh process, queue E exactly once. A valid execution must prove at minimum:

- `1 logical / 1 actual / 0 forecast` high-stage H3 calls;
- zero learned-upscaler calls;
- exact preserved-R sampler/H3 entry hashes;
- exactly 700 VDN attention subcalls;
- 528 returned `vdn_local_sol_all_selected_e` calls;
- 22 native dense-layer local calls;
- 50 native global calls;
- 100 native anchor calls;
- no ordinary production sparse, external Mixed-Grid, or square-expanded local Sol route;
- exact packaged verified SM120 Sol-Attn backend;
- exact Q/K/V preservation for block 2 groups 0, 2 and 10;
- complete all-selected route traces;
- valid Core callback-x0 externalization and outer cleanup;
- exact reviewed source/provenance gates.

Do not interpret media when `execution_valid` is false.

## 9. Evidence to preserve

Preserve the complete ComfyUI log, E report JSON, durable E `.json`/`.pt`, both decoded clips, pre-run R hashes, and any runtime metrics file. If E fails before report emission, preserve the log and emitted durable files and do not immediately rerun.

## 10. Decision gate after E

No follow-up H3 experiment is authorized until the report, witnesses and both decoded clips are reviewed together:

- `execution_valid=false`: diagnose the instrumentation/runtime defect and rerun only E after a causally supported correction;
- `execution_valid=true` but `arithmetic_conformant=false`: localize descriptor/stride/mask/normalization arithmetic from saved witness evidence;
- valid E with broken media despite passing all-selected arithmetic: extend only the bounded E witness required to explain the discrepancy;
- valid E with clean media: use the authoritative design decision table to choose the single permitted next localization experiment.

M/T/C remain conditional. Clean E alone does not authorize permanently dense local attention or a production fix.

## 11. Validation boundary

Hosted CPU/source-contract CI can validate the corrected metric schema, fail-closed behavior, companion integration and source manifests. It cannot establish SM120 arithmetic or media causality. A complete workstation E call remains the empirical gate.
