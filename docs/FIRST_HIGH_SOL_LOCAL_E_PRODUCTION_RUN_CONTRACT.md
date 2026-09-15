# First-High Sol-Local Experiment E — Production Run Contract

This document defines the workstation procedure for the single allowed next CUDA experiment in the MiniMax-H3 first-high artifact investigation: **Experiment E** from the authoritative Sol-local design at ComfyUI-Sol-H3 commit `0b8715faa0a82c730f5aaf0e44b1185e64291e49`.

Experiment E is diagnostic-only. It replays the existing experiment-R first-high state, executes exactly one real high-stage H3 call, keeps VDN restricted local support plus its complement, and changes only the final local Sol-Attn selection so every valid local K block is exact. It does not regenerate R, rerun W, run a full diffusion trajectory, or authorize a production fix.

## 1. Exact reviewed implementation and PR overlays

Run E through the existing **ComfyUI Patcher PR-overlay stack**. Do not replace the custom-node checkouts manually.

| Repository | E PR overlay | Reviewed source | Purpose |
| --- | --- | --- | --- |
| `xmarre/ComfyUI-Sol-H3` | PR #12, stacked on PR #11 | `2eedd341a0f412803afc92702996c07e1702edb7` | E Sol all-selected path and numerical witnesses |
| `xmarre/ComfyUI-VDN-H3-Plus` | PR #16, stacked on PR #15 | `10bf368bee38b9f942961e6d8c374b5af6c4bc16` | E VDN overlay and generated-loader bridge |
| `xmarre/MiniMax-H3-Flow-Aligned-Regenerate` | PR #44, stacked on PR #43 | current PR head; E core `7ea52470ab5ac82b12d58bcb87db82d715b8005c` plus the reviewed bounded runtime-provenance corrections | one-call replay, source/provenance gate, report and durable evidence |

PRs #11, #15 and #43 remain the preserved W evidence bases. E is the next overlay layer; it does not rewrite or repurpose those PRs.

The first real workstation E attempt on the original #44 overlay established that the graph ordering was correct and reached the sampler, but the diagnostic aborted before H3 because E's additional reviewed VDN wrapper layer exposed one extra bounded `closure` level in Core's derived `replacement_chain_dit` provenance. That is an E instrumentation self-effect, not model/media evidence. PR #44 normalizes only this derived current-only closure-depth expansion **after** proving the exact fifty live E → W → captured-production VDN object-patch chains. Any non-closure difference, capture-owned closure difference, key-set difference, or other provenance change still fails closed.

The next workstation attempt passed that gate and exposed one further deterministic E self-effect before H3: `sol_h3/sparse.py` appeared only in the replay runtime's loaded-companion inventory. E's Sol installer must import that production sparse bridge in order to install its bounded local-attention diagnostic wrapper, while R captured the companion-source inventory before the first high H3 call and therefore had not necessarily imported the bridge yet. The file itself is unchanged across Sol PR #11/#12: Git blob `9f462591e3492d0b7af476c16024c79d1237505e`. PR #44 now removes this one replay-only source entry only after proving the live sibling file belongs to the already source-gated Sol package, has exactly that reviewed Git blob, and its recorded SHA-256 equals the actual file bytes. Every other replay-only companion source remains visible to the ordinary fail-closed provenance comparison. This normalization does **not** waive the later lazy-loaded packaged SM120 kernel/compiler provenance; those modules are still verified by E after the kernel is actually loaded.

Neither failed workstation attempt executed an H3 E call, so neither provides operator, arithmetic, or media evidence.

The Flow source manifest pins the exact reviewed executable source bytes. The four installed ComfyUI Core files are verified by their loaded file bytes at runtime, not by a Git branch or tag:

- `comfy.model_sampling`: Git blob `4a55655207115a150d5254c5b364b03a6c256627`
- `comfy.latent_formats`: Git blob `8d8ff6c5f179da8dcdeba854f220fb657de2af1c`
- `comfy.model_patcher`: Git blob `08e69fae149692fed60d886890cc0a8eb80bdf26`
- `comfy.k_diffusion.sampling`: Git blob `4a638008a3df3a2f167e9bf343a258a0f5655ede`

Do not substitute a nearby public ComfyUI revision merely because its Git label looks compatible.

## 2. Preserved R evidence

E consumes the existing R bundle and must not overwrite or regenerate it:

```text
/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json
/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.pt
```

Before changing the active Patcher overlay stack, record the bundle hashes:

```bash
cd /home/toor/ComfyUI
sha256sum \
  output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json \
  output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.pt
```

The R capture ID must remain exactly `234ed062128e43ed8d5ec63e27517b22`.

## 3. ComfyUI Patcher / PR-overlay preparation

Stop ComfyUI completely before rebuilding the PR overlay stack. Keep the already-established production/diagnostic stack and add E on top in this order:

- **Sol-H3:** existing stack through PR #11, then PR #12;
- **VDN-H3-Plus:** existing stack through PR #15, then PR #16;
- **Flow-Aligned-Regenerate:** existing R/W stack through PR #43, then PR #44.

Do not remove PR #11/#15/#43 when adding E. Let ComfyUI Patcher construct the working tree; do not manually switch/reset/rebase/clean the custom-node repositories for this run.

After Patcher has applied the overlays, verify the critical loaded-source bytes:

```bash
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3
test "$(git hash-object sol_h3/first_high_sol_local_diagnostic.py)" = \
  79b3013aa936a836fc4e5c5cf3e80f8ac06a49ac
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
  4eab5193d8e1a3ed51f53a7ab434846494b163d0
test "$(git hash-object h3_flow_regenerate/first_high_sol_local_e_source_delta.json)" = \
  38406fda32593a649d481b5bb48044277150ba2d

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

A successful `test` command is silent and exits zero. Any failure means the active Patcher-built runtime does not contain the reviewed E bytes; do not queue E.

## 4. Process and graph contract

Start a **fresh ComfyUI process** after Patcher rebuilds the exact overlays. Use the same production checkpoint/model configuration, prompt/references, seed, target geometry, complete sampler schedule, Flow/Spectrum/Sol/VDN/DiffAid configuration, model loader and attention backend represented by R.

The active model path is:

1. ordinary production model and companion nodes;
2. ordinary progressive Target Input (`transfer_mode=learned_3d`, `exact_prefix_mode=fallback`);
3. `MiniMax H3 Execution Contract Diagnostics` with `capture_mib=512` and `strict_provenance=true`;
4. its MODEL output into `MiniMax H3 First-High Sol-Local Diagnostic E`;
5. E selects `h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json`;
6. E-patched MODEL goes to the same sampler path used by R.

Do **not** queue R capture/save/replay nodes, W comparison/report nodes, Untwist, Weighted/Mixed-Grid attention-measure research, external/reduced VDN sequence mode, or progressive-only old diagnostic extractor branches.

The high suffix remains exactly:

```text
[0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]
```

E internally stops after the first real high-stage Euler model call while allowing the installed Core callback-x0 externalization and outer-sample cleanup path to complete.

## 5. E report and media wiring

Add `MiniMax H3 First-High Sol-Local E Report`:

- `diagnostic` ← `H3_FLOW_FIRST_HIGH_SOL_LOCAL_E` from E;
- `trigger` ← sampler LATENT from the E-patched model path.

The report returns `report`, `first_high_model_raw`, and `first_high_pre_guidance`. Decode/save both complete clips with the same VAE used for the controlled baseline.

Durable machine evidence is written under:

```text
/home/toor/ComfyUI/output/h3_first_high_sol_local_e/
```

Preserve the generated `.json` and `.pt` files.

## 6. Memory contract

E defaults to a 2 GiB additional diagnostic budget for CUDA and 2 GiB for host memory and performs its own preflight. Optional overrides are `H3_FIRST_HIGH_E_CUDA_BUDGET_GIB` and `H3_FIRST_HIGH_E_CPU_BUDGET_GIB`; do not lower a budget merely to bypass a failure.

## 7. Queue exactly once

From the fresh process, queue the E graph exactly once. A valid E execution must prove at minimum:

- `1 logical / 1 actual / 0 forecast` high-stage H3 calls;
- zero learned-upscaler calls;
- exact preserved-R sampler/H3 entry state hashes;
- exactly 700 VDN attention subcalls;
- 528 returned `vdn_local_sol_all_selected_e` calls;
- 22 native dense-layer local calls;
- 50 native global calls;
- 100 native anchor calls;
- no ordinary production sparse, external Mixed-Grid, or square-expanded local Sol path;
- exact packaged verified SM120 Sol-Attn backend;
- exact Q/K/V preservation for block 2 groups 0, 2 and 10;
- complete all-selected route traces;
- valid Core callback-x0 externalization and outer-sample cleanup;
- exact reviewed source/provenance gates.

Do not interpret media when `execution_valid` is false.

## 8. Evidence to preserve

Preserve the complete ComfyUI log, E report JSON, durable E `.json`/`.pt`, both decoded clips, pre-run R bundle hashes, and any runtime metrics file. If E fails before report emission, preserve the log and any emitted durable files and do not immediately rerun.

## 9. Decision gate after E

No follow-up H3 experiment is authorized until report, witnesses and both decoded clips are reviewed together:

- `execution_valid=false`: diagnose the instrumentation/runtime defect; rerun only E after a causally supported correction;
- `execution_valid=true` but `arithmetic_conformant=false`: localize descriptor/stride/mask/normalization arithmetic from the saved witness evidence;
- valid E with broken media despite passing all-selected arithmetic: extend only the bounded E witness needed to explain the discrepancy;
- valid E with clean media: use the authoritative design decision table to choose the single permitted next localization experiment.

M/T/C remain conditional; clean E alone does not authorize permanently dense local attention or a production fix.

## 10. Structural validation

The exact E Sol/VDN stack and the Flow E implementation passed hosted structural validation. The first bounded post-workstation closure-depth correction passed the full Flow source-contract and Python 3.10/3.11/3.12/3.13 matrix before being advanced onto PR #44. The second bounded correction, for E's exact unchanged `sol_h3/sparse.py` pre-H3 lazy-import self-effect, passed the same complete source-contract, Ruff, formatting, pytest, compileall, build, license and isolated-wheel matrix in run `35034482369`. Hosted CI does not establish SM120 arithmetic or media causality; a valid workstation E call remains the empirical gate.
