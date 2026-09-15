# First-High Sol-Local Experiment E — Production Run Contract

This document defines the workstation procedure for the single allowed next CUDA experiment in the MiniMax-H3 first-high artifact investigation: **Experiment E** from the authoritative Sol-local design at ComfyUI-Sol-H3 commit `0b8715faa0a82c730f5aaf0e44b1185e64291e49`.

Experiment E is diagnostic-only. It replays the existing experiment-R first-high state, executes exactly one real high-stage H3 call, keeps VDN restricted local support plus its complement, and changes only the final local Sol-Attn selection so every valid local K block is exact. It does not regenerate R, rerun W, run a full diffusion trajectory, or authorize a production fix.

## 1. Exact reviewed implementation and PR overlays

Run E through the existing **ComfyUI Patcher PR-overlay stack**. Do not replace the custom-node checkouts manually.

| Repository | E PR overlay | Reviewed executable source | Purpose |
| --- | --- | --- | --- |
| `xmarre/ComfyUI-Sol-H3` | PR #12, stacked on PR #11 | `2eedd341a0f412803afc92702996c07e1702edb7` | E Sol all-selected path and numerical witnesses |
| `xmarre/ComfyUI-VDN-H3-Plus` | PR #16, stacked on PR #15 | `10bf368bee38b9f942961e6d8c374b5af6c4bc16` | E VDN overlay and generated-loader bridge |
| `xmarre/MiniMax-H3-Flow-Aligned-Regenerate` | PR #44, stacked on PR #43 | `7ea52470ab5ac82b12d58bcb87db82d715b8005c` | E one-call replay, source/provenance gate, report and durable evidence |

PRs #11, #15 and #43 remain the preserved W evidence bases. E is the next overlay layer; it does not rewrite or repurpose those PRs.

The Flow source manifest pins the exact reviewed executable source bytes. The four installed ComfyUI Core files are verified by their **loaded file bytes at runtime**, not by a Git branch or tag:

- `comfy.model_sampling`: Git blob `4a55655207115a150d5254c5b364b03a6c256627`
- `comfy.latent_formats`: Git blob `8d8ff6c5f179da8dcdeba854f220fb657de2af1c`
- `comfy.model_patcher`: Git blob `08e69fae149692fed60d886890cc0a8eb80bdf26`
- `comfy.k_diffusion.sampling`: Git blob `4a638008a3df3a2f167e9bf343a258a0f5655ede`

Do not substitute a nearby public ComfyUI revision merely because its Git label looks compatible. The E source gate must accept the actual installed files.

## 2. Preserved R evidence

The E run consumes the existing R bundle and must not overwrite or regenerate it:

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

The R capture ID must remain exactly:

```text
234ed062128e43ed8d5ec63e27517b22
```

## 3. ComfyUI Patcher / PR-overlay preparation

Stop ComfyUI completely before rebuilding the PR overlay stack.

Keep the already-established production/diagnostic stack and add E on top in this order:

- **Sol-H3:** existing stack through PR #11, then PR #12;
- **VDN-H3-Plus:** existing stack through PR #15, then PR #16;
- **Flow-Aligned-Regenerate:** existing R/W stack through PR #43, then PR #44.

Do not remove PR #11/#15/#43 from the Patcher stack when adding E. PR #12/#16/#44 are stacked deltas relative to those preserved W bases. Let ComfyUI Patcher construct the working tree from the PR overlays; do not `git switch`, rebase, reset, clean, or manually replace the custom-node repositories for this run.

After the Patcher has applied the overlay stack, verify the critical loaded-source bytes directly:

```bash
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3
test "$(git hash-object sol_h3/first_high_sol_local_diagnostic.py)" = \
  79b3013aa936a836fc4e5c5cf3e80f8ac06a49ac

test "$(git hash-object sol_h3/__init__.py)" = \
  c4184d48f53a1b0703816c371f616e538acde5a7

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

test "$(git hash-object h3_flow_regenerate/first_high_sol_local_e_source_delta.json)" = \
  8f5bf60a6e0ca84ba55f38fa02045ae399bbc296

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

The ComfyUI checks deliberately hash the installed files rather than compare the checkout's branch/tag/commit name. A successful `test` command is silent and exits zero. Any failure means the active Patcher-built runtime does not contain the reviewed E bytes; do not queue E.

## 4. Process and graph contract

Start a **fresh ComfyUI process** after Patcher has rebuilt the exact overlay stack and the byte checks above pass. Do not reuse a process that previously executed R, W, or another diagnostic arm.

Use the same production model/checkpoint, prompt, references, seed, target geometry, complete sampler schedule, Flow configuration, Spectrum configuration, Sol-H3 configuration, VDN configuration, DiffAid configuration, model loader and attention backend configuration represented by the preserved R capture.

The model path must be composed in this order:

1. ordinary production model and companion nodes;
2. ordinary progressive Target Input configuration with `transfer_mode=learned_3d` and `exact_prefix_mode=fallback`;
3. `MiniMax H3 Execution Contract Diagnostics`:
   - `capture_mib = 512` unless the preserved R graph used another validated value;
   - `strict_provenance = true`;
4. the resulting MODEL into `MiniMax H3 First-High Sol-Local Diagnostic E`;
5. select the explicit preserved R manifest `h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json`;
6. feed the E-patched MODEL to the same sampler path used by R.

Do **not** place any of these on the queued E path:

- R capture/save node;
- R replay node;
- W operator-comparison node or W report node;
- Untwist / clock-trial state;
- weighted or Mixed-Grid attention-measure research mode;
- external/reduced VDN sequence mode;
- progressive-only checkpoint/guidance diagnostic extractors that are not required by E.

The normal high suffix must remain exactly:

```text
[0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]
```

E internally stops after the callback for the **first** real high-stage Euler model call while allowing the installed Core outer-sample externalization and cleanup path to complete.

## 5. E report and media wiring

Add `MiniMax H3 First-High Sol-Local E Report`.

Connect:

- `diagnostic` ← the `H3_FLOW_FIRST_HIGH_SOL_LOCAL_E` output from the E node;
- `trigger` ← the sampler LATENT produced by the E-patched model path.

The report node returns:

1. `report` — the E JSON report;
2. `first_high_model_raw` — decode-ready target-grid video LATENT from the underlying first-high model output;
3. `first_high_pre_guidance` — decode-ready target-grid video LATENT immediately before Flow guidance.

Decode and save **both complete clips** with the same VAE used for the controlled baseline. Do not judge E from selected frames only.

The E node also writes durable machine evidence under:

```text
/home/toor/ComfyUI/output/h3_first_high_sol_local_e/
```

Preserve the generated `.json` and `.pt` files.

## 6. Memory contract

E defaults to a 2 GiB additional diagnostic budget for CUDA and 2 GiB for host memory. The diagnostic performs its own projected-allocation and free-memory preflight before the model call.

Optional overrides exist only for an explicitly reviewed need:

```text
H3_FIRST_HIGH_E_CUDA_BUDGET_GIB
H3_FIRST_HIGH_E_CPU_BUDGET_GIB
```

Do not lower a budget merely to bypass a preflight failure. A budget failure is an invalid E attempt until the reported projected/retained allocation is understood.

## 7. Queue exactly once

From the fresh process, queue the E graph exactly once. Do not queue another generation after a failed or successful E attempt until the emitted report and log are inspected.

A valid E execution must prove, at minimum:

- `1 logical / 1 actual / 0 forecast` high-stage H3 calls;
- zero learned-upscaler calls;
- exact preserved-R sampler/H3 entry state hashes;
- exactly 700 VDN attention subcalls;
- 528 returned `vdn_local_sol_all_selected_e` calls;
- 22 native dense-layer local calls;
- 50 native global calls;
- 100 native anchor calls;
- no ordinary production sparse, external Mixed-Grid, or square-expanded local Sol path;
- the exact packaged verified SM120 Sol-Attn backend;
- exact Q/K/V preservation for witness groups block 2 / groups 0, 2 and 10;
- complete all-selected route traces;
- valid Core callback-x0 externalization and outer-sample cleanup;
- exact reviewed source/provenance gates.

Do not interpret decoded E media when `execution_valid` is false.

## 8. Evidence to preserve

After the single queue, preserve all of the following before restarting or changing code:

- complete ComfyUI console log from startup through E completion/failure;
- E report JSON from `MiniMax H3 First-High Sol-Local E Report`;
- durable E `.json` and `.pt` evidence files from `output/h3_first_high_sol_local_e/`;
- full decoded `first_high_model_raw` clip;
- full decoded `first_high_pre_guidance` clip;
- the two pre-run `sha256sum` values for the R bundle;
- any generated runtime metrics file associated with the queue.

If E fails before report emission, preserve the full log and any durable files that were successfully written; do not rerun immediately.

## 9. Decision gate after E

No follow-up H3 experiment is authorized until the E report, operator witnesses and both decoded clips have been reviewed together.

The immediate interpretation rules are:

- `execution_valid=false`: diagnose the E instrumentation/runtime defect and rerun **only E** after a causally supported correction;
- `execution_valid=true` but `arithmetic_conformant=false`: localize descriptor/stride/mask/normalization arithmetic from the saved witness evidence; do not run another H3 experiment;
- valid E with broken media despite a passing all-selected arithmetic witness: extend only the bounded E witness needed to explain the discrepancy;
- valid E with clean media: use the authoritative design decision table to select the single permitted next localization experiment from the sparse preparation/selector/frozen-reference evidence;
- clean E alone does not authorize permanently dense local attention and does not establish a production fix.

Experiments M, T and C remain conditional. None should be implemented or run merely because E completed.

## 10. Structural validation already established

Before this production contract was written, the exact E Flow/Sol/VDN source set passed hosted structural validation including:

- the normal Sol-H3 CPU contract suite;
- the normal Flow Python 3.10/3.11/3.12/3.13, Ruff, pytest, build, wheel and source-contract lanes;
- the VDN pinned-Comfy/oracle suite, current-Comfy smoke and legacy workflow migration;
- an exact cross-repository source-byte check against the E manifest;
- combined Sol-E and VDN-E targeted tests;
- a generated-custom-node loader smoke that resolves the actual VDN-to-Sol E bridge to the loaded generated Sol companion.

These checks establish structural/source compatibility only. They do not establish SM120 kernel correctness, E media quality, or the causal production fix. The single workstation E run defined above remains the empirical gate.
