# First-High Mapped-Neighbor Arm M — Workstation Run Contract

This document defines the single allowed workstation CUDA call for **M**, the evidence-selected mapped-neighbor arm from `xmarre/ComfyUI-Sol-H3` design commit `0b8715faa0a82c730f5aaf0e44b1185e64291e49`.

R, W and E are complete evidence and must not be regenerated. M is diagnostic-only. It keeps the original diagonal threshold, tau, sink, restricted VDN K/V support, complement, dense layers, global/anchor providers, Q/K/V, adapters, schedule and Spectrum state, and adds only the VDN-mapped physical K64 self/neighbor blocks to the existing local Sol exact set. This run does not authorize a production fix.

## 1. Exact PR-overlay stack

Use ComfyUI Patcher. Do not manually replace, reset, rebase, clean or switch the custom-node repositories.

Apply the existing stacks through these top overlays:

- Sol-H3: PR #11 -> PR #12 -> **PR #13**, head `b95ad7b3bc7028465547b22fc61300cda53eb110`;
- VDN-H3-Plus: PR #15 -> PR #16 -> **PR #17**, head `6ca09ec37cd2dcfad02b790573a79b0462a662f3`;
- Flow-Aligned-Regenerate: PR #43 -> PR #44 -> **PR #45**, validated executable head `88af454146a6de79d6cb08a5539d47de9e756294`.

The older PRs remain independent evidence/contracts underneath M. Do not rerun their graph nodes.

## 2. Validation boundary

Hosted validation for the exact executable M stack is green:

- Sol #13: CI run `35046660423` — `test`, `native-interop`, and `windows-provenance` all succeeded;
- VDN #17: CI run `35046681576` — succeeded;
- Flow #45: CI run `35047000677` — source contracts and Python 3.10/3.11/3.12/3.13 Ruff, formatting, pytest, compileall, build, license and isolated-wheel validation all succeeded.

Two integration defects found during review were corrected before these final runs: Sol #13 preserves the underlying E request parser and treats M's diagnostic route label as history-equivalent to ordinary `vdn_local_sol`; VDN #17 keeps the M dispatcher inert for non-M/E requests. Flow's exact source allowlist pins those corrected bytes.

Hosted CI cannot compile and execute the real SM120 M selector specialization or judge media. The workstation call below remains the empirical gate.

## 3. Preserved R bundle

M consumes the same R bundle used by E:

```text
/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json
/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.pt
```

Before changing overlays, record both file hashes and keep them with the M evidence:

```bash
cd /home/toor/ComfyUI
sha256sum \
  output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json \
  output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.pt
```

The capture ID must remain `234ed062128e43ed8d5ec63e27517b22`; the high suffix must remain exactly:

```text
[0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]
```

## 4. Patcher byte checks

After Patcher builds the active trees, verify the critical reviewed bytes before starting the call.

```bash
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3
test "$(git hash-object sol_h3/first_high_mapped_neighbor_diagnostic.py)" = \
  a0f66f191b752ec01d2780c79525064c63fcda4b
test "$(git hash-object sol_h3/first_high_sol_local_diagnostic.py)" = \
  1c02baa4bae1c17c2ee2ba2e5081012737e83e01
test "$(git hash-object sol_h3/__init__.py)" = \
  0f7a50441f92b7c64ad028d7d44633b231d2d4f9
test "$(git hash-object sol_h3/sparse.py)" = \
  9f462591e3492d0b7af476c16024c79d1237505e

cd /home/toor/ComfyUI/custom_nodes/ComfyUI-VDN-H3-Plus
test "$(git hash-object vdn_h3/first_high_mapped_neighbor_diagnostic.py)" = \
  554733105daa04b297260da0b0b2bc8c4e65d56a
test "$(git hash-object vdn_h3/first_high_mapped_neighbor_bridge.py)" = \
  1a6e46f1fc127054b9d3030f6942834cda465aaa
test "$(git hash-object vdn_h3/first_high_sol_local_diagnostic.py)" = \
  8d0b0c9af1a20bc621635d9931376cca43a71fea
test "$(git hash-object __init__.py)" = \
  d0220b75e56d8a2b4794c913cd1ba251bb719c49

cd /home/toor/ComfyUI/custom_nodes/MiniMax-H3-Flow-Aligned-Regenerate
test "$(git hash-object h3_flow_regenerate/first_high_mapped_neighbor_diagnostic.py)" = \
  c83ea401343703c2411e89fb0121221991f4514a
test "$(git hash-object h3_flow_regenerate/__init__.py)" = \
  ecb1c8c41c23d00bd45d50e9c317e4d463d32f1c
test "$(git hash-object h3_flow_regenerate/first_high_sol_local_diagnostic.py)" = \
  138a27c8fbba0ddc9140b6301187a2b01ce0f1fb
test "$(git hash-object h3_flow_regenerate/first_high_mapped_neighbor_m_source_delta.json)" = \
  635f11c7e1010890c33406dc66dd581857cbab0e
test "$(git hash-object __init__.py)" = \
  2b322d96b2c6f9887f33abebbf61f58ce018624d

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

All successful `test` commands are silent. Any failure means the active Patcher-built runtime differs from the reviewed M stack; do not queue M.

## 5. Graph contract

Start a fresh ComfyUI process after Patcher finishes. Keep the same checkpoint/model, prompt/references, seed, target geometry, model loader, VAE, sampling schedule, Flow/Spectrum/Sol/VDN/DiffAid settings and attention backend represented by R/E.

Use the same graph wiring as the completed E run:

1. ordinary production model and companion nodes;
2. ordinary Progressive Target Input with the established production settings;
3. `MiniMax H3 Execution Contract Diagnostics` with `strict_provenance=true`, `capture_mib=512`;
4. its MODEL output into the diagnostic node. Under Flow PR #45 the visible label is **MiniMax H3 First-High Mapped-Neighbor Diagnostic M**;
5. select `h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json`;
6. send the M-patched MODEL into the same sampler path used by R/E.

Use **MiniMax H3 First-High Mapped-Neighbor M Report** with:

- `diagnostic` from the M node's diagnostic output;
- `trigger` from the sampler LATENT.

Decode/save the returned complete `first_high_model_raw` and `first_high_pre_guidance` clips with the same VAE used for E.

Do not queue R capture/save/replay, W comparison/report, E report, Untwist, Weighted/Mixed-Grid attention-measure research, external/reduced VDN sequence mode, T, C, or any old progressive-only diagnostic extractor.

## 6. Exactly one queue

Queue M exactly once from the fresh process.

A valid M execution must establish at minimum:

- exactly `1 logical / 1 actual / 0 forecast` high-stage H3 calls;
- zero learned-upscaler calls;
- exact preserved-R entry state and source/provenance gates;
- exactly 700 VDN/backend receipts;
- 528 `vdn_local_sol_mapped_neighbor_m` local calls for blocks 2-49 x groups 0-10;
- 22 `vdn_dense_warmup` local calls;
- 50 `vdn_global_native` calls;
- 100 `vdn_anchor_native` calls;
- exactly 528 mapped-route evidence records;
- every mapped descriptor derived from the live VDN gathered plan, bounded to the represented K64 block(s) plus immediate neighbors;
- no removal of an existing threshold/sink/ordinary selection;
- no restricted K/V expansion and no square-Q reconstruction;
- finite recorded original/added/effective selected-pair counts and exact-work increase;
- actual packaged `cute_sm120` provenance at Sana revision `2936c47637380842aaa4a4488fac5006cc542b70` on SM120;
- complete Core callback-x0 externalization and outer cleanup.

M's real CuTe selector specialization is compiled lazily on the workstation from the unchanged packaged SM120 mainloop with the bounded additive selector binding, then the ordinary binding is restored. A compile/runtime fallback is not a valid M result.

## 7. Evidence to return

Preserve and provide:

- complete startup-through-M console log;
- M report JSON;
- durable M `.json` and `.pt` files from:

```text
/home/toor/ComfyUI/output/h3_first_high_mapped_neighbor_m/
```

- complete decoded `first_high_model_raw` clip;
- complete decoded `first_high_pre_guidance` clip;
- the pre-run R JSON/PT SHA-256 values;
- runtime metrics if generated.

Do not immediately rerun if `execution_valid=false`; preserve the exact failure evidence first.

## 8. Decision gate

The only next decision is based on the completed M invariants plus complete media:

- invalid M: fix only the demonstrated M diagnostic/runtime defect before another M call;
- valid M but broken media: the mapped-neighbor intervention is not sufficient; do not promote it and do not automatically run T/C;
- valid M with clean media and retained acceleration: the design supports preparing a **separate** narrow production candidate consisting of a versioned VDN->Sol query-position capability plus additive SM120 forced-route mapping.

Even a clean M call does not authorize removal of the old ordinal-neighbor selections. That is a separate performance change requiring independent evidence.
