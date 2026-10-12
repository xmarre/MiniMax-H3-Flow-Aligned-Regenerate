# CUDA sparse-attention determinism diagnostic

This isolated runner targets the installed **comfy-kitchen 0.2.37**
`cuda.sol_attn_chunked` primitive. It loads no H3 model, changes no ComfyUI
configuration, and does not replace an installed kernel. Apply Flow PR #99
through ComfyUI Patcher to obtain the tools, then stop ComfyUI to free GPU
memory. Use the same Python environment that runs ComfyUI.

## Run

From the installed Flow node directory:

```bash
python tools/reproduce_sol_attn_determinism.py --repeats 32 --production-geometry --stages --output sol-determinism/report.json
```

`--stages` targets Linux/WSL and requires a CUDA toolkit with `nvcc` supporting the current GPU.
The runner downloads a small set of source files at commit
`be003b7c23c5b01328657955b8bc5d3f073d868e` (tag `v0.2.37`), verifies each
against the committed SHA256 manifest, and builds a **separate** diagnostic
library in the report directory. It neither installs that library nor writes
to comfy-kitchen. Download failure, hash mismatch, or compilation failure
stops the staged diagnostic. Without a toolkit, omit `--stages`; installed
primitive execution and workspace observations still run, but source-stage
attribution is unavailable. No manual repository operations are needed.

Return the complete `report.json` and terminal output. Exit 1 denotes observed
wrapper, frozen-core, or compiled-stage instability; exit 2 includes an
unqualified overflow/unsaturated fixture. Exit 0 means this finite protocol
did not find instability in those repetitions.
Exit 0 is not a proof of determinism or evidence that H3 continuation is fixed.
An error report preserves completed cases. Reports are rewritten after each
completed case and on an execution exception.

## Fixed inputs and controls

All tensors are CUDA, fused QKV is BF16, head dimension is 128, and the local
fixture generator seed is 119188. The fixture is generated once per case and
its complete input hashes are checked again after execution. Chunks have
4096 rows with a ragged last chunk. RoPE rotates 96 channels and Q/K weights
are explicitly BF16 before the original wrapper handles them.

| Case | Geometry | Input purpose |
| --- | --- | --- |
| Flat | 4096 tokens, 2 heads | Identical K scores; no overflowing token list expected |
| Overflow | 4096 tokens, 2 heads | Constant Q, alternating ±K; normalized Q/K magnitude 30; identity RoPE |
| Random | 4196 tokens, 2 heads | Nontrivial rotations and a ragged 100-row last chunk |
| Large overflow/random | 36445 tokens, 56 heads | Production dimensions and arguments; synthetic operands |

Large cases use `tau=1`, `topk_ratio=0`, `sink_blocks=[0,101]`,
`sink_q=[0,0]`, `rope_eps=1e-5`, `tail=true`, and no coarse/VSA branch.
Small cases use empty sink ranges. Every case runs `token_aug=256` and a
**primitive-only** `token_aug=0` control. Actual overflow is measured, not
assumed from the fixture name. The overflow CUDA regression fails if the
fixture does not produce reservation counts above 256.

Wrapper repetitions alternate two cold-stat calls and two calls using the
first cold call's **effective producer statistics** frozen exactly. Each pair
uses a new RoPE tensor identity followed by the same identity to exercise cold
and hot packed-RoPE caching. Comparisons stay within the cold/frozen-stat
cohort; cold and stale-stat execution need not have equal results. The runner
does not clear RNG state, caches, or the CUDA allocator.

One complete producer workspace is frozen immediately before the original
`sol_attn_core` call. Core repetitions copy those same bytes and the same
vscale into alternating fresh/reused allocations. This removes mutable
bootstrap/producer history from the core experiment. Undefined workspace
padding is never hashed or exported. Allocation history here means fresh
versus reused tensor allocations; it does not imply a flushed GPU hardware
cache. No deliberately conflicting cross-stream execution is performed.

## Boundaries and evidence

The original wrapper function body runs with a private extension proxy and
private DLPack wrapper; no module globals or model hooks are patched. The proxy
observes effective packed RoPE, cast Q/K weights, kmean/vscale, quantized Q/K/V,
block statistics and output at the real extension boundaries. The complete
fixed QKV tensor has its own SHA256 receipt.

Workspace observations use the original plan offsets, hash only initialized
logical extents, and mask unwritten route-list tails. Reports retain complete
hashes, exact equality, RMS/max differences, and the first changed route list
with both complete lists. Token reports distinguish:

- all unrouted **live** candidate-token counts from the group bitmap;
- candidates inside/above the score histogram window;
- reservations before capacity restriction;
- retained IDs/counts and overflow-group frequency.

Installed-core observations are taken after its combined launch. Their order
in the report is **not** a claim that a particular internal kernel differed
first. With `--stages`, the pinned source library launches the original
finish, block route, token group, tile list, histogram, admission, sorting,
tail merge, and exact functions individually on the same frozen workspace.
Admission comparisons sort a private copy of written IDs to distinguish changed
sets from reservation-order differences corrected by the normal sort kernel.
`source_first_difference` locates the first differing observed
stage in that compiled diagnostic; `source_earliest_divergent_stage` scans all
repetitions in operation order. `source_parity_verified` separately reports
whether all staged outputs match the first installed output bitwise. Do not
attribute a source-probe failure to the installed binary when parity is
unqualified. Compilers, synchronization, and observer timing can change
scheduling and whether a race manifests. Normalization, RoPE and per-token
quantization are fused in the producer. If their combined boundary first
differs, these observations do not distinguish operations inside that kernel.

`core_ms` measures the installed core using CUDA events, with no intermediate
observations inside its interval. It is a diagnostic timing distribution,
not a before/after performance result for a fix. `peak_allocated_bytes` counts
PyTorch allocations including fixtures, clones and observer storage; it is
not the production kernel's incremental memory overhead. Byte-exact reference
snapshots now stay on the tensors' device for the duration of each case.
Repeated observations compare all logical bytes on that device; only baseline
and differing tensors are transferred to host memory for full SHA256 receipts.
An unchanged tensor reuses its baseline SHA256 only after exact byte equality,
including signed-zero and NaN payload bits. Differences retain full hashes,
RMS/max witnesses and concrete changed route lists. This reduces repeated
host copies but raises diagnostic GPU allocation pressure and can itself
perturb allocator-sensitive races; compare multiple runs before attribution.
No full model tensors or replay files are written. Large cases require free
GPU and host memory. The source-stage probe still synchronizes between phases,
so utilization need not remain high throughout this diagnostic.

## Source-grounded hypothesis

Pinned upstream sources:
[chunked wrapper](https://github.com/Comfy-Org/comfy-kitchen/blob/be003b7c23c5b01328657955b8bc5d3f073d868e/comfy_kitchen/backends/cuda/__init__.py),
[workspace and stage orchestration](https://github.com/Comfy-Org/comfy-kitchen/blob/be003b7c23c5b01328657955b8bc5d3f073d868e/comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu),
[token kernels](https://github.com/Comfy-Org/comfy-kitchen/blob/be003b7c23c5b01328657955b8bc5d3f073d868e/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu),
[upstream tests](https://github.com/Comfy-Org/comfy-kitchen/blob/be003b7c23c5b01328657955b8bc5d3f073d868e/tests/test_sol_attn.py).

In `sol_attn_token.cu`, pass 1 builds an integer score histogram. Pass 2
admits whole bins until another bin would exceed the budget. This normally
bounds admissions despite concurrent `atomicAdd` reservations. The top bin
is clamped, however. When it alone exceeds the budget, the threshold becomes
`reference + 80` log2 units, and raw scores at or above that edge can still
reserve slots. The upstream surplus fixture explicitly targets this regime.
Sorting the admitted IDs can canonicalize order but cannot recover candidates
that lost a saturated reservation race. Excess tokens remain in the centroid
tail when they fail to obtain a slot; any correction must preserve their mass.

This source mechanism is conditional. It does not establish that an H3
invocation overflowed. Hash-only model observations do not provide complete
replay operands. The synthetic runner has no claim to reproduce owner-model
quality or the visible continuation defect.

## Validation status and interpretation

CPU tests cover source-model thresholds, exact BF16 hashing, logical workspace
bounds, route-tail masking, comparison witnesses, private wrapper isolation,
candidate/reservation distinction, and source integrity rejection. CUDA tests
are explicitly skipped without a device. Neither CPU tests nor CPU GitHub
Actions establish CUDA compilation, repeated-execution stability, performance,
or output quality.

Four conclusions require separate evidence: a general primitive race;
occurrence with captured effective H3 operands; a corrected upstream primitive;
and restored native A/A′ plus audiovisual continuation quality. No numerical
production correction is included. When exact model replay becomes necessary,
capture complete effective operands at the actual primitive boundary with a
bounded default-off observer; the sampler witness archive is insufficient.
