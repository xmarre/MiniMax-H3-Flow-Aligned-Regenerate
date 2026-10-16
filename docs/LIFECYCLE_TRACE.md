# Prompt lifecycle trace

`H3_LIFECYCLE_TRACE` records where a complete ComfyUI prompt spends wall time
outside transformer arithmetic: text encoding, VAE work, model load and
offload traffic, memory admission, CUDA cache trims, Python garbage
collection, node boundaries and GPU-idle intervals. It is off unless the
environment variable is set before ComfyUI starts.

```bash
export H3_LIFECYCLE_TRACE=1                 # <ComfyUI output>/h3_lifecycle_trace/
# or: export H3_LIFECYCLE_TRACE=/path/to/dir
python main.py ...
python custom_nodes/<this package>/tools/analyze_lifecycle_trace.py \
    <output>/h3_lifecycle_trace/h3_lifecycle_<time>_<prompt>.json
```

One JSON file is written when each prompt finishes.

## What is recorded

| Span | Source |
|---|---|
| `comfy.prompt`, `comfy.node` | `PromptExecutor.execute_async`, `execution.execute` |
| `model_management.load_models_gpu`, `.free_memory`, `.soft_empty_cache` | `comfy.model_management` |
| `model.load`, `model.unload`, `model.use_more_vram` | `LoadedModel`; records model class and bytes moved |
| `text_encoder.*` | `CLIP.encode_from_tokens[_scheduled]` |
| `vae.*` | `VAE.decode/encode[_tiled]` |
| `sampler.cfg_guider_sample` | `CFGGuider.sample` (also used by Continuum chunks) |
| `python.gc` | `gc.callbacks` (every collection, including explicit `gc.collect()`) |
| `flow.<event>` instants | Flow metrics events; stage-end `*_wall` events also probe memory and reset peaks |

Each span stores monotonic start/end times, its parent span, allocator
allocated/reserved bytes, driver-free memory and whether the CUDA stream still
had queued work at each boundary. The trace never synchronizes CUDA, so host
spans around asynchronous GPU work measure host time; the stream-idle flags
and NVML samples show whether the device was actually busy.

## Optional controls

| Variable | Default | Effect |
|---|---|---|
| `H3_LIFECYCLE_TRACE_NVML_MS` | `100` | NVML utilization/power/memory sampling period; `0` disables |
| `H3_LIFECYCLE_TRACE_NVML_INDEX` | `0` | NVML device index |
| `H3_LIFECYCLE_TRACE_STAGE_PEAK` | `1` | reset CUDA peak counters at each sampler start and each Flow stage end so every sampler/stage reports its own peak |

NVML is loaded directly through `libnvidia-ml` (including the WSL path); no
Python package is required. NVML utilization is the fraction of its own sample
window in which a kernel ran, so it resolves intervals of roughly a second, not
individual kernels.

## Reading the report

* **Exclusive time by subsystem** sums only time not covered by a nested span,
  so the rows add up to the prompt wall without double counting.
  `between_nodes` is executor time outside any node; `node_body_other` is node
  time outside any traced call (for example prompt compilation or Python
  orchestration).
* **Model load / offload** lists every full or partial residency change with
  bytes moved and effective GiB/s, and whether it happened inside a sampler
  (memory admission) or elsewhere.
* **Flow stages** turns Flow's `low_stage_wall`, `handoff_probe_wall`,
  `handoff_transfer_wall` and `high_stage_wall` events into stage rows with
  the stage's own peak allocated/reserved bytes (peak counters are reset at
  each stage end; a stage's peak therefore also covers the hand-off work just
  before it), allocator state at the stage end, time spent in model
  residency changes, allocator trims and GC inside the stage, and NVML
  utilization. Compare the peak with VDN's admission `required_mib` receipt
  for the same stage.
* **GPU-idle intervals** are NVML runs below 10% utilization lasting at least
  one second, annotated with the spans and Flow events they overlap.

Extensions that run their own work around node execution (for example
periodic `gc.collect()` plus native heap trimming) appear as `python.gc` spans
and as `between_nodes` time. Native heap trimming itself cannot be wrapped and
is only visible through those gaps.
