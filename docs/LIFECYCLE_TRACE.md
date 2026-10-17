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
| `model.load`, `model.unload`, `model.use_more_vram` | `LoadedModel`; records model class and bytes moved. Linux `model.load` also captures process major/minor faults and host-wide VM swap counters before/after weight restoration. |
| `text_encoder.*` | `CLIP.encode_from_tokens[_scheduled]` |
| `vae.*` | `VAE.decode/encode[_tiled]` |
| `sampler.cfg_guider_sample` | `CFGGuider.sample` (also used by Continuum chunks) |
| `python.gc` | `gc.callbacks` (every collection, including explicit `gc.collect()`) |
| `flow.<event>` instants | All Flow metrics events; only `low_stage_wall`, `handoff_probe_wall`, `handoff_transfer_wall`, and `high_stage_wall` probe memory and reset stage peaks |

Each span stores monotonic start/end times, its parent span, allocator
allocated/reserved bytes, driver-free memory and whether the CUDA stream still
had queued work at each boundary. The trace never synchronizes CUDA, so host
spans around asynchronous GPU work measure host time; the stream-idle flags
and NVML samples help distinguish device activity. A stream query observes only
the caller's current stream, not work queued on other streams. NVML utilization
has driver-side sampling windows (often around a second), so neither signal
provides sub-second, per-kernel idle attribution.

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

* **Exclusive time by subsystem** sums only time not covered by a nested span.
  Rows partition the prompt wall only for strictly nested or disjoint same-thread
  spans. Overlapping siblings (for example concurrent async node work or GC)
  can be double-counted or over-subtracted. Examine the trace intervals before
  interpreting these rows as an exact wall-time partition.
  `between_nodes` is executor time outside any node; `node_body_other` is node
  time outside any traced call (for example prompt compilation or Python
  orchestration).
* **Model load / offload** lists every full or partial residency change with
  bytes moved and effective GiB/s, and whether it happened inside a sampler
  (memory admission) or elsewhere. On Linux/WSL, raw `model.load.fields`
  include `host_pressure_before`, `host_pressure_after` and
  `host_pressure_delta`: process-wide major/minor page faults and host-wide
  `pswpin`, `pswpout`, `pgmajfault` (4-KiB pages or kernel-defined units).
  `memavailable_kib` and `swapfree_kib` are endpoint snapshots, **not**
  cumulative deltas. These counters correlate a slow H2D load with paging,
  but system-wide changes cannot prove this process caused the activity.
  They do not require CUDA synchronization or change memory policy.
* **Flow stages** turns Flow's `low_stage_wall`, `handoff_probe_wall`,
  `handoff_transfer_wall` and `high_stage_wall` events into stage rows with
  the interval's peak allocated/reserved bytes (peak counters are reset at
  each stage end, so a peak can also cover hand-off work before the stage),
  allocator state at the stage end, time spent in model residency changes,
  allocator trims and GC inside the stage, and NVML utilization. The VDN
  admission `required_mib` receipt is a **free/reclaimable-memory target**;
  peak `allocated_mib` and `reserved_mib` are **memory occupancy**. They are
  not directly comparable or interchangeable. Evaluate admission against the
  contemporaneous `free_before_mib`, `free_after_mib`, Core's reusable
  cached allocation accounting, and the trace's driver-free readings. Do not
  reduce VDN's retained headroom using occupancy peaks alone.
* **GPU-idle intervals** are NVML runs below 10% utilization lasting at least
  one second, annotated with the spans and Flow events they overlap.

Extensions that run their own work around node execution (for example
periodic `gc.collect()` plus native heap trimming) appear as `python.gc` spans
and as `between_nodes` time. Native heap trimming itself cannot be wrapped and
is only visible through those gaps.
