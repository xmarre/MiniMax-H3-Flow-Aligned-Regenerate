"""Opt-in, low-overhead lifecycle trace for complete H3 generation prompts.

The trace answers where a prompt's wall time goes *outside* transformer
arithmetic: text-encoder and VAE work, model load/offload traffic, memory
admission, CUDA cache trims, Python garbage collection and the gaps between
ComfyUI nodes. It records monotonic host timestamps only and never
synchronizes CUDA. Span boundary probes query the current CUDA stream
without waiting for completion. That result does not establish that the whole
GPU was idle (other streams may still have work), nor does it classify the
intervening host interval. An optional NVML sampler thread records coarse
device utilization, power and driver-free memory to correlate long host
intervals with device activity; NVML's utilization sampling window is not
100-ms kernel-level attribution.

Enable it before ComfyUI starts::

    H3_LIFECYCLE_TRACE=1            # write to <output>/h3_lifecycle_trace/
    H3_LIFECYCLE_TRACE=/some/dir    # write to an explicit directory

Optional controls:

``H3_LIFECYCLE_TRACE_NVML_MS``
    NVML sampling period in milliseconds (default 100, ``0`` disables).
``H3_LIFECYCLE_TRACE_NVML_INDEX``
    NVML device index (default 0).
``H3_LIFECYCLE_TRACE_STAGE_PEAK``
    ``1`` (default) resets the CUDA allocator's peak counters when a sampler
    span starts and after each Flow stage-end event, so every sampler and
    Flow stage reports its own peak allocated/reserved bytes. ``0`` leaves
    peak counters untouched.

One JSON file is written when each prompt finishes. ``tools/analyze_lifecycle_trace.py``
turns it into a bottleneck report. Nothing is installed unless the variable is
set, and installation never alters arguments, return values or exceptions of
the wrapped functions.
"""

from __future__ import annotations

import contextlib
import contextvars
import ctypes
import functools
import gc
import inspect
import itertools
import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "h3-lifecycle-trace-v1"
ENV_TRACE = "H3_LIFECYCLE_TRACE"
ENV_NVML_MS = "H3_LIFECYCLE_TRACE_NVML_MS"
ENV_NVML_INDEX = "H3_LIFECYCLE_TRACE_NVML_INDEX"
ENV_STAGE_PEAK = "H3_LIFECYCLE_TRACE_STAGE_PEAK"
DEFAULT_MAX_SPANS = 250_000
DEFAULT_MAX_NVML_SAMPLES = 400_000
_TRUE = {"1", "true", "yes", "on"}
_MIB = 1 << 20
# Component walls (e.g. handoff_learned_upscale_wall) may be nested within
# these stage walls and must never reset their enclosing stage's peak.
STAGE_WALL_EVENTS = frozenset(
    {
        "low_stage_wall",
        "handoff_probe_wall",
        "handoff_transfer_wall",
        "high_stage_wall",
    }
)

_log = logging.getLogger("h3_flow_regenerate.lifecycle_trace")
_parent_span: contextvars.ContextVar[int | None] = contextvars.ContextVar("h3_lifecycle_parent_span", default=None)


def _torch_cuda():
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    cuda = getattr(torch, "cuda", None)
    try:
        if cuda is None or not cuda.is_available() or not cuda.is_initialized():
            return None
    except Exception:
        return None
    return cuda


def cuda_memory_probe() -> dict[str, Any] | None:
    """Return allocator and driver memory without initializing or synchronizing CUDA."""
    cuda = _torch_cuda()
    if cuda is None:
        return None
    probe: dict[str, Any] = {}
    try:
        probe["allocated"] = int(cuda.memory_allocated())
        probe["reserved"] = int(cuda.memory_reserved())
        probe["peak_allocated"] = int(cuda.max_memory_allocated())
        probe["peak_reserved"] = int(cuda.max_memory_reserved())
    except Exception as exc:  # allocator backends differ in supported statistics
        probe["allocator_error"] = type(exc).__name__
    try:
        free, total = cuda.mem_get_info()
        probe["driver_free"] = int(free)
        probe["driver_total"] = int(total)
    except Exception as exc:
        probe["driver_error"] = type(exc).__name__
    with contextlib.suppress(Exception):
        probe["stream_idle"] = bool(cuda.current_stream().query())
    return probe


def _reset_cuda_peaks() -> bool:
    cuda = _torch_cuda()
    if cuda is None:
        return False
    try:
        cuda.reset_peak_memory_stats()
    except Exception:
        return False
    return True


def host_pressure_probe() -> dict[str, int] | None:
    """Read bounded Linux paging evidence without a CUDA call or GPU sync.

    Process faults are process-wide; /proc/vmstat swap and fault counters are
    system-wide. Neither metric on its own proves that *this model* paged.
    """
    if sys.platform != "linux":
        return None
    stats: dict[str, int] = {}
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        stats["process_minor_faults"] = int(usage.ru_minflt)
        stats["process_major_faults"] = int(usage.ru_majflt)
    except (ImportError, OSError, ValueError):
        pass
    try:
        with open("/proc/vmstat", encoding="ascii") as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                if key in {"pswpin", "pswpout", "pgmajfault"}:
                    stats["system_" + key] = int(value.strip())
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in {"MemAvailable", "SwapFree"}:
                    stats[key.lower() + "_kib"] = int(value.strip().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return stats or None


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _load_nvml():
    candidates = ["libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1", "nvml.dll"]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class NvmlReader:
    """Minimal ctypes NVML reader; no Python NVML package is required."""

    def __init__(self, index: int = 0) -> None:
        lib = _load_nvml()
        if lib is None:
            raise OSError("NVML library not found")
        if lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit_v2 failed")
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)) != 0:
            raise OSError(f"NVML device {index} unavailable")
        self._lib = lib
        self._handle = handle
        self.index = index

    def read(self) -> tuple[int, int, int, int, int]:
        util = _NvmlUtilization()
        power = ctypes.c_uint()
        memory = _NvmlMemory()
        clock = ctypes.c_uint()
        lib, handle = self._lib, self._handle
        util_ok = lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)) == 0
        power_ok = lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power)) == 0
        memory_ok = lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0
        clock_ok = lib.nvmlDeviceGetClockInfo(handle, ctypes.c_uint(1), ctypes.byref(clock)) == 0
        return (
            int(util.gpu) if util_ok else -1,
            int(util.memory) if util_ok else -1,
            int(power.value) if power_ok else -1,
            int(memory.used) if memory_ok else -1,
            int(clock.value) if clock_ok else -1,
        )


class NvmlSampler:
    """Background sampler of (time, gpu util %, memory util %, power mW, used bytes, SM MHz)."""

    def __init__(
        self,
        read: Callable[[], tuple[int, int, int, int, int]],
        interval_s: float,
        clock: Callable[[], int],
        max_samples: int = DEFAULT_MAX_NVML_SAMPLES,
    ) -> None:
        self._read = read
        self._interval_s = float(interval_s)
        self._clock = clock
        self._max_samples = int(max_samples)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.samples: deque[tuple[int, int, int, int, int, int]] = deque(maxlen=self._max_samples)
        self.dropped = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="h3-lifecycle-nvml", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def sample_once(self) -> None:
        try:
            values = self._read()
        except Exception:
            self.errors += 1
            return
        with self._lock:
            if len(self.samples) >= self._max_samples:
                self.dropped += 1
            self.samples.append((self._clock(), *values))

    def peek(self) -> list[tuple[int, int, int, int, int, int]]:
        with self._lock:
            return list(self.samples)

    def drain(self) -> list[tuple[int, int, int, int, int, int]]:
        with self._lock:
            samples = list(self.samples)
            self.samples.clear()
        return samples

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self._interval_s)


class LifecycleTrace:
    """Bounded in-memory span recorder with explicit parent/child ownership."""

    def __init__(
        self,
        *,
        clock: Callable[[], int] = time.perf_counter_ns,
        wall_clock: Callable[[], int] = time.time_ns,
        memory_probe: Callable[[], dict[str, Any] | None] = cuda_memory_probe,
        reset_peaks: Callable[[], bool] = _reset_cuda_peaks,
        stage_peaks: bool = True,
        max_spans: int = DEFAULT_MAX_SPANS,
    ) -> None:
        self._clock = clock
        self._wall_clock = wall_clock
        self._memory_probe = memory_probe
        self._reset_peaks = reset_peaks
        self.stage_peaks = bool(stage_peaks)
        self._max_spans = int(max_spans)
        # Re-entrant: a GC callback can fire while this thread already holds the lock.
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._gc_started: dict[int, int] = {}
        self.nvml: NvmlSampler | None = None
        self.nvml_status = "disabled"
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.dropped_spans = 0
        self.anchor_clock_ns = self._clock()
        self.anchor_wall_ns = self._wall_clock()

    def now(self) -> int:
        return self._clock()

    def _append(self, record: dict[str, Any]) -> None:
        with self._lock:
            if len(self.spans) >= self._max_spans:
                self.dropped_spans += 1
                return
            self.spans.append(record)

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        subsystem: str,
        *,
        fields: dict[str, Any] | None = None,
        memory: bool = True,
        reset_peaks: bool = False,
    ) -> Iterator[dict[str, Any]]:
        span_id = next(self._ids)
        parent = _parent_span.get()
        token = _parent_span.set(span_id)
        record: dict[str, Any] = {
            "id": span_id,
            "parent": parent,
            "name": name,
            "subsystem": subsystem,
            "thread": threading.get_ident(),
            "fields": dict(fields or {}),
        }
        if reset_peaks and self.stage_peaks:
            record["peaks_reset"] = bool(self._reset_peaks())
        if memory:
            record["memory_start"] = self._safe_probe()
        record["start_ns"] = self._clock()
        try:
            yield record["fields"]
        except BaseException as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["end_ns"] = self._clock()
            _parent_span.reset(token)
            if memory:
                record["memory_end"] = self._safe_probe()
            self._append(record)

    def instant(
        self,
        name: str,
        subsystem: str,
        fields: dict[str, Any] | None = None,
        *,
        memory: bool = False,
        reset_peaks: bool = False,
    ) -> None:
        """Record a point event; optionally probe memory, then reset peaks for the next interval."""
        now = self._clock()
        record = {
            "id": next(self._ids),
            "parent": _parent_span.get(),
            "name": name,
            "subsystem": subsystem,
            "thread": threading.get_ident(),
            "fields": dict(fields or {}),
            "start_ns": now,
            "end_ns": now,
            "instant": True,
        }
        if memory:
            record["memory_end"] = self._safe_probe()
        if reset_peaks and self.stage_peaks:
            record["peaks_reset"] = bool(self._reset_peaks())
        self._append(record)

    def _safe_probe(self) -> dict[str, Any] | None:
        try:
            return self._memory_probe()
        except Exception as exc:
            return {"probe_error": type(exc).__name__}

    def gc_callback(self, phase: str, info: dict[str, Any]) -> None:
        thread = threading.get_ident()
        if phase == "start":
            self._gc_started[thread] = self._clock()
            return
        started = self._gc_started.pop(thread, None)
        if started is None:
            return
        self._append(
            {
                "id": next(self._ids),
                "parent": _parent_span.get(),
                "name": "python.gc",
                "subsystem": "python_gc",
                "thread": thread,
                "fields": {
                    "generation": int(info.get("generation", -1)),
                    "collected": int(info.get("collected", 0)),
                    "uncollectable": int(info.get("uncollectable", 0)),
                },
                "start_ns": started,
                "end_ns": self._clock(),
            }
        )

    def snapshot(self, *, prompt_id: Any = None, reset: bool = False) -> dict[str, Any]:
        with self._lock:
            spans = list(self.spans)
            dropped = self.dropped_spans
            anchor_clock, anchor_wall = self.anchor_clock_ns, self.anchor_wall_ns
            if reset:
                self._reset_buffer()
        samples = [] if self.nvml is None else self.nvml.drain() if reset else self.nvml.peek()
        return {
            "schema": TRACE_SCHEMA,
            "prompt_id": None if prompt_id is None else str(prompt_id),
            "clock": "perf_counter_ns",
            "anchor_clock_ns": anchor_clock,
            "anchor_wall_ns": anchor_wall,
            "pid": os.getpid(),
            "dropped_spans": dropped,
            "stage_peaks": self.stage_peaks,
            "nvml": {
                "status": self.nvml_status,
                "columns": ["t_ns", "gpu_util_pct", "mem_util_pct", "power_mw", "used_bytes", "sm_mhz"],
                "samples": samples,
                "dropped": 0 if self.nvml is None else self.nvml.dropped,
                "errors": 0 if self.nvml is None else self.nvml.errors,
            },
            "spans": spans,
        }


def write_trace_json(trace: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(trace["anchor_wall_ns"] / 1e9))
    prompt = (trace.get("prompt_id") or "prompt").replace(os.sep, "_")[:48]
    target = directory / f"h3_lifecycle_{stamp}_{prompt}.json"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            json.dump(trace, handle, separators=(",", ":"), default=str)
            temporary = Path(handle.name)
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ComfyUI integration


def _patcher_identity(patcher: Any) -> dict[str, Any]:
    inner = getattr(patcher, "model", None)
    identity: dict[str, Any] = {
        "model_class": type(inner).__name__ if inner is not None else type(patcher).__name__,
        "patcher_id": id(patcher),
    }
    for key, method in (("loaded_bytes", "loaded_size"), ("model_bytes", "model_size")):
        with contextlib.suppress(Exception):
            identity[key] = int(getattr(patcher, method)())
    return identity


def _loaded_bytes(patcher: Any) -> int | None:
    try:
        return int(patcher.loaded_size())
    except Exception:
        return None


@functools.lru_cache(maxsize=64)
def _signature(function: Callable) -> inspect.Signature | None:
    try:
        return inspect.signature(function)
    except (TypeError, ValueError):
        return None


def _bound_arguments(function: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    signature = _signature(function)
    if signature is None:
        return {}
    try:
        return dict(signature.bind_partial(*args, **kwargs).arguments)
    except TypeError:
        return {}


def _scalar_fields(fields: dict[str, Any], limit: int = 24) -> dict[str, Any]:
    scalars: dict[str, Any] = {}
    for key, value in fields.items():
        if len(scalars) >= limit:
            break
        if isinstance(value, (bool, int, float)) or value is None:
            scalars[str(key)] = value
        elif isinstance(value, str):
            scalars[str(key)] = value[:160]
    return scalars


class ComfyLifecycleHooks:
    """Install and remove wrappers on ComfyUI lifecycle functions."""

    def __init__(self, trace: LifecycleTrace, output_directory: Callable[[], Path]) -> None:
        self.trace = trace
        self._output_directory = output_directory
        self._restore: list[tuple[Any, str, Any]] = []
        self.installed: list[str] = []
        self.written: list[Path] = []

    def _replace(self, owner: Any, attribute: str, factory: Callable[[Callable], Callable], label: str) -> None:
        original = getattr(owner, attribute, None)
        if original is None or getattr(original, "__h3_lifecycle_wrapped__", False):
            return
        wrapped = factory(original)
        functools.update_wrapper(wrapped, original)
        wrapped.__h3_lifecycle_wrapped__ = True
        setattr(owner, attribute, wrapped)
        self._restore.append((owner, attribute, original))
        self.installed.append(label)

    def _simple(self, name: str, subsystem: str, describe: Callable[..., dict[str, Any]] | None = None):
        trace = self.trace

        def factory(original: Callable) -> Callable:
            def wrapped(*args, **kwargs):
                fields = {}
                if describe is not None:
                    try:
                        fields = describe(original, args, kwargs)
                    except Exception:
                        fields = {}
                with trace.span(name, subsystem, fields=fields):
                    return original(*args, **kwargs)

            return wrapped

        return factory

    def install(self, modules: dict[str, Any] | None = None) -> list[str]:
        modules = modules if modules is not None else sys.modules
        mm = modules.get("comfy.model_management")
        if mm is not None:
            self._install_model_management(mm)
        sd = modules.get("comfy.sd")
        if sd is not None:
            self._install_sd(sd)
        samplers = modules.get("comfy.samplers")
        if samplers is not None and hasattr(samplers, "CFGGuider"):
            self._install_sampler(samplers.CFGGuider)
        execution = modules.get("execution")
        if execution is not None:
            self._install_execution(execution)
        try:
            from .metrics import H3FlowMetrics
        except ImportError:
            H3FlowMetrics = None
        if H3FlowMetrics is not None:
            self._install_flow_metrics(H3FlowMetrics)
        if self.trace.gc_callback not in gc.callbacks:
            gc.callbacks.append(self.trace.gc_callback)
            self.installed.append("gc.callbacks")
        return list(self.installed)

    def uninstall(self) -> None:
        for owner, attribute, original in reversed(self._restore):
            setattr(owner, attribute, original)
        self._restore.clear()
        with contextlib.suppress(ValueError):
            gc.callbacks.remove(self.trace.gc_callback)
        self.installed.clear()

    def _install_model_management(self, mm: Any) -> None:
        trace = self.trace

        def describe_load_models(original, args, kwargs):
            bound = _bound_arguments(original, args, kwargs)
            models = bound.get("models") or ()
            return {
                "models": [_patcher_identity(model) for model in models],
                "memory_required": int(bound.get("memory_required") or 0),
                "force_full_load": bool(bound.get("force_full_load", False)),
            }

        def describe_free_memory(original, args, kwargs):
            bound = _bound_arguments(original, args, kwargs)
            return {
                "memory_required": int(bound.get("memory_required") or 0),
                "device": str(bound.get("device")),
                "keep_loaded": len(bound.get("keep_loaded") or ()),
            }

        self._replace(
            mm,
            "load_models_gpu",
            self._simple("model_management.load_models_gpu", "model_management", describe_load_models),
            "model_management.load_models_gpu",
        )

        def free_memory_factory(original):
            def wrapped(*args, **kwargs):
                try:
                    fields = describe_free_memory(original, args, kwargs)
                except Exception:
                    fields = {}
                with trace.span("model_management.free_memory", "model_management", fields=fields) as out:
                    unloaded = original(*args, **kwargs)
                    with contextlib.suppress(Exception):
                        out["unloaded_models"] = [_patcher_identity(item.model) for item in unloaded or ()]
                    return unloaded

            return wrapped

        self._replace(mm, "free_memory", free_memory_factory, "model_management.free_memory")
        self._replace(
            mm,
            "soft_empty_cache",
            self._simple("model_management.soft_empty_cache", "allocator"),
            "model_management.soft_empty_cache",
        )
        loaded_model = getattr(mm, "LoadedModel", None)
        if loaded_model is None:
            return

        def residency_factory(name: str, direction: str):
            def factory(original):
                def wrapped(self_, *args, **kwargs):
                    patcher = getattr(self_, "model", None)
                    fields = _patcher_identity(patcher) if patcher is not None else {}
                    before = fields.pop("loaded_bytes", None)
                    bound = _bound_arguments(original, (self_, *args), kwargs)
                    for key in ("memory_to_free", "lowvram_model_memory", "extra_memory"):
                        if bound.get(key) is not None:
                            with contextlib.suppress(TypeError, ValueError):
                                fields[key] = int(bound[key])
                    fields["direction"] = direction
                    with trace.span(name, "model_residency", fields=fields) as out:
                        host_before = host_pressure_probe() if name == "model.load" else None
                        try:
                            result = original(self_, *args, **kwargs)
                        finally:
                            if host_before is not None:
                                host_after = host_pressure_probe()
                                out["host_pressure_before"] = host_before
                                if host_after is not None:
                                    out["host_pressure_after"] = host_after
                                    out["host_pressure_delta"] = {
                                        key: host_after[key] - previous
                                        for key, previous in host_before.items()
                                        if key in host_after
                                        and key.endswith(("faults", "pswpin", "pswpout", "pgmajfault"))
                                    }
                        after = _loaded_bytes(patcher) if patcher is not None else None
                        if before is not None and after is not None:
                            out["loaded_bytes_before"] = before
                            out["loaded_bytes_after"] = after
                            out["moved_bytes"] = abs(after - before)
                        if direction == "offload":
                            out["fully_unloaded"] = bool(result)
                        return result

                return wrapped

            return factory

        self._replace(loaded_model, "model_load", residency_factory("model.load", "load"), "LoadedModel.model_load")
        self._replace(
            loaded_model, "model_unload", residency_factory("model.unload", "offload"), "LoadedModel.model_unload"
        )
        self._replace(
            loaded_model,
            "model_use_more_vram",
            residency_factory("model.use_more_vram", "load"),
            "LoadedModel.model_use_more_vram",
        )

    def _install_sd(self, sd: Any) -> None:
        clip = getattr(sd, "CLIP", None)
        if clip is not None:
            for method in ("encode_from_tokens_scheduled", "encode_from_tokens"):
                self._replace(clip, method, self._simple(f"text_encoder.{method}", "text_encoder"), f"CLIP.{method}")
        vae = getattr(sd, "VAE", None)
        if vae is not None:
            for method in ("decode", "decode_tiled", "encode", "encode_tiled"):
                self._replace(vae, method, self._simple(f"vae.{method}", "vae"), f"VAE.{method}")

    def _install_sampler(self, guider: Any) -> None:
        trace = self.trace

        def factory(original):
            def wrapped(*args, **kwargs):
                with trace.span("sampler.cfg_guider_sample", "sampler", reset_peaks=True):
                    return original(*args, **kwargs)

            return wrapped

        self._replace(guider, "sample", factory, "CFGGuider.sample")

    def _install_execution(self, execution: Any) -> None:
        trace = self.trace
        execute = getattr(execution, "execute", None)
        if execute is not None and inspect.iscoroutinefunction(execute):

            def node_factory(original):
                async def wrapped(*args, **kwargs):
                    fields: dict[str, Any] = {}
                    bound = _bound_arguments(original, args, kwargs)
                    node_id = bound.get("current_item")
                    if node_id is not None:
                        fields["node_id"] = str(node_id)
                        with contextlib.suppress(Exception):
                            fields["class_type"] = str(bound["dynprompt"].get_node(node_id)["class_type"])
                    with trace.span("comfy.node", "node", fields=fields) as out:
                        result = await original(*args, **kwargs)
                        with contextlib.suppress(Exception):
                            out["result"] = str(getattr(result[0], "name", result[0]))
                        return result

                return wrapped

            self._replace(execution, "execute", node_factory, "execution.execute")

        executor = getattr(execution, "PromptExecutor", None)
        method = getattr(executor, "execute_async", None) if executor is not None else None
        if method is not None and inspect.iscoroutinefunction(method):
            hooks = self

            def prompt_factory(original):
                async def wrapped(self_, *args, **kwargs):
                    bound = _bound_arguments(original, (self_, *args), kwargs)
                    prompt_id = bound.get("prompt_id")
                    try:
                        with trace.span("comfy.prompt", "prompt", fields={"prompt_id": str(prompt_id)}):
                            return await original(self_, *args, **kwargs)
                    finally:
                        hooks.flush(prompt_id)

                return wrapped

            self._replace(executor, "execute_async", prompt_factory, "PromptExecutor.execute_async")

    def _install_flow_metrics(self, metrics_class: Any) -> None:
        trace = self.trace

        def factory(original):
            def wrapped(self_, kind, **fields):
                with contextlib.suppress(Exception):
                    stage_end = kind in STAGE_WALL_EVENTS
                    trace.instant(
                        f"flow.{kind}",
                        "flow_metrics",
                        _scalar_fields(fields),
                        memory=stage_end,
                        reset_peaks=stage_end,
                    )
                return original(self_, kind, **fields)

            return wrapped

        self._replace(metrics_class, "event", factory, "H3FlowMetrics.event")

    def flush(self, prompt_id: Any = None) -> Path | None:
        try:
            snapshot = self.trace.snapshot(prompt_id=prompt_id, reset=True)
            path = write_trace_json(snapshot, self._output_directory())
        except Exception as exc:
            _log.warning("[H3 lifecycle trace] could not write trace: %s", exc)
            return None
        self.written.append(path)
        _log.info("[H3 lifecycle trace] wrote %s (%d spans)", path, len(snapshot["spans"]))
        return path


_ACTIVE: ComfyLifecycleHooks | None = None


def _default_output_directory(value: str) -> Callable[[], Path]:
    if value.strip().lower() not in _TRUE:
        return lambda: Path(value).expanduser()

    def resolve() -> Path:
        folder_paths = sys.modules.get("folder_paths")
        base = None
        if folder_paths is not None:
            with contextlib.suppress(Exception):
                base = Path(folder_paths.get_output_directory())
        if base is None:
            base = Path(tempfile.gettempdir())
        return base / "h3_lifecycle_trace"

    return resolve


def _env_int(environ: dict[str, str], name: str, default: int) -> int:
    try:
        return int(environ.get(name, default))
    except ValueError:
        return default


def install_from_environment(environ: dict[str, str] | None = None) -> ComfyLifecycleHooks | None:
    """Install ComfyUI lifecycle hooks when ``H3_LIFECYCLE_TRACE`` is set; idempotent."""
    global _ACTIVE
    environ = os.environ if environ is None else environ
    value = str(environ.get(ENV_TRACE, "")).strip()
    if not value or value.lower() in {"0", "false", "no", "off"}:
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    trace = LifecycleTrace(stage_peaks=str(environ.get(ENV_STAGE_PEAK, "1")).strip().lower() in _TRUE)
    interval_ms = _env_int(environ, ENV_NVML_MS, 100)
    if interval_ms > 0:
        try:
            reader = NvmlReader(_env_int(environ, ENV_NVML_INDEX, 0))
            trace.nvml = NvmlSampler(reader.read, interval_ms / 1000.0, trace.now)
            trace.nvml.start()
            trace.nvml_status = f"sampling_every_{interval_ms}ms"
        except Exception as exc:
            trace.nvml_status = f"unavailable:{exc}"
    hooks = ComfyLifecycleHooks(trace, _default_output_directory(value))
    try:
        installed = hooks.install()
    except Exception as exc:  # tracing must never prevent node registration
        hooks.uninstall()
        if trace.nvml is not None:
            trace.nvml.stop()
        _log.warning("[H3 lifecycle trace] disabled after install failure: %s", exc)
        return None
    _ACTIVE = hooks
    _log.info("[H3 lifecycle trace] enabled; nvml=%s; hooks=%s", trace.nvml_status, ", ".join(installed))
    return hooks


__all__ = [
    "ENV_TRACE",
    "STAGE_WALL_EVENTS",
    "TRACE_SCHEMA",
    "ComfyLifecycleHooks",
    "LifecycleTrace",
    "NvmlSampler",
    "cuda_memory_probe",
    "install_from_environment",
    "write_trace_json",
]
