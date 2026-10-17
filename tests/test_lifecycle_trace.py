from __future__ import annotations

import asyncio
import gc
import json
import types

import pytest

from h3_flow_regenerate import lifecycle_trace as lt
from h3_flow_regenerate.lifecycle_report import analyze_trace, format_report
from h3_flow_regenerate.metrics import H3FlowMetrics

GIB = 1 << 30


class FakeClock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000

    def __call__(self) -> int:
        return self.ns

    def advance(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)


class Patcher:
    def __init__(self, name: str, size: int, loaded: int = 0) -> None:
        self.model = type(name, (), {})()
        self.size = size
        self.loaded = loaded

    def loaded_size(self) -> int:
        return self.loaded

    def model_size(self) -> int:
        return self.size


def _fake_comfy(clock: FakeClock):
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = []

    class LoadedModel:
        def __init__(self, patcher):
            self.model = patcher

        def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
            clock.advance(2.0)
            self.model.loaded = self.model.size
            return self.model.model

        def model_unload(self, memory_to_free=None, unpatch_weights=True):
            clock.advance(1.5)
            self.model.loaded = max(0, self.model.loaded - int(memory_to_free))
            return self.model.loaded == 0

        def model_use_more_vram(self, extra_memory, force_patch_weights=False):
            return 0

    def free_memory(memory_required, device, keep_loaded=(), for_dynamic=False, pins_required=0, ram_required=0):
        unloaded = []
        for loaded in list(mm.current_loaded_models) if memory_required > 0 else ():
            if loaded.model_unload(memory_to_free=memory_required):
                mm.current_loaded_models.remove(loaded)
                unloaded.append(loaded)
        mm.soft_empty_cache()
        return unloaded

    def load_models_gpu(
        models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False
    ):
        mm.free_memory(memory_required, "cuda:0")
        for patcher in models:
            loaded = LoadedModel(patcher)
            loaded.model_load()
            mm.current_loaded_models.append(loaded)

    def soft_empty_cache(force=False):
        clock.advance(0.25)

    mm.LoadedModel = LoadedModel
    mm.free_memory = free_memory
    mm.load_models_gpu = load_models_gpu
    mm.soft_empty_cache = soft_empty_cache

    sd = types.ModuleType("comfy.sd")

    class CLIP:
        def __init__(self, patcher):
            self.patcher = patcher

        def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
            mm.load_models_gpu([self.patcher])
            clock.advance(3.0)
            return ("cond", tokens)

    class VAE:
        def decode(self, samples_in, vae_options=None):
            clock.advance(4.0)
            return samples_in

    sd.CLIP = CLIP
    sd.VAE = VAE

    samplers = types.ModuleType("comfy.samplers")

    class CFGGuider:
        def __init__(self, patcher):
            self.patcher = patcher

        def sample(
            self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None
        ):
            mm.free_memory(GIB, "cuda:0")
            mm.load_models_gpu([self.patcher])
            clock.advance(10.0)
            return latent_image

    samplers.CFGGuider = CFGGuider

    execution = types.ModuleType("execution")
    behaviour = {}

    async def execute(
        server,
        dynprompt,
        caches,
        current_item,
        extra_data,
        executed,
        prompt_id,
        execution_list,
        pending_subgraph_results,
        pending_async_nodes,
        ui_outputs,
    ):
        behaviour[current_item]()
        return (types.SimpleNamespace(name="SUCCESS"), None, None)

    class DynPrompt:
        def __init__(self, classes):
            self.classes = classes

        def get_node(self, node_id):
            return {"class_type": self.classes[node_id]}

    class PromptExecutor:
        async def execute_async(self, prompt, prompt_id, extra_data=None, execute_outputs=()):
            dynprompt = DynPrompt({node: spec[0] for node, spec in prompt.items()})
            for node in prompt:
                clock.advance(0.5)
                await execution.execute(None, dynprompt, None, node, {}, set(), prompt_id, None, {}, {}, {})

    execution.execute = execute
    execution.PromptExecutor = PromptExecutor
    execution.behaviour = behaviour
    return {"comfy.model_management": mm, "comfy.sd": sd, "comfy.samplers": samplers, "execution": execution}


def _probe_factory(clock):
    def probe():
        return {
            "allocated": 10 * GIB,
            "reserved": 12 * GIB,
            "peak_allocated": 20 * GIB,
            "peak_reserved": 24 * GIB,
            "driver_free": 30 * GIB,
            "stream_idle": True,
        }

    return probe


def _run_prompt(tmp_path):
    clock = FakeClock()
    modules = _fake_comfy(clock)
    trace = lt.LifecycleTrace(clock=clock, memory_probe=_probe_factory(clock), reset_peaks=lambda: True)
    hooks = lt.ComfyLifecycleHooks(trace, lambda: tmp_path)
    originals = {
        "load": modules["comfy.model_management"].load_models_gpu,
        "unload": modules["comfy.model_management"].LoadedModel.model_unload,
        "execute": modules["execution"].execute,
    }
    hooks.install(modules)
    sd, samplers, execution = modules["comfy.sd"], modules["comfy.samplers"], modules["execution"]
    text_encoder = Patcher("QwenTE", 26 * GIB)
    diffusion = Patcher("H3Model", 20 * GIB)
    clip = sd.CLIP(text_encoder)
    execution.behaviour.update(
        {
            "1": lambda: clip.encode_from_tokens(["chunk-1"]),
            "2": lambda: samplers.CFGGuider(diffusion).sample(None, "latent", None, None),
            "3": lambda: sd.VAE().decode("latent"),
        }
    )
    prompt = {"1": ("TextEncode",), "2": ("Sampler",), "3": ("VAEDecode",)}
    asyncio.run(execution.PromptExecutor().execute_async(prompt, "p-1"))
    return hooks, modules, originals, clock


def test_trace_accounts_prompt_wall_without_double_counting(tmp_path):
    hooks, modules, originals, _clock = _run_prompt(tmp_path)
    try:
        assert len(hooks.written) == 1
        trace = json.loads(hooks.written[0].read_text(encoding="utf-8"))
        summary = analyze_trace(trace)
        assert trace["schema"] == lt.TRACE_SCHEMA
        assert summary["wall_s"] == pytest.approx(24.75)
        assert sum(summary["exclusive_by_subsystem_s"].values()) == pytest.approx(summary["wall_s"])
        exclusive = summary["exclusive_by_subsystem_s"]
        assert exclusive["between_nodes"] == pytest.approx(1.5)
        assert exclusive["text_encoder"] == pytest.approx(3.0)
        assert exclusive["vae"] == pytest.approx(4.0)
        assert exclusive["sampler"] == pytest.approx(10.0)
        assert exclusive["model_residency"] == pytest.approx(2.0 + 1.5 + 2.0)
        assert exclusive["allocator"] == pytest.approx(0.75)

        offloads = [item for item in summary["residency"] if item["event"] == "model.unload"]
        assert offloads == [
            {
                "start_s": pytest.approx(6.25),
                "event": "model.unload",
                "model_class": "QwenTE",
                "moved_mib": 1024.0,
                "seconds": 1.5,
                "gib_per_s": pytest.approx(0.67),
                "fully_unloaded": False,
                "node": "Sampler#2",
                "inside_sampler": True,
            }
        ]
        sampler = summary["samplers"][0]
        assert sampler["seconds"] == pytest.approx(14.0)
        assert sampler["peak_allocated_mib"] == 20480.0
        assert sampler["residency_seconds"] == pytest.approx(4.0)
        report = format_report(summary)
        assert "Sampler#2" in report and "QwenTE" in report
    finally:
        hooks.uninstall()
    mm = modules["comfy.model_management"]
    assert mm.load_models_gpu is originals["load"]
    assert mm.LoadedModel.model_unload is originals["unload"]
    assert modules["execution"].execute is originals["execute"]
    assert hooks.trace.gc_callback not in gc.callbacks


def test_model_load_reports_host_paging_without_changing_execution(monkeypatch, tmp_path):
    clock = FakeClock()
    modules = _fake_comfy(clock)
    snapshots = iter(
        (
            {
                "process_minor_faults": 13,
                "process_major_faults": 5,
                "system_pswpin": 100,
                "system_pswpout": 200,
                "system_pgmajfault": 80,
                "memavailable_kib": 200_000,
            },
            {
                "process_minor_faults": 15,
                "process_major_faults": 8,
                "system_pswpin": 124,
                "system_pswpout": 208,
                "system_pgmajfault": 83,
                "memavailable_kib": 180_000,
            },
        )
    )
    monkeypatch.setattr(lt, "host_pressure_probe", lambda: next(snapshots))
    trace = lt.LifecycleTrace(clock=clock, memory_probe=lambda: None)
    hooks = lt.ComfyLifecycleHooks(trace, lambda: tmp_path)
    hooks.install(modules)
    try:
        patcher = Patcher("QwenTE", 26 * GIB)
        loaded = modules["comfy.model_management"].LoadedModel(patcher)
        assert loaded.model_load() is patcher.model
        receipt = next(span for span in trace.spans if span["name"] == "model.load")
        assert receipt["fields"]["moved_bytes"] == 26 * GIB
        assert receipt["fields"]["host_pressure_delta"] == {
            "process_minor_faults": 2,
            "process_major_faults": 3,
            "system_pswpin": 24,
            "system_pswpout": 8,
            "system_pgmajfault": 3,
        }
        assert receipt["fields"]["host_pressure_before"]["memavailable_kib"] == 200_000
        assert receipt["fields"]["host_pressure_after"]["memavailable_kib"] == 180_000
        summary = analyze_trace(trace.snapshot())
        assert summary["residency"][0]["host_pressure_delta"]["system_pswpin"] == 24
        report = format_report(summary)
        assert "Host paging during model loads" in report
        assert "QwenTE | 2.000 | 3 | 24 | 8 | 200000→180000" in report
    finally:
        hooks.uninstall()


def test_host_paging_probe_failure_does_not_break_model_load(monkeypatch, tmp_path):
    clock = FakeClock()
    modules = _fake_comfy(clock)
    monkeypatch.setattr(lt, "host_pressure_probe", lambda: None)
    trace = lt.LifecycleTrace(clock=clock, memory_probe=lambda: None)
    hooks = lt.ComfyLifecycleHooks(trace, lambda: tmp_path)
    hooks.install(modules)
    try:
        patcher = Patcher("QwenTE", 26 * GIB)
        modules["comfy.model_management"].LoadedModel(patcher).model_load()
        receipt = next(span for span in trace.spans if span["name"] == "model.load")
        assert "host_pressure_delta" not in receipt["fields"]
        assert receipt["fields"]["moved_bytes"] == 26 * GIB
    finally:
        hooks.uninstall()


def test_wrappers_are_transparent_for_results_and_errors(tmp_path):
    clock = FakeClock()
    modules = _fake_comfy(clock)

    def failing_decode(self, samples_in, vae_options=None):
        raise RuntimeError("decode failed")

    modules["comfy.sd"].VAE.decode_tiled = failing_decode
    trace = lt.LifecycleTrace(clock=clock, memory_probe=lambda: None)
    hooks = lt.ComfyLifecycleHooks(trace, lambda: tmp_path)
    hooks.install(modules)
    try:
        vae = modules["comfy.sd"].VAE()
        assert vae.decode("x") == "x"
        with pytest.raises(RuntimeError, match="decode failed"):
            vae.decode_tiled("x")
        errored = [span for span in trace.spans if span["name"] == "vae.decode_tiled"]
        assert errored and errored[0]["error"] == "RuntimeError"
    finally:
        hooks.uninstall()
    assert modules["comfy.sd"].VAE.decode_tiled is failing_decode


def test_install_is_idempotent_and_never_double_wraps(tmp_path):
    clock = FakeClock()
    modules = _fake_comfy(clock)
    hooks = lt.ComfyLifecycleHooks(lt.LifecycleTrace(clock=clock, memory_probe=lambda: None), lambda: tmp_path)
    hooks.install(modules)
    wrapped = modules["comfy.model_management"].free_memory
    second = lt.ComfyLifecycleHooks(lt.LifecycleTrace(clock=clock, memory_probe=lambda: None), lambda: tmp_path)
    second.install(modules)
    try:
        assert modules["comfy.model_management"].free_memory is wrapped
        assert "model_management.free_memory" not in second.installed
    finally:
        second.uninstall()
        hooks.uninstall()


def test_gc_and_flow_events_attach_to_the_enclosing_span(tmp_path):
    clock = FakeClock()
    trace = lt.LifecycleTrace(clock=clock, memory_probe=lambda: None)
    hooks = lt.ComfyLifecycleHooks(trace, lambda: tmp_path)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    hooks.install({})
    try:
        with trace.span("comfy.node", "node", fields={"class_type": "Flow", "node_id": "9"}):
            trace.gc_callback("start", {"generation": 2})
            clock.advance(0.2)
            trace.gc_callback("stop", {"generation": 2, "collected": 7})
            H3FlowMetrics().event("sampler_wall", stage="high", elapsed_s=12.5, payload={"large": [1, 2]})
        node = next(span for span in trace.spans if span["name"] == "comfy.node")
        collected = next(span for span in trace.spans if span["name"] == "python.gc")
        flow = next(span for span in trace.spans if span["name"] == "flow.sampler_wall")
        assert collected["parent"] == node["id"] and collected["fields"]["collected"] == 7
        assert flow["parent"] == node["id"]
        assert flow["fields"] == {"stage": "high", "elapsed_s": 12.5}
        summary = analyze_trace(trace.snapshot())
        assert summary["python_gc"] == {"count": 1, "total_s": 0.2, "max_s": 0.2}
        assert summary["exclusive_by_subsystem_s"]["python_gc"] == pytest.approx(0.2)
    finally:
        hooks.uninstall()
        if gc_was_enabled:
            gc.enable()


def test_nvml_idle_intervals_are_attributed_to_traced_spans():
    clock = FakeClock()
    trace = lt.LifecycleTrace(clock=clock, memory_probe=lambda: None)
    readings = iter([95] * 10 + [0] * 30 + [97] * 10)
    trace.nvml = lt.NvmlSampler(lambda: (next(readings), 0, 300_000, 0, 1800), 0.1, clock)
    with trace.span("comfy.prompt", "prompt"):
        with trace.span("comfy.node", "node", fields={"class_type": "Sampler", "node_id": "1"}):
            for _ in range(10):
                trace.nvml.sample_once()
                clock.advance(0.1)
        with (
            trace.span("comfy.node", "node", fields={"class_type": "TextEncode", "node_id": "2"}),
            trace.span("model.load", "model_residency", fields={"model_class": "QwenTE"}),
        ):
            for _ in range(30):
                trace.nvml.sample_once()
                clock.advance(0.1)
        with trace.span("comfy.node", "node", fields={"class_type": "Sampler", "node_id": "3"}):
            for _ in range(10):
                trace.nvml.sample_once()
                clock.advance(0.1)
    summary = analyze_trace(trace.snapshot(reset=True))
    (idle,) = summary["gpu_idle_intervals"]
    assert idle["seconds"] == pytest.approx(3.0)
    names = {item["name"] for item in idle["covered_by"]}
    assert {"comfy.node", "model.load"} <= names
    assert summary["nvml_prompt"]["idle_fraction"] == pytest.approx(0.6)
    assert not trace.nvml.samples


def test_nvml_sampler_retains_fresh_samples_at_capacity():
    clock = FakeClock()
    readings = iter([10, 20, 30, 40])
    sampler = lt.NvmlSampler(lambda: (next(readings), 0, 300_000, 0, 1800), 0.1, clock, max_samples=2)
    for _ in range(4):
        sampler.sample_once()
        clock.advance(0.1)
    assert sampler.dropped == 2
    assert [s[1] for s in sampler.peek()] == [30, 40]
    assert [s[1] for s in sampler.drain()] == [30, 40]
    assert sampler.peek() == []


def test_span_buffer_is_bounded():
    clock = FakeClock()
    trace = lt.LifecycleTrace(clock=clock, memory_probe=lambda: None, max_spans=3)
    for _ in range(5):
        with trace.span("x", "node"):
            clock.advance(0.01)
    snapshot = trace.snapshot(reset=True)
    assert len(snapshot["spans"]) == 3
    assert snapshot["dropped_spans"] == 2
    assert trace.snapshot()["dropped_spans"] == 0


def test_environment_controls_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "_ACTIVE", None)
    assert lt.install_from_environment({}) is None
    assert lt.install_from_environment({lt.ENV_TRACE: "0"}) is None
    hooks = lt.install_from_environment({lt.ENV_TRACE: str(tmp_path), lt.ENV_NVML_MS: "0"})
    try:
        assert hooks is not None
        assert lt.install_from_environment({lt.ENV_TRACE: str(tmp_path)}) is hooks
        assert hooks.trace.nvml_status == "disabled"
        with hooks.trace.span("comfy.prompt", "prompt"):
            pass
        path = hooks.flush("abc")
        assert path is not None and path.parent == tmp_path
        assert json.loads(path.read_text(encoding="utf-8"))["prompt_id"] == "abc"
    finally:
        hooks.uninstall()
        monkeypatch.setattr(lt, "_ACTIVE", None)


def test_flow_stage_walls_report_per_stage_peaks_and_residency():
    clock = FakeClock()
    peaks = {"allocated": 50 * GIB}
    resets = []

    def probe():
        return {
            "allocated": 40 * GIB,
            "reserved": 60 * GIB,
            "peak_allocated": peaks["allocated"],
            "peak_reserved": 70 * GIB,
            "driver_free": 9 * GIB,
        }

    def reset():
        resets.append(clock())
        peaks["allocated"] = 0
        return True

    trace = lt.LifecycleTrace(clock=clock, memory_probe=probe, reset_peaks=reset)
    hooks = lt.ComfyLifecycleHooks(trace, lambda: None)
    hooks.install({})
    metrics = H3FlowMetrics()
    try:
        with trace.span("comfy.prompt", "prompt"), trace.span("sampler.cfg_guider_sample", "sampler", reset_peaks=True):
            peaks["allocated"] = 61 * GIB
            clock.advance(10.0)
            metrics.event("low_stage_wall", elapsed_ms=10_000.0)
            with trace.span("model.unload", "model_residency", fields={"model_class": "QwenTE"}):
                clock.advance(2.0)
            peaks["allocated"] = 75 * GIB
            clock.advance(18.0)
            metrics.event("high_stage_wall", elapsed_ms=20_000.0)
        summary = analyze_trace(trace.snapshot())
    finally:
        hooks.uninstall()
    low, high = summary["flow_stages"]
    assert (low["stage"], low["seconds"], low["peak_allocated_mib"]) == ("low_stage_wall", 10.0, 61 * 1024.0)
    assert (high["stage"], high["seconds"], high["peak_allocated_mib"]) == ("high_stage_wall", 20.0, 75 * 1024.0)
    assert high["residency_gc_seconds"] == pytest.approx(2.0)
    assert low["residency_gc_seconds"] == 0.0
    assert len(resets) == 3
    assert "## Flow stages" in format_report(summary)


def test_report_explains_peak_headroom_incommensurability():
    from h3_flow_regenerate.lifecycle_report import format_report

    minimal = {
        "prompt_id": "test",
        "wall_s": 0.0,
        "exclusive_by_subsystem_s": {},
        "nodes": [],
        "samplers": [],
        "flow_stages": [],
        "residency": [],
        "gpu_idle_intervals": [],
        "inter_node_gaps": [],
        "python_gc": {"count": 0, "total_s": 0.0, "max_s": 0.0},
    }
    text = format_report(minimal)
    assert "NOT the free/reclaimable-memory" in text
    assert "do not subtract or compare them directly" in text
    assert "process-wide telemetry" in text


def test_nested_component_wall_does_not_reset_transfer_peak():
    """A learned-upscale component wall cannot split its enclosing transfer."""
    clock = FakeClock()
    peak = {"allocated": 0}
    resets = []

    def probe():
        return {
            "allocated": 50 * GIB,
            "reserved": 60 * GIB,
            "peak_allocated": peak["allocated"],
            "peak_reserved": 60 * GIB,
        }

    def reset():
        resets.append(clock())
        peak["allocated"] = 0
        return True

    trace = lt.LifecycleTrace(clock=clock, memory_probe=probe, reset_peaks=reset)
    hooks = lt.ComfyLifecycleHooks(trace, lambda: None)
    hooks.install({})
    metrics = H3FlowMetrics()
    try:
        with (
            trace.span("comfy.prompt", "prompt"),
            trace.span(
                "sampler.cfg_guider_sample",
                "sampler",
                reset_peaks=True,
            ),
        ):
            peak["allocated"] = 61 * GIB
            clock.advance(10)
            metrics.event("low_stage_wall", elapsed_ms=10_000.0)
            peak["allocated"] = 72 * GIB
            clock.advance(1)
            metrics.event("handoff_learned_upscale_wall", elapsed_ms=1_000.0)
            clock.advance(1)
            metrics.event("handoff_transfer_wall", elapsed_ms=2_000.0)
        snapshot = trace.snapshot()
        summary = analyze_trace(snapshot)
    finally:
        hooks.uninstall()

    assert [stage["stage"] for stage in summary["flow_stages"]] == [
        "low_stage_wall",
        "handoff_transfer_wall",
    ]
    assert summary["flow_stages"][1]["peak_allocated_mib"] == 72 * 1024.0
    assert len(resets) == 3
    assert any(span["name"] == "flow.handoff_learned_upscale_wall" for span in snapshot["spans"])
