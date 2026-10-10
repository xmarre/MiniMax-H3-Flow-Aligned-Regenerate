"""Bounded, read-only AV witnesses for the frozen low/probe replay.

The observer keeps baseline operands on temporary CPU disk, never GPU history.
Sampler locals are observed by exact code identity, without replacing a solver.
An unavailable boundary is not evidence of identity or physical state isolation.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import inspect
import json
import math
import pickle
import random
import sys
import tempfile
from pathlib import Path

import torch

from .partitioned_stage import tensor_sha256

POLICY = "h3_low_first_divergence_v1"
_ACTIVE = contextvars.ContextVar("h3_low_first_divergence", default=None)
_OWNER = contextvars.ContextVar("h3_low_replay_owner", default=None)


def _full_sha(value):
    # Stream large contiguous history through bounded host buffers. The hash
    # covers every byte, including dtype-level distinctions such as signed zero.
    flat = value.detach().reshape(-1)
    digest = hashlib.sha256()
    elements = max(1, 8 * 1024 * 1024 // value.element_size())
    for start in range(0, flat.numel(), elements):
        chunk = flat[start : start + elements].to(device="cpu").contiguous().view(torch.uint8)
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _digest(value, *, budget=64 * 1024 * 1024):
    """Full tensor hashes within a byte budget; never silently sample tensors."""
    remaining = budget
    hashed_bytes = 0
    unobserved = []

    def visit(item, path, depth=0):
        nonlocal remaining, hashed_bytes
        if depth > 12:
            unobserved.append(path)
            return {"unobserved": "depth_limit"}
        if torch.is_tensor(item):
            size = item.numel() * item.element_size()
            entry = {"shape": list(item.shape), "dtype": str(item.dtype), "nbytes": size}
            if (
                (remaining is not None and size > remaining)
                or item.device.type == "meta"
                or (item.layout != torch.strided or (remaining is None and not item.is_contiguous()))
            ):
                unobserved.append(path)
                entry["unobserved"] = "byte_budget_or_tensor_layout"
            else:
                if remaining is not None:
                    remaining -= size
                hashed_bytes += size
                entry["sha256"] = _full_sha(item)
            return entry
        if isinstance(item, dict):
            return {
                str(k): visit(v, f"{path}.{k}", depth + 1) for k, v in sorted(item.items(), key=lambda p: str(p[0]))
            }
        if isinstance(item, (list, tuple)):
            return [visit(v, f"{path}.{i}", depth + 1) for i, v in enumerate(item)]
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else {"nonfinite": str(item)}
        unobserved.append(path)
        return {"unobserved_type": f"{type(item).__module__}.{type(item).__qualname__}"}

    tree = visit(value, "root")
    return {
        "sha256": hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest(),
        "hashed_bytes": hashed_bytes,
        "unobserved": unobserved,
        "tree": tree,
    }


def _delta(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"compatible": False, "exact": False, "finite": False, "rms": None, "abs_max": None}
    count, squared, maximum, finite = 0, 0.0, 0.0, True
    av, bv = a.reshape(-1), b.reshape(-1)
    for start in range(0, av.numel(), 1024 * 1024):
        d = av[start : start + 1024 * 1024].float() - bv[start : start + 1024 * 1024].float()
        finite = finite and bool(torch.isfinite(d).all())
        squared += float(d.square().sum(dtype=torch.float64))
        maximum = max(maximum, float(d.abs().max())) if d.numel() else maximum
        count += d.numel()
    return {
        "compatible": True,
        "exact": _full_sha(a) == _full_sha(b),
        "finite": finite,
        "rms": math.sqrt(squared / max(count, 1)) if finite else None,
        "abs_max": maximum if finite else None,
    }


def backend_receipt(guider, options):
    runtime = getattr((getattr(guider, "model_options", None) or {}).get("spectrum_h3_binding"), "runtime", None)
    receipt = {
        "physical_state_isolation_verified": False,
        "model_owned_runtime_buffers": "unobserved",
        "attention_kernel_internal_state": "unobserved",
    }
    if runtime is None:
        receipt["spectrum"] = {"available": False}
    else:
        forecaster = getattr(runtime, "forecaster", None)
        history = getattr(forecaster, "_history", ())
        # Only scalar/tensor data from known fields. No property calls that
        # create snapshots or caches and no synchronization of VDN workers.
        receipt["spectrum"] = {
            "available": True,
            "runtime_id": id(runtime),
            "run_id": getattr(runtime, "active_run_id", None),
            "step_id": getattr(runtime, "active_step_id", None),
            "forecaster_id": id(forecaster),
            "generation": getattr(forecaster, "_generation", None),
            "factor_generation": getattr(forecaster, "_factor_generation", None),
            "history_length": len(history),
            "history": _digest(
                [(getattr(h, "coordinate", None), getattr(h, "feature_flat", None)) for h in history], budget=None
            ),
            "evidence": _digest(
                {
                    k: getattr(forecaster, k, None)
                    for k in ("_evidence_history", "_exact_head_history", "_design", "_cholesky")
                }
            ),
            "controller_fields": _digest(
                vars(getattr(runtime, "_generic_correction_controller", None))
                if hasattr(getattr(runtime, "_generic_correction_controller", None), "__dict__")
                else None
            ),
        }
    stage = options.get("h3_flow_partitioned_stage_v1")
    receipt["partition"] = {
        "owner_id": id(stage) if stage is not None else None,
        "provider_cache_entries": len(getattr(stage, "attention_provider_cache", {})),
        "attention_provider_identity": str(getattr(stage, "attention_provider_identity", None)),
    }
    # Do not import optional backends or initialize any request/pool.
    vdn = sys.modules.get("vdn_h3.runtime")
    current = getattr(vdn, "current_runtime_buffers", None)
    buffers = current() if callable(current) else None
    receipt["vdn"] = {
        "available": buffers is not None,
        "buffers_id": id(buffers) if buffers else None,
        "cache_entries": {
            k: len(getattr(buffers, k, {}))
            for k in ("_scan", "_delta", "_plans", "_kv", "_activations", "_partition_indices")
        },
        "cache_tensor_contents": "unobserved",
    }
    sol = sys.modules.get("sol_h3.runtime")
    request_var = getattr(sol, "_REQUEST", None)
    request = request_var.get() if request_var is not None else None
    receipt["sol"] = {
        "available": request is not None,
        "request_id": id(request) if request else None,
        "backend_transitions": getattr(request, "backend_transitions", None),
        "kernel_id": id(getattr(request, "kernel", None)) if request else None,
    }
    return receipt


def rng_receipt():
    import numpy as np

    return {
        "python_numpy_sha256": hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()))).hexdigest(),
        "torch_cpu_sha256": tensor_sha256(torch.get_rng_state()),
        "initialized_cuda_sha256": [tensor_sha256(s) for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else None,
    }


class LowReplayTrace:
    def __init__(self, *, max_bytes=2 * 1024**3, max_records=512, max_tensor_bytes=128 * 1024**2):
        self._temp = tempfile.TemporaryDirectory(prefix="h3-low-replay-")
        self.root = Path(self._temp.name)
        self.max_bytes, self.max_records, self.max_tensor_bytes = max_bytes, max_records, max_tensor_bytes
        self.bytes = 0
        self.arms = {}
        self.arm = None
        self.stage = None
        self.sampler_code = None
        self.evaluation = -1
        self.native_call = 0
        self._native_count = 0
        self.truncated = []
        self.unobserved_count = 0
        self._first_operands = {}

    def close(self):
        self._temp.cleanup()

    def unobserve(self, reason):
        self.unobserved_count += 1
        if len(self.truncated) < 128 and reason not in self.truncated:
            self.truncated.append(reason)

    def record(self, boundary, value, **metadata):
        if not torch.is_tensor(value):
            self.unobserve(f"{self.arm}/{self.stage}/{boundary}:tensor_unavailable")
            return
        rows = self.arms[self.arm]
        key = (self.stage, self.evaluation, self.native_call, boundary)
        if any(r["key"] == key for r in rows):
            self.unobserve(f"{key}:duplicate_boundary")
            return
        size = value.numel() * value.element_size()
        if len(rows) >= self.max_records or size > self.max_tensor_bytes:
            self.unobserve(f"{key}:record_or_tensor_limit")
            return
        cpu = value.detach().to(device="cpu", copy=True).contiguous()
        row = {
            "key": key,
            "boundary": boundary,
            "stage": self.stage,
            "evaluation": self.evaluation,
            "native_call": self.native_call,
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "sha256": _full_sha(cpu),
            "finite": bool(torch.isfinite(cpu).all()),
            "metadata": metadata,
        }
        if self.arm == "A":
            if self.bytes + size > self.max_bytes:
                self.unobserve(f"{key}:baseline_storage_limit")
                return
            file = self.root / f"A-{len(rows)}.pt"
            torch.save(cpu, file)
            row["file"] = str(file)
            self.bytes += size
        else:
            reference = next((r for r in self.arms.get("A", ()) if r["key"] == key), None)
            if reference is not None:
                original = torch.load(reference["file"], map_location="cpu", weights_only=True)
                row["difference"] = _delta(original, cpu)
                if self.arm == "A_replay" and not row["difference"]["exact"] and self.arm not in self._first_operands:
                    file = self.root / "first-replay.pt"
                    torch.save(cpu, file)
                    self._first_operands[self.arm] = (key, reference["file"], str(file))
        rows.append(row)

    def begin_prediction(self, guider, x, timestep, options):
        self.evaluation += 1
        self.native_call = 0
        self._native_count = 0
        # Retain no frame references: tensors are copied before the frame exits.
        frame = sys._getframe(1)
        solver = None
        try:
            while frame is not None:
                if frame.f_code is self.sampler_code:
                    solver = frame.f_locals.get("x")
                    break
                frame = frame.f_back
            self.record("solver_input", solver, observation="live_exact_sampler_frame_x")
        finally:
            del frame, solver
        self.record(
            "predict_input",
            x,
            backend=backend_receipt(guider, options),
            rng=rng_receipt(),
            cfg=getattr(guider, "cfg", None),
            sigma=float(timestep.detach().flatten()[0]),
        )
        self.record("timestep", timestep)

    def end_prediction(self, result, *, actual, phase, outer, backend=None):
        self.record("predict_output", result, actual=actual, phase=phase, outer_step=outer, backend=backend)

    def callback(self, step, x0, x, total):
        self.record("callback_x", x, callback_step=int(step), total=int(total), observation="before_update")
        self.record("callback_x0", x0, callback_step=int(step))

    def summary(self):
        def public(row):
            return {k: v for k, v in row.items() if k != "file"}

        baseline = self.arms.get("A", [])
        comparisons = {}
        for arm in ("B_target_projection", "A_replay"):
            rows = self.arms.get(arm, [])
            topology = None
            first = None
            semantic_match = True
            backend_comparisons = []
            # Validate whole evaluation identities before comparing native rows.
            # A changed forecast decision can remove native calls entirely.
            a_calls = [r for r in baseline if r["boundary"] == "predict_output"]
            b_calls = [r for r in rows if r["boundary"] == "predict_output"]
            bad_evaluation = None
            for a, b in zip(a_calls, b_calls, strict=False):
                if self.call_identity(a) != self.call_identity(b):
                    bad_evaluation = (a["stage"], a["evaluation"])
                    topology = {"evaluation": bad_evaluation, "reason": "actual_forecast_phase_or_outer_step"}
                    break
            for i, (a, b) in enumerate(zip(baseline, rows, strict=False)):
                if bad_evaluation == (a["stage"], a["evaluation"]):
                    break
                if a["key"] != b["key"] or (a["boundary"] == "timestep" and a["sha256"] != b["sha256"]):
                    topology = {"record": i, "reason": "boundary_order_or_timestep", "a": public(a), "b": public(b)}
                    break
                if a["boundary"] in {"predict_input", "predict_output"}:
                    backend_comparisons.append(
                        {
                            "record": i,
                            "stage": a["stage"],
                            "evaluation": a["evaluation"],
                            "a": a["metadata"].get("backend"),
                            "b": b["metadata"].get("backend"),
                            "rng_equal": a["metadata"].get("rng") == b["metadata"].get("rng"),
                        }
                    )
                cfg_changed = a["boundary"] == "predict_input" and a["metadata"].get("cfg") != b["metadata"].get("cfg")
                if cfg_changed:
                    semantic_match = False
                    first = first or {
                        "record": i,
                        "stage": b["stage"],
                        "evaluation": b["evaluation"],
                        "boundary": "guidance_cfg",
                        "classification": "guidance_configuration",
                    }
                condition_a = a["metadata"].get("conditioning")
                condition_b = b["metadata"].get("conditioning")
                if condition_a != condition_b:
                    semantic_match = False
                    first = first or {
                        "record": i,
                        "stage": b["stage"],
                        "evaluation": b["evaluation"],
                        "boundary": "native_conditioning",
                        "classification": "conditioning_or_mask_payload",
                        "a": condition_a,
                        "b": condition_b,
                    }
                d = b.get("difference")
                if first is None and d is not None and (not d["exact"] or not d["finite"]):
                    first = {
                        "record": i,
                        "stage": b["stage"],
                        "evaluation": b["evaluation"],
                        "boundary": b["boundary"],
                        "difference": d,
                        "classification": self.classify(b),
                        "key": b["key"],
                        "a_sha256": a["sha256"],
                        "b_sha256": b["sha256"],
                    }
            if topology is None and len(baseline) != len(rows):
                topology = {"record": min(len(baseline), len(rows)), "reason": "different_record_count"}
            within_tolerance = (
                bool(baseline)
                and topology is None
                and semantic_match
                and not self.truncated
                and all(
                    r.get("difference", {}).get("finite", False) and r["difference"]["abs_max"] <= 1e-5 for r in rows
                )
            )
            if first is not None and first["boundary"] in {"native_conditioning", "guidance_cfg"}:
                within_tolerance = False
            comparisons[arm] = {
                "first_observed_divergence": first,
                "alignment_failure": topology,
                "observable_backend_comparisons": backend_comparisons,
                "reproduction_within_tolerance": within_tolerance,
                "fully_observed_records_exact": bool(baseline)
                and first is None
                and topology is None
                and not self.truncated,
            }
        sources = {}
        for name in (
            "comfy.samplers",
            "comfy.k_diffusion.sampling",
            "comfy.ldm.minimax.model",
            "comfyui_spectrum_h3.runtime",
            "comfyui_spectrum_h3.sampling",
            "comfyui_spectrum_h3.forecast",
            "vdn_h3.runtime",
            "vdn_h3.hybrid",
            "sol_h3.runtime",
        ):
            source = getattr(sys.modules.get(name), "__file__", None)
            if source and Path(source).is_file():
                sources[name] = {"file": source, "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()}
        return {
            "policy": POLICY,
            "arms": {a: [public(r) for r in rows] for a, rows in self.arms.items()},
            "comparisons": comparisons,
            "recorded_tensor_coverage_complete": not self.truncated,
            "unobserved": list(self.truncated),
            "unobserved_boundary_count": self.unobserved_count,
            "baseline_disk_bytes": self.bytes,
            "loaded_source_files": sources,
            "limits": {
                "baseline_disk_bytes": self.max_bytes,
                "records_per_arm": self.max_records,
                "individual_tensor_bytes": self.max_tensor_bytes,
            },
            "physical_state_isolation_verified": False,
            "limitations": [
                "Callback x is before an update, never a generic post-update callback.",
                "Next live Euler x observes the previous update only with zero churn and matching call order.",
                "Terminal raw state is reconstructed; pre-inverse-scaling solver return is unobserved.",
                "Native witnesses observe the installed diffusion wrapper, not individual transformer blocks.",
                "Backend digests can be partial; occupancy and lifetime identity do not prove cache equality.",
            ],
        }

    @staticmethod
    def call_identity(row):
        return (row["stage"], row["evaluation"], *(row["metadata"].get(k) for k in ("actual", "phase", "outer_step")))

    @staticmethod
    def classify(row):
        boundary = row["boundary"]
        if row["stage"] == "probe":
            return "probe_" + boundary
        if boundary == "solver_input":
            return "solver_interval_before_mask"
        if boundary == "predict_input":
            return "post_mask_model_input"
        if boundary == "predict_output":
            return "actual_prediction" if row["metadata"]["actual"] else "forecast_prediction"
        return boundary

    def divergent_operands(self):
        saved = self._first_operands.get("A_replay")
        first = self.summary()["comparisons"]["A_replay"]["first_observed_divergence"]
        if saved is None or first is None or first.get("key") != saved[0]:
            return {}
        return {
            "first_divergence_A": torch.load(saved[1], map_location="cpu", weights_only=True),
            "first_divergence_A_replay": torch.load(saved[2], map_location="cpu", weights_only=True),
        }


@contextlib.contextmanager
def trace_stage(trace, arm, stage, sampler, guider):
    if trace is None:
        yield
        return
    if _ACTIVE.get() is not None:
        raise RuntimeError("nested low-replay instrumentation is unsupported")
    trace.arm, trace.stage, trace.evaluation = arm, stage, -1
    trace.native_call = 0
    trace._native_count = 0
    function = getattr(sampler, "sampler_function", None)
    trace.sampler_code = getattr(inspect.unwrap(function), "__code__", None) if callable(function) else None
    trace.arms.setdefault(arm, [])
    options = guider.model_options["transformer_options"]
    missing = object()
    previous = options.get("wrappers", missing)
    wrappers = dict(previous) if previous is not missing else {}
    diffusion = dict(wrappers.get("diffusion_model", {}))
    diffusion[POLICY] = [native_trace_wrapper]
    wrappers["diffusion_model"] = diffusion
    options["wrappers"] = wrappers
    token = _ACTIVE.set(trace)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        if previous is missing:
            options.pop("wrappers", None)
        else:
            options["wrappers"] = previous


def active_trace():
    return _ACTIVE.get()


def native_trace_wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
    trace = active_trace()
    if trace is None:
        return executor(x, timestep, context, transformer_options, **kwargs)
    trace._native_count += 1
    trace.native_call = trace._native_count
    conditioning = _digest({"context": context, "kwargs": kwargs})
    trace.record("native_input_video", x[0], conditioning=conditioning)
    trace.record("native_input_audio", x[1])
    trace.record("native_timestep", timestep)
    result = executor(x, timestep, context, transformer_options, **kwargs)
    trace.record("native_output_video", result[0])
    trace.record("native_output_audio", result[1])
    # Reset call ordinal for logical predictor output, after any batched native calls.
    trace.native_call = 0
    return result


def current_trace_owner():
    return _OWNER.get()


@contextlib.contextmanager
def low_trace_scope():
    if _OWNER.get() is not None:
        yield _OWNER.get()
        return
    trace = LowReplayTrace()
    token = _OWNER.set(trace)
    try:
        yield trace
    finally:
        _OWNER.reset(token)
        trace.close()


def traced_partitioned(function):
    from functools import wraps

    @wraps(function)
    def wrapped(executor, guider, *args, **kwargs):
        if not guider.model_options.get("h3_flow_low_sampler_aba_v1", False):
            return function(executor, guider, *args, **kwargs)
        with low_trace_scope():
            return function(executor, guider, *args, **kwargs)

    return wrapped


def record_terminal(trace, arm, stage, boundary, tensor):
    if trace is not None:
        trace.arm, trace.stage = arm, stage
        trace.native_call = 0
        trace.record(boundary, tensor)
