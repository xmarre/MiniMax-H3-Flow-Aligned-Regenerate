"""Post-load lifecycle and active companion observation for first-high diagnostics.

This module extends the bounded execution-contract diagnostic with read-only
observation at two boundaries the base recorder cannot establish from its
pre-load manifest alone:

* after Core ``prepare_sampling`` has loaded/injected each low/probe/high sampler
  lifetime; and
* while the selected low-last/probe/high-first/high-last DIFFUSION_MODEL calls
  are inside companion-owned runtime scopes.

It composes ordinary ModelPatcher wrappers. It does not replace process-global
functions, execute an extra H3 call, invoke the learned upscaler, mutate solver
history, or reset companion state.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

from . import execution_contract_diagnostics as _diag
from . import execution_contract_provenance as _provenance
from . import runtime as _runtime

_EXTENSION_KEY = "h3_flow_execution_contract_runtime_observation_v1"
_PREPARE_KEY = "h3_flow_regenerate.exec_contract.prepare_lifecycle.v1"
_DIFFUSION_KEY = "h3_flow_regenerate.exec_contract.companion_runtime.v1"
_REQUIRED_STAGES = ("low", "probe", "high")
_REQUIRED_COMPANION_SLOTS = ("low_last", "probe", "high_first", "high_last")
_MAX_MODIFIED_MODULES = 1024


def _runtime_callable_signature(value: Any) -> dict[str, Any]:
    identity = _provenance.callable_identity(value)
    source = identity.get("file") or {}
    material = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
    owner = getattr(value, "__self__", None)
    owner_summary = None
    if owner is not None:
        owner_summary = {"type": f"{type(owner).__module__}.{type(owner).__qualname__}"}
        multiplier = getattr(owner, "multiplier", None)
        if isinstance(multiplier, (bool, int, float, str, type(None))):
            owner_summary["multiplier"] = multiplier
        adapter = getattr(owner, "adapter", None)
        if adapter is not None:
            owner_summary["adapter_type"] = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
        module = getattr(owner, "module", None)
        if module is not None:
            owner_summary["module_type"] = f"{type(module).__module__}.{type(module).__qualname__}"
    return {
        "module": identity.get("module"),
        "qualname": identity.get("qualname"),
        "code_digest": identity.get("code_digest"),
        "identity_digest": hashlib.sha256(material).hexdigest(),
        "source_sha256": source.get("sha256"),
        "bound_owner": owner_summary,
    }


def _module_hook_state(module: Any) -> dict[str, Any] | None:
    if module is None:
        return None
    forward = getattr(module, "forward", None)
    pre_hooks = (getattr(module, "_forward_pre_hooks", {}) or {}).values()
    post_hooks = (getattr(module, "_forward_hooks", {}) or {}).values()
    return {
        "type": f"{type(module).__module__}.{type(module).__qualname__}",
        "forward": _runtime_callable_signature(forward) if callable(forward) else None,
        "pre_hooks": [_runtime_callable_signature(hook) for hook in pre_hooks],
        "post_hooks": [_runtime_callable_signature(hook) for hook in post_hooks],
    }


def _keyed_callable_manifest(mapping: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(mapping, dict):
        return result
    for call_type, keyed in mapping.items():
        ordered = []
        if not isinstance(keyed, dict):
            continue
        for key, callables in keyed.items():
            for value in callables:
                ordered.append({"key": str(key), "callable": _runtime_callable_signature(value)})
        result[str(call_type)] = ordered
    return result


def _injection_manifest(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    injections = getattr(model, "injections", {})
    if not isinstance(injections, dict):
        return result
    for key, entries in injections.items():
        result[str(key)] = [
            {
                "inject": _runtime_callable_signature(entry.inject),
                "eject": _runtime_callable_signature(entry.eject),
            }
            for entry in entries
            if callable(getattr(entry, "inject", None)) and callable(getattr(entry, "eject", None))
        ]
    return result


def _object_patch_manifest(model: Any) -> dict[str, Any]:
    patches = getattr(model, "object_patches", {})
    if not isinstance(patches, dict):
        return {}
    return {
        str(key): (
            {"callable": _runtime_callable_signature(value)} if callable(value) else _provenance.safe_value(value)
        )
        for key, value in sorted(patches.items(), key=lambda pair: str(pair[0]))
    }


def _modified_module_manifest(diffusion: Any) -> dict[str, Any]:
    """Capture loaded instance-forward replacements and native forward hooks.

    VDN object patches and Comfy bypass adapters replace nested module ``forward``
    methods at runtime. Looking only at transformer blocks would miss those paths.
    The list is explicitly bounded; truncation fails the structural observation gate.
    """

    named_modules = getattr(diffusion, "named_modules", None)
    if not callable(named_modules):
        return {"count": 0, "limit": _MAX_MODIFIED_MODULES, "truncated": False, "digest": None, "modules": []}

    entries = []
    observed = 0
    for path, module in named_modules():
        namespace = getattr(module, "__dict__", None)
        instance_forward = isinstance(namespace, dict) and "forward" in namespace
        pre_hooks = getattr(module, "_forward_pre_hooks", {}) or {}
        post_hooks = getattr(module, "_forward_hooks", {}) or {}
        if not instance_forward and not pre_hooks and not post_hooks:
            continue
        observed += 1
        if len(entries) >= _MAX_MODIFIED_MODULES:
            continue
        state = _module_hook_state(module)
        if state is None:
            continue
        entries.append(
            {
                "path": str(path or "$"),
                "instance_forward_override": bool(instance_forward),
                **state,
            }
        )

    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"), default=str).encode()
    return {
        "count": observed,
        "limit": _MAX_MODIFIED_MODULES,
        "truncated": observed > _MAX_MODIFIED_MODULES,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "modules": entries,
    }


def model_lifecycle_manifest(
    model: Any,
    *,
    model_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the actually loaded/injected patcher at one sampler lifetime."""

    effective_options = (
        model_options if isinstance(model_options, dict) else (getattr(model, "model_options", None) or {})
    )
    transformer = effective_options.get("transformer_options") or {}
    base = getattr(model, "model", None)
    diffusion = getattr(base, "diffusion_model", None)
    blocks = getattr(diffusion, "blocks", None) or []

    block_state = []
    pre_count = 0
    post_count = 0
    for index, block in enumerate(blocks):
        state = _module_hook_state(block)
        if state is None:
            continue
        pre_count += len(state["pre_hooks"])
        post_count += len(state["post_hooks"])
        block_state.append({"block": index, **state})

    block_encoded = json.dumps(block_state, sort_keys=True, separators=(",", ":"), default=str).encode()
    modified_modules = _modified_module_manifest(diffusion)
    active_injections = _injection_manifest(model)
    active_object_patches = _object_patch_manifest(model)
    effective_wrapper_order = _keyed_callable_manifest(transformer.get("wrappers", {}))
    effective_callbacks = _keyed_callable_manifest(transformer.get("callbacks", {}))
    lifecycle_material = {
        "blocks": block_state,
        "modified_modules": modified_modules["modules"],
        "active_injections": active_injections,
        "active_object_patches": active_object_patches,
        "effective_wrapper_order": effective_wrapper_order,
        "effective_callbacks": effective_callbacks,
    }
    lifecycle_encoded = json.dumps(
        lifecycle_material,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return {
        "patcher_is_injected": bool(getattr(model, "is_injected", False)),
        "load_device": str(getattr(model, "load_device", None)),
        "offload_device": str(getattr(model, "offload_device", None)),
        "active_injections": active_injections,
        "active_object_patches": active_object_patches,
        "effective_wrapper_order": effective_wrapper_order,
        "effective_callbacks": effective_callbacks,
        "diffusion_root": _module_hook_state(diffusion),
        "block_count": len(block_state),
        "block_pre_hook_count": pre_count,
        "block_post_hook_count": post_count,
        "block_runtime_digest": hashlib.sha256(block_encoded).hexdigest(),
        "runtime_lifecycle_digest": hashlib.sha256(lifecycle_encoded).hexdigest(),
        "modified_modules": modified_modules,
        "blocks": block_state,
    }


def _extension_manifest(state: Any) -> dict[str, Any]:
    return state.manifest.setdefault(
        _EXTENSION_KEY,
        {
            "schema_version": 2,
            "observer_wrapper_keys": {
                "prepare_sampling": [_PREPARE_KEY],
                "diffusion_model": [_DIFFUSION_KEY],
            },
            "stage_lifecycles": [],
            "diffusion_actual_counts": {},
            "companion_calls": {},
            "observation_errors": [],
        },
    )


def _record_observation_error(state: Any, scope: str, exc: Exception) -> str:
    message = f"{scope}: {type(exc).__name__}: {exc}"
    extension = _extension_manifest(state)
    if message not in extension["observation_errors"]:
        extension["observation_errors"].append(message)
    return message


def _stage_name(model_options: dict[str, Any] | None) -> str:
    stage = _diag._STAGE.get()
    if stage is not None:
        return str(stage)
    return _diag._stage_name(model_options)


def _prepare_sampling_wrapper(
    executor,
    model,
    noise_shape,
    conds,
    model_options=None,
    force_full_load=False,
    force_offload=False,
):
    result = executor(
        model,
        noise_shape,
        conds,
        model_options=model_options,
        force_full_load=force_full_load,
        force_offload=force_offload,
    )
    record = _diag._ACTIVE.get()
    if record is None:
        return result

    additional_models = []
    if isinstance(result, tuple) and len(result) >= 3 and isinstance(result[2], (list, tuple)):
        for item in result[2]:
            additional_models.append(
                {
                    "type": f"{type(item).__module__}.{type(item).__qualname__}",
                    "patcher_is_injected": bool(getattr(item, "is_injected", False)),
                }
            )

    entry = {
        "stage": _stage_name(model_options),
        "capture_phase": "post_prepare_sampling_load",
        "additional_models": additional_models,
    }
    try:
        entry["model"] = model_lifecycle_manifest(model, model_options=model_options)
    except Exception as exc:
        entry["model"] = {}
        entry["observation_error"] = _record_observation_error(record.state, "post_prepare_sampling_load", exc)
    _extension_manifest(record.state)["stage_lifecycles"].append(entry)
    return result


def _closure_values(value: Any) -> list[tuple[str, Any]]:
    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    closure = getattr(target, "__closure__", None)
    if code is None or closure is None:
        return []
    result = []
    for name, cell in zip(code.co_freevars, closure, strict=True):
        try:
            captured = cell.cell_contents
        except ValueError:
            continue
        result.append((str(name), captured))
    return result


_MAX_VDN_CACHE_KEYS = 16
_VDN_CACHE_FIELDS = (
    ("scan", "_scan"),
    ("delta", "_delta"),
    ("plans", "_plans"),
    ("kv", "_kv"),
    ("activations", "_activations"),
)


def _vdn_retained_cache_snapshot(resources: Any) -> dict[str, Any]:
    """Describe bounded cache topology without reading or hashing scratch tensors."""
    if resources is None:
        return {}
    result: dict[str, Any] = {}
    for label, attribute in _VDN_CACHE_FIELDS:
        mapping = getattr(resources, attribute, None)
        keys = getattr(mapping, "keys", None)
        if not callable(keys):
            continue
        values = list(keys())
        result[label] = {
            "count": len(values),
            "limit": _MAX_VDN_CACHE_KEYS,
            "truncated": len(values) > _MAX_VDN_CACHE_KEYS,
            "keys": [_provenance.safe_value(value) for value in values[:_MAX_VDN_CACHE_KEYS]],
        }
    return result


def _vdn_prefetch_snapshot(resources: Any) -> dict[str, Any] | None:
    """Observe VDN's retained one-block prefetch lifecycle without consuming it."""
    if resources is None:
        return None
    prefetcher = getattr(resources, "_prefetcher", None)
    if prefetcher is None:
        return {"present": False}
    future = getattr(prefetcher, "_future", None)
    done = getattr(future, "done", None)
    return {
        "present": True,
        "generation": _provenance.safe_value(getattr(prefetcher, "_generation", None)),
        "target": _provenance.safe_value(getattr(prefetcher, "_index", None)),
        "future_present": future is not None,
        "future_done": bool(done()) if callable(done) else None,
    }


def _vdn_runtime_snapshot(executor: Any) -> list[dict[str, Any]]:
    found = []
    seen: set[int] = set()
    for wrapper in getattr(executor, "wrappers", ()) or ():
        for capture_name, captured in _closure_values(wrapper):
            ident = id(captured)
            if ident in seen or not type(captured).__module__.startswith("vdn_h3"):
                continue
            if not hasattr(captured, "layout") or not hasattr(captured, "runtime"):
                continue
            seen.add(ident)
            layout = getattr(captured, "layout", None)
            layout_fields = {}
            if layout is not None:
                for name in (
                    "video_start",
                    "video_end",
                    "audio_start",
                    "audio_end",
                    "num_frames",
                    "tokens_per_frame",
                    "frame_size",
                    "text_start",
                    "text_len",
                    "bounds",
                    "full_cover",
                    "seq_len",
                    "anchor_frames",
                ):
                    if hasattr(layout, name):
                        layout_fields[name] = _provenance.safe_value(getattr(layout, name))

            runtime = getattr(captured, "runtime", None)
            current = getattr(runtime, "current", None)
            resources = current() if callable(current) else None
            retained_counts = getattr(resources, "retained_counts", None)
            generation = None if resources is None else getattr(resources, "generation", None)
            found.append(
                {
                    "capture": capture_name,
                    "state_type": f"{type(captured).__module__}.{type(captured).__qualname__}",
                    "owner_wrapper": _runtime_callable_signature(wrapper),
                    "layout_active": layout is not None,
                    "layout": layout_fields,
                    "retain_buffers": bool(getattr(captured, "retain_buffers", False)),
                    "runtime_pool_active": resources is not None,
                    "runtime_pool_retain": None if resources is None else bool(getattr(resources, "retain", False)),
                    "runtime_pool_generation": _provenance.safe_value(generation),
                    "retained_counts": retained_counts() if callable(retained_counts) else None,
                    "retained_cache_keys": _vdn_retained_cache_snapshot(resources),
                    "prefetch": _vdn_prefetch_snapshot(resources),
                }
            )
    return found


def _contextvar_value(module: Any, name: str) -> Any:
    variable = getattr(module, name, None)
    getter = getattr(variable, "get", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except LookupError:
        return None


def _loaded_sol_runtime_module() -> tuple[str | None, Any | None]:
    """Resolve the unique already-loaded Sol runtime without importing another copy."""

    candidates: dict[int, tuple[str, Any]] = {}
    for name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        normalized_name = str(name).replace("\\", "/").lower()
        normalized_file = str(getattr(module, "__file__", "") or "").replace("\\", "/").lower()
        name_match = normalized_name == "sol_h3.runtime" or normalized_name.endswith(".sol_h3.runtime")
        file_match = normalized_file.endswith("/sol_h3/runtime.py")
        if not (name_match or file_match):
            continue
        if not hasattr(module, "_REQUEST") or not hasattr(module, "_FORWARD"):
            continue
        candidates[id(module)] = (str(name), module)
    if not candidates:
        return None, None
    if len(candidates) != 1:
        names = sorted(name for name, _module in candidates.values())
        raise RuntimeError(f"multiple loaded Sol-H3 runtime modules: {names}")
    return next(iter(candidates.values()))


def _sol_runtime_snapshot() -> dict[str, Any] | None:
    module_name, module = _loaded_sol_runtime_module()
    if module is None:
        return None
    request = _contextvar_value(module, "_REQUEST")
    forward = _contextvar_value(module, "_FORWARD")
    result: dict[str, Any] = {
        "module_loaded": True,
        "module_name": module_name,
        "module_file": str(getattr(module, "__file__", None)),
        "request_active": request is not None,
        "forward_active": forward is not None,
    }
    if request is not None:
        for name in (
            "evaluations",
            "eligible_calls",
            "sparse_calls",
            "dense_calls",
            "external_mixed_sol_calls",
            "external_mixed_q_rows",
            "external_mixed_kernel_q_rows",
            "external_mixed_measure_calls",
            "external_mixed_measure_q_rows",
            "external_mixed_measure_kv_rows_before",
            "external_mixed_measure_kv_rows_after",
            "external_mixed_measure_removed_rows",
            "external_mixed_weighted_measure_calls",
            "external_mixed_weighted_measure_q_rows",
            "external_mixed_weighted_measure_kv_rows",
            "vdn_local_sol_calls",
            "vdn_rectangular_sol_calls",
            "vdn_requested_q_rows",
            "vdn_kernel_q_rows",
            "vdn_square_expanded_calls",
            "vdn_square_requested_rows",
            "vdn_square_kernel_rows",
            "exact_blocks",
            "backend_transitions",
            "last_routes",
            "gates",
            "fallbacks",
            "dense_provider_failures",
        ):
            if hasattr(request, name):
                result[name] = _provenance.safe_value(getattr(request, name))
        dense_backends = getattr(request, "dense_attention_backends", None)
        if isinstance(dense_backends, set):
            result["dense_attention_backends"] = sorted(str(item) for item in dense_backends)
        kernel = getattr(request, "kernel", None)
        if kernel is not None:
            result["kernel"] = {
                "backend_name": getattr(kernel, "backend_name", None),
                "source_tree_verified": bool(getattr(kernel, "source_tree_verified", False)),
            }
    if isinstance(forward, tuple) and len(forward) >= 5:
        seen_blocks = forward[3]
        result["forward"] = {
            "evaluation": _provenance.safe_value(forward[2]),
            "seen_blocks": (
                sorted(int(item) for item in seen_blocks)
                if isinstance(seen_blocks, set)
                else _provenance.safe_value(seen_blocks)
            ),
            "routes": _provenance.safe_value(forward[4]),
        }
    return result


def _spectrum_runtime_snapshot(root_model_options: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(root_model_options, dict):
        return None
    binding = root_model_options.get(_runtime.SPECTRUM_BINDING_KEY)
    runtime = getattr(binding, "runtime", None)
    if runtime is None:
        return None
    step = getattr(runtime, "_step", None)
    run = getattr(runtime, "_run", None)
    stats = getattr(runtime, "stats", None)
    result = {
        "active_run_id": getattr(runtime, "active_run_id", None),
        "active_step_id": getattr(runtime, "active_step_id", None),
        "active_policy_step_id": getattr(runtime, "active_policy_step_id", None),
        "active_stage_index": getattr(runtime, "active_stage_index", None),
        "active_solver_phase": getattr(runtime, "active_solver_phase", None),
        "prediction_history_length": getattr(runtime, "prediction_history_length", None),
        "history_labels": _provenance.safe_value(getattr(runtime, "history_labels", None)),
        "last_completed_mode": getattr(runtime, "last_completed_mode", None),
        "last_completed_step_id": getattr(runtime, "last_completed_step_id", None),
        "last_completed_reason": getattr(runtime, "last_completed_reason", None),
    }
    if step is not None:
        result["step"] = {
            name: _provenance.safe_value(getattr(step, name))
            for name in (
                "step_id",
                "policy_step_id",
                "coordinate",
                "mode",
                "reason",
                "stage_index",
                "phase",
                "retain_history",
                "bootstrap_forecast",
                "fallback",
            )
            if hasattr(step, name)
        }
    if run is not None:
        result["run"] = {
            name: _provenance.safe_value(getattr(run, name))
            for name in (
                "run_id",
                "sampler_name",
                "total_steps",
                "policy_steps",
                "stage_count",
                "next_step_id",
                "separate_stage_histories",
                "min_actual_prefix_steps",
                "min_sampler_actual_prefix_steps",
            )
            if hasattr(run, name)
        }
    if stats is not None:
        result["stats"] = {
            name: _provenance.safe_value(getattr(stats, name))
            for name in (
                "actual_steps",
                "forecast_steps",
                "actual_transformer_calls",
                "forecast_model_calls",
                "backend_history_resets",
                "backend_opaque_actual_calls",
            )
            if hasattr(stats, name)
        }
    return result


def _active_companion_snapshot(
    executor: Any,
    transformer: dict[str, Any] | None,
    root_model_options: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "transformer_runtime": _diag._runtime_snapshot({"transformer_options": transformer or {}}),
        "vdn": _vdn_runtime_snapshot(executor),
        "sol": _sol_runtime_snapshot(),
        "spectrum": _spectrum_runtime_snapshot(root_model_options),
    }


def _guarded_companion_snapshot(
    state: Any,
    scope: str,
    executor: Any,
    transformer: dict[str, Any] | None,
    root_model_options: dict[str, Any] | None,
) -> tuple[dict[str, Any], str | None]:
    try:
        return _active_companion_snapshot(executor, transformer, root_model_options), None
    except Exception as exc:
        return {}, _record_observation_error(state, scope, exc)


def _store_companion_call(
    state: Any,
    stage: str,
    before: dict[str, Any],
    after: dict[str, Any],
    errors: list[str],
) -> None:
    extension = _extension_manifest(state)
    counts = extension["diffusion_actual_counts"]
    count = int(counts.get(stage, 0)) + 1
    counts[stage] = count
    entry = {
        "stage": stage,
        "actual_index": count,
        "before": before,
        "after": after,
        "observation_errors": errors,
    }
    slots = extension["companion_calls"]
    if stage == "low":
        slots["low_last"] = entry
    elif stage == "probe":
        slots["probe"] = entry
    elif stage == "high":
        if count == 1:
            slots["high_first"] = entry
        slots["high_last"] = entry


def _make_diffusion_wrapper(root_model_options: dict[str, Any] | None = None):
    def wrapper(executor, *args, **kwargs):
        record = _diag._ACTIVE.get()
        stage = _diag._STAGE.get()
        if record is None or stage not in _REQUIRED_STAGES:
            return executor(*args, **kwargs)

        transformer = kwargs.get("transformer_options")
        if transformer is None:
            for candidate in reversed(args):
                if isinstance(candidate, dict) and _runtime.FLOW_STAGE_KEY in candidate:
                    transformer = candidate
                    break

        active_model_options = getattr(record, "active_model_options", None)
        if not isinstance(active_model_options, dict):
            active_model_options = root_model_options if isinstance(root_model_options, dict) else None
        before, before_error = _guarded_companion_snapshot(
            record.state,
            f"{stage}:before_diffusion_model",
            executor,
            transformer,
            active_model_options,
        )
        result = executor(*args, **kwargs)
        after, after_error = _guarded_companion_snapshot(
            record.state,
            f"{stage}:after_diffusion_model",
            executor,
            transformer,
            active_model_options,
        )
        errors = [item for item in (before_error, after_error) if item is not None]
        _store_companion_call(record.state, str(stage), before, after, errors)
        return result

    return wrapper


def _runtime_observation_gate(report: dict[str, Any]) -> dict[str, Any]:
    provenance = report.get("provenance") or {}
    extension = provenance.get(_EXTENSION_KEY) or {}
    lifecycles = extension.get("stage_lifecycles") or []
    companion_calls = extension.get("companion_calls") or {}
    observation_errors = extension.get("observation_errors") or []

    by_stage: dict[str, list[dict[str, Any]]] = {}
    for entry in lifecycles:
        if isinstance(entry, dict):
            by_stage.setdefault(str(entry.get("stage")), []).append(entry)

    stage_lifecycle_counts = {stage: len(by_stage.get(stage, [])) for stage in _REQUIRED_STAGES}
    stage_lifecycle_complete = all(stage_lifecycle_counts[stage] == 1 for stage in _REQUIRED_STAGES)
    injection_state_consistent = stage_lifecycle_complete and all(
        not entry.get("model", {}).get("active_injections") or bool(entry.get("model", {}).get("patcher_is_injected"))
        for stage in _REQUIRED_STAGES
        for entry in by_stage.get(stage, [])
    )
    lifecycle_digests = [
        entry.get("model", {}).get("runtime_lifecycle_digest")
        for stage in _REQUIRED_STAGES
        for entry in by_stage.get(stage, [])
        if entry.get("model", {}).get("runtime_lifecycle_digest")
    ]
    hook_runtime_stable = len(lifecycle_digests) == len(_REQUIRED_STAGES) and len(set(lifecycle_digests)) == 1
    modified_states = [
        entry.get("model", {}).get("modified_modules") or {}
        for stage in _REQUIRED_STAGES
        for entry in by_stage.get(stage, [])
    ]
    modified_runtime_visible = bool(modified_states) and all(
        int(state.get("count", 0)) > 0 and not bool(state.get("truncated")) for state in modified_states
    )
    companion_slots_complete = all(slot in companion_calls for slot in _REQUIRED_COMPANION_SLOTS)
    expected_actual_counts = {"low": 4, "probe": 1, "high": 2}
    actual_counts = extension.get("diffusion_actual_counts") or {}
    observed_actual_counts = {stage: int(actual_counts.get(stage, 0)) for stage in _REQUIRED_STAGES}
    actual_call_counts_exact = observed_actual_counts == expected_actual_counts

    high_first = companion_calls.get("high_first") or {}
    before = high_first.get("before") or {}
    after = high_first.get("after") or {}
    sol_visible = any(
        bool(snapshot.get("request_active")) and bool(snapshot.get("forward_active"))
        for snapshot in (before.get("sol") or {}, after.get("sol") or {})
        if isinstance(snapshot, dict)
    )
    vdn_visible = any(
        bool(item.get("layout_active")) and bool(item.get("runtime_pool_active"))
        for snapshot in (before, after)
        for item in (snapshot.get("vdn") or [])
        if isinstance(item, dict)
    )
    spectrum_visible = any(
        snapshot.get("spectrum", {}).get("active_run_id") is not None
        and snapshot.get("spectrum", {}).get("active_step_id") is not None
        for snapshot in (before, after)
        if isinstance(snapshot, dict) and isinstance(snapshot.get("spectrum"), dict)
    )
    observation_error_free = not observation_errors

    research_fields = (
        "external_mixed_sol_calls",
        "external_mixed_q_rows",
        "external_mixed_kernel_q_rows",
        "external_mixed_measure_calls",
        "external_mixed_measure_q_rows",
        "external_mixed_measure_kv_rows_before",
        "external_mixed_measure_kv_rows_after",
        "external_mixed_measure_removed_rows",
        "external_mixed_weighted_measure_calls",
        "external_mixed_weighted_measure_q_rows",
        "external_mixed_weighted_measure_kv_rows",
    )
    research_receipts: dict[str, Any] = {}
    missing_research_receipts: list[str] = []
    nonzero_research_receipts: dict[str, Any] = {}
    for slot in _REQUIRED_COMPANION_SLOTS:
        call = companion_calls.get(slot) or {}
        for phase in ("before", "after"):
            sol = (call.get(phase) or {}).get("sol")
            if not isinstance(sol, dict):
                missing_research_receipts.append(f"{slot}.{phase}.sol")
                continue
            for field in research_fields:
                key = f"{slot}.{phase}.{field}"
                if field not in sol:
                    missing_research_receipts.append(key)
                    continue
                value = sol[field]
                research_receipts[key] = value
                if not isinstance(value, (bool, int, float)) or float(value) != 0.0:
                    nonzero_research_receipts[key] = value
    sol_research_receipts_complete = not missing_research_receipts
    sol_research_zero = sol_research_receipts_complete and not nonzero_research_receipts

    runtime_contract_complete = all(
        (
            stage_lifecycle_complete,
            injection_state_consistent,
            hook_runtime_stable,
            modified_runtime_visible,
            companion_slots_complete,
            actual_call_counts_exact,
            sol_visible,
            vdn_visible,
            spectrum_visible,
            observation_error_free,
            sol_research_receipts_complete,
            sol_research_zero,
        )
    )
    return {
        "stage_lifecycle_complete": stage_lifecycle_complete,
        "stage_lifecycle_counts": stage_lifecycle_counts,
        "injection_state_consistent": injection_state_consistent,
        "hook_runtime_stable": hook_runtime_stable,
        "runtime_lifecycle_digests": lifecycle_digests,
        "modified_runtime_visible": modified_runtime_visible,
        "modified_module_counts": [int(state.get("count", 0)) for state in modified_states],
        "modified_module_truncated": [bool(state.get("truncated")) for state in modified_states],
        "companion_slots_complete": companion_slots_complete,
        "actual_call_counts_exact": actual_call_counts_exact,
        "expected_actual_call_counts": expected_actual_counts,
        "observed_actual_call_counts": observed_actual_counts,
        "sol_active_high_first_visible": sol_visible,
        "vdn_active_high_first_visible": vdn_visible,
        "spectrum_active_high_first_visible": spectrum_visible,
        "observation_error_free": observation_error_free,
        "observation_errors": observation_errors,
        "sol_research_receipts_complete": sol_research_receipts_complete,
        "sol_research_zero": sol_research_zero,
        "sol_research_receipts": research_receipts,
        "missing_sol_research_receipts": missing_research_receipts,
        "nonzero_sol_research_receipts": nonzero_research_receipts,
        "runtime_contract_complete": runtime_contract_complete,
    }


class H3ExecutionContractDiagnostics(_diag.H3ExecutionContractDiagnostics):
    DESCRIPTION = (
        _diag.H3ExecutionContractDiagnostics.DESCRIPTION
        + " It also records post-load lifecycle and active VDN/Sol/Spectrum runtime state."
    )

    def apply(self, model, capture_mib, strict_provenance):
        patched, state = _diag.patch_execution_contract_diagnostics(
            model,
            capture_mib=int(capture_mib),
            strict_provenance=bool(strict_provenance),
        )
        import comfy.patcher_extension

        _diag._append_wrapper(
            patched,
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            _PREPARE_KEY,
            _prepare_sampling_wrapper,
        )
        _diag._append_wrapper(
            patched,
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _DIFFUSION_KEY,
            _make_diffusion_wrapper(),
        )
        _extension_manifest(state)
        return patched, state, json.dumps(state.manifest, indent=2, sort_keys=True, default=str)


class H3ExecutionContractReport(_diag.H3ExecutionContractReport):
    def extract(self, diagnostic, trigger):
        _ = trigger
        if not isinstance(diagnostic, _diag._State):
            raise TypeError("invalid H3 execution-contract diagnostic handle")
        if not diagnostic.complete:
            raise RuntimeError("no completed execution-contract capture is available")
        report = diagnostic.complete.pop()
        runtime_gate = _runtime_observation_gate(report)
        observation_gate = report.setdefault("observation_gate", {})
        base_structural = bool(observation_gate.get("structural_candidate"))
        observation_gate["base_structural_candidate"] = base_structural
        observation_gate["runtime_contract"] = runtime_gate
        observation_gate["structural_candidate"] = bool(base_structural and runtime_gate["runtime_contract_complete"])
        return (json.dumps(report, indent=2, sort_keys=True, default=str),)


NODE_CLASS_MAPPINGS = {
    "H3ExecutionContractDiagnostics": H3ExecutionContractDiagnostics,
    "H3ExecutionContractReport": H3ExecutionContractReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ExecutionContractDiagnostics": "MiniMax H3 Execution Contract Diagnostics",
    "H3ExecutionContractReport": "MiniMax H3 Execution Contract Report",
}
