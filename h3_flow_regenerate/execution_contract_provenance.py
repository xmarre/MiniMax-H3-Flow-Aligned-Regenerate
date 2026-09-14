"""Installed-runtime provenance for first-high H3 contract diagnostics.

The collector runs inside the active ComfyUI process. It records imported
source files, git state, effective wrapper/replacement order, hook identities,
attention/provider callables, compiler/runtime flags, and a bounded fingerprint
of the actually loaded H3 model. It deliberately does not hash multi-gigabyte
checkpoint files on the diagnostic hot path.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from . import comfy_compat as _comfy_compat
from . import runtime as _runtime

_REQUIRED_MODULES = (
    "comfy.samplers",
    "comfy.model_base",
    "comfy.patcher_extension",
    "comfy.ldm.minimax.model",
)

_COMPANION_PATH_MARKERS: dict[str, tuple[str, ...]] = {
    "flow": ("minimax-h3-flow-aligned-regenerate",),
    "sol_h3": ("comfyui-sol-h3",),
    "vdn_h3": ("comfyui-vdn-h3-plus",),
    "spectrum_h3": ("comfyui-spectrum-minimax-h3",),
    "diffaid": ("comfyui-diffaid-patches",),
    "untwist": ("comfyui-untwisting-rope", "comfyui-flux2-untwisting-rope"),
    "kj": ("comfyui-kjnodes",),
    "latent_upscaler": (
        "comfyui_minimax_h3_latent_upscaler-plus",
        "comfyui-minimax-h3-latent-upscaler-plus",
    ),
}
_OPTIONAL_COMPANION_PATH_MARKERS: dict[str, tuple[str, ...]] = {
    "dora_loader": ("comfyui-dora-dynamic-lora-loader",),
}
_MAX_COMPANION_FILES = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(directory: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


@functools.lru_cache(maxsize=64)
def _repo_state(root: str) -> dict[str, Any]:
    directory = Path(root)
    head = _run_git(directory, "rev-parse", "HEAD")
    status = _run_git(directory, "status", "--porcelain", "--untracked-files=normal")
    return {
        "available": head is not None and status is not None,
        "root": root,
        "head": head,
        "dirty": None if status is None else bool(status),
    }


@functools.lru_cache(maxsize=256)
def file_identity(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        source = Path(path)
        resolved = source.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None
    root = _run_git(resolved.parent, "rev-parse", "--show-toplevel")
    git = _repo_state(root) if root else {"available": False, "root": None, "head": None, "dirty": None}
    return {
        "path": str(source),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "git": git,
    }


def _captured_value_identity(value: Any, *, depth: int, seen: set[int]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch.is_tensor(value):
        return {
            "tensor": True,
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if callable(value):
        return {"callable": callable_identity(value, _depth=depth + 1, _seen=seen)}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def callable_identity(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> dict[str, Any]:
    target = getattr(value, "__func__", value)
    cls = target if inspect.isclass(target) else type(target)
    module = getattr(target, "__module__", cls.__module__)
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", cls.__qualname__))
    source_target = target
    try:
        source_file = inspect.getsourcefile(source_target) or inspect.getfile(source_target)
    except (OSError, TypeError):
        source_target = cls
        try:
            source_file = inspect.getsourcefile(source_target) or inspect.getfile(source_target)
        except (OSError, TypeError):
            source_file = None
    code = getattr(target, "__code__", None)
    closure = getattr(target, "__closure__", None)
    defaults = getattr(target, "__defaults__", None)
    kwdefaults = getattr(target, "__kwdefaults__", None)
    if code is None and callable(target):
        call_method = target.__call__
        code = getattr(call_method, "__code__", None)
        closure = getattr(call_method, "__closure__", None)
        defaults = getattr(call_method, "__defaults__", None)
        kwdefaults = getattr(call_method, "__kwdefaults__", None)
    code_digest = None
    if code is not None:
        material = code.co_code + repr(code.co_consts).encode() + repr(code.co_names).encode()
        code_digest = hashlib.sha256(material).hexdigest()
    result = {
        "module": str(module),
        "qualname": str(qualname),
        "code_digest": code_digest,
        "file": file_identity(source_file),
    }
    if _depth >= 2 or code is None:
        return result
    seen = set() if _seen is None else _seen
    ident = id(target)
    if ident in seen:
        result["recursive_capture"] = True
        return result
    seen.add(ident)
    try:
        captures = {}
        if closure is not None:
            for name, cell in zip(code.co_freevars, closure, strict=True):
                try:
                    captured = cell.cell_contents
                except ValueError:
                    captures[name] = {"empty_cell": True}
                else:
                    captures[name] = _captured_value_identity(captured, depth=_depth, seen=seen)
        if captures:
            result["closure"] = captures
        if defaults:
            result["defaults"] = [_captured_value_identity(item, depth=_depth, seen=seen) for item in defaults]
        if kwdefaults:
            result["kwdefaults"] = {
                str(key): _captured_value_identity(item, depth=_depth, seen=seen)
                for key, item in sorted(kwdefaults.items())
            }
    finally:
        seen.remove(ident)
    return result


def safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch.is_tensor(value):
        return {
            "tensor": True,
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sample_digest": _runtime._tensor_signature(value, max_values=64).hex(),
        }
    if isinstance(value, dict):
        return {
            str(key): safe_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "uuid"
        }
    if isinstance(value, (list, tuple)):
        return [safe_value(item, depth=depth + 1) for item in value]
    if dataclasses.is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: safe_value(getattr(value, field.name), depth=depth + 1)
                for field in dataclasses.fields(value)
                if field.name != "learned_upscaler"
            },
        }
    if hasattr(value, "cond"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "cond": safe_value(value.cond, depth=depth + 1),
        }
    if type(value).__module__.startswith("comfy.ldm.minimax") and hasattr(value, "__dict__"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                str(key): safe_value(item, depth=depth + 1)
                for key, item in sorted(vars(value).items())
                if not str(key).startswith("_")
            },
        }
    if callable(value):
        return {"callable": callable_identity(value)}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def fingerprint(value: Any) -> tuple[str, Any]:
    normalized = safe_value(value)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest(), normalized


def alias_graph(value: Any, *, max_depth: int = 8, max_nodes: int = 4096) -> list[dict[str, Any]]:
    """Return stable labels for repeated runtime object aliases.

    Raw object ids are deliberately not serialized. They are only used while
    building invocation-local equivalence classes; reports contain deterministic
    alias labels and structural paths, which are meaningful only within a run.
    """

    seen: dict[int, list[str]] = {}
    types: dict[int, str] = {}
    expanded: set[int] = set()
    node_count = 0

    def visit(item: Any, path: str, depth: int) -> None:
        nonlocal node_count
        if depth > max_depth or node_count >= max_nodes:
            return
        if item is None or isinstance(item, (bool, int, float, str, bytes)):
            return
        node_count += 1
        ident = id(item)
        seen.setdefault(ident, []).append(path)
        types.setdefault(ident, f"{type(item).__module__}.{type(item).__qualname__}")
        if ident in expanded:
            return
        expanded.add(ident)
        if torch.is_tensor(item) or callable(item):
            return
        if isinstance(item, dict):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                if str(key) != "uuid":
                    visit(child, f"{path}.{key}", depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", depth + 1)
            return
        if dataclasses.is_dataclass(item):
            for field in dataclasses.fields(item):
                if field.name != "learned_upscaler":
                    visit(getattr(item, field.name), f"{path}.{field.name}", depth + 1)

    visit(value, "$", 0)
    groups = [(ident, paths) for ident, paths in seen.items() if len(paths) > 1]
    groups.sort(key=lambda pair: pair[1])
    return [
        {"alias": f"alias_{index + 1}", "type": types[ident], "paths": paths}
        for index, (ident, paths) in enumerate(groups)
    ]


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
                ordered.append({"key": str(key), "callable": callable_identity(value)})
        result[str(call_type)] = ordered
    return result


def _wrapper_manifest(model: Any) -> dict[str, Any]:
    return _keyed_callable_manifest(getattr(model, "wrappers", {}))


def _callback_manifest(model: Any) -> dict[str, Any]:
    return _keyed_callable_manifest(getattr(model, "callbacks", {}))


def _injection_manifest(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    injections = getattr(model, "injections", {})
    if not isinstance(injections, dict):
        return result
    for key, entries in injections.items():
        result[str(key)] = [
            {
                "inject": callable_identity(entry.inject),
                "eject": callable_identity(entry.eject),
            }
            for entry in entries
            if hasattr(entry, "inject") and hasattr(entry, "eject")
        ]
    return result


def _object_patch_manifest(model: Any) -> dict[str, Any]:
    patches = getattr(model, "object_patches", {})
    if not isinstance(patches, dict):
        return {}
    return {
        str(key): callable_identity(value) if callable(value) else safe_value(value)
        for key, value in sorted(patches.items(), key=lambda pair: str(pair[0]))
    }


def _runtime_function_manifest() -> dict[str, Any]:
    return {
        "runtime._run_progressive": callable_identity(_runtime._run_progressive),
        "runtime.apply_guidance": callable_identity(_runtime.apply_guidance),
        "runtime.build_handoff_state": callable_identity(_runtime.build_handoff_state),
        "comfy_compat.flow_predict_wrapper": callable_identity(_comfy_compat.flow_predict_wrapper),
    }


def _explicit_attribute(owner: Any, name: str) -> Any:
    """Read only attributes actually declared on an object/class.

    Some Comfy model-config objects implement warning-emitting ``__getattr__``
    fallbacks. Provenance must not manufacture attribute probes just to discover
    a source path.
    """
    if owner is None:
        return None
    try:
        namespace = vars(owner)
    except TypeError:
        namespace = {}
    if name in namespace:
        return namespace[name]
    for cls in type(owner).__mro__:
        if name in vars(cls):
            try:
                return getattr(owner, name)
            except Exception:
                return None
    return None


def _cached_patcher_checkpoint(model: Any) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve a diffusion checkpoint from Comfy's loader reload factory.

    Core ModelPatcher clones preserve ``cached_patcher_init``. KJ's
    DiffusionModelLoaderKJ stores ``(_load_diffusion_model_kj,
    (unet_path, model_options, extra_state_dict))`` there, so arg0 is the exact
    source file even though the loaded MiniMax model/config exposes no pathname.
    """
    cached = _explicit_attribute(model, "cached_patcher_init")
    if not isinstance(cached, tuple) or len(cached) < 2 or not callable(cached[0]):
        return None, None
    args = cached[1]
    if not isinstance(args, tuple) or not args:
        return None, None
    candidate = args[0]
    if not isinstance(candidate, (str, os.PathLike)) or not str(candidate):
        return None, None
    try:
        resolved = Path(candidate).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None, None
    if not resolved.is_file():
        return None, None
    return str(resolved), {
        "kind": "cached_patcher_init_arg0",
        "factory": callable_identity(cached[0]),
        "argument_index": 0,
    }


def _model_fingerprint(model: Any) -> dict[str, Any]:
    base = getattr(model, "model", None)
    diffusion = getattr(base, "diffusion_model", None)
    if diffusion is None:
        return {"available": False}
    parameters = list(diffusion.named_parameters())
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(f"{name}|{tuple(parameter.shape)}|{parameter.dtype}".encode())
    if parameters:
        count = min(12, len(parameters))
        indexes = sorted({round(i * (len(parameters) - 1) / max(count - 1, 1)) for i in range(count)})
        for index in indexes:
            name, parameter = parameters[index]
            digest.update(name.encode())
            digest.update(_runtime._tensor_signature(parameter, max_values=16))

    checkpoint = None
    checkpoint_source = None
    model_config = _explicit_attribute(base, "model_config")
    for owner_name, owner in (
        ("patcher", model),
        ("base", base),
        ("model_config", model_config),
        ("diffusion", diffusion),
    ):
        if owner is None:
            continue
        for attr in ("checkpoint_name", "ckpt_name", "model_path", "model_file", "filename", "source_path"):
            value = _explicit_attribute(owner, attr)
            if isinstance(value, (str, os.PathLike)) and str(value):
                checkpoint = str(value)
                checkpoint_source = {"kind": "declared_attribute", "owner": owner_name, "attribute": attr}
                break
        if checkpoint:
            break
    if checkpoint is None:
        checkpoint, checkpoint_source = _cached_patcher_checkpoint(model)
    checkpoint_stat = None
    if checkpoint:
        try:
            resolved = Path(checkpoint).resolve(strict=True)
            stat = resolved.stat()
            checkpoint_stat = {
                "resolved_path": str(resolved),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError:
            checkpoint_stat = {"resolved_path": None, "size": None, "mtime_ns": None}

    return {
        "available": True,
        "class": f"{type(diffusion).__module__}.{type(diffusion).__qualname__}",
        "parameter_count": len(parameters),
        "runtime_fingerprint": digest.hexdigest(),
        "checkpoint_path": checkpoint,
        "checkpoint_source": checkpoint_source,
        "checkpoint_stat": checkpoint_stat,
        "exact_checkpoint_file_identity": False,
        "checkpoint_note": "bounded loaded-model fingerprint only; exact checkpoint SHA remains a replay gate",
    }


def _hook_manifest(diffusion: Any) -> list[dict[str, Any]]:
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None:
        return []
    result = []
    for index, block in enumerate(blocks):
        pre_hooks = (getattr(block, "_forward_pre_hooks", {}) or {}).values()
        post_hooks = (getattr(block, "_forward_hooks", {}) or {}).values()
        result.append(
            {
                "block": index,
                "forward": callable_identity(block.forward),
                "pre_hooks": [callable_identity(hook) for hook in pre_hooks],
                "post_hooks": [callable_identity(hook) for hook in post_hooks],
            }
        )
    return result


def _provider_manifest(provider: Any) -> dict[str, Any] | None:
    if provider is None:
        return None
    methods = {}
    for name in ("upscale", "execute", "__call__"):
        method = getattr(provider, name, None)
        if callable(method):
            methods[name] = callable_identity(method)
    fields = {}
    if hasattr(provider, "__dict__"):
        for key, value in sorted(vars(provider).items()):
            if str(key).startswith("_"):
                continue
            if callable(value):
                fields[str(key)] = {"callable": callable_identity(value)}
            elif isinstance(value, (bool, int, float, str, type(None))):
                fields[str(key)] = value
            else:
                fields[str(key)] = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    return {
        "type": f"{type(provider).__module__}.{type(provider).__qualname__}",
        "methods": methods,
        "fields": fields,
    }


def _compiler_manifest() -> dict[str, Any]:
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_config = getattr(dynamo, "config", None)
    cuda_backend = getattr(torch.backends, "cuda", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    return {
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "matmul_allow_tf32": getattr(getattr(cuda_backend, "matmul", None), "allow_tf32", None),
        "cudnn_allow_tf32": getattr(cudnn_backend, "allow_tf32", None),
        "dynamo_dynamic_shapes": getattr(dynamo_config, "dynamic_shapes", None),
        "dynamo_assume_static_by_default": getattr(dynamo_config, "assume_static_by_default", None),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_MODULE_LOADING",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TORCH_LOGS",
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCH_CUDNN_V8_API_ENABLED",
            )
        },
    }


def _path_matches(path: str, markers: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(marker in normalized for marker in markers)


def _loaded_source_group(markers: tuple[str, ...]) -> list[dict[str, Any]]:
    matches: list[tuple[str, str]] = []
    for module_name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None) if module is not None else None
        if not isinstance(path, str) or not path.lower().endswith((".py", ".pyw")):
            continue
        if _path_matches(path, markers):
            matches.append((str(module_name), path))
    result = []
    seen_paths: set[str] = set()
    for module_name, path in sorted(matches):
        identity = file_identity(path)
        resolved = None if identity is None else identity.get("resolved_path")
        dedupe = str(resolved or path)
        if dedupe in seen_paths:
            continue
        seen_paths.add(dedupe)
        result.append({"module": module_name, "file": identity})
        if len(result) >= _MAX_COMPANION_FILES:
            break
    return result


def _loaded_companion_manifest() -> dict[str, Any]:
    groups = {label: _loaded_source_group(markers) for label, markers in _COMPANION_PATH_MARKERS.items()}
    groups.update({label: _loaded_source_group(markers) for label, markers in _OPTIONAL_COMPANION_PATH_MARKERS.items()})
    return groups


def _source_group_complete(entries: list[dict[str, Any]]) -> bool:
    if not entries:
        return False
    return all(
        isinstance(entry.get("file"), dict) and bool(entry["file"].get("git", {}).get("available")) for entry in entries
    )


def collect_manifest(
    model: Any,
    *,
    model_options: dict[str, Any] | None = None,
    phase: str = "node_apply_preflight",
) -> dict[str, Any]:
    # Provenance must describe the files and git state that exist for this
    # invocation, not whatever happened to be cached earlier in a long-lived
    # Comfy process.
    file_identity.cache_clear()
    _repo_state.cache_clear()

    effective_options = (
        model_options if isinstance(model_options, dict) else (getattr(model, "model_options", None) or {})
    )
    effective_transformer = effective_options.get("transformer_options") or {}

    imported: dict[str, Any] = {}
    unresolved: list[str] = []
    for module_name in _REQUIRED_MODULES:
        module = sys.modules.get(module_name)
        identity = file_identity(getattr(module, "__file__", None)) if module is not None else None
        imported[module_name] = identity
        if identity is None or not identity["git"]["available"]:
            unresolved.append(f"source:{module_name}")

    companions = _loaded_companion_manifest()
    for label in _COMPANION_PATH_MARKERS:
        if not _source_group_complete(companions.get(label, [])):
            unresolved.append(f"companion:{label}")

    patcher_wrappers = _wrapper_manifest(model)
    effective_wrappers = _keyed_callable_manifest(effective_transformer.get("wrappers", {}))
    wrappers = effective_wrappers or patcher_wrappers
    for wrapper_type, entries in wrappers.items():
        for position, entry in enumerate(entries):
            source = entry["callable"].get("file")
            if source is None or not source["git"]["available"]:
                unresolved.append(f"wrapper:{wrapper_type}:{position}:{entry['key']}")

    patcher_callbacks = _callback_manifest(model)
    effective_callbacks = _keyed_callable_manifest(effective_transformer.get("callbacks", {}))
    callbacks = effective_callbacks or patcher_callbacks
    for callback_type, entries in callbacks.items():
        for position, entry in enumerate(entries):
            source = entry["callable"].get("file")
            if source is None or not source["git"]["available"]:
                unresolved.append(f"callback:{callback_type}:{position}:{entry['key']}")

    runtime_functions = _runtime_function_manifest()
    for name, identity in runtime_functions.items():
        source = identity.get("file")
        if source is None or not source["git"]["available"]:
            unresolved.append(f"runtime:{name}")

    pr35_predict = runtime_functions["comfy_compat.flow_predict_wrapper"]
    pr36_progressive = runtime_functions["runtime._run_progressive"]
    validation_stack = {
        "pr35_predict_active": pr35_predict["module"].endswith("handoff_checkpoint_diagnostic")
        and pr35_predict["qualname"].endswith("_flow_predict_capture_wrapper"),
        "pr36_progressive_active": pr36_progressive["module"].endswith("guidance_anchor_transport_validation")
        and pr36_progressive["qualname"].endswith("_run_progressive_anchor_wrapper"),
    }
    if not validation_stack["pr35_predict_active"]:
        unresolved.append("overlay:pr35_checkpoint_diagnostic")
    if not validation_stack["pr36_progressive_active"]:
        unresolved.append("overlay:pr36_guidance_anchor_transport")

    base = getattr(model, "model", None)
    diffusion = getattr(base, "diffusion_model", None)
    model_fingerprint = _model_fingerprint(model)
    if not model_fingerprint.get("available"):
        unresolved.append("model:runtime_fingerprint")

    replacement = (effective_transformer.get("patches_replace") or {}).get("dit") or {}
    replacement_manifest = {
        str(key): callable_identity(value) if callable(value) else safe_value(value)
        for key, value in replacement.items()
    }
    attention = effective_transformer.get("optimized_attention_override")
    progressive = effective_options.get(_runtime.PROGRESSIVE_KEY)
    provider = getattr(progressive, "learned_upscaler", None)

    manifest = {
        "schema_version": 3,
        "capture_phase": str(phase),
        "collected_ns": time.time_ns(),
        "cwd": os.getcwd(),
        "python": sys.version,
        "compiler_runtime": _compiler_manifest(),
        "imported_sources": imported,
        "loaded_companion_sources": companions,
        "active_wrapper_order": wrappers,
        "patcher_wrapper_order": patcher_wrappers,
        "active_callbacks": callbacks,
        "patcher_callbacks": patcher_callbacks,
        "active_injections": _injection_manifest(model),
        "patcher_is_injected": bool(getattr(model, "is_injected", False)),
        "active_object_patches": _object_patch_manifest(model),
        "active_runtime_functions": runtime_functions,
        "validation_stack": validation_stack,
        "replacement_chain_dit": replacement_manifest,
        "optimized_attention_override": (
            callable_identity(attention) if callable(attention) else safe_value(attention)
        ),
        "model": model_fingerprint,
        "progressive_config": safe_value(progressive),
        "learned_provider": _provider_manifest(provider),
        "block_hooks": _hook_manifest(diffusion),
        "model_options_alias_graph": alias_graph(effective_options),
        "unresolved": sorted(set(unresolved)),
    }
    manifest["gate_complete"] = not manifest["unresolved"]
    manifest["replay_checkpoint_identity_exact"] = bool(model_fingerprint.get("exact_checkpoint_file_identity"))
    return manifest
