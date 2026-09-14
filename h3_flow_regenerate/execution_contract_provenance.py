"""Installed-runtime provenance for first-high H3 contract diagnostics.

The collector runs inside the active ComfyUI process. It records the imported
source files, git state, effective wrapper/replacement order, hook identities,
attention provider, and a bounded fingerprint of the actually loaded H3 model.
It deliberately does not hash multi-gigabyte checkpoint files on the hot path.
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


def callable_identity(value: Any) -> dict[str, Any]:
    target = getattr(value, "__func__", value)
    cls = target if inspect.isclass(target) else type(target)
    module = getattr(target, "__module__", cls.__module__)
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", cls.__qualname__))
    try:
        source_file = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError):
        source_file = None
    code = getattr(target, "__code__", None)
    code_digest = None
    if code is not None:
        material = code.co_code + repr(code.co_consts).encode() + repr(code.co_names).encode()
        code_digest = hashlib.sha256(material).hexdigest()
    return {
        "module": str(module),
        "qualname": str(qualname),
        "code_digest": code_digest,
        "file": file_identity(source_file),
    }


def safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch.is_tensor(value):
        return {
            "tensor": True,
            "shape": [int(x) for x in value.shape],
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
            "cond": safe_value(getattr(value, "cond"), depth=depth + 1),
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


def _wrapper_manifest(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for wrapper_type, keyed in getattr(model, "wrappers", {}).items():
        ordered = []
        for key, wrappers in keyed.items():
            for wrapper in wrappers:
                ordered.append({"key": str(key), "callable": callable_identity(wrapper)})
        result[str(wrapper_type)] = ordered
    return result


def _runtime_function_manifest() -> dict[str, Any]:
    return {
        "runtime._run_progressive": callable_identity(_runtime._run_progressive),
        "runtime.apply_guidance": callable_identity(_runtime.apply_guidance),
        "runtime.build_handoff_state": callable_identity(_runtime.build_handoff_state),
        "comfy_compat.flow_predict_wrapper": callable_identity(_comfy_compat.flow_predict_wrapper),
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
    for owner in (model, base, getattr(base, "model_config", None), diffusion):
        if owner is None:
            continue
        for attr in ("checkpoint_name", "ckpt_name", "model_path", "model_file", "filename", "source_path"):
            value = getattr(owner, attr, None)
            if isinstance(value, (str, os.PathLike)) and str(value):
                checkpoint = str(value)
                break
        if checkpoint:
            break
    checkpoint_stat = None
    if checkpoint:
        try:
            resolved = Path(checkpoint).resolve(strict=True)
            stat = resolved.stat()
            checkpoint_stat = {"resolved_path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            checkpoint_stat = {"resolved_path": None, "size": None, "mtime_ns": None}

    return {
        "available": True,
        "class": f"{type(diffusion).__module__}.{type(diffusion).__qualname__}",
        "parameter_count": len(parameters),
        "runtime_fingerprint": digest.hexdigest(),
        "checkpoint_path": checkpoint,
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
        result.append(
            {
                "block": index,
                "forward": callable_identity(block.forward),
                "pre_hooks": [callable_identity(hook) for hook in (getattr(block, "_forward_pre_hooks", {}) or {}).values()],
                "post_hooks": [callable_identity(hook) for hook in (getattr(block, "_forward_hooks", {}) or {}).values()],
            }
        )
    return result


def collect_manifest(model: Any) -> dict[str, Any]:
    imported: dict[str, Any] = {}
    unresolved: list[str] = []
    for module_name in _REQUIRED_MODULES:
        module = sys.modules.get(module_name)
        identity = file_identity(getattr(module, "__file__", None)) if module is not None else None
        imported[module_name] = identity
        if identity is None or not identity["git"]["available"]:
            unresolved.append(f"source:{module_name}")

    wrappers = _wrapper_manifest(model)
    for wrapper_type, entries in wrappers.items():
        for position, entry in enumerate(entries):
            source = entry["callable"].get("file")
            if source is None or not source["git"]["available"]:
                unresolved.append(f"wrapper:{wrapper_type}:{position}:{entry['key']}")

    runtime_functions = _runtime_function_manifest()
    for name, identity in runtime_functions.items():
        source = identity.get("file")
        if source is None or not source["git"]["available"]:
            unresolved.append(f"runtime:{name}")

    base = getattr(model, "model", None)
    diffusion = getattr(base, "diffusion_model", None)
    model_fingerprint = _model_fingerprint(model)
    if not model_fingerprint.get("available"):
        unresolved.append("model:runtime_fingerprint")

    transformer = (getattr(model, "model_options", None) or {}).get("transformer_options") or {}
    replacement = ((transformer.get("patches_replace") or {}).get("dit") or {})
    replacement_manifest = {
        str(key): callable_identity(value) if callable(value) else safe_value(value)
        for key, value in replacement.items()
    }
    attention = transformer.get("optimized_attention_override")
    progressive = (getattr(model, "model_options", None) or {}).get(_runtime.PROGRESSIVE_KEY)
    provider = getattr(progressive, "learned_upscaler", None)

    manifest = {
        "schema_version": 1,
        "collected_ns": time.time_ns(),
        "cwd": os.getcwd(),
        "python": sys.version,
        "torch": {
            "version": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
        },
        "imported_sources": imported,
        "active_wrapper_order": wrappers,
        "active_runtime_functions": runtime_functions,
        "replacement_chain_dit": replacement_manifest,
        "optimized_attention_override": callable_identity(attention) if callable(attention) else safe_value(attention),
        "model": model_fingerprint,
        "progressive_config": safe_value(progressive),
        "learned_provider": callable_identity(provider) if callable(provider) else safe_value(provider),
        "block_hooks": _hook_manifest(diffusion),
        "unresolved": sorted(set(unresolved)),
    }
    manifest["gate_complete"] = not manifest["unresolved"]
    manifest["replay_checkpoint_identity_exact"] = bool(model_fingerprint.get("exact_checkpoint_file_identity"))
    return manifest
