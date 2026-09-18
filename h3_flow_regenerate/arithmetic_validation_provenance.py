"""Source/runtime provenance receipts for arithmetic-validation campaigns.

The capture path is deliberately stdlib-only. It records Git state and hashes
of files identified from the running ComfyUI process as loaded module __file__
values. The campaign gate validates these receipts independently of manifest
source-stack claims.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE_PROVENANCE_KIND = "h3_arithmetic_validation_source_provenance_v2"
RUNTIME_SOURCE_RECEIPT_KIND = "h3_arithmetic_validation_runtime_source_receipt_v1"
REPOSITORIES = ("comfyui", "flow", "sol", "vdn", "continuum")
OVERLAY_REPOSITORIES = ("flow", "sol", "vdn", "continuum")
_RUNTIME_REPOSITORY_SIGNATURES = {
    "flow": ("h3_flow_regenerate/runtime.py",),
    "sol": ("sol_h3/runtime.py",),
    "vdn": ("vdn_h3/partitioned_runtime.py",),
    "continuum": ("continuation.py", "v3"),
}
_RUNTIME_CRITICAL_MODULES = {
    "comfyui": ("comfy.model_management",),
    "flow": ("h3_flow_regenerate.runtime",),
    "sol": ("sol_h3.runtime",),
    "vdn": ("vdn_h3.partitioned_runtime",),
    "continuum": (),
}


class SourceProvenanceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceProvenanceError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(payload)


def _sha256_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _runtime_file_receipt(module: str, path: Path) -> dict[str, str]:
    path = path.resolve()
    _require(path.is_file(), f"runtime module file is missing: {path}")
    return {
        "module": module,
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _module_file(module_name: str) -> Path | None:
    module = sys.modules.get(module_name)
    raw = getattr(module, "__file__", None) if module is not None else None
    return Path(raw).resolve() if isinstance(raw, str) and raw else None


def _entrypoint_module(root: Path) -> tuple[str, Path]:
    target = (root / "__init__.py").resolve()
    candidates = []
    for name, module in tuple(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw:
            continue
        try:
            path = Path(raw).resolve()
        except OSError:
            continue
        if path == target:
            candidates.append(name)
    _require(candidates, f"running ComfyUI has no loaded custom-node entrypoint for {root}")
    return sorted(candidates)[0], target


def capture_running_comfyui_source_receipt() -> dict[str, Any]:
    """Capture source paths and custom-node load order from the live ComfyUI process."""
    try:
        import nodes as comfy_nodes
    except ImportError as exc:
        raise SourceProvenanceError("runtime source receipt must be captured inside ComfyUI") from exc

    registry = getattr(comfy_nodes, "LOADED_MODULE_DIRS", None)
    _require(
        isinstance(registry, dict) and registry,
        "running ComfyUI exposes no populated nodes.LOADED_MODULE_DIRS registry",
    )
    registry_entries = [
        (index, str(loader_name), Path(raw_root).resolve())
        for index, (loader_name, raw_root) in enumerate(registry.items())
    ]

    matched: dict[str, tuple[int, str, Path]] = {}
    for name, signatures in _RUNTIME_REPOSITORY_SIGNATURES.items():
        candidates = [
            (index, loader_name, root)
            for index, loader_name, root in registry_entries
            if all((root / signature).exists() for signature in signatures)
        ]
        _require(
            len(candidates) == 1,
            f"running ComfyUI custom-node registry resolves {name} to {len(candidates)} roots",
        )
        matched[name] = candidates[0]
    _require(
        len({str(entry[2]) for entry in matched.values()}) == len(OVERLAY_REPOSITORIES),
        "running ComfyUI custom-node roots are not distinct",
    )
    overlay_order = [name for name, _entry in sorted(matched.items(), key=lambda item: item[1][0])]

    raw_nodes_file = getattr(comfy_nodes, "__file__", None)
    _require(isinstance(raw_nodes_file, str) and raw_nodes_file, "running ComfyUI nodes.__file__ is missing")
    nodes_file = Path(raw_nodes_file).resolve()
    comfy_root = nodes_file.parent
    repositories: dict[str, dict[str, Any]] = {
        "comfyui": {
            "root": str(comfy_root),
            "loader_name": "nodes",
            "loader_index": None,
            "loaded_files": [_runtime_file_receipt("nodes", nodes_file)],
        }
    }

    for name in OVERLAY_REPOSITORIES:
        index, loader_name, root = matched[name]
        entry_module, entry_file = _entrypoint_module(root)
        loaded_files = [_runtime_file_receipt(entry_module, entry_file)]
        for module_name in _RUNTIME_CRITICAL_MODULES[name]:
            module_file = _module_file(module_name)
            if module_file is None:
                continue
            try:
                module_file.relative_to(root)
            except ValueError as exc:
                raise SourceProvenanceError(
                    f"runtime module {module_name} resolved outside {name} root: {module_file}"
                ) from exc
            if all(item["path"] != str(module_file) for item in loaded_files):
                loaded_files.append(_runtime_file_receipt(module_name, module_file))
        repositories[name] = {
            "root": str(root),
            "loader_name": loader_name,
            "loader_index": index,
            "loaded_files": loaded_files,
        }

    for module_name in _RUNTIME_CRITICAL_MODULES["comfyui"]:
        module_file = _module_file(module_name)
        if module_file is None:
            continue
        try:
            module_file.relative_to(comfy_root)
        except ValueError as exc:
            raise SourceProvenanceError(
                f"runtime module {module_name} resolved outside ComfyUI root: {module_file}"
            ) from exc
        repositories["comfyui"]["loaded_files"].append(_runtime_file_receipt(module_name, module_file))

    return {
        "schema_version": 1,
        "kind": RUNTIME_SOURCE_RECEIPT_KIND,
        "capture_origin": "running_comfyui_process",
        "registry_source": "nodes.LOADED_MODULE_DIRS",
        "process_id": os.getpid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "overlay_order": overlay_order,
        "repositories": repositories,
    }


def validate_runtime_source_receipt(
    value: dict[str, Any],
    *,
    verify_files: bool,
) -> dict[str, Any]:
    _require(isinstance(value, dict), "runtime source receipt root must be an object")
    _require(value.get("schema_version") == 1, "unsupported runtime source receipt schema")
    _require(value.get("kind") == RUNTIME_SOURCE_RECEIPT_KIND, "unexpected runtime source receipt kind")
    _require(value.get("capture_origin") == "running_comfyui_process", "runtime receipt capture origin is invalid")
    _require(value.get("registry_source") == "nodes.LOADED_MODULE_DIRS", "runtime receipt registry source is invalid")
    _require(type(value.get("process_id")) is int and value["process_id"] > 0, "runtime receipt process_id is invalid")
    _require(
        isinstance(value.get("python_executable"), str) and value["python_executable"],
        "runtime receipt python executable is missing",
    )
    overlay = value.get("overlay_order")
    _require(
        isinstance(overlay, list)
        and len(overlay) == len(OVERLAY_REPOSITORIES)
        and set(overlay) == set(OVERLAY_REPOSITORIES),
        "runtime receipt custom-node overlay order is incomplete or ambiguous",
    )
    repositories = value.get("repositories")
    _require(
        isinstance(repositories, dict) and set(repositories) == set(REPOSITORIES),
        "runtime receipt repositories are incomplete",
    )

    loader_indices: dict[str, int] = {}
    normalized: dict[str, Any] = {}
    for name in REPOSITORIES:
        entry = repositories[name]
        _require(isinstance(entry, dict), f"runtime receipt repository {name} is malformed")
        root_raw = entry.get("root")
        _require(isinstance(root_raw, str) and root_raw, f"runtime receipt {name} root is missing")
        root = Path(root_raw)
        _require(root.is_absolute(), f"runtime receipt {name} root is not absolute")
        if name in OVERLAY_REPOSITORIES:
            loader_name = entry.get("loader_name")
            loader_index = entry.get("loader_index")
            _require(isinstance(loader_name, str) and loader_name, f"runtime receipt {name} loader name is missing")
            _require(type(loader_index) is int and loader_index >= 0, f"runtime receipt {name} loader index is invalid")
            loader_indices[name] = loader_index
            if verify_files:
                for signature in _RUNTIME_REPOSITORY_SIGNATURES[name]:
                    _require((root / signature).exists(), f"runtime receipt {name} root lacks signature {signature}")
        loaded = entry.get("loaded_files")
        _require(isinstance(loaded, list) and loaded, f"runtime receipt {name} has no loaded module files")
        normalized_files = []
        for item in loaded:
            _require(isinstance(item, dict), f"runtime receipt {name} loaded-file entry is malformed")
            module = item.get("module")
            path_raw = item.get("path")
            digest = item.get("sha256")
            _require(isinstance(module, str) and module, f"runtime receipt {name} module label is missing")
            _require(isinstance(path_raw, str) and path_raw, f"runtime receipt {name} module path is missing")
            _require(_sha256_digest(digest), f"runtime receipt {name} module SHA-256 is invalid")
            path = Path(path_raw)
            _require(path.is_absolute(), f"runtime receipt {name} module path is not absolute")
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise SourceProvenanceError(
                    f"runtime receipt {name} module path is outside repository root: {path}"
                ) from exc
            if verify_files:
                _require(path.is_file(), f"runtime receipt loaded module is missing: {path}")
                _require(_sha256_file(path) == digest, f"runtime receipt loaded module changed after capture: {path}")
            normalized_files.append({"module": module, "path": path_raw, "sha256": digest})
        normalized[name] = {
            "root": root_raw,
            "loaded_files": normalized_files,
        }

    _require(
        len(set(loader_indices.values())) == len(loader_indices),
        "runtime receipt custom-node loader indices are not unique",
    )
    derived_overlay = [name for name, _index in sorted(loader_indices.items(), key=lambda item: item[1])]
    _require(overlay == derived_overlay, "runtime receipt overlay order disagrees with loader indices")
    return {
        "overlay_order": list(overlay),
        "repositories": normalized,
    }


def _loaded_file_match(working: bytes, head: bytes) -> tuple[bool, str, str]:
    """Return canonical tracked-file identity while preserving Windows CRLF transport."""
    raw_digest = _sha256_bytes(working)
    head_digest = _sha256_bytes(head)
    if raw_digest == head_digest:
        return True, "exact", head_digest
    if b"\r\n" in working and _sha256_bytes(working.replace(b"\r\n", b"\n")) == head_digest:
        return True, "crlf_to_lf", head_digest
    return False, "mismatch", raw_digest


def _git(root: Path, *args: str, check: bool = True) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise SourceProvenanceError(f"cannot execute git for {root}: {exc}") from exc
    if check and result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise SourceProvenanceError(f"git {' '.join(args)} failed for {root}: {error or result.returncode}")
    return result.stdout


def _relative_file(root: Path, path: Path) -> tuple[Path, str]:
    root = root.resolve()
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SourceProvenanceError(f"loaded file {path} is outside repository {root}") from exc
    _require(path.is_file(), f"loaded file is missing: {path}")
    return path, relative.as_posix()


def _untracked_fingerprint(root: Path) -> list[dict[str, str]]:
    raw = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    names = sorted(name for name in raw.decode("utf-8", errors="surrogateescape").split("\0") if name)
    result = []
    for name in names:
        path = root / name
        if path.is_symlink():
            digest = _sha256_bytes(
                ("symlink:" + str(path.readlink())).encode(
                    "utf-8",
                    errors="surrogateescape",
                )
            )
        elif path.is_file():
            digest = _sha256_file(path)
        else:
            digest = _sha256_bytes(b"other")
        result.append({"path": name, "sha256": digest})
    return result


def capture_repository(root: Path, loaded_files: list[tuple[str, Path]]) -> dict[str, Any]:
    root = root.resolve()
    _require(root.is_dir(), f"repository root is missing: {root}")
    head = _git(root, "rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    _require(len(head) == 40, f"repository HEAD is not a full Git SHA: {root}")

    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    tracked_diff = _git(root, "diff", "--binary", "HEAD")
    staged_diff = _git(root, "diff", "--binary", "--cached", "HEAD")
    untracked = _untracked_fingerprint(root)
    worktree_receipt = {
        "status_sha256": _sha256_bytes(status),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "staged_diff_sha256": _sha256_bytes(staged_diff),
        "untracked": untracked,
    }

    captured_files = []
    for module_name, raw_path in loaded_files:
        _require(isinstance(module_name, str) and module_name, "loaded module label is empty")
        path, relative = _relative_file(root, raw_path)
        working = path.read_bytes()
        head_bytes = _git(root, "show", f"HEAD:{relative}", check=False)
        tracked = bool(_git(root, "ls-files", "--error-unmatch", "--", relative, check=False))
        raw_sha256 = _sha256_bytes(working)
        head_sha256 = _sha256_bytes(head_bytes) if tracked else None
        matches_head = False
        match_mode = "untracked"
        canonical_sha256 = raw_sha256
        if tracked:
            matches_head, match_mode, canonical_sha256 = _loaded_file_match(
                working,
                head_bytes,
            )
        captured_files.append(
            {
                "module": module_name,
                "path": str(path),
                "relative_path": relative,
                "sha256": raw_sha256,
                "canonical_sha256": canonical_sha256,
                "head_sha256": head_sha256,
                "match_mode": match_mode,
                "tracked": tracked,
                "matches_head": matches_head,
            }
        )

    _require(captured_files, f"repository {root} has no loaded module file receipt")
    remote = _git(root, "remote", "get-url", "origin", check=False).decode("utf-8", errors="replace").strip()
    return {
        "root": str(root),
        "head": head,
        "dirty": bool(status),
        "working_tree": worktree_receipt,
        "working_tree_sha256": _canonical_sha256(worktree_receipt),
        "remote": remote or None,
        "loaded_files": captured_files,
    }


def capture_source_provenance(*, runtime_receipt: dict[str, Any]) -> dict[str, Any]:
    runtime = validate_runtime_source_receipt(runtime_receipt, verify_files=True)
    repositories = {name: Path(runtime["repositories"][name]["root"]) for name in REPOSITORIES}
    loaded_files = {
        name: [(item["module"], Path(item["path"])) for item in runtime["repositories"][name]["loaded_files"]]
        for name in REPOSITORIES
    }
    captured = {name: capture_repository(repositories[name], loaded_files[name]) for name in REPOSITORIES}
    return {
        "schema_version": 2,
        "kind": SOURCE_PROVENANCE_KIND,
        "python_executable": str(Path(sys.executable).resolve()),
        "runtime_receipt_sha256": _canonical_sha256(runtime_receipt),
        "runtime_receipt": runtime_receipt,
        "overlay_order": list(runtime["overlay_order"]),
        "repositories": captured,
    }


def validate_source_provenance(
    value: dict[str, Any],
    *,
    expected_stack: dict[str, str],
    expected_dirty: dict[str, bool],
) -> str:
    _require(isinstance(value, dict), "source provenance root must be an object")
    _require(value.get("schema_version") == 2, "unsupported source provenance schema")
    _require(value.get("kind") == SOURCE_PROVENANCE_KIND, "unexpected source provenance kind")
    runtime_receipt = value.get("runtime_receipt")
    runtime_receipt_sha256 = value.get("runtime_receipt_sha256")
    _require(
        _sha256_digest(runtime_receipt_sha256) and runtime_receipt_sha256 == _canonical_sha256(runtime_receipt),
        "source provenance runtime receipt digest is invalid",
    )
    runtime = validate_runtime_source_receipt(runtime_receipt, verify_files=False)
    overlay = value.get("overlay_order")
    _require(
        isinstance(overlay, list)
        and len(overlay) == len(OVERLAY_REPOSITORIES)
        and set(overlay) == set(OVERLAY_REPOSITORIES),
        "source provenance custom-node overlay order is incomplete or ambiguous",
    )
    _require(
        overlay == runtime["overlay_order"],
        "source provenance overlay order disagrees with the running ComfyUI receipt",
    )
    repositories = value.get("repositories")
    _require(isinstance(repositories, dict), "source provenance repositories are missing")

    identity_repositories = {}
    for name in REPOSITORIES:
        entry = repositories.get(name)
        _require(isinstance(entry, dict), f"source provenance is missing repository {name}")
        _require(
            entry.get("root") == runtime["repositories"][name]["root"],
            f"source provenance {name} root disagrees with the running ComfyUI receipt",
        )
        head = entry.get("head")
        dirty = entry.get("dirty")
        working_tree = entry.get("working_tree")
        working_tree_sha256 = entry.get("working_tree_sha256")
        _require(head == expected_stack.get(name), f"source provenance {name} HEAD disagrees with run source_stack")
        _require(
            isinstance(working_tree, dict),
            f"source provenance {name} working-tree receipt is missing",
        )
        _require(
            _sha256_digest(working_tree_sha256) and working_tree_sha256 == _canonical_sha256(working_tree),
            f"source provenance {name} working-tree digest is invalid",
        )
        _require(type(dirty) is bool, f"source provenance {name} dirty state is invalid")
        _require(
            dirty == expected_dirty.get(name),
            f"source provenance {name} dirty state disagrees with run source_dirty",
        )
        if not dirty:
            empty_digest = _sha256_bytes(b"")
            _require(
                working_tree.get("status_sha256") == empty_digest
                and working_tree.get("tracked_diff_sha256") == empty_digest
                and working_tree.get("staged_diff_sha256") == empty_digest
                and working_tree.get("untracked") == [],
                f"source provenance {name} claims clean state with nonempty working-tree evidence",
            )
        loaded = entry.get("loaded_files")
        _require(isinstance(loaded, list) and loaded, f"source provenance {name} has no loaded module files")
        identity_files = []
        runtime_files = {
            (item["module"], item["path"], item["sha256"]) for item in runtime["repositories"][name]["loaded_files"]
        }
        captured_runtime_files = set()
        for item in loaded:
            _require(isinstance(item, dict), f"source provenance {name} loaded-file receipt is malformed")
            module = item.get("module")
            path = item.get("path")
            relative = item.get("relative_path")
            digest = item.get("sha256")
            canonical_digest = item.get("canonical_sha256")
            head_digest = item.get("head_sha256")
            match_mode = item.get("match_mode")
            _require(isinstance(module, str) and module, f"source provenance {name} module label is missing")
            _require(isinstance(path, str) and path, f"source provenance {name} loaded-file path is missing")
            _require(isinstance(relative, str) and relative, f"source provenance {name} relative path is missing")
            _require(_sha256_digest(digest), f"source provenance {name} SHA-256 is invalid")
            _require(_sha256_digest(canonical_digest), f"source provenance {name} canonical SHA-256 is invalid")
            _require(_sha256_digest(head_digest), f"source provenance {name} HEAD SHA-256 is invalid")
            _require(item.get("tracked") is True, f"source provenance {name} loaded file is untracked")
            _require(item.get("matches_head") is True, f"source provenance {name} loaded file differs from HEAD")
            _require(
                match_mode in {"exact", "crlf_to_lf"},
                f"source provenance {name} loaded-file match mode is invalid",
            )
            _require(
                canonical_digest == head_digest,
                f"source provenance {name} canonical loaded-file digest mismatch",
            )
            if match_mode == "exact":
                _require(
                    digest == head_digest,
                    f"source provenance {name} exact loaded-file digest mismatch",
                )
            captured_runtime_files.add((module, path, digest))
            identity_files.append(
                {
                    "module": module,
                    "relative_path": relative,
                    "sha256": canonical_digest,
                }
            )
        _require(
            captured_runtime_files == runtime_files,
            f"source provenance {name} loaded files disagree with the running ComfyUI receipt",
        )
        identity_repositories[name] = {
            "head": head,
            "dirty": dirty,
            "loaded_files": sorted(
                identity_files,
                key=lambda item: (item["module"], item["relative_path"]),
            ),
        }

    identity = {
        "overlay_order": list(overlay),
        "repositories": identity_repositories,
    }
    return _canonical_sha256(identity)


__all__ = [
    "OVERLAY_REPOSITORIES",
    "REPOSITORIES",
    "RUNTIME_SOURCE_RECEIPT_KIND",
    "SOURCE_PROVENANCE_KIND",
    "SourceProvenanceError",
    "capture_repository",
    "capture_running_comfyui_source_receipt",
    "capture_source_provenance",
    "validate_runtime_source_receipt",
    "validate_source_provenance",
]
