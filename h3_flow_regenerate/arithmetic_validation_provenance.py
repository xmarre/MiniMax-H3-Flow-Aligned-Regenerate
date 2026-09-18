"""Source/runtime provenance receipts for arithmetic-validation campaigns.

The capture path is deliberately stdlib-only. It records Git state and hashes
of files identified from the running ComfyUI process as loaded module __file__
values. The campaign gate validates these receipts independently of manifest
source-stack claims.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE_PROVENANCE_KIND = "h3_arithmetic_validation_source_provenance_v1"
REPOSITORIES = ("comfyui", "flow", "sol", "vdn", "continuum")
OVERLAY_REPOSITORIES = ("flow", "sol", "vdn", "continuum")


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


def capture_source_provenance(
    *,
    repositories: dict[str, Path],
    loaded_files: dict[str, list[tuple[str, Path]]],
    overlay_order: list[str],
) -> dict[str, Any]:
    _require(
        set(repositories) == set(REPOSITORIES),
        "provenance requires comfyui/flow/sol/vdn/continuum repositories",
    )
    _require(
        len(overlay_order) == len(OVERLAY_REPOSITORIES) and set(overlay_order) == set(OVERLAY_REPOSITORIES),
        "overlay order must contain flow/sol/vdn/continuum exactly once",
    )
    captured = {name: capture_repository(repositories[name], loaded_files.get(name, [])) for name in REPOSITORIES}
    return {
        "schema_version": 1,
        "kind": SOURCE_PROVENANCE_KIND,
        "python_executable": str(Path(sys.executable).resolve()),
        "overlay_order": list(overlay_order),
        "repositories": captured,
    }


def validate_source_provenance(
    value: dict[str, Any],
    *,
    expected_stack: dict[str, str],
    expected_dirty: dict[str, bool],
) -> str:
    _require(isinstance(value, dict), "source provenance root must be an object")
    _require(value.get("schema_version") == 1, "unsupported source provenance schema")
    _require(value.get("kind") == SOURCE_PROVENANCE_KIND, "unexpected source provenance kind")
    overlay = value.get("overlay_order")
    _require(
        isinstance(overlay, list)
        and len(overlay) == len(OVERLAY_REPOSITORIES)
        and set(overlay) == set(OVERLAY_REPOSITORIES),
        "source provenance custom-node overlay order is incomplete or ambiguous",
    )
    repositories = value.get("repositories")
    _require(isinstance(repositories, dict), "source provenance repositories are missing")

    identity_repositories = {}
    for name in REPOSITORIES:
        entry = repositories.get(name)
        _require(isinstance(entry, dict), f"source provenance is missing repository {name}")
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
        for item in loaded:
            _require(isinstance(item, dict), f"source provenance {name} loaded-file receipt is malformed")
            module = item.get("module")
            relative = item.get("relative_path")
            digest = item.get("sha256")
            canonical_digest = item.get("canonical_sha256")
            head_digest = item.get("head_sha256")
            match_mode = item.get("match_mode")
            _require(isinstance(module, str) and module, f"source provenance {name} module label is missing")
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
            identity_files.append(
                {
                    "module": module,
                    "relative_path": relative,
                    "sha256": canonical_digest,
                }
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
    "SOURCE_PROVENANCE_KIND",
    "SourceProvenanceError",
    "capture_repository",
    "capture_source_provenance",
    "validate_source_provenance",
]
