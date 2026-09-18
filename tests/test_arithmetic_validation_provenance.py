from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from h3_flow_regenerate.arithmetic_validation_provenance import (
    SourceProvenanceError,
    _loaded_file_match,
    capture_repository,
    validate_source_provenance,
)


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _repo(tmp_path, name, relative):
    root = tmp_path / name
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("value = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root, path


def test_capture_repository_proves_loaded_file_matches_head(tmp_path):
    root, path = _repo(tmp_path, "flow", "pkg/runtime.py")

    receipt = capture_repository(
        root,
        [("pkg.runtime", path)],
    )

    assert receipt["dirty"] is False
    assert len(receipt["head"]) == 40
    assert len(receipt["working_tree_sha256"]) == 64
    loaded = receipt["loaded_files"][0]
    assert loaded["relative_path"] == "pkg/runtime.py"
    assert loaded["tracked"] is True
    assert loaded["matches_head"] is True
    assert loaded["match_mode"] == "exact"
    assert loaded["sha256"] == loaded["head_sha256"]
    assert loaded["canonical_sha256"] == loaded["head_sha256"]


def test_capture_repository_detects_dirty_loaded_file(tmp_path):
    root, path = _repo(tmp_path, "flow", "pkg/runtime.py")
    path.write_text("value = 2\n", encoding="utf-8")

    receipt = capture_repository(root, [("pkg.runtime", path)])

    assert receipt["dirty"] is True
    assert receipt["loaded_files"][0]["matches_head"] is False


def test_capture_repository_rejects_loaded_file_outside_repo(tmp_path):
    root, _path = _repo(tmp_path, "flow", "pkg/runtime.py")
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(SourceProvenanceError, match="outside repository"):
        capture_repository(root, [("pkg.runtime", outside)])


def test_validate_source_provenance_uses_loaded_bytes_and_overlay_order():
    heads = {
        "flow": "1" * 40,
        "sol": "2" * 40,
        "vdn": "3" * 40,
        "continuum": "4" * 40,
    }
    repositories = {}
    empty_digest = hashlib.sha256(b"").hexdigest()
    working_tree = {
        "status_sha256": empty_digest,
        "tracked_diff_sha256": empty_digest,
        "staged_diff_sha256": empty_digest,
        "untracked": [],
    }
    working_tree_sha256 = hashlib.sha256(
        json.dumps(
            working_tree,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    digests = {
        "flow": "a" * 64,
        "sol": "b" * 64,
        "vdn": "c" * 64,
        "continuum": "d" * 64,
    }
    for name, head in heads.items():
        digest = digests[name]
        repositories[name] = {
            "head": head,
            "dirty": False,
            "working_tree": working_tree,
            "working_tree_sha256": working_tree_sha256,
            "loaded_files": [
                {
                    "module": f"{name}.runtime",
                    "relative_path": "runtime.py",
                    "sha256": digest,
                    "canonical_sha256": digest,
                    "head_sha256": digest,
                    "match_mode": "exact",
                    "tracked": True,
                    "matches_head": True,
                }
            ],
        }
    value = {
        "schema_version": 1,
        "kind": "h3_arithmetic_validation_source_provenance_v1",
        "overlay_order": ["continuum", "flow", "vdn", "sol"],
        "repositories": repositories,
    }

    identity = validate_source_provenance(
        value,
        expected_stack=heads,
        expected_dirty={name: False for name in heads},
    )

    assert len(identity) == 64



def test_capture_cli_is_stdlib_only():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools" / "capture_arithmetic_validation_provenance.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr



def test_loaded_file_match_accepts_only_crlf_transport_normalization():
    head = b"first\nsecond\n"
    matches, mode, canonical = _loaded_file_match(
        b"first\r\nsecond\r\n",
        head,
    )
    assert matches is True
    assert mode == "crlf_to_lf"
    assert canonical == hashlib.sha256(head).hexdigest()

    matches, mode, canonical = _loaded_file_match(
        b"first\rsecond\n",
        head,
    )
    assert matches is False
    assert mode == "mismatch"
    assert canonical == hashlib.sha256(b"first\rsecond\n").hexdigest()
