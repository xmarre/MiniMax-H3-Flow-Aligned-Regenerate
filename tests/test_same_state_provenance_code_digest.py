from __future__ import annotations

import copy

import pytest

from h3_flow_regenerate import same_state_high_replay as replay


def _callable_identity(code_digest: str, *, source_sha: str = "same-source") -> dict:
    return {
        "module": "pkg.runtime",
        "qualname": "wrapper",
        "code_digest": code_digest,
        "file": {
            "path": "/repo/runtime.py",
            "resolved_path": "/repo/runtime.py",
            "sha256": source_sha,
            "git": {
                "available": True,
                "root": "/repo",
                "head": "capture-head",
                "dirty": False,
            },
        },
    }


def _full_provenance(code_digest: str, *, source_sha: str = "same-source") -> dict:
    return {
        "schema_version": 3,
        "gate_complete": True,
        "active_runtime_functions": {
            "runtime.wrapper": _callable_identity(code_digest, source_sha=source_sha),
        },
    }


def _legacy_persisted_equivalence(full: dict) -> dict:
    """Model the pre-fix causal identity: git removed, raw code_digest retained."""
    persisted = copy.deepcopy(full)
    persisted["active_runtime_functions"]["runtime.wrapper"]["file"].pop("git")
    return persisted


def test_cross_process_equivalence_ignores_process_local_callable_code_digest():
    capture = _full_provenance("capture-process-digest")
    replay_job = _full_provenance("replay-process-digest")

    capture_identity = replay._provenance_equivalence_identity(capture)
    replay_identity = replay._provenance_equivalence_identity(replay_job)

    assert capture_identity == replay_identity
    assert "code_digest" not in capture_identity["active_runtime_functions"]["runtime.wrapper"]


def test_cross_process_equivalence_still_rejects_source_sha_change():
    capture = replay._provenance_equivalence_identity(_full_provenance("capture-process-digest", source_sha="source-a"))
    replay_job = replay._provenance_equivalence_identity(
        _full_provenance("replay-process-digest", source_sha="source-b")
    )

    assert replay._provenance_diff_paths(capture, replay_job) == [
        "$.active_runtime_functions.runtime.wrapper.file.sha256"
    ]


def test_legacy_persisted_equivalence_with_raw_code_digest_remains_usable():
    full = _full_provenance("capture-process-digest")
    persisted = _legacy_persisted_equivalence(full)
    bundle = {
        "provenance_identity": full,
        "provenance_equivalence_identity": persisted,
        "provenance_equivalence_digest": replay._sha_json(persisted),
    }

    derived = replay._bundle_provenance_equivalence_identity(bundle)

    assert derived == replay._provenance_equivalence_identity(full)
    assert "code_digest" not in derived["active_runtime_functions"]["runtime.wrapper"]


def test_legacy_persisted_equivalence_cannot_hide_source_tamper():
    full = _full_provenance("capture-process-digest", source_sha="source-a")
    persisted = _legacy_persisted_equivalence(full)
    persisted["active_runtime_functions"]["runtime.wrapper"]["file"]["sha256"] = "source-b"
    bundle = {
        "provenance_identity": full,
        "provenance_equivalence_identity": persisted,
        "provenance_equivalence_digest": replay._sha_json(persisted),
    }

    with pytest.raises(RuntimeError, match="inconsistent with full provenance"):
        replay._bundle_provenance_equivalence_identity(bundle)
