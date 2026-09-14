from __future__ import annotations

from pathlib import Path


SOURCE = Path("h3_flow_regenerate/same_state_high_replay.py")
TESTS = Path("tests/test_same_state_high_replay.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


source = SOURCE.read_text()
source = replace_once(
    source,
    '''def _bundle_provenance_equivalence_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    persisted = manifest.get("provenance_equivalence_identity")
    persisted_digest = manifest.get("provenance_equivalence_digest")
    if isinstance(persisted, dict):
        if not isinstance(persisted_digest, str) or _sha_json(persisted) != persisted_digest:
            raise RuntimeError("same-state replay provenance-equivalence digest is inconsistent")
        return persisted
    legacy = manifest.get("provenance_identity")
    if not isinstance(legacy, dict):
        raise RuntimeError("same-state replay bundle lacks installed-runtime provenance identity")
    return _provenance_equivalence_identity(legacy)
''',
    '''def _bundle_provenance_equivalence_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    full_identity = manifest.get("provenance_identity")
    if not isinstance(full_identity, dict):
        raise RuntimeError("same-state replay bundle lacks installed-runtime provenance identity")
    derived = _provenance_equivalence_identity(full_identity)

    persisted = manifest.get("provenance_equivalence_identity")
    persisted_digest = manifest.get("provenance_equivalence_digest")
    if persisted is None and persisted_digest is None:
        return derived
    if not isinstance(persisted, dict) or not isinstance(persisted_digest, str):
        raise RuntimeError("same-state replay provenance-equivalence identity is incomplete")
    if _sha_json(persisted) != persisted_digest:
        raise RuntimeError("same-state replay provenance-equivalence digest is inconsistent")
    if _canonical_json(persisted) != _canonical_json(derived):
        differences = _provenance_diff_paths(derived, persisted)
        detail = "" if not differences else "; differing fields: " + ", ".join(differences)
        raise RuntimeError(
            "same-state replay provenance-equivalence identity is inconsistent with full provenance" + detail
        )
    return derived
''',
    "provenance-equivalence helper",
)
source = replace_once(
    source,
    '''    provenance_identity = manifest.get("provenance_identity")
    if not isinstance(provenance_identity, dict) or _sha_json(provenance_identity) != manifest.get("provenance_digest"):
        raise RuntimeError("same-state replay provenance manifest digest is inconsistent")
    runtime_policy = manifest.get("first_high_runtime_policy")
''',
    '''    provenance_identity = manifest.get("provenance_identity")
    if not isinstance(provenance_identity, dict) or _sha_json(provenance_identity) != manifest.get("provenance_digest"):
        raise RuntimeError("same-state replay provenance manifest digest is inconsistent")
    _bundle_provenance_equivalence_identity(manifest)
    runtime_policy = manifest.get("first_high_runtime_policy")
''',
    "bundle-validation",
)
SOURCE.write_text(source)

tests = TESTS.read_text()
name = "test_bundle_equivalence_identity_cannot_override_validated_full_provenance"
if name in tests:
    raise SystemExit("tamper test already present unexpectedly")
tests += r'''


def test_bundle_equivalence_identity_cannot_override_validated_full_provenance():
    full_identity = {
        "loaded_companion_sources": {
            "flow": [
                {
                    "module": "h3_flow_regenerate.runtime",
                    "file": {
                        "path": "/custom_nodes/flow/h3_flow_regenerate/runtime.py",
                        "resolved_path": "/custom_nodes/flow/h3_flow_regenerate/runtime.py",
                        "sha256": "production-source-a",
                        "git": {"available": True, "head": "capture-head", "dirty": False},
                    },
                }
            ]
        },
        "patcher_is_injected": True,
    }
    derived = replay._provenance_equivalence_identity(full_identity)
    manifest = {
        "provenance_identity": copy.deepcopy(full_identity),
        "provenance_digest": replay._sha_json(full_identity),
        "provenance_equivalence_identity": copy.deepcopy(derived),
        "provenance_equivalence_digest": replay._sha_json(derived),
    }

    assert replay._bundle_provenance_equivalence_identity(manifest) == derived

    tampered = copy.deepcopy(manifest)
    tampered_equivalence = tampered["provenance_equivalence_identity"]
    tampered_equivalence["loaded_companion_sources"]["flow"][0]["file"]["sha256"] = "production-source-b"
    tampered["provenance_equivalence_digest"] = replay._sha_json(tampered_equivalence)

    try:
        replay._bundle_provenance_equivalence_identity(tampered)
    except RuntimeError as exc:
        message = str(exc)
        assert "inconsistent with full provenance" in message
        assert "sha256" in message
    else:
        raise AssertionError("tampered causal provenance identity was accepted")
'''
TESTS.write_text(tests)
