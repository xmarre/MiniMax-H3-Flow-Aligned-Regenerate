from __future__ import annotations

from pathlib import Path

import pytest

from h3_flow_regenerate.validation_fixture import (
    REQUIRED_FIXTURE_ASSETS,
    REQUIRED_RUN_OUTPUTS,
    build_fixture_manifest,
    build_run_receipt,
    validate_candidate_pair,
    validate_fixture_manifest,
    validate_run_receipt,
)


def _write(root: Path, name: str, payload: bytes) -> str:
    path = root / name
    path.write_bytes(payload)
    return str(path)


def _spec(tmp_path: Path):
    assets = {name: _write(tmp_path, f"{name}.bin", f"asset:{name}".encode()) for name in REQUIRED_FIXTURE_ASSETS}
    references = [_write(tmp_path, f"reference-{index}.bin", f"reference:{index}".encode()) for index in range(7)]
    return {
        "assets": assets,
        "references": references,
        "runtime": {
            "core": "installed-core-manifest-sha256",
            "model": "model-sha256",
            "qwen_encoder": "qwen-encoder-sha256",
            "vdn": "vdn-identity",
            "upscaler": "upscaler-sha256",
            "quantization_compiler": "compiler-provenance",
        },
        "sampler_seed": 384415445495679,
        "wildcard_seed": 237862536758780,
    }


def _run_spec(tmp_path: Path, domain: str):
    settings = {
        "audio_position_domain": domain,
        "prefix_transformer_context": "exact_target_partitioned",
        "vdn_linear_diagnostic": "normal",
        "audio_guided_overlap_ticks": 4,
        "audio_guided_overlap_mode": "model_timestep_only",
        "sampler": "same",
        "schedule": "same",
        "handoff": "same",
        "prompt": "same",
    }
    outputs = {
        name: _write(tmp_path, f"{domain}-{name}.bin", f"{domain}:{name}".encode()) for name in REQUIRED_RUN_OUTPUTS
    }
    return {"settings": settings, "outputs": outputs}


def test_fixture_manifest_hashes_complete_cpu_persisted_boundary_state(tmp_path):
    manifest = build_fixture_manifest(_spec(tmp_path))
    assert validate_fixture_manifest(manifest) == manifest["fixture_sha256"]
    assert len(manifest["references"]) == 7
    assert set(manifest["assets"]) == set(REQUIRED_FIXTURE_ASSETS)
    assert all("sha256" in receipt and "bytes" in receipt for receipt in manifest["assets"].values())


def test_fixture_identity_changes_if_one_captured_byte_changes(tmp_path):
    spec = _spec(tmp_path)
    first = build_fixture_manifest(spec)
    Path(spec["assets"]["sampler_noise"]).write_bytes(b"different-noise")
    second = build_fixture_manifest(spec)
    assert first["fixture_sha256"] != second["fixture_sha256"]


def test_run_receipt_hashes_outputs_and_detects_tampering(tmp_path):
    fixture = build_fixture_manifest(_spec(tmp_path))
    receipt = build_run_receipt(fixture, _run_spec(tmp_path, "legacy_target"))
    assert validate_run_receipt(fixture, receipt) == receipt["settings"]
    assert all(
        len(receipt["outputs"][f"{name}_sha256"]) == 64 and receipt["outputs"][f"{name}_bytes"] > 0
        for name in REQUIRED_RUN_OUTPUTS
    )

    receipt["outputs"]["metrics_bytes"] += 1
    with pytest.raises(ValueError, match="receipt digest mismatch"):
        validate_run_receipt(fixture, receipt)


def test_pair_validator_accepts_only_one_axis_audio_position_change(tmp_path):
    fixture = build_fixture_manifest(_spec(tmp_path))
    control = build_run_receipt(fixture, _run_spec(tmp_path, "legacy_target"))
    candidate = build_run_receipt(fixture, _run_spec(tmp_path, "source_carrier"))
    validate_candidate_pair(fixture, control, candidate)

    candidate_spec = _run_spec(tmp_path, "source_carrier")
    candidate_spec["settings"]["schedule"] = "different"
    different = build_run_receipt(fixture, candidate_spec)
    with pytest.raises(ValueError, match="beyond audio_position_domain"):
        validate_candidate_pair(fixture, control, different)


def test_pair_validator_rejects_different_fixture(tmp_path):
    fixture = build_fixture_manifest(_spec(tmp_path))
    other_spec = _spec(tmp_path)
    Path(other_spec["assets"]["sampler_noise"]).write_bytes(b"other-noise")
    other_fixture = build_fixture_manifest(other_spec)
    control = build_run_receipt(fixture, _run_spec(tmp_path, "legacy_target"))
    candidate = build_run_receipt(other_fixture, _run_spec(tmp_path, "source_carrier"))
    with pytest.raises(ValueError, match="matched fixture"):
        validate_candidate_pair(fixture, control, candidate)


def test_fixture_validator_rejects_missing_runtime_identity_even_with_recomputed_digest(tmp_path):
    import hashlib
    import json

    manifest = build_fixture_manifest(_spec(tmp_path))
    manifest["runtime"].pop("qwen_encoder")
    body = {key: value for key, value in manifest.items() if key != "fixture_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    manifest["fixture_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="runtime identities"):
        validate_fixture_manifest(manifest)


def test_run_receipt_validator_rejects_noncanonical_output_hash_with_recomputed_receipt_digest(tmp_path):
    import hashlib
    import json

    fixture = build_fixture_manifest(_spec(tmp_path))
    receipt = build_run_receipt(fixture, _run_spec(tmp_path, "legacy_target"))
    receipt["outputs"]["metrics_sha256"] = "z" * 64
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="canonical metrics_sha256"):
        validate_run_receipt(fixture, receipt)


def test_pair_validator_requires_explicit_controlled_settings(tmp_path):
    fixture = build_fixture_manifest(_spec(tmp_path))
    control_spec = _run_spec(tmp_path, "legacy_target")
    candidate_spec = _run_spec(tmp_path, "source_carrier")
    control_spec["settings"].pop("prompt")
    candidate_spec["settings"].pop("prompt")
    control = build_run_receipt(fixture, control_spec)
    candidate = build_run_receipt(fixture, candidate_spec)
    with pytest.raises(ValueError, match="missing controlled settings"):
        validate_candidate_pair(fixture, control, candidate)
