from __future__ import annotations

from pathlib import Path

import pytest

from h3_flow_regenerate.validation_fixture import (
    REQUIRED_FIXTURE_ASSETS,
    build_fixture_manifest,
    validate_candidate_pair,
    validate_fixture_manifest,
)


def _write(root: Path, name: str, payload: bytes) -> str:
    path = root / name
    path.write_bytes(payload)
    return str(path)


def _spec(tmp_path: Path):
    assets = {
        name: _write(tmp_path, f"{name}.bin", f"asset:{name}".encode())
        for name in REQUIRED_FIXTURE_ASSETS
    }
    references = [
        _write(tmp_path, f"reference-{index}.bin", f"reference:{index}".encode())
        for index in range(7)
    ]
    return {
        "assets": assets,
        "references": references,
        "runtime": {
            "core": "installed-core-manifest-sha256",
            "model": "model-sha256",
            "vdn": "vdn-identity",
            "upscaler": "upscaler-sha256",
            "quantization_compiler": "compiler-provenance",
        },
        "sampler_seed": 384415445495679,
        "wildcard_seed": 237862536758780,
    }


def _receipt(fixture_sha256: str, domain: str):
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
        "log_sha256": "1" * 64,
        "metrics_sha256": "2" * 64,
        "mp4_sha256": "3" * 64,
        "pre_patch_boundary_sha256": "4" * 64,
    }
    return {"schema": 1, "fixture_sha256": fixture_sha256, "settings": settings, "outputs": outputs}


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


def test_pair_validator_accepts_only_one_axis_audio_position_change(tmp_path):
    fixture = build_fixture_manifest(_spec(tmp_path))
    control = _receipt(fixture["fixture_sha256"], "legacy_target")
    candidate = _receipt(fixture["fixture_sha256"], "source_carrier")
    validate_candidate_pair(fixture, control, candidate)

    candidate["settings"]["schedule"] = "different"
    with pytest.raises(ValueError, match="beyond audio_position_domain"):
        validate_candidate_pair(fixture, control, candidate)


def test_pair_validator_rejects_different_fixture(tmp_path):
    fixture = build_fixture_manifest(_spec(tmp_path))
    control = _receipt(fixture["fixture_sha256"], "legacy_target")
    candidate = _receipt("f" * 64, "source_carrier")
    with pytest.raises(ValueError, match="matched fixture"):
        validate_candidate_pair(fixture, control, candidate)
