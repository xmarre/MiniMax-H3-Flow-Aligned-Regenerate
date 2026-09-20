"""Bounded manifests for matched exact-prefix continuation validation.

This module never captures or mutates live ComfyUI state. It hashes CPU-persisted
fixture artifacts produced by the existing Continuum State/Session path or a
test-owned boundary capture, then proves that control and candidate receipts use
one identical fixture and differ only in the selected audio-position policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURE_SCHEMA = 1
RUN_RECEIPT_SCHEMA = 1
REQUIRED_RUN_OUTPUTS = (
    "log",
    "metrics",
    "mp4",
    "pre_patch_boundary",
)
REQUIRED_FIXTURE_ASSETS = (
    "first_chunk_latents",
    "carry_latents",
    "sampler_noise",
    "compiled_conditioning",
    "conditioning_manifest",
    "physical_prompt_intervals",
    "rng_state",
    "stage_schedule",
    "patcher_source_manifest",
    "pre_video_seam_boundary_frames",
)
REQUIRED_RUNTIME_IDENTITIES = (
    "core",
    "model",
    "qwen_encoder",
    "vdn",
    "upscaler",
    "quantization_compiler",
)
ARCHIVED_EVIDENCE_SHA256 = {
    "00545/Pasted text(20260920-160548).txt": "2a28806856485b6aafcfb493261ce51a6c59e56b3712176675aac5948abb2fba",
    "00545/metrics_00545_.json": "95fc1ca5e9246fb8a0856dc6a3b381aa0f1716dd1bdcd501798293db0686fedb",
    "00545/MiniMax_H3_00012-audio(1).mp4": "b11b41caeae7d45e71746c72d67e2ae6e4480b9e42bee689de2c9bac5abd95b4",
    "00546/Pasted text(20260920-174204).txt": "c0f9fdd0ab79f39001ef2d3ad95418172665e05f29d9c4fdc26c79ff50e6fac3",
    "00546/metrics_00546_.json": "19fbf9af2a3ac421a4fa4e84104a652b28aca6e48306edd7a451657e03455dcd",
    "00546/MiniMax_H3_00013-audio.mp4": "cf562082f2b46c0dbe12c03657ae5ffb8924ca34cd16d2a1d96d0e7747a4595e",
}


def sha256_file(path: str | Path) -> tuple[str, int]:
    source = Path(path)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _hashed_file(path: str | Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {"sha256": digest, "bytes": int(size)}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_fixture_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    """Hash one already-persisted matched fixture without retaining filesystem paths."""

    if not isinstance(spec, dict):
        raise ValueError("fixture specification must be a mapping")
    assets = spec.get("assets")
    runtime = spec.get("runtime")
    references = spec.get("references")
    archived = spec.get("archived_evidence")
    if not isinstance(assets, dict):
        raise ValueError("fixture specification requires an assets mapping")
    if not isinstance(runtime, dict):
        raise ValueError("fixture specification requires a runtime mapping")
    if not isinstance(references, list) or len(references) != 7:
        raise ValueError("fixture specification requires exactly seven ordered reference assets")

    missing_assets = [name for name in REQUIRED_FIXTURE_ASSETS if not assets.get(name)]
    if missing_assets:
        raise ValueError(f"fixture specification is missing required assets: {missing_assets!r}")
    missing_runtime = [name for name in REQUIRED_RUNTIME_IDENTITIES if not str(runtime.get(name, "")).strip()]
    if missing_runtime:
        raise ValueError(f"fixture specification is missing runtime identities: {missing_runtime!r}")

    hashed_assets = {name: _hashed_file(assets[name]) for name in REQUIRED_FIXTURE_ASSETS}
    hashed_references = [_hashed_file(path) for path in references]

    archived_receipts: dict[str, Any] = {}
    if archived is not None:
        if not isinstance(archived, dict):
            raise ValueError("archived_evidence must map canonical evidence names to paths")
        missing = [name for name in ARCHIVED_EVIDENCE_SHA256 if name not in archived]
        if missing:
            raise ValueError(f"archived evidence mapping is incomplete: {missing!r}")
        for name, expected in ARCHIVED_EVIDENCE_SHA256.items():
            receipt = _hashed_file(archived[name])
            if receipt["sha256"] != expected:
                raise ValueError(f"archived evidence hash mismatch for {name!r}")
            archived_receipts[name] = receipt

    manifest = {
        "schema": FIXTURE_SCHEMA,
        "assets": hashed_assets,
        "references": hashed_references,
        "runtime": {name: str(runtime[name]) for name in REQUIRED_RUNTIME_IDENTITIES},
        "sampler_seed": int(spec["sampler_seed"]),
        "wildcard_seed": None if spec.get("wildcard_seed") is None else int(spec["wildcard_seed"]),
        "archived_evidence": archived_receipts,
    }
    manifest["fixture_sha256"] = _canonical_digest(manifest)
    return manifest


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable") from exc


def validate_fixture_manifest(manifest: dict[str, Any]) -> str:
    if not isinstance(manifest, dict) or manifest.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("unsupported exact-prefix fixture manifest schema")
    supplied = str(manifest.get("fixture_sha256", ""))
    body = {key: value for key, value in manifest.items() if key != "fixture_sha256"}
    calculated = _canonical_digest(body)
    if supplied != calculated:
        raise ValueError("exact-prefix fixture manifest digest mismatch")
    if len(manifest.get("references") or ()) != 7:
        raise ValueError("exact-prefix fixture manifest does not contain seven ordered references")
    for name in REQUIRED_FIXTURE_ASSETS:
        receipt = (manifest.get("assets") or {}).get(name)
        if not isinstance(receipt, dict) or len(str(receipt.get("sha256", ""))) != 64:
            raise ValueError(f"exact-prefix fixture asset receipt is missing for {name!r}")
    return supplied


def build_run_receipt(
    fixture: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Hash one completed continuation run against one immutable fixture."""

    fixture_digest = validate_fixture_manifest(fixture)
    if not isinstance(spec, dict):
        raise ValueError("run receipt specification must be a mapping")
    settings = spec.get("settings")
    outputs = spec.get("outputs")
    if not isinstance(settings, dict):
        raise ValueError("run receipt specification requires a settings mapping")
    if not isinstance(outputs, dict):
        raise ValueError("run receipt specification requires an outputs mapping")
    missing = [name for name in REQUIRED_RUN_OUTPUTS if not outputs.get(name)]
    if missing:
        raise ValueError(f"run receipt specification is missing outputs: {missing!r}")

    hashed_outputs: dict[str, Any] = {}
    for name in REQUIRED_RUN_OUTPUTS:
        digest, size = sha256_file(outputs[name])
        hashed_outputs[f"{name}_sha256"] = digest
        hashed_outputs[f"{name}_bytes"] = int(size)

    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "fixture_sha256": fixture_digest,
        "settings": _json_copy(settings, label="run settings"),
        "outputs": hashed_outputs,
    }
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    return receipt


def validate_run_receipt(
    fixture: dict[str, Any],
    receipt: dict[str, Any],
    *,
    label: str = "run",
) -> dict[str, Any]:
    fixture_digest = validate_fixture_manifest(fixture)
    if not isinstance(receipt, dict) or receipt.get("schema") != RUN_RECEIPT_SCHEMA:
        raise ValueError(f"{label} run receipt has unsupported schema")
    if receipt.get("fixture_sha256") != fixture_digest:
        raise ValueError(f"{label} run does not reference the matched fixture")
    if not isinstance(receipt.get("settings"), dict):
        raise ValueError(f"{label} run receipt is missing settings")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{label} run receipt is missing output hashes")
    for name in REQUIRED_RUN_OUTPUTS:
        digest = str(outputs.get(f"{name}_sha256", ""))
        size = outputs.get(f"{name}_bytes")
        if len(digest) != 64:
            raise ValueError(f"{label} run receipt is missing {name}_sha256")
        if type(size) is not int or size < 0:
            raise ValueError(f"{label} run receipt is missing {name}_bytes")

    supplied = str(receipt.get("receipt_sha256", ""))
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    calculated = _canonical_digest(body)
    if supplied != calculated:
        raise ValueError(f"{label} run receipt digest mismatch")
    return receipt["settings"]


def validate_candidate_pair(
    fixture: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    """Require one-axis legacy_target versus source_carrier continuation receipts."""

    validate_fixture_manifest(fixture)
    for label, receipt in (("control", control), ("candidate", candidate)):
        validate_run_receipt(fixture, receipt, label=label)

    control_settings = dict(control["settings"])
    candidate_settings = dict(candidate["settings"])
    control_domain = control_settings.pop("audio_position_domain", None)
    candidate_domain = candidate_settings.pop("audio_position_domain", None)
    if control_domain != "legacy_target" or candidate_domain != "source_carrier":
        raise ValueError("matched pair must compare legacy_target control with source_carrier candidate")
    if control_settings != candidate_settings:
        raise ValueError("matched pair changes settings beyond audio_position_domain")

    required = {
        "prefix_transformer_context": "exact_target_partitioned",
        "vdn_linear_diagnostic": "normal",
        "audio_guided_overlap_ticks": 4,
        "audio_guided_overlap_mode": "model_timestep_only",
    }
    for name, expected in required.items():
        if control_settings.get(name) != expected:
            raise ValueError(f"matched pair requires {name}={expected!r}")
    # Output files are content-addressed by validate_run_receipt before this
    # one-axis settings comparison. Media quality remains a separate human gate.


__all__ = [
    "ARCHIVED_EVIDENCE_SHA256",
    "FIXTURE_SCHEMA",
    "REQUIRED_FIXTURE_ASSETS",
    "REQUIRED_RUNTIME_IDENTITIES",
    "REQUIRED_RUN_OUTPUTS",
    "RUN_RECEIPT_SCHEMA",
    "build_fixture_manifest",
    "build_run_receipt",
    "sha256_file",
    "validate_candidate_pair",
    "validate_fixture_manifest",
    "validate_run_receipt",
]
