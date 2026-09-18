from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from h3_flow_regenerate import arithmetic_validation_campaign as campaign


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> dict:
    path.write_bytes(data)
    return {"path": path.name, "sha256": _sha256(path)}


def _metrics(logical: int, actual: int, sampler_s: float) -> dict:
    events = [
        {
            "kind": "sampler_wall",
            "fields": {"elapsed_ms": sampler_s * 1000.0},
        }
    ]
    for index in range(logical):
        events.append(
            {
                "kind": "model_call",
                "fields": {"actual": index < actual},
            }
        )
    return {"schema_version": 1, "events": events, "counters": {}}


def _sol_log(
    *,
    compile_misses: int,
    condition: str,
    process_id: int,
    source_verify_count: int = 1,
) -> str:
    reasons = {"new_request": 1}
    if condition == "geometry_bias_mutated":
        reasons = {"new_geometry": 1}
    record = {
        "success": True,
        "validation": {
            "compile_hits": 4,
            "compile_misses": compile_misses,
            "hits": 1,
            "misses": 1,
            "failures": 0,
            "miss_reasons": reasons,
        },
        "runtime_lease": {
            "source_verify_count": source_verify_count,
            "request_id": f"sol-h3-{process_id}-1",
            "device_identity": {
                "type": "cuda",
                "index": 0,
                "process": process_id,
            },
        },
    }
    return "INFO comfy.sol_h3 Sol-H3 " + json.dumps(record, sort_keys=True)


def _diagnostic_report(*, compile_misses: int) -> dict:
    replay = {
        target: {
            "first_compile_misses": 1,
            "primed_compile_misses": 0,
            "retained_proof_hit": True,
            "host_wall_s": 0.1,
        }
        for target in campaign.REPLAY_TARGETS
    }
    return {
        "status": "pass",
        "validation_failures": 0,
        "compile_hits": 4,
        "compile_misses": compile_misses,
        "replay_reports": replay,
    }


def _identity() -> dict:
    digest = "a" * 64
    return {
        "workflow_sha256": digest,
        "prompt_sha256": digest,
        "reference_media_sha256": digest,
        "model_stack_sha256": digest,
        "adapter_stack_sha256": digest,
        "patch_stack_sha256": digest,
        "decoder_sha256": digest,
        "sampler_settings_sha256": digest,
        "conditioning_sha256": digest,
        "geometry_sha256": digest,
        "seed": 1,
        "continuum_revision": "continuum-fixed",
        "device_identity": "cuda:0-sm120",
        "driver": "driver-fixed",
        "torch": "torch-fixed",
        "torch_cuda": "cuda-fixed",
        "cutlass_dsl": "cutlass-fixed",
        "triton": "triton-fixed",
    }


def _source_stack(implementation: str) -> dict:
    marker = {
        "released_target": "3",
        "partitioned_preserved": "2",
        "partitioned_fixed": "3",
    }[implementation]
    return {
        "flow": marker * 40,
        "sol": marker * 40,
        "vdn": marker * 40,
        "continuum": "4" * 40,
    }


def _add_run(
    tmp_path: Path,
    runs: list[dict],
    *,
    run_id: str,
    implementation: str,
    condition: str,
    sequence: int,
    identity_digest: str,
    sampler_s: float,
    e2e_s: float,
    pair_id: str | None = None,
    diagnostic_mode: bool = False,
) -> None:
    partitioned = implementation != "released_target"
    logical, actual = (18, 14) if partitioned else (17, 13)
    compile_misses = 0 if condition == "primed" else 1
    process_id = {
        "released_target": 1001,
        "partitioned_preserved": 1002,
        "partitioned_fixed": 1003,
    }[implementation]

    metrics_path = tmp_path / f"{run_id}.metrics.json"
    metrics_path.write_text(
        json.dumps(_metrics(logical, actual, sampler_s)),
        encoding="utf-8",
    )
    log_path = tmp_path / f"{run_id}.log.txt"
    log_path.write_text(
        _sol_log(
            compile_misses=compile_misses,
            condition=condition,
            process_id=process_id,
        )
        + f"\n[INFO] Prompt executed in {e2e_s:.2f} seconds",
        encoding="utf-8",
    )
    video_path = tmp_path / f"{run_id}.video.bin"
    audio_path = tmp_path / f"{run_id}.audio.bin"

    artifacts = {
        "metrics": {"path": metrics_path.name, "sha256": _sha256(metrics_path)},
        "log": {"path": log_path.name, "sha256": _sha256(log_path)},
        "video": _write(video_path, f"video:{run_id}".encode()),
        "audio": _write(audio_path, f"audio:{run_id}".encode()),
    }
    if diagnostic_mode:
        diagnostics_path = tmp_path / f"{run_id}.diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(_diagnostic_report(compile_misses=compile_misses)),
            encoding="utf-8",
        )
        artifacts["sol_diagnostics"] = {
            "path": diagnostics_path.name,
            "sha256": _sha256(diagnostics_path),
        }

    run = {
        "id": run_id,
        "implementation": implementation,
        "condition": condition,
        "sequence": sequence,
        "pair_id": pair_id,
        "frozen_identity_sha256": identity_digest,
        "source_stack": _source_stack(implementation),
        "timing": {"sampler_s": sampler_s, "e2e_s": e2e_s},
        "fresh_process": condition == "cold",
        "fresh_sol_requests": condition != "cold",
        "changed_contract_revalidated": condition
        in {
            "numerical_invalidated",
            "geometry_bias_mutated",
        },
        "compiler_cache_state": ("isolated_empty" if condition == "cold" else "retained_same_process"),
        "process_anchor_run_id": (
            None if condition == "cold" else f"{implementation}-cold"
        ),
        "diagnostic_mode": diagnostic_mode,
        "decoded_media": {
            "video_pass": True,
            "audio_pass": True,
            "checks": {
                "motion": True,
                "continuity": True,
                "prefix_seam": True,
                "prompt_adherence": True,
                "texture_artifacts": True,
                "audio_seam": True,
                "audio_intelligibility": True,
            },
        },
        "artifacts": artifacts,
    }
    if condition in {"numerical_invalidated", "geometry_bias_mutated"}:
        run["mutation"] = {
            "kind": ("geometry" if condition == "geometry_bias_mutated" else "preprocess_generation"),
            "base_field": ("geometry_sha256" if condition == "geometry_bias_mutated" else "conditioning_sha256"),
            "before_sha256": "a" * 64,
            "after_sha256": "c" * 64,
        }
    runs.append(run)


def _manifest(tmp_path: Path) -> dict:
    identity = _identity()
    identity_digest = campaign._canonical_sha256(identity)
    runs: list[dict] = []

    sequence = 0
    for pair in range(3):
        _add_run(
            tmp_path,
            runs,
            run_id=f"control-p{pair}",
            implementation="released_target",
            condition="primed",
            sequence=sequence,
            identity_digest=identity_digest,
            sampler_s=300.0 + pair,
            e2e_s=350.0 + pair,
            pair_id=f"pair-{pair}",
        )
        sequence += 1
        _add_run(
            tmp_path,
            runs,
            run_id=f"fixed-p{pair}",
            implementation="partitioned_fixed",
            condition="primed",
            sequence=sequence,
            identity_digest=identity_digest,
            sampler_s=280.0 + pair,
            e2e_s=330.0 + pair,
            pair_id=f"pair-{pair}",
        )
        sequence += 1

    for implementation in campaign.IMPLEMENTATIONS:
        for condition in campaign.CONDITIONS:
            if condition == "primed" and implementation in {
                "released_target",
                "partitioned_fixed",
            }:
                continue
            diagnostic = condition == "cold" and implementation != "released_target"
            _add_run(
                tmp_path,
                runs,
                run_id=f"{implementation}-{condition}",
                implementation=implementation,
                condition=condition,
                sequence=sequence,
                identity_digest=identity_digest,
                sampler_s=290.0,
                e2e_s=340.0,
                diagnostic_mode=diagnostic,
            )
            sequence += 1

    return {
        "schema_version": 1,
        "kind": campaign.CAMPAIGN_KIND,
        "frozen_identity": identity,
        "runs": runs,
    }


def test_campaign_gate_accepts_complete_matched_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    report = campaign.validate_campaign_manifest(_manifest(tmp_path), root=tmp_path)

    assert report.pair_count == 3
    assert report.sampler_pair_wins == 3
    assert report.e2e_pair_wins == 3
    assert report.sampler_median_delta_s == pytest.approx(20.0)
    assert report.e2e_median_delta_s == pytest.approx(20.0)
    assert report.diagnostic_runs == 2


def test_campaign_gate_rejects_primed_compile_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    target = next(run for run in manifest["runs"] if run["id"] == "fixed-p0")
    log_path = tmp_path / target["artifacts"]["log"]["path"]
    log_path.write_text(
        _sol_log(
            compile_misses=1,
            condition="primed",
            process_id=1003,
        )
        + "\n[INFO] Prompt executed in 330.00 seconds",
        encoding="utf-8",
    )
    target["artifacts"]["log"]["sha256"] = _sha256(log_path)

    with pytest.raises(campaign.CampaignEvidenceError, match="primed condition observed compiler misses"):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_media_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    manifest["runs"][0]["decoded_media"]["audio_pass"] = False

    with pytest.raises(campaign.CampaignEvidenceError, match="failed decoded audio"):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_nonrepeatable_e2e_advantage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    target = next(run for run in manifest["runs"] if run["id"] == "fixed-p1")
    target["timing"]["e2e_s"] = 400.0
    log_path = tmp_path / target["artifacts"]["log"]["path"]
    log_path.write_text(
        _sol_log(
            compile_misses=0,
            condition="primed",
            process_id=1003,
        )
        + "\n[INFO] Prompt executed in 400.00 seconds",
        encoding="utf-8",
    )
    target["artifacts"]["log"]["sha256"] = _sha256(log_path)

    with pytest.raises(campaign.CampaignEvidenceError, match="has no E2E advantage"):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_source_stack_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    target = next(
        run
        for run in manifest["runs"]
        if run["implementation"] == "partitioned_fixed" and run["condition"] == "geometry_bias_mutated"
    )
    target["source_stack"]["sol"] = "f" * 40

    with pytest.raises(campaign.CampaignEvidenceError, match="source stack changed"):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_artifact_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    manifest["runs"][0]["artifacts"]["video"]["sha256"] = "0" * 64

    with pytest.raises(campaign.CampaignEvidenceError, match="hash mismatch"):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)



def test_campaign_gate_rejects_same_process_claim_when_runtime_pid_differs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    target = next(run for run in manifest["runs"] if run["id"] == "fixed-p0")
    log_path = tmp_path / target["artifacts"]["log"]["path"]
    log_path.write_text(
        _sol_log(
            compile_misses=0,
            condition="primed",
            process_id=9999,
        )
        + "\n[INFO] Prompt executed in 330.00 seconds",
        encoding="utf-8",
    )
    target["artifacts"]["log"]["sha256"] = _sha256(log_path)

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="is not from the same process as its cold anchor",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)



def test_sol_totals_accept_dense_only_request_without_source_binding():
    text = (
        _sol_log(
            compile_misses=1,
            condition="cold",
            process_id=4242,
            source_verify_count=1,
        )
        + "\n"
        + _sol_log(
            compile_misses=0,
            condition="cold",
            process_id=4242,
            source_verify_count=0,
        )
    )
    totals = campaign._sol_totals(campaign._sol_records(text))

    assert totals["process_id"] == 4242
    assert totals["source_verified_requests"] == 1
