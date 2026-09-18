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
            "fields": {"elapsed_ms": sampler_s * 1000.0, "failed": False},
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


def _runtime_device_identity(process_id: int, process_generation: str) -> dict:
    return {
        "type": "cuda",
        "index": 0,
        "current_device": 0,
        "name": "Test SM120",
        "uuid": "GPU-test-sm120",
        "total_memory": 96 * 1024**3,
        "multi_processor_count": 188,
        "sm": [12, 0],
        "cuda_driver_version": 13000,
        "process": process_id,
        "process_generation": process_generation,
        "context_scope": "pytorch-primary-process-device",
    }


def _runtime_environment() -> dict:
    return {
        "python": "3.12.0",
        "platform": "Linux",
        "torch": "2.10.0",
        "torch_cuda": "13.0",
        "triton": "3.6.0",
        "nvidia_cutlass_dsl": "4.3.2",
        "cuda_python": "13.0.0",
        "apache_tvm_ffi": "0.1.0",
    }


def _sol_log(
    *,
    compile_misses: int,
    condition: str,
    process_id: int,
    process_generation: str,
    source_verify_count: int = 1,
    request_serial: int = 1,
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
            "request_id": f"sol-h3-{process_id}-{request_serial}",
            "source_generation": "d" * 64 if source_verify_count else None,
            "implementation_generation": "e" * 64 if source_verify_count else None,
            "device_identity": (
                _runtime_device_identity(process_id, process_generation) if source_verify_count else None
            ),
            "compiler_environment": (_runtime_environment() if source_verify_count else {}),
        },
    }
    return "INFO comfy.sol_h3 Sol-H3 " + json.dumps(record, sort_keys=True)


def _diagnostic_report(*, compile_misses: int, request_id: str) -> dict:
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
        "success": True,
        "validation_failures": 0,
        "compile_hits": 4,
        "compile_misses": 0,
        "request_ids": [request_id],
        "request_reports": [
            {
                "summary_index": 0,
                "targets": sorted(campaign.REPLAY_TARGETS),
                "request_id": request_id,
                "source_generation": "d" * 64,
                "implementation_generation": "e" * 64,
                "compile_hits": 4,
                "compile_misses": compile_misses,
            }
        ],
        "replay_reports": replay,
    }


def _identity() -> dict:
    digest = "a" * 64
    environment = _runtime_environment()
    device = _runtime_device_identity(1, "0" * 32)
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
        "device_identity": campaign._stable_device_fingerprint(device),
        "driver": device["cuda_driver_version"],
        "python": environment["python"],
        "platform": environment["platform"],
        "torch": environment["torch"],
        "torch_cuda": environment["torch_cuda"],
        "cutlass_dsl": environment["nvidia_cutlass_dsl"],
        "triton": environment["triton"],
        "cuda_python": environment["cuda_python"],
        "apache_tvm_ffi": environment["apache_tvm_ffi"],
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


def _source_provenance(implementation: str) -> dict:
    stack = _source_stack(implementation)
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
    module_paths = {
        "flow": ("h3_flow_regenerate.runtime", "h3_flow_regenerate/runtime.py"),
        "sol": ("sol_h3.runtime", "sol_h3/runtime.py"),
        "vdn": ("vdn_h3.partitioned_runtime", "vdn_h3/partitioned_runtime.py"),
        "continuum": ("continuum.v3.driving_nodes", "v3/driving_nodes.py"),
    }
    for name, head in stack.items():
        module, relative = module_paths[name]
        digest = hashlib.sha256(f"{head}:{relative}".encode()).hexdigest()
        repositories[name] = {
            "root": f"/installed/{name}",
            "head": head,
            "dirty": False,
            "working_tree": working_tree,
            "working_tree_sha256": working_tree_sha256,
            "remote": f"https://github.com/example/{name}.git",
            "loaded_files": [
                {
                    "module": module,
                    "path": f"/installed/{name}/{relative}",
                    "relative_path": relative,
                    "sha256": digest,
                    "canonical_sha256": digest,
                    "head_sha256": digest,
                    "match_mode": "exact",
                    "tracked": True,
                    "matches_head": True,
                }
            ],
        }
    return {
        "schema_version": 1,
        "kind": "h3_arithmetic_validation_source_provenance_v1",
        "python_executable": "/python",
        "overlay_order": ["continuum", "flow", "vdn", "sol"],
        "repositories": repositories,
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
    process_id_override: int | None = None,
    process_generation_override: str | None = None,
    process_anchor_run_id: str | None = None,
    compile_misses_override: int | None = None,
) -> None:
    partitioned = implementation != "released_target"
    logical, actual = (18, 14) if partitioned else (17, 13)
    compile_misses = (0 if condition == "primed" else 1) if compile_misses_override is None else compile_misses_override
    default_process_id = {
        "released_target": 1001,
        "partitioned_preserved": 1002,
        "partitioned_fixed": 1003,
    }[implementation]
    default_process_generation = {
        "released_target": "1" * 32,
        "partitioned_preserved": "2" * 32,
        "partitioned_fixed": "3" * 32,
    }[implementation]
    process_id = default_process_id if process_id_override is None else process_id_override
    process_generation = (
        default_process_generation if process_generation_override is None else process_generation_override
    )

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
            process_generation=process_generation,
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
            json.dumps(
                _diagnostic_report(
                    compile_misses=compile_misses,
                    request_id=f"sol-h3-{process_id}-1",
                )
            ),
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
        "source_dirty": {
            "flow": False,
            "sol": False,
            "vdn": False,
            "continuum": False,
        },
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
            None
            if condition == "cold"
            else (process_anchor_run_id if process_anchor_run_id is not None else f"{implementation}-cold")
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
    source_provenance = {}
    for implementation in campaign.IMPLEMENTATIONS:
        path = tmp_path / f"{implementation}.source-provenance.json"
        path.write_text(
            json.dumps(_source_provenance(implementation)),
            encoding="utf-8",
        )
        source_provenance[implementation] = {
            "path": path.name,
            "sha256": _sha256(path),
        }
    runs: list[dict] = []
    sequence = 0

    # Each implementation starts with its fresh-process cold anchor.
    for implementation in campaign.IMPLEMENTATIONS:
        _add_run(
            tmp_path,
            runs,
            run_id=f"{implementation}-cold",
            implementation=implementation,
            condition="cold",
            sequence=sequence,
            identity_digest=identity_digest,
            sampler_s=290.0,
            e2e_s=340.0,
            diagnostic_mode=implementation != "released_target",
        )
        sequence += 1

    # Each arm also has a canonical same-arm primed repeat after its own cold run.
    for implementation in campaign.IMPLEMENTATIONS:
        _add_run(
            tmp_path,
            runs,
            run_id=f"{implementation}-primed",
            implementation=implementation,
            condition="primed",
            sequence=sequence,
            identity_digest=identity_digest,
            sampler_s=290.0,
            e2e_s=340.0,
        )
        sequence += 1

    # Prime both timing modes in the same process before measured pairs. The
    # control warmup should already hit its executable cache; the first fixed
    # warmup may populate partitioned-specific executable keys.
    _add_run(
        tmp_path,
        runs,
        run_id="pair-warmup-control",
        implementation="released_target",
        condition=campaign.PAIR_WARMUP_CONDITION,
        sequence=sequence,
        identity_digest=identity_digest,
        sampler_s=290.0,
        e2e_s=340.0,
        process_id_override=1001,
        process_generation_override="1" * 32,
        process_anchor_run_id="released_target-cold",
        compile_misses_override=0,
    )
    sequence += 1
    _add_run(
        tmp_path,
        runs,
        run_id="pair-warmup-fixed",
        implementation="partitioned_fixed",
        condition=campaign.PAIR_WARMUP_CONDITION,
        sequence=sequence,
        identity_digest=identity_digest,
        sampler_s=290.0,
        e2e_s=340.0,
        process_id_override=1001,
        process_generation_override="1" * 32,
        process_anchor_run_id="released_target-cold",
        compile_misses_override=1,
    )
    sequence += 1

    # Promotion timing pairs now alternate after both modes are resident.
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
            process_id_override=1001,
            process_generation_override="1" * 32,
            process_anchor_run_id="released_target-cold",
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
            process_id_override=1001,
            process_generation_override="1" * 32,
            process_anchor_run_id="released_target-cold",
        )
        sequence += 1

    # Invalidation cases remain in the process anchored by each cold run.
    for implementation in campaign.IMPLEMENTATIONS:
        for condition in ("numerical_invalidated", "geometry_bias_mutated"):
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
            )
            sequence += 1

    return {
        "schema_version": 1,
        "kind": campaign.CAMPAIGN_KIND,
        "frozen_identity": identity,
        "source_provenance": source_provenance,
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
    assert len(report.cold_reports) == 3
    assert {item["run_id"] for item in report.setup_reports} == {
        "released_target-cold",
        "partitioned_preserved-cold",
        "partitioned_fixed-cold",
        "pair-warmup-control",
        "pair-warmup-fixed",
    }
    assert report.pair_reports[0]["sampler_delta_pct"] > 0.0
    assert report.pair_reports[0]["e2e_delta_pct"] > 0.0


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
            process_id=1001,
            process_generation="1" * 32,
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
            process_id=1001,
            process_generation="1" * 32,
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

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match=r"source stack|different source stack",
    ):
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
            process_generation="9" * 32,
        )
        + "\n[INFO] Prompt executed in 330.00 seconds",
        encoding="utf-8",
    )
    target["artifacts"]["log"]["sha256"] = _sha256(log_path)

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="is not from the same process ID as its cold anchor",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_sol_totals_accept_dense_only_request_without_source_binding():
    text = (
        _sol_log(
            compile_misses=1,
            condition="cold",
            process_id=4242,
            process_generation="a" * 32,
            source_verify_count=1,
        )
        + "\n"
        + _sol_log(
            compile_misses=0,
            condition="cold",
            process_id=4242,
            process_generation="a" * 32,
            source_verify_count=0,
        )
    )
    totals = campaign._sol_totals(campaign._sol_records(text))

    assert totals["process_id"] == 4242
    assert totals["process_generation"] == "a" * 32
    assert totals["source_verified_requests"] == 1


def test_campaign_gate_rejects_recycled_pid_with_new_process_generation(
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
            process_id=1001,
            process_generation="f" * 32,
        )
        + "\n[INFO] Prompt executed in 330.00 seconds",
        encoding="utf-8",
    )
    target["artifacts"]["log"]["sha256"] = _sha256(log_path)

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="same process generation",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_diagnostic_from_different_sol_request(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    target = next(run for run in manifest["runs"] if run["id"] == "partitioned_fixed-cold")
    diagnostics_path = tmp_path / target["artifacts"]["sol_diagnostics"]["path"]
    report = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    report["request_ids"] = ["sol-h3-1003-999"]
    report["request_reports"][0]["request_id"] = "sol-h3-1003-999"
    diagnostics_path.write_text(json.dumps(report), encoding="utf-8")
    target["artifacts"]["sol_diagnostics"]["sha256"] = _sha256(diagnostics_path)

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="absent from the run log",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_nonadjacent_pair_in_complete_order(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    control = next(run for run in manifest["runs"] if run["id"] == "control-p0")
    fixed = next(run for run in manifest["runs"] if run["id"] == "fixed-p0")
    preserved = next(run for run in manifest["runs"] if run["id"] == "partitioned_preserved-primed")
    original_fixed_sequence = fixed["sequence"]
    for run in manifest["runs"]:
        if run["sequence"] >= original_fixed_sequence:
            run["sequence"] += 1
    preserved["sequence"] = original_fixed_sequence
    assert fixed["sequence"] - control["sequence"] == 2

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="not adjacent in the complete campaign order",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_run_before_cold_anchor(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    cold = next(run for run in manifest["runs"] if run["id"] == "partitioned_fixed-cold")
    primed = next(run for run in manifest["runs"] if run["id"] == "fixed-p0")
    cold["sequence"], primed["sequence"] = primed["sequence"], cold["sequence"]

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="appears before its cold process anchor",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_runtime_environment_drift(
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
    text = log_path.read_text(encoding="utf-8")
    line, e2e = text.split("\n", 1)
    record = json.loads(line.split("Sol-H3 ", 1)[1])
    record["runtime_lease"]["compiler_environment"]["triton"] = "different"
    log_path.write_text(
        "INFO comfy.sol_h3 Sol-H3 " + json.dumps(record, sort_keys=True) + "\n" + e2e,
        encoding="utf-8",
    )
    target["artifacts"]["log"]["sha256"] = _sha256(log_path)

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="runtime triton differs",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_dirty_source_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    manifest["runs"][0]["source_dirty"]["sol"] = True

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="uses dirty sol source",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_missing_shared_process_fixed_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    manifest["runs"] = [run for run in manifest["runs"] if run["id"] != "pair-warmup-fixed"]

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="lacks pre-measurement warmup of both control and fixed modes",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_pair_warmup_after_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    warmup = next(run for run in manifest["runs"] if run["id"] == "pair-warmup-fixed")
    first_pair = next(run for run in manifest["runs"] if run["id"] == "control-p0")
    warmup["sequence"], first_pair["sequence"] = first_pair["sequence"], warmup["sequence"]

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="lacks pre-measurement warmup of both control and fixed modes",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)



def test_campaign_gate_rejects_loaded_source_file_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    entry = manifest["source_provenance"]["partitioned_fixed"]
    path = tmp_path / entry["path"]
    provenance = json.loads(path.read_text(encoding="utf-8"))
    loaded = provenance["repositories"]["sol"]["loaded_files"][0]
    loaded["matches_head"] = False
    path.write_text(json.dumps(provenance), encoding="utf-8")
    entry["sha256"] = _sha256(path)

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="loaded file differs from HEAD",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_source_provenance_head_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    entry = manifest["source_provenance"]["partitioned_fixed"]
    path = tmp_path / entry["path"]
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["repositories"]["flow"]["head"] = "9" * 40
    path.write_text(json.dumps(provenance), encoding="utf-8")
    entry["sha256"] = _sha256(path)

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="HEAD disagrees with run source_stack",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)



def test_campaign_gate_rejects_failed_sampler_wall(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    target = next(run for run in manifest["runs"] if run["id"] == "control-p0")
    metrics_path = tmp_path / target["artifacts"]["metrics"]["path"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["events"][0]["fields"]["failed"] = True
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    target["artifacts"]["metrics"]["sha256"] = _sha256(metrics_path)

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="failed sampler_wall interval",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)


def test_campaign_gate_rejects_preserved_pair_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign,
        "validate_partitioned_runtime_evidence",
        lambda *args, **kwargs: object(),
    )
    manifest = _manifest(tmp_path)
    target = next(run for run in manifest["runs"] if run["id"] == "pair-warmup-fixed")
    target["implementation"] = "partitioned_preserved"
    target["source_stack"] = _source_stack("partitioned_preserved")

    with pytest.raises(
        campaign.CampaignEvidenceError,
        match="pair warmup uses unsupported implementation",
    ):
        campaign.validate_campaign_manifest(manifest, root=tmp_path)
