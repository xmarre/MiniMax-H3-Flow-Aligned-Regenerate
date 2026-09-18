"""Fail-closed promotion gate for the staged arithmetic-validation campaign.

This validator is offline. It consumes already-produced metrics, logs, decoded
media, and diagnostic reports. It does not execute H3/CUDA or alter runtime
policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .partitioned_runtime_gate import RuntimeGateError, validate_partitioned_runtime_evidence

CAMPAIGN_KIND = "h3_arithmetic_validation_campaign_v1"
IMPLEMENTATIONS = ("released_target", "partitioned_preserved", "partitioned_fixed")
EVIDENCE_CONDITIONS = ("cold", "primed", "numerical_invalidated", "geometry_bias_mutated")
PAIR_WARMUP_CONDITION = "pair_warmup"
CONDITIONS = (*EVIDENCE_CONDITIONS, PAIR_WARMUP_CONDITION)
RELEASED_TARGET_WHOLE_COUNTS = (17, 13, 4)
PARTITIONED_WHOLE_COUNTS = (18, 14, 4)
PARTITIONED_LATEST_COUNTS = (9, 7, 2)
REPLAY_TARGETS = {
    "ordinary_low",
    "ordinary_continuation_high",
    "partitioned_suffix",
}
REQUIRED_IDENTITY = {
    "workflow_sha256",
    "prompt_sha256",
    "reference_media_sha256",
    "model_stack_sha256",
    "adapter_stack_sha256",
    "patch_stack_sha256",
    "decoder_sha256",
    "sampler_settings_sha256",
    "conditioning_sha256",
    "geometry_sha256",
    "seed",
    "continuum_revision",
    "device_identity",
    "driver",
    "python",
    "platform",
    "torch",
    "torch_cuda",
    "cutlass_dsl",
    "triton",
    "cuda_python",
    "apache_tvm_ffi",
}
REQUIRED_ARTIFACTS = ("metrics", "log", "video", "audio")
MEDIA_CHECKS = {
    "motion",
    "continuity",
    "prefix_seam",
    "prompt_adherence",
    "texture_artifacts",
    "audio_seam",
    "audio_intelligibility",
}
COLD_CACHE_STATES = {"isolated_empty", "isolated_retained"}
_PROMPT_EXECUTED_RE = re.compile(r"Prompt executed in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds")
_SOL_REQUEST_ID_RE = re.compile(r"^sol-h3-([1-9][0-9]*)-([1-9][0-9]*)$")


class CampaignEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CampaignReport:
    frozen_identity_sha256: str
    runs: int
    diagnostic_runs: int
    pair_count: int
    sampler_pair_wins: int
    e2e_pair_wins: int
    sampler_median_delta_s: float
    e2e_median_delta_s: float
    decoded_video_passes: int
    decoded_audio_passes: int
    implementation_source_digests: dict[str, str]
    pair_reports: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignEvidenceError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _hex_digest(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _stable_device_identity(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "Sol device identity is missing")
    stable = {
        name: value.get(name)
        for name in (
            "type",
            "index",
            "name",
            "uuid",
            "total_memory",
            "multi_processor_count",
            "sm",
            "cuda_driver_version",
            "context_scope",
        )
    }
    _require(stable["type"] == "cuda", "campaign requires a CUDA Sol runtime")
    _require(stable["sm"] == [12, 0], f"campaign requires SM120, got {stable['sm']!r}")
    _require(
        isinstance(stable["cuda_driver_version"], int),
        "Sol device identity omitted CUDA driver version",
    )
    return stable


def _stable_device_fingerprint(value: Any) -> str:
    return _canonical_sha256(_stable_device_identity(value))


def _runtime_environment(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "Sol compiler environment is missing")
    required = {
        "python",
        "platform",
        "torch",
        "torch_cuda",
        "triton",
        "nvidia_cutlass_dsl",
        "cuda_python",
        "apache_tvm_ffi",
    }
    missing = required - set(value)
    _require(not missing, f"Sol compiler environment is missing {sorted(missing)}")
    return {
        "python": value["python"],
        "platform": value["platform"],
        "torch": value["torch"],
        "torch_cuda": value["torch_cuda"],
        "cutlass_dsl": value["nvidia_cutlass_dsl"],
        "triton": value["triton"],
        "cuda_python": value["cuda_python"],
        "apache_tvm_ffi": value["apache_tvm_ffi"],
    }


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignEvidenceError(f"{label} is missing or not numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise CampaignEvidenceError(f"{label} must be finite and positive")
    return number


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(run: dict[str, Any], name: str, root: Path) -> Path:
    artifacts = run.get("artifacts")
    _require(isinstance(artifacts, dict), f"run {run.get('id')!r} has no artifacts object")
    entry = artifacts.get(name)
    _require(isinstance(entry, dict), f"run {run.get('id')!r} is missing artifact {name!r}")
    raw_path = entry.get("path")
    expected = entry.get("sha256")
    _require(isinstance(raw_path, str) and raw_path, f"artifact {name!r} has no path")
    _require(_hex_digest(expected, 64), f"artifact {name!r} has invalid SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    _require(path.is_file(), f"artifact {name!r} is missing: {path}")
    _require(_file_sha256(path) == expected, f"artifact {name!r} hash mismatch")
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignEvidenceError(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CampaignEvidenceError(f"cannot read log {path}: {exc}") from exc


def _e2e_s(text: str) -> float:
    matches = [float(value) for value in _PROMPT_EXECUTED_RE.findall(text)]
    _require(
        len(matches) == 1,
        f"run log must contain exactly one 'Prompt executed in ... seconds' receipt; got {len(matches)}",
    )
    return matches[0]


def _sampler_s(metrics: dict[str, Any]) -> float:
    events = metrics.get("events")
    _require(isinstance(events, list), "metrics events must be a list")
    values = []
    for event in events:
        if not isinstance(event, dict) or event.get("kind") != "sampler_wall":
            continue
        fields = event.get("fields")
        elapsed = fields.get("elapsed_ms") if isinstance(fields, dict) else None
        _require(
            isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool),
            "sampler_wall elapsed_ms is invalid",
        )
        values.append(float(elapsed) / 1000.0)
    _require(values, "metrics contain no sampler_wall events")
    return sum(values)


def _counts(metrics: dict[str, Any]) -> tuple[int, int, int]:
    events = metrics.get("events")
    _require(isinstance(events, list), "metrics events must be a list")
    calls = [event for event in events if isinstance(event, dict) and event.get("kind") == "model_call"]
    _require(calls, "metrics contain no model_call events")
    actual = 0
    for event in calls:
        fields = event.get("fields")
        _require(isinstance(fields, dict), "model_call fields are invalid")
        marker = fields.get("actual")
        _require(type(marker) is bool, "model_call actual marker is missing")
        actual += int(marker)
    return len(calls), actual, len(calls) - actual


def _sol_records(text: str) -> list[dict[str, Any]]:
    marker = "Sol-H3 "
    records = []
    for line in text.splitlines():
        at = line.find(marker)
        if at < 0:
            continue
        try:
            value = json.loads(line[at + len(marker) :].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("validation"), dict):
            records.append(value)
    _require(records, "run log contains no Sol-H3 validation summaries")
    return records


def _sol_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "compile_hits": 0,
        "compile_misses": 0,
        "validation_hits": 0,
        "validation_misses": 0,
        "validation_failures": 0,
        "miss_reasons": {},
        "process_ids": set(),
        "process_generations": set(),
        "source_verified_requests": 0,
        "request_provenance": {},
        "runtime_environments": {},
        "device_identities": {},
    }
    for record in records:
        validation = record.get("validation")
        lease = record.get("runtime_lease")
        _require(isinstance(validation, dict), "Sol validation summary is malformed")
        _require(isinstance(lease, dict), "Sol runtime lease summary is malformed")
        request_id = lease.get("request_id")
        match = _SOL_REQUEST_ID_RE.match(request_id) if isinstance(request_id, str) else None
        _require(match is not None, "Sol runtime lease request_id is missing or malformed")
        result["process_ids"].add(int(match.group(1)))
        _require(
            request_id not in result["request_provenance"],
            f"duplicate Sol runtime lease request_id {request_id!r}",
        )

        source_verify_count = lease.get("source_verify_count")
        _require(
            type(source_verify_count) is int and source_verify_count in {0, 1},
            "Sol Request source verification count is outside the request-owned contract",
        )
        result["source_verified_requests"] += source_verify_count
        if source_verify_count == 1:
            device_identity = lease.get("device_identity")
            _require(
                isinstance(device_identity, dict),
                "source-verified Sol Request omitted device identity",
            )
            _require(
                device_identity.get("process") == int(match.group(1)),
                "Sol request ID and device process identity disagree",
            )
            process_generation = device_identity.get("process_generation")
            _require(
                _hex_digest(process_generation, 32),
                "source-verified Sol Request omitted process-generation provenance",
            )
            result["process_generations"].add(process_generation)
            stable_device = _stable_device_identity(device_identity)
            device_digest = _canonical_sha256(stable_device)
            result["device_identities"][device_digest] = stable_device
            runtime_environment = _runtime_environment(lease.get("compiler_environment"))
            environment_digest = _canonical_sha256(runtime_environment)
            result["runtime_environments"][environment_digest] = runtime_environment
            source_generation = lease.get("source_generation")
            implementation_generation = lease.get("implementation_generation")
            _require(
                _hex_digest(source_generation, 64),
                "source-verified Sol Request omitted source-generation provenance",
            )
            _require(
                _hex_digest(implementation_generation, 64),
                "source-verified Sol Request omitted implementation-generation provenance",
            )
        else:
            source_generation = lease.get("source_generation")
            implementation_generation = lease.get("implementation_generation")
        result["request_provenance"][request_id] = {
            "source_verify_count": source_verify_count,
            "source_generation": source_generation,
            "implementation_generation": implementation_generation,
        }
        result["compile_hits"] += int(validation.get("compile_hits", 0))
        result["compile_misses"] += int(validation.get("compile_misses", 0))
        result["validation_hits"] += int(validation.get("hits", 0))
        result["validation_misses"] += int(validation.get("misses", 0))
        result["validation_failures"] += int(validation.get("failures", 0))
        reasons = validation.get("miss_reasons")
        if isinstance(reasons, dict):
            for name, count in reasons.items():
                if isinstance(name, str) and isinstance(count, int):
                    result["miss_reasons"][name] = result["miss_reasons"].get(name, 0) + count
    _require(result["validation_failures"] == 0, "Sol arithmetic validation failed")
    _require(
        result["source_verified_requests"] > 0,
        "run contains no Sol Request that bound and verified the sparse runtime",
    )
    _require(
        len(result["process_ids"]) == 1,
        "one run log contains Sol Requests from multiple process identities",
    )
    _require(
        len(result["process_generations"]) == 1,
        "one run log contains source-verified Sol Requests from multiple process generations",
    )
    _require(
        len(result["device_identities"]) == 1,
        "one run log contains multiple source-verified device identities",
    )
    _require(
        len(result["runtime_environments"]) == 1,
        "one run log contains multiple source-verified compiler environments",
    )
    result["process_id"] = next(iter(result.pop("process_ids")))
    result["process_generation"] = next(iter(result.pop("process_generations")))
    result["device_fingerprint"], result["device_identity"] = next(iter(result.pop("device_identities").items()))
    _, result["runtime_environment"] = next(iter(result.pop("runtime_environments").items()))
    return result


def _source_digest(run: dict[str, Any]) -> str:
    stack = run.get("source_stack")
    dirty = run.get("source_dirty")
    _require(isinstance(stack, dict), f"run {run.get('id')!r} has no source_stack")
    _require(isinstance(dirty, dict), f"run {run.get('id')!r} has no source_dirty map")
    required = {"flow", "sol", "vdn", "continuum"}
    _require(required.issubset(stack), f"source_stack is missing {sorted(required - set(stack))}")
    _require(required.issubset(dirty), f"source_dirty is missing {sorted(required - set(dirty))}")
    for name in required:
        _require(_hex_digest(stack.get(name), 40), f"source_stack.{name} is not a full Git SHA")
        _require(type(dirty.get(name)) is bool, f"source_dirty.{name} is not boolean")
        _require(not dirty[name], f"run {run.get('id')!r} uses dirty {name} source")
    return _canonical_sha256({"source_stack": stack, "source_dirty": dirty})


def _media(run: dict[str, Any]) -> None:
    acceptance = run.get("decoded_media")
    _require(isinstance(acceptance, dict), f"run {run.get('id')!r} has no decoded_media result")
    _require(acceptance.get("video_pass") is True, f"run {run.get('id')!r} failed decoded video")
    _require(acceptance.get("audio_pass") is True, f"run {run.get('id')!r} failed decoded audio")
    checks = acceptance.get("checks")
    _require(isinstance(checks, dict), f"run {run.get('id')!r} has no decoded-media check detail")
    missing = MEDIA_CHECKS - set(checks)
    _require(not missing, f"run {run.get('id')!r} decoded-media checks are missing {sorted(missing)}")
    failed = sorted(name for name in MEDIA_CHECKS if checks.get(name) is not True)
    _require(not failed, f"run {run.get('id')!r} failed decoded-media checks {failed}")


def _mutation(run: dict[str, Any]) -> None:
    mutation = run.get("mutation")
    _require(isinstance(mutation, dict), f"run {run.get('id')!r} has no explicit mutation receipt")
    kind = mutation.get("kind")
    allowed = {
        "lora_strength",
        "preprocess_generation",
        "geometry",
        "bias",
        "geometry_and_bias",
    }
    _require(kind in allowed, f"run {run.get('id')!r} has unsupported mutation kind {kind!r}")
    base_field = mutation.get("base_field")
    _require(
        isinstance(base_field, str) and base_field in REQUIRED_IDENTITY and base_field.endswith("_sha256"),
        f"run {run.get('id')!r} mutation base_field does not name a frozen digest field",
    )
    before = mutation.get("before_sha256")
    after = mutation.get("after_sha256")
    _require(_hex_digest(before, 64), f"run {run.get('id')!r} mutation before_sha256 is invalid")
    _require(_hex_digest(after, 64), f"run {run.get('id')!r} mutation after_sha256 is invalid")
    _require(before != after, f"run {run.get('id')!r} mutation did not change its contract digest")
    if run["condition"] == "geometry_bias_mutated":
        _require(
            kind in {"geometry", "bias", "geometry_and_bias"},
            f"run {run.get('id')!r} geometry/bias condition has unrelated mutation kind",
        )


def _diagnostic(run: dict[str, Any], report: dict[str, Any], sol: dict[str, Any]) -> None:
    _require(report.get("status") == "pass", f"run {run.get('id')!r} diagnostics are not marked pass")
    _require(report.get("success") is True, f"run {run.get('id')!r} diagnostics do not report success")
    _require(int(report.get("validation_failures", 0)) == 0, "diagnostic report has validation failures")
    if run["implementation"] in {"partitioned_preserved", "partitioned_fixed"}:
        request_ids = report.get("request_ids")
        request_reports = report.get("request_reports")
        _require(
            isinstance(request_ids, list) and request_ids,
            f"run {run.get('id')!r} multi-request diagnostic omitted request_ids",
        )
        _require(
            len(request_ids) == len(set(request_ids)),
            f"run {run.get('id')!r} multi-request diagnostic contains duplicate request_ids",
        )
        _require(
            isinstance(request_reports, list) and len(request_reports) == len(request_ids),
            f"run {run.get('id')!r} multi-request diagnostic request_reports do not match request_ids",
        )
        diagnostic_ids = {item.get("request_id") for item in request_reports if isinstance(item, dict)}
        _require(
            diagnostic_ids == set(request_ids),
            f"run {run.get('id')!r} multi-request diagnostic request report identities disagree",
        )
        for item in request_reports:
            _require(isinstance(item, dict), "diagnostic request report is malformed")
            request_id = item.get("request_id")
            log_provenance = sol["request_provenance"].get(request_id)
            _require(
                log_provenance is not None,
                f"run {run.get('id')!r} diagnostics reference Sol Request {request_id!r} absent from the run log",
            )
            _require(
                item.get("source_generation") == log_provenance.get("source_generation"),
                f"run {run.get('id')!r} diagnostic source generation disagrees with the run log",
            )
            _require(
                item.get("implementation_generation") == log_provenance.get("implementation_generation"),
                f"run {run.get('id')!r} diagnostic implementation generation disagrees with the run log",
            )
        replay = report.get("replay_reports")
        _require(isinstance(replay, dict) and REPLAY_TARGETS.issubset(replay), "required replay targets are missing")
        for target in REPLAY_TARGETS:
            item = replay[target]
            _require(isinstance(item, dict), f"replay target {target!r} is malformed")
            _require(
                int(item.get("primed_compile_misses", -1)) == 0,
                f"replay target {target!r} recompiled when primed",
            )
            _require(item.get("retained_proof_hit") is True, f"replay target {target!r} did not reuse proof")
            if run["condition"] == "cold" and run.get("compiler_cache_state") == "isolated_empty":
                _require(
                    int(item.get("first_compile_misses", 0)) > 0,
                    f"replay target {target!r} observed no first-executable compilation",
                )
    # Replay runs before the live arithmetic gate. On an isolated-empty
    # executable cache, its first arm may perform the compilation and leave the
    # live Request summary with zero compile misses. Cold executable evidence is
    # therefore the per-target first replay arm checked above, not the aggregate
    # live validation counter.


def validate_campaign_manifest(manifest: dict[str, Any], *, root: Path) -> CampaignReport:
    _require(isinstance(manifest, dict), "campaign root must be an object")
    _require(manifest.get("schema_version") == 1, "unsupported campaign schema")
    _require(manifest.get("kind") == CAMPAIGN_KIND, "unexpected campaign kind")

    identity = manifest.get("frozen_identity")
    _require(isinstance(identity, dict), "frozen_identity is missing")
    missing = REQUIRED_IDENTITY - set(identity)
    _require(not missing, f"frozen_identity is missing {sorted(missing)}")
    for name in REQUIRED_IDENTITY:
        if name.endswith("_sha256"):
            _require(_hex_digest(identity.get(name), 64), f"frozen_identity.{name} is invalid")
    _require(
        _hex_digest(identity.get("device_identity"), 64),
        "frozen_identity.device_identity must be the stable Sol device SHA-256",
    )
    _require(type(identity.get("seed")) is int, "frozen_identity.seed must be an integer")
    identity_digest = _canonical_sha256(identity)

    runs = manifest.get("runs")
    _require(isinstance(runs, list) and runs, "campaign contains no runs")
    ids: set[str] = set()
    sequences: set[int] = set()
    source_digests = {name: set() for name in IMPLEMENTATIONS}
    by_impl_condition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    validated = []
    diagnostics = 0

    for run in runs:
        _require(isinstance(run, dict), "campaign run is not an object")
        run_id = run.get("id")
        implementation = run.get("implementation")
        condition = run.get("condition")
        sequence = run.get("sequence")
        _require(isinstance(run_id, str) and run_id, "campaign run has no id")
        _require(run_id not in ids, f"duplicate run id {run_id!r}")
        ids.add(run_id)
        _require(implementation in IMPLEMENTATIONS, f"run {run_id!r} has unknown implementation")
        _require(condition in CONDITIONS, f"run {run_id!r} has unknown condition")
        _require(type(sequence) is int and sequence >= 0, f"run {run_id!r} has invalid sequence")
        _require(sequence not in sequences, f"duplicate sequence {sequence}")
        sequences.add(sequence)
        _require(run.get("frozen_identity_sha256") == identity_digest, f"run {run_id!r} frozen identity mismatch")
        run_source_digest = _source_digest(run)
        source_digests[implementation].add(run_source_digest)

        for name in REQUIRED_ARTIFACTS:
            _artifact(run, name, root)
        metrics = _json(_artifact(run, "metrics", root), f"run {run_id} metrics")
        text = _log(_artifact(run, "log", root))
        timing = run.get("timing")
        _require(isinstance(timing, dict), f"run {run_id!r} has no timing")
        sampler = _positive(timing.get("sampler_s"), f"{run_id}.sampler_s")
        e2e = _positive(timing.get("e2e_s"), f"{run_id}.e2e_s")
        _require(e2e >= sampler, f"run {run_id!r} E2E is shorter than sampler")
        measured_e2e = _e2e_s(text)
        _require(
            abs(measured_e2e - e2e) <= 0.02,
            f"run {run_id!r} E2E timing disagrees with its ComfyUI log ({e2e:.6f}s vs {measured_e2e:.6f}s)",
        )
        measured_sampler = _sampler_s(metrics)
        _require(abs(measured_sampler - sampler) <= 0.005, f"run {run_id!r} sampler timing disagrees with metrics")

        counts = _counts(metrics)
        expected_counts = (
            RELEASED_TARGET_WHOLE_COUNTS if implementation == "released_target" else PARTITIONED_WHOLE_COUNTS
        )
        _require(
            counts == expected_counts,
            f"run {run_id!r} whole-run counts are {counts}, expected {expected_counts}",
        )

        sol = _sol_totals(_sol_records(text))
        _require(
            sol["device_fingerprint"] == identity["device_identity"],
            f"run {run_id!r} device identity differs from the frozen campaign device",
        )
        _require(
            sol["device_identity"]["cuda_driver_version"] == identity["driver"],
            f"run {run_id!r} CUDA driver differs from the frozen campaign driver",
        )
        for name, value in sol["runtime_environment"].items():
            _require(
                identity.get(name) == value,
                f"run {run_id!r} runtime {name} differs from the frozen campaign environment",
            )
        fresh_process = run.get("fresh_process")
        diagnostic_mode = run.get("diagnostic_mode")
        _require(type(fresh_process) is bool, f"run {run_id!r} fresh_process is not boolean")
        _require(type(diagnostic_mode) is bool, f"run {run_id!r} diagnostic_mode is not boolean")
        if condition == PAIR_WARMUP_CONDITION:
            _require(not diagnostic_mode, f"run {run_id!r} pair warmup must use low-overhead mode")
            _require(not run.get("pair_id"), f"run {run_id!r} pair warmup must not carry a measurement pair_id")
        cache_state = run.get("compiler_cache_state")
        _require(isinstance(cache_state, str) and cache_state, f"run {run_id!r} compiler_cache_state is missing")

        if condition == "cold":
            _require(fresh_process, f"run {run_id!r} cold condition is not a fresh process")
            _require(
                cache_state in COLD_CACHE_STATES,
                f"run {run_id!r} cold compiler cache state is not isolated/documented",
            )
        else:
            _require(not fresh_process, f"run {run_id!r} non-cold condition claims a fresh process")
            _require(run.get("fresh_sol_requests") is True, f"run {run_id!r} did not force fresh Sol Requests")
            _require(
                cache_state == "retained_same_process",
                f"run {run_id!r} non-cold compiler cache state is not retained_same_process",
            )
        if condition == "primed":
            _require(sol["compile_misses"] == 0, f"run {run_id!r} primed condition observed compiler misses")
        elif condition in {"numerical_invalidated", "geometry_bias_mutated"}:
            _mutation(run)
            _require(
                run["mutation"]["before_sha256"] == identity[run["mutation"]["base_field"]],
                f"run {run_id!r} mutation before_sha256 does not match its frozen base field",
            )
            _require(run.get("changed_contract_revalidated") is True, f"run {run_id!r} lacks revalidation receipt")
            _require(sol["validation_misses"] > 0, f"run {run_id!r} changed contract produced no validation miss")
        if condition == "geometry_bias_mutated":
            reasons = sol["miss_reasons"]
            _require(
                int(reasons.get("new_geometry", 0)) + int(reasons.get("new_bias", 0)) > 0,
                f"run {run_id!r} geometry/bias mutation produced no matching miss reason",
            )

        if diagnostic_mode:
            diagnostics += 1
            _diagnostic(
                run,
                _json(_artifact(run, "sol_diagnostics", root), f"run {run_id} diagnostics"),
                sol,
            )

        if implementation != "released_target":
            try:
                validate_partitioned_runtime_evidence(
                    metrics,
                    text,
                    expected_logical=PARTITIONED_LATEST_COUNTS[0],
                    expected_actual=PARTITIONED_LATEST_COUNTS[1],
                    expected_forecast=PARTITIONED_LATEST_COUNTS[2],
                    require_performance_accounting=diagnostic_mode,
                )
            except RuntimeGateError as exc:
                raise CampaignEvidenceError(f"run {run_id!r} failed partitioned runtime gate: {exc}") from exc

        _media(run)
        entry = dict(run)
        entry["_sampler_s"] = sampler
        entry["_e2e_s"] = e2e
        entry["_source_digest"] = run_source_digest
        entry["_process_id"] = sol["process_id"]
        entry["_process_generation"] = sol["process_generation"]
        validated.append(entry)
        by_impl_condition.setdefault((implementation, condition), []).append(entry)

    by_run_id = {run["id"]: run for run in validated}
    for run in validated:
        if run["condition"] == "cold":
            _require(
                run.get("process_anchor_run_id") in {None, ""},
                f"run {run['id']!r} cold condition must not point at another process anchor",
            )
            continue
        anchor_id = run.get("process_anchor_run_id")
        _require(
            isinstance(anchor_id, str) and anchor_id,
            f"run {run['id']!r} non-cold condition has no cold process anchor",
        )
        anchor = by_run_id.get(anchor_id)
        _require(anchor is not None, f"run {run['id']!r} process anchor {anchor_id!r} does not exist")
        _require(
            anchor["condition"] == "cold",
            f"run {run['id']!r} process anchor {anchor_id!r} is not cold evidence",
        )
        timing_pair_run = (
            run["condition"] == "primed"
            and run["implementation"] in {"released_target", "partitioned_fixed"}
            and isinstance(run.get("pair_id"), str)
            and bool(run.get("pair_id"))
        )
        shared_pair_process_run = timing_pair_run or run["condition"] == PAIR_WARMUP_CONDITION
        if not shared_pair_process_run:
            _require(
                anchor["implementation"] == run["implementation"],
                f"run {run['id']!r} process anchor belongs to a different implementation",
            )
        _require(
            anchor["_source_digest"] == run["_source_digest"],
            f"run {run['id']!r} process anchor uses a different source stack",
        )
        _require(
            anchor["sequence"] < run["sequence"],
            f"run {run['id']!r} appears before its cold process anchor",
        )
        _require(
            anchor["_process_id"] == run["_process_id"],
            f"run {run['id']!r} is not from the same process ID as its cold anchor",
        )
        _require(
            anchor["_process_generation"] == run["_process_generation"],
            f"run {run['id']!r} is not from the same process generation as its cold anchor",
        )

    for implementation in IMPLEMENTATIONS:
        _require(len(source_digests[implementation]) == 1, f"{implementation} source stack changed within campaign")
        for condition in EVIDENCE_CONDITIONS:
            _require(
                by_impl_condition.get((implementation, condition)),
                f"missing {implementation}/{condition} evidence",
            )
        canonical_primed = [run for run in by_impl_condition[(implementation, "primed")] if not run.get("pair_id")]
        _require(
            canonical_primed,
            f"missing unpaired same-arm primed evidence for {implementation}",
        )
    for implementation in ("partitioned_preserved", "partitioned_fixed"):
        _require(
            any(run["implementation"] == implementation and run["diagnostic_mode"] for run in validated),
            f"missing diagnostic replay/CUDA evidence for {implementation}",
        )

    paired = [
        run
        for run in validated
        if run["condition"] == "primed"
        and run["implementation"] in {"released_target", "partitioned_fixed"}
        and isinstance(run.get("pair_id"), str)
        and run["pair_id"]
    ]
    pair_ids = sorted({run["pair_id"] for run in paired})
    _require(len(pair_ids) >= 3, "promotion requires at least three paired primed repetitions")
    _require(all(not run["diagnostic_mode"] for run in paired), "paired timing runs must use low-overhead mode")
    ordered = sorted(paired, key=lambda run: run["sequence"])
    paired_processes = {(run["_process_id"], run["_process_generation"], run["_source_digest"]) for run in paired}
    _require(
        len(paired_processes) == 1,
        "all measured timing pairs must share one already-primed process/source stack",
    )
    pair_process = next(iter(paired_processes))
    first_pair_sequence = min(run["sequence"] for run in paired)
    pair_warmups = [
        run
        for run in validated
        if run["condition"] == PAIR_WARMUP_CONDITION
        and (run["_process_id"], run["_process_generation"], run["_source_digest"]) == pair_process
        and run["sequence"] < first_pair_sequence
    ]
    warmup_implementations = {run["implementation"] for run in pair_warmups}
    _require(
        {"released_target", "partitioned_fixed"}.issubset(warmup_implementations),
        "measured timing process lacks pre-measurement warmup of both control and fixed modes",
    )

    pair_reports = []
    sampler_deltas = []
    e2e_deltas = []
    for pair_id in pair_ids:
        group = [run for run in paired if run["pair_id"] == pair_id]
        _require(len(group) == 2, f"pair {pair_id!r} must contain exactly two runs")
        by_impl = {run["implementation"]: run for run in group}
        _require(set(by_impl) == {"released_target", "partitioned_fixed"}, f"pair {pair_id!r} is incomplete")
        positions = sorted(ordered.index(run) for run in group)
        _require(positions[1] == positions[0] + 1, f"pair {pair_id!r} is not adjacent among paired timing runs")
        control = by_impl["released_target"]
        fixed = by_impl["partitioned_fixed"]
        _require(
            abs(int(control["sequence"]) - int(fixed["sequence"])) == 1,
            f"pair {pair_id!r} is not adjacent in the complete campaign order",
        )
        _require(
            control["_source_digest"] == fixed["_source_digest"],
            f"pair {pair_id!r} control/fixed source stacks differ",
        )
        _require(
            control["_process_id"] == fixed["_process_id"]
            and control["_process_generation"] == fixed["_process_generation"],
            f"pair {pair_id!r} was not measured in one shared primed process",
        )
        sampler_delta = control["_sampler_s"] - fixed["_sampler_s"]
        e2e_delta = control["_e2e_s"] - fixed["_e2e_s"]
        _require(sampler_delta > 0.0, f"pair {pair_id!r} has no sampler advantage")
        _require(e2e_delta > 0.0, f"pair {pair_id!r} has no E2E advantage")
        sampler_deltas.append(sampler_delta)
        e2e_deltas.append(e2e_delta)
        pair_reports.append(
            {
                "pair_id": pair_id,
                "control_run": control["id"],
                "fixed_run": fixed["id"],
                "sampler_delta_s": sampler_delta,
                "e2e_delta_s": e2e_delta,
            }
        )

    return CampaignReport(
        frozen_identity_sha256=identity_digest,
        runs=len(validated),
        diagnostic_runs=diagnostics,
        pair_count=len(pair_ids),
        sampler_pair_wins=len(pair_ids),
        e2e_pair_wins=len(pair_ids),
        sampler_median_delta_s=float(statistics.median(sampler_deltas)),
        e2e_median_delta_s=float(statistics.median(e2e_deltas)),
        decoded_video_passes=len(validated),
        decoded_audio_passes=len(validated),
        implementation_source_digests={name: next(iter(source_digests[name])) for name in IMPLEMENTATIONS},
        pair_reports=tuple(pair_reports),
    )


__all__ = [
    "CAMPAIGN_KIND",
    "PAIR_WARMUP_CONDITION",
    "CampaignEvidenceError",
    "CampaignReport",
    "validate_campaign_manifest",
]
