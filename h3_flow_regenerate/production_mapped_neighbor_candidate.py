"""Bounded production-v4 first-high candidate replay.

This validation layer reuses the hash-validated experiment-R first-high bundle but
executes the ordinary installed VDN retained path and ordinary packaged Sol-H3 ABI.
It does not import or install W/E/M operator substitutions, selector replacements,
or route-label normalization. Its only runtime instrumentation is a bounded
backend-receipt sink plus the existing execution-contract recorder.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib
import json
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import comfy_compat as _comfy_compat
from . import execution_contract_diagnostics as _diag
from . import runtime as _runtime
from . import same_state_high_replay as _replay
from .execution_contract_provenance import callable_identity as _callable_identity
from .geometry import unpack_streams

SCHEMA_VERSION = 1
MODE = "production_mapped_neighbor_v4_candidate"
DESIGN_COMMIT = "941b8571b099d18d1fe4e4deef9f04fd343b1098"
EXPECTED_CAPTURE_ID = "234ed062128e43ed8d5ec63e27517b22"
STATE_KEY = "h3_flow_production_mapped_neighbor_candidate_v1"
RECEIPTS_KEY = "attention_backend_receipts_v1"
_OUTER_KEY = "h3_flow_regenerate.production_mapped_neighbor_candidate.execute.v1"
_SAMPLER_KEY = "h3_flow_regenerate.production_mapped_neighbor_candidate.sampler_entry.v1"
_SOURCE_MANIFEST = "production_mapped_neighbor_candidate_source_delta.json"
_REQUIRED_SUFFIX = [0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]
_SOL_CONTRACT = "sana-sol-engine-sol-attn-64-rect-sm120-mapped-neighbor-v4"
_MAPPED_RECEIPT_TAG = "vdn_mapped_neighbor_v1"
_MAPPED_POLICY = "k64-union-radius1-additive-interval4-v1"
_MAX_REPORTS = 4
_EXPECTED_BLOCKS = 50
_EXPECTED_PACKED_ROWS = 56349
_EXPECTED_VIDEO_SPAN = (3101, 56349)
_EXPECTED_LOCAL_Q_ROWS = (4096,) + (5120,) * 9 + (1024,)
_EXPECTED_WINDOW_KV_ROWS = (14365, 19485) + (20509,) * 7 + (16413, 11293)
_EXPECTED_ENTRY_HASHES = {
    "first_high_sampler_input_video": "79dca62849b2a159061a1d6204828af7a4c5995c41beec256c1719a5932a41ee",
    "first_high_sampler_input_audio": "f5fd588101bf2ab68eb5aeeb49bfaf8e58f4c71aa99864de89cb1ddef8c859f3",
    "first_high_h3_input_video": "b49f17a317530795be43bc775486777b07d5b5dbb28996819033cdede64195c0",
    "first_high_h3_input_audio": "f54b22ff25a675c47a4c32aba54b5642033b048441ad6eec9bf5ef935cd29002",
}
_FORBIDDEN_DIAGNOSTIC_MODULE_SUFFIXES = (
    "first_high_operator_comparison",
    "first_high_sol_local_diagnostic",
    "first_high_mapped_neighbor_diagnostic",
    "first_high_operator_diagnostic",
    "first_high_sol_local_receipt_tap",
    "first_high_sol_local_witness_bridge",
)
_EXPECTED_SOURCE_ENTRY_KEYS = frozenset(
    {
        ("flow", "h3_flow_regenerate.production_mapped_neighbor_candidate", "."),
        ("flow", "h3_flow_regenerate.production_mapped_neighbor_candidate", "../__init__.py"),
        ("sol", "sol_h3.interop", "."),
        ("sol", "sol_h3.runtime", "."),
        ("sol", "sol_h3.sparse", "."),
        ("sol", "sol_h3.provenance", "."),
        ("sol", "sol_h3.provenance", "sol_manifest.json"),
        ("sol", "sol_h3.mapped_neighbors", "."),
        ("sol", "sol_h3._vendor.sol_attn.interface", "."),
        ("sol", "sol_h3._vendor.sol_attn.sm120.kernel", "."),
        ("sol", "sol_h3._vendor.sol_attn.sm120.mainloop", "."),
        ("vdn", "vdn_h3.hybrid", "."),
        ("vdn", "vdn_h3.retained", "."),
        ("vdn", "vdn_h3.softmax_provider", "."),
        ("vdn", "vdn_h3.query_positions", "."),
    }
)
_PRIVATE_REPORT_KEYS = frozenset({"_media_raw", "_media_pre", "_media_final"})


class _FirstCallComplete(BaseException):
    def __init__(self, token: object, x0: torch.Tensor):
        super().__init__("production mapped-neighbor candidate completed its single first-high model call")
        self.token = token
        self.x0 = x0


@dataclass(slots=True)
class _ReceiptSink:
    """Identity-stable receipt owner across Core model-option copies."""

    items: list[Any] = field(default_factory=list)
    limit: int = 800

    def append(self, value: Any) -> None:
        if len(self.items) >= self.limit:
            raise RuntimeError("production candidate receipt budget exceeded")
        self.items.append(value)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


@dataclass(slots=True)
class _State:
    replay: _replay._ReplayState
    source_gate: dict[str, Any]
    complete: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_REPORTS))


@dataclass(slots=True)
class _Call:
    state: _State
    token: object = field(default_factory=object)
    completed: bool = False
    raw_x0: torch.Tensor | None = None


_ACTIVE_CALL: contextvars.ContextVar[_Call | None] = contextvars.ContextVar(
    "h3_flow_production_mapped_neighbor_candidate_call", default=None
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _valid_git_blob_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _source_entry_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("owner", "")),
        str(entry.get("module", "")),
        str(entry.get("relative_path", ".")),
    )


def _load_source_manifest() -> tuple[dict[str, Any], str]:
    path = Path(__file__).with_name(_SOURCE_MANIFEST)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("production candidate source-delta manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("production candidate source-delta manifest schema is unsupported")
    if manifest.get("design_commit") != DESIGN_COMMIT:
        raise RuntimeError("production candidate source-delta manifest targets the wrong design commit")
    if manifest.get("sol_pr_head") != "1459b34853a39d1054fd5c8643de28a7b0e240c7":
        raise RuntimeError("production candidate source-delta manifest targets the wrong Sol production head")
    if manifest.get("vdn_pr_head") != "333d63f81d33fe29dc1f1f637c5f4a4396880f99":
        raise RuntimeError("production candidate source-delta manifest targets the wrong VDN production head")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("production candidate source-delta manifest has no reviewed entries")
    keys = [_source_entry_key(entry) for entry in entries if isinstance(entry, dict)]
    if len(keys) != len(entries) or len(keys) != len(set(keys)):
        raise RuntimeError("production candidate source-delta manifest entries are invalid or duplicated")
    if frozenset(keys) != _EXPECTED_SOURCE_ENTRY_KEYS:
        raise RuntimeError("production candidate source-delta manifest differs from the reviewed exact entry set")
    return manifest, _sha_json(manifest)


def _resolve_source_entry(entry: dict[str, Any]) -> Path:
    module_name = entry.get("module")
    relative = entry.get("relative_path", ".")
    if not isinstance(module_name, str) or not module_name or not isinstance(relative, str):
        raise RuntimeError("production candidate source entry has invalid module/path metadata")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"production candidate required source module is not importable: {module_name}") from exc
    raw_file = getattr(module, "__file__", None)
    if not isinstance(raw_file, str) or not raw_file:
        raise RuntimeError(f"production candidate source module has no file: {module_name}")
    base = Path(raw_file).resolve(strict=True)
    candidate = base if relative == "." else (base.parent / relative).resolve(strict=True)
    if not candidate.is_file():
        raise RuntimeError(f"production candidate source path is not a file: {candidate}")
    return candidate


def _forbidden_diagnostic_modules() -> list[str]:
    result = []
    for name in sys.modules:
        lowered = name.lower()
        if any(lowered.endswith(suffix) for suffix in _FORBIDDEN_DIAGNOSTIC_MODULE_SUFFIXES):
            result.append(name)
    return sorted(result)


def _verify_source_manifest() -> dict[str, Any]:
    manifest, digest = _load_source_manifest()
    observed = []
    for entry in manifest["entries"]:
        path = _resolve_source_entry(entry)
        expected_blob = entry.get("candidate_git_blob_sha")
        capture_sha = entry.get("capture_sha256")
        if not _valid_git_blob_sha(expected_blob):
            raise RuntimeError("production candidate source entry lacks exact candidate Git blob identity")
        if capture_sha is not None and not _valid_sha256(capture_sha):
            raise RuntimeError("production candidate source entry has invalid R-capture SHA-256")
        actual_blob = _git_blob_sha(path)
        actual_sha = _sha256_file(path)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"production candidate source bytes differ from reviewed candidate: {path}: "
                f"{actual_blob} != {expected_blob}"
            )
        observed.append(
            {
                **entry,
                "path": str(path),
                "git_blob_sha": actual_blob,
                "sha256": actual_sha,
            }
        )

    try:
        from sol_h3.provenance import CONTRACT as sol_contract
        from sol_h3.provenance import REVISION as sana_revision
        from sol_h3.provenance import verify_source as verify_sol_source
        from vdn_h3.softmax_provider import PROVIDER_API_VERSION
    except Exception as exc:
        raise RuntimeError("production candidate cannot import paired Sol/VDN production contracts") from exc
    verified = verify_sol_source()
    if sol_contract != _SOL_CONTRACT or verified.get("contract") != _SOL_CONTRACT:
        raise RuntimeError("production candidate requires the mapped-neighbor-v4 packaged Sol contract")
    if int(PROVIDER_API_VERSION) != 4:
        raise RuntimeError("production candidate requires VDN provider API v4")
    forbidden = _forbidden_diagnostic_modules()
    if forbidden:
        raise RuntimeError(
            "production candidate process has W/E/M operator diagnostic modules loaded: " + ", ".join(forbidden)
        )
    return {
        "manifest_digest": digest,
        "entries": observed,
        "sol_contract": sol_contract,
        "sana_revision": sana_revision,
        "vdn_provider_api": int(PROVIDER_API_VERSION),
        "forbidden_diagnostic_modules_loaded": forbidden,
        "operator_substitution_absent": True,
    }


def _source_gate_by_path(source_gate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for entry in source_gate.get("entries", []):
        path = str(entry.get("path", ""))
        if not path or path in result:
            raise RuntimeError("production candidate source gate has ambiguous resolved paths")
        result[path] = entry
    return result


def _normalize_approved_callables(value: Any, source_gate: dict[str, Any], *, side: str) -> Any:
    """Collapse only exact source-backed identities from reviewed changed files.

    Capture-side identities must carry the manifest's exact R SHA-256; current-side
    identities must carry the verified candidate SHA-256. The collapse keeps the
    callable qualname and reviewed candidate blob in the comparable identity, so a
    different runtime owner cannot be hidden by the source-delta allowance.
    """
    by_path = _source_gate_by_path(source_gate)
    if isinstance(value, dict):
        file_info = value.get("file")
        if isinstance(file_info, dict):
            raw_path = file_info.get("resolved_path") or file_info.get("path")
            entry = by_path.get(str(raw_path)) if isinstance(raw_path, str) else None
            if entry is not None and isinstance(value.get("module"), str) and isinstance(value.get("qualname"), str):
                expected = entry.get("capture_sha256") if side == "capture" else entry.get("sha256")
                observed = file_info.get("sha256")
                if expected is None:
                    if side == "capture":
                        raise RuntimeError(f"new production source unexpectedly appears in R capture: {raw_path}")
                elif observed != expected:
                    raise RuntimeError(
                        f"{side} callable source identity differs from reviewed delta for {raw_path}: "
                        f"{observed!r} != {expected!r}"
                    )
                return {
                    "approved_production_source_delta_callable": {
                        "owner": entry.get("owner"),
                        "module": entry.get("module"),
                        "qualname": value.get("qualname"),
                        "candidate_git_blob_sha": entry.get("candidate_git_blob_sha"),
                    }
                }
        result = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if side == "current" and key in {STATE_KEY, _OUTER_KEY, _SAMPLER_KEY}:
                continue
            result[key] = _normalize_approved_callables(item, source_gate, side=side)
        return result
    if isinstance(value, list):
        return [_normalize_approved_callables(item, source_gate, side=side) for item in value]
    return value


def _companion_source_diff(
    capture_groups: Any,
    current_groups: Any,
    source_gate: dict[str, Any],
    *,
    limit: int = 24,
) -> list[str]:
    if not isinstance(capture_groups, dict) or not isinstance(current_groups, dict):
        return ["$.loaded_companion_sources"]
    approved = _source_gate_by_path(source_gate)
    differences: list[str] = []
    for label in sorted(set(capture_groups) | set(current_groups)):
        if len(differences) >= limit:
            break
        path = f"$.loaded_companion_sources.{label}"
        if label not in capture_groups or label not in current_groups:
            differences.append(path + " (group missing)")
            continue
        capture, capture_problems = _replay._companion_source_map(capture_groups[label])
        current, current_problems = _replay._companion_source_map(current_groups[label])
        differences.extend(path + ".capture_identity: " + item for item in capture_problems)
        differences.extend(path + ".current_identity: " + item for item in current_problems)
        if len(differences) >= limit:
            break
        for source_path, capture_entry in sorted(capture.items()):
            if len(differences) >= limit:
                break
            delta = approved.get(source_path)
            if delta is not None:
                base_sha = delta.get("capture_sha256")
                if base_sha is None:
                    differences.append(path + f".new_source_present_in_capture[{source_path}]")
                    continue
                if capture_entry.get("sha256") != base_sha:
                    differences.append(path + f".capture_delta_sha256[{source_path}]")
                    continue
                current_entry = current.get(source_path)
                if current_entry is not None and current_entry.get("sha256") != delta.get("sha256"):
                    differences.append(path + f".current_delta_sha256[{source_path}]")
                continue
            try:
                resolved = Path(source_path).resolve(strict=True)
                disk_sha = _sha256_file(resolved)
            except (OSError, RuntimeError):
                differences.append(path + f".capture_source[{source_path}].unreadable")
                continue
            if str(resolved) != source_path or disk_sha != capture_entry.get("sha256"):
                differences.append(path + f".capture_source[{source_path}].on_disk_sha256")
                continue
            current_entry = current.get(source_path)
            if current_entry is not None and current_entry.get("sha256") != capture_entry.get("sha256"):
                differences.append(path + f".shared_source[{source_path}].sha256")
        if len(differences) >= limit:
            break
        for source_path, current_entry in sorted(current.items()):
            if source_path in capture:
                continue
            delta = approved.get(source_path)
            if (
                delta is not None
                and delta.get("capture_sha256") is None
                and current_entry.get("sha256") == delta.get("sha256")
            ):
                continue
            differences.append(path + f".current_only_source[{source_path}]")
            if len(differences) >= limit:
                break
    return differences[:limit]


def _callable_equivalence_identity(value: Any) -> dict[str, Any]:
    identity = _callable_identity(value)
    normalized = _replay._provenance_equivalence_identity({"value": identity})
    result = normalized.get("value") if isinstance(normalized, dict) else None
    if not isinstance(result, dict):
        raise RuntimeError("production candidate could not normalize validation wrapper callable identity")
    return result


def _wrapper_specs():
    import comfy.patcher_extension

    return (
        (comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _OUTER_KEY, _outer_wrapper),
        (comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, _SAMPLER_KEY, _sampler_entry_wrapper),
    )


def _normalize_validation_wrapper_order(current: dict[str, Any], guider: Any) -> dict[str, Any]:
    """Remove only the two proven validation wrappers from comparable R provenance.

    R predates this validation layer, so its wrapper-order manifests cannot contain
    these entries. Prove the live ModelPatcher owns exactly the expected callables
    at the expected wrapper types before removing their manifest entries. This is
    instrumentation normalization only; production Sol/VDN wrappers remain exact.
    """
    patcher = getattr(guider, "model_patcher", None)
    runtime_wrappers = getattr(patcher, "wrappers", None)
    if not isinstance(runtime_wrappers, dict):
        raise RuntimeError("production candidate cannot inspect live ModelPatcher wrapper ownership")
    specs = _wrapper_specs()
    for wrapper_type, key, expected_callable in specs:
        locations = [
            existing_type
            for existing_type, keyed in runtime_wrappers.items()
            if isinstance(keyed, dict) and key in keyed
        ]
        if len(locations) != 1 or locations[0] != wrapper_type:
            raise RuntimeError(f"production candidate live wrapper key ownership changed for {key}: {locations!r}")
        keyed = runtime_wrappers.get(wrapper_type)
        values = keyed.get(key) if isinstance(keyed, dict) else None
        if not isinstance(values, (list, tuple)) or len(values) != 1 or values[0] is not expected_callable:
            raise RuntimeError(f"production candidate live wrapper callable identity changed for {key}")

    result = dict(current)
    for field_name in ("active_wrapper_order", "patcher_wrapper_order"):
        manifest = result.get(field_name)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"production candidate provenance lacks {field_name}")
        rebuilt = dict(manifest)
        for wrapper_type, key, expected_callable in specs:
            manifest_key = str(wrapper_type)
            entries = rebuilt.get(manifest_key)
            if not isinstance(entries, list):
                raise RuntimeError(f"production candidate provenance lacks wrapper list {field_name}.{manifest_key}")
            matches = [
                (index, item) for index, item in enumerate(entries) if isinstance(item, dict) and item.get("key") == key
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"production candidate provenance must contain exactly one {field_name} entry for {key}"
                )
            index, item = matches[0]
            expected_identity = _callable_equivalence_identity(expected_callable)
            if _canonical_json(item.get("callable")) != _canonical_json(expected_identity):
                raise RuntimeError(f"production candidate provenance callable changed for {field_name}:{key}")
            rebuilt[manifest_key] = [entry for ordinal, entry in enumerate(entries) if ordinal != index]
        result[field_name] = rebuilt
    return result


def _provenance_gate(state: _State, record: _diag._Record, guider: Any) -> dict[str, Any]:
    capture = _replay._bundle_provenance_equivalence_identity(state.replay.manifest)
    current = _replay._provenance_equivalence_identity(record.state.manifest)
    current = _normalize_validation_wrapper_order(current, guider)
    capture_sources = capture.get("loaded_companion_sources") if isinstance(capture, dict) else None
    current_sources = current.get("loaded_companion_sources") if isinstance(current, dict) else None
    source_differences = _companion_source_diff(capture_sources, current_sources, state.source_gate)
    capture_base = {k: v for k, v in capture.items() if k != "loaded_companion_sources"}
    current_base = {k: v for k, v in current.items() if k != "loaded_companion_sources"}
    capture_norm = _normalize_approved_callables(capture_base, state.source_gate, side="capture")
    current_norm = _normalize_approved_callables(current_base, state.source_gate, side="current")
    semantic_differences = _replay._provenance_diff_paths(capture_norm, current_norm, limit=24)
    differences = (semantic_differences + source_differences)[:24]
    return {
        "exact_except_reviewed_production_delta": not differences,
        "unexpected_differences": differences,
        "source_manifest_digest": state.source_gate.get("manifest_digest"),
        "operator_substitution_absent": state.source_gate.get("operator_substitution_absent") is True,
    }


def _numeric_attr(value: Any, name: str) -> float | None:
    raw = getattr(value, name, None)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return None


def _sampling_runtime_identity(guider: Any) -> dict[str, Any]:
    import comfy.latent_formats
    import comfy.model_sampling

    patcher = getattr(guider, "model_patcher", None)
    base = getattr(patcher, "model", None)
    sampling = getattr(base, "model_sampling", None)
    latent = getattr(base, "latent_format", None)
    diffusion = getattr(base, "diffusion_model", None)
    final_layer = getattr(diffusion, "final_layer", None)
    video_out = getattr(final_layer, "video_out", None)
    head_count = None
    weight = getattr(video_out, "weight", None)
    out_features = getattr(video_out, "out_features", None)
    if torch.is_tensor(weight) and type(out_features) is int and out_features > 0:
        head_count = int(weight.shape[0]) // int(out_features)
    return {
        "model_sampling_class": (
            None if sampling is None else f"{type(sampling).__module__}.{type(sampling).__qualname__}"
        ),
        "model_sampling_av": isinstance(sampling, comfy.model_sampling.ModelSamplingAV),
        "model_sampling_const": isinstance(sampling, comfy.model_sampling.CONST),
        "noise_scale": _numeric_attr(sampling, "noise_scale"),
        "shift": _numeric_attr(sampling, "shift"),
        "audio_shift": _numeric_attr(sampling, "audio_shift"),
        "latent_format_class": None if latent is None else f"{type(latent).__module__}.{type(latent).__qualname__}",
        "latent_format_minimax_h3_av": isinstance(latent, comfy.latent_formats.MiniMaxH3AV),
        "latent_scale_factor": _numeric_attr(latent, "scale_factor"),
        "final_head_count": head_count,
    }


def _sampling_runtime_ok(identity: dict[str, Any]) -> bool:
    noise_scale = identity.get("noise_scale")
    return bool(
        identity.get("model_sampling_av") is True
        and identity.get("model_sampling_const") is True
        and identity.get("latent_format_minimax_h3_av") is True
        and isinstance(noise_scale, float)
        and math.isfinite(noise_scale)
        and noise_scale > 0.0
        and type(identity.get("final_head_count")) is int
        and identity["final_head_count"] > 0
    )


def _export_contract(guider: Any, raw_x0: torch.Tensor, returned: Any) -> dict[str, Any]:
    if not torch.is_tensor(raw_x0) or not torch.is_tensor(returned):
        raise RuntimeError("production candidate export contract requires packed tensor callback/output values")
    base = getattr(getattr(guider, "model_patcher", None), "model", None)
    process_out = getattr(base, "process_latent_out", None)
    if not callable(process_out):
        raise RuntimeError("production candidate cannot resolve active Core process_latent_out")
    expected = process_out(raw_x0.to(torch.float32))
    if not torch.is_tensor(expected):
        raise RuntimeError("production candidate process_latent_out did not return a tensor")
    exact = (
        tuple(returned.shape) == tuple(expected.shape)
        and returned.dtype == expected.dtype
        and returned.device == expected.device
        and torch.equal(returned, expected)
    )
    return {
        "exact_core_callback_x0_externalization": exact,
        "raw_x0_sha256": _replay._tensor_sha256(raw_x0),
        "expected_export_sha256": _replay._tensor_sha256(expected),
        "returned_sha256": _replay._tensor_sha256(returned),
        "returned_shape": [int(dim) for dim in returned.shape],
        "boundary": "SAMPLER_SAMPLE callback x0 -> CFGGuider.inner_sample process_latent_out",
    }


def _core_cleanup_contract(guider: Any, options: dict[str, Any]) -> dict[str, Any]:
    inner_absent = not hasattr(guider, "inner_model")
    loaded_absent = not hasattr(guider, "loaded_models")
    pool_absent = "multigpu_thread_pool" not in options
    return {
        "inner_model_absent": inner_absent,
        "loaded_models_absent": loaded_absent,
        "multigpu_thread_pool_absent": pool_absent,
        "complete": inner_absent and loaded_absent and pool_absent,
    }


def _snapshot_report(record: _diag._Record) -> dict[str, Any]:
    result = {}
    for name, expected in _EXPECTED_ENTRY_HASHES.items():
        snapshot = record.snapshots.get(name)
        actual = None if snapshot is None else _replay._tensor_sha256(snapshot.tensor)
        result[name] = {"sha256": actual, "expected_sha256": expected, "exact": actual == expected}
    return result


def _decode_ready_snapshot(record: _diag._Record, name: str, expected_shape: tuple[int, ...]) -> torch.Tensor:
    snapshot = record.snapshots.get(name)
    if snapshot is None or not torch.is_tensor(snapshot.tensor):
        raise RuntimeError(f"production candidate is missing decode-ready media snapshot: {name}")
    value = snapshot.tensor
    if value.ndim != 5 or tuple(int(dim) for dim in value.shape) != expected_shape:
        raise RuntimeError(
            f"production candidate media {name} has {tuple(int(dim) for dim in value.shape)}, expected {expected_shape}"
        )
    return value


def _decode_final_video(output: torch.Tensor, target_shapes: list[tuple[int, ...]]) -> torch.Tensor:
    video, _audio = unpack_streams(output, target_shapes)
    expected = target_shapes[0]
    if video.ndim != 5 or tuple(int(dim) for dim in video.shape) != expected:
        raise RuntimeError("production candidate final packed output did not unpack to captured target video geometry")
    return video


def _validate_backend_receipts(receipts: _ReceiptSink) -> dict[str, Any]:
    items = list(receipts.items)
    parsed = []
    malformed = []
    for index, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) not in {3, 4}:
            malformed.append(index)
            continue
        owner, block, route = item[:3]
        fields = item[3] if len(item) == 4 else None
        if (
            owner != "sol_h3"
            or type(block) is not int
            or not 0 <= block < _EXPECTED_BLOCKS
            or not isinstance(route, str)
        ):
            malformed.append(index)
            continue
        parsed.append((block, route, fields))
    routes = Counter(route for _block, route, _fields in parsed)
    blocks = Counter(block for block, _route, _fields in parsed)
    expected_routes = Counter(
        {
            "vdn_local_sol_mapped_v1": 528,
            "vdn_dense_warmup": 22,
            "vdn_global_native": 50,
            "vdn_anchor_native": 100,
        }
    )
    per_block_ok = len(blocks) == _EXPECTED_BLOCKS and all(blocks.get(block) == 14 for block in range(_EXPECTED_BLOCKS))
    per_block_route_topology_ok = True
    for block in range(_EXPECTED_BLOCKS):
        block_routes = Counter(route for item_block, route, _fields in parsed if item_block == block)
        expected_block_routes = Counter(
            {
                "vdn_dense_warmup" if block < 2 else "vdn_local_sol_mapped_v1": 11,
                "vdn_global_native": 1,
                "vdn_anchor_native": 2,
            }
        )
        if block_routes != expected_block_routes:
            per_block_route_topology_ok = False
            break

    mapped = [(block, fields) for block, route, fields in parsed if route == "vdn_local_sol_mapped_v1"]
    mapped_fields_valid = True
    mapped_by_block: dict[int, list[tuple[Any, ...]]] = {}
    mapped_identities_by_group: dict[int, set[tuple[Any, ...]]] = {}
    owners = set()
    plans = set()
    for block, fields in mapped:
        if not isinstance(fields, tuple) or len(fields) != 12:
            mapped_fields_valid = False
            continue
        (
            tag,
            owner_generation,
            plan_digest,
            group_index,
            q_rows,
            kv_rows,
            sink_rows,
            map_digest,
            descriptor_digest,
            policy,
            contract,
            executed,
        ) = fields
        valid = bool(
            tag == _MAPPED_RECEIPT_TAG
            and isinstance(owner_generation, str)
            and owner_generation
            and _valid_sha256(plan_digest)
            and type(group_index) is int
            and 0 <= group_index < 11
            and type(q_rows) is int
            and q_rows > 0
            and type(kv_rows) is int
            and kv_rows > 0
            and sink_rows == _EXPECTED_VIDEO_SPAN[0]
            and _valid_sha256(map_digest)
            and _valid_sha256(descriptor_digest)
            and policy == _MAPPED_POLICY
            and contract == _SOL_CONTRACT
            and executed is True
        )
        mapped_fields_valid = mapped_fields_valid and valid
        if valid:
            owners.add(owner_generation)
            plans.add(plan_digest)
            mapped_by_block.setdefault(block, []).append(fields)
            mapped_identities_by_group.setdefault(group_index, set()).add(
                (q_rows, kv_rows, sink_rows, map_digest, descriptor_digest)
            )
    mapped_geometry_ok = True
    for block in range(2, _EXPECTED_BLOCKS):
        values = sorted(mapped_by_block.get(block, []), key=lambda value: value[3])
        if len(values) != 11:
            mapped_geometry_ok = False
            continue
        if [value[3] for value in values] != list(range(11)):
            mapped_geometry_ok = False
        if tuple(value[4] for value in values) != _EXPECTED_LOCAL_Q_ROWS:
            mapped_geometry_ok = False
        if tuple(value[5] for value in values) != _EXPECTED_WINDOW_KV_ROWS:
            mapped_geometry_ok = False
    mapped_identity_consistent = bool(
        set(mapped_identities_by_group) == set(range(11))
        and all(len(values) == 1 for values in mapped_identities_by_group.values())
    )
    warmup_only_first_two = (
        all((route != "vdn_dense_warmup") or block in {0, 1} for block, route, _fields in parsed)
        and sum(1 for block, route, _fields in parsed if route == "vdn_dense_warmup" and block == 0) == 11
        and sum(1 for block, route, _fields in parsed if route == "vdn_dense_warmup" and block == 1) == 11
    )
    no_native_local_fallback = not any(
        route.startswith("vdn_local_native") or route.startswith("kernel_unavailable:")
        for _block, route, _fields in parsed
    )
    valid = bool(
        len(items) == 700
        and not malformed
        and routes == expected_routes
        and per_block_ok
        and per_block_route_topology_ok
        and len(mapped) == 528
        and mapped_fields_valid
        and mapped_geometry_ok
        and mapped_identity_consistent
        and len(owners) == 1
        and len(plans) == 1
        and warmup_only_first_two
        and no_native_local_fallback
    )
    return {
        "count": len(items),
        "malformed_indices": malformed,
        "routes": dict(routes),
        "expected_routes": dict(expected_routes),
        "per_block_14_calls_exact": per_block_ok,
        "per_block_route_topology_exact": per_block_route_topology_ok,
        "mapped_receipt_count": len(mapped),
        "mapped_fields_valid": mapped_fields_valid,
        "mapped_geometry_exact": mapped_geometry_ok,
        "mapped_identity_consistent_by_group": mapped_identity_consistent,
        "owner_generation_count": len(owners),
        "plan_digest_count": len(plans),
        "warmup_only_first_two_blocks": warmup_only_first_two,
        "no_native_local_fallback": no_native_local_fallback,
        "valid": valid,
    }


def _sampler_entry_wrapper(
    executor,
    model_wrap,
    sigmas,
    extra_args,
    callback,
    noise,
    latent_image=None,
    denoise_mask=None,
    disable_pbar=False,
):
    options = extra_args.get("model_options") if isinstance(extra_args, dict) else None
    state = (options or {}).get(STATE_KEY)
    if not isinstance(state, _State) or _diag._stage_name(options) != "high":
        return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    record = _diag._ACTIVE.get()
    if record is None:
        raise RuntimeError("production candidate sampler-entry validation requires active execution diagnostics")
    _replay._validate_replay_sampler_entry(state.replay, record)
    call = _ACTIVE_CALL.get()
    if not isinstance(call, _Call) or call.state is not state:
        raise RuntimeError("production candidate sampler entry has no matching completion owner")
    try:
        return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    except _FirstCallComplete as exc:
        if exc.token is not call.token:
            raise
        if call.completed or not torch.is_tensor(exc.x0):
            raise RuntimeError("production candidate first-call completion sentinel is invalid") from None
        call.completed = True
        call.raw_x0 = exc.x0
        return exc.x0


def _outer_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    guider = executor.class_obj
    options = getattr(guider, "model_options", None)
    state = (options or {}).get(STATE_KEY)
    if not isinstance(state, _State):
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    record = _diag._ACTIVE.get()
    if record is None or not record.state.strict_provenance or not record.state.manifest.get("gate_complete"):
        raise RuntimeError("production candidate requires strict, complete execution-contract diagnostics")
    replay = state.replay
    _replay._assert_checkpoint_identity_unchanged(replay.exact_checkpoint_identity)
    if not isinstance(options, dict):
        raise RuntimeError("production candidate requires mutable guider model options")
    config = options.get(_runtime.PROGRESSIVE_KEY)
    if not _diag._eligible(config):
        raise RuntimeError("production candidate requires the captured progressive Target Input config")
    current_sampler = _runtime.sampler_name(sampler)
    if str(replay.manifest.get("sampler")) != current_sampler or current_sampler != "sample_euler":
        raise RuntimeError("production candidate is bounded to the captured Euler sampler")
    if int(seed or 0) != int(replay.manifest["seed"]):
        raise RuntimeError("production candidate seed differs from R capture")
    if denoise_mask is not None or "high_denoise_mask" in replay.tensors:
        raise RuntimeError("production candidate does not support protected/masked replay")
    if not isinstance(latent_shapes, list):
        raise RuntimeError("production candidate requires mutable target latent-shape metadata")
    target_shapes = [tuple(int(dim) for dim in shape) for shape in replay.manifest["target_shapes"]]
    if [tuple(int(dim) for dim in shape) for shape in latent_shapes] != target_shapes:
        raise RuntimeError("production candidate target geometry differs from R capture")
    if _runtime._schedule_signature(sigmas) != str(replay.manifest["original_schedule_digest"]):
        raise RuntimeError("production candidate caller schedule differs from R capture")
    if record.pristine_cond_digest != str(replay.manifest["pristine_conditioning_digest"]):
        raise RuntimeError("production candidate pristine target conditioning differs from R capture")

    provenance_gate = _provenance_gate(state, record, guider)
    if not provenance_gate["exact_except_reviewed_production_delta"]:
        raise RuntimeError(
            "production candidate runtime provenance differs beyond reviewed Sol/VDN/Flow delta: "
            + ", ".join(provenance_gate["unexpected_differences"][:12])
        )
    sampling_runtime = _sampling_runtime_identity(guider)
    if not _sampling_runtime_ok(sampling_runtime):
        raise RuntimeError("production candidate runtime is not the expected MiniMax-H3 AV CONST contract")

    binding = _runtime._resolve_binding(guider)
    if binding is None:
        raise RuntimeError("production candidate requires active Flow binding")
    if binding.active_capture is not None or binding.active_guidance_run is not None:
        raise RuntimeError("production candidate entered with stale Flow capture/guidance state")
    if not _replay._dict_equal(_replay._guidance_config_dict(binding), replay.manifest.get("guidance_config")):
        raise RuntimeError("production candidate Flow guidance config differs from R capture")
    if not _replay._dict_equal(_replay._spectrum_config_dict(guider), replay.manifest.get("spectrum_config")):
        raise RuntimeError("production candidate Spectrum config differs from R capture")

    trajectory, run_id = _replay._rebuild_trajectory(replay.manifest, replay.tensors)
    current_guidance = _replay._guidance_config_dict(binding)
    if current_guidance is not None and current_guidance.get("mode") != "off" and trajectory is None:
        raise RuntimeError("production candidate guidance is active but R bundle has no trajectory")
    handoff_index = int(replay.manifest["handoff_index"])
    high_sigmas = sigmas[handoff_index:]
    captured_high = [float(value) for value in replay.manifest["high_sigmas"]]
    live_high = [float(value) for value in high_sigmas.detach().to(device="cpu", dtype=torch.float64).tolist()]
    if captured_high != _REQUIRED_SUFFIX or live_high != captured_high:
        raise RuntimeError("production candidate must preserve the complete captured high suffix")

    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("production candidate requires mutable transformer options")
    if RECEIPTS_KEY in transformer:
        raise RuntimeError("production candidate backend receipt key is already owned")
    if any(str(key).startswith("h3_first_high_") for key in transformer):
        raise RuntimeError("production candidate cannot execute with W/E/M transformer instrumentation")
    if "h3_flow_untwist_clock_trial_v1" in transformer or "h3_flow_sampling_context" in transformer:
        raise RuntimeError("production candidate requires the established no-Untwist R control")
    sol_config = transformer.get("sol_h3_runtime_v1")
    if not isinstance(sol_config, dict) or sol_config.get("backend") != "sol":
        raise RuntimeError("production candidate requires active Sol-H3 backend")
    if _ACTIVE_CALL.get() is not None:
        raise RuntimeError("nested production candidate execution is unsupported")

    replay_latent = _replay._tensor_for_replay(replay, "high_latent_image")
    replay_noise = _replay._tensor_for_replay(replay, "high_noise_argument")
    receipt_sink = _ReceiptSink()
    metric_start = binding.metrics.counters
    event_start = len(binding.metrics.events)
    previous_progressive = options.pop(_runtime.PROGRESSIVE_KEY)
    previous_trajectory = binding.trajectory
    previous_guidance_run_id = binding.guidance_run_id
    previous_capture_enabled = binding.capture_enabled
    binding.trajectory = trajectory
    binding.guidance_run_id = run_id
    binding.capture_enabled = False
    transformer[RECEIPTS_KEY] = receipt_sink
    call = _Call(state=state)
    call_token = _ACTIVE_CALL.set(call)
    completed = False
    diagnostic_output = None
    export_contract = None
    cleanup_contract = None
    replay_error: BaseException | None = None

    def stop_after_first(step, x0, x, _total):
        _ = x
        if int(step) != 0:
            raise RuntimeError("production candidate callback reached a second Euler step")
        raise _FirstCallComplete(call.token, x0)

    try:
        if record.pristine_conds is None:
            raise RuntimeError("production candidate record is missing pristine target conditioning")
        _runtime._reset_guider_conds(guider, template=record.pristine_conds)
        with _runtime._flow_stage_contract(guider, "high"), _runtime._high_stage_contract(guider):
            diagnostic_output = executor(
                replay_noise,
                replay_latent,
                sampler,
                high_sigmas,
                None,
                stop_after_first,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        if not call.completed or call.raw_x0 is None:
            raise RuntimeError("production candidate callback sentinel did not terminate after first model call")
        export_contract = _export_contract(guider, call.raw_x0, diagnostic_output)
        cleanup_contract = _core_cleanup_contract(guider, options)
        if export_contract["exact_core_callback_x0_externalization"] is not True:
            raise RuntimeError("production candidate callback x0 bypassed Core externalization")
        if cleanup_contract["complete"] is not True:
            raise RuntimeError("production candidate bounded return bypassed Core outer-sample cleanup")
        completed = True
        return diagnostic_output
    except BaseException as exc:
        replay_error = exc
        raise
    finally:
        _ACTIVE_CALL.reset(call_token)
        transformer.pop(RECEIPTS_KEY, None)
        binding.trajectory = previous_trajectory
        binding.guidance_run_id = previous_guidance_run_id
        binding.capture_enabled = previous_capture_enabled
        options[_runtime.PROGRESSIVE_KEY] = previous_progressive

        if completed and replay_error is None and diagnostic_output is not None and call.raw_x0 is not None:
            metric_delta = _diag._counter_delta(metric_start, binding.metrics.counters)
            events = binding.metrics.events[event_start:]
            topology = {
                "logical": int(metric_delta.get("sampler_logical_calls", 0)),
                "actual": int(metric_delta.get("transformer_actual_nfe", 0)),
                "forecast": int(metric_delta.get("spectrum_forecast_calls", 0)),
                "high_logical": int(metric_delta.get("sampler_logical_calls_high", 0)),
                "high_actual": int(metric_delta.get("transformer_actual_nfe_high", 0)),
                "high_forecast": int(metric_delta.get("spectrum_forecast_calls_high", 0)),
                "upscaler_calls": sum(1 for event in events if event.kind == "handoff_learned_upscale_wall"),
            }
            expected_topology = {
                "logical": 1,
                "actual": 1,
                "forecast": 0,
                "high_logical": 1,
                "high_actual": 1,
                "high_forecast": 0,
                "upscaler_calls": 0,
            }
            snapshots = _snapshot_report(record)
            entry_exact = all(value["exact"] is True for value in snapshots.values())
            receipts = _validate_backend_receipts(receipt_sink)
            target_video_shape = tuple(int(dim) for dim in target_shapes[0])
            raw_media = _decode_ready_snapshot(record, "first_high_model_raw_video", target_video_shape)
            pre_media = _decode_ready_snapshot(record, "first_high_pre_guidance_video", target_video_shape)
            final_media = _decode_final_video(diagnostic_output, target_shapes)
            execution_valid = bool(
                topology == expected_topology
                and entry_exact
                and receipts["valid"]
                and provenance_gate["exact_except_reviewed_production_delta"]
                and provenance_gate["operator_substitution_absent"]
                and _sampling_runtime_ok(sampling_runtime)
                and export_contract is not None
                and export_contract["exact_core_callback_x0_externalization"] is True
                and cleanup_contract is not None
                and cleanup_contract["complete"] is True
            )
            report = {
                "schema_version": SCHEMA_VERSION,
                "kind": "minimax_h3_production_mapped_neighbor_candidate_report",
                "design_commit": DESIGN_COMMIT,
                "capture_id": replay.manifest["capture_id"],
                "mode": MODE,
                "completed_first_call_only": True,
                "production_path": True,
                "operator_substitution_absent": provenance_gate["operator_substitution_absent"],
                "final_trajectory_available": False,
                "execution_valid": execution_valid,
                "media_clean": None,
                "media_assessment_required": True,
                "topology": topology,
                "topology_exact": topology == expected_topology,
                "source_gate": state.source_gate,
                "provenance_gate": provenance_gate,
                "sampling_runtime": sampling_runtime,
                "sampling_runtime_ok": _sampling_runtime_ok(sampling_runtime),
                "snapshot_hashes": snapshots,
                "entry_state_exact": entry_exact,
                "backend_receipts": receipts,
                "no_hidden_native_local_fallback": receipts["no_native_local_fallback"],
                "export_contract": export_contract,
                "core_cleanup": cleanup_contract,
                "decode_ready_media_available": True,
                "media_contract": {
                    "first_high_model_raw_shape": [int(dim) for dim in raw_media.shape],
                    "first_high_model_raw_sha256": _replay._tensor_sha256(raw_media),
                    "first_high_pre_guidance_shape": [int(dim) for dim in pre_media.shape],
                    "first_high_pre_guidance_sha256": _replay._tensor_sha256(pre_media),
                    "first_high_final_shape": [int(dim) for dim in final_media.shape],
                    "first_high_final_sha256": _replay._tensor_sha256(final_media),
                },
                "diagnostic_x0_sha256": _replay._tensor_sha256(call.raw_x0),
                "diagnostic_output_sha256": _replay._tensor_sha256(diagnostic_output),
                "decision_gate": (
                    "invalid-production-candidate: fix the failed structural/runtime gate before any full trajectory"
                    if not execution_valid
                    else "valid-production-candidate-structure: inspect all three decoded media before full trajectory"
                ),
            }
            state.complete.append(
                {
                    **report,
                    "_media_raw": raw_media,
                    "_media_pre": pre_media,
                    "_media_final": final_media,
                }
            )


def patch_production_mapped_neighbor_candidate(model: Any, manifest_path: str) -> tuple[Any, _State]:
    diagnostic = (getattr(model, "model_options", None) or {}).get(_diag.DIAGNOSTIC_KEY)
    if not isinstance(diagnostic, _diag._State):
        raise RuntimeError("production candidate must be applied after MiniMax H3 Execution Contract Diagnostics")
    if not diagnostic.strict_provenance or not diagnostic.manifest.get("gate_complete"):
        raise RuntimeError("production candidate requires strict, complete installed-runtime provenance")
    resolved, manifest, tensors = _replay._load_bundle(manifest_path)
    if str(manifest.get("capture_id", "")) != EXPECTED_CAPTURE_ID:
        raise RuntimeError(f"production candidate requires preserved R capture {EXPECTED_CAPTURE_ID}")
    exact_checkpoint = _replay._checkpoint_identity(diagnostic.manifest)
    if not _replay._dict_equal(exact_checkpoint, manifest.get("exact_checkpoint_identity")):
        raise RuntimeError("production candidate exact checkpoint identity differs from R capture")
    source_gate = _verify_source_manifest()
    replay_state = _replay._ReplayState(
        manifest_path=str(resolved),
        manifest=manifest,
        tensors=tensors,
        exact_checkpoint_identity=exact_checkpoint,
    )
    patched = model.clone()
    _comfy_compat._copy_model_options(patched)
    transformer = patched.model_options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("production candidate requires mutable transformer options")
    if "h3_flow_untwist_clock_trial_v1" in transformer or "h3_flow_sampling_context" in transformer:
        raise RuntimeError("production candidate requires the established no-Untwist R control")
    if (
        STATE_KEY in patched.model_options
        or _replay.CAPTURE_STATE_KEY in patched.model_options
        or _replay.REPLAY_STATE_KEY in patched.model_options
    ):
        raise RuntimeError("production candidate cannot be combined with R capture/replay wrappers")
    if any(str(key).startswith("h3_flow_first_high_") for key in patched.model_options):
        raise RuntimeError("production candidate cannot be combined with W/E/M first-high wrappers")

    state = _State(replay=replay_state, source_gate=source_gate)
    patched.model_options[STATE_KEY] = state

    import comfy.patcher_extension

    _diag._insert_relative(
        patched,
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        _runtime.OUTER_WRAPPER_KEY,
        _OUTER_KEY,
        _outer_wrapper,
        before=True,
    )
    _diag._insert_relative(
        patched,
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        _diag._SAMPLER_KEY,
        _SAMPLER_KEY,
        _sampler_entry_wrapper,
        before=False,
    )
    return patched, state


def _saved_bundles() -> list[str]:
    return [name for name in _replay._saved_replay_manifests() if name.endswith(".json")]


class H3ProductionMappedNeighborCandidate:
    CATEGORY = "MiniMax H3/flow regenerate/validation"
    DESCRIPTION = (
        "Replay the preserved R first-high state for exactly one production model call using ordinary VDN provider-v4 "
        "and packaged mapped-neighbor Sol-H3. No W/E/M operator substitution is installed."
    )
    RETURN_TYPES = ("MODEL", "H3_FLOW_PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE")
    RETURN_NAMES = ("model", "candidate")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        bundles = _saved_bundles() or ["<no saved R bundle>"]
        return {"required": {"model": ("MODEL",), "bundle": (bundles,)}}

    def apply(self, model, bundle):
        if str(bundle) == "<no saved R bundle>":
            raise RuntimeError("no saved R bundle exists under output/h3_flow_replay")
        path = _replay._resolve_replay_bundle_selector(str(bundle))
        return patch_production_mapped_neighbor_candidate(model, str(path))


class H3ProductionMappedNeighborCandidateReport:
    CATEGORY = "MiniMax H3/flow regenerate/validation"
    DESCRIPTION = (
        "Emit the bounded production-v4 first-high report and decode-ready raw, pre-guidance, and final video latents. "
        "media_clean stays unset until all three decoded clips are manually inspected."
    )
    RETURN_TYPES = ("STRING", "LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("report", "first_high_model_raw", "first_high_pre_guidance", "first_high_final")
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "candidate": ("H3_FLOW_PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE",),
                "trigger": ("LATENT",),
            }
        }

    def extract(self, candidate, trigger):
        _ = trigger
        if not isinstance(candidate, _State):
            raise TypeError("invalid production mapped-neighbor candidate handle")
        if not candidate.complete:
            raise RuntimeError("no completed production mapped-neighbor candidate report is available")
        entry = candidate.complete.pop()
        raw = entry.get("_media_raw")
        pre = entry.get("_media_pre")
        final = entry.get("_media_final")
        if not all(torch.is_tensor(value) for value in (raw, pre, final)):
            raise RuntimeError("completed production candidate report is missing decode-ready media")
        public = {key: value for key, value in entry.items() if key not in _PRIVATE_REPORT_KEYS}
        return (
            json.dumps(public, indent=2, sort_keys=True, default=str),
            {"samples": raw},
            {"samples": pre},
            {"samples": final},
        )


NODE_CLASS_MAPPINGS = {
    "H3ProductionMappedNeighborCandidate": H3ProductionMappedNeighborCandidate,
    "H3ProductionMappedNeighborCandidateReport": H3ProductionMappedNeighborCandidateReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ProductionMappedNeighborCandidate": "MiniMax H3 Production Mapped-Neighbor Candidate",
    "H3ProductionMappedNeighborCandidateReport": "MiniMax H3 Production Mapped-Neighbor Candidate Report",
}


__all__ = [
    "DESIGN_COMMIT",
    "EXPECTED_CAPTURE_ID",
    "MODE",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SCHEMA_VERSION",
    "_validate_backend_receipts",
    "patch_production_mapped_neighbor_candidate",
]
