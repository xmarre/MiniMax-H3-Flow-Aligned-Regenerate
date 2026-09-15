"""Bounded first-high Sol-local root-cause diagnostic E.

E reuses the preserved experiment-R first-high bundle and executes one real Euler
model call. VDN keeps its released restricted local support and linear complement.
Only the final Sol local kernel selection is changed to all-selected. Three
same-input operator witnesses are sidecars of block 2 groups 0, 2 and 10; they add
no H3 calls and are excluded from production backend receipts/counters.

This module is diagnostic-only. It never changes production progressive policy and
never promotes a production fix from its own result.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import math
import os
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import comfy_compat as _comfy_compat
from . import execution_contract_diagnostics as _diag
from . import first_high_operator_comparison as _w
from . import runtime as _runtime
from . import same_state_high_replay as _replay
from .execution_contract_provenance import callable_identity as _callable_identity

SCHEMA_VERSION = 1
DESIGN_COMMIT = "0b8715faa0a82c730f5aaf0e44b1185e64291e49"
STATE_KEY = "h3_flow_first_high_sol_local_diagnostic_v1"
REQUEST_KEY = "h3_first_high_sol_local_diagnostic_v1"
RECEIPTS_KEY = "h3_first_high_sol_local_receipts_v1"
EVIDENCE_KEY = "h3_first_high_sol_local_evidence_v1"
MODE = "all_selected_e"
_OUTER_KEY = "h3_flow_regenerate.first_high_sol_local.execute.v1"
_SAMPLER_KEY = "h3_flow_regenerate.first_high_sol_local.sampler_entry.v1"
_SOURCE_MANIFEST = "first_high_sol_local_e_source_delta.json"
_REQUIRED_SUFFIX = [0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]
_MAX_REPORTS = 4
_EXPECTED_PACKED_ROWS = 56349
_EXPECTED_VIDEO_SPAN = (3101, 56349)
_EXPECTED_BLOCKS = 50
_EXPECTED_LOCAL_Q_ROWS = (4096,) + (5120,) * 9 + (1024,)
_EXPECTED_WINDOW_KV_ROWS = (14365, 19485) + (20509,) * 7 + (16413, 11293)
_EXPECTED_WITNESS_GROUPS = (0, 2, 10)
_EXPECTED_CAPTURE_ID = "234ed062128e43ed8d5ec63e27517b22"
_EXPECTED_ENTRY_HASHES = {
    "first_high_sampler_input_video": "79dca62849b2a159061a1d6204828af7a4c5995c41beec256c1719a5932a41ee",
    "first_high_sampler_input_audio": "f5fd588101bf2ab68eb5aeeb49bfaf8e58f4c71aa99864de89cb1ddef8c859f3",
    "first_high_h3_input_video": "b49f17a317530795be43bc775486777b07d5b5dbb28996819033cdede64195c0",
    "first_high_h3_input_audio": "f54b22ff25a675c47a4c32aba54b5642033b048441ad6eec9bf5ef935cd29002",
}
_EXPECTED_BLOCK0_QKV = "38cc6dc9bec83456a41e8ea3183b242442e8c9a8fe368185131e6f5666672936"
_EXPECTED_GATE_FINGERPRINT = "55afe755b441274502061cbd29de1febe9e469d6047fb391a40fc0cad1b9f082"
_EXPECTED_ADAPTER_DIGEST = "409e9cd7cd1657493d6cd5d7fadf88e79aab726b50b508f8ab727bf562b72d61"
_OWNER_COMPANION_GROUP = {"flow": "flow", "sol": "sol_h3", "vdn": "vdn_h3"}
_MEDIA_RAW_KEY = "_first_high_model_raw_video"
_MEDIA_PRE_KEY = "_first_high_pre_guidance_video"
_PRIVATE_EVIDENCE_KEYS = frozenset({"_transformer_options"})
_DEFAULT_DIAGNOSTIC_BUDGET_BYTES = 2 * 1024**3
_CUDA_BUDGET_ENV = "H3_FIRST_HIGH_E_CUDA_BUDGET_GIB"
_CPU_BUDGET_ENV = "H3_FIRST_HIGH_E_CPU_BUDGET_GIB"
_ARITH_MEAN_ABS_LIMIT = 0.002
_ARITH_REL_L2_LIMIT = 0.005
_ARITH_CATASTROPHIC_MAX_FLOOR = 0.5
_ARITH_CATASTROPHIC_REFERENCE_PEAK_MULTIPLIER = 4.0
_REQUIRED_SOURCE_ENTRY_KEYS = frozenset(
    {
        ("flow", "h3_flow_regenerate.first_high_sol_local_diagnostic", "."),
        ("flow", "h3_flow_regenerate.first_high_sol_local_diagnostic", "../__init__.py"),
        ("flow", "h3_flow_regenerate.first_high_operator_comparison", "."),
        ("sol", "sol_h3.first_high_sol_local_diagnostic", "."),
        ("sol", "sol_h3.first_high_sol_local_witness_bridge", "."),
        ("sol", "sol_h3.first_high_sol_local_receipt_tap", "."),
        ("sol", "sol_h3.first_high_operator_diagnostic", "."),
        ("sol", "sol_h3", "."),
        ("vdn", "vdn_h3.first_high_sol_local_diagnostic", "."),
        ("vdn", "vdn_h3.first_high_sol_local_bridge", "."),
        ("vdn", "vdn_h3.first_high_operator_diagnostic", "."),
        ("vdn", "vdn_h3.first_high_operator_sol_bridge", "."),
        ("vdn", "vdn_h3", "../__init__.py"),
        ("core", "comfy.model_sampling", "."),
        ("core", "comfy.latent_formats", "."),
        ("core", "comfy.model_patcher", "."),
        ("core", "comfy.k_diffusion.sampling", "."),
    }
)


class _FirstCallComplete(BaseException):
    def __init__(self, token: object, x0: torch.Tensor):
        super().__init__("first-high Sol-local E completed its single model call")
        self.token = token
        self.x0 = x0


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


@dataclass(slots=True)
class _Sink:
    """Clone-stable bounded owner across Core model-option copies."""

    items: list[Any] = field(default_factory=list)
    limit: int = 4096

    def append(self, value: Any) -> None:
        if len(self.items) >= self.limit:
            raise RuntimeError("first-high Sol-local E diagnostic sink bound exceeded")
        self.items.append(value)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


_ACTIVE_CALL: contextvars.ContextVar[_Call | None] = contextvars.ContextVar(
    "h3_flow_first_high_sol_local_call", default=None
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
    return isinstance(value, str) and len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _source_entry_key(value: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("owner", "")),
        str(value.get("module", "")),
        str(value.get("relative_path", ".")),
    )


def _load_source_manifest() -> tuple[dict[str, Any], str]:
    path = Path(__file__).with_name(_SOURCE_MANIFEST)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("first-high Sol-local E source-delta manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("first-high Sol-local E source-delta manifest schema is unsupported")
    if manifest.get("design_commit") != DESIGN_COMMIT:
        raise RuntimeError("first-high Sol-local E source-delta manifest targets the wrong design commit")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("first-high Sol-local E source-delta manifest has no reviewed entries")
    keys = [_source_entry_key(item) for item in entries if isinstance(item, dict)]
    if len(keys) != len(entries) or len(keys) != len(set(keys)):
        raise RuntimeError("first-high Sol-local E source-delta manifest has invalid or duplicate entries")
    if frozenset(keys) != _REQUIRED_SOURCE_ENTRY_KEYS:
        raise RuntimeError("first-high Sol-local E source-delta manifest differs from the reviewed exact entry set")
    return manifest, _sha_json(manifest)


def _resolve_source_entry(entry: dict[str, Any]) -> Path:
    # PR #43 already installs the namespace-aware resolver used by W. Reuse that
    # exact proven loader contract rather than assuming bare custom-node packages.
    return _w._resolve_source_entry(entry)


def _verify_source_manifest() -> dict[str, Any]:
    manifest, digest = _load_source_manifest()
    observed = []
    for raw in manifest["entries"]:
        path = _resolve_source_entry(raw)
        expected_blob = raw.get("candidate_git_blob_sha")
        base_blob = raw.get("base_git_blob_sha")
        base_sha256 = raw.get("base_sha256")
        if not _valid_git_blob_sha(expected_blob):
            raise RuntimeError("first-high Sol-local E source entry lacks candidate Git blob identity")
        if base_blob is not None and not _valid_git_blob_sha(base_blob):
            raise RuntimeError("first-high Sol-local E source entry has invalid base Git blob identity")
        changed_existing = base_blob is not None and base_blob != expected_blob
        if changed_existing and not _valid_sha256(base_sha256):
            raise RuntimeError("first-high Sol-local E changed source entry lacks exact R base SHA-256")
        if not changed_existing and base_sha256 is not None:
            raise RuntimeError("first-high Sol-local E source entry has unexpected R base SHA-256")
        actual_blob = _git_blob_sha(path)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"first-high Sol-local E source bytes differ from reviewed candidate: {path}: "
                f"{actual_blob} != {expected_blob}"
            )
        observed.append(
            {
                **raw,
                "path": str(path),
                "git_blob_sha": actual_blob,
                "sha256": _sha256_file(path),
            }
        )
    return {"manifest_digest": digest, "entries": observed}


def _source_path(source_gate: dict[str, Any], owner: str, module: str, relative: str = ".") -> str:
    matches = [
        str(entry.get("path", ""))
        for entry in source_gate.get("entries", [])
        if entry.get("owner") == owner and entry.get("module") == module and entry.get("relative_path", ".") == relative
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(f"first-high Sol-local E source gate does not uniquely identify {owner}:{module}:{relative}")
    return matches[0]


def _normalize_vdn_object_patches(
    current: dict[str, Any],
    capture: dict[str, Any],
    guider: Any,
    source_gate: dict[str, Any],
) -> dict[str, Any]:
    current_patches = current.get("active_object_patches")
    capture_patches = capture.get("active_object_patches")
    patcher = getattr(guider, "model_patcher", None)
    runtime_patches = getattr(patcher, "object_patches", None)
    if not isinstance(current_patches, dict) or not isinstance(capture_patches, dict):
        raise RuntimeError("first-high Sol-local E provenance lacks active VDN object-patch manifests")
    if not isinstance(runtime_patches, dict):
        raise RuntimeError("first-high Sol-local E cannot inspect live model object patches")
    expected_keys = {f"diffusion_model.blocks.{index}.attn.forward" for index in range(_EXPECTED_BLOCKS)}
    wrapped = {
        str(key): value
        for key, value in runtime_patches.items()
        if getattr(value, "_h3_first_high_sol_local_diagnostic_v1", False) is True
    }
    if set(wrapped) != expected_keys:
        raise RuntimeError(
            "first-high Sol-local E VDN diagnostic object-patch set is not exactly 50 H3 attention patches"
        )
    e_source = _source_path(source_gate, "vdn", "vdn_h3.first_high_sol_local_diagnostic")
    w_source = _source_path(source_gate, "vdn", "vdn_h3.first_high_operator_diagnostic")
    rebuilt = dict(current_patches)
    for key in sorted(expected_keys):
        e_wrapper = wrapped[key]
        e_identity = _callable_identity(e_wrapper)
        e_file = (e_identity.get("file") or {}).get("resolved_path") if isinstance(e_identity, dict) else None
        if e_file != e_source:
            raise RuntimeError(f"first-high Sol-local E VDN wrapper source changed for {key}: {e_file!r}")
        w_wrapper = getattr(e_wrapper, "_h3_first_high_sol_local_original_forward", None)
        if not callable(w_wrapper) or getattr(w_wrapper, "_h3_first_high_operator_diagnostic_v1", False) is not True:
            raise RuntimeError(f"first-high Sol-local E VDN wrapper lost underlying W evidence wrapper for {key}")
        w_identity = _callable_identity(w_wrapper)
        w_file = (w_identity.get("file") or {}).get("resolved_path") if isinstance(w_identity, dict) else None
        if w_file != w_source:
            raise RuntimeError(f"first-high Sol-local E underlying W wrapper source changed for {key}: {w_file!r}")
        production = getattr(w_wrapper, "_h3_first_high_operator_original_forward", None)
        if not callable(production):
            raise RuntimeError(f"first-high Sol-local E VDN chain lacks underlying production forward for {key}")
        if key not in current_patches or key not in capture_patches:
            raise RuntimeError(f"first-high Sol-local E provenance is missing VDN identity for {key}")
        production_identity = _w._callable_equivalence_identity(production)
        if _canonical_json(production_identity) != _canonical_json(capture_patches[key]):
            detail = _replay._provenance_diff_paths(capture_patches[key], production_identity, limit=8)
            raise RuntimeError(
                f"first-high Sol-local E underlying VDN production forward differs from R for {key}"
                + ("; " + ", ".join(detail) if detail else "")
            )
        rebuilt[key] = capture_patches[key]
    result = dict(current)
    result["active_object_patches"] = rebuilt
    return result


def _wrapper_specs():
    import comfy.patcher_extension

    return (
        (comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _OUTER_KEY, _outer_wrapper),
        (comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, _SAMPLER_KEY, _sampler_entry_wrapper),
    )


def _normalize_wrapper_order(current: dict[str, Any], guider: Any) -> dict[str, Any]:
    patcher = getattr(guider, "model_patcher", None)
    runtime_wrappers = getattr(patcher, "wrappers", None)
    if not isinstance(runtime_wrappers, dict):
        raise RuntimeError("first-high Sol-local E cannot inspect live ModelPatcher wrapper ownership")
    specs = _wrapper_specs()
    for wrapper_type, key, expected_callable in specs:
        locations = [
            existing_type
            for existing_type, keyed in runtime_wrappers.items()
            if isinstance(keyed, dict) and key in keyed
        ]
        if len(locations) != 1 or locations[0] != wrapper_type:
            raise RuntimeError(f"first-high Sol-local E live wrapper key ownership changed for {key}: {locations!r}")
        keyed = runtime_wrappers.get(wrapper_type)
        values = keyed.get(key) if isinstance(keyed, dict) else None
        if not isinstance(values, (list, tuple)) or len(values) != 1 or values[0] is not expected_callable:
            raise RuntimeError(f"first-high Sol-local E live wrapper callable identity changed for {key}")

    result = dict(current)
    for field_name in ("active_wrapper_order", "patcher_wrapper_order"):
        manifest = result.get(field_name)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"first-high Sol-local E provenance lacks {field_name}")
        rebuilt = dict(manifest)
        for wrapper_type, key, expected_callable in specs:
            manifest_key = str(wrapper_type)
            entries = rebuilt.get(manifest_key)
            if not isinstance(entries, list):
                raise RuntimeError(f"first-high Sol-local E provenance lacks wrapper list {field_name}.{manifest_key}")
            matches = [
                (index, item) for index, item in enumerate(entries) if isinstance(item, dict) and item.get("key") == key
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"first-high Sol-local E provenance must contain exactly one {field_name} entry for {key}"
                )
            index, item = matches[0]
            expected_identity = _w._callable_equivalence_identity(expected_callable)
            if _canonical_json(item.get("callable")) != _canonical_json(expected_identity):
                raise RuntimeError(f"first-high Sol-local E provenance callable changed for {field_name}:{key}")
            rebuilt[manifest_key] = [entry for ordinal, entry in enumerate(entries) if ordinal != index]
        result[field_name] = rebuilt
    return result


def _verify_capture_base_sources(capture: dict[str, Any], source_gate: dict[str, Any]) -> dict[str, Any]:
    groups = capture.get("loaded_companion_sources")
    if not isinstance(groups, dict):
        raise RuntimeError("first-high Sol-local E R provenance lacks companion source inventory")
    checked = []
    for entry in source_gate.get("entries", []):
        base_blob = entry.get("base_git_blob_sha")
        candidate_blob = entry.get("candidate_git_blob_sha")
        if base_blob is None or base_blob == candidate_blob:
            continue
        owner = str(entry.get("owner", ""))
        group_name = _OWNER_COMPANION_GROUP.get(owner)
        if group_name is None:
            raise RuntimeError(f"first-high Sol-local E changed source has no R companion owner mapping: {owner!r}")
        expected_sha256 = entry.get("base_sha256")
        if not _valid_sha256(expected_sha256):
            raise RuntimeError("first-high Sol-local E changed source lacks reviewed R base SHA-256")
        capture_sources, problems = _replay._companion_source_map(groups.get(group_name))
        if problems:
            raise RuntimeError(
                f"first-high Sol-local E R companion source inventory is invalid for {group_name}: "
                + "; ".join(problems)
            )
        source_path = str(entry.get("path", ""))
        captured = capture_sources.get(source_path)
        if captured is None:
            raise RuntimeError(f"first-high Sol-local E R provenance is missing reviewed base source: {source_path}")
        if captured.get("sha256") != expected_sha256:
            raise RuntimeError(f"first-high Sol-local E R base source differs from reviewed bytes: {source_path}")
        checked.append({"owner": owner, "path": source_path, "sha256": expected_sha256, "exact": True})
    return {"exact": True, "checked": checked}


def _allowed_provenance_difference(path: str, source_gate: dict[str, Any]) -> bool:
    if not path.startswith("$.loaded_companion_sources."):
        return False
    for entry in source_gate.get("entries", []):
        base_blob = entry.get("base_git_blob_sha")
        candidate_blob = entry.get("candidate_git_blob_sha")
        if base_blob == candidate_blob:
            continue
        source_path = str(entry.get("path", ""))
        if source_path and f"[{source_path}]" in path:
            return True
    return False


def _provenance_gate(state: _State, record: _diag._Record, guider: Any) -> dict[str, Any]:
    capture = _replay._bundle_provenance_equivalence_identity(state.replay.manifest)
    current = _replay._provenance_equivalence_identity(record.state.manifest)
    capture_base_sources = _verify_capture_base_sources(capture, state.source_gate)
    current = _normalize_vdn_object_patches(current, capture, guider, state.source_gate)
    current = _normalize_wrapper_order(current, guider)
    differences = _replay._cross_process_provenance_diff_paths(capture, current, limit=128)
    unexpected = [path for path in differences if not _allowed_provenance_difference(path, state.source_gate)]
    return {
        "differences": differences,
        "unexpected_differences": unexpected,
        "capture_base_sources": capture_base_sources,
        "exact_except_reviewed_e_delta": capture_base_sources["exact"] and not unexpected,
    }


def _target_shapes_digest(manifest: dict[str, Any]) -> str:
    return _sha_json([[int(dim) for dim in shape] for shape in manifest["target_shapes"]])


def _request_tuple(state: _State) -> tuple[tuple[str, Any], ...]:
    manifest = state.replay.manifest
    high = [float(value) for value in manifest.get("high_sigmas", [])]
    if high != _REQUIRED_SUFFIX:
        raise RuntimeError(f"first-high Sol-local E requires the captured high suffix; got {high}")
    capture_id = str(manifest.get("capture_id", ""))
    if capture_id != _EXPECTED_CAPTURE_ID:
        raise RuntimeError(
            f"first-high Sol-local E requires preserved R capture {_EXPECTED_CAPTURE_ID}, got {capture_id}"
        )
    return (
        ("api", 1),
        ("capture_id", capture_id),
        ("mode", MODE),
        ("stage", "high"),
        ("logical_call_limit", 1),
        ("sigma", float(high[0])),
        ("target_shapes_digest", _target_shapes_digest(manifest)),
        ("source_contract_digest", str(state.source_gate["manifest_digest"])),
    )


def _budget_bytes(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return _DEFAULT_DIAGNOSTIC_BUDGET_BYTES
    try:
        gib = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"first-high Sol-local E {name} must be a finite GiB value") from exc
    if not math.isfinite(gib) or not 0.25 <= gib <= 64.0:
        raise RuntimeError(f"first-high Sol-local E {name} must be in [0.25, 64] GiB")
    return int(gib * 1024**3)


def _planned_witness_cpu_bytes() -> int:
    heads = 56
    dim = 128
    total = 0
    for group in _EXPECTED_WITNESS_GROUPS:
        q_rows = _EXPECTED_LOCAL_Q_ROWS[group]
        kv_rows = _EXPECTED_WINDOW_KV_ROWS[group]
        q_blocks = (q_rows + 63) // 64
        k_blocks = (kv_rows + 63) // 64
        route_groups = (k_blocks + 63) // 64
        qkv = (q_rows + 2 * kv_rows) * heads * dim * 2
        saved_outputs = 3 * q_rows * heads * dim * 2
        summaries = 2 * k_blocks * heads * dim * 2
        threshold = q_blocks * heads * 4
        traces = 2 * q_blocks * heads * route_groups * 2 * 4
        route_fields = 2 * q_blocks * heads * k_blocks * 4
        lse = q_rows * heads * 4
        total += qkv + saved_outputs + summaries + threshold + traces + route_fields + lse
    return total


def _projected_cuda_sidecar_bytes() -> int:
    heads = 56
    dim = 128
    group = max(
        _EXPECTED_WITNESS_GROUPS, key=lambda item: _EXPECTED_LOCAL_Q_ROWS[item] + 2 * _EXPECTED_WINDOW_KV_ROWS[item]
    )
    q_rows = _EXPECTED_LOCAL_Q_ROWS[group]
    kv_rows = _EXPECTED_WINDOW_KV_ROWS[group]
    logical_qkv = (q_rows + 2 * kv_rows) * heads * dim * 2
    # Conservative bound: logical live QKV domain plus six Q-sized BF16 results
    # and 256 MiB for summaries, traces, FP32 tile/reference scratch and runtime
    # workspace. QKV is caller-owned rather than cloned, so this intentionally
    # overstates E's incremental allocation.
    q_output = q_rows * heads * dim * 2
    return logical_qkv + 6 * q_output + 256 * 1024**2


def _cpu_available_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        if hasattr(os, "sysconf"):
            try:
                return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError):
                pass
    raise RuntimeError("first-high Sol-local E cannot prove host-memory headroom")


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, AttributeError, OSError):
        return None


def _process_maxrss_bytes() -> int | None:
    try:
        import resource
        import sys

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _memory_snapshot(device: torch.device) -> dict[str, Any]:
    try:
        free_cuda, total_cuda = torch.cuda.mem_get_info(device)
    except TypeError:
        with torch.cuda.device(device):
            free_cuda, total_cuda = torch.cuda.mem_get_info()
    return {
        "cuda_allocator_current_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_allocator_process_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_free_bytes": int(free_cuda),
        "cuda_total_bytes": int(total_cuda),
        "cpu_rss_bytes": _process_rss_bytes(),
        "cpu_process_peak_rss_bytes": _process_maxrss_bytes(),
    }


def _memory_preflight(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("first-high Sol-local E requires CUDA SM120")
    cuda_budget = _budget_bytes(_CUDA_BUDGET_ENV)
    cpu_budget = _budget_bytes(_CPU_BUDGET_ENV)
    planned_cpu = _planned_witness_cpu_bytes()
    projected_cuda = _projected_cuda_sidecar_bytes()
    if planned_cpu > cpu_budget:
        raise RuntimeError(
            "first-high Sol-local E projected CPU witness storage exceeds the configured diagnostic budget: "
            f"{planned_cpu} > {cpu_budget}"
        )
    if projected_cuda > cuda_budget:
        raise RuntimeError(
            "first-high Sol-local E projected CUDA sidecar bound exceeds the configured diagnostic budget: "
            f"{projected_cuda} > {cuda_budget}"
        )
    snapshot = _memory_snapshot(device)
    cpu_available = _cpu_available_bytes()
    if int(snapshot["cuda_free_bytes"]) < cuda_budget:
        raise RuntimeError(
            "first-high Sol-local E has insufficient CUDA headroom for the configured diagnostic budget: "
            f"{snapshot['cuda_free_bytes']} < {cuda_budget}"
        )
    if int(cpu_available) < cpu_budget:
        raise RuntimeError(
            "first-high Sol-local E has insufficient host headroom for the configured diagnostic budget: "
            f"{cpu_available} < {cpu_budget}"
        )
    return {
        "cuda_budget_env": _CUDA_BUDGET_ENV,
        "cpu_budget_env": _CPU_BUDGET_ENV,
        "cuda_budget_bytes": cuda_budget,
        "cpu_budget_bytes": cpu_budget,
        "projected_cuda_sidecar_upper_bound_bytes": projected_cuda,
        "planned_cpu_witness_storage_bytes": planned_cpu,
        "cpu_available_bytes": int(cpu_available),
        "before": snapshot,
    }


def _unique_cpu_tensor_storage_bytes(value: Any) -> int:
    seen: set[tuple[int, int]] = set()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                return
            storage = item.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            seen.add(key)
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return sum(size for _, size in seen)


def _memory_completion(
    device: torch.device,
    preflight: dict[str, Any],
    evidence: _Sink,
) -> dict[str, Any]:
    after = _memory_snapshot(device)
    before = preflight["before"]
    cpu_evidence = _unique_cpu_tensor_storage_bytes(evidence.items)
    cpu_peak_before = before.get("cpu_process_peak_rss_bytes")
    cpu_peak_after = after.get("cpu_process_peak_rss_bytes")
    cuda_peak_before = int(before["cuda_allocator_process_peak_bytes"])
    cuda_peak_after = int(after["cuda_allocator_process_peak_bytes"])
    return {
        **preflight,
        "after": after,
        "actual_retained_cpu_evidence_storage_bytes": cpu_evidence,
        "retained_cpu_evidence_within_budget": cpu_evidence <= int(preflight["cpu_budget_bytes"]),
        "cuda_allocator_new_process_high_water_bytes": max(0, cuda_peak_after - cuda_peak_before),
        "cpu_new_process_high_water_bytes": (
            None
            if cpu_peak_before is None or cpu_peak_after is None
            else max(0, int(cpu_peak_after) - int(cpu_peak_before))
        ),
        "peak_semantics": (
            "process high-water values are read without resetting global allocator/RSS statistics; "
            "new-high-water deltas are exact only when E exceeds the pre-existing process peak"
        ),
    }


def _snapshot_report(record: _diag._Record, manifest: dict[str, Any]) -> dict[str, Any]:
    result = _w._snapshot_report(record, manifest)
    for name, expected in _EXPECTED_ENTRY_HASHES.items():
        observed = (result.get(name) or {}).get("w_sha256")
        result[name]["design_expected_sha256"] = expected
        result[name]["design_expected_equal"] = observed == expected
    return result


def _validate_vdn_receipts(receipts: _Sink) -> dict[str, Any]:
    subcalls = [item for item in receipts if isinstance(item, dict)]
    kinds = Counter(str(item.get("kind")) for item in subcalls)
    local = [item for item in subcalls if item.get("kind") == "local"]
    nonlocal_calls = [item for item in subcalls if item.get("kind") in {"global", "anchor"}]
    local_routes = Counter(str(item.get("provider_route")) for item in local)
    nonlocal_routes = Counter(str(item.get("provider_route")) for item in nonlocal_calls)
    packed_rows = {int(item.get("packed_rows", -1)) for item in subcalls}
    video_spans = {(int(item.get("video_start", -1)), int(item.get("video_end", -1))) for item in subcalls}
    gate_values = {item.get("gate_fingerprint") for item in subcalls}
    adapters: dict[str, str] = {}
    adapters_complete = True
    for block in range(_EXPECTED_BLOCKS):
        values = {item.get("adapter_fingerprint") for item in subcalls if item.get("block") == block}
        if len(values) != 1 or not _valid_sha256(next(iter(values), None)):
            adapters_complete = False
            continue
        adapters[str(block)] = next(iter(values))
    adapter_digest = _sha_json(adapters) if adapters_complete and len(adapters) == _EXPECTED_BLOCKS else None
    block0_qkv = next(
        (
            item.get("pre_attention_qkv_digest")
            for item in subcalls
            if item.get("block") == 0 and item.get("kind") == "global"
        ),
        None,
    )
    geometry = bool(
        packed_rows == {_EXPECTED_PACKED_ROWS}
        and video_spans == {_EXPECTED_VIDEO_SPAN}
        and _w._block_receipt_topology_ok(subcalls)
        and _w._block_receipt_group_geometry_ok(subcalls, "native_window")
    )
    support = all(
        item.get("support_mode") == "restricted_window"
        and item.get("complement_executed") is True
        and item.get("canonical_full_kv") is False
        and int(item.get("original_sink_rows", -1)) == _EXPECTED_VIDEO_SPAN[0]
        for item in local
    )
    nonlocal_support = all(
        int(item.get("kv_rows", -1)) == _EXPECTED_PACKED_ROWS
        and item.get("support_mode") == "full"
        and item.get("complement_executed") is False
        for item in nonlocal_calls
    )
    expected_kinds = kinds == Counter({"local": 550, "global": 50, "anchor": 100})
    expected_local_routes = local_routes == Counter({"vdn_dense_warmup": 22, "vdn_local_sol_all_selected_e": 528})
    expected_nonlocal_routes = nonlocal_routes == Counter({"vdn_global_native": 50, "vdn_anchor_native": 100})
    gate_exact = gate_values == {_EXPECTED_GATE_FINGERPRINT}
    adapter_exact = adapter_digest == _EXPECTED_ADAPTER_DIGEST
    qkv_exact = block0_qkv == _EXPECTED_BLOCK0_QKV
    return {
        "count": len(subcalls),
        "kind_counts": dict(kinds),
        "local_routes": dict(local_routes),
        "nonlocal_routes": dict(nonlocal_routes),
        "packed_rows": sorted(packed_rows),
        "video_spans": sorted(video_spans),
        "expected_700_subcalls": len(subcalls) == 700 and expected_kinds,
        "expected_local_routes": expected_local_routes,
        "expected_nonlocal_routes": expected_nonlocal_routes,
        "restricted_support_and_complement_preserved": support,
        "nonlocal_full_support_preserved": nonlocal_support,
        "geometry_exact": geometry,
        "gate_fingerprint": next(iter(gate_values)) if len(gate_values) == 1 else None,
        "gate_fingerprint_exact": gate_exact,
        "adapter_fingerprints_by_block": adapters,
        "adapter_fingerprint_digest": adapter_digest,
        "adapter_fingerprint_digest_exact": adapter_exact,
        "first_block_pre_attention_qkv_digest": block0_qkv,
        "first_block_pre_attention_qkv_digest_exact": qkv_exact,
        "valid": bool(
            len(subcalls) == 700
            and expected_kinds
            and expected_local_routes
            and expected_nonlocal_routes
            and support
            and nonlocal_support
            and geometry
            and gate_exact
            and adapter_exact
            and qkv_exact
        ),
    }


def _validate_backend_receipts(evidence: _Sink, capture_id: str) -> dict[str, Any]:
    items = [item for item in evidence if isinstance(item, dict) and item.get("kind") == "sol_backend_receipt"]
    routes = Counter(str(item.get("route")) for item in items)
    blocks = Counter(int(item.get("block_index", -1)) for item in items)
    capture_ok = bool(items and all(item.get("capture_id") == capture_id for item in items))
    per_block_ok = len(blocks) == 50 and all(blocks.get(block) == 14 for block in range(50))
    expected = Counter(
        {
            "vdn_local_sol_all_selected_e": 528,
            "vdn_dense_warmup": 22,
            "vdn_global_native": 50,
            "vdn_anchor_native": 100,
        }
    )
    return {
        "count": len(items),
        "routes": dict(routes),
        "per_block_counts": {str(key): value for key, value in sorted(blocks.items())},
        "capture_id_exact": capture_ok,
        "per_block_14_calls_exact": per_block_ok,
        "expected_700_backend_routes": len(items) == 700 and routes == expected,
        "valid": bool(len(items) == 700 and routes == expected and capture_ok and per_block_ok),
    }


def _validate_sol_counter_isolation(record: _diag._Record) -> dict[str, Any]:
    companion = _replay._high_first_companion_observation(record.state.manifest) or {}
    observation_errors = companion.get("observation_errors") if isinstance(companion, dict) else None
    sol_after = ((companion.get("after") or {}).get("sol") or {}) if isinstance(companion, dict) else {}
    zero_fields = (
        "sparse_calls",
        "external_mixed_sol_calls",
        "external_mixed_q_rows",
        "external_mixed_kernel_q_rows",
        "external_mixed_measure_calls",
        "external_mixed_measure_q_rows",
        "external_mixed_measure_kv_rows_before",
        "external_mixed_measure_kv_rows_after",
        "external_mixed_measure_removed_rows",
        "external_mixed_weighted_measure_calls",
        "external_mixed_weighted_measure_q_rows",
        "external_mixed_weighted_measure_kv_rows",
        "vdn_local_sol_calls",
        "vdn_rectangular_sol_calls",
        "vdn_requested_q_rows",
        "vdn_kernel_q_rows",
        "vdn_square_expanded_calls",
        "vdn_square_requested_rows",
        "vdn_square_kernel_rows",
    )
    zero_fields_present = bool(isinstance(sol_after, dict) and all(field in sol_after for field in zero_fields))
    zero = bool(
        zero_fields_present
        and all(type(sol_after[field]) in {int, float} and float(sol_after[field]) == 0.0 for field in zero_fields)
    )
    expected_dense_only = bool(
        type(sol_after.get("evaluations")) in {int, float}
        and int(sol_after["evaluations"]) == 1
        and type(sol_after.get("eligible_calls")) in {int, float}
        and int(sol_after["eligible_calls"]) == 22
        and type(sol_after.get("dense_calls")) in {int, float}
        and int(sol_after["dense_calls"]) == 22
    )
    valid = bool(not (observation_errors or []) and zero and expected_dense_only)
    return {
        "valid": valid,
        "observation_errors": observation_errors or [],
        "zero_fields": list(zero_fields),
        "zero_fields_present": zero_fields_present,
        "ordinary_sparse_external_square_zero": zero,
        "expected_one_evaluation_and_22_dense_locals": expected_dense_only,
        "sol_after": sol_after,
    }


def _metric_gate(metrics: dict[str, Any]) -> bool:
    if not metrics.get("finite"):
        return False
    peak = float(metrics.get("reference_peak_abs", 0.0))
    catastrophic = float(
        metrics.get(
            "catastrophic_max_abs_limit",
            max(_ARITH_CATASTROPHIC_MAX_FLOOR, _ARITH_CATASTROPHIC_REFERENCE_PEAK_MULTIPLIER * peak),
        )
    )
    return bool(
        float(metrics.get("mean_abs", math.inf)) <= _ARITH_MEAN_ABS_LIMIT
        and float(metrics.get("rel_l2", math.inf)) <= _ARITH_REL_L2_LIMIT
        and float(metrics.get("max_abs", math.inf)) <= catastrophic
    )


def _frozen_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics.get("finite")
        and float(metrics.get("mean_abs", math.inf)) <= _ARITH_MEAN_ABS_LIMIT
        and float(metrics.get("rel_l2", math.inf)) <= _ARITH_REL_L2_LIMIT
    )


def _validate_witnesses(evidence: _Sink) -> dict[str, Any]:
    witnesses = [item for item in evidence if isinstance(item, dict) and item.get("kind") == "operator_witness"]
    identity = sorted((item.get("block_index"), item.get("group_index")) for item in witnesses)
    expected_identity = [(2, group) for group in _EXPECTED_WITNESS_GROUPS]
    reports = []
    all_selected_conformant = True
    frozen_conformant = True
    prep_conformant = True
    route_conformant = True
    input_integrity = True
    debug_conformant = True
    all_selected_trace_conformant = True
    complete = identity == expected_identity
    for item in sorted(witnesses, key=lambda value: int(value.get("group_index", -1))):
        group = int(item.get("group_index", -1))
        completed = item.get("completed") is True
        complete = complete and completed
        input_ok = bool(
            item.get("input_exact_on_entry") is True
            and item.get("input_exact_after_sidecars") is True
            and item.get("preserved_qkv_sha256") == item.get("entry_qkv_sha256")
            and item.get("preserved_qkv_sha256") == item.get("exit_qkv_sha256")
        )
        input_integrity = input_integrity and input_ok
        debug_ok = item.get("debug_specializations_conform") is True
        debug_conformant = debug_conformant and debug_ok
        trace_count_ok = bool(
            item.get("all_selected_trace_complete") is True
            and type(item.get("all_selected_selected_block_pairs")) is int
            and item.get("all_selected_selected_block_pairs") == item.get("all_selected_expected_block_pairs")
        )
        all_selected_trace_conformant = all_selected_trace_conformant and trace_count_ok
        all_selected = item.get("all_selected_vs_native") or {}
        all_selected_ok = bool(item.get("all_selected_arithmetic_gate_pass") is True and _metric_gate(all_selected))
        all_selected_conformant = all_selected_conformant and all_selected_ok
        summary = item.get("summary_metrics") or {}
        summary_ok = all(
            value.get("finite") is True and float(value.get("rel_l2", math.inf)) <= 0.01
            for value in (summary.get("kc") or {}, summary.get("vc") or {}, summary.get("threshold") or {})
        )
        prep_conformant = prep_conformant and summary_ok
        route_ok = bool(
            item.get("route_trace_matches_independent") is True and int(item.get("route_mismatch_count", -1)) == 0
        )
        route_conformant = route_conformant and route_ok
        frozen = item.get("frozen_route_reference") or {}
        frozen_ok = bool(
            frozen.get("finite") is True
            and int(frozen.get("score_chunk_keys", -1)) == 1024
            and int(frozen.get("max_live_score_bytes_fp32", 2**63)) <= 64 * 1024 * 4
            and all(
                _frozen_gate(frozen.get(name) or {})
                for name in (
                    "output",
                    "numerator_scaled_to_reference_rowmax",
                    "denominator_scaled_to_reference_rowmax",
                    "lse",
                )
            )
        )
        frozen_conformant = frozen_conformant and frozen_ok
        reports.append(
            {
                "block_index": 2,
                "group_index": group,
                "completed": completed,
                "q_contract": item.get("q_contract"),
                "k_contract": item.get("k_contract"),
                "v_contract": item.get("v_contract"),
                "original_sink_rows": item.get("original_sink_rows"),
                "scale": item.get("scale"),
                "input_exact_on_entry": item.get("input_exact_on_entry"),
                "input_exact_after_sidecars": item.get("input_exact_after_sidecars"),
                "input_integrity_exact": input_ok,
                "all_selected_vs_native": all_selected,
                "all_selected_conformant": all_selected_ok,
                "all_selected_selected_block_pairs": item.get("all_selected_selected_block_pairs"),
                "all_selected_expected_block_pairs": item.get("all_selected_expected_block_pairs"),
                "all_selected_trace_complete": trace_count_ok,
                "sparse_selected_block_pairs": item.get("sparse_selected_block_pairs"),
                "summary_metrics": summary,
                "summary_conformant": summary_ok,
                "route_trace_matches_independent": route_ok,
                "route_mismatch_count": item.get("route_mismatch_count"),
                "route_mismatch_examples": item.get("route_mismatch_examples"),
                "frozen_route_reference": frozen,
                "frozen_route_conformant": frozen_ok,
                "debug_sparse_vs_ordinary": item.get("debug_sparse_vs_ordinary"),
                "lse_specialization_vs_ordinary": item.get("lse_specialization_vs_ordinary"),
                "debug_all_selected_vs_returned": item.get("debug_all_selected_vs_returned"),
                "debug_specializations_conform": debug_ok,
                "packaged_backend": item.get("packaged_backend"),
                "packaged_source_tree_verified": item.get("packaged_source_tree_verified"),
            }
        )
    provenance = [item for item in evidence if isinstance(item, dict) and item.get("kind") == "sol_sparse_provenance"]
    provenance_ok = bool(
        len(provenance) == 1
        and isinstance(provenance[0].get("provenance"), dict)
        and provenance[0]["provenance"].get("contract") == "sana-sol-engine-sol-attn-64-rect-sm120-v3"
        and provenance[0]["provenance"].get("source") == "sana-sol-engine"
        and provenance[0]["provenance"].get("revision") == "2936c47637380842aaa4a4488fac5006cc542b70"
        and provenance[0]["provenance"].get("compute_capability") == [12, 0]
    )
    execution_valid = bool(
        complete and provenance_ok and input_integrity and debug_conformant and all_selected_trace_conformant
    )
    arithmetic_conformant = bool(execution_valid and all_selected_conformant)
    sparse_conformant = bool(execution_valid and prep_conformant and route_conformant and frozen_conformant)
    return {
        "identity": identity,
        "expected_identity": expected_identity,
        "complete": complete,
        "reports": reports,
        "packaged_sparse_provenance": provenance[0].get("provenance") if len(provenance) == 1 else None,
        "packaged_sparse_provenance_valid": provenance_ok,
        "input_integrity_exact": input_integrity,
        "debug_specializations_conform": debug_conformant,
        "all_selected_trace_conformant": all_selected_trace_conformant,
        "all_selected_sdpa_conformant": all_selected_conformant,
        "preparation_conformant": prep_conformant,
        "selector_trace_conformant": route_conformant,
        "frozen_route_mixed_arithmetic_conformant": frozen_conformant,
        "execution_valid": execution_valid,
        "arithmetic_conformant": arithmetic_conformant,
        "production_sparse_conformant_to_independent_witness": sparse_conformant,
    }


def _sanitize_evidence(value: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        return {str(key): _sanitize_evidence(item) for key, item in value.items() if key not in _PRIVATE_EVIDENCE_KEYS}
    if isinstance(value, list):
        return [_sanitize_evidence(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_evidence(item) for item in value)
    return value


def _public_tensor_inventory(value: Any, prefix: str = "$") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if torch.is_tensor(value):
        result.append(
            {
                "path": prefix,
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "device": str(value.device),
                "sha256": _replay._tensor_sha256(value),
            }
        )
    elif isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            result.extend(_public_tensor_inventory(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.extend(_public_tensor_inventory(item, f"{prefix}[{index}]"))
    return result


def _atomic_write_bytes(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def _persist_evidence(
    capture_id: str,
    evidence: _Sink,
    raw_media: torch.Tensor,
    pre_media: torch.Tensor,
    report: dict[str, Any],
) -> dict[str, Any]:
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory()).resolve() / "h3_first_high_sol_local_e"
    except (ImportError, AttributeError, OSError) as exc:
        raise RuntimeError("first-high Sol-local E cannot resolve the ComfyUI output directory") from exc
    root.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_evidence(evidence.items)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "design_commit": DESIGN_COMMIT,
        "evidence": sanitized,
        "first_high_model_raw_video": raw_media.detach().to(device="cpu", copy=True),
        "first_high_pre_guidance_video": pre_media.detach().to(device="cpu", copy=True),
    }
    tensor_inventory = _public_tensor_inventory(payload)
    stem = f"{capture_id}-all-selected-e"
    tensor_path = root / f"{stem}.pt"
    json_path = root / f"{stem}.json"
    _atomic_write_bytes(tensor_path, lambda path: torch.save(payload, path))
    tensor_file_sha = _sha256_file(tensor_path)
    public_report = dict(report)
    public_report["durable_evidence"] = {
        "tensor_path": str(tensor_path),
        "tensor_file_sha256": tensor_file_sha,
        "tensor_inventory": tensor_inventory,
        "json_path": str(json_path),
    }
    encoded = json.dumps(public_report, indent=2, sort_keys=True, default=str).encode("utf-8")
    _atomic_write_bytes(json_path, lambda path: path.write_bytes(encoded))
    return {
        "tensor_path": str(tensor_path),
        "tensor_file_sha256": tensor_file_sha,
        "json_path": str(json_path),
        "json_file_sha256": _sha256_file(json_path),
        "tensor_inventory": tensor_inventory,
    }


def _core_cleanup_contract(guider: Any, options: dict[str, Any]) -> dict[str, Any]:
    return _w._core_cleanup_contract(guider, options)


def _sampling_runtime_identity(guider: Any) -> dict[str, Any]:
    return _w._sampling_runtime_identity(guider)


def _sampling_runtime_ok(identity: dict[str, Any]) -> bool:
    return _w._sampling_runtime_ok(identity)


def _export_contract(guider: Any, raw_x0: torch.Tensor, returned: Any) -> dict[str, Any]:
    return _w._export_contract(guider, raw_x0, returned)


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
        raise RuntimeError("first-high Sol-local E sampler-entry validation requires active execution diagnostics")
    _replay._validate_replay_sampler_entry(state.replay, record)
    call = _ACTIVE_CALL.get()
    if not isinstance(call, _Call) or call.state is not state:
        raise RuntimeError("first-high Sol-local E sampler entry has no matching execution-local completion owner")
    try:
        return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    except _FirstCallComplete as exc:
        if exc.token is not call.token:
            raise
        if call.completed:
            raise RuntimeError("first-high Sol-local E sampler callback completed more than once") from None
        if not torch.is_tensor(exc.x0):
            raise RuntimeError("first-high Sol-local E sampler callback x0 is not a packed tensor") from None
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
        raise RuntimeError("first-high Sol-local E requires strict, complete execution-contract diagnostics")
    replay = state.replay
    _replay._assert_checkpoint_identity_unchanged(replay.exact_checkpoint_identity)
    if not isinstance(options, dict):
        raise RuntimeError("first-high Sol-local E requires mutable guider model options")
    config = options.get(_runtime.PROGRESSIVE_KEY)
    if not _diag._eligible(config):
        raise RuntimeError("first-high Sol-local E requires the captured progressive Target Input config")
    current_sampler = _runtime.sampler_name(sampler)
    if str(replay.manifest.get("sampler")) != current_sampler or current_sampler != "sample_euler":
        raise RuntimeError("first-high Sol-local E is bounded to the captured Euler sampler")
    if int(seed or 0) != int(replay.manifest["seed"]):
        raise RuntimeError("first-high Sol-local E seed differs from R capture")
    if denoise_mask is not None or "high_denoise_mask" in replay.tensors:
        raise RuntimeError("first-high Sol-local E does not support protected/masked replay")
    if not isinstance(latent_shapes, list):
        raise RuntimeError("first-high Sol-local E requires mutable target latent-shape metadata")
    target_shapes = [tuple(int(dim) for dim in shape) for shape in replay.manifest["target_shapes"]]
    if [tuple(int(dim) for dim in shape) for shape in latent_shapes] != target_shapes:
        raise RuntimeError("first-high Sol-local E target geometry differs from R capture")
    if _runtime._schedule_signature(sigmas) != str(replay.manifest["original_schedule_digest"]):
        raise RuntimeError("first-high Sol-local E caller schedule differs from R capture")
    if record.pristine_cond_digest != str(replay.manifest["pristine_conditioning_digest"]):
        raise RuntimeError("first-high Sol-local E pristine target conditioning differs from R capture")

    provenance_gate = _provenance_gate(state, record, guider)
    if not provenance_gate["exact_except_reviewed_e_delta"]:
        raise RuntimeError(
            "first-high Sol-local E runtime provenance differs beyond the reviewed source delta: "
            + ", ".join(provenance_gate["unexpected_differences"][:12])
        )
    sampling_runtime = _sampling_runtime_identity(guider)
    sampling_runtime_ok = _sampling_runtime_ok(sampling_runtime)
    if not sampling_runtime_ok:
        raise RuntimeError("first-high Sol-local E runtime is not the expected MiniMax-H3 AV CONST contract")

    binding = _runtime._resolve_binding(guider)
    if binding is None:
        raise RuntimeError("first-high Sol-local E requires the active Flow binding")
    if binding.active_capture is not None or binding.active_guidance_run is not None:
        raise RuntimeError("first-high Sol-local E entered with stale Flow capture/guidance state")
    current_guidance = _replay._guidance_config_dict(binding)
    if not _replay._dict_equal(current_guidance, replay.manifest.get("guidance_config")):
        raise RuntimeError("first-high Sol-local E Flow guidance config differs from R capture")
    current_spectrum = _replay._spectrum_config_dict(guider)
    if not _replay._dict_equal(current_spectrum, replay.manifest.get("spectrum_config")):
        raise RuntimeError("first-high Sol-local E Spectrum config differs from R capture")

    trajectory, run_id = _replay._rebuild_trajectory(replay.manifest, replay.tensors)
    if current_guidance is not None and current_guidance.get("mode") != "off" and trajectory is None:
        raise RuntimeError("first-high Sol-local E guidance is active but R bundle has no trajectory")
    handoff_index = int(replay.manifest["handoff_index"])
    high_sigmas = sigmas[handoff_index:]
    captured_high = [float(value) for value in replay.manifest["high_sigmas"]]
    live_high = [float(value) for value in high_sigmas.detach().to(device="cpu", dtype=torch.float64).tolist()]
    if captured_high != _REQUIRED_SUFFIX or live_high != captured_high:
        raise RuntimeError("first-high Sol-local E must keep the complete captured high suffix")

    replay_latent = _replay._tensor_for_replay(replay, "high_latent_image")
    replay_noise = _replay._tensor_for_replay(replay, "high_noise_argument")
    memory_preflight = _memory_preflight(replay_latent.device)
    request = _request_tuple(state)
    receipt_sink = _Sink(limit=800)
    evidence_sink = _Sink(limit=4096)
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("first-high Sol-local E requires mutable transformer options")
    for key in (REQUEST_KEY, RECEIPTS_KEY, EVIDENCE_KEY):
        if key in transformer:
            raise RuntimeError(f"first-high Sol-local E private transformer option already exists: {key}")
    if transformer.get("h3_first_high_operator_diagnostic_v1") is not None:
        raise RuntimeError("first-high Sol-local E cannot coexist with W")
    if "h3_flow_untwist_clock_trial_v1" in transformer or "h3_flow_sampling_context" in transformer:
        raise RuntimeError("first-high Sol-local E requires the established no-Untwist R control")
    sol_config = transformer.get("sol_h3_runtime_v1")
    if not isinstance(sol_config, dict) or sol_config.get("backend") != "sol":
        raise RuntimeError("first-high Sol-local E requires the active Sol-H3 backend companion")
    if _ACTIVE_CALL.get() is not None:
        raise RuntimeError("nested first-high Sol-local E execution is unsupported")

    metric_start = binding.metrics.counters
    event_start = len(binding.metrics.events)
    previous_progressive = options.pop(_runtime.PROGRESSIVE_KEY)
    previous_trajectory = binding.trajectory
    previous_guidance_run_id = binding.guidance_run_id
    previous_capture_enabled = binding.capture_enabled
    binding.trajectory = trajectory
    binding.guidance_run_id = run_id
    binding.capture_enabled = False
    transformer[REQUEST_KEY] = request
    transformer[RECEIPTS_KEY] = receipt_sink
    transformer[EVIDENCE_KEY] = evidence_sink
    call = _Call(state=state)
    call_token = _ACTIVE_CALL.set(call)
    completed = False
    diagnostic_x0 = None
    diagnostic_output = None
    export_contract = None
    cleanup_contract = None
    replay_error: BaseException | None = None

    def stop_after_first(step, x0, x, _total):
        _ = x
        if int(step) != 0:
            raise RuntimeError("first-high Sol-local E callback reached a second Euler step")
        raise _FirstCallComplete(call.token, x0)

    try:
        if record.pristine_conds is None:
            raise RuntimeError("first-high Sol-local E record is missing pristine target conditions")
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
            raise RuntimeError("first-high Sol-local E callback sentinel did not terminate after the first model call")
        diagnostic_x0 = call.raw_x0
        export_contract = _export_contract(guider, diagnostic_x0, diagnostic_output)
        cleanup_contract = _core_cleanup_contract(guider, options)
        if export_contract["exact_core_callback_x0_externalization"] is not True:
            raise RuntimeError("first-high Sol-local E callback x0 bypassed Core externalization")
        if cleanup_contract["complete"] is not True:
            raise RuntimeError("first-high Sol-local E bounded return bypassed Core outer-sample cleanup")
        completed = True
        return diagnostic_output
    except BaseException as exc:
        replay_error = exc
        raise
    finally:
        _ACTIVE_CALL.reset(call_token)
        transformer.pop(REQUEST_KEY, None)
        transformer.pop(RECEIPTS_KEY, None)
        transformer.pop(EVIDENCE_KEY, None)
        binding.trajectory = previous_trajectory
        binding.guidance_run_id = previous_guidance_run_id
        binding.capture_enabled = previous_capture_enabled
        options[_runtime.PROGRESSIVE_KEY] = previous_progressive

        if completed and replay_error is None:
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
            snapshots = _snapshot_report(record, replay.manifest)
            entry_exact = all(
                (snapshots.get(name) or {}).get("design_expected_equal") is True for name in _EXPECTED_ENTRY_HASHES
            )
            target_video_shape = tuple(int(dim) for dim in replay.manifest["target_shapes"][0])
            raw_media = _w._decode_ready_video_snapshot(record, "first_high_model_raw_video", target_video_shape)
            pre_media = _w._decode_ready_video_snapshot(record, "first_high_pre_guidance_video", target_video_shape)
            vdn_receipts = _validate_vdn_receipts(receipt_sink)
            backend_receipts = _validate_backend_receipts(evidence_sink, str(replay.manifest["capture_id"]))
            witnesses = _validate_witnesses(evidence_sink)
            sol_counter_isolation = _validate_sol_counter_isolation(record)
            memory_report = _memory_completion(replay_latent.device, memory_preflight, evidence_sink)
            memory_valid = bool(memory_report["retained_cpu_evidence_within_budget"])
            execution_valid = bool(
                topology == expected_topology
                and memory_valid
                and entry_exact
                and vdn_receipts["valid"]
                and backend_receipts["valid"]
                and witnesses["execution_valid"]
                and sol_counter_isolation["valid"]
                and provenance_gate["exact_except_reviewed_e_delta"]
                and sampling_runtime_ok
                and export_contract is not None
                and export_contract["exact_core_callback_x0_externalization"] is True
                and cleanup_contract is not None
                and cleanup_contract["complete"] is True
            )
            arithmetic_conformant = bool(execution_valid and witnesses["arithmetic_conformant"])
            report = {
                "schema_version": SCHEMA_VERSION,
                "kind": "minimax_h3_first_high_sol_local_e_report",
                "design_commit": DESIGN_COMMIT,
                "capture_id": replay.manifest["capture_id"],
                "mode": MODE,
                "completed_first_call_only": True,
                "final_trajectory_available": False,
                "operator_modified_from_r": True,
                "r_attribution_valid_not_applicable": True,
                "execution_valid": execution_valid,
                "arithmetic_conformant": arithmetic_conformant,
                "media_clean": None,
                "media_assessment_required": True,
                "topology": topology,
                "topology_exact": topology == expected_topology,
                "memory": memory_report,
                "source_gate": state.source_gate,
                "provenance_gate": provenance_gate,
                "sampling_runtime": sampling_runtime,
                "sampling_runtime_ok": sampling_runtime_ok,
                "request": dict(request),
                "snapshot_hashes": snapshots,
                "entry_state_exact": entry_exact,
                "vdn_receipts": vdn_receipts,
                "backend_receipts": backend_receipts,
                "operator_witnesses": witnesses,
                "ordinary_sol_counter_isolation": sol_counter_isolation,
                "export_contract": export_contract,
                "core_cleanup": cleanup_contract,
                "decode_ready_media_available": True,
                "media_contract": {
                    "first_high_model_raw_shape": [int(dim) for dim in raw_media.shape],
                    "first_high_model_raw_sha256": _replay._tensor_sha256(raw_media),
                    "first_high_pre_guidance_shape": [int(dim) for dim in pre_media.shape],
                    "first_high_pre_guidance_sha256": _replay._tensor_sha256(pre_media),
                },
                "diagnostic_x0_sha256": _replay._tensor_sha256(diagnostic_x0),
                "diagnostic_x0_shape": [int(dim) for dim in diagnostic_x0.shape],
                "diagnostic_output_sha256": _replay._tensor_sha256(diagnostic_output),
                "diagnostic_output_shape": [int(dim) for dim in diagnostic_output.shape],
                "decision_gate": (
                    "invalid-e: fix the diagnostic defect and rerun only E"
                    if not execution_valid
                    else (
                        (
                            "all-selected-arithmetic-mismatch: localize descriptor/stride/"
                            "mask/normalization before any new H3 run"
                        )
                        if not arithmetic_conformant
                        else (
                            "valid-e: decode E media; run only the design-table follow-up "
                            "selected by that media plus witness result"
                        )
                    )
                ),
            }
            durable = _persist_evidence(str(replay.manifest["capture_id"]), evidence_sink, raw_media, pre_media, report)
            report["durable_evidence"] = durable
            state.complete.append({**report, _MEDIA_RAW_KEY: raw_media, _MEDIA_PRE_KEY: pre_media})


def patch_first_high_sol_local_diagnostic(model: Any, manifest_path: str) -> tuple[Any, _State]:
    diagnostic = (getattr(model, "model_options", None) or {}).get(_diag.DIAGNOSTIC_KEY)
    if not isinstance(diagnostic, _diag._State):
        raise RuntimeError("first-high Sol-local E must be applied after MiniMax H3 Execution Contract Diagnostics")
    if not diagnostic.strict_provenance or not diagnostic.manifest.get("gate_complete"):
        raise RuntimeError("first-high Sol-local E requires strict, complete installed-runtime provenance")
    resolved, manifest, tensors = _replay._load_bundle(manifest_path)
    if str(manifest.get("capture_id", "")) != _EXPECTED_CAPTURE_ID:
        raise RuntimeError(f"first-high Sol-local E requires preserved R capture {_EXPECTED_CAPTURE_ID}")
    exact_checkpoint = _replay._checkpoint_identity(diagnostic.manifest)
    if not _replay._dict_equal(exact_checkpoint, manifest.get("exact_checkpoint_identity")):
        raise RuntimeError("first-high Sol-local E exact checkpoint identity differs from R capture")
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
        raise RuntimeError("first-high Sol-local E requires mutable transformer options")
    if "h3_flow_untwist_clock_trial_v1" in transformer or "h3_flow_sampling_context" in transformer:
        raise RuntimeError("first-high Sol-local E requires the established no-Untwist R control")
    if (
        STATE_KEY in patched.model_options
        or _w.STATE_KEY in patched.model_options
        or _replay.CAPTURE_STATE_KEY in patched.model_options
        or _replay.REPLAY_STATE_KEY in patched.model_options
    ):
        raise RuntimeError("first-high Sol-local E cannot be combined with R capture/replay or W wrappers")
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


class H3FirstHighSolLocalDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Experiment E: replay the preserved R first-high state once, keep VDN restricted support/complement, and "
        "make only Sol local attention all-selected. Also captures bounded same-input operator witnesses at block 2 "
        "groups 0, 2 and 10. Diagnostic only; no production fix is promoted."
    )
    RETURN_TYPES = ("MODEL", "H3_FLOW_FIRST_HIGH_SOL_LOCAL_E")
    RETURN_NAMES = ("model", "diagnostic")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        bundles = _saved_bundles()
        if not bundles:
            bundles = ["<no saved R bundle>"]
        return {"required": {"model": ("MODEL",), "bundle": (bundles,)}}

    def apply(self, model, bundle):
        if str(bundle) == "<no saved R bundle>":
            raise RuntimeError("no saved R bundle exists under output/h3_flow_replay")
        path = _replay._resolve_replay_bundle_selector(str(bundle))
        return patch_first_high_sol_local_diagnostic(model, str(path))


class H3FirstHighSolLocalDiagnosticReport:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Emit E execution/arithmetic validity and decode-ready first-high media after the one-call sampler returns. "
        "media_clean remains unset until the decoded result is inspected. Costly operator evidence is persisted to "
        "the ComfyUI output directory before this node can report completion."
    )
    RETURN_TYPES = ("STRING", "LATENT", "LATENT")
    RETURN_NAMES = ("report", "first_high_model_raw", "first_high_pre_guidance")
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diagnostic": ("H3_FLOW_FIRST_HIGH_SOL_LOCAL_E",),
                "trigger": ("LATENT",),
            }
        }

    def extract(self, diagnostic, trigger):
        _ = trigger
        if not isinstance(diagnostic, _State):
            raise TypeError("invalid first-high Sol-local E diagnostic handle")
        if not diagnostic.complete:
            raise RuntimeError("no completed first-high Sol-local E report is available")
        entry = diagnostic.complete[-1]
        raw = entry.get(_MEDIA_RAW_KEY)
        pre = entry.get(_MEDIA_PRE_KEY)
        if not torch.is_tensor(raw) or not torch.is_tensor(pre):
            raise RuntimeError("completed first-high Sol-local E report is missing decode-ready media")
        diagnostic.complete.pop()
        public = {key: value for key, value in entry.items() if key not in {_MEDIA_RAW_KEY, _MEDIA_PRE_KEY}}
        return (
            json.dumps(public, indent=2, sort_keys=True, default=str),
            {"samples": raw},
            {"samples": pre},
        )


NODE_CLASS_MAPPINGS = {
    "H3FirstHighSolLocalDiagnostic": H3FirstHighSolLocalDiagnostic,
    "H3FirstHighSolLocalDiagnosticReport": H3FirstHighSolLocalDiagnosticReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FirstHighSolLocalDiagnostic": "MiniMax H3 First-High Sol-Local Diagnostic E",
    "H3FirstHighSolLocalDiagnosticReport": "MiniMax H3 First-High Sol-Local E Report",
}
