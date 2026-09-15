"""Bounded first-high operator comparison W.

W reuses an existing, hash-validated experiment-R bundle and executes exactly the
first high Euler model call in a fresh process. It changes only the explicit VDN
local-attention operator selected by ``mode``. The original high sigma suffix is
kept intact so MiniMax-H3 PDD/final-head selection sees the same schedule.

This module is diagnostic-only. It never changes production progressive policy,
never regenerates the R bundle, and never reports R ``attribution_valid`` for an
operator-modified run.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib
import json
import math
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

SCHEMA_VERSION = 1
STATE_KEY = "h3_flow_first_high_operator_comparison_v1"
REQUEST_KEY = "h3_first_high_operator_diagnostic_v1"
RECEIPTS_KEY = "h3_first_high_operator_receipts_v1"
_OUTER_KEY = "h3_flow_regenerate.first_high_operator.execute.v1"
_SAMPLER_KEY = "h3_flow_regenerate.first_high_operator.sampler_entry.v1"
_SOURCE_MANIFEST = "first_high_w_source_delta.json"
_DESIGN_COMMIT = "41b54405867a31e1a9e3261d4d79469a640682c4"
_ALLOWED_MODES = ("native_window", "native_full_support")
_REQUIRED_SUFFIX = [0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]
_MAX_REPORTS = 4
_EXPECTED_PACKED_ROWS = 56349
_EXPECTED_VIDEO_SPAN = (3101, 56349)
_EXPECTED_BLOCKS = 50
_EXPECTED_LOCAL_Q_ROWS = (4096,) + (5120,) * 9 + (1024,)
_EXPECTED_WINDOW_KV_ROWS = (14365, 19485) + (20509,) * 7 + (16413, 11293)
_OWNER_COMPANION_GROUP = {"flow": "flow", "sol": "sol_h3", "vdn": "vdn_h3"}
_REQUIRED_SOURCE_ENTRY_KEYS = frozenset(
    {
        ("flow", "h3_flow_regenerate.first_high_operator_comparison", "."),
        ("flow", "h3_flow_regenerate.first_high_operator_comparison", "../__init__.py"),
        ("sol", "sol_h3.first_high_operator_diagnostic", "."),
        ("sol", "sol_h3", "."),
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
        super().__init__("first-high W diagnostic completed its single model call")
        self.token = token
        self.x0 = x0


@dataclass(slots=True)
class _State:
    replay: _replay._ReplayState
    mode: str
    source_gate: dict[str, Any]
    complete: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_REPORTS))


@dataclass(slots=True)
class _Call:
    state: _State
    token: object = field(default_factory=object)
    completed: bool = False
    raw_x0: torch.Tensor | None = None


_ACTIVE_CALL: contextvars.ContextVar[_Call | None] = contextvars.ContextVar(
    "h3_flow_first_high_operator_call", default=None
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
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


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
        raw = path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("first-high W source-delta manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("first-high W source-delta manifest schema is unsupported")
    if manifest.get("design_commit") != _DESIGN_COMMIT:
        raise RuntimeError("first-high W source-delta manifest targets the wrong design commit")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("first-high W source-delta manifest has no reviewed entries")
    keys = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("first-high W source-delta entry is not a dictionary")
        keys.append(_source_entry_key(entry))
    if len(keys) != len(set(keys)):
        raise RuntimeError("first-high W source-delta manifest contains duplicate source entries")
    if frozenset(keys) != _REQUIRED_SOURCE_ENTRY_KEYS:
        raise RuntimeError("first-high W source-delta manifest entry set differs from the reviewed W contract")
    digest = _sha_json(manifest)
    return manifest, digest


def _resolve_source_entry(entry: dict[str, Any]) -> Path:
    module_name = entry.get("module")
    relative = entry.get("relative_path", ".")
    if not isinstance(module_name, str) or not module_name or not isinstance(relative, str):
        raise RuntimeError("first-high W source-delta entry has invalid path metadata")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"first-high W required source module is not importable: {module_name}") from exc
    raw_file = getattr(module, "__file__", None)
    if not isinstance(raw_file, str) or not raw_file:
        raise RuntimeError(f"first-high W source module has no file: {module_name}")
    base = Path(raw_file).resolve(strict=True)
    candidate = base if relative == "." else (base.parent / relative).resolve(strict=True)
    if not candidate.is_file():
        raise RuntimeError(f"first-high W source-delta path is not a file: {candidate}")
    return candidate


def _verify_source_manifest() -> dict[str, Any]:
    manifest, digest = _load_source_manifest()
    observed = []
    for raw in manifest["entries"]:
        path = _resolve_source_entry(raw)
        expected_blob = raw.get("candidate_git_blob_sha")
        base_blob = raw.get("base_git_blob_sha")
        base_sha256 = raw.get("base_sha256")
        if not _valid_git_blob_sha(expected_blob):
            raise RuntimeError("first-high W source-delta entry lacks candidate Git blob identity")
        if base_blob is not None and not _valid_git_blob_sha(base_blob):
            raise RuntimeError("first-high W source-delta entry has invalid base Git blob identity")
        changed_existing = base_blob is not None and base_blob != expected_blob
        if changed_existing and not _valid_sha256(base_sha256):
            raise RuntimeError("first-high W changed source entry lacks exact R base SHA-256")
        if not changed_existing and base_sha256 is not None:
            raise RuntimeError("first-high W source entry has an unexpected R base SHA-256")
        actual_blob = _git_blob_sha(path)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"first-high W source bytes differ from reviewed candidate: {path}: {actual_blob} != {expected_blob}"
            )
        observed.append(
            {
                "owner": raw.get("owner"),
                "module": raw.get("module"),
                "relative_path": raw.get("relative_path", "."),
                "path": str(path),
                "base_git_blob_sha": base_blob,
                "base_sha256": base_sha256,
                "candidate_git_blob_sha": expected_blob,
                "git_blob_sha": actual_blob,
                "sha256": _sha256_file(path),
                "reason": raw.get("reason"),
            }
        )
    return {"manifest_digest": digest, "entries": observed}


def _target_shapes_digest(manifest: dict[str, Any]) -> str:
    return _sha_json([[int(dim) for dim in shape] for shape in manifest["target_shapes"]])


def _request_tuple(state: _State) -> tuple[tuple[str, Any], ...]:
    manifest = state.replay.manifest
    high = [float(v) for v in manifest.get("high_sigmas", [])]
    if high != _REQUIRED_SUFFIX:
        raise RuntimeError(f"first-high W requires the original high suffix; got {high}")
    return (
        ("api", 1),
        ("capture_id", str(manifest["capture_id"])),
        ("mode", state.mode),
        ("stage", "high"),
        ("logical_call_limit", 1),
        ("sigma", float(high[0])),
        ("target_shapes_digest", _target_shapes_digest(manifest)),
        ("source_contract_digest", str(state.source_gate["manifest_digest"])),
    )


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
        "latent_format_class": (None if latent is None else f"{type(latent).__module__}.{type(latent).__qualname__}"),
        "latent_format_minimax_h3_av": isinstance(latent, comfy.latent_formats.MiniMaxH3AV),
        "latent_scale_factor": _numeric_attr(latent, "scale_factor"),
        "final_head_count": head_count,
    }


def _sampling_runtime_ok(identity: dict[str, Any]) -> bool:
    noise_scale = identity.get("noise_scale")
    head_count = identity.get("final_head_count")
    return bool(
        identity.get("model_sampling_av") is True
        and identity.get("model_sampling_const") is True
        and identity.get("latent_format_minimax_h3_av") is True
        and isinstance(noise_scale, float)
        and math.isfinite(noise_scale)
        and noise_scale > 0.0
        and type(head_count) is int
        and head_count > 0
    )


def _export_contract(guider: Any, raw_x0: torch.Tensor, returned: Any) -> dict[str, Any]:
    if not torch.is_tensor(raw_x0) or not torch.is_tensor(returned):
        raise RuntimeError("first-high W export contract requires packed tensor callback/output values")
    patcher = getattr(guider, "model_patcher", None)
    base = getattr(patcher, "model", None)
    process_out = getattr(base, "process_latent_out", None)
    if not callable(process_out):
        raise RuntimeError("first-high W cannot resolve the active Core process_latent_out transform")
    raw_float = raw_x0.to(torch.float32)
    expected = process_out(raw_float)
    if not torch.is_tensor(expected):
        raise RuntimeError("first-high W process_latent_out did not return a packed tensor")
    exact = (
        tuple(returned.shape) == tuple(expected.shape)
        and returned.dtype == expected.dtype
        and returned.device == expected.device
        and torch.equal(returned, expected)
    )
    process_out_identity = (
        tuple(expected.shape) == tuple(raw_float.shape)
        and expected.dtype == raw_float.dtype
        and expected.device == raw_float.device
        and torch.equal(expected, raw_float)
    )
    return {
        "exact_core_callback_x0_externalization": exact,
        "process_latent_out_identity_for_this_x0": process_out_identity,
        "raw_x0_shape": [int(dim) for dim in raw_x0.shape],
        "raw_x0_dtype": str(raw_x0.dtype),
        "raw_x0_device": str(raw_x0.device),
        "raw_x0_sha256": _replay._tensor_sha256(raw_x0),
        "expected_export_shape": [int(dim) for dim in expected.shape],
        "expected_export_dtype": str(expected.dtype),
        "expected_export_device": str(expected.device),
        "expected_export_sha256": _replay._tensor_sha256(expected),
        "returned_shape": [int(dim) for dim in returned.shape],
        "returned_dtype": str(returned.dtype),
        "returned_device": str(returned.device),
        "returned_sha256": _replay._tensor_sha256(returned),
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


def _vdn_w_source_path(source_gate: dict[str, Any]) -> str:
    matches = [
        str(entry.get("path", ""))
        for entry in source_gate.get("entries", [])
        if entry.get("owner") == "vdn"
        and entry.get("module") == "vdn_h3.first_high_operator_diagnostic"
        and entry.get("relative_path", ".") == "."
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError("first-high W source gate does not identify exactly one VDN diagnostic source file")
    return matches[0]


def _callable_equivalence_identity(value: Any) -> dict[str, Any]:
    identity = _callable_identity(value)
    normalized = _replay._provenance_equivalence_identity({"active_object_patches": {"value": identity}})
    return normalized["active_object_patches"]["value"]


def _normalize_vdn_w_object_patches(
    current: dict[str, Any],
    capture: dict[str, Any],
    guider: Any,
    source_gate: dict[str, Any],
) -> dict[str, Any]:
    """Prove W's construction wrapper delegates to the captured VDN closure.

    The VDN diagnostic overlay is installed before ApplyVDN constructs its fifty
    object patches, so the preflight provenance legitimately sees a thin W-aware
    wrapper instead of R's direct ``vdn_forward`` closure. We do not exempt that
    difference by path. Each live wrapper must come from the exact reviewed W
    source, expose its exact underlying production closure, and that closure must
    equal the corresponding R object-patch identity after the same provenance
    normalization. Only then is that one entry replaced for comparison.
    """
    current_patches = current.get("active_object_patches")
    capture_patches = capture.get("active_object_patches")
    patcher = getattr(guider, "model_patcher", None)
    runtime_patches = getattr(patcher, "object_patches", None)
    if not isinstance(current_patches, dict) or not isinstance(capture_patches, dict):
        raise RuntimeError("first-high W provenance lacks active VDN object-patch manifests")
    if not isinstance(runtime_patches, dict):
        raise RuntimeError("first-high W cannot inspect live model object patches")

    expected_keys = {f"diffusion_model.blocks.{index}.attn.forward" for index in range(_EXPECTED_BLOCKS)}
    wrapped = {
        str(key): value
        for key, value in runtime_patches.items()
        if getattr(value, "_h3_first_high_operator_diagnostic_v1", False) is True
    }
    if set(wrapped) != expected_keys:
        missing = sorted(expected_keys - set(wrapped))
        extra = sorted(set(wrapped) - expected_keys)
        raise RuntimeError(
            "first-high W VDN diagnostic object-patch set is not exactly the 50 H3 attention patches; "
            f"missing={missing[:4]} extra={extra[:4]}"
        )

    source_path = _vdn_w_source_path(source_gate)
    rebuilt = dict(current_patches)
    for key in sorted(expected_keys):
        wrapper = wrapped[key]
        wrapper_identity = _callable_identity(wrapper)
        file_info = wrapper_identity.get("file") if isinstance(wrapper_identity, dict) else None
        resolved = file_info.get("resolved_path") if isinstance(file_info, dict) else None
        if resolved != source_path:
            raise RuntimeError(f"first-high W VDN wrapper source identity changed for {key}: {resolved!r}")
        original = getattr(wrapper, "_h3_first_high_operator_original_forward", None)
        if not callable(original):
            raise RuntimeError(f"first-high W VDN wrapper lacks its underlying production forward for {key}")
        if key not in current_patches or key not in capture_patches:
            raise RuntimeError(f"first-high W provenance is missing VDN object-patch identity for {key}")
        original_identity = _callable_equivalence_identity(original)
        capture_identity = capture_patches[key]
        if _canonical_json(original_identity) != _canonical_json(capture_identity):
            detail = _replay._provenance_diff_paths(capture_identity, original_identity, limit=8)
            raise RuntimeError(
                f"first-high W underlying VDN production forward differs from R for {key}"
                + ("; " + ", ".join(detail) if detail else "")
            )
        rebuilt[key] = capture_identity

    result = dict(current)
    result["active_object_patches"] = rebuilt
    return result


def _verify_capture_base_sources(capture: dict[str, Any], source_gate: dict[str, Any]) -> dict[str, Any]:
    groups = capture.get("loaded_companion_sources")
    if not isinstance(groups, dict):
        raise RuntimeError("first-high W R provenance lacks companion source inventory")
    checked = []
    for entry in source_gate.get("entries", []):
        base_blob = entry.get("base_git_blob_sha")
        candidate_blob = entry.get("candidate_git_blob_sha")
        if base_blob is None or base_blob == candidate_blob:
            continue
        owner = str(entry.get("owner", ""))
        group_name = _OWNER_COMPANION_GROUP.get(owner)
        if group_name is None:
            raise RuntimeError(f"first-high W changed source has no R companion owner mapping: {owner!r}")
        expected_sha256 = entry.get("base_sha256")
        if not _valid_sha256(expected_sha256):
            raise RuntimeError("first-high W changed source lacks a reviewed R base SHA-256")
        capture_sources, problems = _replay._companion_source_map(groups.get(group_name))
        if problems:
            raise RuntimeError(
                f"first-high W R companion source inventory is invalid for {group_name}: " + "; ".join(problems)
            )
        source_path = str(entry.get("path", ""))
        captured = capture_sources.get(source_path)
        if captured is None:
            raise RuntimeError(f"first-high W R provenance is missing reviewed base source: {source_path}")
        captured_sha256 = captured.get("sha256")
        if captured_sha256 != expected_sha256:
            raise RuntimeError(
                f"first-high W R base source differs from reviewed bytes: {source_path}: "
                f"{captured_sha256} != {expected_sha256}"
            )
        checked.append(
            {
                "owner": owner,
                "path": source_path,
                "capture_sha256": captured_sha256,
                "expected_base_sha256": expected_sha256,
                "exact": True,
            }
        )
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
    current = _normalize_vdn_w_object_patches(current, capture, guider, state.source_gate)
    differences = _replay._cross_process_provenance_diff_paths(capture, current, limit=64)
    unexpected = [path for path in differences if not _allowed_provenance_difference(path, state.source_gate)]
    return {
        "differences": differences,
        "unexpected_differences": unexpected,
        "capture_base_sources": capture_base_sources,
        "exact_except_reviewed_w_delta": capture_base_sources["exact"] and not unexpected,
    }


def _snapshot_hash(record: _diag._Record, name: str) -> str | None:
    snapshot = record.snapshots.get(name)
    return None if snapshot is None else _replay._tensor_sha256(snapshot.tensor)


def _snapshot_report(record: _diag._Record, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get("expected_snapshot_hashes") or {}
    names = (
        "first_high_sampler_input_video",
        "first_high_sampler_input_audio",
        "first_high_h3_input_video",
        "first_high_h3_input_audio",
        "first_high_h3_velocity_video",
        "first_high_h3_velocity_audio",
        "first_high_model_raw_video",
        "first_high_model_raw_audio",
        "first_high_pre_guidance_video",
    )
    result = {}
    for name in names:
        current = _snapshot_hash(record, name)
        result[name] = {
            "capture_sha256": expected.get(name),
            "w_sha256": current,
            "entry_equal": (
                current is not None and expected.get(name) is not None and current == expected.get(name)
                if name
                in {
                    "first_high_sampler_input_video",
                    "first_high_sampler_input_audio",
                    "first_high_h3_input_video",
                    "first_high_h3_input_audio",
                }
                else None
            ),
        }
    return result


def _block_receipt_topology_ok(subcalls: list[dict[str, Any]]) -> bool:
    by_block: dict[int, list[dict[str, Any]]] = {}
    for item in subcalls:
        block = item.get("block")
        if type(block) is not int or not 0 <= block < _EXPECTED_BLOCKS:
            return False
        by_block.setdefault(block, []).append(item)
    if set(by_block) != set(range(_EXPECTED_BLOCKS)):
        return False
    for items in by_block.values():
        kinds = Counter(str(item.get("kind")) for item in items)
        if kinds != Counter({"local": 11, "global": 1, "anchor": 2}):
            return False
        local_groups = sorted(item.get("group_index") for item in items if item.get("kind") == "local")
        anchor_groups = sorted(item.get("group_index") for item in items if item.get("kind") == "anchor")
        global_groups = [item.get("group_index") for item in items if item.get("kind") == "global"]
        if local_groups != list(range(11)) or anchor_groups != [0, 1] or global_groups != [None]:
            return False
        q_rows = [item.get("q_rows") for item in items]
        if any(type(rows) is not int or rows <= 0 for rows in q_rows):
            return False
        if sum(q_rows) != _EXPECTED_PACKED_ROWS:
            return False
    return True


def _block_receipt_group_geometry_ok(subcalls: list[dict[str, Any]], mode: str) -> bool:
    by_block: dict[int, list[dict[str, Any]]] = {}
    for item in subcalls:
        block = item.get("block")
        if type(block) is not int or not 0 <= block < _EXPECTED_BLOCKS:
            return False
        by_block.setdefault(block, []).append(item)
    if set(by_block) != set(range(_EXPECTED_BLOCKS)):
        return False
    expected_local_kv = (
        (_EXPECTED_PACKED_ROWS,) * len(_EXPECTED_LOCAL_Q_ROWS)
        if mode == "native_full_support"
        else _EXPECTED_WINDOW_KV_ROWS
    )
    for items in by_block.values():
        globals_ = [item for item in items if item.get("kind") == "global"]
        locals_ = sorted(
            (item for item in items if item.get("kind") == "local"),
            key=lambda item: item.get("group_index", -1),
        )
        anchors = sorted(
            (item for item in items if item.get("kind") == "anchor"),
            key=lambda item: item.get("group_index", -1),
        )
        if len(globals_) != 1 or len(locals_) != 11 or len(anchors) != 2:
            return False
        if globals_[0].get("q_rows") != _EXPECTED_VIDEO_SPAN[0] or globals_[0].get("kv_rows") != _EXPECTED_PACKED_ROWS:
            return False
        if tuple(item.get("q_rows") for item in locals_) != _EXPECTED_LOCAL_Q_ROWS:
            return False
        if tuple(item.get("kv_rows") for item in locals_) != expected_local_kv:
            return False
        if tuple(item.get("q_rows") for item in anchors) != (1024, 1024):
            return False
        if tuple(item.get("kv_rows") for item in anchors) != (_EXPECTED_PACKED_ROWS, _EXPECTED_PACKED_ROWS):
            return False
    return True


def _validate_receipts(receipts: list[Any], mode: str) -> dict[str, Any]:
    subcalls = [item for item in receipts if isinstance(item, dict)]
    kinds = Counter(str(item.get("kind")) for item in subcalls)
    local_routes = Counter(str(item.get("provider_route")) for item in subcalls if item.get("kind") == "local")
    packed_rows = {int(item.get("packed_rows", -1)) for item in subcalls}
    video_spans = {(int(item.get("video_start", -1)), int(item.get("video_end", -1))) for item in subcalls}
    local = [item for item in subcalls if item.get("kind") == "local"]
    nonlocal_calls = [item for item in subcalls if item.get("kind") in {"global", "anchor"}]
    expected_local_route = "vdn_local_native_window_w" if mode == "native_window" else "vdn_local_native_full_w"
    support_ok = all(
        item.get("support_mode") == ("restricted_window" if mode == "native_window" else "canonical_full")
        for item in local
    )
    complement_ok = all(bool(item.get("complement_executed")) == (mode == "native_window") for item in local)
    full_kv_ok = True
    if mode == "native_full_support":
        full_kv_ok = all(
            item.get("canonical_full_kv") is True and int(item.get("kv_rows", -1)) == _EXPECTED_PACKED_ROWS
            for item in local
        )
    else:
        full_kv_ok = all(0 < int(item.get("kv_rows", -1)) <= _EXPECTED_PACKED_ROWS for item in local)
    nonlocal_ok = all(
        int(item.get("kv_rows", -1)) == _EXPECTED_PACKED_ROWS
        and item.get("support_mode") == "full"
        and item.get("complement_executed") is False
        for item in nonlocal_calls
    )
    geometry_ok = packed_rows == {_EXPECTED_PACKED_ROWS} and video_spans == {_EXPECTED_VIDEO_SPAN}
    per_block_ok = _block_receipt_topology_ok(subcalls)
    per_group_geometry_ok = _block_receipt_group_geometry_ok(subcalls, mode)
    fingerprints_ok = bool(subcalls) and all(
        isinstance(item.get("gate_fingerprint"), str) and isinstance(item.get("adapter_fingerprint"), str)
        for item in subcalls
    )
    pre_attention_digest = next(
        (
            item.get("pre_attention_qkv_digest")
            for item in subcalls
            if item.get("block") == 0 and item.get("kind") == "global"
        ),
        None,
    )
    pre_attention_digest_present = isinstance(pre_attention_digest, str) and len(pre_attention_digest) == 64
    expected = kinds == Counter({"local": 550, "global": 50, "anchor": 100})
    return {
        "count": len(subcalls),
        "kind_counts": dict(kinds),
        "local_routes": dict(local_routes),
        "packed_rows": sorted(packed_rows),
        "video_spans": sorted(video_spans),
        "support_ok": support_ok,
        "complement_ok": complement_ok,
        "canonical_or_window_kv_ok": full_kv_ok,
        "nonlocal_full_support_ok": nonlocal_ok,
        "geometry_ok": geometry_ok,
        "per_block_topology_ok": per_block_ok,
        "per_group_geometry_ok": per_group_geometry_ok,
        "fingerprints_present": fingerprints_ok,
        "expected_700_subcalls": expected and len(subcalls) == 700,
        "expected_local_route": local_routes == Counter({expected_local_route: 550}),
        "first_block_pre_attention_qkv_digest": pre_attention_digest,
        "first_block_pre_attention_qkv_digest_present": pre_attention_digest_present,
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
        raise RuntimeError("first-high W sampler-entry validation requires active execution diagnostics")
    _replay._validate_replay_sampler_entry(state.replay, record)
    call = _ACTIVE_CALL.get()
    if not isinstance(call, _Call) or call.state is not state:
        raise RuntimeError("first-high W sampler entry has no matching execution-local completion owner")
    try:
        return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    except _FirstCallComplete as exc:
        if exc.token is not call.token:
            raise
        if call.completed:
            raise RuntimeError("first-high W sampler callback completed more than once") from None
        if not torch.is_tensor(exc.x0):
            raise RuntimeError("first-high W sampler callback x0 is not a packed tensor") from None
        call.completed = True
        call.raw_x0 = exc.x0
        # Catch at SAMPLER_SAMPLE so KSAMPLER cannot take another Euler step, while
        # CFGGuider.inner_sample/outer_sample still finish their normal externalize
        # and cleanup lifecycle above this boundary.
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
        raise RuntimeError("first-high W requires strict, complete execution-contract diagnostics")
    replay = state.replay
    _replay._assert_checkpoint_identity_unchanged(replay.exact_checkpoint_identity)
    if not isinstance(options, dict):
        raise RuntimeError("first-high W requires mutable guider model options")
    config = options.get(_runtime.PROGRESSIVE_KEY)
    if not _diag._eligible(config):
        raise RuntimeError("first-high W requires the original progressive Target Input config")
    current_sampler = _runtime.sampler_name(sampler)
    if str(replay.manifest.get("sampler")) != current_sampler or current_sampler != "sample_euler":
        raise RuntimeError("first-high W is bounded to the captured Euler sampler")
    if int(seed or 0) != int(replay.manifest["seed"]):
        raise RuntimeError("first-high W seed differs from R capture")
    if denoise_mask is not None or "high_denoise_mask" in replay.tensors:
        raise RuntimeError("first-high W does not support protected/masked replay")
    if not isinstance(latent_shapes, list):
        raise RuntimeError("first-high W requires mutable target latent-shape metadata")
    target_shapes = [tuple(int(dim) for dim in shape) for shape in replay.manifest["target_shapes"]]
    if [tuple(int(dim) for dim in shape) for shape in latent_shapes] != target_shapes:
        raise RuntimeError("first-high W target geometry differs from R capture")
    if _runtime._schedule_signature(sigmas) != str(replay.manifest["original_schedule_digest"]):
        raise RuntimeError("first-high W caller schedule differs from R capture")
    if record.pristine_cond_digest != str(replay.manifest["pristine_conditioning_digest"]):
        raise RuntimeError("first-high W pristine target conditioning differs from R capture")

    provenance_gate = _provenance_gate(state, record, guider)
    if not provenance_gate["exact_except_reviewed_w_delta"]:
        raise RuntimeError(
            "first-high W runtime provenance differs beyond the reviewed source delta: "
            + ", ".join(provenance_gate["unexpected_differences"][:12])
        )
    sampling_runtime = _sampling_runtime_identity(guider)
    sampling_runtime_ok = _sampling_runtime_ok(sampling_runtime)
    if not sampling_runtime_ok:
        raise RuntimeError("first-high W runtime is not the expected MiniMax-H3 AV flow sampling contract")

    binding = _runtime._resolve_binding(guider)
    if binding is None:
        raise RuntimeError("first-high W requires the active Flow binding")
    if binding.active_capture is not None or binding.active_guidance_run is not None:
        raise RuntimeError("first-high W entered with stale Flow capture/guidance state")
    current_guidance = _replay._guidance_config_dict(binding)
    if not _replay._dict_equal(current_guidance, replay.manifest.get("guidance_config")):
        raise RuntimeError("first-high W Flow guidance config differs from R capture")
    current_spectrum = _replay._spectrum_config_dict(guider)
    if not _replay._dict_equal(current_spectrum, replay.manifest.get("spectrum_config")):
        raise RuntimeError("first-high W Spectrum config differs from R capture")

    trajectory, run_id = _replay._rebuild_trajectory(replay.manifest, replay.tensors)
    if current_guidance is not None and current_guidance.get("mode") != "off" and trajectory is None:
        raise RuntimeError("first-high W guidance is active but R bundle has no captured trajectory")

    handoff_index = int(replay.manifest["handoff_index"])
    high_sigmas = sigmas[handoff_index:]
    captured_high = [float(value) for value in replay.manifest["high_sigmas"]]
    live_high = [float(value) for value in high_sigmas.detach().to(device="cpu", dtype=torch.float64).tolist()]
    if captured_high != _REQUIRED_SUFFIX or live_high != captured_high:
        raise RuntimeError("first-high W must keep the complete original captured high suffix")

    replay_latent = _replay._tensor_for_replay(replay, "high_latent_image")
    replay_noise = _replay._tensor_for_replay(replay, "high_noise_argument")
    request = _request_tuple(state)
    receipt_sink: list[Any] = []
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("first-high W requires mutable transformer options")
    for key in (REQUEST_KEY, RECEIPTS_KEY):
        if key in transformer:
            raise RuntimeError(f"first-high W private transformer option already exists: {key}")
    sol_config = transformer.get("sol_h3_runtime_v1")
    if not isinstance(sol_config, dict) or sol_config.get("backend") != "sol":
        raise RuntimeError("first-high W requires the active Sol-H3 backend companion")
    if _ACTIVE_CALL.get() is not None:
        raise RuntimeError("nested first-high W execution is unsupported")

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
            raise RuntimeError("first-high W callback reached a second Euler step")
        raise _FirstCallComplete(call.token, x0)

    try:
        if record.pristine_conds is None:
            raise RuntimeError("first-high W record is missing pristine target conditions")
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
            raise RuntimeError("first-high W Euler callback sentinel did not terminate after the first model call")
        diagnostic_x0 = call.raw_x0
        export_contract = _export_contract(guider, diagnostic_x0, diagnostic_output)
        cleanup_contract = _core_cleanup_contract(guider, options)
        if export_contract["exact_core_callback_x0_externalization"] is not True:
            raise RuntimeError("first-high W callback x0 did not traverse the canonical Core externalization path")
        if cleanup_contract["complete"] is not True:
            raise RuntimeError("first-high W bounded sampler return bypassed Core outer-sample cleanup")
        completed = True
        return diagnostic_output
    except BaseException as exc:
        replay_error = exc
        raise
    finally:
        _ACTIVE_CALL.reset(call_token)
        transformer.pop(REQUEST_KEY, None)
        transformer.pop(RECEIPTS_KEY, None)
        binding.trajectory = previous_trajectory
        binding.guidance_run_id = previous_guidance_run_id
        binding.capture_enabled = previous_capture_enabled
        options[_runtime.PROGRESSIVE_KEY] = previous_progressive

        if completed and replay_error is None:
            metric_delta = _diag._counter_delta(metric_start, binding.metrics.counters)
            events = binding.metrics.events[event_start:]
            upscaler_calls = sum(1 for event in events if event.kind == "handoff_learned_upscale_wall")
            topology = {
                "logical": int(metric_delta.get("sampler_logical_calls", 0)),
                "actual": int(metric_delta.get("transformer_actual_nfe", 0)),
                "forecast": int(metric_delta.get("spectrum_forecast_calls", 0)),
                "high_logical": int(metric_delta.get("sampler_logical_calls_high", 0)),
                "high_actual": int(metric_delta.get("transformer_actual_nfe_high", 0)),
                "high_forecast": int(metric_delta.get("spectrum_forecast_calls_high", 0)),
                "upscaler_calls": int(upscaler_calls),
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
                snapshots[name]["entry_equal"] is True
                for name in (
                    "first_high_sampler_input_video",
                    "first_high_sampler_input_audio",
                    "first_high_h3_input_video",
                    "first_high_h3_input_audio",
                )
            )
            receipt_report = _validate_receipts(receipt_sink, state.mode)
            companion = _replay._high_first_companion_observation(record) or {}
            sol_after = ((companion.get("after") or {}).get("sol") or {}) if isinstance(companion, dict) else {}
            sol_zero = bool(
                int(sol_after.get("vdn_local_sol_calls", 0) or 0) == 0
                and int(sol_after.get("sparse_calls", 0) or 0) == 0
                and int(sol_after.get("external_mixed_sol_calls", 0) or 0) == 0
            )
            invariants = bool(
                topology == expected_topology
                and entry_exact
                and receipt_report["expected_700_subcalls"]
                and receipt_report["expected_local_route"]
                and receipt_report["support_ok"]
                and receipt_report["complement_ok"]
                and receipt_report["canonical_or_window_kv_ok"]
                and receipt_report["nonlocal_full_support_ok"]
                and receipt_report["geometry_ok"]
                and receipt_report["per_block_topology_ok"]
                and receipt_report["per_group_geometry_ok"]
                and receipt_report["fingerprints_present"]
                and receipt_report["first_block_pre_attention_qkv_digest_present"]
                and sol_zero
                and provenance_gate["exact_except_reviewed_w_delta"]
                and sampling_runtime_ok
                and export_contract is not None
                and export_contract["exact_core_callback_x0_externalization"] is True
                and cleanup_contract is not None
                and cleanup_contract["complete"] is True
            )
            state.complete.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "minimax_h3_first_high_operator_comparison_report",
                    "capture_id": replay.manifest["capture_id"],
                    "mode": state.mode,
                    "completed_first_call_only": True,
                    "final_trajectory_available": False,
                    "operator_modified_from_r": True,
                    "r_attribution_valid_not_applicable": True,
                    "w_invariants_valid": invariants,
                    "topology": topology,
                    "topology_exact": topology == expected_topology,
                    "source_gate": state.source_gate,
                    "provenance_gate": provenance_gate,
                    "sampling_runtime": sampling_runtime,
                    "sampling_runtime_ok": sampling_runtime_ok,
                    "request": dict(request),
                    "snapshot_hashes": snapshots,
                    "entry_state_exact": entry_exact,
                    "receipts": receipt_report,
                    "sol_local_sparse_zero": sol_zero,
                    "sol_after": sol_after,
                    "export_contract": export_contract,
                    "core_cleanup": cleanup_contract,
                    "diagnostic_x0_sha256": _replay._tensor_sha256(diagnostic_x0),
                    "diagnostic_x0_shape": [int(dim) for dim in diagnostic_x0.shape],
                    "diagnostic_output_sha256": _replay._tensor_sha256(diagnostic_output),
                    "diagnostic_output_shape": [int(dim) for dim in diagnostic_output.shape],
                    "decision_gate": (
                        "valid-arm: compare decoded first-high diagnostic media"
                        if invariants
                        else "invalid-arm: do not interpret media"
                    ),
                }
            )


def patch_first_high_operator_comparison(
    model: Any,
    manifest_path: str,
    mode: str,
) -> tuple[Any, _State]:
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"unsupported first-high W mode: {mode}")
    diagnostic = (getattr(model, "model_options", None) or {}).get(_diag.DIAGNOSTIC_KEY)
    if not isinstance(diagnostic, _diag._State):
        raise RuntimeError("first-high W must be applied after MiniMax H3 Execution Contract Diagnostics")
    if not diagnostic.strict_provenance or not diagnostic.manifest.get("gate_complete"):
        raise RuntimeError("first-high W requires strict, complete installed-runtime provenance")
    resolved, manifest, tensors = _replay._load_bundle(manifest_path)
    exact_checkpoint = _replay._checkpoint_identity(diagnostic.manifest)
    if not _replay._dict_equal(exact_checkpoint, manifest.get("exact_checkpoint_identity")):
        raise RuntimeError("first-high W exact checkpoint identity differs from R capture")
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
        raise RuntimeError("first-high W requires mutable transformer options")
    if "h3_flow_untwist_clock_trial_v1" in transformer or "h3_flow_sampling_context" in transformer:
        raise RuntimeError("first-high W requires the established no-Untwist R control")
    if (
        STATE_KEY in patched.model_options
        or _replay.CAPTURE_STATE_KEY in patched.model_options
        or _replay.REPLAY_STATE_KEY in patched.model_options
    ):
        raise RuntimeError("first-high W cannot be combined with R capture/replay wrappers")
    state = _State(replay=replay_state, mode=mode, source_gate=source_gate)
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


class H3FirstHighOperatorComparison:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Experiment W: replay the exact saved R first-high state and stop after one real Euler model call. "
        "native_window keeps VDN restricted support + linear complement; native_full_support keeps the same Q groups "
        "with canonical full packed K/V and no complement. Diagnostic only."
    )
    RETURN_TYPES = ("MODEL", "H3_FLOW_FIRST_HIGH_OPERATOR_W")
    RETURN_NAMES = ("model", "comparison")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        bundles = _saved_bundles()
        if not bundles:
            bundles = ["<no saved R bundle>"]
        return {
            "required": {
                "model": ("MODEL",),
                "bundle": (bundles,),
                "mode": (list(_ALLOWED_MODES), {"default": "native_window"}),
            }
        }

    def apply(self, model, bundle, mode):
        if str(bundle) == "<no saved R bundle>":
            raise RuntimeError("no saved R bundle exists under output/h3_flow_replay")
        path = _replay._resolve_replay_bundle_selector(str(bundle))
        return patch_first_high_operator_comparison(model, str(path), str(mode))


class H3FirstHighOperatorComparisonReport:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Emit W invariants after the one-call sampler returns its diagnostic x0. "
        "final_trajectory_available is always false; do not treat the trigger as a completed diffusion trajectory."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comparison": ("H3_FLOW_FIRST_HIGH_OPERATOR_W",),
                "trigger": ("LATENT",),
            }
        }

    def extract(self, comparison, trigger):
        _ = trigger
        if not isinstance(comparison, _State):
            raise TypeError("invalid first-high W comparison handle")
        if not comparison.complete:
            raise RuntimeError("no completed first-high W report is available")
        return (json.dumps(comparison.complete.pop(), indent=2, sort_keys=True, default=str),)


NODE_CLASS_MAPPINGS = {
    "H3FirstHighOperatorComparison": H3FirstHighOperatorComparison,
    "H3FirstHighOperatorComparisonReport": H3FirstHighOperatorComparisonReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FirstHighOperatorComparison": "MiniMax H3 First-High Operator Comparison W",
    "H3FirstHighOperatorComparisonReport": "MiniMax H3 First-High Operator W Report",
}
