"""Flow-side adapter for the evidence-selected mapped-neighbor arm M.

The reviewed E one-call replay/cleanup machinery is reused unchanged.  This
stacked diagnostic overlay changes only the request mode, exact source allowlist,
route validators, durable evidence name, and user-visible arm identity required
for the single M media call selected by the authoritative design.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from . import first_high_sol_local_diagnostic as e

MODE = "mapped_neighbor_m"
ROUTE = "vdn_local_sol_mapped_neighbor_m"
_SOURCE_MANIFEST = "first_high_mapped_neighbor_m_source_delta.json"
_EXPECTED_M_KEYS = frozenset(
    {
        ("flow", "h3_flow_regenerate.first_high_sol_local_diagnostic", "."),
        ("flow", "h3_flow_regenerate.first_high_sol_local_diagnostic", "../__init__.py"),
        ("flow", "h3_flow_regenerate.first_high_operator_comparison", "."),
        ("flow", "h3_flow_regenerate", "."),
        ("flow", "h3_flow_regenerate.first_high_mapped_neighbor_diagnostic", "."),
        ("sol", "sol_h3.first_high_sol_local_diagnostic", "."),
        ("sol", "sol_h3.first_high_sol_local_witness_bridge", "."),
        ("sol", "sol_h3.first_high_sol_local_receipt_tap", "."),
        ("sol", "sol_h3.first_high_operator_diagnostic", "."),
        ("sol", "sol_h3.first_high_mapped_neighbor_diagnostic", "."),
        ("sol", "sol_h3", "."),
        ("vdn", "vdn_h3.first_high_sol_local_diagnostic", "."),
        ("vdn", "vdn_h3.first_high_sol_local_bridge", "."),
        ("vdn", "vdn_h3.first_high_operator_diagnostic", "."),
        ("vdn", "vdn_h3.first_high_operator_sol_bridge", "."),
        ("vdn", "vdn_h3.first_high_mapped_neighbor_diagnostic", "."),
        ("vdn", "vdn_h3.first_high_mapped_neighbor_bridge", "."),
        ("vdn", "vdn_h3", "../__init__.py"),
        ("core", "comfy.model_sampling", "."),
        ("core", "comfy.latent_formats", "."),
        ("core", "comfy.model_patcher", "."),
        ("core", "comfy.k_diffusion.sampling", "."),
    }
)

# The existing E wrapper/report reads MODE dynamically.  Changing the module
# constant on this stacked PR keeps wrapper keys and the one-call lifecycle
# identical while giving M a separately identifiable immutable request.
e.MODE = MODE

_ORIGINAL_VALIDATE_VDN = e._validate_vdn_receipts
_ORIGINAL_VALIDATE_BACKEND = e._validate_backend_receipts
_ORIGINAL_VALIDATE_SOL_COUNTERS = e._validate_sol_counter_isolation


def _verify_m_source_manifest() -> dict[str, Any]:
    path = Path(__file__).with_name(_SOURCE_MANIFEST)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("mapped-neighbor M source-delta manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("mapped-neighbor M source-delta manifest schema is unsupported")
    if manifest.get("design_commit") != e.DESIGN_COMMIT or manifest.get("mode") != MODE:
        raise RuntimeError("mapped-neighbor M source-delta manifest targets the wrong design/mode")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("mapped-neighbor M source-delta manifest has no reviewed entries")
    keys = [e._source_entry_key(item) for item in entries if isinstance(item, dict)]
    if len(keys) != len(entries) or len(keys) != len(set(keys)) or frozenset(keys) != _EXPECTED_M_KEYS:
        raise RuntimeError("mapped-neighbor M source-delta manifest differs from the reviewed exact entry set")

    observed = []
    for raw in entries:
        resolved = e._resolve_source_entry(raw)
        candidate = raw.get("candidate_git_blob_sha")
        base = raw.get("base_git_blob_sha")
        base_sha256 = raw.get("base_sha256")
        if not e._valid_git_blob_sha(candidate):
            raise RuntimeError("mapped-neighbor M source entry lacks candidate Git blob identity")
        if base is not None and not e._valid_git_blob_sha(base):
            raise RuntimeError("mapped-neighbor M source entry has invalid R-base Git blob identity")
        changed_existing = base is not None and base != candidate
        if changed_existing and not e._valid_sha256(base_sha256):
            raise RuntimeError("mapped-neighbor M changed source entry lacks exact R-base SHA-256")
        if not changed_existing and base_sha256 is not None:
            raise RuntimeError("mapped-neighbor M source entry has unexpected R-base SHA-256")
        actual = e._git_blob_sha(resolved)
        if actual != candidate:
            raise RuntimeError(
                f"mapped-neighbor M source bytes differ from reviewed candidate: {resolved}: {actual} != {candidate}"
            )
        observed.append({**raw, "path": str(resolved), "git_blob_sha": actual, "sha256": e._sha256_file(resolved)})
    return {"manifest_digest": e._sha_json(manifest), "entries": observed}


e._verify_source_manifest = _verify_m_source_manifest


def _normalized_vdn_sink(receipts):
    normalized = e._Sink(limit=800)
    actual_routes = Counter()
    for item in receipts:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        copy = dict(item)
        if copy.get("kind") == "local":
            actual_routes[str(copy.get("provider_route"))] += 1
            if copy.get("provider_route") == ROUTE:
                copy["provider_route"] = "vdn_local_sol_all_selected_e"
        normalized.append(copy)
    return normalized, actual_routes


def _validate_vdn_m(receipts):
    normalized, actual_routes = _normalized_vdn_sink(receipts)
    result = dict(_ORIGINAL_VALIDATE_VDN(normalized))
    expected = Counter({"vdn_dense_warmup": 22, ROUTE: 528})
    route_ok = actual_routes == expected
    result["local_routes"] = dict(actual_routes)
    result["expected_local_routes"] = route_ok
    result["mapped_neighbor_route_exact"] = route_ok
    result["valid"] = bool(result.get("valid") and route_ok)
    return result


def _validate_backend_m(evidence, capture_id: str):
    normalized = e._Sink(limit=4096)
    actual_routes = Counter()
    for item in evidence:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        copy = dict(item)
        if copy.get("kind") == "sol_backend_receipt":
            actual_routes[str(copy.get("route"))] += 1
            if copy.get("route") == ROUTE:
                copy["route"] = "vdn_local_sol_all_selected_e"
        normalized.append(copy)
    result = dict(_ORIGINAL_VALIDATE_BACKEND(normalized, capture_id))
    expected = Counter(
        {
            ROUTE: 528,
            "vdn_dense_warmup": 22,
            "vdn_global_native": 50,
            "vdn_anchor_native": 100,
        }
    )
    route_ok = actual_routes == expected
    result["routes"] = dict(actual_routes)
    result["expected_700_backend_routes"] = bool(sum(actual_routes.values()) == 700 and route_ok)
    result["mapped_neighbor_route_exact"] = route_ok
    result["valid"] = bool(result.get("valid") and route_ok)
    return result


def _validate_mapped_routes(evidence) -> dict[str, Any]:
    routes = [item for item in evidence if isinstance(item, dict) and item.get("kind") == "mapped_neighbor_route"]
    identity = Counter((int(item.get("block_index", -1)), int(item.get("group_index", -1))) for item in routes)
    expected_identity = Counter((block, group) for block in range(2, 50) for group in range(11))
    identity_ok = identity == expected_identity and len(routes) == 528
    reports = []
    valid = identity_ok
    total_original = 0
    total_added = 0
    total_effective = 0
    total_pairs = 0
    max_increase = 0.0
    for item in sorted(
        routes,
        key=lambda value: (int(value.get("block_index", -1)), int(value.get("group_index", -1))),
    ):
        block = int(item.get("block_index", -1))
        group = int(item.get("group_index", -1))
        q_rows = int(item.get("q_rows", -1))
        kv_rows = int(item.get("kv_rows", -1))
        q_blocks = int(item.get("q_blocks", -1))
        k_blocks = int(item.get("k_blocks", -1))
        original = int(item.get("original_selected_pairs", -1))
        added = int(item.get("added_selected_pairs", -1))
        effective = int(item.get("effective_selected_pairs", -1))
        pairs = int(item.get("total_block_pairs", -1))
        intervals = item.get("mapped_neighbor_intervals")
        expected_q = e._EXPECTED_LOCAL_Q_ROWS[group] if 0 <= group < 11 else -1
        expected_kv = e._EXPECTED_WINDOW_KV_ROWS[group] if 0 <= group < 11 else -1
        interval_ok = bool(
            isinstance(intervals, list)
            and len(intervals) == q_blocks == (q_rows + 63) // 64
            and all(
                isinstance(pair, list)
                and len(pair) == 2
                and all(type(value) is int for value in pair)
                and 0 <= pair[0] < pair[1] <= k_blocks
                and pair[1] - pair[0] <= 4
                for pair in intervals
            )
        )
        increase = float(item.get("exact_work_increase_fraction", math.inf))
        item_ok = bool(
            2 <= block < 50
            and 0 <= group < 11
            and q_rows == expected_q
            and kv_rows == expected_kv
            and int(item.get("original_sink_rows", -1)) == e._EXPECTED_VIDEO_SPAN[0]
            and item.get("reason") == "mapped_neighbor"
            and item.get("additive_only") is True
            and item.get("restricted_domain_unchanged") is True
            and isinstance(item.get("descriptor_sha256"), str)
            and len(item["descriptor_sha256"]) == 64
            and isinstance(item.get("query_positions_sha256"), str)
            and len(item["query_positions_sha256"]) == 64
            and interval_ok
            and original > 0
            and added >= 0
            and effective == original + added
            and pairs > 0
            and effective <= pairs
            and math.isfinite(increase)
            and 0.0 <= increase <= 0.25
        )
        valid = valid and item_ok
        total_original += max(original, 0)
        total_added += max(added, 0)
        total_effective += max(effective, 0)
        total_pairs += max(pairs, 0)
        max_increase = max(max_increase, increase if math.isfinite(increase) else math.inf)
        reports.append(
            {
                "block_index": block,
                "group_index": group,
                "valid": item_ok,
                "original_selected_pairs": original,
                "added_selected_pairs": added,
                "effective_selected_pairs": effective,
                "total_block_pairs": pairs,
                "exact_work_increase_fraction": increase,
                "descriptor_sha256": item.get("descriptor_sha256"),
                "query_positions_sha256": item.get("query_positions_sha256"),
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
    valid = bool(valid and provenance_ok and total_added > 0)
    aggregate_increase = float(total_added) / float(total_original) if total_original else math.inf
    return {
        "identity_complete": identity_ok,
        "expected_route_records": 528,
        "route_records": len(routes),
        "reports": reports,
        "packaged_sparse_provenance": provenance[0].get("provenance") if len(provenance) == 1 else None,
        "packaged_sparse_provenance_valid": provenance_ok,
        "total_original_selected_pairs": total_original,
        "total_added_selected_pairs": total_added,
        "total_effective_selected_pairs": total_effective,
        "total_block_pairs": total_pairs,
        "aggregate_exact_work_increase_fraction": aggregate_increase,
        "maximum_per_call_exact_work_increase_fraction": max_increase,
        "additive_only": True,
        "restricted_domain_unchanged": True,
        "execution_valid": valid,
        # M is a policy arm, not an all-selected arithmetic arm.  Completion of
        # every mapped call implies the unchanged production all-selected gate
        # passed for each distinct layout; route evidence then proves only the
        # reviewed additive selection delta was applied.
        "arithmetic_conformant": valid,
        "production_sparse_conformant_to_independent_witness": None,
        "mapped_neighbor_policy_evidence_valid": valid,
    }


def _persist_m(capture_id, evidence, raw_media, pre_media, report):
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory()).resolve() / "h3_first_high_mapped_neighbor_m"
    except (ImportError, AttributeError, OSError) as exc:
        raise RuntimeError("mapped-neighbor M cannot resolve the ComfyUI output directory") from exc
    root.mkdir(parents=True, exist_ok=True)

    report["kind"] = "minimax_h3_first_high_mapped_neighbor_m_report"
    report["mode"] = MODE
    report["selected_arm"] = "M"
    report["decision_gate"] = (
        "invalid-m: fix only the demonstrated M diagnostic defect before another M call"
        if not report.get("execution_valid")
        else "valid-m: decode and compare complete M media against preserved R/W/E; production promotion remains gated"
    )
    sanitized = e._sanitize_evidence(evidence.items)
    payload = {
        "schema_version": e.SCHEMA_VERSION,
        "capture_id": capture_id,
        "design_commit": e.DESIGN_COMMIT,
        "mode": MODE,
        "evidence": sanitized,
        "first_high_model_raw_video": raw_media.detach().to(device="cpu", copy=True),
        "first_high_pre_guidance_video": pre_media.detach().to(device="cpu", copy=True),
    }
    tensor_inventory = e._public_tensor_inventory(payload)
    stem = f"{capture_id}-mapped-neighbor-m"
    tensor_path = root / f"{stem}.pt"
    json_path = root / f"{stem}.json"
    e._atomic_write_bytes(tensor_path, lambda path: torch.save(payload, path))
    tensor_file_sha = e._sha256_file(tensor_path)
    public_report = dict(report)
    public_report["durable_evidence"] = {
        "tensor_path": str(tensor_path),
        "tensor_file_sha256": tensor_file_sha,
        "tensor_inventory": tensor_inventory,
        "json_path": str(json_path),
    }
    encoded = json.dumps(public_report, indent=2, sort_keys=True, default=str).encode("utf-8")
    e._atomic_write_bytes(json_path, lambda path: path.write_bytes(encoded))
    return {
        "tensor_path": str(tensor_path),
        "tensor_file_sha256": tensor_file_sha,
        "json_path": str(json_path),
        "json_file_sha256": e._sha256_file(json_path),
        "tensor_inventory": tensor_inventory,
    }


e._validate_vdn_receipts = _validate_vdn_m
e._validate_backend_receipts = _validate_backend_m
e._validate_witnesses = _validate_mapped_routes
e._validate_sol_counter_isolation = _ORIGINAL_VALIDATE_SOL_COUNTERS
e._persist_evidence = _persist_m

# Preserve the existing workflow wiring/type IDs so the user can replace only the
# Patcher overlays.  The visible node names make it explicit that the stacked PR
# is executing M rather than replaying all-selected E.
e.H3FirstHighSolLocalDiagnostic.DESCRIPTION = (
    "Arm M: replay the preserved R first-high state once and add only VDN-mapped physical local neighbors to the "
    "existing Sol exact route. Restricted K/V support, complement, threshold, sink and schedule remain unchanged."
)
e.H3FirstHighSolLocalDiagnosticReport.DESCRIPTION = (
    "Emit the bounded mapped-neighbor M validity/route-work report and decode-ready first-high raw/pre-guidance media."
)
e.NODE_DISPLAY_NAME_MAPPINGS["H3FirstHighSolLocalDiagnostic"] = "MiniMax H3 First-High Mapped-Neighbor Diagnostic M"
e.NODE_DISPLAY_NAME_MAPPINGS["H3FirstHighSolLocalDiagnosticReport"] = "MiniMax H3 First-High Mapped-Neighbor M Report"

__all__ = ["MODE", "ROUTE"]
