import hashlib
import json
import os
from pathlib import Path

path = Path("h3_flow_regenerate/first_high_sol_local_diagnostic.py")
text = path.read_text(encoding="utf-8")

text = text.replace("import contextvars\n", "import contextlib\nimport contextvars\n", 1)
text = text.replace(
    '    "first_high_h3_input_video": "b49f17a317530795be43e8ea318b3b242442e8c9a8fe368185131e6f5666672936" if False else "b49f17a317530795be43bc775486777b07d5b5dbb28996819033cdede64195c0",\n',
    '    "first_high_h3_input_video": "b49f17a317530795be43bc775486777b07d5b5dbb28996819033cdede64195c0",\n',
    1,
)

anchor = "_ARITH_CATASTROPHIC_REFERENCE_PEAK_MULTIPLIER = 4.0\n"
required = r'''_REQUIRED_SOURCE_ENTRY_KEYS = frozenset(
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
'''
if required not in text:
    text = text.replace(anchor, anchor + required, 1)

old = '''    keys = [_source_entry_key(item) for item in entries if isinstance(item, dict)]
    if len(keys) != len(entries) or len(keys) != len(set(keys)):
        raise RuntimeError("first-high Sol-local E source-delta manifest has invalid or duplicate entries")
    return manifest, _sha_json(manifest)
'''
new = '''    keys = [_source_entry_key(item) for item in entries if isinstance(item, dict)]
    if len(keys) != len(entries) or len(keys) != len(set(keys)):
        raise RuntimeError("first-high Sol-local E source-delta manifest has invalid or duplicate entries")
    if frozenset(keys) != _REQUIRED_SOURCE_ENTRY_KEYS:
        raise RuntimeError("first-high Sol-local E source-delta manifest differs from the reviewed exact entry set")
    return manifest, _sha_json(manifest)
'''
if old not in text:
    raise SystemExit("source manifest validation anchor not found")
text = text.replace(old, new, 1)

start = text.index("def _allowed_provenance_difference(")
end = text.index("\ndef _provenance_gate(", start)
replacement = r'''def _allowed_provenance_difference(path: str, source_gate: dict[str, Any]) -> bool:
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
'''
text = text[:start] + replacement + text[end:]

backend_anchor = "def _metric_gate(metrics: dict[str, Any]) -> bool:\n"
insert = r'''def _validate_sol_counter_isolation(record: _diag._Record) -> dict[str, Any]:
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
    zero_fields_present = bool(
        isinstance(sol_after, dict) and all(field in sol_after for field in zero_fields)
    )
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


'''
if insert not in text:
    text = text.replace(backend_anchor, insert + backend_anchor, 1)

start = text.index("def _validate_witnesses(")
end = text.index("\ndef _sanitize_evidence(", start)
replacement = r'''def _validate_witnesses(evidence: _Sink) -> dict[str, Any]:
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
            item.get("route_trace_matches_independent") is True
            and int(item.get("route_mismatch_count", -1)) == 0
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
        complete
        and provenance_ok
        and input_integrity
        and debug_conformant
        and all_selected_trace_conformant
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
'''
text = text[:start] + replacement + text[end:]

old = '''    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
'''
new = '''    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
'''
if old not in text:
    raise SystemExit("atomic cleanup anchor not found")
text = text.replace(old, new, 1)

old = '''            vdn_receipts = _validate_vdn_receipts(receipt_sink)
            backend_receipts = _validate_backend_receipts(evidence_sink, str(replay.manifest["capture_id"]))
            witnesses = _validate_witnesses(evidence_sink)
            execution_valid = bool(
                topology == expected_topology
                and entry_exact
                and vdn_receipts["valid"]
                and backend_receipts["valid"]
                and witnesses["execution_valid"]
'''
new = '''            vdn_receipts = _validate_vdn_receipts(receipt_sink)
            backend_receipts = _validate_backend_receipts(evidence_sink, str(replay.manifest["capture_id"]))
            witnesses = _validate_witnesses(evidence_sink)
            sol_counter_isolation = _validate_sol_counter_isolation(record)
            execution_valid = bool(
                topology == expected_topology
                and entry_exact
                and vdn_receipts["valid"]
                and backend_receipts["valid"]
                and witnesses["execution_valid"]
                and sol_counter_isolation["valid"]
'''
if old not in text:
    raise SystemExit("final validation anchor not found")
text = text.replace(old, new, 1)

old = '''                "backend_receipts": backend_receipts,
                "operator_witnesses": witnesses,
                "export_contract": export_contract,
'''
new = '''                "backend_receipts": backend_receipts,
                "operator_witnesses": witnesses,
                "ordinary_sol_counter_isolation": sol_counter_isolation,
                "export_contract": export_contract,
'''
if old not in text:
    raise SystemExit("report counter anchor not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

# Pin the already-checkpointed Sol E implementation byte identity.  The Flow file
# itself is finalized after Ruff formatting by the workflow.
manifest_path = Path("h3_flow_regenerate/first_high_sol_local_e_source_delta.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for entry in manifest["entries"]:
    if entry.get("owner") == "sol" and entry.get("module") == "sol_h3.first_high_sol_local_diagnostic":
        entry["candidate_git_blob_sha"] = "30020cf98796549c3ec53004ebb0e4d0421ae908"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
