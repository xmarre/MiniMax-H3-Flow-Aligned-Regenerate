from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from h3_flow_regenerate import production_mapped_neighbor_candidate as candidate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "h3_flow_regenerate" / "production_mapped_neighbor_candidate_source_delta.json"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _mapped_fields(group_index: int, q_rows: int, kv_rows: int) -> tuple[object, ...]:
    return (
        "vdn_mapped_neighbor_v1",
        "vdn-owner-test",
        "a" * 64,
        group_index,
        q_rows,
        kv_rows,
        3101,
        "b" * 64,
        "c" * 64,
        "k64-union-radius1-additive-interval4-v1",
        "sana-sol-engine-sol-attn-64-rect-sm120-mapped-neighbor-v4",
        True,
    )


def _valid_receipts() -> candidate._ReceiptSink:
    sink = candidate._ReceiptSink()
    for block in range(50):
        if block < 2:
            for _group_index in range(11):
                sink.append(("sol_h3", block, "vdn_dense_warmup"))
        else:
            for group_index, (q_rows, kv_rows) in enumerate(
                zip(candidate._EXPECTED_LOCAL_Q_ROWS, candidate._EXPECTED_WINDOW_KV_ROWS, strict=True)
            ):
                sink.append(
                    (
                        "sol_h3",
                        block,
                        "vdn_local_sol_mapped_v1",
                        _mapped_fields(group_index, q_rows, kv_rows),
                    )
                )
        sink.append(("sol_h3", block, "vdn_global_native"))
        sink.append(("sol_h3", block, "vdn_anchor_native"))
        sink.append(("sol_h3", block, "vdn_anchor_native"))
    return sink


def _module_file(root: Path, module: str) -> Path:
    path = root.joinpath(*module.split("."))
    package = path / "__init__.py"
    return package if package.is_file() else path.with_suffix(".py")


def _install_fake_patcher_extension(monkeypatch):
    comfy = types.ModuleType("comfy")
    patcher_extension = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        OUTER_SAMPLE = "outer_sample"
        SAMPLER_SAMPLE = "sampler_sample"

    patcher_extension.WrappersMP = WrappersMP
    comfy.patcher_extension = patcher_extension
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", patcher_extension)
    return WrappersMP


def _wrapper_manifest_entry(key: str, function) -> dict[str, object]:
    return {"key": key, "callable": candidate._callable_equivalence_identity(function)}


def test_candidate_identity_is_distinct_and_bound_to_authoritative_design():
    assert candidate.SCHEMA_VERSION == 1
    assert candidate.MODE == "production_mapped_neighbor_v4_candidate"
    assert candidate.DESIGN_COMMIT == "941b8571b099d18d1fe4e4deef9f04fd343b1098"
    assert candidate.EXPECTED_CAPTURE_ID == "234ed062128e43ed8d5ec63e27517b22"
    assert candidate.STATE_KEY not in {
        "h3_flow_first_high_operator_comparison_v1",
        "h3_flow_first_high_sol_local_e_v1",
        "h3_flow_first_high_mapped_neighbor_m_v1",
    }


def test_candidate_module_does_not_import_w_e_or_m_operator_modules():
    tree = ast.parse(Path(candidate.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = {
        "first_high_operator_comparison",
        "first_high_sol_local_diagnostic",
        "first_high_mapped_neighbor_diagnostic",
        "first_high_operator_diagnostic",
        "first_high_sol_local_receipt_tap",
        "first_high_sol_local_witness_bridge",
    }
    assert not any(any(name.endswith(suffix) for suffix in forbidden) for name in imported)


def test_valid_backend_receipts_require_exact_production_topology():
    sink = _valid_receipts()
    report = candidate._validate_backend_receipts(sink)

    assert len(sink.items) == 700
    assert report["routes"] == {
        "vdn_dense_warmup": 22,
        "vdn_global_native": 50,
        "vdn_anchor_native": 100,
        "vdn_local_sol_mapped_v1": 528,
    }
    assert report["mapped_receipt_count"] == 528
    assert report["per_block_14_calls_exact"] is True
    assert report["per_block_route_topology_exact"] is True
    assert report["mapped_fields_valid"] is True
    assert report["mapped_geometry_exact"] is True
    assert report["mapped_identity_consistent_by_group"] is True
    assert report["warmup_only_first_two_blocks"] is True
    assert report["no_native_local_fallback"] is True
    assert report["valid"] is True


def test_receipt_gate_rejects_native_local_fallback_and_old_kernel_contract():
    sink = _valid_receipts()
    index = next(i for i, item in enumerate(sink.items) if item[2] == "vdn_local_sol_mapped_v1")
    block = sink.items[index][1]
    sink.items[index] = ("sol_h3", block, "vdn_local_native_mapping:owner")
    report = candidate._validate_backend_receipts(sink)
    assert report["no_native_local_fallback"] is False
    assert report["valid"] is False

    sink = _valid_receipts()
    index = next(i for i, item in enumerate(sink.items) if item[2] == "vdn_local_sol_mapped_v1")
    owner, block, route, fields = sink.items[index]
    mutated = (*fields[:10], "sana-sol-engine-sol-attn-64-rect-sm120-v3", fields[11])
    sink.items[index] = (owner, block, route, mutated)
    report = candidate._validate_backend_receipts(sink)
    assert report["mapped_fields_valid"] is False
    assert report["valid"] is False


def test_receipt_gate_rejects_per_block_route_swaps_with_same_aggregate_counts():
    sink = _valid_receipts()
    block0_global = next(
        i for i, item in enumerate(sink.items) if item[1] == 0 and item[2] == "vdn_global_native"
    )
    block1_anchor = next(
        i for i, item in enumerate(sink.items) if item[1] == 1 and item[2] == "vdn_anchor_native"
    )
    sink.items[block0_global] = ("sol_h3", 0, "vdn_anchor_native")
    sink.items[block1_anchor] = ("sol_h3", 1, "vdn_global_native")

    report = candidate._validate_backend_receipts(sink)
    assert report["routes"] == {
        "vdn_dense_warmup": 22,
        "vdn_global_native": 50,
        "vdn_anchor_native": 100,
        "vdn_local_sol_mapped_v1": 528,
    }
    assert report["per_block_14_calls_exact"] is True
    assert report["per_block_route_topology_exact"] is False
    assert report["valid"] is False


def test_receipt_gate_rejects_group_mapping_identity_drift_between_blocks():
    sink = _valid_receipts()
    index = next(
        i
        for i, item in enumerate(sink.items)
        if item[1] == 2 and item[2] == "vdn_local_sol_mapped_v1" and item[3][3] == 0
    )
    owner, block, route, fields = sink.items[index]
    mutated = (*fields[:7], "d" * 64, *fields[8:])
    sink.items[index] = (owner, block, route, mutated)

    report = candidate._validate_backend_receipts(sink)
    assert report["mapped_fields_valid"] is True
    assert report["mapped_geometry_exact"] is True
    assert report["mapped_identity_consistent_by_group"] is False
    assert report["valid"] is False


def test_receipt_gate_has_exact_50_block_14_subcall_distribution():
    sink = _valid_receipts()
    blocks = Counter(item[1] for item in sink.items)
    assert blocks == Counter({block: 14 for block in range(50)})


def test_approved_callable_normalization_requires_exact_source_identity(tmp_path):
    path = tmp_path / "runtime.py"
    path.write_text("pass\n", encoding="utf-8")
    gate = {
        "entries": [
            {
                "owner": "sol",
                "module": "sol_h3.runtime",
                "path": str(path.resolve()),
                "candidate_git_blob_sha": _git_blob_sha(path),
                "capture_sha256": "1" * 64,
                "sha256": "2" * 64,
            }
        ]
    }
    capture = {
        "module": "sol_h3.runtime",
        "qualname": "BlockPatch",
        "code_digest": "old",
        "file": {"resolved_path": str(path.resolve()), "sha256": "1" * 64},
    }
    current = {
        "module": "sol_h3.runtime",
        "qualname": "BlockPatch",
        "code_digest": "new",
        "file": {"resolved_path": str(path.resolve()), "sha256": "2" * 64},
    }
    capture_normalized = candidate._normalize_approved_callables(capture, gate, side="capture")
    current_normalized = candidate._normalize_approved_callables(current, gate, side="current")
    assert capture_normalized == current_normalized

    current["file"]["sha256"] = "3" * 64
    with pytest.raises(RuntimeError, match="current callable source identity differs"):
        candidate._normalize_approved_callables(current, gate, side="current")


def test_validation_wrapper_normalization_proves_live_ownership_before_removal(monkeypatch):
    wrappers = _install_fake_patcher_extension(monkeypatch)
    keep = {"key": "production-wrapper", "callable": {"module": "production"}}
    current = {
        "active_wrapper_order": {
            wrappers.OUTER_SAMPLE: [
                keep,
                _wrapper_manifest_entry(candidate._OUTER_KEY, candidate._outer_wrapper),
            ],
            wrappers.SAMPLER_SAMPLE: [
                _wrapper_manifest_entry(candidate._SAMPLER_KEY, candidate._sampler_entry_wrapper)
            ],
        },
        "patcher_wrapper_order": {
            wrappers.OUTER_SAMPLE: [
                keep,
                _wrapper_manifest_entry(candidate._OUTER_KEY, candidate._outer_wrapper),
            ],
            wrappers.SAMPLER_SAMPLE: [
                _wrapper_manifest_entry(candidate._SAMPLER_KEY, candidate._sampler_entry_wrapper)
            ],
        },
    }
    live = {
        wrappers.OUTER_SAMPLE: {candidate._OUTER_KEY: [candidate._outer_wrapper]},
        wrappers.SAMPLER_SAMPLE: {candidate._SAMPLER_KEY: [candidate._sampler_entry_wrapper]},
    }
    guider = types.SimpleNamespace(model_patcher=types.SimpleNamespace(wrappers=live))

    normalized = candidate._normalize_validation_wrapper_order(current, guider)
    for field in ("active_wrapper_order", "patcher_wrapper_order"):
        assert normalized[field][wrappers.OUTER_SAMPLE] == [keep]
        assert normalized[field][wrappers.SAMPLER_SAMPLE] == []

    live[wrappers.SAMPLER_SAMPLE][candidate._SAMPLER_KEY] = [lambda: None]
    with pytest.raises(RuntimeError, match="live wrapper callable identity changed"):
        candidate._normalize_validation_wrapper_order(current, guider)


def test_source_delta_manifest_is_exactly_the_reviewed_production_runtime_delta():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["design_commit"] == candidate.DESIGN_COMMIT
    assert manifest["flow_base"] == "4ae2e35f77ed961151ab5695bef9e1dbe277cc54"
    assert manifest["sol_pr"] == 14
    assert manifest["sol_pr_head"] == "8b049e39d000b0d283f2f01e8477f74cf5c2d608"
    assert manifest["vdn_pr"] == 18
    assert manifest["vdn_pr_head"] == "333d63f81d33fe29dc1f1f637c5f4a4396880f99"

    keys = {(item["owner"], item["module"], item["relative_path"]) for item in manifest["entries"]}
    expected = {
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
    assert keys == expected
    assert frozenset(expected) == candidate._EXPECTED_SOURCE_ENTRY_KEYS
    assert len(keys) == len(manifest["entries"])


def test_flow_source_delta_blob_identities_match_current_candidate_tree():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        if entry["owner"] != "flow":
            continue
        module_file = _module_file(ROOT, entry["module"])
        path = (
            module_file
            if entry["relative_path"] == "."
            else (module_file.parent / entry["relative_path"]).resolve()
        )
        assert path.is_file()
        assert _git_blob_sha(path) == entry["candidate_git_blob_sha"]


@pytest.mark.skipif(
    not os.environ.get("SOL_PATH") or not os.environ.get("VDN_PATH"),
    reason="paired production sources not supplied",
)
def test_paired_sol_vdn_source_delta_blob_identities_match_pinned_pr_heads():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    roots = {"sol": Path(os.environ["SOL_PATH"]), "vdn": Path(os.environ["VDN_PATH"])}
    for entry in manifest["entries"]:
        owner = entry["owner"]
        if owner not in roots:
            continue
        module_file = _module_file(roots[owner], entry["module"])
        path = (
            module_file
            if entry["relative_path"] == "."
            else (module_file.parent / entry["relative_path"]).resolve()
        )
        assert path.is_file(), path
        assert _git_blob_sha(path) == entry["candidate_git_blob_sha"], path


def test_root_registration_exposes_only_new_production_validation_nodes():
    root_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_NODE_CLASS_MAPPINGS" in root_source
    assert "PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_NODE_DISPLAY_NAME_MAPPINGS" in root_source
    assert "first_high_mapped_neighbor_diagnostic" not in root_source
    assert "first_high_sol_local_diagnostic" not in root_source
    assert "first_high_operator_comparison" not in root_source
