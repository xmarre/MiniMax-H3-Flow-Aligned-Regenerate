# ruff: noqa: I001
from __future__ import annotations

import copy
import importlib

import pytest


root = importlib.import_module("__init__")


def _chains():
    capture = {}
    current = {}
    for index in range(50):
        key = str(("double_block", index))
        base = {
            "module": "comfy.model_patcher",
            "qualname": "replacement",
            "file": {"resolved_path": "/runtime/model_patcher.py", "sha256": "a" * 64},
        }
        capture[key] = copy.deepcopy(base)
        current[key] = copy.deepcopy(base)

    key0 = str(("double_block", 0))
    capture[key0]["closure"] = {
        "previous": {
            "callable": {
                "module": "vdn_h3.hybrid",
                "qualname": "vdn_forward",
            }
        }
    }
    current[key0] = copy.deepcopy(capture[key0])
    current[key0]["closure"]["previous"]["callable"]["closure"] = {"state": {"type": "vdn_h3.runtime.State"}}

    key1 = str(("double_block", 1))
    current[key1]["closure"] = {
        "underlying": {
            "callable": {
                "module": "vdn_h3.first_high_sol_local_diagnostic",
                "qualname": "wrapped",
            }
        }
    }
    return capture, current


def _source_record(path, module, sha256="b" * 64):
    return {
        "module": module,
        "file": {
            "resolved_path": str(path),
            "sha256": sha256,
        },
    }


def _lazy_sparse_fixture(tmp_path, monkeypatch):
    package = tmp_path / "sol_h3"
    package.mkdir()
    sol_init = package / "__init__.py"
    sparse = package / "sparse.py"
    other = package / "other.py"
    sol_init.write_text("# sol init\n", encoding="utf-8")
    sparse.write_text("# sparse\n", encoding="utf-8")
    other.write_text("# other\n", encoding="utf-8")
    expected_sha256 = "b" * 64
    monkeypatch.setattr(
        root._FIRST_HIGH_SOL_LOCAL_MODULE,
        "_git_blob_sha",
        lambda path: root._E_LAZY_SOL_SPARSE_BLOB if path == sparse else "0" * 40,
    )
    monkeypatch.setattr(
        root._FIRST_HIGH_SOL_LOCAL_MODULE,
        "_sha256_file",
        lambda path: expected_sha256 if path == sparse else "c" * 64,
    )
    source_gate = {
        "entries": [
            {
                "owner": "sol",
                "module": "sol_h3",
                "relative_path": ".",
                "path": str(sol_init),
            }
        ]
    }
    capture = {
        "loaded_companion_sources": {
            "sol_h3": [_source_record(sol_init, "generated.sol_h3", "c" * 64)],
        }
    }
    current = copy.deepcopy(capture)
    current["loaded_companion_sources"]["sol_h3"].append(
        _source_record(sparse, "generated.sol_h3.sparse", expected_sha256)
    )
    return capture, current, source_gate, sparse, other


def test_e_replacement_chain_accepts_only_current_only_closure_depth_expansion():
    capture_chain, current_chain = _chains()
    normalized = root._normalize_e_replacement_chain(
        {"replacement_chain_dit": current_chain},
        {"replacement_chain_dit": capture_chain},
    )
    assert normalized["replacement_chain_dit"] == capture_chain


def test_e_replacement_chain_rejects_non_closure_difference():
    capture_chain, current_chain = _chains()
    current_chain[str(("double_block", 7))]["module"] = "unexpected.module"
    with pytest.raises(RuntimeError, match="differs from R"):
        root._normalize_e_replacement_chain(
            {"replacement_chain_dit": current_chain},
            {"replacement_chain_dit": capture_chain},
        )


def test_e_replacement_chain_rejects_change_inside_capture_owned_closure():
    capture_chain, current_chain = _chains()
    key = str(("double_block", 0))
    current_chain[key]["closure"]["previous"]["callable"]["module"] = "unexpected.module"
    with pytest.raises(RuntimeError, match="differs from R"):
        root._normalize_e_replacement_chain(
            {"replacement_chain_dit": current_chain},
            {"replacement_chain_dit": capture_chain},
        )


def test_e_replacement_chain_rejects_key_set_change():
    capture_chain, current_chain = _chains()
    current_chain["unexpected"] = {"module": "x"}
    with pytest.raises(RuntimeError, match="key set differs"):
        root._normalize_e_replacement_chain(
            {"replacement_chain_dit": current_chain},
            {"replacement_chain_dit": capture_chain},
        )


def test_e_lazy_sparse_normalizes_only_exact_reviewed_self_import(tmp_path, monkeypatch):
    capture, current, source_gate, sparse, _other = _lazy_sparse_fixture(tmp_path, monkeypatch)
    normalized = root._normalize_e_lazy_sol_sparse_source(current, capture, source_gate)
    normalized_map, problems = root._FIRST_HIGH_SOL_LOCAL_MODULE._replay._companion_source_map(
        normalized["loaded_companion_sources"]["sol_h3"]
    )
    assert problems == []
    assert str(sparse) not in normalized_map
    assert normalized["loaded_companion_sources"]["sol_h3"] == capture["loaded_companion_sources"]["sol_h3"]


def test_e_lazy_sparse_rejects_unreviewed_sparse_bytes(tmp_path, monkeypatch):
    capture, current, source_gate, _sparse, _other = _lazy_sparse_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(root._FIRST_HIGH_SOL_LOCAL_MODULE, "_git_blob_sha", lambda _path: "0" * 40)
    with pytest.raises(RuntimeError, match="reviewed PR #11/#12 bytes"):
        root._normalize_e_lazy_sol_sparse_source(current, capture, source_gate)


def test_e_lazy_sparse_rejects_snapshot_sha_mismatch(tmp_path, monkeypatch):
    capture, current, source_gate, sparse, _other = _lazy_sparse_fixture(tmp_path, monkeypatch)
    current["loaded_companion_sources"]["sol_h3"][-1]["file"]["sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="provenance SHA-256"):
        root._normalize_e_lazy_sol_sparse_source(current, capture, source_gate)
    assert sparse.exists()


def test_e_lazy_sparse_leaves_unrelated_replay_only_source_visible(tmp_path, monkeypatch):
    capture, current, source_gate, sparse, other = _lazy_sparse_fixture(tmp_path, monkeypatch)
    current["loaded_companion_sources"]["sol_h3"].append(_source_record(other, "generated.sol_h3.other", "e" * 64))
    normalized = root._normalize_e_lazy_sol_sparse_source(current, capture, source_gate)
    current_map, problems = root._FIRST_HIGH_SOL_LOCAL_MODULE._replay._companion_source_map(
        normalized["loaded_companion_sources"]["sol_h3"]
    )
    assert problems == []
    assert str(sparse) not in current_map
    assert str(other) in current_map
