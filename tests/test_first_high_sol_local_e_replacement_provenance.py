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
    current[key0]["closure"]["previous"]["callable"]["closure"] = {
        "state": {"type": "vdn_h3.runtime.State"}
    }

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
