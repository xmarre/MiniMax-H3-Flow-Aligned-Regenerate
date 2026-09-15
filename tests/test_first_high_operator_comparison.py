from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from h3_flow_regenerate import first_high_operator_comparison as w


def _request_state(*, high_sigmas=None, mode="native_window"):
    return SimpleNamespace(
        replay=SimpleNamespace(
            manifest={
                "capture_id": "capture-w",
                "high_sigmas": list(w._REQUIRED_SUFFIX if high_sigmas is None else high_sigmas),
                "target_shapes": [[1, 24, 52, 64, 64], [1, 32, 2, 292]],
            }
        ),
        mode=mode,
        source_gate={"manifest_digest": "a" * 64},
    )


def _source_gate(path="/custom_nodes/flow/h3_flow_regenerate/first_high_operator_comparison.py"):
    return {
        "entries": [
            {
                "owner": "flow",
                "module": "h3_flow_regenerate.first_high_operator_comparison",
                "relative_path": ".",
                "path": path,
                "base_git_blob_sha": None,
                "candidate_git_blob_sha": "b" * 40,
            }
        ]
    }


def _wrapper_entry(key, qualname, path):
    return {
        "key": key,
        "callable": {
            "module": "h3_flow_regenerate.first_high_operator_comparison",
            "qualname": qualname,
            "file": {"resolved_path": path},
        },
    }


def _receipts(mode: str):
    local_route = "vdn_local_native_window_w" if mode == "native_window" else "vdn_local_native_full_w"
    local_rows = [4655] * 10 + [4650]
    result = []
    for block in range(50):
        common = {
            "block": block,
            "packed_rows": 56349,
            "video_start": 3101,
            "video_end": 56349,
            "gate_fingerprint": "gate",
            "adapter_fingerprint": "adapter",
        }
        result.append(
            {
                **common,
                "kind": "global",
                "group_index": None,
                "q_rows": 3101,
                "kv_rows": 56349,
                "support_mode": "full",
                "complement_executed": False,
                "provider_route": "vdn_global_native",
                "pre_attention_qkv_digest": "c" * 64 if block == 0 else None,
            }
        )
        for group_index, q_rows in enumerate(local_rows):
            result.append(
                {
                    **common,
                    "kind": "local",
                    "group_index": group_index,
                    "q_rows": q_rows,
                    "kv_rows": 56349 if mode == "native_full_support" else 13341,
                    "support_mode": "canonical_full" if mode == "native_full_support" else "restricted_window",
                    "complement_executed": mode == "native_window",
                    "provider_route": local_route,
                    "canonical_full_kv": mode == "native_full_support",
                }
            )
        for anchor_index in range(2):
            result.append(
                {
                    **common,
                    "kind": "anchor",
                    "group_index": anchor_index,
                    "q_rows": 1024,
                    "kv_rows": 56349,
                    "support_mode": "full",
                    "complement_executed": False,
                    "provider_route": "vdn_anchor_native",
                }
            )
    return result


def test_request_preserves_complete_r_suffix_and_exact_schema():
    request = w._request_tuple(_request_state())
    assert tuple(name for name, _value in request) == (
        "api",
        "capture_id",
        "mode",
        "stage",
        "logical_call_limit",
        "sigma",
        "target_shapes_digest",
        "source_contract_digest",
    )
    assert dict(request)["sigma"] == w._REQUIRED_SUFFIX[0]
    assert dict(request)["logical_call_limit"] == 1

    changed = list(w._REQUIRED_SUFFIX)
    changed[-1] = 0.1
    with pytest.raises(RuntimeError, match="original high suffix"):
        w._request_tuple(_request_state(high_sigmas=changed))


def test_provenance_normalization_removes_only_exact_reviewed_w_wrappers():
    gate = _source_gate()
    source_path = gate["entries"][0]["path"]
    production = {
        "key": "h3_flow_regenerate.first_high_operator.production_like_name",
        "callable": {"qualname": "production", "file": {"resolved_path": "/production.py"}},
    }
    identity = {
        "active_wrapper_order": {
            "outer_sample": [
                _wrapper_entry(w._OUTER_KEY, "_outer_wrapper", source_path),
                production,
            ],
            "sampler_sample": [_wrapper_entry(w._SAMPLER_KEY, "_sampler_entry_wrapper", source_path)],
        },
        "patcher_wrapper_order": {
            "outer_sample": [_wrapper_entry(w._OUTER_KEY, "_outer_wrapper", source_path)],
            "sampler_sample": [_wrapper_entry(w._SAMPLER_KEY, "_sampler_entry_wrapper", source_path)],
        },
    }
    original = copy.deepcopy(identity)

    normalized = w._without_w_diagnostic_wrappers(identity, gate)

    assert normalized["active_wrapper_order"]["outer_sample"] == [production]
    assert normalized["active_wrapper_order"]["sampler_sample"] == []
    assert normalized["patcher_wrapper_order"]["outer_sample"] == []
    assert normalized["patcher_wrapper_order"]["sampler_sample"] == []
    assert identity == original


def test_provenance_normalization_rejects_foreign_or_duplicate_w_wrapper_identity():
    gate = _source_gate()
    source_path = gate["entries"][0]["path"]
    foreign = {
        "active_wrapper_order": {
            "outer_sample": [_wrapper_entry(w._OUTER_KEY, "_outer_wrapper", "/foreign.py")],
            "sampler_sample": [_wrapper_entry(w._SAMPLER_KEY, "_sampler_entry_wrapper", source_path)],
        }
    }
    with pytest.raises(RuntimeError, match="missing diagnostic wrapper identity"):
        w._without_w_diagnostic_wrappers(foreign, gate)

    duplicate_entry = _wrapper_entry(w._OUTER_KEY, "_outer_wrapper", source_path)
    duplicate = {
        "active_wrapper_order": {
            "outer_sample": [duplicate_entry, copy.deepcopy(duplicate_entry)],
            "sampler_sample": [_wrapper_entry(w._SAMPLER_KEY, "_sampler_entry_wrapper", source_path)],
        }
    }
    with pytest.raises(RuntimeError, match="installed more than once"):
        w._without_w_diagnostic_wrappers(duplicate, gate)


def test_source_delta_allowance_is_exact_path_changed_companion_only():
    changed_path = "/custom_nodes/sol_h3/sol_h3/__init__.py"
    unchanged_core = "/ComfyUI/comfy/model_sampling.py"
    gate = {
        "entries": [
            {
                "path": changed_path,
                "base_git_blob_sha": "1" * 40,
                "candidate_git_blob_sha": "2" * 40,
            },
            {
                "path": unchanged_core,
                "base_git_blob_sha": "3" * 40,
                "candidate_git_blob_sha": "3" * 40,
            },
        ]
    }
    assert w._allowed_provenance_difference(
        f"$.loaded_companion_sources.sol_h3.shared_source[{changed_path}].sha256",
        gate,
    )
    assert not w._allowed_provenance_difference(
        "$.loaded_companion_sources.vdn_h3.shared_source[/other/__init__.py].sha256",
        gate,
    )
    assert not w._allowed_provenance_difference(
        f"$.loaded_companion_sources.core.shared_source[{unchanged_core}].sha256",
        gate,
    )
    assert not w._allowed_provenance_difference("$.active_wrapper_order.outer_sample.length (8 != 9)", gate)
    assert not w._allowed_provenance_difference("$.something.first_high_operator_unrelated", gate)


@pytest.mark.parametrize("mode", ["native_window", "native_full_support"])
def test_receipt_validation_requires_exact_700_subcall_geometry(mode):
    report = w._validate_receipts(_receipts(mode), mode)
    assert report["expected_700_subcalls"] is True
    assert report["expected_local_route"] is True
    assert report["support_ok"] is True
    assert report["complement_ok"] is True
    assert report["canonical_or_window_kv_ok"] is True
    assert report["nonlocal_full_support_ok"] is True
    assert report["geometry_ok"] is True
    assert report["per_block_topology_ok"] is True
    assert report["first_block_pre_attention_qkv_digest_present"] is True


def test_receipt_validation_rejects_wrong_geometry_block_partition_and_route():
    receipts = _receipts("native_window")
    wrong_geometry = copy.deepcopy(receipts)
    wrong_geometry[0]["packed_rows"] = 56348
    assert w._validate_receipts(wrong_geometry, "native_window")["geometry_ok"] is False

    wrong_partition = copy.deepcopy(receipts)
    first_local = next(item for item in wrong_partition if item["block"] == 0 and item["kind"] == "local")
    first_local["group_index"] = 10
    assert w._validate_receipts(wrong_partition, "native_window")["per_block_topology_ok"] is False

    wrong_route = copy.deepcopy(receipts)
    first_local = next(item for item in wrong_route if item["kind"] == "local")
    first_local["provider_route"] = "vdn_local_sol"
    assert w._validate_receipts(wrong_route, "native_window")["expected_local_route"] is False


def test_provenance_gate_ignores_exact_w_wrappers_but_not_similar_wrapper(monkeypatch):
    gate = _source_gate()
    source_path = gate["entries"][0]["path"]
    capture = {
        "active_wrapper_order": {"outer_sample": [{"key": "production"}], "sampler_sample": []},
        "loaded_companion_sources": {},
    }
    current = {
        "active_wrapper_order": {
            "outer_sample": [
                _wrapper_entry(w._OUTER_KEY, "_outer_wrapper", source_path),
                {"key": "production"},
            ],
            "sampler_sample": [_wrapper_entry(w._SAMPLER_KEY, "_sampler_entry_wrapper", source_path)],
        },
        "loaded_companion_sources": {},
    }
    monkeypatch.setattr(w._replay, "_bundle_provenance_equivalence_identity", lambda _manifest: capture)
    monkeypatch.setattr(w._replay, "_provenance_equivalence_identity", lambda _manifest: current)
    state = SimpleNamespace(replay=SimpleNamespace(manifest={}), source_gate=gate)
    record = SimpleNamespace(state=SimpleNamespace(manifest={}))

    report = w._provenance_gate(state, record)
    assert report["differences"] == []
    assert report["exact_except_reviewed_w_delta"] is True

    current["active_wrapper_order"]["outer_sample"].append({"key": "first_high_operator_production"})
    report = w._provenance_gate(state, record)
    assert report["exact_except_reviewed_w_delta"] is False
    assert any("active_wrapper_order" in item for item in report["unexpected_differences"])
