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


def _vdn_source_gate(path="/custom_nodes/vdn/vdn_h3/first_high_operator_diagnostic.py"):
    return {
        "entries": [
            {
                "owner": "vdn",
                "module": "vdn_h3.first_high_operator_diagnostic",
                "relative_path": ".",
                "path": path,
                "base_git_blob_sha": None,
                "base_sha256": None,
                "candidate_git_blob_sha": "b" * 40,
            }
        ]
    }


def _vdn_key(index: int) -> str:
    return f"diffusion_model.blocks.{index}.attn.forward"


def _fake_vdn_object_patches():
    runtime = {}
    originals = {}
    capture = {}
    current = {}
    for index in range(w._EXPECTED_BLOCKS):
        key = _vdn_key(index)

        def original(*_args, _index=index, **_kwargs):
            return _index

        def wrapper(*_args, _index=index, **_kwargs):
            return _index

        wrapper._h3_first_high_operator_diagnostic_v1 = True
        wrapper._h3_first_high_operator_original_forward = original
        runtime[key] = wrapper
        originals[original] = {"production_identity": index}
        capture[key] = {"production_identity": index}
        current[key] = {"diagnostic_identity": index}
    return runtime, originals, capture, current


def _receipts(mode: str):
    local_route = "vdn_local_native_window_w" if mode == "native_window" else "vdn_local_native_full_w"
    local_rows = list(w._EXPECTED_LOCAL_Q_ROWS)
    window_kv_rows = list(w._EXPECTED_WINDOW_KV_ROWS)
    result = []
    for block in range(50):
        common = {
            "block": block,
            "packed_rows": 56349,
            "video_start": 3101,
            "video_end": 56349,
            "gate_fingerprint": "d" * 64,
            "adapter_fingerprint": "e" * 64,
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
                    "kv_rows": 56349 if mode == "native_full_support" else window_kv_rows[group_index],
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


def test_vdn_object_patch_normalization_proves_exact_underlying_r_forward(monkeypatch):
    gate = _vdn_source_gate()
    source_path = gate["entries"][0]["path"]
    runtime, originals, capture_patches, current_patches = _fake_vdn_object_patches()
    unrelated_key = "diffusion_model.unrelated.forward"
    current_patches[unrelated_key] = {"identity": "unchanged"}
    capture_patches[unrelated_key] = {"identity": "unchanged"}
    runtime[unrelated_key] = lambda: None

    wrappers = {value for key, value in runtime.items() if key != unrelated_key}
    monkeypatch.setattr(
        w,
        "_callable_identity",
        lambda value: {
            "file": {
                "resolved_path": source_path if value in wrappers else "/production.py",
            }
        },
    )
    monkeypatch.setattr(w, "_callable_equivalence_identity", lambda value: copy.deepcopy(originals[value]))

    current = {"active_object_patches": copy.deepcopy(current_patches)}
    capture = {"active_object_patches": copy.deepcopy(capture_patches)}
    original_current = copy.deepcopy(current)
    guider = SimpleNamespace(model_patcher=SimpleNamespace(object_patches=runtime))

    normalized = w._normalize_vdn_w_object_patches(current, capture, guider, gate)

    for index in range(w._EXPECTED_BLOCKS):
        key = _vdn_key(index)
        assert normalized["active_object_patches"][key] == capture_patches[key]
    assert normalized["active_object_patches"][unrelated_key] == {"identity": "unchanged"}
    assert current == original_current


def test_vdn_object_patch_normalization_fails_closed_on_patch_set_and_source(monkeypatch):
    gate = _vdn_source_gate()
    source_path = gate["entries"][0]["path"]
    runtime, originals, capture_patches, current_patches = _fake_vdn_object_patches()
    current = {"active_object_patches": current_patches}
    capture = {"active_object_patches": capture_patches}

    missing_runtime = dict(runtime)
    missing_runtime.pop(_vdn_key(49))
    with pytest.raises(RuntimeError, match="not exactly the 50 H3 attention patches"):
        w._normalize_vdn_w_object_patches(
            current,
            capture,
            SimpleNamespace(model_patcher=SimpleNamespace(object_patches=missing_runtime)),
            gate,
        )

    wrong = runtime[_vdn_key(0)]
    monkeypatch.setattr(
        w,
        "_callable_identity",
        lambda value: {
            "file": {
                "resolved_path": "/foreign.py" if value is wrong else source_path,
            }
        },
    )
    monkeypatch.setattr(w, "_callable_equivalence_identity", lambda value: copy.deepcopy(originals[value]))
    with pytest.raises(RuntimeError, match="wrapper source identity changed"):
        w._normalize_vdn_w_object_patches(
            current,
            capture,
            SimpleNamespace(model_patcher=SimpleNamespace(object_patches=runtime)),
            gate,
        )


def test_vdn_object_patch_normalization_rejects_changed_underlying_forward(monkeypatch):
    gate = _vdn_source_gate()
    source_path = gate["entries"][0]["path"]
    runtime, originals, capture_patches, current_patches = _fake_vdn_object_patches()
    changed_original = runtime[_vdn_key(17)]._h3_first_high_operator_original_forward
    originals[changed_original] = {"production_identity": "changed"}
    monkeypatch.setattr(
        w,
        "_callable_identity",
        lambda _value: {"file": {"resolved_path": source_path}},
    )
    monkeypatch.setattr(w, "_callable_equivalence_identity", lambda value: copy.deepcopy(originals[value]))

    with pytest.raises(RuntimeError, match="underlying VDN production forward differs from R"):
        w._normalize_vdn_w_object_patches(
            {"active_object_patches": current_patches},
            {"active_object_patches": capture_patches},
            SimpleNamespace(model_patcher=SimpleNamespace(object_patches=runtime)),
            gate,
        )


def test_capture_base_source_gate_requires_exact_recorded_bytes():
    source_path = "/home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3/sol_h3/__init__.py"
    gate = {
        "entries": [
            {
                "owner": "sol",
                "path": source_path,
                "base_git_blob_sha": "1" * 40,
                "candidate_git_blob_sha": "2" * 40,
                "base_sha256": "a" * 64,
            }
        ]
    }
    capture = {
        "loaded_companion_sources": {
            "sol_h3": [
                {
                    "module": "sol_h3",
                    "file": {"resolved_path": source_path, "sha256": "a" * 64},
                }
            ]
        }
    }

    report = w._verify_capture_base_sources(capture, gate)
    assert report["exact"] is True
    assert report["checked"] == [
        {
            "owner": "sol",
            "path": source_path,
            "capture_sha256": "a" * 64,
            "expected_base_sha256": "a" * 64,
            "exact": True,
        }
    ]

    changed = copy.deepcopy(capture)
    changed["loaded_companion_sources"]["sol_h3"][0]["file"]["sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="R base source differs"):
        w._verify_capture_base_sources(changed, gate)

    missing = {"loaded_companion_sources": {"sol_h3": []}}
    with pytest.raises(RuntimeError, match="R provenance is missing reviewed base source"):
        w._verify_capture_base_sources(missing, gate)


def test_source_delta_allowance_is_exact_path_changed_companion_only():
    changed_path = "/custom_nodes/sol_h3/sol_h3/__init__.py"
    unchanged_core = "/ComfyUI/comfy/model_sampling.py"
    gate = {
        "entries": [
            {
                "path": changed_path,
                "base_git_blob_sha": "1" * 40,
                "base_sha256": "a" * 64,
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
    assert report["expected_nonlocal_routes"] is True
    assert report["support_ok"] is True
    assert report["complement_ok"] is True
    assert report["canonical_or_window_kv_ok"] is True
    assert report["nonlocal_full_support_ok"] is True
    assert report["geometry_ok"] is True
    assert report["per_block_topology_ok"] is True
    assert report["per_group_geometry_ok"] is True
    assert report["gate_fingerprint_consistent"] is True
    assert report["adapter_fingerprints_complete"] is True
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

    wrong_q_geometry = copy.deepcopy(receipts)
    block_zero_locals = [item for item in wrong_q_geometry if item["block"] == 0 and item["kind"] == "local"]
    block_zero_locals[0]["q_rows"] += 1
    block_zero_locals[1]["q_rows"] -= 1
    report = w._validate_receipts(wrong_q_geometry, "native_window")
    assert report["per_block_topology_ok"] is True
    assert report["per_group_geometry_ok"] is False

    wrong_kv_geometry = copy.deepcopy(receipts)
    first_local = next(item for item in wrong_kv_geometry if item["block"] == 0 and item["kind"] == "local")
    first_local["kv_rows"] += 1024
    assert w._validate_receipts(wrong_kv_geometry, "native_window")["per_group_geometry_ok"] is False

    wrong_route = copy.deepcopy(receipts)
    first_local = next(item for item in wrong_route if item["kind"] == "local")
    first_local["provider_route"] = "vdn_local_sol"
    assert w._validate_receipts(wrong_route, "native_window")["expected_local_route"] is False

    wrong_nonlocal_route = copy.deepcopy(receipts)
    first_global = next(item for item in wrong_nonlocal_route if item["kind"] == "global")
    first_global["provider_route"] = "foreign_global"
    assert w._validate_receipts(wrong_nonlocal_route, "native_window")["expected_nonlocal_routes"] is False

    wrong_gate = copy.deepcopy(receipts)
    wrong_gate[0]["gate_fingerprint"] = "f" * 64
    assert w._validate_receipts(wrong_gate, "native_window")["gate_fingerprint_consistent"] is False

    wrong_adapter = copy.deepcopy(receipts)
    wrong_adapter[0]["adapter_fingerprint"] = "f" * 64
    assert w._validate_receipts(wrong_adapter, "native_window")["adapter_fingerprints_complete"] is False


def test_receipt_sink_is_bounded_mutable_owner():
    sink = w._ReceiptSink()
    sink.append({"block": 0})
    assert len(sink) == 1
    assert list(sink) == [{"block": 0}]


def test_provenance_gate_normalizes_only_proven_vdn_wrapper_delta(monkeypatch):
    gate = _vdn_source_gate()
    capture = {
        "active_object_patches": {},
        "active_wrapper_order": {"outer_sample": [{"key": "production"}]},
        "loaded_companion_sources": {},
    }
    current = copy.deepcopy(capture)
    monkeypatch.setattr(w._replay, "_bundle_provenance_equivalence_identity", lambda _manifest: capture)
    monkeypatch.setattr(w._replay, "_provenance_equivalence_identity", lambda _manifest: current)
    monkeypatch.setattr(
        w,
        "_normalize_vdn_w_object_patches",
        lambda value, _capture, _guider, _gate: copy.deepcopy(value),
    )
    state = SimpleNamespace(replay=SimpleNamespace(manifest={}), source_gate=gate)
    record = SimpleNamespace(state=SimpleNamespace(manifest={}))
    guider = SimpleNamespace()

    report = w._provenance_gate(state, record, guider)
    assert report["differences"] == []
    assert report["capture_base_sources"] == {"exact": True, "checked": []}
    assert report["exact_except_reviewed_w_delta"] is True

    current["active_wrapper_order"]["outer_sample"].append({"key": "unrelated-change"})
    report = w._provenance_gate(state, record, guider)
    assert report["exact_except_reviewed_w_delta"] is False
    assert any("active_wrapper_order" in item for item in report["unexpected_differences"])
