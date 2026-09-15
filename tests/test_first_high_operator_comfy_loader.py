from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import __init__ as plugin_root
from h3_flow_regenerate import first_high_operator_comparison as w


def _deny_bare_import(name: str):
    raise ModuleNotFoundError(name)


def test_w_source_resolver_uses_loaded_comfyui_namespaced_module(monkeypatch):
    bare = "h3_flow_regenerate.first_high_operator_comparison"
    synthetic = "custom_nodes.synthetic_flow.h3_flow_regenerate.first_high_operator_comparison"
    original = sys.modules.pop(bare, None)
    monkeypatch.setitem(sys.modules, synthetic, w)
    monkeypatch.setattr(plugin_root.importlib, "import_module", _deny_bare_import)
    try:
        resolved = plugin_root._resolve_w_source_entry({"module": bare, "relative_path": "."})
    finally:
        sys.modules.pop(synthetic, None)
        if original is not None:
            sys.modules[bare] = original

    assert resolved == Path(w.__file__).resolve(strict=True)


def test_w_source_resolver_fails_closed_on_distinct_suffix_sources(monkeypatch, tmp_path):
    bare = "h3_flow_regenerate.first_high_operator_comparison"
    synthetic = "custom_nodes.synthetic_flow.h3_flow_regenerate.first_high_operator_comparison"
    foreign_name = "custom_nodes.foreign_flow.h3_flow_regenerate.first_high_operator_comparison"
    foreign_path = tmp_path / "first_high_operator_comparison.py"
    foreign_path.write_text("# foreign\n", encoding="utf-8")
    foreign = ModuleType(foreign_name)
    foreign.__file__ = str(foreign_path)

    original = sys.modules.pop(bare, None)
    monkeypatch.setitem(sys.modules, synthetic, w)
    monkeypatch.setitem(sys.modules, foreign_name, foreign)
    monkeypatch.setattr(plugin_root.importlib, "import_module", _deny_bare_import)
    try:
        with pytest.raises(RuntimeError, match="resolves ambiguously"):
            plugin_root._resolve_w_source_entry({"module": bare, "relative_path": "."})
    finally:
        sys.modules.pop(synthetic, None)
        sys.modules.pop(foreign_name, None)
        if original is not None:
            sys.modules[bare] = original


def _fake_wrapper_identity(value):
    if value is w._outer_wrapper:
        return {"identity": "w-outer"}
    if value is w._sampler_entry_wrapper:
        return {"identity": "w-sampler"}
    raise AssertionError("unexpected callable")


def _wrapper_fixture():
    specs = (
        ("outer_sample", w._OUTER_KEY, w._outer_wrapper),
        ("sampler_sample", w._SAMPLER_KEY, w._sampler_entry_wrapper),
    )
    runtime = {
        "outer_sample": {
            w._OUTER_KEY: [w._outer_wrapper],
            "production.outer": [lambda: None],
        },
        "sampler_sample": {
            "production.sampler": [lambda: None],
            w._SAMPLER_KEY: [w._sampler_entry_wrapper],
        },
    }
    active = {
        "outer_sample": [
            {"key": w._OUTER_KEY, "callable": {"identity": "w-outer"}},
            {"key": "production.outer", "callable": {"identity": "production-outer"}},
        ],
        "sampler_sample": [
            {"key": "production.sampler", "callable": {"identity": "production-sampler"}},
            {"key": w._SAMPLER_KEY, "callable": {"identity": "w-sampler"}},
        ],
    }
    current = {
        "active_wrapper_order": copy.deepcopy(active),
        "patcher_wrapper_order": copy.deepcopy(active),
        "unchanged": {"value": 1},
    }
    return specs, runtime, current


def test_w_wrapper_provenance_normalizer_removes_only_exact_owned_wrappers(monkeypatch):
    specs, runtime, current = _wrapper_fixture()
    original = copy.deepcopy(current)
    monkeypatch.setattr(plugin_root, "_w_wrapper_specs", lambda: specs)
    monkeypatch.setattr(w, "_callable_equivalence_identity", _fake_wrapper_identity)
    guider = SimpleNamespace(model_patcher=SimpleNamespace(wrappers=runtime))

    normalized = plugin_root._normalize_w_wrapper_order(current, guider)

    expected_outer = [{"key": "production.outer", "callable": {"identity": "production-outer"}}]
    expected_sampler = [{"key": "production.sampler", "callable": {"identity": "production-sampler"}}]
    for field_name in ("active_wrapper_order", "patcher_wrapper_order"):
        assert normalized[field_name]["outer_sample"] == expected_outer
        assert normalized[field_name]["sampler_sample"] == expected_sampler
    assert normalized["unchanged"] == {"value": 1}
    assert current == original


def test_w_wrapper_provenance_normalizer_fails_closed_on_wrong_callable_or_owner(monkeypatch):
    specs, runtime, current = _wrapper_fixture()
    monkeypatch.setattr(plugin_root, "_w_wrapper_specs", lambda: specs)
    monkeypatch.setattr(w, "_callable_equivalence_identity", _fake_wrapper_identity)
    guider = SimpleNamespace(model_patcher=SimpleNamespace(wrappers=runtime))

    wrong_manifest = copy.deepcopy(current)
    wrong_manifest["active_wrapper_order"]["outer_sample"][0]["callable"] = {"identity": "foreign"}
    with pytest.raises(RuntimeError, match="provenance wrapper callable identity changed"):
        plugin_root._normalize_w_wrapper_order(wrong_manifest, guider)

    wrong_runtime = copy.deepcopy(runtime)
    wrong_runtime["outer_sample"][w._OUTER_KEY] = [lambda: None]
    with pytest.raises(RuntimeError, match="live wrapper callable identity changed"):
        plugin_root._normalize_w_wrapper_order(
            current,
            SimpleNamespace(model_patcher=SimpleNamespace(wrappers=wrong_runtime)),
        )

    wrong_owner = copy.deepcopy(runtime)
    wrong_owner["foreign_type"] = {w._OUTER_KEY: wrong_owner["outer_sample"].pop(w._OUTER_KEY)}
    with pytest.raises(RuntimeError, match="live wrapper key ownership changed"):
        plugin_root._normalize_w_wrapper_order(
            current,
            SimpleNamespace(model_patcher=SimpleNamespace(wrappers=wrong_owner)),
        )


def test_w_wrapper_provenance_normalizer_does_not_hide_unrelated_wrapper(monkeypatch):
    specs, runtime, current = _wrapper_fixture()
    monkeypatch.setattr(plugin_root, "_w_wrapper_specs", lambda: specs)
    monkeypatch.setattr(w, "_callable_equivalence_identity", _fake_wrapper_identity)
    runtime["outer_sample"]["foreign.outer"] = [lambda: None]
    current["active_wrapper_order"]["outer_sample"].append(
        {"key": "foreign.outer", "callable": {"identity": "foreign"}}
    )
    current["patcher_wrapper_order"]["outer_sample"].append(
        {"key": "foreign.outer", "callable": {"identity": "foreign"}}
    )

    normalized = plugin_root._normalize_w_wrapper_order(
        current,
        SimpleNamespace(model_patcher=SimpleNamespace(wrappers=runtime)),
    )

    for field_name in ("active_wrapper_order", "patcher_wrapper_order"):
        assert any(entry.get("key") == "foreign.outer" for entry in normalized[field_name]["outer_sample"])
